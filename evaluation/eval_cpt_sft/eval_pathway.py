#!/usr/bin/env python3
"""
检验「固定位置 / 跨层通路」假说。

背景问题
--------
SFT 可能在每一层都有更新，但是否：
  1) 每层只改了矩阵中的特定位置（行/列/元素）？
  2) 各层改的是同一组 hidden dimension，形成一条「通路」？
  3) 每个训练 step 都在更新同一批位置？

本脚本用 **最终 checkpoint 的 ΔW = W_trained - W_base** 回答 (1)(2)；
(3) 需要额外提供中间 checkpoint（--step_checkpoints）。

核心指标
--------
A. 元素集中度：多少比例的元素贡献了 50%/90% 的 ||ΔW||_F^2 能量
   - 若绝大多数元素都活跃（接近 100%）→ 非稀疏、非固定少数位置
   - 若极少数元素贡献大部分能量 → 位置高度集中

B. 维度 profile：对每层某矩阵，按行/列算 L2 范数，得到长度为 hidden_size 的向量
   - q_proj 行 profile：输出维（head 聚合前的 out dim）
   - o_proj 列 profile：写回 residual 的维
   - down_proj 列 profile：MLP 输出到 residual 的维

C. 跨层通路相关：
   - 相邻层 profile 的 Pearson 相关
   - top-k 维度的 Jaccard overlap
   - 与「随机打乱维度」的 null baseline 对比
   - 若 corr ≈ shuffled、overlap ≈ k/dim → **不支持**固定通路
   - 若 corr 显著高于 shuffled、overlap 显著高于 k/dim → **支持**通路假说

用法
----
python3 eval_pathway.py \\
    --base  /path/to/Qwen3-8B-MchRiskModel-Init \\
    --sft   /path/to/Qwen3-8b-mch-risk \\
    --cpt   /path/to/cpt-checkpoint          # 可选，用于对比 \\
    --output_dir ./output/pathway

# 若有中间 checkpoint，检验「每步是否更新同一位置」：
python3 eval_pathway.py \\
    --base /path/to/base \\
    --sft  /path/to/final \\
    --step_checkpoints 500:/path/to/ckpt-500 1000:/path/to/ckpt-1000 2000:/path/to/final \\
    --output_dir ./output/pathway
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

# 与 eval_delta_w.py 保持一致的加载逻辑
LAYER_MODULE_RE = re.compile(
    r"model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight"
)

# 用于通路分析的子模块：(key 后缀, profile 轴, 语义说明)
PATHWAY_MODULES = [
    ("self_attn.q_proj.weight", "row", "q_proj row / out dim"),
    ("self_attn.o_proj.weight", "col", "o_proj col / residual dim"),
    ("mlp.down_proj.weight", "col", "down_proj col / residual dim"),
]


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.load(open(index))["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        from safetensors.torch import load_file
        return {k: str(single) for k in load_file(str(single)).keys()}
    raise FileNotFoundError(f"No safetensors in {model_dir}")


def load_tensor(model_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def delta_tensor(base_dir: Path, trained_dir: Path, bm, tm, key: str) -> Optional[torch.Tensor]:
    if key not in bm or key not in tm:
        return None
    wb = load_tensor(base_dir, bm, key).float()
    wt = load_tensor(trained_dir, tm, key).float()
    if wb.shape != wt.shape:
        return None
    return wt - wb


def element_concentration(delta: torch.Tensor) -> Dict[str, float]:
    """返回元素能量集中度与活跃比例。"""
    e = (delta.reshape(-1) ** 2).numpy()
    total = float(e.sum())
    n = len(e)
    if total < 1e-20:
        return {
            "n_elements": n,
            "pct_elements_for_50pct_energy": float("nan"),
            "pct_elements_for_90pct_energy": float("nan"),
            "pct_active_gt_1e8": 0.0,
            "gini_energy": float("nan"),
        }

    sorted_e = np.sort(e)[::-1]
    csum = np.cumsum(sorted_e) / total
    pct50 = (int(np.searchsorted(csum, 0.5)) + 1) / n * 100.0
    pct90 = (int(np.searchsorted(csum, 0.9)) + 1) / n * 100.0
    active = float((np.abs(delta.numpy()) > 1e-8).mean() * 100.0)

    # Gini on squared delta (能量不平等程度，越高越集中)
    sorted_asc = np.sort(e)
    idx = np.arange(1, n + 1)
    gini = float((np.sum((2 * idx - n - 1) * sorted_asc)) / (n * total))

    return {
        "n_elements": n,
        "pct_elements_for_50pct_energy": pct50,
        "pct_elements_for_90pct_energy": pct90,
        "pct_active_gt_1e8": active,
        "gini_energy": gini,
    }


def dim_profile(delta: torch.Tensor, axis: str) -> np.ndarray:
    d = delta.numpy()
    if axis == "row":
        return np.linalg.norm(d, axis=1)
    if axis == "col":
        return np.linalg.norm(d, axis=0)
    raise ValueError(axis)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a), len(b))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / k


def cross_layer_pathway(
    profiles: Dict[int, np.ndarray],
    top_k: int,
    seed: int = 0,
) -> Dict[str, float]:
    """相邻层 + 多层 skip 的通路统计。"""
    layers = sorted(profiles)
    if len(layers) < 2:
        return {}

    dim = len(next(iter(profiles.values())))
    rng = np.random.default_rng(seed)

    adj_corr, adj_perm, adj_overlap = [], [], []
    for i in range(len(layers) - 1):
        a, b = layers[i], layers[i + 1]
        pa, pb = profiles[a], profiles[b]
        adj_corr.append(pearson(pa, pb))
        adj_perm.append(pearson(pa, pb[rng.permutation(len(pb))]))
        adj_overlap.append(topk_overlap(pa, pb, top_k))

    skip_corrs = {}
    for skip in (2, 4, 8, 16):
        vals = []
        for i in range(len(layers) - skip):
            vals.append(pearson(profiles[layers[i]], profiles[layers[i + skip]]))
        skip_corrs[f"corr_skip_{skip}"] = float(np.nanmean(vals))

    return {
        "dim_size": dim,
        "top_k": top_k,
        "random_overlap_expect": top_k / dim,
        "adj_layer_corr_mean": float(np.nanmean(adj_corr)),
        "adj_layer_corr_shuffled_mean": float(np.nanmean(adj_perm)),
        "adj_layer_topk_overlap_mean": float(np.mean(adj_overlap)),
        **skip_corrs,
    }


def collect_keys(base_map: Dict[str, str], module_suffix: str) -> Dict[int, str]:
    out = {}
    for key in base_map:
        if key.endswith(module_suffix) and LAYER_MODULE_RE.match(key):
            layer = int(LAYER_MODULE_RE.match(key).group(1))
            out[layer] = key
    return out


def analyze_model(
    label: str,
    base_dir: Path,
    trained_dir: Path,
    bm: Dict[str, str],
    tm: Dict[str, str],
    top_k: int,
) -> Tuple[List[dict], Dict[str, dict], Dict[str, Dict[int, np.ndarray]]]:
    """返回 element 行、pathway 汇总、各模块 profile。"""
    element_rows = []
    pathway_summary = {}
    all_profiles: Dict[str, Dict[int, np.ndarray]] = {}

    for module_suffix, axis, desc in PATHWAY_MODULES:
        keys = collect_keys(bm, module_suffix)
        profiles = {}
        conc_list = []

        for layer in sorted(keys):
            key = keys[layer]
            d = delta_tensor(base_dir, trained_dir, bm, tm, key)
            if d is None:
                continue

            conc = element_concentration(d)
            conc_list.append(conc)
            element_rows.append({
                "model": label,
                "module": module_suffix,
                "layer": layer,
                **conc,
            })
            profiles[layer] = dim_profile(d, axis)

        all_profiles[module_suffix] = profiles
        if not profiles:
            continue

        stats = cross_layer_pathway(profiles, top_k=top_k)
        stats["module"] = module_suffix
        stats["axis"] = axis
        stats["description"] = desc
        stats["mean_pct_active"] = float(np.mean([c["pct_active_gt_1e8"] for c in conc_list]))
        stats["mean_pct_for_50pct_energy"] = float(np.mean([c["pct_elements_for_50pct_energy"] for c in conc_list]))
        stats["mean_pct_for_90pct_energy"] = float(np.mean([c["pct_elements_for_90pct_energy"] for c in conc_list]))
        stats["mean_gini_energy"] = float(np.mean([c["gini_energy"] for c in conc_list]))
        pathway_summary[module_suffix] = stats

        print(f"[{label}] {desc}")
        print(f"  active={stats['mean_pct_active']:.1f}%  "
              f"50%energy@{stats['mean_pct_for_50pct_energy']:.1f}% elems  "
              f"90%energy@{stats['mean_pct_for_90pct_energy']:.1f}% elems")
        print(f"  adj corr={stats['adj_layer_corr_mean']:.4f}  "
              f"shuffled={stats['adj_layer_corr_shuffled_mean']:.4f}  "
              f"top-{top_k} overlap={stats['adj_layer_topk_overlap_mean']:.3f}  "
              f"(random={stats['random_overlap_expect']:.3f})")

    return element_rows, pathway_summary, all_profiles


def analyze_step_consistency(
    base_dir: Path,
    bm: Dict[str, str],
    step_checkpoints: List[Tuple[int, Path]],
    module_suffix: str,
    axis: str,
    top_k: int,
) -> List[dict]:
    """
    检验「每步是否更新同一批维度」：
    将各 step 相对 base 的 top-k 维度，与最终 step 的 top-k 做 Jaccard。
    """
    if len(step_checkpoints) < 2:
        return []

    step_checkpoints = sorted(step_checkpoints, key=lambda x: x[0])
    final_step, final_dir = step_checkpoints[-1]
    final_tm = load_weight_map(final_dir)
    keys = collect_keys(bm, module_suffix)

    final_profiles = {}
    for layer, key in keys.items():
        d = delta_tensor(base_dir, final_dir, bm, final_tm, key)
        if d is not None:
            final_profiles[layer] = dim_profile(d, axis)

    rows = []
    for step, ckpt_dir in step_checkpoints[:-1]:
        tm = load_weight_map(ckpt_dir)
        overlaps = []
        for layer, key in keys.items():
            if layer not in final_profiles:
                continue
            d = delta_tensor(base_dir, ckpt_dir, bm, tm, key)
            if d is None:
                continue
            prof = dim_profile(d, axis)
            overlaps.append(topk_overlap(prof, final_profiles[layer], top_k))

        rows.append({
            "module": module_suffix,
            "step": step,
            "final_step": final_step,
            "top_k": top_k,
            "mean_topk_jaccard_with_final": float(np.mean(overlaps)) if overlaps else float("nan"),
            "random_expect": top_k / len(next(iter(final_profiles.values()))) if final_profiles else float("nan"),
        })
        print(f"[step] {module_suffix} step={step} vs final={final_step}: "
              f"jaccard={rows[-1]['mean_topk_jaccard_with_final']:.3f}")

    return rows


def interpret_pathway(summary: Dict[str, dict]) -> Dict[str, str]:
    """根据指标给出可读结论（规则化，供人工复核）。"""
    notes = {}
    for module, s in summary.items():
        active = s["mean_pct_active"]
        corr = s["adj_layer_corr_mean"]
        shuf = s["adj_layer_corr_shuffled_mean"]
        overlap = s["adj_layer_topk_overlap_mean"]
        rand = s["random_overlap_expect"]
        pct90 = s["mean_pct_for_90pct_energy"]

        parts = []
        if active > 95:
            parts.append("元素级几乎全量活跃，不像只改极少数固定坐标")
        elif pct90 < 20:
            parts.append("能量高度集中在少数元素，存在固定位置倾向")
        else:
            parts.append("元素能量中等集中，非极端稀疏")

        corr_gain = corr - shuf if not (np.isnan(corr) or np.isnan(shuf)) else 0.0
        overlap_gain = overlap - rand

        if corr_gain > 0.05 and overlap_gain > rand * 0.5:
            parts.append("跨层维度 profile 显著高于随机基线，支持「通路」假说")
        elif corr_gain > 0.02 or overlap_gain > rand * 0.25:
            parts.append("跨层有弱相关，但通路证据不充分")
        else:
            parts.append("跨层相关≈随机，不支持固定 hidden-dim 通路")

        notes[module] = "；".join(parts)
    return notes


def plot_corr_heatmap(
    profiles: Dict[int, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    layers = sorted(profiles)
    n = len(layers)
    if n < 2:
        return
    mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            mat[i, j] = pearson(profiles[li], profiles[lj])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(layers, fontsize=7, rotation=90)
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def parse_step_checkpoints(spec: List[str]) -> List[Tuple[int, Path]]:
    out = []
    for item in spec:
        step_str, path = item.split(":", 1)
        out.append((int(step_str), Path(path)))
    return out


def main():
    p = argparse.ArgumentParser(description="ΔW 通路 / 固定位置分析")
    p.add_argument("--base", required=True, help="Base 模型目录")
    p.add_argument("--sft", required=True, help="SFT 最终 checkpoint")
    p.add_argument("--cpt", default=None, help="可选 CPT checkpoint，用于对比")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=128, help="top-k 维度用于 overlap/Jaccard")
    p.add_argument(
        "--step_checkpoints",
        nargs="*",
        default=[],
        help="中间 checkpoint，格式: STEP:PATH，如 500:/path/ckpt-500 2000:/path/final",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    sft = Path(args.sft)

    bm = load_weight_map(base)
    sm = load_weight_map(sft)

    all_element_rows = []
    all_pathway = {}
    all_notes = {}

    for label, td, tm in [("sft", sft, sm)] + (
        [("cpt", Path(args.cpt), load_weight_map(Path(args.cpt)))] if args.cpt else []
    ):
        elem_rows, pathway, profiles = analyze_model(label, base, td, bm, tm, args.top_k)
        all_element_rows.extend(elem_rows)
        all_pathway[label] = pathway
        all_notes[label] = interpret_pathway(pathway)

        # 热力图：仅 SFT q_proj
        q_prof = profiles.get("self_attn.q_proj.weight", {})
        if label == "sft" and q_prof:
            plot_corr_heatmap(
                q_prof,
                "SFT q_proj row-profile corr across layers",
                out / "pathway_qproj_heatmap.png",
            )

    # step 一致性
    step_rows = []
    if args.step_checkpoints:
        steps = parse_step_checkpoints(args.step_checkpoints)
        for module_suffix, axis, _ in PATHWAY_MODULES:
            step_rows.extend(
                analyze_step_consistency(base, bm, steps, module_suffix, axis, args.top_k)
            )

    # 写文件
    with open(out / "element_concentration.csv", "w", newline="") as f:
        fields = [
            "model", "module", "layer", "n_elements",
            "pct_elements_for_50pct_energy", "pct_elements_for_90pct_energy",
            "pct_active_gt_1e8", "gini_energy",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_element_rows)

    summary = {
        "pathway_stats": all_pathway,
        "interpretation": all_notes,
        "how_to_read": {
            "fixed_positions": (
                "若 pct_active≈100% 且 pct_elements_for_90pct_energy 也很大，"
                "说明不是「只改少数固定元素」，而是密集中等幅度更新。"
            ),
            "pathway": (
                "若 adj_layer_corr_mean 显著 > adj_layer_corr_shuffled_mean，"
                "且 adj_layer_topk_overlap_mean 显著 > random_overlap_expect，"
                "说明相邻层在同一批 hidden dimension 上改动更大，支持通路假说。"
            ),
            "per_step": (
                "需提供 --step_checkpoints。"
                "若早期 step 的 top-k 维度与 final 的 Jaccard 很高，"
                "说明训练过程中持续更新同一批位置。"
            ),
        },
    }
    with open(out / "pathway_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if step_rows:
        with open(out / "step_consistency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
            w.writeheader()
            w.writerows(step_rows)

    print("\n=== Interpretation ===")
    for label, notes in all_notes.items():
        print(f"[{label}]")
        for module, text in notes.items():
            print(f"  {module}: {text}")

    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
