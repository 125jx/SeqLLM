#!/usr/bin/env python3
"""
CPT vs SFT 权重几何分析（子空间 / 低秩 / 与 base 主子空间重叠）。

动机
----
summary.json 里有个反常点：SFT 对主干权重的改动量（backbone r_l≈0.0216）是
CPT（≈0.00155）的 ~14 倍，但遗忘的却是 CPT。这说明「遗忘与否」不取决于 ΔW 的
大小，而取决于 ΔW 的方向落在哪个子空间。本脚本量化这一点。

三组指标（逐权重矩阵 ΔW = W_trained - W_base）
------------------------------------------------
1. 低秩性 / 子空间紧凑度：
   - stable_rank = ||ΔW||_F^2 / sigma_max^2   （廉价，全层都算）
   - effective_rank = exp(-Σ p_i log p_i), p_i = σ_i^2 / Σσ^2  （需完整谱，可选）
   - rank_for_90pct_energy = 达到 90% 谱能量所需奇异值个数 / 满秩       （可选）
   低秩 → 更新压缩在少数方向 → 支持「特定子空间」假说。

2. 与 base 主子空间的能量重叠：
   对 W_base 做截断 SVD 取 top-k 方向 U_k(输出侧) / V_k(输入侧)，
   - proj_left_k  = ||U_k^T ΔW||_F^2 / ||ΔW||_F^2
   - proj_right_k = ||ΔW V_k||_F^2  / ||ΔW||_F^2
   随机零假设基线 ≈ k / dim。
   若 CPT 的重叠显著高于 SFT（且高于基线）→ CPT 更扰动 base 主方向 → 干扰 → 遗忘；
   SFT 更多落在低能量尾部 / 补空间 → 无损。

3. 任务向量夹角：
   cos(ΔW_SFT, ΔW_CPT)（逐矩阵 flatten）。接近 0 → 两者更新不同子空间。

用法
----
python3 eval_subspace.py \
    --base /path/Qwen3-8B-MchRiskModel-Init \
    --cpt  /path/saves/qwen8b_pt_6144/checkpoint-20000 \
    --sft  /path/Qwen/Qwen3-8b-mch-risk \
    --output_dir ./output/subspace \
    --k_list 64 256 \
    --device auto \
    --full_spectrum          # 可选：额外算 effective_rank（较慢）
    --spectrum_stride 4      # full_spectrum 时每隔几层算一次
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

# 只分析主干 layer 内的线性权重（embed / lm_head 另有特殊性，此处不含）
LAYER_MODULE_RE = re.compile(
    r"model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight"
)
MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
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


def load_tensor(model_dir: Path, weight_map: Dict[str, str], key: str, device: str) -> torch.Tensor:
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    return t.to(device=device, dtype=torch.float32)


def parse_layer_module(key: str):
    m = LAYER_MODULE_RE.match(key)
    if not m:
        return None, None
    layer = int(m.group(1))
    module = m.group(2)  # e.g. self_attn.q_proj
    return layer, module


def base_lowrank_subspace(w_base: torch.Tensor, max_k: int):
    """返回 base 的 top-max_k 左/右奇异向量 U_k(m x k), V_k(n x k)。"""
    q = min(max_k, min(w_base.shape) - 1)
    # svd_lowrank: A ≈ U diag(S) V^T, U:(m,q) V:(n,q)
    U, S, V = torch.svd_lowrank(w_base, q=q, niter=4)
    return U, V


def spectrum_metrics(delta: torch.Tensor) -> Dict[str, float]:
    """完整奇异值谱的指标（较慢）。"""
    sv = torch.linalg.svdvals(delta)  # 降序
    sv2 = sv ** 2
    total = float(sv2.sum())
    if total < 1e-20:
        return {"effective_rank": float("nan"), "rank_for_90pct_energy_frac": float("nan")}
    p = (sv2 / sv2.sum()).clamp_min(1e-20)
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())
    csum = torch.cumsum(sv2, dim=0) / total
    r90 = int(torch.searchsorted(csum, torch.tensor(0.9, device=csum.device)).item()) + 1
    return {
        "effective_rank": eff_rank,
        "rank_for_90pct_energy_frac": r90 / len(sv),
    }


def matrix_metrics(
    delta: torch.Tensor,
    fro: float,
    Uk: torch.Tensor,
    Vk: torch.Tensor,
    k_list: List[int],
) -> Dict[str, float]:
    out = {"fro": fro}
    if fro < 1e-12:
        out["sigma_max"] = 0.0
        out["stable_rank"] = float("nan")
        for k in k_list:
            out[f"proj_left_k{k}"] = float("nan")
            out[f"proj_right_k{k}"] = float("nan")
        return out

    # sigma_max via lowrank (q=1)
    _, s1, _ = torch.svd_lowrank(delta, q=1, niter=4)
    sigma_max = float(s1[0].item())
    out["sigma_max"] = sigma_max
    out["stable_rank"] = (fro ** 2) / (sigma_max ** 2) if sigma_max > 1e-12 else float("nan")

    fro2 = fro ** 2
    for k in k_list:
        kk = min(k, Uk.shape[1], Vk.shape[1])
        left = torch.linalg.norm(Uk[:, :kk].transpose(0, 1) @ delta).item() ** 2
        right = torch.linalg.norm(delta @ Vk[:, :kk]).item() ** 2
        out[f"proj_left_k{k}"] = left / fro2
        out[f"proj_right_k{k}"] = right / fro2
    return out


def main():
    p = argparse.ArgumentParser(description="CPT vs SFT 权重子空间 / 低秩 / 主子空间重叠分析")
    p.add_argument("--base", required=True)
    p.add_argument("--cpt", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k_list", type=int, nargs="+", default=[64, 256])
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--full_spectrum", action="store_true", help="额外算 effective_rank（慢）")
    p.add_argument("--spectrum_stride", type=int, default=4, help="full_spectrum 时每隔几层算一次")
    p.add_argument("--layer_stride", type=int, default=1, help="只分析每隔几层（加速）")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base, cpt, sft = Path(args.base), Path(args.cpt), Path(args.sft)
    print("Base:", base, "\nCPT:", cpt, "\nSFT:", sft, "\nOut:", out)

    bm = load_weight_map(base)
    cm = load_weight_map(cpt)
    sm = load_weight_map(sft)

    max_k = max(args.k_list)
    per_matrix_rows: List[dict] = []
    cos_rows: List[dict] = []

    # 收集可分析的 key（三方都在、shape 一致、且是 layer 模块）
    keys = []
    for key in sorted(bm):
        layer, module = parse_layer_module(key)
        if layer is None:
            continue
        if layer % args.layer_stride != 0:
            continue
        if key not in cm or key not in sm:
            continue
        keys.append((layer, module, key))

    print(f"Analyzing {len(keys)} weight matrices ...")

    for idx, (layer, module, key) in enumerate(keys):
        w_base = load_tensor(base, bm, key, device)
        w_cpt = load_tensor(cpt, cm, key, device)
        w_sft = load_tensor(sft, sm, key, device)

        Uk, Vk = base_lowrank_subspace(w_base, max_k)

        d_cpt = w_cpt - w_base
        d_sft = w_sft - w_base
        fro_cpt = float(torch.linalg.norm(d_cpt).item())
        fro_sft = float(torch.linalg.norm(d_sft).item())

        do_spectrum = args.full_spectrum and (layer % args.spectrum_stride == 0)

        for label, d, fro in [("cpt", d_cpt, fro_cpt), ("sft", d_sft, fro_sft)]:
            row = {"model": label, "layer": layer, "module": module,
                   "out_dim": d.shape[0], "in_dim": d.shape[1]}
            row.update(matrix_metrics(d, fro, Uk, Vk, args.k_list))
            if do_spectrum:
                row.update(spectrum_metrics(d))
            per_matrix_rows.append(row)

        # 任务向量夹角
        denom = fro_cpt * fro_sft
        cos = float((d_cpt.reshape(-1) @ d_sft.reshape(-1)).item() / denom) if denom > 1e-12 else float("nan")
        cos_rows.append({"layer": layer, "module": module, "cos_sft_cpt": cos})

        # 及时释放
        del w_base, w_cpt, w_sft, d_cpt, d_sft, Uk, Vk
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[{idx + 1}/{len(keys)}] L{layer} {module}  "
              f"fro(cpt/sft)={fro_cpt:.3f}/{fro_sft:.3f}  "
              f"stable_rank(cpt/sft)={per_matrix_rows[-2]['stable_rank']:.1f}/"
              f"{per_matrix_rows[-1]['stable_rank']:.1f}  cos={cos:.3f}")

    # ---------- 写 CSV ----------
    fields = ["model", "layer", "module", "out_dim", "in_dim", "fro", "sigma_max", "stable_rank"]
    for k in args.k_list:
        fields += [f"proj_left_k{k}", f"proj_right_k{k}"]
    if args.full_spectrum:
        fields += ["effective_rank", "rank_for_90pct_energy_frac"]
    with open(out / "per_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_matrix_rows:
            w.writerow(r)
    with open(out / "taskvector_cosine.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "module", "cos_sft_cpt"])
        w.writeheader()
        w.writerows(cos_rows)

    # ---------- 汇总 ----------
    summary = build_summary(per_matrix_rows, cos_rows, args.k_list, args.full_spectrum)
    with open(out / "subspace_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------- 作图 ----------
    main_k = max(args.k_list)
    plot_stable_rank(per_matrix_rows, out)
    plot_subspace_overlap(per_matrix_rows, summary, main_k, out)
    plot_overlap_bars(per_matrix_rows, main_k, out)
    plot_cosine(cos_rows, out)
    if args.full_spectrum:
        plot_effective_rank(per_matrix_rows, out)

    print("\n=== Key takeaways ===")
    for line in summary["takeaways"]:
        print("  -", line)
    print(f"\nDone. Outputs in {out}")


def _layer_mean(rows, model, field):
    by_layer = defaultdict(list)
    for r in rows:
        if r["model"] == model and field in r and not _isnan(r[field]):
            by_layer[r["layer"]].append(r[field])
    return {l: float(np.mean(v)) for l, v in by_layer.items()}


def _isnan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False


def build_summary(rows, cos_rows, k_list, full_spectrum):
    summary = {"by_model": {}, "by_module": {}, "cosine": {}, "takeaways": []}
    main_k = max(k_list)

    for model in ("cpt", "sft"):
        sub = [r for r in rows if r["model"] == model]
        entry = {
            "mean_fro": float(np.mean([r["fro"] for r in sub])),
            "mean_stable_rank": float(np.nanmean([r["stable_rank"] for r in sub])),
        }
        for k in k_list:
            entry[f"mean_proj_left_k{k}"] = float(np.nanmean([r[f"proj_left_k{k}"] for r in sub]))
            entry[f"mean_proj_right_k{k}"] = float(np.nanmean([r[f"proj_right_k{k}"] for r in sub]))
        if full_spectrum:
            er = [r["effective_rank"] for r in sub if "effective_rank" in r and not _isnan(r["effective_rank"])]
            if er:
                entry["mean_effective_rank"] = float(np.mean(er))
        summary["by_model"][model] = entry

    # 逐模块类型
    for module in MODULE_ORDER:
        d = {}
        for model in ("cpt", "sft"):
            sub = [r for r in rows if r["model"] == model and r["module"] == module]
            if not sub:
                continue
            d[model] = {
                "mean_stable_rank": float(np.nanmean([r["stable_rank"] for r in sub])),
                f"mean_proj_left_k{main_k}": float(np.nanmean([r[f"proj_left_k{main_k}"] for r in sub])),
                f"mean_proj_right_k{main_k}": float(np.nanmean([r[f"proj_right_k{main_k}"] for r in sub])),
            }
        if d:
            summary["by_module"][module] = d

    cvals = [c["cos_sft_cpt"] for c in cos_rows if not _isnan(c["cos_sft_cpt"])]
    summary["cosine"]["mean_cos_sft_cpt"] = float(np.mean(cvals)) if cvals else float("nan")

    # 简要自动解读
    cpt_e, sft_e = summary["by_model"]["cpt"], summary["by_model"]["sft"]
    pl = f"mean_proj_left_k{main_k}"
    summary["takeaways"].append(
        f"改动量 ||ΔW||_F 均值: SFT={sft_e['mean_fro']:.4f} vs CPT={cpt_e['mean_fro']:.4f} "
        f"(SFT/CPT={sft_e['mean_fro'] / max(cpt_e['mean_fro'], 1e-9):.1f}x)"
    )
    summary["takeaways"].append(
        f"stable_rank 均值: SFT={sft_e['mean_stable_rank']:.1f} vs CPT={cpt_e['mean_stable_rank']:.1f} "
        f"(越低越像低维子空间更新)"
    )
    summary["takeaways"].append(
        f"与 base top-{main_k} 主子空间(输出侧)能量重叠: SFT={sft_e[pl]:.3f} vs CPT={cpt_e[pl]:.3f} "
        f"(越高越扰动 base 主方向)"
    )
    summary["takeaways"].append(
        f"任务向量 cos(SFT,CPT) 均值={summary['cosine']['mean_cos_sft_cpt']:.3f} "
        f"(接近 0 表示 SFT/CPT 更新不同子空间)"
    )
    return summary


def plot_stable_rank(rows, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in [("cpt", "tab:blue"), ("sft", "tab:orange")]:
        m = _layer_mean(rows, model, "stable_rank")
        if not m:
            continue
        xs = sorted(m)
        ax.plot(xs, [m[x] for x in xs], marker="o", markersize=3,
                label=model.upper(), color=color, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("stable rank  (||ΔW||_F^2 / σ_max^2)")
    ax.set_title("ΔW stable rank per layer (lower = more low-rank / compact subspace)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "stable_rank_lines.png", dpi=150)
    plt.close(fig)
    print("Plot saved:", out / "stable_rank_lines.png")


def plot_effective_rank(rows, out):
    have = [r for r in rows if "effective_rank" in r and not _isnan(r["effective_rank"])]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in [("cpt", "tab:blue"), ("sft", "tab:orange")]:
        m = _layer_mean(have, model, "effective_rank")
        if not m:
            continue
        xs = sorted(m)
        ax.plot(xs, [m[x] for x in xs], marker="o", markersize=3,
                label=model.upper(), color=color, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("effective rank  exp(-Σ p logp)")
    ax.set_title("ΔW effective rank per layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "effective_rank_lines.png", dpi=150)
    plt.close(fig)
    print("Plot saved:", out / "effective_rank_lines.png")


def plot_subspace_overlap(rows, summary, main_k, out):
    field = f"proj_left_k{main_k}"
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in [("cpt", "tab:blue"), ("sft", "tab:orange")]:
        m = _layer_mean(rows, model, field)
        if not m:
            continue
        xs = sorted(m)
        ax.plot(xs, [m[x] for x in xs], marker="o", markersize=3,
                label=model.upper(), color=color, linewidth=2)
    # 随机基线 k/out_dim（取常见 out_dim 均值近似）
    dims = [r["out_dim"] for r in rows]
    if dims:
        null = main_k / float(np.mean(dims))
        ax.axhline(null, color="gray", linestyle="--", linewidth=1,
                   label=f"random null ≈ k/dim = {null:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"proj energy onto base top-{main_k} (output side)")
    ax.set_title("ΔW overlap with base principal subspace (higher = more interference)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "subspace_overlap_lines.png", dpi=150)
    plt.close(fig)
    print("Plot saved:", out / "subspace_overlap_lines.png")


def plot_overlap_bars(rows, main_k, out):
    field = f"proj_left_k{main_k}"
    modules = [m for m in MODULE_ORDER if any(r["module"] == m for r in rows)]
    x = np.arange(len(modules))
    w = 0.35

    def mean_for(model, module):
        vals = [r[field] for r in rows if r["model"] == model and r["module"] == module and not _isnan(r[field])]
        return float(np.mean(vals)) if vals else 0.0

    cpt = [mean_for("cpt", m) for m in modules]
    sft = [mean_for("sft", m) for m in modules]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, cpt, w, label="CPT", color="tab:blue")
    ax.bar(x + w / 2, sft, w, label="SFT", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(".")[-1] for m in modules], rotation=30, ha="right")
    ax.set_ylabel(f"proj energy onto base top-{main_k} (output side)")
    ax.set_title("Overlap with base principal subspace by module")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "subspace_overlap_bars.png", dpi=150)
    plt.close(fig)
    print("Plot saved:", out / "subspace_overlap_bars.png")


def plot_cosine(cos_rows, out):
    by_layer = defaultdict(list)
    for c in cos_rows:
        if not _isnan(c["cos_sft_cpt"]):
            by_layer[c["layer"]].append(c["cos_sft_cpt"])
    if not by_layer:
        return
    xs = sorted(by_layer)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, [float(np.mean(by_layer[x])) for x in xs],
            marker="o", markersize=3, color="tab:green", linewidth=2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(ΔW_SFT, ΔW_CPT)")
    ax.set_title("Task-vector alignment SFT vs CPT (≈0 = different subspaces)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "taskvector_cosine.png", dpi=150)
    plt.close(fig)
    print("Plot saved:", out / "taskvector_cosine.png")


if __name__ == "__main__":
    main()
