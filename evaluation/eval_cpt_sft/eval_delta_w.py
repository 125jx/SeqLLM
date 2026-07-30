#!/usr/bin/env python3
"""
CPT vs SFT ΔW 分析（按图片方法）

公式：r_l = ||W_trained - W_base||_F / ||W_base||_F

分组：
  - embed_tokens / lm_head（全局）
  - layer 0, 1, 2, ...（每层所有 backbone 权重矩阵取均值，不再拆 q/k/v/o）
  - 新增 token 行（embed / lm_head 拆分，单独列出）

输出图：
  1. 折线图：横轴 layer，纵轴 r_l，CPT vs SFT
  2. 分组柱状图：embed / backbone均值 / lm_head / new_tokens，CPT vs SFT 并排
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

WEIGHT_SUFFIXES = (
    "embed_tokens.weight", "lm_head.weight",
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
LAYER_RE = re.compile(
    r"model\.layers\.(\d+)\.(?:self_attn\.\w+|mlp\.\w+)\.weight"
)
DEFAULT_ORIGINAL_VOCAB_SIZE = 151643


def load_weight_map(model_dir):
    p = model_dir / "model.safetensors.index.json"
    if p.exists():
        return json.load(open(p))["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        from safetensors.torch import load_file
        return {k: str(single) for k in load_file(str(single)).keys()}
    raise FileNotFoundError("No safetensors in {}".format(model_dir))


def load_tensor(model_dir, weight_map, key):
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def relative_frobenius(w_trained, w_base):
    b = w_base.float()
    d = w_trained.float() - b
    bn = b.norm("fro").item()
    return float("nan") if bn < 1e-12 else d.norm("fro").item() / bn


def parse_key(key):
    if key == "model.embed_tokens.weight":
        return "embed_tokens", -1
    if key == "lm_head.weight":
        return "lm_head", -1
    m = LAYER_RE.match(key)
    if m:
        return "layer", int(m.group(1))
    return "other", -1


def analyze_pair(base_dir, trained_dir, label, base_map, trained_map, vocab_size):
    per_key = []
    layer_vals = defaultdict(list)
    embed_split = []
    only_keys = []

    for key in sorted(base_map):
        if not key.endswith(WEIGHT_SUFFIXES) or key not in trained_map:
            continue
        wb = load_tensor(base_dir, base_map, key)
        wt = load_tensor(trained_dir, trained_map, key)
        if wb.shape != wt.shape:
            continue

        group, layer = parse_key(key)
        r = relative_frobenius(wt, wb)
        per_key.append({"model": label, "key": key, "group": group, "layer": layer, "r_l": r})
        print("[{}] {} r={:.6f}".format(label, key, r))

        if group == "layer":
            layer_vals[layer].append(r)
        if key in ("model.embed_tokens.weight", "lm_head.weight"):
            split = min(vocab_size, wb.shape[0])
            if split < wb.shape[0]:
                for sl, s, e in [("original_vocab", 0, split), ("new_tokens", split, wb.shape[0])]:
                    embed_split.append({
                        "model": label, "module": key.split(".")[-2],
                        "slice": sl, "r_l": relative_frobenius(wt[s:e], wb[s:e]),
                    })

    for key in sorted(trained_map):
        if key.endswith(".weight") and key not in base_map:
            only_keys.append(key)

    # 聚合：每层一个 r_l（层内所有矩阵均值）
    rows = []
    for layer in sorted(layer_vals):
        rows.append({
            "model": label, "group": "layer", "layer": layer,
            "r_l": float(np.mean(layer_vals[layer])),
        })
    for r in per_key:
        if r["group"] in ("embed_tokens", "lm_head"):
            rows.append({
                "model": label, "group": r["group"], "layer": -1, "r_l": r["r_l"],
            })

    return rows, embed_split, only_keys, per_key


def build_summary(rows, embed_split, sft_only):
    summary = {"by_group": {}, "by_layer": {}, "embed_split": {}, "sft_only_keys": sft_only}
    for r in rows:
        if r["group"] == "layer":
            summary["by_layer"].setdefault(str(r["layer"]), {})[r["model"]] = r["r_l"]
        else:
            summary["by_group"].setdefault(r["group"], {})[r["model"]] = r["r_l"]
    for r in embed_split:
        k = "{}_{}".format(r["module"], r["slice"])
        summary["embed_split"].setdefault(k, {})[r["model"]] = r["r_l"]

    # backbone 全层均值
    layer_rs = defaultdict(list)
    for r in rows:
        if r["group"] == "layer":
            layer_rs[r["model"]].append(r["r_l"])
    summary["backbone_mean"] = {m: float(np.mean(v)) for m, v in layer_rs.items()}
    return summary


def plot_layer_lines(rows, out_dir):
    """折线图：layer 0~N，CPT vs SFT"""
    style = {
        "cpt": {"color": "#2563EB", "label": "CPT"},
        "sft": {"color": "#EA580C", "label": "SFT"},
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.4,
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)

    series = {}
    for model, cfg in style.items():
        sub = sorted(
            [r for r in rows if r["model"] == model and r["group"] == "layer"],
            key=lambda x: x["layer"],
        )
        if sub:
            series[model] = {
                "xs": [r["layer"] for r in sub],
                "ys": [r["r_l"] for r in sub],
                **cfg,
            }

    all_ys = [y for s in series.values() for y in s["ys"]]
    y_min = min(all_ys) * 0.95
    y_max = max(all_ys) * 1.05
    ax.set_ylim(y_min, y_max)

    for s in series.values():
        ax.plot(
            s["xs"], s["ys"],
            label=s["label"],
            color=s["color"],
            marker="o",
            markersize=9,
            markevery=4,
            markerfacecolor="white",
            markeredgewidth=2.8,
            markeredgecolor=s["color"],
            linewidth=4.0,
            zorder=3,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(r"$r_\ell$ (Relative Frobenius Norm)")
    ax.set_xlim(-0.5, max(r["layer"] for r in rows if r["group"] == "layer") + 0.5)
    ax.set_xticks(range(0, 36, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.55, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        borderpad=0.8,
    )
    leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_dir / "layer_lines.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Plot saved: {}".format(out_dir / "layer_lines.png"))


def plot_module_bars(rows, embed_split, out_dir):
    """分组柱状图：embed / backbone均值 / lm_head / new_tokens"""
    def get_r(model, group):
        for r in rows:
            if r["model"] == model and r["group"] == group:
                return r["r_l"]
        return 0.0

    def mean_new_token(model):
        vals = [r["r_l"] for r in embed_split
                if r["model"] == model and r["slice"] == "new_tokens"]
        return float(np.mean(vals)) if vals else 0.0

    groups = [
        ("embed_tokens", lambda m: get_r(m, "embed_tokens")),
        ("backbone\n(layer mean)", lambda m: float(np.mean(
            [r["r_l"] for r in rows if r["model"] == m and r["group"] == "layer"] or [0]))),
        ("lm_head", lambda m: get_r(m, "lm_head")),
        ("new_tokens\n(embed+lm_head)", mean_new_token),
    ]

    x = np.arange(len(groups))
    w = 0.35
    cpt = [fn("cpt") for _, fn in groups]
    sft = [fn("sft") for _, fn in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, cpt, w, label="CPT", color="tab:blue")
    ax.bar(x + w / 2, sft, w, label="SFT", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("r_l")
    ax.set_title("Mean Relative Frobenius Norm by Module Group")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "module_bars.png", dpi=150)
    plt.close(fig)
    print("Plot saved: {}".format(out_dir / "module_bars.png"))


def plot_embed_split(embed_split, out_dir):
    if not embed_split:
        return
    slices = ["original_vocab", "new_tokens"]
    x = np.arange(len(slices))
    w = 0.35

    def mean_slice(model, sl):
        vals = [r["r_l"] for r in embed_split if r["model"] == model and r["slice"] == sl]
        return float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, [mean_slice("cpt", s) for s in slices], w, label="CPT", color="tab:blue")
    ax.bar(x + w / 2, [mean_slice("sft", s) for s in slices], w, label="SFT", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(["original vocab", "new tokens"])
    ax.set_ylabel("r_l")
    ax.set_title("Embed / LM Head: Original vs New Tokens")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "embed_split.png", dpi=150)
    plt.close(fig)
    print("Plot saved: {}".format(out_dir / "embed_split.png"))


def main():
    p = argparse.ArgumentParser(description="CPT vs SFT Delta-W analysis")
    p.add_argument("--base", required=True)
    p.add_argument("--cpt", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--original_vocab_size", type=int, default=DEFAULT_ORIGINAL_VOCAB_SIZE)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base, cpt, sft = Path(args.base), Path(args.cpt), Path(args.sft)
    print("Base:", base, "\nCPT:", cpt, "\nSFT:", sft, "\nOut:", out)

    bm, cm, sm = load_weight_map(base), load_weight_map(cpt), load_weight_map(sft)
    rows, embed_split, sft_only = [], [], {"cpt": [], "sft": []}

    for label, td, tm in [("cpt", cpt, cm), ("sft", sft, sm)]:
        r, e, o, _ = analyze_pair(base, td, label, bm, tm, args.original_vocab_size)
        rows.extend(r)
        embed_split.extend(e)
        sft_only[label] = o

    # CSV：按层聚合后的主结果
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "group", "layer", "r_l"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "embed_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "module", "slice", "r_l"])
        w.writeheader()
        w.writerows(embed_split)

    summary = build_summary(rows, embed_split, sft_only)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_layer_lines(rows, out)
    plot_module_bars(rows, embed_split, out)
    plot_embed_split(embed_split, out)
    print("Done. Outputs in", out)


if __name__ == "__main__":
    main()
