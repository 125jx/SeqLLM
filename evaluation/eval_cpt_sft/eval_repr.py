#!/usr/bin/env python3
"""
语言认知探针：base / CPT / SFT 三个 checkpoint 在纯语言输入上的"表示/输出"漂移

对应分析（喂纯中文语言文本，不带 prefix / 行为 token）：
  ① 输入 token embedding 表：对共享词表 token 算 cosine(base_emb[w], model_emb[w]) 取平均
     （直接读权重，不需前向）
  ② 各层 hidden state 对 base 的相似度（主图）：mean-pool over valid tokens 得每层向量，
     算 base vs CPT、base vs SFT 的 linear-CKA + per-example cosine，逐层画曲线
  ③ next-token 分布对 base 的 KL：KL(p_base || p_model) 在共享词表上（renormalize）逐 token 平均

预期：SFT 全程贴近 base（KL 小、相似度≈1），CPT 在深层急剧下滑、KL 大（表示被改写 → 遗忘）。

内存策略（单卡即可，逐个模型加载）：
  - 用 base tokenizer 统一分词一次，同一批 input_ids 喂给三个模型（纯中文文本不会用到新增 token，对齐成立）
  - base pass：缓存逐层 mean-pool hidden + 把 next-token log-prob 写到磁盘 memmap
  - cpt/sft pass：前向时逐位置载入 base 的 log-prob，当场累加 KL，只额外缓存 hidden
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_ORIGINAL_VOCAB_SIZE = 151643


# --------------------------- 探针数据 ---------------------------
def load_probe_texts(args):
    """返回纯语言文本列表。支持 txt / jsonl / lm_eval log_samples。"""
    texts = []
    path = Path(args.probe_file)
    if args.probe_format == "txt":
        texts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    elif args.probe_format == "jsonl":
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            t = obj.get(args.text_key)
            if t is None:
                # 兜底：常见字段
                for k in ("text", "question", "prompt", "content", "doc"):
                    if k in obj and isinstance(obj[k], str):
                        t = obj[k]
                        break
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    elif args.probe_format == "lm_eval":
        # lm_eval --log_samples 的 jsonl：从 doc 里拼出题干纯文本
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            doc = obj.get("doc", obj)
            q = doc.get("question") or doc.get("query") or ""
            opts = []
            for key in ("A", "B", "C", "D"):
                if doc.get(key):
                    opts.append("{}. {}".format(key, doc[key]))
            t = (q + ("\n" + "\n".join(opts) if opts else "")).strip()
            if t:
                texts.append(t)
    else:
        raise ValueError("Unknown probe_format: {}".format(args.probe_format))

    if not texts:
        raise RuntimeError("No probe texts loaded from {}".format(path))
    if args.n_samples > 0:
        texts = texts[: args.n_samples]
    print("Loaded {} probe texts from {} ({})".format(len(texts), path, args.probe_format))
    return texts


# --------------------------- embedding 权重 cosine（①）---------------------------
def load_weight_map(model_dir):
    p = model_dir / "model.safetensors.index.json"
    if p.exists():
        return json.load(open(p))["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        from safetensors.torch import load_file
        return {k: str(single) for k in load_file(str(single)).keys()}
    raise FileNotFoundError("No safetensors in {}".format(model_dir))


def load_named_tensor(model_dir, weight_map, key):
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def embedding_cosine(base_dir, model_dir, shared_vocab):
    """对共享词表 token 逐行算 cosine，取平均。返回 embed_tokens / lm_head 两个数。"""
    out = {}
    bm, mm = load_weight_map(base_dir), load_weight_map(model_dir)
    for key, name in (("model.embed_tokens.weight", "embed_tokens"), ("lm_head.weight", "lm_head")):
        if key not in bm or key not in mm:
            continue
        wb = load_named_tensor(base_dir, bm, key).float()[:shared_vocab]
        wm = load_named_tensor(model_dir, mm, key).float()[:shared_vocab]
        cos = torch.nn.functional.cosine_similarity(wb, wm, dim=1)
        out[name] = float(cos.mean())
    return out


# --------------------------- 前向：hidden + logprob ---------------------------
def load_model(path, dtype):
    print("Loading model:", path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    return model


@torch.no_grad()
def forward_collect(model, batches, num_layers, hidden_size, shared_vocab,
                    logprob_memmap=None, base_logprob_memmap=None):
    """
    对一批已 tokenize 的样本前向，收集：
      - mean_hidden: [N, num_layers+1, H]（每层对有效 token 做 mean-pool）
      - 若 logprob_memmap 给定：把每个采样位置的 next-token log-prob（共享词表, renorm）写入
      - 若 base_logprob_memmap 给定：读回 base logprob，当场累加 KL(base || model)
    返回 (mean_hidden, kl_sum, kl_count)
    """
    N = sum(len(b["input_ids"]) for b in batches)
    mean_hidden = np.zeros((N, num_layers + 1, hidden_size), dtype=np.float16)
    kl_sum, kl_count = 0.0, 0
    pos_cursor = 0  # 全局采样位置游标（与 base pass 的写入顺序一致）
    sample_idx = 0
    device = next(model.parameters()).device

    for b in batches:
        input_ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        hs = out.hidden_states  # tuple(len=num_layers+1) of [B, T, H]
        logits = out.logits      # [B, T, V]

        # ① 逐层 mean-pool
        mask = attn.unsqueeze(-1).float()  # [B,T,1]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B,1]
        for li, h in enumerate(hs):
            pooled = (h.float() * mask).sum(dim=1) / denom  # [B,H]
            bs = pooled.shape[0]
            mean_hidden[sample_idx:sample_idx + bs, li, :] = pooled.cpu().numpy().astype(np.float16)

        # ③ next-token 分布：位置 t 预测 token t+1，取每个样本前 kl_positions 个有效位置
        if logprob_memmap is not None or base_logprob_memmap is not None:
            lp_full = torch.log_softmax(logits[:, :, :shared_vocab].float(), dim=-1)  # renorm over shared vocab
            B, T = input_ids.shape
            for bi in range(B):
                valid = int(attn[bi].sum().item())
                # 预测位置为 0..valid-2（最后一个位置没有 target，这里只取分布本身不需 target）
                n_pos = min(valid, b["kl_pos"][bi])
                for t in range(n_pos):
                    lp = lp_full[bi, t].cpu().numpy().astype(np.float16)
                    if logprob_memmap is not None:
                        logprob_memmap[pos_cursor] = lp
                    if base_logprob_memmap is not None:
                        base_lp = base_logprob_memmap[pos_cursor].astype(np.float32)
                        p_base = np.exp(base_lp)
                        kl = float(np.sum(p_base * (base_lp - lp.astype(np.float32))))
                        kl_sum += kl
                        kl_count += 1
                    pos_cursor += 1

        sample_idx += input_ids.shape[0]

    return mean_hidden, kl_sum, kl_count


# --------------------------- 相似度度量 ---------------------------
def linear_cka(X, Y):
    """linear CKA between [N,Hx] and [N,Hy]（对旋转/缩放鲁棒）。"""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    if xx < 1e-12 or yy < 1e-12:
        return float("nan")
    return float(xy / (xx * yy))


def per_example_cosine(X, Y):
    """逐样本 cosine(X_i, Y_i) 取平均。"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(Xn * Yn, axis=1)))


# --------------------------- 画图 ---------------------------
def plot_layer_similarity(layer_rows, out_dir, metric):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in (("cpt", "tab:blue"), ("sft", "tab:orange")):
        sub = sorted([r for r in layer_rows if r["model"] == model], key=lambda x: x["layer"])
        if not sub:
            continue
        ax.plot([r["layer"] for r in sub], [r[metric] for r in sub],
                label=model.upper(), color=color, marker="o", markersize=3, linewidth=2)
    ax.set_xlabel("Layer (0 = embedding output)")
    ax.set_ylabel("{} vs base".format(metric.upper()))
    ax.set_title("Per-layer representation similarity to base ({})".format(metric.upper()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = out_dir / "layer_{}.png".format(metric)
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print("Plot saved:", fp)


# --------------------------- 主流程 ---------------------------
def build_batches(tokenizer, texts, max_len, batch_size, kl_positions):
    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        valid = enc["attention_mask"].sum(dim=1).tolist()
        kl_pos = [min(int(v), kl_positions) for v in valid]
        batches.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "kl_pos": kl_pos,
        })
    return batches


def total_kl_positions(batches):
    return int(sum(sum(b["kl_pos"]) for b in batches))


def main():
    ap = argparse.ArgumentParser(description="语言认知探针：base/CPT/SFT 表示&输出漂移")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--probe_file", required=True)
    ap.add_argument("--probe_format", default="txt", choices=["txt", "jsonl", "lm_eval"])
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--kl_positions", type=int, default=64,
                    help="每条样本参与 KL 平均的有效位置数上限")
    ap.add_argument("--original_vocab_size", type=int, default=DEFAULT_ORIGINAL_VOCAB_SIZE)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    base_dir, cpt_dir, sft_dir = Path(args.base), Path(args.cpt), Path(args.sft)

    # 用 base tokenizer 统一分词，保证三模型 next-token 位置严格对齐
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = load_probe_texts(args)
    batches = build_batches(tokenizer, texts, args.max_len, args.batch_size, args.kl_positions)
    n_pos = total_kl_positions(batches)
    shared_vocab = args.original_vocab_size
    print("Total KL positions: {}, shared_vocab: {}".format(n_pos, shared_vocab))

    # base logprob 落盘 memmap（float16）
    base_lp_path = out / "base_logprob.f16.memmap"
    base_lp = np.memmap(str(base_lp_path), dtype=np.float16, mode="w+", shape=(n_pos, shared_vocab))

    # ---- Pass 1: base ----
    base_model = load_model(str(base_dir), dtype)
    cfg = base_model.config
    num_layers, hidden_size = cfg.num_hidden_layers, cfg.hidden_size
    base_hidden, _, _ = forward_collect(
        base_model, batches, num_layers, hidden_size, shared_vocab, logprob_memmap=base_lp)
    base_lp.flush()
    del base_model
    torch.cuda.empty_cache()

    # ---- Pass 2/3: cpt / sft ----
    layer_rows = []
    kl_summary = {}
    for label, mdir in (("cpt", cpt_dir), ("sft", sft_dir)):
        model = load_model(str(mdir), dtype)
        mh, kl_sum, kl_cnt = forward_collect(
            model, batches, num_layers, hidden_size, shared_vocab, base_logprob_memmap=base_lp)
        del model
        torch.cuda.empty_cache()

        for li in range(num_layers + 1):
            X = base_hidden[:, li, :].astype(np.float32)
            Y = mh[:, li, :].astype(np.float32)
            layer_rows.append({
                "model": label, "layer": li,
                "cka": linear_cka(X, Y),
                "cosine": per_example_cosine(X, Y),
            })
        kl_summary[label] = kl_sum / max(kl_cnt, 1)
        print("[{}] mean KL(base||model) = {:.6f} over {} positions".format(label, kl_summary[label], kl_cnt))

    # ---- ① embedding cosine ----
    emb_summary = {
        "cpt": embedding_cosine(base_dir, cpt_dir, shared_vocab),
        "sft": embedding_cosine(base_dir, sft_dir, shared_vocab),
    }

    # ---- 输出 ----
    with open(out / "layer_similarity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "cka", "cosine"])
        w.writeheader()
        w.writerows(layer_rows)

    summary = {
        "embedding_cosine": emb_summary,      # ① 越接近 1 越像 base
        "next_token_kl": kl_summary,          # ③ 越大越偏离 base（遗忘）
        "layer_similarity_head_tail": {       # 便捷查看首/尾层
            m: {
                "layer0_cka": next(r["cka"] for r in layer_rows if r["model"] == m and r["layer"] == 0),
                "last_layer_cka": next(r["cka"] for r in layer_rows if r["model"] == m and r["layer"] == num_layers),
                "last_layer_cosine": next(r["cosine"] for r in layer_rows if r["model"] == m and r["layer"] == num_layers),
            } for m in ("cpt", "sft")
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_layer_similarity(layer_rows, out, "cka")
    plot_layer_similarity(layer_rows, out, "cosine")

    # 清理大文件
    del base_lp
    try:
        base_lp_path.unlink()
    except OSError:
        pass

    print("\nDone. Outputs in", out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
