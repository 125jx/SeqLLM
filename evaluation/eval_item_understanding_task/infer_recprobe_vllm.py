#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecProbe —— vLLM 批量推理，产出判分用的 preds.jsonl
======================================================================
读 build_recprobe.py 生成的评测集 (含 prompt/task_id)，用指定 ckpt 跑一遍，
每行写 {"task_id": ..., "output": "<模型原文>"}，直接喂给 score_recprobe.py。

默认：关闭 thinking + 贪心解码 (temperature=0)。
  RecProbe 是确定性选择/排序题，关 thinking 输出直接给「答案：X」，判分最稳。
  如需让模型显式推理，加 --enable-thinking（会同时切到采样解码 temp=0.6）。

用法：
    python3 infer_recprobe_vllm.py \
        --model /apdcephfs/zjx121/apdcephfs_cq11/share_303717182/bobjxzhang/Qwen/OneRec-8B \
        --gold  ./all_tasks.jsonl \
        --out   ./preds_onerec8b.jsonl \
        --tp 8

随后判分：
    python3 score_recprobe.py --gold ./all_tasks.jsonl \
        --pred ./preds_onerec8b.jsonl --report ./score_onerec8b.json
"""
import argparse
import json
import os
import sys


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ckpt 目录")
    ap.add_argument("--gold", default="./all_tasks.jsonl",
                    help="评测集 (含 prompt/task_id)")
    ap.add_argument("--out", required=True, help="预测输出 jsonl")
    ap.add_argument("--tp", type=int, default=8, help="tensor parallel size = GPU 数")
    ap.add_argument("--gpu-mem", type=float, default=0.30, help="gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0=自动(thinking 时 2048，否则 64)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="开启 Qwen3 思考模式 (默认关闭)")
    ap.add_argument("--temperature", type=float, default=-1.0,
                    help="<0=自动(thinking 时 0.6，否则 0 贪心)")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条 (调试)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = load_jsonl(args.gold)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        print("[ERROR] 评测集为空", file=sys.stderr)
        sys.exit(1)
    print(f"[load] {len(rows)} 条评测样本 <- {args.gold}", flush=True)

    # ---- 解码参数 ----
    think = args.enable_thinking
    temp = args.temperature if args.temperature >= 0 else (0.6 if think else 0.0)
    max_new = args.max_new_tokens if args.max_new_tokens > 0 else (2048 if think else 64)
    print(f"[cfg] thinking={think}  temperature={temp}  max_new_tokens={max_new}  "
          f"tp={args.tp}", flush=True)

    # ---- chat template (prompt 自带角色前缀，只放 user) ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        try:
            t = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=think)
        except TypeError:
            # 个别 tokenizer 不认 enable_thinking kwarg
            t = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t)

    # ---- vLLM ----
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        temperature=temp,
        top_p=0.95 if temp > 0 else 1.0,
        top_k=20 if temp > 0 else -1,
        max_tokens=max_new,
    )

    outputs = llm.generate(texts, sp)

    # ---- 写出 (vLLM 保序返回) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outputs):
            text = o.outputs[0].text if o.outputs else ""
            f.write(json.dumps(
                {"task_id": r["task_id"], "output": text},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"[done] 写出 {n} 条预测 -> {args.out}", flush=True)
    print(f"[next] python3 score_recprobe.py --gold {args.gold} "
          f"--pred {args.out} --report {os.path.splitext(args.out)[0]}_score.json",
          flush=True)


if __name__ == "__main__":
    main()
