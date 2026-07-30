#!/usr/bin/env python3
"""
从 C-Eval 验证集导出 probe_ceval.txt（每行一条纯中文题干+选项，不带 prefix）。

在太极 / 评测环境里运行（需要 datasets，可联网下载数据）：
  python3 prepare_probe_ceval.py
  python3 prepare_probe_ceval.py --output data/probe_ceval.txt --n_samples 500

也可从 lm_eval --log_samples 的 jsonl 转换：
  python3 prepare_probe_ceval.py --from_lm_eval /path/to/samples.jsonl
"""
import argparse
import json
import random
from pathlib import Path


def format_ceval_row(row):
    q = (row.get("question") or row.get("query") or "").strip()
    opts = []
    for key in ("A", "B", "C", "D"):
        if row.get(key):
            opts.append("{}. {}".format(key, row[key]))
    text = q
    if opts:
        text = text + "\n" + "\n".join(opts)
    return text.strip()


def load_from_datasets(n_samples, seed):
    from datasets import get_dataset_config_names, load_dataset

    names = get_dataset_config_names("ceval/ceval-exam")
    rows = []
    for name in names:
        ds = load_dataset("ceval/ceval-exam", name, split="val", trust_remote_code=True)
        for row in ds:
            rows.append(format_ceval_row(row))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = [r for r in rows if r]
    if n_samples > 0:
        rows = rows[:n_samples]
    return rows


def load_from_lm_eval(path, n_samples):
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        doc = obj.get("doc", obj)
        text = format_ceval_row(doc)
        if text:
            rows.append(text)
    if n_samples > 0:
        rows = rows[:n_samples]
    return rows


def main():
    ap = argparse.ArgumentParser(description="Prepare probe_ceval.txt for eval_repr.py")
    ap.add_argument("--output", default=None, help="default: <script_dir>/data/probe_ceval.txt")
    ap.add_argument("--n_samples", type=int, default=500, help="0 = keep all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_lm_eval", default=None, help="lm_eval --log_samples jsonl path")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out = Path(args.output) if args.output else script_dir / "data" / "probe_ceval.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_lm_eval:
        rows = load_from_lm_eval(args.from_lm_eval, args.n_samples)
        source = "lm_eval:{}".format(args.from_lm_eval)
    else:
        rows = load_from_datasets(args.n_samples, args.seed)
        source = "ceval/ceval-exam (val, all subjects)"

    if not rows:
        raise RuntimeError("No probe texts exported")

    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("Wrote {} lines to {}".format(len(rows), out))
    print("Source:", source)
    print("Example:\n", rows[0][:200], "...")


if __name__ == "__main__":
    main()
