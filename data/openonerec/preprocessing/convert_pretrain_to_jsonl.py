#!/usr/bin/env python3
"""将提取出的 pretrain Parquet 转成每行 {"text": "..."} 的 JSONL。"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    parser = argparse.ArgumentParser(description="将 pretrain Parquet 转为 JSONL")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=data_dir / "pretrain_parquet",
        help="输入 Parquet 文件或目录（默认：../data/pretrain_parquet）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=data_dir / "pretrain_jsonl",
        help="输出 JSONL 文件或目录（默认：../data/pretrain_jsonl）",
    )
    return parser.parse_args()


def extract_text(raw_segments):
    if isinstance(raw_segments, str):
        raw_segments = json.loads(raw_segments)
    if not isinstance(raw_segments, (list, tuple)):
        return None
    parts = [
        segment["text"]
        for segment in raw_segments
        if isinstance(segment, dict)
        and segment.get("type") == "text"
        and isinstance(segment.get("text"), str)
    ]
    return "".join(parts) if parts else None


def convert_file(input_path, output_path):
    frame = pd.read_parquet(input_path, columns=["segments"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for raw_segments in tqdm(
            frame["segments"], total=len(frame), desc=input_path.stem
        ):
            try:
                text = extract_text(raw_segments)
            except (json.JSONDecodeError, TypeError):
                text = None
            if not text:
                skipped += 1
                continue
            output_file.write(
                json.dumps({"text": text}, ensure_ascii=False) + "\n"
            )
            written += 1

    print(f"写入：{output_path}（{written:,} 条，跳过 {skipped:,} 条）")


def main():
    args = parse_args()
    if args.input.is_file():
        output_path = args.output
        if output_path.suffix.lower() != ".jsonl":
            output_path = output_path / f"{args.input.stem}.jsonl"
        convert_file(args.input, output_path)
        return

    if not args.input.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{args.input}")

    input_files = sorted(args.input.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{args.input}")
    for input_path in input_files:
        convert_file(input_path, args.output / f"{input_path.stem}.jsonl")


if __name__ == "__main__":
    main()
