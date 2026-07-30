#!/usr/bin/env python3
"""从已下载的 RecIF Parquet 中提取预训练数据。"""

import argparse
import json
import random
import uuid
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


SID_FORMAT = "<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>"
HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10
ITEM_TEMPLATES = (
    lambda sid, caption: json.dumps(
        {"视频ID": sid, "视频内容": caption}, ensure_ascii=False
    ),
    lambda sid, caption: f"视频{sid} 展示了以下内容：{caption}",
    lambda sid, caption: f"视频{sid} 的内容完整描述如下：{caption}",
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    openonerec_dir = script_dir.parent
    default_data_dir = openonerec_dir / "RecIF"
    default_output_dir = openonerec_dir / "data" / "pretrain_parquet"
    parser = argparse.ArgumentParser(
        description="从 RecIF 原始 Parquet 提取 OpenOneRec 预训练数据"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="RecIF 数据目录（默认：../RecIF）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="输出目录（默认：../data/pretrain_parquet）",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("video_rec", "item_understand"),
        default=("video_rec", "item_understand"),
        help="要提取的任务，默认全部提取",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pid2sid(path):
    print(f"读取 SID 映射：{path}")
    mapping = pd.read_parquet(path, columns=["pid", "sid"])
    return dict(zip(mapping["pid"], mapping["sid"]))


def make_sid(pid, pid2sid):
    code = pid2sid.get(pid)
    if code is None or len(code) < 3:
        return ""
    return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])


def make_sids(pids, pid2sid):
    if pids is None:
        return ""
    return "".join(filter(None, (make_sid(pid, pid2sid) for pid in pids)))


def segments(text):
    return json.dumps([{"type": "text", "text": text}], ensure_ascii=False)


def write_parquet(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["source", "uuid", "segments", "metadata"])
    frame.to_parquet(output_path, index=False)
    print(f"写入：{output_path}（{len(frame):,} 条）")


def extract_video_rec(metadata_path, pid2sid, output_path):
    columns = ["uid", "split", "hist_video_pid", "target_video_pid"]
    frame = pd.read_parquet(metadata_path, columns=columns)
    rows = []
    for row in tqdm(
        frame.itertuples(index=False), total=len(frame), desc="video_rec"
    ):
        if row.split != 0 or row.hist_video_pid is None or row.target_video_pid is None:
            continue
        history = make_sids(row.hist_video_pid[-HIST_MAX_LEN:], pid2sid)
        target = make_sids(row.target_video_pid[:TARGET_MAX_LEN], pid2sid)
        if not history or not target:
            continue
        rows.append(
            {
                "source": "RecIF_VideoRec_Pretrain",
                "uuid": str(uuid.uuid4()),
                "segments": segments(history + target),
                "metadata": json.dumps({"uid": int(row.uid)}, ensure_ascii=False),
            }
        )
    write_parquet(rows, output_path)


def extract_item_understand(caption_path, pid2sid, output_path, seed):
    random.seed(seed)
    try:
        frame = pd.read_parquet(caption_path, columns=["pid", "dense_caption"])
        caption_column = "dense_caption"
    except (KeyError, ValueError):
        frame = pd.read_parquet(caption_path, columns=["pid", "caption"])
        caption_column = "caption"

    rows = []
    for row in tqdm(
        frame.itertuples(index=False), total=len(frame), desc="item_understand"
    ):
        caption = getattr(row, caption_column)
        if not isinstance(caption, str) or not caption:
            continue
        sid = make_sid(row.pid, pid2sid)
        if not sid:
            continue
        text = random.choice(ITEM_TEMPLATES)(sid, caption)
        rows.append(
            {
                "source": "RecIF_ItemUnderstand_Pretrain",
                "uuid": str(uuid.uuid4()),
                "segments": segments(text),
                "metadata": json.dumps(
                    {"pid": int(row.pid), "sid": sid}, ensure_ascii=False
                ),
            }
        )
    write_parquet(rows, output_path)


def main():
    args = parse_args()
    metadata_path = args.data_dir / "onerec_bench_release.parquet"
    pid2sid_path = args.data_dir / "video_ad_pid2sid.parquet"
    caption_path = args.data_dir / "pid2caption.parquet"

    required = [metadata_path]
    if "video_rec" in args.tasks or "item_understand" in args.tasks:
        required.append(pid2sid_path)
    if "item_understand" in args.tasks:
        required.append(caption_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    pid2sid = None
    if "video_rec" in args.tasks or "item_understand" in args.tasks:
        pid2sid = load_pid2sid(pid2sid_path)

    if "video_rec" in args.tasks:
        extract_video_rec(
            metadata_path, pid2sid, args.output_dir / "pretrain_video_rec.parquet"
        )
    if "item_understand" in args.tasks:
        extract_item_understand(
            caption_path,
            pid2sid,
            args.output_dir / "pretrain_item_understand.parquet",
            args.seed,
        )


if __name__ == "__main__":
    main()
