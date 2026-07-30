#!/usr/bin/env python3
"""将 RecIF 原始数据转为 Alpaca SFT JSONL。

支持两类任务：
1. sid_caption：SID -> caption 语义理解
2. video_rec：视频推荐（新 template），并额外输出按 target 拆条扩样的版本

用法：
  python prepare_sft.py --tasks sid_caption video_rec
  python prepare_sft.py --tasks video_rec
"""

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

# video_rec 新 template，参考 convert_video_rec_format.py
VIDEO_INSTRUCTION = (
    "你是一个视频推荐引擎，通过学习用户的视频观看历史序列，"
    "预测用户可能感兴趣的下一个视频内容。"
)
VIDEO_TASK_DESC = "基于用户的视频观看历史行为序列，完成以下个性化视频推荐任务。"
VIDEO_SUBTASK_DESC = (
    "下方数据记录了用户历史观看过的视频序列。"
    "请分析用户的观看行为模式，推断用户当前的观看兴趣，"
    "输出最可能感兴趣的下一批视频推荐序列。"
)

# sid_caption prompts，沿用 item_understand / train.jsonl 风格
CAPTION_SYSTEM_PROMPTS = [
    "你是一名视频描述生成器，请根据下面的视频token生成视频描述。",
    "你是一个专业的视频内容分析助手，能够理解视频token并生成准确的描述。",
    "你是一位视频理解专家，擅长将视频token转换为详细的文字描述。",
    "作为视频内容解析助手，你需要根据视频token提供精准的内容描述。",
    "你是一个智能视频解说员，可以根据视频token创建生动的描述。",
    "你具备理解视频token并生成高质量描述的能力。",
    "你是视频内容描述专家，能够将视频token转化为易懂的文字说明。",
    "作为AI视频分析助手，你可以根据视频token生成详细准确的描述。",
    "作为视频理解专家，你的任务是将视频token转换为准确的文字描述。",
    "你是一个多媒体内容解读助手，能够解析视频token并生成相应描述。",
]
CAPTION_USER_PROMPTS = [
    "请描述 {sid} 的内容",
    "这段视频 {sid} 展示了什么？",
    "请解释 {sid} 中的内容",
    "能否说明 {sid} 里发生了什么？",
    "请分析 {sid} 的具体内容",
    "{sid} 这个视频讲的是什么？",
    "请详细描述 {sid}",
    "告诉我 {sid} 的内容是什么",
    "请为 {sid} 生成描述",
    "{sid} 包含哪些内容？",
    "请说明视频 {sid} 的主要内容",
    "描述一下 {sid} 中展现的场景",
    "{sid} 这段内容是关于什么的？",
    "请解读 {sid} 的视频内容",
    "能描述下 {sid} 吗？",
    "{sid} 里面有什么？",
    "请对 {sid} 进行内容说明",
    "这个 {sid} 是什么内容？",
    "分析 {sid} 并给出描述",
    "请阐述 {sid} 的内容细节",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    openonerec_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="准备 OpenOneRec SFT JSONL 数据")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=openonerec_dir / "RecIF",
        help="RecIF 原始数据目录（默认：../RecIF）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=openonerec_dir / "data" / "sft_jsonl",
        help="SFT JSONL 输出目录（默认：../data/sft_jsonl）",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("sid_caption", "video_rec"),
        default=("sid_caption", "video_rec"),
        help="要处理的任务",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hist-max-len",
        type=int,
        default=HIST_MAX_LEN,
        help="video_rec 历史序列最大长度",
    )
    parser.add_argument(
        "--target-max-len",
        type=int,
        default=TARGET_MAX_LEN,
        help="video_rec 每个样本最多保留的 target 数",
    )
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
        return []
    sids = []
    for pid in pids:
        sid = make_sid(pid, pid2sid)
        if sid:
            sids.append(sid)
    return sids


def open_jsonl_writer(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def write_row(fout, row):
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_video_input(hist_sids):
    return (
        f"<task>{VIDEO_TASK_DESC}</task>\n"
        f"<subtask>{VIDEO_SUBTASK_DESC}</subtask>\n\n"
        f"用户观看过的视频：{''.join(hist_sids)}"
    )


def prepare_sid_caption(data_dir, output_dir, seed):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    caption_path = data_dir / "pid2caption.parquet"
    pid2sid_path = data_dir / "video_ad_pid2sid.parquet"
    for path in (caption_path, pid2sid_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")

    random.seed(seed)
    pid2sid = load_pid2sid(pid2sid_path)

    try:
        frame = pd.read_parquet(caption_path, columns=["pid", "dense_caption"])
        caption_column = "dense_caption"
    except (KeyError, ValueError):
        frame = pd.read_parquet(caption_path, columns=["pid", "caption"])
        caption_column = "caption"

    output_path = output_dir / "sft_sid_caption.jsonl"
    count = 0
    with open_jsonl_writer(output_path) as fout:
        for row in tqdm(
            frame.itertuples(index=False), total=len(frame), desc="sid_caption"
        ):
            caption = getattr(row, caption_column)
            if not isinstance(caption, str) or not caption.strip():
                continue
            sid = make_sid(row.pid, pid2sid)
            if not sid:
                continue
            write_row(
                fout,
                {
                    "metadata": {
                        "pid": int(row.pid),
                        "sid": sid,
                        "domain": "video_ad",
                        "direction": "sid2caption",
                    },
                    "uuid": str(uuid.uuid4()),
                    "source": "RecIF_SidCaption",
                    "instruction": random.choice(CAPTION_SYSTEM_PROMPTS),
                    "input": random.choice(CAPTION_USER_PROMPTS).format(sid=sid),
                    "output": caption.strip(),
                },
            )
            count += 1
    print(f"写入：{output_path}（{count:,} 条）")


def prepare_video_rec(data_dir, output_dir, hist_max_len, target_max_len):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    metadata_path = data_dir / "onerec_bench_release.parquet"
    pid2sid_path = data_dir / "video_ad_pid2sid.parquet"
    for path in (metadata_path, pid2sid_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")

    pid2sid = load_pid2sid(pid2sid_path)
    columns = ["uid", "split", "hist_video_pid", "target_video_pid"]
    frame = pd.read_parquet(metadata_path, columns=columns)

    multi_path = output_dir / "sft_video_rec.jsonl"
    expand_path = output_dir / "sft_video_rec_expand.jsonl"
    multi_count = 0
    expand_count = 0

    with open_jsonl_writer(multi_path) as multi_fout, open_jsonl_writer(
        expand_path
    ) as expand_fout:
        for row in tqdm(
            frame.itertuples(index=False), total=len(frame), desc="video_rec"
        ):
            if (
                row.split != 0
                or row.hist_video_pid is None
                or row.target_video_pid is None
            ):
                continue

            hist_sids = make_sids(row.hist_video_pid[-hist_max_len:], pid2sid)
            target_sids = make_sids(row.target_video_pid[:target_max_len], pid2sid)
            if not hist_sids or not target_sids:
                continue

            uid = int(row.uid)
            input_text = build_video_input(hist_sids)
            base = {
                "source": "RecIF_VideoRec",
                "uid": uid,
                "instruction": VIDEO_INSTRUCTION,
                "input": input_text,
            }

            # 1) 原始规模：一次输出全部 target
            sample = dict(base)
            sample["output"] = "".join(target_sids)
            write_row(multi_fout, sample)
            multi_count += 1

            # 2) 扩样：每个 target 单独成一条，约扩大 target_max_len 倍
            for target_sid in target_sids:
                sample = dict(base)
                sample["output"] = target_sid
                write_row(expand_fout, sample)
                expand_count += 1

    print(f"写入：{multi_path}（{multi_count:,} 条）")
    print(f"写入：{expand_path}（{expand_count:,} 条）")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "sid_caption" in args.tasks:
        print("=" * 60)
        print("处理 sid_caption")
        print("=" * 60)
        prepare_sid_caption(args.data_dir, args.output_dir, args.seed)

    if "video_rec" in args.tasks:
        print("=" * 60)
        print("处理 video_rec（新 template + 扩样）")
        print("=" * 60)
        prepare_video_rec(
            args.data_dir,
            args.output_dir,
            args.hist_max_len,
            args.target_max_len,
        )

    print("全部完成。")


if __name__ == "__main__":
    main()
