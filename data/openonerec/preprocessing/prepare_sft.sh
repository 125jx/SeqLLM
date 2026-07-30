#!/usr/bin/env bash
# 将 RecIF 转为 SFT JSONL：sid_caption + video_rec（含扩样）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENONEREC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${OPENONEREC_DIR}/RecIF}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPENONEREC_DIR}/data/sft_jsonl}"

echo "准备 SFT 数据"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sft.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tasks sid_caption video_rec

echo "完成，输出目录：${OUTPUT_DIR}"
echo "  - sft_sid_caption.jsonl"
echo "  - sft_video_rec.jsonl"
echo "  - sft_video_rec_expand.jsonl"
