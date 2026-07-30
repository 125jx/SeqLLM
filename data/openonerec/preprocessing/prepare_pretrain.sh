#!/usr/bin/env bash
# 依次从 RecIF 提取预训练 Parquet，再转换成 JSONL。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENONEREC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${OPENONEREC_DIR}/RecIF}"
PARQUET_DIR="${PARQUET_DIR:-${OPENONEREC_DIR}/data/pretrain_parquet}"
JSONL_DIR="${JSONL_DIR:-${OPENONEREC_DIR}/data/pretrain_jsonl}"

echo "[1/2] 提取 video_rec 和 item_understand 预训练数据"
"${PYTHON_BIN}" "${SCRIPT_DIR}/extract_pretrain.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${PARQUET_DIR}" \
    --tasks video_rec item_understand

echo "[2/2] 转换 Parquet 为 JSONL"
"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_pretrain_to_jsonl.py" \
    --input "${PARQUET_DIR}/pretrain_video_rec.parquet" \
    --output "${JSONL_DIR}/pretrain_video_rec.jsonl"
"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_pretrain_to_jsonl.py" \
    --input "${PARQUET_DIR}/pretrain_item_understand.parquet" \
    --output "${JSONL_DIR}/pretrain_item_understand.jsonl"

echo "完成，JSONL 输出目录：${JSONL_DIR}"
