#!/bin/bash
# ==============================================================================
# Video-Rec Evaluation 启动器（YAML 驱动，与 LLaMA-Factory 训练风格一致）
#
# 用法:
#   bash examples/onerec/run_eval_video_rec.sh
#   bash examples/onerec/run_eval_video_rec.sh /path/to/custom.yaml
#   MODEL_PATH=/path/to/ckpt bash examples/onerec/run_eval_video_rec.sh
#   OUTPUT_DIR=/path/to/dir bash examples/onerec/run_eval_video_rec.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/onerec/run_eval_video_rec.sh
#
# YAML 路径默认 = examples/onerec/eval_video_rec.yaml；
# CLI 环境变量可覆盖 YAML 中的关键字段（MODEL_PATH / OUTPUT_DIR / SAMPLE_SIZE /
# CONSTRAINED / CODEBOOK_PATH / TP_SIZE / GPU_MEM_UTIL / DTYPE / MAX_MODEL_LEN /
# CHAT_TEMPLATE_FILE / DATASET / DATA_PATH ...）
# ==============================================================================

# bash examples/onerec/run_eval_video_rec.sh

set -euo pipefail

# ---------------- 路径 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval_video_rec.py"

CONFIG="${1:-${SCRIPT_DIR}/eval_video_rec.yaml}"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: YAML config not found: ${CONFIG}"
    exit 1
fi
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: eval script not found: ${EVAL_SCRIPT}"
    exit 1
fi

# ---------------- GPU / vLLM ----------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# ---------------- 日志 ----------------
LOG_DIR="${SCRIPT_DIR}/logs_v2"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
TAG="$(basename "${CONFIG%.yaml}")"
LOG_FILE="${LOG_DIR}/${TAG}_${TS}.log"

# ---------------- 可选覆盖（CLI > YAML）----------------
OVERRIDES=()
if [ -n "${MODEL_PATH:-}" ];          then OVERRIDES+=(--model_name_or_path "${MODEL_PATH}"); fi
if [ -n "${OUTPUT_DIR:-}" ];          then OVERRIDES+=(--output_dir "${OUTPUT_DIR}"); fi
if [ -n "${SAMPLE_SIZE:-}" ];         then OVERRIDES+=(--sample_size "${SAMPLE_SIZE}"); fi
if [ -n "${DATASET:-}" ];             then OVERRIDES+=(--dataset "${DATASET}"); fi
if [ -n "${DATA_PATH:-}" ];           then OVERRIDES+=(--data_path "${DATA_PATH}"); fi
if [ -n "${CHAT_TEMPLATE_FILE:-}" ];  then OVERRIDES+=(--chat_template_file "${CHAT_TEMPLATE_FILE}"); fi
if [ -n "${CONSTRAINED:-}" ];         then OVERRIDES+=(--constrained "${CONSTRAINED}"); fi
if [ -n "${CODEBOOK_PATH:-}" ];       then OVERRIDES+=(--codebook_path "${CODEBOOK_PATH}"); fi
if [ -n "${NUM_BEAMS:-}" ];           then OVERRIDES+=(--num_beams "${NUM_BEAMS}"); fi
if [ -n "${MAX_NEW_TOKENS:-}" ];      then OVERRIDES+=(--max_new_tokens "${MAX_NEW_TOKENS}"); fi
if [ -n "${BATCH_SIZE:-}" ];          then OVERRIDES+=(--batch_size "${BATCH_SIZE}"); fi
if [ -n "${TP_SIZE:-}" ];             then OVERRIDES+=(--tensor_parallel_size "${TP_SIZE}"); fi
if [ -n "${GPU_MEM_UTIL:-}" ];        then OVERRIDES+=(--gpu_memory_utilization "${GPU_MEM_UTIL}"); fi
if [ -n "${DTYPE:-}" ];               then OVERRIDES+=(--dtype "${DTYPE}"); fi
if [ -n "${MAX_MODEL_LEN:-}" ];       then OVERRIDES+=(--max_model_len "${MAX_MODEL_LEN}"); fi
if [ -n "${MAX_LOGPROBS:-}" ];        then OVERRIDES+=(--max_logprobs "${MAX_LOGPROBS}"); fi

# 若用户没显式给 TP_SIZE，则按 GPU 数自动设置（覆盖 YAML 里的 tensor_parallel_size）
if [ -z "${TP_SIZE:-}" ]; then
    OVERRIDES+=(--tensor_parallel_size "${NUM_GPUS}")
fi

echo "=========================================="
echo "Video-Rec Evaluation"
echo "=========================================="
echo "  config            : ${CONFIG}"
echo "  eval script       : ${EVAL_SCRIPT}"
echo "  log file          : ${LOG_FILE}"
echo "  CUDA_VISIBLE_DEVS : ${CUDA_VISIBLE_DEVICES} (auto TP=${NUM_GPUS})"
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
    echo "  overrides         : ${OVERRIDES[*]}"
fi
echo "=========================================="

python3 -u "${EVAL_SCRIPT}" \
    --config "${CONFIG}" \
    "${OVERRIDES[@]}" 2>&1 | tee "${LOG_FILE}"

echo "=========================================="
echo "Done. Log: ${LOG_FILE}"
echo "=========================================="
