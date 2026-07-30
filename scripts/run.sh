#!/usr/bin/env bash
# 减少警告输出
export TRANSFORMERS_VERBOSITY=error
# 过滤 SDPA output_attentions 警告（注意力蒸馏的预期行为）
export PYTHONWARNINGS="ignore::UserWarning"
# 禁用 transformers 的特定警告
export HF_HUB_DISABLE_PROGRESS_BARS=1

set -euo pipefail

# Usage (defaults match the command you provided):
#   bash submitjob/run-pt.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 YAML=examples/train_pt/xxx.yaml LOG_FILE=log/qwen/xxx.log bash submitjob/run-pt.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${FORCE_TORCHRUN:=1}"
: "${YAML:=examples/onerec/prefix_guided_sft.yaml}"
: "${LOG_FILE:=log/onerec_sft_nextitem.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

echo "[run] cwd=$(pwd)"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] FORCE_TORCHRUN=${FORCE_TORCHRUN}"
echo "[run] YAML=${YAML}"
echo "[run] LOG_FILE=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" FORCE_TORCHRUN="${FORCE_TORCHRUN}" \
  lmf train "${YAML}" 2>&1 | tee "${LOG_FILE}"


