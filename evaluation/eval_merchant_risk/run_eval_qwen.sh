#!/bin/bash
# ==============================================================================
# eval_qwen.py 运行脚本
# 支持两种模式: api (调用 vLLM 服务) / local (直接加载模型)
# 支持通过环境变量覆盖配置，例如:
#   MODE=local TP=8 BATCH_SIZE=256 bash evaluation/eval_merchant_risk/run_eval_qwen.sh
#   START_INDEX=0 END_INDEX=60000 bash evaluation/eval_merchant_risk/run_eval_qwen.sh
# ==============================================================================
set -euo pipefail

# ========== 模型 ==========
: "${MODEL_PATH:=/apdcephfs/zjx121/apdcephfs_cq11/share_303717182/bobjxzhang/Qwen/cot_rl/qwen3_8b_sft_cot_rl2}"

# ========== 输入 / 输出 ==========
# 单文件评估；若配置了 START_INDEX / END_INDEX，OUTPUT_FILE 会自动拼上
# "_s{START}_e{END}" 后缀，避免多卡跑同一份输入时互相覆盖。
: "${INPUT_FILE:=/apdcephfs/zjx121/apdcephfs_cq11/share_303717182/bobjxzhang/data/wechat_pay_sft_cot_rl_result/eval/origin/screening_mch_seq_part1.20260425.jsonl}"
: "${OUTPUT_FILE:=/apdcephfs/zjx121/apdcephfs_cq11/share_303717182/bobjxzhang/data/wechat_pay_sft_cot_rl_result/eval/mch/eval_qwen3_8b_sft_cot_rl2_part1.jsonl}"

# ========== API 模式配置 ==========
: "${BASE_URL:=http://localhost:8000/v1}"
: "${NUM_WORKERS:=128}"
# 启动 vLLM 服务命令示例（另开终端）:
# max-model-len = max_input_tokens + max_new_tokens = 8192 + 3000 = 11192
# vllm serve ${MODEL_PATH} \
#     --tensor-parallel-size 8 \
#     --max-model-len 12000 \
#     --gpu-memory-utilization 0.9 \
#     --port 8000

# ========== 本地模式配置 ==========
: "${TENSOR_PARALLEL:=8}"
: "${BATCH_SIZE:=256}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

# ========== 其他配置 ==========
: "${MAX_NEW_TOKENS:=3000}"
: "${MAX_INPUT_TOKENS:=8192}"
: "${TOP_LOGPROBS:=20}"
: "${LIMIT:=}"              # 如 LIMIT="--limit 100"，留空不限制
: "${MARK_TRUNCATED:=true}" # 是否记录 truncated / truncated_reason
: "${RESUME:=1}"            # 1=断点续跑，0=覆盖写

# ========== 数据分片（多卡并行跑同一文件时使用）==========
# 留空则不分片；用法示例:
#   START_INDEX=0     END_INDEX=60000
#   START_INDEX=60000 END_INDEX=120000
: "${START_INDEX:=}"
: "${END_INDEX:=}"

# ==============================================================================
# 运行模式: api 或 local
# ==============================================================================
: "${MODE:=local}"

# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_qwen.py"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "错误: 找不到评估脚本 ${EVAL_SCRIPT}"
    exit 1
fi
if [ -z "${INPUT_FILE}" ] || [ -z "${OUTPUT_FILE}" ]; then
    echo "错误: INPUT_FILE 或 OUTPUT_FILE 未设置"
    exit 1
fi
if [ ! -f "${INPUT_FILE}" ]; then
    echo "错误: 输入文件不存在: ${INPUT_FILE}"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# ----- 组装 start/end_index，并为 OUTPUT 生成分片后缀 -----
SHARD_ARGS=()
SHARD_SUFFIX=""
if [ -n "${START_INDEX}" ]; then
    SHARD_ARGS+=(--start_index "${START_INDEX}")
    SHARD_SUFFIX="${SHARD_SUFFIX}_s${START_INDEX}"
fi
if [ -n "${END_INDEX}" ]; then
    SHARD_ARGS+=(--end_index "${END_INDEX}")
    SHARD_SUFFIX="${SHARD_SUFFIX}_e${END_INDEX}"
fi

FINAL_OUTPUT="${OUTPUT_FILE}"
if [ -n "${SHARD_SUFFIX}" ]; then
    out_dir="$(dirname "${OUTPUT_FILE}")"
    out_base="$(basename "${OUTPUT_FILE}")"
    if [[ "${out_base}" == *.* ]]; then
        out_stem="${out_base%.*}"
        out_ext=".${out_base##*.}"
    else
        out_stem="${out_base}"
        out_ext=""
    fi
    FINAL_OUTPUT="${out_dir}/${out_stem}${SHARD_SUFFIX}${out_ext}"
fi

mkdir -p "$(dirname "${FINAL_OUTPUT}")"

RESUME_ARGS=()
if [ "${RESUME}" = "1" ] || [ "${RESUME}" = "true" ]; then
    RESUME_ARGS+=(--resume)
fi

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
    # 允许 LIMIT="--limit 100" 或 LIMIT="100"
    if [[ "${LIMIT}" == --* ]]; then
        # shellcheck disable=SC2206
        LIMIT_ARGS=(${LIMIT})
    else
        LIMIT_ARGS=(--limit "${LIMIT}")
    fi
fi

export CUDA_VISIBLE_DEVICES

echo ""
echo "##############################################"
echo "# MODE  : ${MODE}"
echo "# MODEL : ${MODEL_PATH}"
echo "# INPUT : ${INPUT_FILE}"
echo "# OUTPUT: ${FINAL_OUTPUT}"
echo "# GPUS  : ${CUDA_VISIBLE_DEVICES}"
echo "# SHARD : start=${START_INDEX:-<none>}  end=${END_INDEX:-<none>}"
echo "##############################################"

if [ "${MODE}" = "api" ]; then
    echo "运行模式: API 调用 | 服务: ${BASE_URL} | 并发: ${NUM_WORKERS}"
    python3 "${EVAL_SCRIPT}" \
        --mode api \
        --base_url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --input "${INPUT_FILE}" \
        --output "${FINAL_OUTPUT}" \
        --workers "${NUM_WORKERS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_input_tokens "${MAX_INPUT_TOKENS}" \
        --top_logprobs "${TOP_LOGPROBS}" \
        --mark_truncated "${MARK_TRUNCATED}" \
        "${LIMIT_ARGS[@]}" \
        "${SHARD_ARGS[@]}" \
        "${RESUME_ARGS[@]}"

elif [ "${MODE}" = "local" ]; then
    echo "运行模式: 本地模型加载 | TP: ${TENSOR_PARALLEL}, Batch: ${BATCH_SIZE}"
    python3 "${EVAL_SCRIPT}" \
        --mode local \
        --model "${MODEL_PATH}" \
        --input "${INPUT_FILE}" \
        --output "${FINAL_OUTPUT}" \
        --tp "${TENSOR_PARALLEL}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_input_tokens "${MAX_INPUT_TOKENS}" \
        --top_logprobs "${TOP_LOGPROBS}" \
        --mark_truncated "${MARK_TRUNCATED}" \
        "${LIMIT_ARGS[@]}" \
        "${SHARD_ARGS[@]}" \
        "${RESUME_ARGS[@]}"

else
    echo "错误: 未知模式 ${MODE}，请设置为 api 或 local"
    exit 1
fi

echo ""
echo "=========================================="
echo "评估任务完成: ${FINAL_OUTPUT}"
echo "=========================================="
