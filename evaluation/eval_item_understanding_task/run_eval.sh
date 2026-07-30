#!/usr/bin/env bash
# Item Understanding Task 一键评测：vLLM 推理 -> 确定性判分
# 用法：
#   bash run_eval.sh                       # 用默认 OneRec-8B，单机 8 卡
#   bash run_eval.sh /path/to/ckpt 8       # 指定 ckpt + tp=8
#   bash run_eval.sh /path/to/ckpt 8 think # 第三参数 think -> 开启思考模式
set -euo pipefail

cd "$(dirname "$0")"

# 单机 8 卡
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL="${1:-/apdcephfs_cq11/share_303717182/bobjxzhang/Qwen/OneRec-8B}"
TP="${2:-8}"
THINK="${3:-}"

GOLD="./all_tasks.jsonl"
TAG="$(basename "$MODEL")"
PRED="./results/preds_${TAG}.jsonl"
REPORT="./results/score_${TAG}.json"

THINK_FLAG=""
if [ "$THINK" = "think" ]; then
  THINK_FLAG="--enable-thinking"
  PRED="./preds_${TAG}_think.jsonl"
  REPORT="./score_${TAG}_think.json"
fi

echo "==== Item Understanding Eval ===="
echo "model : $MODEL"
echo "tp    : $TP   think: ${THINK:-off}"
echo "gold  : $GOLD"
echo "pred  : $PRED"
echo "======================="

python3 infer_recprobe_vllm.py \
  --model "$MODEL" \
  --gold  "$GOLD" \
  --out   "$PRED" \
  --tp    "$TP" \
  $THINK_FLAG

python3 score_recprobe.py \
  --gold "$GOLD" \
  --pred "$PRED" \
  --report "$REPORT"
