#!/bin/bash
# 「固定位置 / 跨层通路」分析
# Usage:
#   bash run_pathway.sh
#   bash run_pathway.sh /path/to/cpt-checkpoint
#
# 若有中间 checkpoint（检验每步是否更新同一位置）：
#   STEP_CKS="500:/path/checkpoint-500 1000:/path/checkpoint-1000 2000:/path/final" bash run_pathway.sh
set -euo pipefail

ROOT="/apdcephfs_cq11/share_303717182/bobjxzhang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${ROOT}/Qwen/Qwen3-8B-MchRiskModel-Init"
CPT_MODEL="${1:-${ROOT}/saves/qwen8b_pt_6144/checkpoint-32000}"
SFT_MODEL="${ROOT}/Qwen/Qwen3-8b-mch-risk"
OUTPUT_DIR="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}/pathway"
TOP_K="${TOP_K:-128}"

# 可选：STEP_CKS="500:/path/ckpt-500 2000:/path/final"
STEP_ARGS=()
if [[ -n "${STEP_CKS:-}" ]]; then
    # shellcheck disable=SC2206
    STEP_ARGS=(--step_checkpoints ${STEP_CKS})
fi

mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/eval_pathway.py" \
    --base "${BASE_MODEL}" \
    --sft "${SFT_MODEL}" \
    --cpt "${CPT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_k "${TOP_K}" \
    "${STEP_ARGS[@]}"

echo ""
echo "Done:"
echo "  ${OUTPUT_DIR}/pathway_summary.json       # 汇总 + 自动解读"
echo "  ${OUTPUT_DIR}/element_concentration.csv  # 每层元素集中度"
echo "  ${OUTPUT_DIR}/pathway_qproj_heatmap.png  # q_proj 跨层相关热力图"
echo "  ${OUTPUT_DIR}/step_consistency.csv       # 若设置了 STEP_CKS"
