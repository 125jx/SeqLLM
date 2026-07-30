#!/bin/bash
# CPT vs SFT 权重几何分析（低秩 / 与 base 主子空间重叠 / 任务向量夹角）
# Usage:
#   bash run_subspace.sh
#   bash run_subspace.sh /path/to/cpt/checkpoint          # 可选：覆盖 CPT 路径
#   FULL_SPECTRUM=1 bash run_subspace.sh                  # 额外算 effective_rank（慢）
set -euo pipefail

ROOT="/apdcephfs_cq11/share_303717182/bobjxzhang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${ROOT}/Qwen/Qwen3-8B-MchRiskModel-Init"
CPT_MODEL="${1:-${ROOT}/saves/qwen8b_pt_6144/checkpoint-32000}"
SFT_MODEL="${ROOT}/Qwen/Qwen3-8b-mch-risk"
OUTPUT_DIR="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}/subspace"
DEVICE="${DEVICE:-auto}"
K_LIST="${K_LIST:-64 256}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

for name in BASE_MODEL CPT_MODEL SFT_MODEL; do
    eval "p=\${${name}}"
    if [[ ! -f "${p}/model.safetensors.index.json" && ! -f "${p}/model.safetensors" ]]; then
        echo "ERROR: ${name} not found: ${p}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "CPT vs SFT Weight-Subspace Analysis"
echo "  Base : ${BASE_MODEL}"
echo "  CPT  : ${CPT_MODEL}"
echo "  SFT  : ${SFT_MODEL}"
echo "  Out  : ${OUTPUT_DIR}"
echo "  Dev  : ${DEVICE}   K=[${K_LIST}]   layer_stride=${LAYER_STRIDE}"
echo "============================================"

EXTRA=()
if [[ -n "${FULL_SPECTRUM:-}" ]]; then
    EXTRA=(--full_spectrum --spectrum_stride "${SPECTRUM_STRIDE:-4}")
fi

python3 "${SCRIPT_DIR}/eval_subspace.py" \
    --base "${BASE_MODEL}" \
    --cpt "${CPT_MODEL}" \
    --sft "${SFT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --k_list ${K_LIST} \
    --layer_stride "${LAYER_STRIDE}" \
    "${EXTRA[@]}"

echo ""
echo "Done:"
echo "  ${OUTPUT_DIR}/subspace_summary.json        # 汇总 + 自动解读"
echo "  ${OUTPUT_DIR}/per_matrix.csv               # 逐权重矩阵指标"
echo "  ${OUTPUT_DIR}/taskvector_cosine.csv"
echo "  ${OUTPUT_DIR}/stable_rank_lines.png        # 逐层 stable rank (低秩性)"
echo "  ${OUTPUT_DIR}/subspace_overlap_lines.png   # 逐层与 base 主子空间重叠"
echo "  ${OUTPUT_DIR}/subspace_overlap_bars.png    # 按模块分组重叠对比"
echo "  ${OUTPUT_DIR}/taskvector_cosine.png        # SFT/CPT 任务向量夹角"
echo "  ${OUTPUT_DIR}/effective_rank_lines.png     # 若设置 FULL_SPECTRUM=1"
