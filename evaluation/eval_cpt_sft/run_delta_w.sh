#!/bin/bash
# CPT vs SFT ΔW 分析 — 一键运行
# Usage: bash run.sh
#        bash run.sh /path/to/cpt/checkpoint   # 可选：覆盖 CPT 路径
set -euo pipefail

ROOT="/apdcephfs_cq11/share_303717182/bobjxzhang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${ROOT}/Qwen/Qwen3-8B-MchRiskModel-Init"
CPT_MODEL="${1:-${ROOT}/saves/qwen8b_pt_6144/checkpoint-32000}"
SFT_MODEL="${ROOT}/Qwen/Qwen3-8b-mch-risk"
OUTPUT_DIR="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}/delta_w"
ORIGINAL_VOCAB_SIZE="${ORIGINAL_VOCAB_SIZE:-151643}"

for name in BASE_MODEL CPT_MODEL SFT_MODEL; do
    eval "p=\${${name}}"
    if [[ ! -f "${p}/model.safetensors.index.json" && ! -f "${p}/model.safetensors" ]]; then
        echo "ERROR: ${name} not found: ${p}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "CPT vs SFT Delta-W Analysis"
echo "  Base : ${BASE_MODEL}"
echo "  CPT  : ${CPT_MODEL}"
echo "  SFT  : ${SFT_MODEL}"
echo "  Out  : ${OUTPUT_DIR}"
echo "============================================"

python3 "${SCRIPT_DIR}/eval_delta_w.py" \
    --base "${BASE_MODEL}" \
    --cpt "${CPT_MODEL}" \
    --sft "${SFT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --original_vocab_size "${ORIGINAL_VOCAB_SIZE}"

echo ""
echo "Done:"
echo "  ${OUTPUT_DIR}/layer_lines.png   # 折线图：逐层 r_l"
echo "  ${OUTPUT_DIR}/module_bars.png   # 柱状图：模块分组对比"
echo "  ${OUTPUT_DIR}/embed_split.png   # 原始词表 vs 新增 token"
echo "  ${OUTPUT_DIR}/results.csv"
echo "  ${OUTPUT_DIR}/embed_split.csv"
echo "  ${OUTPUT_DIR}/summary.json"
