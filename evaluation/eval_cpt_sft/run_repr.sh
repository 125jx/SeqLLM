#!/bin/bash
# 语言认知探针：base / CPT / SFT 在纯语言输入上的表示&输出漂移
#   ① embedding 表 cosine  ② 逐层 hidden 相似度(CKA/cosine)  ③ next-token KL
# Usage: bash run_repr.sh                       # 用默认探针文件
#        bash run_repr.sh /path/to/probe.txt    # 覆盖探针文件（每行一条纯中文文本）
set -euo pipefail

ROOT="/apdcephfs_cq11/share_303717182/bobjxzhang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${ROOT}/Qwen/Qwen3-8B-MchRiskModel-Init"
CPT_MODEL="${CPT_MODEL:-${ROOT}/saves/qwen8b_pt_6144/checkpoint-32000}"
SFT_MODEL="${ROOT}/Qwen/Qwen3-8b-mch-risk"
OUTPUT_DIR="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}/repr"
ORIGINAL_VOCAB_SIZE="${ORIGINAL_VOCAB_SIZE:-151643}"

# 探针数据：优先命令行参数，其次环境变量，最后默认路径
# 建议用「C-Eval 掉分那批语言样本」的纯题干文本（不带 prefix / 行为 token）
PROBE_FILE="${1:-${PROBE_FILE:-${SCRIPT_DIR}/data/probe_ceval.txt}}"
PROBE_FORMAT="${PROBE_FORMAT:-txt}"   # txt | jsonl | lm_eval

N_SAMPLES="${N_SAMPLES:-200}"
MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
KL_POSITIONS="${KL_POSITIONS:-64}"

for name in BASE_MODEL CPT_MODEL SFT_MODEL; do
    eval "p=\${${name}}"
    if [[ ! -f "${p}/model.safetensors.index.json" && ! -f "${p}/model.safetensors" ]]; then
        echo "ERROR: ${name} not found: ${p}"; exit 1
    fi
done
if [[ ! -f "${PROBE_FILE}" ]]; then
    echo "ERROR: probe file not found: ${PROBE_FILE}"
    echo ""
    echo "probe_ceval 不是现成数据，需要在你自己的太极/GPU 环境里先生成："
    echo "  方式 A（推荐）: bash ${SCRIPT_DIR}/prepare_probe_ceval.sh"
    echo "  方式 B: 若已跑过 eval_ceval.sh --log_samples，把 jsonl 转过来："
    echo "          bash ${SCRIPT_DIR}/prepare_probe_ceval.sh /path/to/samples.jsonl"
    echo "  方式 C: 任意纯中文 .txt（每行一条），然后 bash run_repr.sh /path/to/that.txt"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Representation / Language-cognition Probe"
echo "  Base  : ${BASE_MODEL}"
echo "  CPT   : ${CPT_MODEL}"
echo "  SFT   : ${SFT_MODEL}"
echo "  Probe : ${PROBE_FILE} (${PROBE_FORMAT}), n=${N_SAMPLES}"
echo "  Out   : ${OUTPUT_DIR}"
echo "============================================"

python3 "${SCRIPT_DIR}/eval_repr.py" \
    --base "${BASE_MODEL}" \
    --cpt "${CPT_MODEL}" \
    --sft "${SFT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --probe_file "${PROBE_FILE}" \
    --probe_format "${PROBE_FORMAT}" \
    --n_samples "${N_SAMPLES}" \
    --max_len "${MAX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --kl_positions "${KL_POSITIONS}" \
    --original_vocab_size "${ORIGINAL_VOCAB_SIZE}"

echo ""
echo "Done:"
echo "  ${OUTPUT_DIR}/layer_cka.png        # 主图：逐层 CKA 相似度 vs base，CPT/SFT"
echo "  ${OUTPUT_DIR}/layer_cosine.png     # 逐层 per-example cosine vs base"
echo "  ${OUTPUT_DIR}/layer_similarity.csv"
echo "  ${OUTPUT_DIR}/summary.json         # ① embedding cosine ② 逐层首尾 ③ next-token KL"
