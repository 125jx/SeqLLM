#!/bin/bash
# 在太极评测环境里生成 probe_ceval.txt（我这边 Cursor 环境跑不了，需你在 GPU 机器上执行）
# Usage:
#   bash prepare_probe_ceval.sh
#   bash prepare_probe_ceval.sh /path/to/lm_eval_samples.jsonl   # 从已有 log_samples 转
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SCRIPT_DIR}/data/probe_ceval.txt"
N_SAMPLES="${N_SAMPLES:-500}"

# 与 eval_ceval.sh 一致，下载 C-Eval 时需要
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"

mkdir -p "${SCRIPT_DIR}/data"

if [[ $# -ge 1 ]]; then
    echo "Convert lm_eval log_samples -> ${OUTPUT}"
    python3 "${SCRIPT_DIR}/prepare_probe_ceval.py" \
        --from_lm_eval "$1" \
        --output "${OUTPUT}" \
        --n_samples "${N_SAMPLES}"
else
    echo "Download C-Eval val split -> ${OUTPUT}"
    python3 -c "import datasets" 2>/dev/null || {
        echo "ERROR: 需要 pip install datasets"
        exit 1
    }
    python3 "${SCRIPT_DIR}/prepare_probe_ceval.py" \
        --output "${OUTPUT}" \
        --n_samples "${N_SAMPLES}"
fi

echo ""
echo "Done. Then run:"
echo "  bash ${SCRIPT_DIR}/run_repr.sh"
