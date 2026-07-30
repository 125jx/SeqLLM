#!/bin/bash
# 一键运行 CPT vs SFT 全部分析（delta_w / subspace / pathway / repr）
# PROBE_FILE=/apdcephfs_cq11/share_303717182/bobjxzhang/tmp/eval_cpt_sft/data/probe_ceval.txt bash run_all.sh /apdcephfs_cq11/share_303717182/bobjxzhang/saves/qwen8b_pt_6144/checkpoint-33000 /apdcephfs_cq11/share_303717182/bobjxzhang/tmp/eval_cpt_sft/output1
# Usage:
#   bash run_all.sh                                  # 用各脚本默认 CPT / 输出路径
#   bash run_all.sh /path/to/cpt/ckpt                # 覆盖 CPT 路径
#   bash run_all.sh /path/to/cpt/ckpt /path/to/out   # 同时覆盖 CPT 路径 + 输出根目录
#
# 也可用环境变量设置（与位置参数等价，位置参数优先）：
#   CPT_CKPT=/path/to/cpt/ckpt   OUTPUT_ROOT=/path/to/out   bash run_all.sh
#
# 输出结构：四个子任务分别写到 ${OUTPUT_ROOT}/{delta_w,subspace,pathway,repr}，
#           日志写到 ${OUTPUT_ROOT}/logs/<task>_<时间戳>.log
#
# 常用开关（环境变量）：
#   ONLY="delta_w subspace"   只跑指定子任务（空格分隔，名字见下方 ALL_TASKS）
#   SKIP="repr"               跳过指定子任务
#   STOP_ON_ERROR=1           某个子任务失败即整体退出（默认失败也继续，最后汇总）
#
# 子脚本自身的环境变量（如 TOP_K / K_LIST / FULL_SPECTRUM / N_SAMPLES ...）
# 会被继承，可直接在命令行前面加，例如：
#   FULL_SPECTRUM=1 TOP_K=256 bash run_all.sh
#
# repr 特别说明：它的 $1 是「探针文件」而非 CPT 路径，CPT 由本脚本经 CPT_MODEL 自动传入。
#   若要给 repr 指定探针文件，用 PROBE_FILE 环境变量：
#   PROBE_FILE=/path/to/probe.txt  bash run_all.sh /path/cpt/ckpt /path/out
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CPT ckpt：位置参数 $1 优先，其次环境变量 CPT_CKPT，否则留空（各子脚本用自己的默认值）
CPT_ARG="${1:-${CPT_CKPT:-}}"

# 输出根目录：位置参数 $2 优先，其次环境变量 OUTPUT_ROOT，否则默认 ${SCRIPT_DIR}/output
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-${SCRIPT_DIR}/output}}"
export OUTPUT_ROOT   # 透传给子脚本，让它们写到 ${OUTPUT_ROOT}/<task>

ALL_TASKS="repr"

# 解析 ONLY / SKIP
declare -a TASKS=()
for t in ${ONLY:-$ALL_TASKS}; do
    skip=0
    for s in ${SKIP:-}; do
        [[ "$t" == "$s" ]] && skip=1 && break
    done
    [[ $skip -eq 0 ]] && TASKS+=("$t")
done

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "############################################"
echo "# CPT vs SFT — Run-All"
echo "#   tasks : ${TASKS[*]}"
echo "#   cpt   : ${CPT_ARG:-<default in each script>}"
echo "#   out   : ${OUTPUT_ROOT}"
echo "#   logs  : ${LOG_DIR}/<task>_${STAMP}.log"
echo "############################################"

declare -A STATUS
declare -A ELAPSED
overall_start=$(date +%s)

for task in "${TASKS[@]}"; do
    script="${SCRIPT_DIR}/run_${task}.sh"
    log="${LOG_DIR}/${task}_${STAMP}.log"

    if [[ ! -f "${script}" ]]; then
        echo ">>> [${task}] SKIP: ${script} not found"
        STATUS[$task]="MISSING"
        continue
    fi

    echo ""
    echo ">>> [${task}] start  ($(date '+%H:%M:%S'))  ->  ${log}"
    t0=$(date +%s)

    # 注意参数约定不一致：
    #   delta_w / subspace / pathway  ->  $1 = CPT ckpt
    #   repr                          ->  $1 = PROBE_FILE，CPT 走环境变量 CPT_MODEL
    # 因此 repr 单独处理：CPT 用 CPT_MODEL 传，位置参数留给探针文件 PROBE_FILE。
    if [[ "${task}" == "repr" ]]; then
        if [[ -n "${PROBE_FILE:-}" ]]; then
            CPT_MODEL="${CPT_ARG:-${CPT_MODEL:-}}" bash "${script}" "${PROBE_FILE}" 2>&1 | tee "${log}"
        else
            CPT_MODEL="${CPT_ARG:-${CPT_MODEL:-}}" bash "${script}" 2>&1 | tee "${log}"
        fi
    elif [[ -n "${CPT_ARG}" ]]; then
        bash "${script}" "${CPT_ARG}" 2>&1 | tee "${log}"
    else
        bash "${script}" 2>&1 | tee "${log}"
    fi
    rc=${PIPESTATUS[0]}

    t1=$(date +%s)
    ELAPSED[$task]=$(( t1 - t0 ))

    if [[ ${rc} -eq 0 ]]; then
        STATUS[$task]="OK"
        echo ">>> [${task}] done   (${ELAPSED[$task]}s)"
    else
        STATUS[$task]="FAIL(rc=${rc})"
        echo ">>> [${task}] FAIL   (rc=${rc}, ${ELAPSED[$task]}s)  见日志: ${log}"
        if [[ -n "${STOP_ON_ERROR:-}" ]]; then
            echo ">>> STOP_ON_ERROR=1，终止后续任务。"
            break
        fi
    fi
done

overall_end=$(date +%s)

echo ""
echo "############################################"
echo "# Summary  (total $(( overall_end - overall_start ))s)"
echo "############################################"
fail=0
for task in "${TASKS[@]}"; do
    st="${STATUS[$task]:-SKIPPED}"
    el="${ELAPSED[$task]:-0}"
    printf "  %-10s %-14s %ss\n" "${task}" "${st}" "${el}"
    [[ "${st}" == FAIL* ]] && fail=1
done
echo "  logs: ${LOG_DIR}"

exit ${fail}
