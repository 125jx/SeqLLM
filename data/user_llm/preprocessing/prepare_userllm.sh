#!/usr/bin/env bash
# 依次处理 MovieLens-20M 与 Amazon Movies & TV。
#
# 用法：
#   bash prepare_userllm.sh
#   bash prepare_userllm.sh --datasets movielens
#   bash prepare_userllm.sh --datasets amazon --stages warmup favcategory reviewg
#   bash prepare_userllm.sh --datasets both --stages pretrain warmup fav
#   bash prepare_userllm.sh --datasets movielens amazon --stages all
#
# 可选环境变量：
#   PYTHON_BIN   默认 python3
#
# 统一 stage 名（会按数据集自动映射）：
#   pretrain   预训练窗口 + id_descriptions（amazon 额外写 review_store）
#   warmup     warmup_train / warmup_eval
#   fav        movielens→favgenre，amazon→favcategory
#   favgenre   仅 movielens
#   favcategory 仅 amazon
#   reviewg    仅 amazon
#   all        该数据集的全部 stage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=()
STAGES=()
EXTRA_ARGS=()

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --stages)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        STAGES+=("$1")
        shift
      done
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      # 兼容位置参数：prepare_userllm.sh movielens pretrain warmup
      if [[ "$1" == "movielens" || "$1" == "amazon" || "$1" == "both" ]]; then
        DATASETS+=("$1")
      else
        STAGES+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=("both")
fi
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=("all")
fi

# 展开 both，并固定先 movielens 后 amazon
RESOLVED_DATASETS=()
for ds in "${DATASETS[@]}"; do
  case "$ds" in
    both)
      RESOLVED_DATASETS+=("movielens" "amazon")
      ;;
    movielens|amazon)
      RESOLVED_DATASETS+=("$ds")
      ;;
    *)
      echo "[error] 未知数据集: $ds（可选 movielens / amazon / both）" >&2
      exit 1
      ;;
  esac
done

# 去重但保序
UNIQUE_DATASETS=()
for ds in "${RESOLVED_DATASETS[@]}"; do
  skip=0
  for seen in "${UNIQUE_DATASETS[@]:-}"; do
    if [[ "$seen" == "$ds" ]]; then
      skip=1
      break
    fi
  done
  if [[ $skip -eq 0 ]]; then
    UNIQUE_DATASETS+=("$ds")
  fi
done

map_stages_for_dataset() {
  local dataset="$1"
  local -a mapped=()
  local stage
  for stage in "${STAGES[@]}"; do
    case "$stage" in
      all)
        if [[ "$dataset" == "movielens" ]]; then
          mapped=("pretrain" "warmup" "favgenre")
        else
          mapped=("pretrain" "warmup" "favcategory" "reviewg")
        fi
        printf '%s\n' "${mapped[@]}"
        return 0
        ;;
    esac
  done

  for stage in "${STAGES[@]}"; do
    case "$stage" in
      pretrain|warmup)
        mapped+=("$stage")
        ;;
      fav)
        if [[ "$dataset" == "movielens" ]]; then
          mapped+=("favgenre")
        else
          mapped+=("favcategory")
        fi
        ;;
      favgenre)
        if [[ "$dataset" == "movielens" ]]; then
          mapped+=("favgenre")
        else
          echo "[warn] favgenre 仅适用于 movielens，跳过 amazon" >&2
        fi
        ;;
      favcategory)
        if [[ "$dataset" == "amazon" ]]; then
          mapped+=("favcategory")
        else
          echo "[warn] favcategory 仅适用于 amazon，跳过 movielens" >&2
        fi
        ;;
      reviewg)
        if [[ "$dataset" == "amazon" ]]; then
          mapped+=("reviewg")
        else
          echo "[warn] reviewg 仅适用于 amazon，跳过 movielens" >&2
        fi
        ;;
      *)
        echo "[error] 未知 stage: $stage" >&2
        echo "可选: pretrain warmup fav favgenre favcategory reviewg all" >&2
        exit 1
        ;;
    esac
  done

  if [[ ${#mapped[@]} -eq 0 ]]; then
    echo "[warn] ${dataset}: 没有可执行的 stage，跳过" >&2
    return 1
  fi
  printf '%s\n' "${mapped[@]}"
}

run_movielens() {
  local -a stages=("$@")
  echo "============================================================"
  echo "[MovieLens] stages: ${stages[*]}"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_movielens20m.py" \
    --stages "${stages[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

run_amazon() {
  local -a stages=("$@")
  echo "============================================================"
  echo "[Amazon] stages: ${stages[*]}"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_amazon_movies_tv.py" \
    --stages "${stages[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "datasets: ${UNIQUE_DATASETS[*]}"
echo "requested stages: ${STAGES[*]}"
echo

for ds in "${UNIQUE_DATASETS[@]}"; do
  mapfile -t ds_stages < <(map_stages_for_dataset "$ds" || true)
  if [[ ${#ds_stages[@]} -eq 0 ]]; then
    continue
  fi
  case "$ds" in
    movielens) run_movielens "${ds_stages[@]}" ;;
    amazon) run_amazon "${ds_stages[@]}" ;;
  esac
  echo
done

echo "全部完成。"
