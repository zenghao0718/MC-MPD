#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to an exclude_<class>/multishot directory}
CKPT_PATH=${CKPT_PATH:?Set CKPT_PATH to the class checkpoint}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"$(dirname "$(dirname "${CKPT_PATH}")")/freq_stats.pt"}
CKPT_STEP=${CKPT_STEP:-0}
SHOT_LIST=${SHOT_LIST:-"0,1,2,5,10,20"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
EVAL_REPEATS=${EVAL_REPEATS:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
ZERO_SHOT_METADATA_PER_CLASS=${ZERO_SHOT_METADATA_PER_CLASS:-1024}
MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS:-0}
MODEL_MODE=${MODEL_MODE:-auto}
BRANCH_MODE=${BRANCH_MODE:-}

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint does not exist: ${CKPT_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

branch_args=()
if [[ -n "${BRANCH_MODE}" ]]; then
  branch_args=(--branch_mode "${BRANCH_MODE}")
fi

python test_ddfsd_multishot.py \
  --data_root "${DATA_ROOT}" --output_dir "${OUTPUT_DIR}" --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" --exclude_class "${EXCLUDE_CLASS}" --ckpt_path "${CKPT_PATH}" \
  --ckpt_step "${CKPT_STEP}" --freq_stats_path "${FREQ_STATS_PATH}" \
  --model_mode "${MODEL_MODE}" "${branch_args[@]}" \
  --shot_list "${SHOT_LIST}" --eval_repeats "${EVAL_REPEATS}" --eval_seeds "${EVAL_SEEDS}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_eval_query_per_class "${MAX_EVAL_QUERY_PER_CLASS}" \
  --zero_shot_metadata_per_class "${ZERO_SHOT_METADATA_PER_CLASS}" \
  --tau 0.2 --tau_r 0.1 --use_fp16 True --pretrained False
