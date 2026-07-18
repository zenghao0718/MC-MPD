#!/usr/bin/env bash
# Generic DDFSD branch-mode (dual/single_branch_rgb_train/freq-only) diagnostic evaluation wrapper.
set -euo pipefail

EXCLUDE_CLASS=${EXCLUDE_CLASS:?EXCLUDE_CLASS is required}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
CKPT_STEPS=${CKPT_STEPS:-"2500,5000,7500,10000,12500,15000"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
NUM_SUPPORT_TEST=${NUM_SUPPORT_TEST:-10}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-"${OUTPUT_PATH}/branch_modes"}

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${FREQ_STATS_PATH}" ]]; then
    echo "Frequency stats do not exist: ${FREQ_STATS_PATH}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

python tools/eval_ddfsd_branch_modes.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_PATH}" \
    --exclude_class "${EXCLUDE_CLASS}" \
    --ckpt_steps "${CKPT_STEPS}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --num_support_test "${NUM_SUPPORT_TEST}" \
    --eval_seeds "${EVAL_SEEDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tau 0.2 \
    --tau_r 0.1 \
    --use_fp16 True \
    --pretrained False \
    --out_dir "${OUT_DIR}"
