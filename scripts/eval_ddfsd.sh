#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
TOTAL_STEPS=${TOTAL_STEPS:-15000}
CKPT_PATH=${CKPT_PATH:-"${OUTPUT_PATH}/ckpt/ddfsd_step[${TOTAL_STEPS}].pth"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the GenImage root used by the checkpoint." >&2
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${FREQ_STATS_PATH}" ]]; then
    echo "Frequency stats do not exist: ${FREQ_STATS_PATH}" >&2
    echo "Evaluation must use the freq_stats.pt from the same run directory." >&2
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

python test_ddfsd.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --exclude_class "${EXCLUDE_CLASS}" \
    --ckpt_path "${CKPT_PATH}" \
    --ckpt_step "${TOTAL_STEPS}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --num_support_test 10 \
    --eval_repeats 5 \
    --eval_seeds "${EVAL_SEEDS}" \
    --tau 0.2 \
    --tau_r 0.1 \
    --use_fp16 True \
    --pretrained False
