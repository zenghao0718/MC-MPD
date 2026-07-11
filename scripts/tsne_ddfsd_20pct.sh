#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-42}
NUM_SAMPLES_EACH_CLASS=${NUM_SAMPLES_EACH_CLASS:-1024}
FEATURE_MODE=${FEATURE_MODE:-rgb}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_20pct/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_20pct_steps30000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
PERPLEXITY=${PERPLEXITY:-30}
USE_FP16=${USE_FP16:-True}
TOTAL_STEPS=${TOTAL_STEPS:-30000}

resolve_ckpt_path() {
    if [[ -n "${CKPT_PATH:-}" ]]; then
        echo "${CKPT_PATH}"
        return 0
    fi

    local candidates=(
        "${OUTPUT_PATH}/checkpoint_step_${TOTAL_STEPS}.pth"
        "${OUTPUT_PATH}/ckpt/checkpoint_step_${TOTAL_STEPS}.pth"
        "${OUTPUT_PATH}/ckpt/ddfsd_step[${TOTAL_STEPS}].pth"
        "${OUTPUT_PATH}/ckpt/ddfsd_step_${TOTAL_STEPS}.pth"
        "${OUTPUT_PATH}/ckpt/ddfsd_step${TOTAL_STEPS}.pth"
        "${OUTPUT_PATH}/ddfsd_step[${TOTAL_STEPS}].pth"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "${candidates[0]}"
}

if [[ "${TOTAL_STEPS}" != "30000" ]]; then
    echo "This 1/5 t-SNE script only supports step30000 checkpoints." >&2
    exit 1
fi

CKPT_PATH=$(resolve_ckpt_path)

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the 1/5 GenImage root that contains real/ ADM/ BigGAN/ glide/ Midjourney/ SD/ VQDM/." >&2
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint does not exist: ${CKPT_PATH}" >&2
    echo "Expected a step30000 checkpoint; supported names include checkpoint_step_30000.pth and ckpt/ddfsd_step[30000].pth." >&2
    exit 1
fi

if [[ ! -f "${FREQ_STATS_PATH}" ]]; then
    echo "Frequency stats do not exist: ${FREQ_STATS_PATH}" >&2
    echo "Use the freq_stats.pt from the same run directory; this script never recomputes frequency stats." >&2
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

python visualization/tsne_ddfsd.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --exclude_class "${EXCLUDE_CLASS}" \
    --feature_mode "${FEATURE_MODE}" \
    --num_samples_each_class "${NUM_SAMPLES_EACH_CLASS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --perplexity "${PERPLEXITY}" \
    --use_fp16 "${USE_FP16}"
