#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-42}
NUM_SAMPLES_EACH_CLASS=${NUM_SAMPLES_EACH_CLASS:-1024}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_full"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
RUN_CONFIG=${RUN_CONFIG:-"ddfsd_full_steps200000"}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"${RUN_ROOT}/${RUN_CONFIG}"}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
PERPLEXITY=${PERPLEXITY:-30}
USE_FP16=${USE_FP16:-True}
EXCLUDE_CLASSES=${EXCLUDE_CLASSES:-"Midjourney glide ADM SD VQDM BigGAN"}
FEATURE_MODES=${FEATURE_MODES:-"rgb freq add concat"}

read -r -a EXCLUDE_ARRAY <<< "${EXCLUDE_CLASSES}"
read -r -a FEATURE_ARRAY <<< "${FEATURE_MODES}"

for exclude_class in "${EXCLUDE_ARRAY[@]}"; do
    for feature_mode in "${FEATURE_ARRAY[@]}"; do
        EXCLUDE_CLASS="${exclude_class}" \
        FEATURE_MODE="${feature_mode}" \
        SEED="${SEED}" \
        NUM_SAMPLES_EACH_CLASS="${NUM_SAMPLES_EACH_CLASS}" \
        DATA_ROOT="${DATA_ROOT}" \
        RUN_ROOT="${RUN_ROOT}" \
        OUTPUT_PATH="${BASE_OUTPUT_DIR}/exclude_${exclude_class}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        PERPLEXITY="${PERPLEXITY}" \
        USE_FP16="${USE_FP16}" \
        bash scripts/tsne_ddfsd_full.sh
    done
done

EXCLUDE_CSV=$(IFS=,; echo "${EXCLUDE_ARRAY[*]}")
FEATURE_CSV=$(IFS=,; echo "${FEATURE_ARRAY[*]}")

python visualization/plot_tsne_grid.py \
    --run_dir "${BASE_OUTPUT_DIR}" \
    --output_dir "${BASE_OUTPUT_DIR}/visualization/seed${SEED}" \
    --seed "${SEED}" \
    --exclude_classes "${EXCLUDE_CSV}" \
    --feature_modes "${FEATURE_CSV}"
