#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/Dual-Domain-Few-Shot-AIGI-Detector"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"main_full_steps15000"}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
RUN_DIR=${RUN_DIR:-"${RUN_ROOT}/${EXPERIMENT_NAME}/exclude_${EXCLUDE_CLASS}"}
CKPT_PATH=${CKPT_PATH:-"${RUN_DIR}/ckpt/ddfsd_step[15000].pth"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${RUN_DIR}/freq_stats.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RUN_DIR}/formal_eval_step15000"}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

case "${EXCLUDE_CLASS}" in
  ADM|BigGAN|glide|Midjourney|SD|VQDM) ;;
  *) echo "Invalid EXCLUDE_CLASS: ${EXCLUDE_CLASS}" >&2; exit 2 ;;
esac
[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "Step-15000 checkpoint does not exist: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${FREQ_STATS_PATH}" ]] || { echo "Frequency stats do not exist: ${FREQ_STATS_PATH}" >&2; exit 1; }

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$(realpath -m "${OUTPUT_DIR}")/" in
  "$(realpath -m "${REPO_ROOT}")/"*)
    echo "OUTPUT_DIR must be outside the repository: ${OUTPUT_DIR}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
python test_ddfsd.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --exclude_class "${EXCLUDE_CLASS}" \
  --model_mode dual \
  --branch_mode dual \
  --ckpt_path "${CKPT_PATH}" \
  --ckpt_step 15000 \
  --freq_stats_path "${FREQ_STATS_PATH}" \
  --num_support_test 10 \
  --eval_repeats 5 \
  --eval_seeds 42,101,102,103,104 \
  --max_eval_query_per_class 0 \
  --tau 0.2 \
  --tau_r 0.1 \
  --use_fp16 True \
  --pretrained False 2>&1 | tee "${OUTPUT_DIR}/eval.log"
