#!/usr/bin/env bash
set -euo pipefail

CLASSES=${CLASSES:-"ADM BigGAN glide Midjourney SD VQDM"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUNS_ROOT=${RUNS_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_10pct_steps15000"}
CKPT_STEP=${CKPT_STEP:-15000}
SHOT_LIST=${SHOT_LIST:-"0,1,2,5,10,20"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
ZERO_SHOT_METADATA_PER_CLASS=${ZERO_SHOT_METADATA_PER_CLASS:-1024}

for class_name in ${CLASSES}; do
  class_root="${RUNS_ROOT}/exclude_${class_name}"
  ckpt_path="${class_root}/ckpt/ddfsd_step[${CKPT_STEP}].pth"
  freq_path="${class_root}/freq_stats.pt"
  output_dir="${class_root}/multishot"
  [[ -f "${ckpt_path}" ]] || { echo "[${class_name}] missing checkpoint: ${ckpt_path}" >&2; exit 1; }
  [[ -f "${freq_path}" ]] || { echo "[${class_name}] missing freq stats: ${freq_path}" >&2; exit 1; }
  echo "[${class_name}] CKPT_PATH=${ckpt_path} OUTPUT_DIR=${output_dir}"
  EXCLUDE_CLASS="${class_name}" DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${output_dir}" \
    CKPT_PATH="${ckpt_path}" FREQ_STATS_PATH="${freq_path}" CKPT_STEP="${CKPT_STEP}" \
    SHOT_LIST="${SHOT_LIST}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" ZERO_SHOT_METADATA_PER_CLASS="${ZERO_SHOT_METADATA_PER_CLASS}" \
    bash scripts/eval_ddfsd_multishot.sh 2>&1 | tee "${class_root}/multishot_eval.log"
done

python tools/summarize_ddfsd_multishot.py --input_root "${RUNS_ROOT}" --output_dir "${RUNS_ROOT}/multishot_summary"
