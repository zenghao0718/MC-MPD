#!/usr/bin/env bash
# ADM-only 10-shot main-protocol parity precheck. Run inside screen on AutoDL.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
CKPT_ROOT=${CKPT_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_ckpts_step15000"}
FREQ_STATS_ROOT=${FREQ_STATS_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/shared_full_freq_stats"}
REFERENCE_MAIN_ROOT=${REFERENCE_MAIN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_eval_step15000"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_shot_ablation_main_protocol_precheck/exclude_ADM/shot_10"}
CKPT_STEP=${CKPT_STEP:-15000}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
ZERO_SHOT_METADATA_PER_CLASS=${ZERO_SHOT_METADATA_PER_CLASS:-1024}
MODEL_MODE=${MODEL_MODE:-dual}
BRANCH_MODE=${BRANCH_MODE:-dual}
TAU=${TAU:-0.2}
TAU_R=${TAU_R:-0.1}
PYTHON_BIN=${PYTHON_BIN:-python}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
GIT_COMMIT=${GIT_COMMIT:-"$(git rev-parse HEAD)"}

CKPT_PATH=${CKPT_PATH:-"${CKPT_ROOT}/exclude_ADM/ckpt/ddfsd_step[${CKPT_STEP}].pth"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${FREQ_STATS_ROOT}/exclude_ADM/freq_stats.pt"}
REFERENCE_CSV=${REFERENCE_CSV:-"${REFERENCE_MAIN_ROOT}/exclude_ADM/formal_eval/step_${CKPT_STEP}/ddfsd_eval_per_seed.csv"}
PARITY_CSV=${PARITY_CSV:-"${OUTPUT_DIR}/ddfsd_10shot_parity.csv"}

[[ "${CKPT_STEP}" == "15000" ]] || {
  echo "ADM parity precheck requires CKPT_STEP=15000." >&2
  exit 1
}
[[ "${EVAL_SEEDS}" == "42,101,102,103,104" ]] || {
  echo "ADM parity precheck requires EVAL_SEEDS=42,101,102,103,104." >&2
  exit 1
}
case "${SKIP_COMPLETED}" in 0|1) ;; *) echo "SKIP_COMPLETED must be 0 or 1." >&2; exit 1 ;; esac

"${PYTHON_BIN}" tools/check_ddfsd_10shot_parity.py \
  --reference_csv "${REFERENCE_CSV}" \
  --expected_seeds "${EVAL_SEEDS}" \
  --expected_ckpt_step "${CKPT_STEP}" \
  --expected_exclude_class ADM \
  --validate_reference_only

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "ADM checkpoint does not exist: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${FREQ_STATS_PATH}" ]] || { echo "ADM freq stats do not exist: ${FREQ_STATS_PATH}" >&2; exit 1; }
[[ ! -e "${PARITY_CSV}" ]] || {
  echo "Refusing to overwrite existing parity CSV: ${PARITY_CSV}" >&2
  exit 1
}

if [[ -d "${OUTPUT_DIR}" ]]; then
  "${PYTHON_BIN}" tools/check_ddfsd_shot_completion.py \
    --output_dir "${OUTPUT_DIR}" \
    --git_commit "${GIT_COMMIT}" \
    --exclude_class ADM \
    --shot 10 \
    --seeds "${EVAL_SEEDS}" \
    --data_root "${DATA_ROOT}" \
    --ckpt_path "${CKPT_PATH}" \
    --ckpt_step "${CKPT_STEP}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --checkpoint_model_mode "${MODEL_MODE}" \
    --model_mode "${MODEL_MODE}" \
    --branch_mode "${BRANCH_MODE}" \
    --tau "${TAU}" \
    --tau_r "${TAU_R}" \
    --max_eval_query_per_class 0 \
    --zero_shot_metadata_per_class "${ZERO_SHOT_METADATA_PER_CLASS}"
  [[ "${SKIP_COMPLETED}" == "1" ]] || {
    echo "Refusing to overwrite complete ADM 10-shot output: ${OUTPUT_DIR}" >&2
    exit 1
  }
  echo "[SKIP EVAL] ADM 10-shot output is complete and configuration-matched."
else
  mkdir -p "${OUTPUT_DIR}"
  "${PYTHON_BIN}" test_ddfsd.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    --seed 42 \
    --exclude_class ADM \
    --ckpt_path "${CKPT_PATH}" \
    --ckpt_step "${CKPT_STEP}" \
    --model_mode "${MODEL_MODE}" \
    --branch_mode "${BRANCH_MODE}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --num_support_test 10 \
    --eval_repeats 5 \
    --eval_seeds "${EVAL_SEEDS}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --max_eval_query_per_class 0 \
    --zero_shot_metadata_per_class "${ZERO_SHOT_METADATA_PER_CLASS}" \
    --tau "${TAU}" \
    --tau_r "${TAU_R}" \
    --save_manifest true \
    --manifest_path "${OUTPUT_DIR}/support_query_manifest.csv" \
    --use_fp16 true \
    --pretrained false 2>&1 | tee "${OUTPUT_DIR}/eval.log"
fi

"${PYTHON_BIN}" tools/check_ddfsd_10shot_parity.py \
  --new_csv "${OUTPUT_DIR}/ddfsd_eval_per_seed.csv" \
  --reference_csv "${REFERENCE_CSV}" \
  --output_csv "${PARITY_CSV}" \
  --expected_seeds "${EVAL_SEEDS}" \
  --expected_ckpt_step "${CKPT_STEP}" \
  --expected_exclude_class ADM \
  --tolerance 1e-6

echo "ADM 10-shot parity precheck PASS: ${PARITY_CSV}"
