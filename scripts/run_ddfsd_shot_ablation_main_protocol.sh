#!/usr/bin/env bash
# Formal main-protocol shot ablation. Run this long evaluation inside screen on AutoDL.
set -euo pipefail

ALL_CLASSES="ADM BigGAN glide Midjourney SD VQDM"
CLASSES=${CLASSES:-"${ALL_CLASSES}"}
SHOTS=${SHOTS:-"0 5 10 20 30 50"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
CKPT_ROOT=${CKPT_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_ckpts_step15000"}
FREQ_STATS_ROOT=${FREQ_STATS_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/shared_full_freq_stats"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_shot_ablation_main_protocol_step15000"}
REFERENCE_MAIN_ROOT=${REFERENCE_MAIN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_eval_step15000"}
CKPT_STEP=${CKPT_STEP:-15000}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
ZERO_SHOT_METADATA_PER_CLASS=${ZERO_SHOT_METADATA_PER_CLASS:-1024}
MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS:-0}
MODEL_MODE=${MODEL_MODE:-dual}
BRANCH_MODE=${BRANCH_MODE:-dual}
TAU=${TAU:-0.2}
TAU_R=${TAU_R:-0.1}
PYTHON_BIN=${PYTHON_BIN:-python}
AGGREGATE=${AGGREGATE:-auto}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

CKPT_PATH_TEMPLATE=${CKPT_PATH_TEMPLATE:-"${CKPT_ROOT}/exclude_{class}/ckpt/ddfsd_step[{step}].pth"}
FREQ_STATS_PATH_TEMPLATE=${FREQ_STATS_PATH_TEMPLATE:-"${FREQ_STATS_ROOT}/exclude_{class}/freq_stats.pt"}
SHOT_OUTPUT_TEMPLATE=${SHOT_OUTPUT_TEMPLATE:-"${OUTPUT_ROOT}/exclude_{class}/shot_{shot}"}
REFERENCE_CSV_TEMPLATE=${REFERENCE_CSV_TEMPLATE:-"${REFERENCE_MAIN_ROOT}/exclude_{class}/formal_eval/step_{step}/ddfsd_eval_per_seed.csv"}

case "${AGGREGATE}" in auto|0|1) ;; *) echo "AGGREGATE must be auto, 0, or 1." >&2; exit 1 ;; esac
case "${SKIP_COMPLETED}" in 0|1) ;; *) echo "SKIP_COMPLETED must be 0 or 1." >&2; exit 1 ;; esac
[[ "${MAX_EVAL_QUERY_PER_CLASS}" == "0" ]] || {
  echo "Formal main-protocol evaluation requires MAX_EVAL_QUERY_PER_CLASS=0." >&2
  exit 1
}

render_template() {
  local value=$1
  local class_name=$2
  local shot=$3
  value=${value//\{class\}/${class_name}}
  value=${value//\{shot\}/${shot}}
  value=${value//\{step\}/${CKPT_STEP}}
  printf '%s\n' "${value}"
}

is_complete() {
  local output_dir=$1
  local shot=$2
  if [[ "${shot}" == "0" ]]; then
    local name
    for name in ddfsd_zero_shot_per_seed.csv ddfsd_zero_shot_summary.csv \
      zero_shot_metadata_manifest.csv zero_shot_invalid_images.csv zero_shot_config.json eval.log; do
      [[ -s "${output_dir}/${name}" ]] || return 1
    done
  else
    local name
    for name in ddfsd_eval_per_seed.csv ddfsd_eval_summary.csv \
      support_query_manifest.csv config.json eval.log; do
      [[ -s "${output_dir}/${name}" ]] || return 1
    done
  fi
}

echo "protocol=main-protocol formal shot ablation"
echo "CLASSES=${CLASSES}"
echo "SHOTS=${SHOTS}"
echo "EVAL_SEEDS=${EVAL_SEEDS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "FREQ_STATS_ROOT=${FREQ_STATS_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "REFERENCE_MAIN_ROOT=${REFERENCE_MAIN_ROOT}"
echo "CKPT_STEP=${CKPT_STEP}"
echo "MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS}"
echo "Run this command in a named screen session; this script performs evaluation only."

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

read -r -a requested_classes <<< "${CLASSES}"
read -r -a requested_shots <<< "${SHOTS}"
[[ ${#requested_classes[@]} -gt 0 ]] || { echo "CLASSES is empty." >&2; exit 1; }
[[ ${#requested_shots[@]} -gt 0 ]] || { echo "SHOTS is empty." >&2; exit 1; }

declare -A seen_classes=()
declare -A seen_shots=()
for shot in "${requested_shots[@]}"; do
  case " 0 5 10 20 30 50 " in *" ${shot} "*) ;; *) echo "Unsupported formal shot: ${shot}" >&2; exit 1 ;; esac
  [[ -z "${seen_shots[${shot}]+x}" ]] || { echo "Duplicate shot: ${shot}" >&2; exit 1; }
  seen_shots[${shot}]=1
done

for class_name in "${requested_classes[@]}"; do
  case " ${ALL_CLASSES} " in *" ${class_name} "*) ;; *) echo "Unknown class: ${class_name}" >&2; exit 1 ;; esac
  [[ -z "${seen_classes[${class_name}]+x}" ]] || { echo "Duplicate class: ${class_name}" >&2; exit 1; }
  seen_classes[${class_name}]=1

  ckpt_path=$(render_template "${CKPT_PATH_TEMPLATE}" "${class_name}" "")
  freq_stats_path=$(render_template "${FREQ_STATS_PATH_TEMPLATE}" "${class_name}" "")
  [[ -f "${ckpt_path}" ]] || { echo "[${class_name}] checkpoint missing: ${ckpt_path}" >&2; exit 1; }
  [[ -f "${freq_stats_path}" ]] || { echo "[${class_name}] freq stats missing: ${freq_stats_path}" >&2; exit 1; }

  for shot in "${requested_shots[@]}"; do
    output_dir=$(render_template "${SHOT_OUTPUT_TEMPLATE}" "${class_name}" "${shot}")
    if is_complete "${output_dir}" "${shot}"; then
      if [[ "${SKIP_COMPLETED}" == "1" ]]; then
        echo "[SKIP] exclude_${class_name}/shot_${shot} is complete"
        continue
      fi
      echo "Refusing to overwrite complete output: ${output_dir}" >&2
      exit 1
    fi
    if [[ -d "${output_dir}" ]]; then
      echo "Refusing incomplete existing output directory: ${output_dir}" >&2
      exit 1
    fi
    mkdir -p "${output_dir}"
    echo "[RUN] exclude_class=${class_name} shot=${shot} output=${output_dir}"
    if [[ "${shot}" == "0" ]]; then
      "${PYTHON_BIN}" test_ddfsd_zero_shot_main_protocol.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${output_dir}" \
        --num_workers "${NUM_WORKERS}" \
        --seed 42 \
        --exclude_class "${class_name}" \
        --ckpt_path "${ckpt_path}" \
        --ckpt_step "${CKPT_STEP}" \
        --model_mode "${MODEL_MODE}" \
        --branch_mode "${BRANCH_MODE}" \
        --freq_stats_path "${freq_stats_path}" \
        --eval_repeats 5 \
        --eval_seeds "${EVAL_SEEDS}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --zero_shot_metadata_per_class "${ZERO_SHOT_METADATA_PER_CLASS}" \
        --tau "${TAU}" \
        --tau_r "${TAU_R}" \
        --use_fp16 true \
        --pretrained false 2>&1 | tee "${output_dir}/eval.log"
    else
      "${PYTHON_BIN}" test_ddfsd.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${output_dir}" \
        --num_workers "${NUM_WORKERS}" \
        --seed 42 \
        --exclude_class "${class_name}" \
        --ckpt_path "${ckpt_path}" \
        --ckpt_step "${CKPT_STEP}" \
        --model_mode "${MODEL_MODE}" \
        --branch_mode "${BRANCH_MODE}" \
        --freq_stats_path "${freq_stats_path}" \
        --num_support_test "${shot}" \
        --eval_repeats 5 \
        --eval_seeds "${EVAL_SEEDS}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --max_eval_query_per_class "${MAX_EVAL_QUERY_PER_CLASS}" \
        --tau "${TAU}" \
        --tau_r "${TAU_R}" \
        --save_manifest true \
        --manifest_path "${output_dir}/support_query_manifest.csv" \
        --use_fp16 true \
        --pretrained false 2>&1 | tee "${output_dir}/eval.log"
    fi
  done
done

should_aggregate=0
if [[ "${AGGREGATE}" == "1" ]]; then
  should_aggregate=1
elif [[ "${AGGREGATE}" == "auto" && "${CLASSES}" == "${ALL_CLASSES}" && "${SHOTS}" == "0 5 10 20 30 50" ]]; then
  should_aggregate=1
fi

if [[ "${should_aggregate}" == "1" ]]; then
  "${PYTHON_BIN}" tools/summarize_ddfsd_shot_ablation_main_protocol.py \
    --input_root "${OUTPUT_ROOT}" \
    --output_dir "${OUTPUT_ROOT}/summary" \
    --reference_main_root "${REFERENCE_MAIN_ROOT}" \
    --reference_csv_template "${REFERENCE_CSV_TEMPLATE}" \
    --ckpt_step "${CKPT_STEP}" \
    --eval_seeds "${EVAL_SEEDS}"
else
  echo "[AGGREGATE] skipped; run the summarizer after all six classes and six shots exist."
fi
