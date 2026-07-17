#!/usr/bin/env bash
set -euo pipefail

ALL_CLASSES="ADM BigGAN glide Midjourney SD VQDM"
CLASSES=${CLASSES:-"${ALL_CLASSES}"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
CKPT_ROOT=${CKPT_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_ckpts_step15000"}
FREQ_STATS_ROOT=${FREQ_STATS_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/shared_full_freq_stats"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_eval_step15000"}
CKPT_STEP=${CKPT_STEP:-15000}
CKPT_PATH_TEMPLATE=${CKPT_PATH_TEMPLATE:-"${CKPT_ROOT}/exclude_{class}/ckpt/ddfsd_step[{step}].pth"}
FREQ_STATS_PATH_TEMPLATE=${FREQ_STATS_PATH_TEMPLATE:-"${FREQ_STATS_ROOT}/exclude_{class}/freq_stats.pt"}
MULTISHOT_OUTPUT_TEMPLATE=${MULTISHOT_OUTPUT_TEMPLATE:-"${OUTPUT_ROOT}/exclude_{class}/multishot"}
SHOT_LIST=${SHOT_LIST:-"0,1,2,5,10,20"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
ZERO_SHOT_METADATA_PER_CLASS=${ZERO_SHOT_METADATA_PER_CLASS:-1024}
MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS:-0}
AGGREGATE=${AGGREGATE:-auto}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
FORCE_RERUN=${FORCE_RERUN:-0}

case "${AGGREGATE}" in auto|0|1) ;; *) echo "AGGREGATE must be auto, 0, or 1; got: ${AGGREGATE}" >&2; exit 1 ;; esac
case "${SKIP_COMPLETED}" in 0|1) ;; *) echo "SKIP_COMPLETED must be 0 or 1; got: ${SKIP_COMPLETED}" >&2; exit 1 ;; esac
case "${FORCE_RERUN}" in 0|1) ;; *) echo "FORCE_RERUN must be 0 or 1; got: ${FORCE_RERUN}" >&2; exit 1 ;; esac

render_template() {
  local value=$1
  local class_name=$2
  value=${value//\{class\}/${class_name}}
  value=${value//\{step\}/${CKPT_STEP}}
  printf '%s\n' "${value}"
}

absolute_path() {
  python -c 'import os,sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' "$1"
}

is_complete() {
  local directory=$1
  local name
  for name in multishot_per_seed.csv multishot_summary.csv support_query_manifest.csv zero_shot_metadata_manifest.csv multishot_config.json; do
    [[ -s "${directory}/${name}" ]] || return 1
  done
}

validate_output_path() {
  local output_abs=$1
  local output_root_abs=$2
  local ckpt_root_abs=$3
  local freq_root_abs=$4
  case "${output_abs}" in "${output_root_abs}"/*) ;; *)
    echo "Resolved output directory is outside OUTPUT_ROOT: ${output_abs}" >&2
    exit 1
  esac
  if [[ "${output_abs}" == "${output_root_abs}" || "${output_abs}" == "${ckpt_root_abs}" || "${output_abs}" == "${freq_root_abs}" ]]; then
    echo "Refusing unsafe output directory: ${output_abs}" >&2
    exit 1
  fi
}

echo "DATA_ROOT=${DATA_ROOT}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "FREQ_STATS_ROOT=${FREQ_STATS_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "CKPT_STEP=${CKPT_STEP}"
echo "SHOT_LIST=${SHOT_LIST}"
echo "EVAL_SEEDS=${EVAL_SEEDS}"
echo "CLASSES=${CLASSES}"
echo "AGGREGATE=${AGGREGATE}"
echo "SKIP_COMPLETED=${SKIP_COMPLETED}"
echo "FORCE_RERUN=${FORCE_RERUN}"
echo "MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS}"
echo "CKPT_PATH_TEMPLATE=${CKPT_PATH_TEMPLATE}"
echo "FREQ_STATS_PATH_TEMPLATE=${FREQ_STATS_PATH_TEMPLATE}"
echo "MULTISHOT_OUTPUT_TEMPLATE=${MULTISHOT_OUTPUT_TEMPLATE}"

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
output_root_abs=$(absolute_path "${OUTPUT_ROOT}")
ckpt_root_abs=$(absolute_path "${CKPT_ROOT}")
freq_root_abs=$(absolute_path "${FREQ_STATS_ROOT}")
case "${output_root_abs}" in
  "${ckpt_root_abs}"|"${ckpt_root_abs}"/*)
    echo "OUTPUT_ROOT must not be the checkpoint root or a child of it: ${output_root_abs}" >&2; exit 1 ;;
  "${freq_root_abs}"|"${freq_root_abs}"/*)
    echo "OUTPUT_ROOT must not be the frequency-stats root or a child of it: ${output_root_abs}" >&2; exit 1 ;;
esac
mkdir -p "${OUTPUT_ROOT}"

read -r -a requested_classes <<< "${CLASSES}"
[[ ${#requested_classes[@]} -gt 0 ]] || { echo "CLASSES is empty." >&2; exit 1; }
declare -A seen_classes=()
for class_name in "${requested_classes[@]}"; do
  case " ${ALL_CLASSES} " in *" ${class_name} "*) ;; *) echo "Unknown class: ${class_name}" >&2; exit 1 ;; esac
  [[ -z "${seen_classes[${class_name}]+x}" ]] || { echo "Duplicate class in CLASSES: ${class_name}" >&2; exit 1; }
  seen_classes[${class_name}]=1

  ckpt_path=$(render_template "${CKPT_PATH_TEMPLATE}" "${class_name}")
  freq_path=$(render_template "${FREQ_STATS_PATH_TEMPLATE}" "${class_name}")
  output_dir=$(render_template "${MULTISHOT_OUTPUT_TEMPLATE}" "${class_name}")
  output_abs=$(absolute_path "${output_dir}")
  validate_output_path "${output_abs}" "${output_root_abs}" "${ckpt_root_abs}" "${freq_root_abs}"

  echo "exclude_class=${class_name}"
  echo "resolved ckpt_path=${ckpt_path}"
  echo "resolved freq_stats_path=${freq_path}"
  echo "resolved output_dir=${output_abs}"
  [[ -f "${ckpt_path}" ]] || { echo "[${class_name}] checkpoint does not exist: ${ckpt_path}" >&2; exit 1; }
  [[ -f "${freq_path}" ]] || { echo "[${class_name}] freq stats do not exist: ${freq_path}" >&2; exit 1; }

  if [[ "${FORCE_RERUN}" == "1" && -d "${output_abs}" ]]; then
    echo "[FORCE_RERUN] deleting validated class output: ${output_abs}"
    rm -rf -- "${output_abs}"
  elif is_complete "${output_abs}"; then
    if [[ "${SKIP_COMPLETED}" == "1" ]]; then
      echo "[SKIP] ${class_name} already completed"
      continue
    fi
    echo "[${class_name}] complete results already exist: ${output_abs}; use SKIP_COMPLETED=1 or FORCE_RERUN=1." >&2
    exit 1
  elif [[ -d "${output_abs}" ]]; then
    echo "[${class_name}] incomplete result directory exists: ${output_abs}" >&2
    echo "Use a new MULTISHOT_OUTPUT_TEMPLATE or set FORCE_RERUN=1." >&2
    exit 1
  fi

  log_dir="${OUTPUT_ROOT}/exclude_${class_name}"
  mkdir -p "${log_dir}"
  EXCLUDE_CLASS="${class_name}" DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${output_abs}" \
    CKPT_PATH="${ckpt_path}" FREQ_STATS_PATH="${freq_path}" CKPT_STEP="${CKPT_STEP}" \
    SHOT_LIST="${SHOT_LIST}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" ZERO_SHOT_METADATA_PER_CLASS="${ZERO_SHOT_METADATA_PER_CLASS}" \
    MAX_EVAL_QUERY_PER_CLASS="${MAX_EVAL_QUERY_PER_CLASS}" \
    bash scripts/eval_ddfsd_multishot.sh 2>&1 | tee "${log_dir}/multishot_eval.log"
done

should_aggregate=0
if [[ "${AGGREGATE}" == "1" ]]; then
  should_aggregate=1
elif [[ "${AGGREGATE}" == "auto" ]]; then
  if [[ ${#requested_classes[@]} -eq 6 ]]; then
    should_aggregate=1
    for class_name in ${ALL_CLASSES}; do
      [[ -n "${seen_classes[${class_name}]+x}" ]] || should_aggregate=0
      class_output=$(render_template "${MULTISHOT_OUTPUT_TEMPLATE}" "${class_name}")
      is_complete "${class_output}" || should_aggregate=0
    done
  fi
fi

if [[ "${should_aggregate}" == "1" ]]; then
  python tools/summarize_ddfsd_multishot.py \
    --input_root "${OUTPUT_ROOT}" --output_dir "${OUTPUT_ROOT}/multishot_summary" \
    --result_dir_template "${MULTISHOT_OUTPUT_TEMPLATE}" --ckpt_step "${CKPT_STEP}"
else
  echo "[AGGREGATE] skipped: mode=${AGGREGATE}, requested classes=${CLASSES}"
fi
