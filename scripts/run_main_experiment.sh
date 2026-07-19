#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-all}
CLASSES=${CLASSES:-"ADM BigGAN glide Midjourney SD VQDM"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/Dual-Domain-Few-Shot-AIGI-Detector"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"main_full_steps15000"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

case "${MODE}" in train|eval|all) ;; *) echo "MODE must be train, eval, or all: ${MODE}" >&2; exit 2 ;; esac
case "$(realpath -m "${RUN_ROOT}")/" in
  "$(realpath -m "${REPO_ROOT}")/"*)
    echo "RUN_ROOT must be outside the repository: ${RUN_ROOT}" >&2
    exit 1
    ;;
esac

for class_name in ${CLASSES}; do
  case "${class_name}" in ADM|BigGAN|glide|Midjourney|SD|VQDM) ;; *) echo "Invalid class: ${class_name}" >&2; exit 2 ;; esac
  run_dir="${RUN_ROOT}/${EXPERIMENT_NAME}/exclude_${class_name}"
  mkdir -p "${run_dir}/logs"
  echo "[$(date -Is)] Starting ${MODE} for exclude_${class_name}"

  if [[ "${MODE}" == train || "${MODE}" == all ]]; then
    set +e
    EXCLUDE_CLASS="${class_name}" RUN_ROOT="${RUN_ROOT}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
      bash "${SCRIPT_DIR}/train_main.sh" 2>&1 | tee "${run_dir}/logs/train_driver.log"
    status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
      echo "FAILED: train exclude_${class_name}, exit code ${status}" >&2
      exit "${status}"
    fi
  fi

  if [[ "${MODE}" == eval || "${MODE}" == all ]]; then
    set +e
    EXCLUDE_CLASS="${class_name}" RUN_ROOT="${RUN_ROOT}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
      bash "${SCRIPT_DIR}/eval_main.sh" 2>&1 | tee "${run_dir}/logs/eval_driver.log"
    status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
      echo "FAILED: eval exclude_${class_name}, exit code ${status}" >&2
      exit "${status}"
    fi
  fi
  echo "[$(date -Is)] Completed exclude_${class_name}"
done

echo "SUCCESS: mode=${MODE}; classes=${CLASSES}"
