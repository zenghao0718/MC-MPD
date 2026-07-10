#!/usr/bin/env bash
# Sequentially run the full DDFSD train+eval+diagnostics pipeline for all six
# leave-one-out exclude classes with margins m_rf=1.3 / m_ff=0.7:
#   ADM -> BigGAN -> glide -> Midjourney -> SD -> VQDM
#
# Meant to be launched inside `screen` so it survives disconnects. On a
# class's TRAIN failure, the failure is recorded and the script proceeds to
# the next class (it does NOT silently skip -- see
# ${RUN_ROOT}/${EXPERIMENT_NAME}/all6_pipeline_status.log and each class's
# own PIPELINE_STATUS.txt / logs/).
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/miniGenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"ddfsd_10pct_steps15000_mrf1p3_mff0p7"}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
M_RF=${M_RF:-1.3}
M_FF=${M_FF:-0.7}
CLASSES=("ADM" "BigGAN" "glide" "Midjourney" "SD" "VQDM")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${RUN_ROOT}/${EXPERIMENT_NAME}"
STATUS_LOG="${EXPERIMENT_ROOT}/all6_pipeline_status.log"
mkdir -p "${EXPERIMENT_ROOT}"
echo "=== all6 margin pipeline start: $(date -Iseconds) DATA_ROOT=${DATA_ROOT} M_RF=${M_RF} M_FF=${M_FF} ===" >> "${STATUS_LOG}"

export DATA_ROOT RUN_ROOT NUM_WORKERS SEED M_RF M_FF

for CLASS in "${CLASSES[@]}"; do
    OUTPUT_PATH="${EXPERIMENT_ROOT}/exclude_${CLASS}"
    FREQ_STATS_PATH="${OUTPUT_PATH}/freq_stats.pt"

    echo "============================================================" | tee -a "${STATUS_LOG}"
    echo "=== [$(date -Iseconds)] START exclude_class=${CLASS} OUTPUT_PATH=${OUTPUT_PATH} ===" | tee -a "${STATUS_LOG}"
    echo "============================================================" | tee -a "${STATUS_LOG}"

    EXCLUDE_CLASS="${CLASS}" OUTPUT_PATH="${OUTPUT_PATH}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
        bash scripts/run_ddfsd_one_class_pipeline.sh
    EXIT_CODE=$?

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[$(date -Iseconds)] exclude_class=${CLASS} pipeline_result=OK" | tee -a "${STATUS_LOG}"
    else
        echo "[$(date -Iseconds)] exclude_class=${CLASS} pipeline_result=FAILED exit_code=${EXIT_CODE}" | tee -a "${STATUS_LOG}"
        echo "[$(date -Iseconds)] exclude_class=${CLASS} FAILED -- see ${OUTPUT_PATH}/logs/ and ${OUTPUT_PATH}/PIPELINE_STATUS.txt -- continuing to next class" | tee -a "${STATUS_LOG}"
    fi
done

echo "=== all6 margin pipeline end: $(date -Iseconds) ===" | tee -a "${STATUS_LOG}"
