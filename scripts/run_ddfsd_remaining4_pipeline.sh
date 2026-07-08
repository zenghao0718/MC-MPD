#!/usr/bin/env bash
# Sequentially run the full DDFSD train+eval pipeline for the four remaining exclude classes:
# glide -> Midjourney -> SD -> VQDM. Meant to be launched inside `screen` so it survives
# disconnects. On a class's TRAIN failure, the failure is recorded and the script proceeds
# to the next class (it does NOT silently skip -- see RUN_ROOT/remaining4_pipeline_status.log).
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
CLASSES=("glide" "Midjourney" "SD" "VQDM")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STATUS_LOG="${RUN_ROOT}/remaining4_pipeline_status.log"
mkdir -p "${RUN_ROOT}"
echo "=== remaining4 pipeline start: $(date -Iseconds) ===" >> "${STATUS_LOG}"

export DATA_ROOT RUN_ROOT NUM_WORKERS SEED

for CLASS in "${CLASSES[@]}"; do
    echo "============================================================" | tee -a "${STATUS_LOG}"
    echo "=== [$(date -Iseconds)] START exclude_class=${CLASS} ===" | tee -a "${STATUS_LOG}"
    echo "============================================================" | tee -a "${STATUS_LOG}"

    EXCLUDE_CLASS="${CLASS}" bash scripts/run_ddfsd_one_class_pipeline.sh
    EXIT_CODE=$?

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[$(date -Iseconds)] exclude_class=${CLASS} pipeline_result=OK" | tee -a "${STATUS_LOG}"
    else
        echo "[$(date -Iseconds)] exclude_class=${CLASS} pipeline_result=FAILED exit_code=${EXIT_CODE}" | tee -a "${STATUS_LOG}"
        echo "[$(date -Iseconds)] exclude_class=${CLASS} FAILED -- see ${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${CLASS}/logs/ and PIPELINE_STATUS.txt -- continuing to next class" | tee -a "${STATUS_LOG}"
    fi
done

echo "=== remaining4 pipeline end: $(date -Iseconds) ===" | tee -a "${STATUS_LOG}"
