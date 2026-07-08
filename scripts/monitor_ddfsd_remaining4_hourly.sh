#!/usr/bin/env bash
# Lightweight hourly status monitor for the remaining-4-class DDFSD pipeline.
# Appends one status block per hour to RUN_ROOT/remaining4_hourly_status.log.
# Stops automatically once run_ddfsd_remaining4_pipeline.sh (and train_ddfsd.py) are no
# longer running. Does NOT poll more than once per hour.
set -uo pipefail

RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
STATUS_LOG="${RUN_ROOT}/remaining4_hourly_status.log"
INTERVAL_SEC=${INTERVAL_SEC:-3600}

mkdir -p "${RUN_ROOT}"

get_current_class() {
    ps aux | grep '[t]rain_ddfsd.py' | grep -oP '(?<=--exclude_class )\S+' | head -n1
}

while true; do
    PIPELINE_RUNNING=$(pgrep -f "run_ddfsd_remaining4_pipeline.sh" || true)
    TRAIN_RUNNING=$(pgrep -f "train_ddfsd.py" || true)

    {
        echo "===== $(date -Iseconds) ====="
        if [[ -z "${PIPELINE_RUNNING}" && -z "${TRAIN_RUNNING}" ]]; then
            echo "pipeline_running=NO train_running=NO -- monitor stopping."
        else
            CURRENT_CLASS=$(get_current_class)
            echo "pipeline_running=$([[ -n ${PIPELINE_RUNNING} ]] && echo YES || echo NO)"
            echo "train_running=$([[ -n ${TRAIN_RUNNING} ]] && echo YES || echo NO)"
            echo "current_class=${CURRENT_CLASS:-UNKNOWN}"

            if [[ -n "${CURRENT_CLASS}" ]]; then
                OUTPUT_PATH="${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${CURRENT_CLASS}"
                LOG_FILE="${OUTPUT_PATH}/logs/train_${CURRENT_CLASS}_10pct.log"
                echo "log_path=${LOG_FILE}"
                LATEST_STEP_LINE=$(grep -a "Validation val_seen\|Validation val_unseen" "${LOG_FILE}" 2>/dev/null | tail -n1)
                echo "latest_validation_line=${LATEST_STEP_LINE:-NA}"
                DEBUG_DIR_LOG=$(ls -t "${OUTPUT_PATH}/logs"/*_log.txt 2>/dev/null | head -n1)
                if [[ -n "${DEBUG_DIR_LOG}" ]]; then
                    LATEST_DEBUG_LINE=$(grep -a "DEBUG {" "${DEBUG_DIR_LOG}" 2>/dev/null | tail -n1)
                    echo "latest_debug_line=${LATEST_DEBUG_LINE:-NA}"
                fi
                echo "checkpoints_present:"
                find "${OUTPUT_PATH}/ckpt" -maxdepth 1 -type f -name "*.pth" -printf "  %p %s bytes\n" 2>/dev/null
            fi

            echo "gpu_status:"
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
        fi
        echo ""
    } >> "${STATUS_LOG}" 2>&1

    if [[ -z "${PIPELINE_RUNNING}" && -z "${TRAIN_RUNNING}" ]]; then
        break
    fi

    sleep "${INTERVAL_SEC}"
done
