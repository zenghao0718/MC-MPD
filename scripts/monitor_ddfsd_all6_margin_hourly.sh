#!/usr/bin/env bash
# Lightweight hourly status monitor for the all-6-class m_rf=1.3/m_ff=0.7 DDFSD pipeline.
# Appends one status block per hour to ${RUN_ROOT}/${EXPERIMENT_NAME}/all6_hourly_status.log.
# Stops automatically once run_ddfsd_all6_margin_pipeline.sh (and train_ddfsd.py) are no
# longer running. Does NOT poll more than once per hour (INTERVAL_SEC=3600 by default).
set -uo pipefail

RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"ddfsd_10pct_steps15000_mrf1p3_mff0p7"}
EXPERIMENT_ROOT="${RUN_ROOT}/${EXPERIMENT_NAME}"
STATUS_LOG="${EXPERIMENT_ROOT}/all6_hourly_status.log"
INTERVAL_SEC=${INTERVAL_SEC:-3600}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${EXPERIMENT_ROOT}"

get_current_class() {
    ps aux | grep '[t]rain_ddfsd.py' | grep -oP '(?<=--exclude_class )\S+' | head -n1
}

print_debug_summary() {
    local debug_line="$1"
    python3 - "${debug_line}" <<'PYEOF'
import ast
import sys

line = sys.argv[1]
idx = line.find("DEBUG {")
if idx == -1:
    print("  (could not locate DEBUG dict in latest log line)")
    sys.exit(0)
try:
    record = ast.literal_eval(line[idx + len("DEBUG "):].strip())
except (ValueError, SyntaxError):
    print("  (failed to parse DEBUG dict)")
    sys.exit(0)

def g(key, fmt="{:.6f}"):
    val = record.get(key)
    if val is None:
        return "NA"
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val)

print(f"  step={record.get('step', 'NA')}")
print(f"  loss_total={g('loss_total')} loss_dual={g('loss_dual')} loss_sep={g('loss_sep')} loss_rf={g('loss_rf')} loss_ff={g('loss_ff')}")
print(f"  weighted_loss_sep={g('weighted_loss_sep')} weighted_sep_to_dual_ratio={g('weighted_sep_to_dual_ratio')}")
print(f"  alpha_mean={g('alpha_mean')} alpha_min={g('alpha_min')} alpha_max={g('alpha_max')}")
print(f"  proto_rf_fused_mean={g('proto_rf_fused_mean')} p10={g('proto_rf_fused_p10')} p50={g('proto_rf_fused_p50')} p90={g('proto_rf_fused_p90')}")
print(f"  proto_ff_fused_mean={g('proto_ff_fused_mean')} p10={g('proto_ff_fused_p10')} p50={g('proto_ff_fused_p50')} p90={g('proto_ff_fused_p90')}")
print(f"  rf_violation_rate={g('rf_violation_rate')} ff_violation_rate={g('ff_violation_rate')}")
PYEOF
}

while true; do
    PIPELINE_RUNNING=$(pgrep -f "run_ddfsd_all6_margin_pipeline.sh" || true)
    TRAIN_RUNNING=$(pgrep -f "train_ddfsd.py" || true)

    {
        echo "===== $(date -Iseconds) ====="
        if [[ -z "${PIPELINE_RUNNING}" && -z "${TRAIN_RUNNING}" ]]; then
            echo "pipeline_running=NO train_running=NO -- monitor stopping."
        else
            CURRENT_CLASS=$(get_current_class)
            echo "pipeline_running=$([[ -n ${PIPELINE_RUNNING} ]] && echo YES || echo NO)"
            echo "train_running=$([[ -n ${TRAIN_RUNNING} ]] && echo YES || echo NO)"
            echo "current_exclude_class=${CURRENT_CLASS:-UNKNOWN}"

            echo "gpu_status:"
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'

            if [[ -n "${CURRENT_CLASS}" ]]; then
                OUTPUT_PATH="${EXPERIMENT_ROOT}/exclude_${CURRENT_CLASS}"
                echo "output_path=${OUTPUT_PATH}"

                LATEST_CKPT=$(ls -t "${OUTPUT_PATH}/ckpt"/*.pth 2>/dev/null | head -n1)
                echo "latest_checkpoint=${LATEST_CKPT:-NA}"

                DEBUG_DIR_LOG=$(ls -t "${OUTPUT_PATH}/logs"/*_log.txt 2>/dev/null | head -n1)
                if [[ -n "${DEBUG_DIR_LOG}" ]]; then
                    echo "log_path=${DEBUG_DIR_LOG}"
                    LATEST_DEBUG_LINE=$(grep -a "DEBUG {" "${DEBUG_DIR_LOG}" 2>/dev/null | tail -n1)
                    if [[ -n "${LATEST_DEBUG_LINE}" ]]; then
                        print_debug_summary "${LATEST_DEBUG_LINE}"
                    else
                        echo "  (no DEBUG step record yet)"
                    fi
                else
                    echo "log_path=NA (no *_log.txt found yet)"
                fi
            fi
        fi
        echo ""
    } >> "${STATUS_LOG}" 2>&1

    if [[ -z "${PIPELINE_RUNNING}" && -z "${TRAIN_RUNNING}" ]]; then
        break
    fi

    sleep "${INTERVAL_SEC}"
done
