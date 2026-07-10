#!/usr/bin/env bash
# Sequential DDFSD 1/5 (20%) pipeline for the six GenImage leave-one-out classes.
# A class is skipped only when its last pipeline state is COMPLETE and all key
# checkpoints/CSVs/reports are present. Failed classes are recorded and the next
# class still runs, allowing safe class-level continuation on a single GPU.
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_20pct/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
RUN_CONFIG=${RUN_CONFIG:-"ddfsd_20pct_steps30000"}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"${RUN_ROOT}/${RUN_CONFIG}"}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-16}
TOTAL_STEPS=${TOTAL_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-200}
LR_STEP=${LR_STEP:-10000}
LR_GAMMA=${LR_GAMMA:-0.5}

RGB_BACKBONE_LR=${RGB_BACKBONE_LR:-3e-5}
FREQ_BACKBONE_LR=${FREQ_BACKBONE_LR:-3e-5}
RGB_HEAD_LR=${RGB_HEAD_LR:-1e-4}
FREQ_HEAD_LR=${FREQ_HEAD_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
TAU=${TAU:-0.2}
TAU_R=${TAU_R:-0.1}
M_RF=${M_RF:-1.2}
M_FF=${M_FF:-0.6}
LAMBDA_FF=${LAMBDA_FF:-0.5}
LAMBDA_SEP_TARGET=${LAMBDA_SEP_TARGET:-0.03}
LAMBDA_SEP_WARMUP_START=${LAMBDA_SEP_WARMUP_START:-5000}
LAMBDA_SEP_WARMUP_END=${LAMBDA_SEP_WARMUP_END:-15000}
BRANCH_DROPOUT_DUAL_PROB=${BRANCH_DROPOUT_DUAL_PROB:-0.90}
BRANCH_DROPOUT_RGB_PROB=${BRANCH_DROPOUT_RGB_PROB:-0.05}
BRANCH_DROPOUT_FREQ_PROB=${BRANCH_DROPOUT_FREQ_PROB:-0.05}
AUTO_COMPUTE_FREQ_STATS=${AUTO_COMPUTE_FREQ_STATS:-True}
USE_FP16=${USE_FP16:-True}
PRETRAINED=${PRETRAINED:-True}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS:-0}

EXPECTED_CKPT_STEPS=${EXPECTED_CKPT_STEPS:-"5000,10000,15000,20000,25000,30000"}
FORMAL_EVAL_STEPS=${FORMAL_EVAL_STEPS:-"5000,10000,15000,20000,25000,30000"}
BRANCH_MODE_STEPS=${BRANCH_MODE_STEPS:-"5000,10000,15000,20000,25000,30000"}
ALPHA_GRID_STEPS=${ALPHA_GRID_STEPS:-"15000,25000,30000"}
TRAIN_STATS_STEPS=${TRAIN_STATS_STEPS:-"5000,10000,15000,20000,25000,30000"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
SUPPORT_SHOT=${SUPPORT_SHOT:-10}
BRANCH_MODES=${BRANCH_MODES:-"dual,rgb-only,freq-only"}
ALPHA_MODES=${ALPHA_MODES:-"adaptive,0.0,0.25,0.5,0.75,1.0"}
ALPHA_BRANCH_MODES=${ALPHA_BRANCH_MODES:-"dual"}

# ONLY_CLASSES is the preferred subset override. CLASSES and EXCLUDE_CLASSES are
# accepted as aliases. Values may be comma-, semicolon-, or whitespace-separated.
CLASS_SPEC=${ONLY_CLASSES:-${CLASSES:-${EXCLUDE_CLASSES:-"ADM,BigGAN,glide,Midjourney,SD,VQDM"}}}
CLASS_SPEC_NORMALIZED=${CLASS_SPEC//,/ }
CLASS_SPEC_NORMALIZED=${CLASS_SPEC_NORMALIZED//;/ }
CLASS_SPEC_NORMALIZED=${CLASS_SPEC_NORMALIZED//$'\t'/ }
read -r -a REQUESTED_CLASSES <<< "${CLASS_SPEC_NORMALIZED}"
ALL_CLASSES=("ADM" "BigGAN" "glide" "Midjourney" "SD" "VQDM")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STATUS_FILE="${EXPERIMENT_ROOT}/PIPELINE_STATUS_ALL.txt"
CONFIG_FILE="${EXPERIMENT_ROOT}/PIPELINE_CONFIG_ALL.txt"
GLOBAL_LOG_DIR="${EXPERIMENT_ROOT}/logs"
SUMMARY_DIR="${EXPERIMENT_ROOT}/summary"
ATTEMPT_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${EXPERIMENT_ROOT}" "${GLOBAL_LOG_DIR}" "${SUMMARY_DIR}"

status_append() {
    printf '%s\n' "$*" >> "${STATUS_FILE}"
}

latest_pipeline_state() {
    local path=$1
    [[ -f "${path}" ]] || return 1
    awk -F= '/^pipeline_state=/{state=$2} END{if (state != "") print state}' "${path}"
}

validate_completeness_csv() {
    local csv_path=$1
    local expected_class=$2
    [[ -s "${csv_path}" ]] || return 1
    python - "${csv_path}" "${expected_class}" <<'PYEOF'
import csv
import sys

path, expected_class = sys.argv[1:3]
core = {
    "class_directory", "pipeline_status", "config_snapshot", "freq_stats", "checkpoints",
    "formal_summary", "formal_per_seed", "branch_modes_summary",
    "branch_modes_per_seed", "alpha_grid_summary", "alpha_grid_per_seed",
    "train_alpha_loss",
}
try:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
except (OSError, csv.Error) as exc:
    print(f"Cannot read completeness CSV {path}: {exc}", file=sys.stderr)
    sys.exit(1)

found = {}
for row in rows:
    if str(row.get("exclude_class", "")).strip() != expected_class:
        continue
    component = str(row.get("component", "")).strip()
    if component in core:
        if component in found:
            print(f"Duplicate completeness component {component} in {path}", file=sys.stderr)
            sys.exit(1)
        found[component] = str(row.get("status", "")).strip()

missing = sorted(core - set(found))
bad = sorted(f"{name}={status}" for name, status in found.items() if status != "完整")
if missing or bad:
    print(f"Core completeness failed for {expected_class}: missing={missing}, bad={bad}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

is_allowed_class() {
    case "$1" in
        ADM|BigGAN|glide|Midjourney|SD|VQDM) return 0 ;;
        *) return 1 ;;
    esac
}

class_is_complete() {
    local class_name=$1
    local output_path="${EXPERIMENT_ROOT}/exclude_${class_name}"
    local status_path="${output_path}/PIPELINE_STATUS.txt"
    local raw_step step
    local -a steps

    [[ "$(latest_pipeline_state "${status_path}" 2>/dev/null || true)" == "COMPLETE" ]] || return 1
    [[ -s "${output_path}/PIPELINE_CONFIG.txt" ]] || return 1
    [[ -f "${output_path}/freq_stats.pt" ]] || return 1

    IFS=',' read -r -a steps <<< "${EXPECTED_CKPT_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        [[ -f "${output_path}/ckpt/ddfsd_step[${step}].pth" ]] || return 1
    done

    [[ -s "${output_path}/formal_eval/ddfsd_eval_per_seed_all_steps.csv" ]] || return 1
    [[ -s "${output_path}/formal_eval/ddfsd_eval_summary_all_steps.csv" ]] || return 1
    [[ -s "${output_path}/branch_modes/ddfsd_${class_name}_branch_modes_per_seed.csv" ]] || return 1
    [[ -s "${output_path}/branch_modes/ddfsd_${class_name}_branch_modes_summary.csv" ]] || return 1
    [[ -s "${output_path}/alpha_grid/ddfsd_${class_name}_alpha_grid_per_seed.csv" ]] || return 1
    [[ -s "${output_path}/alpha_grid/ddfsd_${class_name}_alpha_grid_summary.csv" ]] || return 1
    [[ -s "${output_path}/csv/ddfsd_${class_name}_train_alpha_loss_by_ckpt.csv" ]] || return 1
    [[ -s "${output_path}/summary/ddfsd_${class_name}_20pct_summary.md" ]] || return 1
    [[ -s "${output_path}/summary/ddfsd_${class_name}_20pct_summary.csv" ]] || return 1
    validate_completeness_csv \
        "${output_path}/summary/ddfsd_${class_name}_20pct_completeness.csv" "${class_name}" || return 1
    return 0
}

all_six_classes_complete() {
    local class_name
    for class_name in "${ALL_CLASSES[@]}"; do
        class_is_complete "${class_name}" || return 1
    done
    return 0
}

write_config_snapshot() {
    local candidate="${EXPERIMENT_ROOT}/.PIPELINE_CONFIG_ALL.${ATTEMPT_ID}.tmp"
    local alternate="${EXPERIMENT_ROOT}/PIPELINE_CONFIG_ALL_${ATTEMPT_ID}.txt"
    {
        printf 'DATA_ROOT=%s\n' "${DATA_ROOT}"
        printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
        printf 'RUN_CONFIG=%s\n' "${RUN_CONFIG}"
        printf 'EXPERIMENT_ROOT=%s\n' "${EXPERIMENT_ROOT}"
        printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
        printf 'SEED=%s\n' "${SEED}"
        printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
        printf 'TOTAL_STEPS=%s\n' "${TOTAL_STEPS}"
        printf 'SAVE_INTERVAL=%s\n' "${SAVE_INTERVAL}"
        printf 'EVAL_INTERVAL=%s\n' "${EVAL_INTERVAL}"
        printf 'LOG_INTERVAL=%s\n' "${LOG_INTERVAL}"
        printf 'LR_STEP=%s\n' "${LR_STEP}"
        printf 'LR_GAMMA=%s\n' "${LR_GAMMA}"
        printf 'EXPECTED_CKPT_STEPS=%s\n' "${EXPECTED_CKPT_STEPS}"
        printf 'FORMAL_EVAL_STEPS=%s\n' "${FORMAL_EVAL_STEPS}"
        printf 'BRANCH_MODE_STEPS=%s\n' "${BRANCH_MODE_STEPS}"
        printf 'ALPHA_GRID_STEPS=%s\n' "${ALPHA_GRID_STEPS}"
        printf 'TRAIN_STATS_STEPS=%s\n' "${TRAIN_STATS_STEPS}"
        printf 'EVAL_SEEDS=%s\n' "${EVAL_SEEDS}"
        printf 'SUPPORT_SHOT=%s\n' "${SUPPORT_SHOT}"
        printf 'BRANCH_MODES=%s\n' "${BRANCH_MODES}"
        printf 'ALPHA_MODES=%s\n' "${ALPHA_MODES}"
        printf 'ALPHA_BRANCH_MODES=%s\n' "${ALPHA_BRANCH_MODES}"
        printf 'EVAL_BATCH_SIZE=%s\n' "${EVAL_BATCH_SIZE}"
        printf 'MAX_EVAL_QUERY_PER_CLASS=%s\n' "${MAX_EVAL_QUERY_PER_CLASS}"
        printf 'RGB_BACKBONE_LR=%s\n' "${RGB_BACKBONE_LR}"
        printf 'FREQ_BACKBONE_LR=%s\n' "${FREQ_BACKBONE_LR}"
        printf 'RGB_HEAD_LR=%s\n' "${RGB_HEAD_LR}"
        printf 'FREQ_HEAD_LR=%s\n' "${FREQ_HEAD_LR}"
        printf 'WEIGHT_DECAY=%s\n' "${WEIGHT_DECAY}"
        printf 'TAU=%s\n' "${TAU}"
        printf 'TAU_R=%s\n' "${TAU_R}"
        printf 'M_RF=%s\n' "${M_RF}"
        printf 'M_FF=%s\n' "${M_FF}"
        printf 'LAMBDA_FF=%s\n' "${LAMBDA_FF}"
        printf 'LAMBDA_SEP_TARGET=%s\n' "${LAMBDA_SEP_TARGET}"
        printf 'LAMBDA_SEP_WARMUP_START=%s\n' "${LAMBDA_SEP_WARMUP_START}"
        printf 'LAMBDA_SEP_WARMUP_END=%s\n' "${LAMBDA_SEP_WARMUP_END}"
        printf 'BRANCH_DROPOUT_DUAL_PROB=%s\n' "${BRANCH_DROPOUT_DUAL_PROB}"
        printf 'BRANCH_DROPOUT_RGB_PROB=%s\n' "${BRANCH_DROPOUT_RGB_PROB}"
        printf 'BRANCH_DROPOUT_FREQ_PROB=%s\n' "${BRANCH_DROPOUT_FREQ_PROB}"
        printf 'AUTO_COMPUTE_FREQ_STATS=%s\n' "${AUTO_COMPUTE_FREQ_STATS}"
        printf 'USE_FP16=%s\n' "${USE_FP16}"
        printf 'PRETRAINED=%s\n' "${PRETRAINED}"
        printf 'TRAIN_EPISODE=3-way,5-shot,5-query\n'
        printf 'VALIDATION_SHOT=10\n'
    } > "${candidate}"

    if [[ ! -e "${CONFIG_FILE}" ]]; then
        mv -- "${candidate}" "${CONFIG_FILE}"
        return 0
    fi
    if cmp -s -- "${candidate}" "${CONFIG_FILE}"; then
        rm -f -- "${candidate}"
        return 0
    fi
    mv -- "${candidate}" "${alternate}"
    status_append "config_status=CHANGED_PRESERVED canonical=${CONFIG_FILE} new=${alternate}"
    return 1
}

status_append "pipeline_attempt_start=${ATTEMPT_ID} time=$(date -Iseconds)"
status_append "requested_classes=${CLASS_SPEC}"
if ! write_config_snapshot; then
    status_append "failure_reason=pipeline_config_mismatch_refusing_to_mix_runs"
    status_append "pipeline_state=FAILED"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "PIPELINE_CONFIG_ALL.txt differs from this invocation; preserved a timestamped candidate and stopped." >&2
    exit 2
fi

FAILURES=()
VALID_CLASS_COUNT=0

for CLASS_NAME in "${REQUESTED_CLASSES[@]}"; do
    CLASS_NAME="${CLASS_NAME//$'\t'/}"
    CLASS_NAME="${CLASS_NAME//$'\r'/}"
    CLASS_NAME="${CLASS_NAME//$'\n'/}"
    [[ -n "${CLASS_NAME}" ]] || continue
    if ! is_allowed_class "${CLASS_NAME}"; then
        echo "Invalid class '${CLASS_NAME}'; expected ADM, BigGAN, glide, Midjourney, SD, or VQDM." >&2
        status_append "class_${CLASS_NAME}_status=FAILED_INVALID_CLASS"
        FAILURES+=("invalid_class_${CLASS_NAME}")
        continue
    fi
    VALID_CLASS_COUNT=$((VALID_CLASS_COUNT + 1))

    CLASS_OUTPUT_PATH="${EXPERIMENT_ROOT}/exclude_${CLASS_NAME}"
    CLASS_FREQ_STATS_PATH="${CLASS_OUTPUT_PATH}/freq_stats.pt"
    CLASS_LOG_DIR="${CLASS_OUTPUT_PATH}/logs"
    CLASS_DRIVER_LOG="${CLASS_LOG_DIR}/all_classes_driver_${ATTEMPT_ID}.log"
    mkdir -p "${CLASS_LOG_DIR}"

    echo "============================================================" | tee -a "${CLASS_DRIVER_LOG}"
    echo "[$(date -Iseconds)] class=${CLASS_NAME} output=${CLASS_OUTPUT_PATH}" | tee -a "${CLASS_DRIVER_LOG}"
    echo "============================================================" | tee -a "${CLASS_DRIVER_LOG}"

    if class_is_complete "${CLASS_NAME}"; then
        echo "Verified COMPLETE status plus key checkpoints/CSVs; safely skipping ${CLASS_NAME}." | tee -a "${CLASS_DRIVER_LOG}"
        status_append "class_${CLASS_NAME}_status=SKIPPED_COMPLETE"
        continue
    fi

    DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" RUN_CONFIG="${RUN_CONFIG}" \
        EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" NUM_WORKERS="${NUM_WORKERS}" SEED="${SEED}" \
        BATCH_SIZE="${BATCH_SIZE}" TOTAL_STEPS="${TOTAL_STEPS}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
        EVAL_INTERVAL="${EVAL_INTERVAL}" LOG_INTERVAL="${LOG_INTERVAL}" \
        LR_STEP="${LR_STEP}" LR_GAMMA="${LR_GAMMA}" \
        EXCLUDE_CLASS="${CLASS_NAME}" OUTPUT_PATH="${CLASS_OUTPUT_PATH}" \
        FREQ_STATS_PATH="${CLASS_FREQ_STATS_PATH}" EXPECTED_CKPT_STEPS="${EXPECTED_CKPT_STEPS}" \
        FORMAL_EVAL_STEPS="${FORMAL_EVAL_STEPS}" BRANCH_MODE_STEPS="${BRANCH_MODE_STEPS}" \
        ALPHA_GRID_STEPS="${ALPHA_GRID_STEPS}" TRAIN_STATS_STEPS="${TRAIN_STATS_STEPS}" \
        EVAL_SEEDS="${EVAL_SEEDS}" SUPPORT_SHOT="${SUPPORT_SHOT}" \
        BRANCH_MODES="${BRANCH_MODES}" ALPHA_MODES="${ALPHA_MODES}" \
        ALPHA_BRANCH_MODES="${ALPHA_BRANCH_MODES}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
        MAX_EVAL_QUERY_PER_CLASS="${MAX_EVAL_QUERY_PER_CLASS}" \
        RGB_BACKBONE_LR="${RGB_BACKBONE_LR}" FREQ_BACKBONE_LR="${FREQ_BACKBONE_LR}" \
        RGB_HEAD_LR="${RGB_HEAD_LR}" FREQ_HEAD_LR="${FREQ_HEAD_LR}" WEIGHT_DECAY="${WEIGHT_DECAY}" \
        TAU="${TAU}" TAU_R="${TAU_R}" M_RF="${M_RF}" M_FF="${M_FF}" LAMBDA_FF="${LAMBDA_FF}" \
        LAMBDA_SEP_TARGET="${LAMBDA_SEP_TARGET}" LAMBDA_SEP_WARMUP_START="${LAMBDA_SEP_WARMUP_START}" \
        LAMBDA_SEP_WARMUP_END="${LAMBDA_SEP_WARMUP_END}" \
        BRANCH_DROPOUT_DUAL_PROB="${BRANCH_DROPOUT_DUAL_PROB}" \
        BRANCH_DROPOUT_RGB_PROB="${BRANCH_DROPOUT_RGB_PROB}" \
        BRANCH_DROPOUT_FREQ_PROB="${BRANCH_DROPOUT_FREQ_PROB}" \
        AUTO_COMPUTE_FREQ_STATS="${AUTO_COMPUTE_FREQ_STATS}" USE_FP16="${USE_FP16}" PRETRAINED="${PRETRAINED}" \
        bash scripts/run_ddfsd_one_class_pipeline_20pct.sh 2>&1 | tee -a "${CLASS_DRIVER_LOG}"
    CLASS_EXIT=${PIPESTATUS[0]}

    if [[ ${CLASS_EXIT} -eq 0 ]] && class_is_complete "${CLASS_NAME}"; then
        status_append "class_${CLASS_NAME}_status=COMPLETE"
    else
        status_append "class_${CLASS_NAME}_status=FAILED exit_code=${CLASS_EXIT} status_file=${CLASS_OUTPUT_PATH}/PIPELINE_STATUS.txt"
        FAILURES+=("${CLASS_NAME}")
        echo "${CLASS_NAME} failed or did not pass key-result verification; continuing to the next class." | tee -a "${CLASS_DRIVER_LOG}" >&2
    fi
done

if [[ ${VALID_CLASS_COUNT} -eq 0 ]]; then
    status_append "failure_reason=no_valid_classes_requested"
    status_append "pipeline_state=FAILED"
    status_append "pipeline_end=$(date -Iseconds)"
    exit 2
fi

# Write a pre-summary state so the all-class report can include an authoritative
# snapshot. A summary failure appends PARTIAL afterwards.
if [[ ${#FAILURES[@]} -eq 0 ]] && all_six_classes_complete; then
    status_append "pipeline_state=SUMMARY_PENDING_ALL_COMPLETE"
elif [[ ${#FAILURES[@]} -eq 0 ]]; then
    status_append "pipeline_state=SUMMARY_PENDING_SELECTION_COMPLETE"
else
    status_append "pipeline_state=PARTIAL"
fi

ALL_SUMMARY_MD="${SUMMARY_DIR}/ddfsd_20pct_all_classes_summary.md"
ALL_SUMMARY_CSV="${SUMMARY_DIR}/ddfsd_20pct_all_classes_summary.csv"
ALL_COMPLETENESS_CSV="${SUMMARY_DIR}/ddfsd_20pct_all_classes_completeness.csv"
ALL_MISSING_CSV="${SUMMARY_DIR}/ddfsd_20pct_missing_files_and_failures.csv"

echo "============================================================"
echo "Generating all-class DDFSD 20pct summary from currently available results."
echo "============================================================"
ALL_SUMMARY_HISTORY_DIR="${SUMMARY_DIR}/history/${ATTEMPT_ID}"
if [[ -e "${ALL_SUMMARY_MD}" || -e "${ALL_SUMMARY_CSV}" || -e "${ALL_COMPLETENESS_CSV}" || -e "${ALL_MISSING_CSV}" ]]; then
    mkdir -p "${ALL_SUMMARY_HISTORY_DIR}"
    for SUMMARY_ARTIFACT in "${ALL_SUMMARY_MD}" "${ALL_SUMMARY_CSV}" "${ALL_COMPLETENESS_CSV}" "${ALL_MISSING_CSV}"; do
        if [[ -e "${SUMMARY_ARTIFACT}" ]]; then
            cp -- "${SUMMARY_ARTIFACT}" "${ALL_SUMMARY_HISTORY_DIR}/$(basename "${SUMMARY_ARTIFACT}")"
        fi
    done
    status_append "all_classes_summary_previous_outputs_preserved=${ALL_SUMMARY_HISTORY_DIR}"
fi

python tools/summarize_ddfsd_20pct.py all-classes \
    --run-root "${EXPERIMENT_ROOT}" \
    2>&1 | tee -a "${GLOBAL_LOG_DIR}/summarize_all_classes_20pct.log"
SUMMARY_EXIT=${PIPESTATUS[0]}

REQUESTED_COMPLETENESS_FULL=1
for CLASS_NAME in "${REQUESTED_CLASSES[@]}"; do
    is_allowed_class "${CLASS_NAME}" || continue
    validate_completeness_csv "${ALL_COMPLETENESS_CSV}" "${CLASS_NAME}" || REQUESTED_COMPLETENESS_FULL=0
done
ALL_COMPLETENESS_FULL=1
for CLASS_NAME in "${ALL_CLASSES[@]}"; do
    validate_completeness_csv "${ALL_COMPLETENESS_CSV}" "${CLASS_NAME}" || ALL_COMPLETENESS_FULL=0
done

if [[ ${SUMMARY_EXIT} -eq 0 && -s "${ALL_SUMMARY_MD}" && -s "${ALL_SUMMARY_CSV}" && -s "${ALL_MISSING_CSV}" && ${REQUESTED_COMPLETENESS_FULL} -eq 1 ]]; then
    if [[ ${ALL_COMPLETENESS_FULL} -eq 1 ]]; then
        status_append "all_classes_summary_status=OK core_completeness=ALL_FULL"
    else
        status_append "all_classes_summary_status=OK core_completeness=REQUESTED_FULL_ALL_PARTIAL"
    fi
else
    status_append "all_classes_summary_status=FAILED exit_code=${SUMMARY_EXIT} requested_core_completeness=${REQUESTED_COMPLETENESS_FULL}"
    FAILURES+=("all_classes_summary")
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    FAILURE_LIST=$(IFS=,; printf '%s' "${FAILURES[*]}")
    status_append "failure_classes_or_stages=${FAILURE_LIST}"
    status_append "pipeline_state=PARTIAL"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "All-class pipeline finished PARTIAL: ${FAILURE_LIST}" >&2
    exit 1
fi

if all_six_classes_complete && [[ ${ALL_COMPLETENESS_FULL} -eq 1 ]]; then
    status_append "pipeline_state=COMPLETE"
else
    status_append "pipeline_state=SELECTION_COMPLETE"
fi
status_append "pipeline_end=$(date -Iseconds)"
echo "Requested DDFSD 20pct classes finished successfully."
exit 0
