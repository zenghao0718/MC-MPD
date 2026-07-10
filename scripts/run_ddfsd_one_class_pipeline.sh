#!/usr/bin/env bash
# Full DDFSD pipeline for one exclude_class: train -> checkpoint check -> formal eval
# (all checkpoints) -> branch-mode diagnosis -> fixed-alpha diagnosis -> train-log parse.
#
# Usage: EXCLUDE_CLASS=glide bash scripts/run_ddfsd_one_class_pipeline.sh
#
# On failure of the training stage, this script exits non-zero immediately (does not run
# any downstream stage) so the caller (run_ddfsd_remaining4_pipeline.sh) can record the
# failure and move on to the next class. Downstream diagnostic stages are best-effort:
# a failure in one of them is logged but does not abort the remaining stages for this class.
set -uo pipefail

EXCLUDE_CLASS=${EXCLUDE_CLASS:?EXCLUDE_CLASS is required}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
# Margin hyper-parameters passthrough (defaults match train_ddfsd_10pct.sh).
M_RF=${M_RF:-1.2}
M_FF=${M_FF:-0.6}
LOG_INTERVAL=${LOG_INTERVAL:-200}

FORMAL_EVAL_STEPS=${FORMAL_EVAL_STEPS:-"2500,5000,7500,10000,12500,15000"}
BRANCH_MODE_STEPS=${BRANCH_MODE_STEPS:-"2500,5000,7500,10000,12500,15000"}
ALPHA_GRID_STEPS=${ALPHA_GRID_STEPS:-"7500,12500,15000"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_PATH}/logs" "${OUTPUT_PATH}/csv"

STATUS_FILE="${OUTPUT_PATH}/PIPELINE_STATUS.txt"
echo "exclude_class=${EXCLUDE_CLASS} pipeline_start=$(date -Iseconds)" > "${STATUS_FILE}"

echo "############################################################"
echo "# [1/6] TRAIN exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"

export DATA_ROOT RUN_ROOT NUM_WORKERS SEED EXCLUDE_CLASS OUTPUT_PATH FREQ_STATS_PATH M_RF M_FF LOG_INTERVAL

bash scripts/train_ddfsd_10pct.sh 2>&1 | tee "${OUTPUT_PATH}/logs/train_${EXCLUDE_CLASS}_10pct.log"
TRAIN_EXIT=${PIPESTATUS[0]}

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo "train_status=FAILED exit_code=${TRAIN_EXIT}" >> "${STATUS_FILE}"
    echo "TRAIN FAILED for ${EXCLUDE_CLASS}, exit code ${TRAIN_EXIT}. See ${OUTPUT_PATH}/logs/train_${EXCLUDE_CLASS}_10pct.log" >&2
    exit "${TRAIN_EXIT}"
fi
echo "train_status=OK" >> "${STATUS_FILE}"

echo "############################################################"
echo "# [1b/6] CHECKPOINT CHECK exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
ls -lh "${OUTPUT_PATH}/ckpt"/*.pth 2>&1 | tee -a "${STATUS_FILE}" || true
for STEP in 2500 5000 7500 10000 12500 15000; do
    CKPT_PATH="${OUTPUT_PATH}/ckpt/ddfsd_step[${STEP}].pth"
    if [[ -f "${CKPT_PATH}" ]]; then
        echo "ckpt_step_${STEP}=OK" >> "${STATUS_FILE}"
    else
        echo "ckpt_step_${STEP}=MISSING" >> "${STATUS_FILE}"
    fi
done

echo "############################################################"
echo "# [2/6] FORMAL DUAL 5-SEED 10-SHOT EVAL exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
CKPT_STEPS="${FORMAL_EVAL_STEPS}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
    DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" EXCLUDE_CLASS="${EXCLUDE_CLASS}" \
    OUTPUT_PATH="${OUTPUT_PATH}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
    bash scripts/eval_ddfsd_formal_all_steps.sh 2>&1 | tee "${OUTPUT_PATH}/logs/eval_${EXCLUDE_CLASS}_formal.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "formal_eval_status=FAILED" >> "${STATUS_FILE}"
else
    echo "formal_eval_status=OK" >> "${STATUS_FILE}"
fi

echo "############################################################"
echo "# [3/6] BRANCH-MODE DIAGNOSIS (dual/rgb-only/freq-only) exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
mkdir -p "${OUTPUT_PATH}/branch_modes"
EXCLUDE_CLASS="${EXCLUDE_CLASS}" DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" \
    OUTPUT_PATH="${OUTPUT_PATH}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
    CKPT_STEPS="${BRANCH_MODE_STEPS}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
    OUT_DIR="${OUTPUT_PATH}/branch_modes" \
    bash scripts/eval_ddfsd_branch_modes.sh 2>&1 | tee "${OUTPUT_PATH}/branch_modes/eval_${EXCLUDE_CLASS}_branch_modes.log.tmp"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "branch_modes_status=FAILED" >> "${STATUS_FILE}"
else
    echo "branch_modes_status=OK" >> "${STATUS_FILE}"
fi
mv "${OUTPUT_PATH}/branch_modes/eval_${EXCLUDE_CLASS}_branch_modes.log.tmp" "${OUTPUT_PATH}/branch_modes/eval_${EXCLUDE_CLASS}_branch_modes.log" 2>/dev/null || true

echo "############################################################"
echo "# [4/6] FIXED ALPHA DIAGNOSIS (adaptive,0.0,0.25,0.5,0.75,1.0) exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
mkdir -p "${OUTPUT_PATH}/alpha_grid"
DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${OUTPUT_PATH}/alpha_grid" EXCLUDE_CLASS="${EXCLUDE_CLASS}" \
    CKPT_DIR="${OUTPUT_PATH}/ckpt" CKPT_STEPS="${ALPHA_GRID_STEPS}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
    SUPPORT_SHOT=10 EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
    ALPHA_MODES="adaptive,0.0,0.25,0.5,0.75,1.0" BRANCH_MODES="dual" \
    bash scripts/eval_ddfsd_alpha_grid.sh 2>&1 | tee "${OUTPUT_PATH}/alpha_grid/eval_${EXCLUDE_CLASS}_alpha_grid.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "alpha_grid_status=FAILED" >> "${STATUS_FILE}"
else
    echo "alpha_grid_status=OK" >> "${STATUS_FILE}"
fi

echo "############################################################"
echo "# [5/7] PARSE TRAINING-TIME ALPHA/LOSS BY CHECKPOINT exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
python tools/parse_ddfsd_train_alpha_loss.py \
    --exclude_class "${EXCLUDE_CLASS}" \
    --output_path "${OUTPUT_PATH}" \
    --ckpt_steps "2500,5000,7500,10000,12500,15000" \
    --out_csv "${OUTPUT_PATH}/csv/ddfsd_${EXCLUDE_CLASS}_train_alpha_loss_by_ckpt.csv" \
    2>&1 | tee "${OUTPUT_PATH}/logs/parse_${EXCLUDE_CLASS}_train_alpha_loss.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "train_log_parse_status=FAILED" >> "${STATUS_FILE}"
else
    echo "train_log_parse_status=OK" >> "${STATUS_FILE}"
fi

echo "############################################################"
echo "# [6/7] PARSE MARGIN DIAGNOSTICS (proto_rf_/proto_ff_ distances, violation rate) exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
python tools/parse_ddfsd_margin_diagnostics.py \
    --exclude_class "${EXCLUDE_CLASS}" \
    --output_path "${OUTPUT_PATH}" \
    --log_interval "${LOG_INTERVAL}" \
    --ckpt_steps "2500,5000,7500,10000,12500,15000" \
    --out_csv_by_interval "${OUTPUT_PATH}/csv/ddfsd_${EXCLUDE_CLASS}_margin_diagnostics_by_log_interval.csv" \
    --out_csv_by_ckpt "${OUTPUT_PATH}/csv/ddfsd_${EXCLUDE_CLASS}_margin_diagnostics_by_ckpt.csv" \
    2>&1 | tee "${OUTPUT_PATH}/logs/parse_${EXCLUDE_CLASS}_margin_diagnostics.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "margin_diagnostics_parse_status=FAILED" >> "${STATUS_FILE}"
else
    echo "margin_diagnostics_parse_status=OK" >> "${STATUS_FILE}"
fi

echo "pipeline_end=$(date -Iseconds)" >> "${STATUS_FILE}"
echo "############################################################"
echo "# [7/7] DONE with training+eval stages for exclude_class=${EXCLUDE_CLASS}"
echo "# (per-class markdown summary is generated afterwards)"
echo "############################################################"
cat "${STATUS_FILE}"
exit 0
