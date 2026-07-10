#!/usr/bin/env bash
# DDFSD 1/5 (20%) pipeline for one GenImage leave-one-out class.
#
# The pipeline is intentionally restart-safe at evaluation-stage granularity:
# completed checkpoint CSVs are never evaluated again, missing checkpoints are
# evaluated into unique attempt directories, and canonical CSVs are created only
# when absent. A partial training run is not restarted in place because
# train_ddfsd.py has no resume contract and doing so would overwrite checkpoints.
set -uo pipefail

EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_20pct/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
RUN_CONFIG=${RUN_CONFIG:-"ddfsd_20pct_steps30000"}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"${RUN_ROOT}/${RUN_CONFIG}"}
OUTPUT_PATH=${OUTPUT_PATH:-"${EXPERIMENT_ROOT}/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}

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
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
MAX_EVAL_QUERY_PER_CLASS=${MAX_EVAL_QUERY_PER_CLASS:-0}

is_allowed_class() {
    case "$1" in
        ADM|BigGAN|glide|Midjourney|SD|VQDM) return 0 ;;
        *) return 1 ;;
    esac
}

if ! is_allowed_class "${EXCLUDE_CLASS}"; then
    echo "Invalid EXCLUDE_CLASS='${EXCLUDE_CLASS}'. Expected one of: ADM, BigGAN, glide, Midjourney, SD, VQDM." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

path_is_strictly_within() {
    local parent_path=$1
    local child_path=$2
    python - "${parent_path}" "${child_path}" <<'PYEOF'
import os
import sys

parent = os.path.realpath(os.path.abspath(sys.argv[1]))
child = os.path.realpath(os.path.abspath(sys.argv[2]))
try:
    inside = os.path.commonpath((parent, child)) == parent and child != parent
except ValueError:
    inside = False
if not inside:
    print(f"Path containment check failed: child={child} parent={parent}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

if ! path_is_strictly_within "${EXPERIMENT_ROOT}" "${OUTPUT_PATH}"; then
    echo "OUTPUT_PATH must resolve inside EXPERIMENT_ROOT. Override EXPERIMENT_ROOT together with OUTPUT_PATH when relocating a run." >&2
    exit 3
fi
if ! path_is_strictly_within "${OUTPUT_PATH}" "${FREQ_STATS_PATH}"; then
    echo "FREQ_STATS_PATH must resolve inside OUTPUT_PATH." >&2
    exit 3
fi

LOGS_DIR="${OUTPUT_PATH}/logs"
CSV_DIR="${OUTPUT_PATH}/csv"
FORMAL_EVAL_DIR="${OUTPUT_PATH}/formal_eval"
BRANCH_MODE_DIR="${OUTPUT_PATH}/branch_modes"
ALPHA_GRID_DIR="${OUTPUT_PATH}/alpha_grid"
SUMMARY_DIR="${OUTPUT_PATH}/summary"
STATUS_FILE="${OUTPUT_PATH}/PIPELINE_STATUS.txt"
CONFIG_FILE="${OUTPUT_PATH}/PIPELINE_CONFIG.txt"
ATTEMPT_ID="$(date +%Y%m%d_%H%M%S)_$$"

mkdir -p \
    "${LOGS_DIR}" "${CSV_DIR}" "${SUMMARY_DIR}" \
    "${FORMAL_EVAL_DIR}/logs" "${FORMAL_EVAL_DIR}/attempts" \
    "${BRANCH_MODE_DIR}/logs" "${BRANCH_MODE_DIR}/attempts" \
    "${ALPHA_GRID_DIR}/logs" "${ALPHA_GRID_DIR}/attempts"

status_append() {
    printf '%s\n' "$*" >> "${STATUS_FILE}"
}

latest_pipeline_state() {
    local path=$1
    [[ -f "${path}" ]] || return 1
    awk -F= '/^pipeline_state=/{state=$2} END{if (state != "") print state}' "${path}"
}

validate_result_csv() {
    local csv_kind=$1
    local csv_path=$2
    local expected_steps=$3
    local expected_modes=${4:-""}
    [[ -s "${csv_path}" ]] || return 1
    python - \
        "${csv_kind}" "${csv_path}" "${EXCLUDE_CLASS}" "${expected_steps}" \
        "${EVAL_SEEDS}" "${expected_modes}" "${SUPPORT_SHOT}" <<'PYEOF'
import csv
import math
import sys

kind, path, expected_class, raw_steps, raw_seeds, raw_modes, raw_shot = sys.argv[1:8]
steps = tuple(int(item.strip()) for item in raw_steps.split(",") if item.strip())
seeds = tuple(int(item.strip()) for item in raw_seeds.split(",") if item.strip())
modes = tuple(item.strip() for item in raw_modes.split(",") if item.strip())
support_shot = int(raw_shot)


def fail(message):
    print(f"CSV validation failed [{kind}] {path}: {message}", file=sys.stderr)
    raise SystemExit(1)


def finite(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{field} is not numeric: {value!r}")
    if not math.isfinite(number):
        fail(f"{field} is not finite: {value!r}")
    return number


def integer(value, field):
    number = finite(value, field)
    if not number.is_integer():
        fail(f"{field} is not an integer: {value!r}")
    return int(number)


def metric(value, field):
    number = finite(value, field)
    if number < 0.0 or number > 1.0:
        fail(f"{field} is outside [0, 1]: {number}")
    return number


def alpha_mode(value):
    token = str(value).strip().lower()
    if token == "adaptive":
        return "adaptive"
    number = finite(token, "alpha_mode")
    return f"{number:g}"


try:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        rows = list(reader)
except (OSError, csv.Error) as exc:
    fail(f"cannot read CSV: {exc}")

if not rows:
    fail("no data rows")

common = {"exclude_class", "ckpt_step"}
required_by_kind = {
    "formal_per_seed": common | {"seed", "support_shot", "split", "acc", "ap"},
    "formal_summary": common | {"support_shot", "acc_mean", "acc_std", "ap_mean", "ap_std", "eval_seeds"},
    "branch_per_seed": common | {"branch_mode", "seed", "support_shot", "split", "acc", "ap"},
    "branch_summary": common | {"branch_mode", "support_shot", "acc_mean", "acc_std", "ap_mean", "ap_std", "eval_seeds"},
    "alpha_per_seed": common | {"branch_mode", "alpha_mode", "seed", "support_shot", "split", "acc", "ap", "adaptive_alpha_mean"},
    "alpha_summary": common | {"branch_mode", "alpha_mode", "support_shot", "acc_mean", "acc_std", "ap_mean", "ap_std", "eval_seeds", "adaptive_alpha_mean"},
    "train": common | {"matched_train_step", "step_delta", "alpha_mean", "alpha_min", "alpha_max", "loss_rf", "loss_ff", "loss_sep", "lambda_sep_current"},
}
if kind not in required_by_kind:
    fail(f"unknown CSV kind: {kind}")
missing_headers = sorted(required_by_kind[kind] - headers)
if missing_headers:
    fail(f"missing columns: {missing_headers}")

normalized_modes = tuple(alpha_mode(mode) for mode in modes) if kind.startswith("alpha_") else modes
actual_keys = []
for row_index, row in enumerate(rows, start=2):
    if str(row.get("exclude_class", "")).strip() != expected_class:
        fail(f"row {row_index} exclude_class mismatch: {row.get('exclude_class')!r}")
    step = integer(row.get("ckpt_step"), f"row {row_index} ckpt_step")
    if step not in steps:
        fail(f"row {row_index} unexpected ckpt_step={step}")

    if kind == "train":
        matched = integer(row.get("matched_train_step"), f"row {row_index} matched_train_step")
        delta = integer(row.get("step_delta"), f"row {row_index} step_delta")
        if matched != step or delta != 0:
            fail(f"row {row_index} checkpoint/matched step mismatch: {step}/{matched}, delta={delta}")
        for field in ("alpha_mean", "alpha_min", "alpha_max", "loss_rf", "loss_ff", "loss_sep", "lambda_sep_current"):
            finite(row.get(field), f"row {row_index} {field}")
        actual_keys.append((step,))
        continue

    if integer(row.get("support_shot"), f"row {row_index} support_shot") != support_shot:
        fail(f"row {row_index} support_shot is not {support_shot}")
    if kind.endswith("per_seed") and str(row.get("split", "")).strip() != "val":
        fail(f"row {row_index} split is not val")

    if kind.endswith("per_seed"):
        metric(row.get("acc"), f"row {row_index} acc")
        metric(row.get("ap"), f"row {row_index} ap")
        seed = integer(row.get("seed"), f"row {row_index} seed")
        if seed not in seeds:
            fail(f"row {row_index} unexpected seed={seed}")
    else:
        metric(row.get("acc_mean"), f"row {row_index} acc_mean")
        metric(row.get("ap_mean"), f"row {row_index} ap_mean")
        for field in ("acc_std", "ap_std"):
            if finite(row.get(field), f"row {row_index} {field}") < 0:
                fail(f"row {row_index} {field} is negative")
        row_seeds = {
            integer(item.strip(), f"row {row_index} eval_seeds")
            for item in str(row.get("eval_seeds", "")).split(",")
            if item.strip()
        }
        if row_seeds != set(seeds):
            fail(f"row {row_index} eval_seeds mismatch: {sorted(row_seeds)}")

    if kind.startswith("formal_"):
        actual_keys.append((step, seed) if kind.endswith("per_seed") else (step,))
    elif kind.startswith("branch_"):
        mode = str(row.get("branch_mode", "")).strip()
        if mode not in modes:
            fail(f"row {row_index} unexpected branch_mode={mode!r}")
        actual_keys.append((step, mode, seed) if kind.endswith("per_seed") else (step, mode))
    elif kind.startswith("alpha_"):
        branch_mode = str(row.get("branch_mode", "")).strip()
        if branch_mode != "dual":
            fail(f"row {row_index} alpha branch_mode is not dual: {branch_mode!r}")
        mode = alpha_mode(row.get("alpha_mode"))
        if mode not in normalized_modes:
            fail(f"row {row_index} unexpected alpha_mode={mode!r}")
        metric(row.get("adaptive_alpha_mean"), f"row {row_index} adaptive_alpha_mean")
        actual_keys.append((step, mode, seed) if kind.endswith("per_seed") else (step, mode))

if kind == "formal_per_seed":
    expected_keys = {(step, seed) for step in steps for seed in seeds}
elif kind == "formal_summary" or kind == "train":
    expected_keys = {(step,) for step in steps}
elif kind == "branch_per_seed":
    expected_keys = {(step, mode, seed) for step in steps for mode in modes for seed in seeds}
elif kind == "branch_summary":
    expected_keys = {(step, mode) for step in steps for mode in modes}
elif kind == "alpha_per_seed":
    expected_keys = {(step, mode, seed) for step in steps for mode in normalized_modes for seed in seeds}
else:
    expected_keys = {(step, mode) for step in steps for mode in normalized_modes}

if len(actual_keys) != len(set(actual_keys)):
    fail("duplicate grid keys")
if set(actual_keys) != expected_keys:
    missing = sorted(expected_keys - set(actual_keys), key=str)
    extra = sorted(set(actual_keys) - expected_keys, key=str)
    fail(f"grid mismatch: missing={missing}, extra={extra}")
PYEOF
}

validate_completeness_csv() {
    local csv_path=$1
    [[ -s "${csv_path}" ]] || return 1
    python - "${csv_path}" "${EXCLUDE_CLASS}" <<'PYEOF'
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
        print(f"Completeness class mismatch in {path}", file=sys.stderr)
        sys.exit(1)
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

atomic_install_csv() {
    local source_path=$1
    local target_path=$2
    local history_dir=$3
    local temporary_path="${target_path}.tmp.${ATTEMPT_ID}"
    mkdir -p "$(dirname "${target_path}")"
    if [[ -e "${target_path}" ]]; then
        mkdir -p "${history_dir}"
        cp -- "${target_path}" "${history_dir}/$(basename "${target_path}")" || return 1
    fi
    cp -- "${source_path}" "${temporary_path}" || return 1
    mv -f -- "${temporary_path}" "${target_path}"
}

merge_step_csvs() {
    local target_path=$1
    local step_root=$2
    local source_name=$3
    local expected_steps=$4
    python - "${target_path}" "${step_root}" "${source_name}" "${expected_steps}" <<'PYEOF'
import csv
import os
import sys

target, step_root, source_name, raw_steps = sys.argv[1:5]
steps = [item.strip() for item in raw_steps.split(",") if item.strip()]

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)

if os.path.exists(target):
    print(f"Aggregate attempt target already exists: {target}", file=sys.stderr)
    sys.exit(3)

fieldnames = None
all_rows = []
missing = []
for step in steps:
    source = os.path.join(step_root, f"step_{step}", source_name)
    if not os.path.isfile(source):
        missing.append(source)
        continue
    try:
        current_fields, rows = read_csv(source)
    except (OSError, csv.Error) as exc:
        print(f"Cannot read {source}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not current_fields:
        print(f"CSV has no header: {source}", file=sys.stderr)
        sys.exit(2)
    if fieldnames is None:
        fieldnames = current_fields
    elif current_fields != fieldnames:
        print(f"CSV header mismatch: {source}", file=sys.stderr)
        sys.exit(2)
    all_rows.extend(rows)

if missing or fieldnames is None:
    for path in missing:
        print(f"Missing step CSV: {path}", file=sys.stderr)
    sys.exit(2)

os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
temporary = f"{target}.tmp.{os.getpid()}"
try:
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
except FileExistsError:
    print(f"Temporary target already exists; not overwriting: {temporary}", file=sys.stderr)
    sys.exit(3)
with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
try:
    os.link(temporary, target)
except FileExistsError:
    print(f"Target appeared concurrently; not overwriting: {target}", file=sys.stderr)
    os.unlink(temporary)
    sys.exit(3)
os.unlink(temporary)
print(f"Merged {len(all_rows)} rows -> {target}")
PYEOF
}

all_checkpoints_exist() {
    local raw_step step
    local -a steps
    IFS=',' read -r -a steps <<< "${EXPECTED_CKPT_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        [[ -f "${OUTPUT_PATH}/ckpt/ddfsd_step[${step}].pth" ]] || return 1
    done
    return 0
}

formal_aggregate_complete() {
    validate_result_csv "formal_per_seed" \
        "${FORMAL_EVAL_DIR}/ddfsd_eval_per_seed_all_steps.csv" "${FORMAL_EVAL_STEPS}" &&
        validate_result_csv "formal_summary" \
            "${FORMAL_EVAL_DIR}/ddfsd_eval_summary_all_steps.csv" "${FORMAL_EVAL_STEPS}"
}

branch_aggregate_complete() {
    validate_result_csv "branch_per_seed" \
        "${BRANCH_MODE_DIR}/ddfsd_${EXCLUDE_CLASS}_branch_modes_per_seed.csv" \
        "${BRANCH_MODE_STEPS}" "${BRANCH_MODES}" &&
        validate_result_csv "branch_summary" \
            "${BRANCH_MODE_DIR}/ddfsd_${EXCLUDE_CLASS}_branch_modes_summary.csv" \
            "${BRANCH_MODE_STEPS}" "${BRANCH_MODES}"
}

alpha_aggregate_complete() {
    validate_result_csv "alpha_per_seed" \
        "${ALPHA_GRID_DIR}/ddfsd_${EXCLUDE_CLASS}_alpha_grid_per_seed.csv" \
        "${ALPHA_GRID_STEPS}" "${ALPHA_MODES}" &&
        validate_result_csv "alpha_summary" \
            "${ALPHA_GRID_DIR}/ddfsd_${EXCLUDE_CLASS}_alpha_grid_summary.csv" \
            "${ALPHA_GRID_STEPS}" "${ALPHA_MODES}"
}

formal_steps_complete() {
    local raw_step step step_dir
    local -a steps
    IFS=',' read -r -a steps <<< "${FORMAL_EVAL_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${FORMAL_EVAL_DIR}/step_${step}"
        validate_result_csv "formal_per_seed" "${step_dir}/ddfsd_eval_per_seed.csv" "${step}" || return 1
        validate_result_csv "formal_summary" "${step_dir}/ddfsd_eval_summary.csv" "${step}" || return 1
    done
}

branch_steps_complete() {
    local raw_step step step_dir
    local per_seed_name="ddfsd_${EXCLUDE_CLASS}_branch_modes_per_seed.csv"
    local summary_name="ddfsd_${EXCLUDE_CLASS}_branch_modes_summary.csv"
    local -a steps
    IFS=',' read -r -a steps <<< "${BRANCH_MODE_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${BRANCH_MODE_DIR}/step_${step}"
        validate_result_csv "branch_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${BRANCH_MODES}" || return 1
        validate_result_csv "branch_summary" "${step_dir}/${summary_name}" "${step}" "${BRANCH_MODES}" || return 1
    done
}

alpha_steps_complete() {
    local raw_step step step_dir
    local per_seed_name="ddfsd_${EXCLUDE_CLASS}_alpha_grid_per_seed.csv"
    local summary_name="ddfsd_${EXCLUDE_CLASS}_alpha_grid_summary.csv"
    local -a steps
    IFS=',' read -r -a steps <<< "${ALPHA_GRID_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${ALPHA_GRID_DIR}/step_${step}"
        validate_result_csv "alpha_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${ALPHA_MODES}" || return 1
        validate_result_csv "alpha_summary" "${step_dir}/${summary_name}" "${step}" "${ALPHA_MODES}" || return 1
    done
}

formal_stage_complete() {
    formal_steps_complete && formal_aggregate_complete
}

branch_stage_complete() {
    branch_steps_complete && branch_aggregate_complete
}

alpha_stage_complete() {
    alpha_steps_complete && alpha_aggregate_complete
}

class_outputs_complete() {
    [[ "$(latest_pipeline_state "${STATUS_FILE}" 2>/dev/null || true)" == "COMPLETE" ]] || return 1
    [[ -s "${CONFIG_FILE}" && -f "${FREQ_STATS_PATH}" ]] || return 1
    all_checkpoints_exist || return 1
    formal_stage_complete || return 1
    branch_stage_complete || return 1
    alpha_stage_complete || return 1
    validate_result_csv "train" \
        "${CSV_DIR}/ddfsd_${EXCLUDE_CLASS}_train_alpha_loss_by_ckpt.csv" \
        "${TRAIN_STATS_STEPS}" || return 1
    [[ -s "${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_summary.md" ]] || return 1
    [[ -s "${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_summary.csv" ]] || return 1
    validate_completeness_csv \
        "${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_completeness.csv" || return 1
    return 0
}

write_config_snapshot() {
    local candidate="${OUTPUT_PATH}/.PIPELINE_CONFIG.${ATTEMPT_ID}.tmp"
    local alternate="${OUTPUT_PATH}/PIPELINE_CONFIG_${ATTEMPT_ID}.txt"
    {
        printf 'EXCLUDE_CLASS=%s\n' "${EXCLUDE_CLASS}"
        printf 'DATA_ROOT=%s\n' "${DATA_ROOT}"
        printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
        printf 'RUN_CONFIG=%s\n' "${RUN_CONFIG}"
        printf 'OUTPUT_PATH=%s\n' "${OUTPUT_PATH}"
        printf 'FREQ_STATS_PATH=%s\n' "${FREQ_STATS_PATH}"
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
status_append "exclude_class=${EXCLUDE_CLASS}"
if ! write_config_snapshot; then
    status_append "failure_reason=pipeline_config_mismatch_refusing_to_mix_runs"
    status_append "pipeline_state=FAILED"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "PIPELINE_CONFIG.txt differs from this invocation; preserved a timestamped candidate and stopped." >&2
    exit 22
fi

if class_outputs_complete; then
    status_append "pipeline_skip_reason=verified_complete_config_and_outputs"
    status_append "pipeline_state=COMPLETE"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "DDFSD 20pct pipeline already complete for ${EXCLUDE_CLASS}; verified config, status, and strict CSV grids, skipping."
    exit 0
fi

export DATA_ROOT RUN_ROOT RUN_CONFIG NUM_WORKERS SEED EXCLUDE_CLASS OUTPUT_PATH FREQ_STATS_PATH
export BATCH_SIZE TOTAL_STEPS SAVE_INTERVAL EVAL_INTERVAL LOG_INTERVAL LR_STEP LR_GAMMA
export RGB_BACKBONE_LR FREQ_BACKBONE_LR RGB_HEAD_LR FREQ_HEAD_LR WEIGHT_DECAY
export TAU TAU_R M_RF M_FF LAMBDA_FF LAMBDA_SEP_TARGET LAMBDA_SEP_WARMUP_START LAMBDA_SEP_WARMUP_END
export BRANCH_DROPOUT_DUAL_PROB BRANCH_DROPOUT_RGB_PROB BRANCH_DROPOUT_FREQ_PROB
export AUTO_COMPUTE_FREQ_STATS USE_FP16 PRETRAINED

echo "############################################################"
echo "# [1/7] TRAIN / RESUME CHECK exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"

IFS=',' read -r -a EXPECTED_STEPS_ARRAY <<< "${EXPECTED_CKPT_STEPS}"
EXISTING_EXPECTED=0
MISSING_EXPECTED=()
for RAW_STEP in "${EXPECTED_STEPS_ARRAY[@]}"; do
    STEP="${RAW_STEP//[[:space:]]/}"
    [[ -n "${STEP}" ]] || continue
    if [[ -f "${OUTPUT_PATH}/ckpt/ddfsd_step[${STEP}].pth" ]]; then
        EXISTING_EXPECTED=$((EXISTING_EXPECTED + 1))
    else
        MISSING_EXPECTED+=("${STEP}")
    fi
done

if [[ ${#MISSING_EXPECTED[@]} -eq 0 ]]; then
    echo "All expected checkpoints already exist; training is not rerun."
    status_append "train_status=SKIPPED_CHECKPOINTS_COMPLETE"
elif [[ ${EXISTING_EXPECTED} -gt 0 ]] || compgen -G "${OUTPUT_PATH}/ckpt/*.pth" >/dev/null; then
    status_append "train_status=FAILED_PARTIAL_CHECKPOINTS missing_steps=${MISSING_EXPECTED[*]}"
    status_append "failure_reason=partial_training_output_preserved_no_in_place_resume"
    status_append "pipeline_state=FAILED"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "Partial checkpoints exist under ${OUTPUT_PATH}/ckpt." >&2
    echo "train_ddfsd.py has no safe resume contract; refusing to overwrite them." >&2
    exit 20
else
    TRAIN_LOG="${LOGS_DIR}/train_${EXCLUDE_CLASS}_20pct.log"
    bash scripts/train_ddfsd_20pct.sh 2>&1 | tee -a "${TRAIN_LOG}"
    TRAIN_EXIT=${PIPESTATUS[0]}
    if [[ ${TRAIN_EXIT} -ne 0 ]]; then
        status_append "train_status=FAILED exit_code=${TRAIN_EXIT} log=${TRAIN_LOG}"
        status_append "failure_reason=training_command_failed"
        status_append "pipeline_state=FAILED"
        status_append "pipeline_end=$(date -Iseconds)"
        echo "Training failed for ${EXCLUDE_CLASS}; downstream stages were not started." >&2
        exit "${TRAIN_EXIT}"
    fi
    status_append "train_status=OK log=${TRAIN_LOG}"
fi

echo "############################################################"
echo "# [2/7] CHECKPOINT CHECK exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"

MISSING_EXPECTED=()
for RAW_STEP in "${EXPECTED_STEPS_ARRAY[@]}"; do
    STEP="${RAW_STEP//[[:space:]]/}"
    [[ -n "${STEP}" ]] || continue
    CKPT_PATH="${OUTPUT_PATH}/ckpt/ddfsd_step[${STEP}].pth"
    if [[ -f "${CKPT_PATH}" ]]; then
        status_append "ckpt_step_${STEP}=OK path=${CKPT_PATH}"
    else
        MISSING_EXPECTED+=("${STEP}")
        status_append "ckpt_step_${STEP}=MISSING path=${CKPT_PATH}"
    fi
done

if [[ ${#MISSING_EXPECTED[@]} -gt 0 ]] || [[ ! -f "${FREQ_STATS_PATH}" ]]; then
    status_append "checkpoint_status=FAILED missing_steps=${MISSING_EXPECTED[*]} freq_stats_exists=$([[ -f "${FREQ_STATS_PATH}" ]] && echo yes || echo no)"
    status_append "failure_reason=required_training_artifacts_missing"
    status_append "pipeline_state=FAILED"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "Required checkpoints or freq_stats.pt are missing; downstream stages were not started." >&2
    exit 21
fi
status_append "checkpoint_status=OK"

run_formal_stage() {
    if formal_stage_complete; then
        echo "Formal evaluation step and aggregate CSVs are complete; skipping the stage."
        return 0
    fi

    local stage_failed=0
    local raw_step step step_dir attempt_dir attempt_log command_exit source_dir history_dir
    local aggregate_attempt_dir aggregate_history_dir candidate_per_seed candidate_summary
    local -a steps
    IFS=',' read -r -a steps <<< "${FORMAL_EVAL_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${FORMAL_EVAL_DIR}/step_${step}"
        if validate_result_csv "formal_per_seed" "${step_dir}/ddfsd_eval_per_seed.csv" "${step}" &&
            validate_result_csv "formal_summary" "${step_dir}/ddfsd_eval_summary.csv" "${step}"; then
            echo "formal step ${step}: existing complete CSVs kept"
            continue
        fi

        attempt_dir="${FORMAL_EVAL_DIR}/attempts/step_${step}_${ATTEMPT_ID}"
        attempt_log="${FORMAL_EVAL_DIR}/logs/eval_${EXCLUDE_CLASS}_step_${step}.log"
        mkdir -p "${attempt_dir}"
        CKPT_STEPS="${step}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
            DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" EXCLUDE_CLASS="${EXCLUDE_CLASS}" \
            OUTPUT_PATH="${OUTPUT_PATH}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
            FORMAL_EVAL_DIR="${attempt_dir}" \
            bash scripts/eval_ddfsd_formal_all_steps.sh 2>&1 | tee -a "${attempt_log}"
        command_exit=${PIPESTATUS[0]}
        source_dir="${attempt_dir}/step_${step}"
        history_dir="${FORMAL_EVAL_DIR}/history/${ATTEMPT_ID}/step_${step}"
        if [[ ${command_exit} -eq 0 ]] &&
            validate_result_csv "formal_per_seed" "${source_dir}/ddfsd_eval_per_seed.csv" "${step}" &&
            validate_result_csv "formal_summary" "${source_dir}/ddfsd_eval_summary.csv" "${step}"; then
            atomic_install_csv "${source_dir}/ddfsd_eval_per_seed.csv" \
                "${step_dir}/ddfsd_eval_per_seed.csv" "${history_dir}" || stage_failed=1
            atomic_install_csv "${source_dir}/ddfsd_eval_summary.csv" \
                "${step_dir}/ddfsd_eval_summary.csv" "${history_dir}" || stage_failed=1
            status_append "formal_step_${step}_refreshed_from=${attempt_dir} history=${history_dir}"
        else
            stage_failed=1
        fi
        if ! validate_result_csv "formal_per_seed" "${step_dir}/ddfsd_eval_per_seed.csv" "${step}" ||
            ! validate_result_csv "formal_summary" "${step_dir}/ddfsd_eval_summary.csv" "${step}"; then
            echo "formal step ${step}: FAILED (exit=${command_exit})" >&2
            stage_failed=1
        fi
    done

    if [[ ${stage_failed} -eq 0 ]]; then
        aggregate_attempt_dir="${FORMAL_EVAL_DIR}/attempts/aggregate_${ATTEMPT_ID}"
        aggregate_history_dir="${FORMAL_EVAL_DIR}/history/${ATTEMPT_ID}/aggregate"
        candidate_per_seed="${aggregate_attempt_dir}/ddfsd_eval_per_seed_all_steps.csv"
        candidate_summary="${aggregate_attempt_dir}/ddfsd_eval_summary_all_steps.csv"
        mkdir -p "${aggregate_attempt_dir}"
        merge_step_csvs "${candidate_per_seed}" \
            "${FORMAL_EVAL_DIR}" "ddfsd_eval_per_seed.csv" "${FORMAL_EVAL_STEPS}" || stage_failed=1
        merge_step_csvs "${candidate_summary}" \
            "${FORMAL_EVAL_DIR}" "ddfsd_eval_summary.csv" "${FORMAL_EVAL_STEPS}" || stage_failed=1
        if [[ ${stage_failed} -eq 0 ]] &&
            validate_result_csv "formal_per_seed" "${candidate_per_seed}" "${FORMAL_EVAL_STEPS}" &&
            validate_result_csv "formal_summary" "${candidate_summary}" "${FORMAL_EVAL_STEPS}"; then
            atomic_install_csv "${candidate_per_seed}" \
                "${FORMAL_EVAL_DIR}/ddfsd_eval_per_seed_all_steps.csv" "${aggregate_history_dir}" || stage_failed=1
            atomic_install_csv "${candidate_summary}" \
                "${FORMAL_EVAL_DIR}/ddfsd_eval_summary_all_steps.csv" "${aggregate_history_dir}" || stage_failed=1
            status_append "formal_aggregate_refreshed_from=${aggregate_attempt_dir} history=${aggregate_history_dir}"
        else
            stage_failed=1
        fi
    fi
    [[ ${stage_failed} -eq 0 ]] && formal_stage_complete
}

run_branch_stage() {
    if branch_stage_complete; then
        echo "Branch-mode step and aggregate CSVs are complete; skipping the stage."
        return 0
    fi

    local stage_failed=0
    local raw_step step step_dir attempt_dir attempt_log command_exit history_dir
    local aggregate_attempt_dir aggregate_history_dir candidate_per_seed candidate_summary
    local per_seed_name="ddfsd_${EXCLUDE_CLASS}_branch_modes_per_seed.csv"
    local summary_name="ddfsd_${EXCLUDE_CLASS}_branch_modes_summary.csv"
    local -a steps
    IFS=',' read -r -a steps <<< "${BRANCH_MODE_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${BRANCH_MODE_DIR}/step_${step}"
        if validate_result_csv "branch_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${BRANCH_MODES}" &&
            validate_result_csv "branch_summary" "${step_dir}/${summary_name}" "${step}" "${BRANCH_MODES}"; then
            echo "branch-mode step ${step}: existing complete CSVs kept"
            continue
        fi

        attempt_dir="${BRANCH_MODE_DIR}/attempts/step_${step}_${ATTEMPT_ID}"
        attempt_log="${BRANCH_MODE_DIR}/logs/eval_${EXCLUDE_CLASS}_step_${step}.log"
        mkdir -p "${attempt_dir}"
        EXCLUDE_CLASS="${EXCLUDE_CLASS}" DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" \
            OUTPUT_PATH="${OUTPUT_PATH}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
            CKPT_STEPS="${step}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
            NUM_SUPPORT_TEST="${SUPPORT_SHOT}" OUT_DIR="${attempt_dir}" \
            bash scripts/eval_ddfsd_branch_modes.sh 2>&1 | tee -a "${attempt_log}"
        command_exit=${PIPESTATUS[0]}
        history_dir="${BRANCH_MODE_DIR}/history/${ATTEMPT_ID}/step_${step}"
        if [[ ${command_exit} -eq 0 ]] &&
            validate_result_csv "branch_per_seed" "${attempt_dir}/${per_seed_name}" "${step}" "${BRANCH_MODES}" &&
            validate_result_csv "branch_summary" "${attempt_dir}/${summary_name}" "${step}" "${BRANCH_MODES}"; then
            atomic_install_csv "${attempt_dir}/${per_seed_name}" \
                "${step_dir}/${per_seed_name}" "${history_dir}" || stage_failed=1
            atomic_install_csv "${attempt_dir}/${summary_name}" \
                "${step_dir}/${summary_name}" "${history_dir}" || stage_failed=1
            status_append "branch_step_${step}_refreshed_from=${attempt_dir} history=${history_dir}"
        else
            stage_failed=1
        fi
        if ! validate_result_csv "branch_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${BRANCH_MODES}" ||
            ! validate_result_csv "branch_summary" "${step_dir}/${summary_name}" "${step}" "${BRANCH_MODES}"; then
            echo "branch-mode step ${step}: FAILED (exit=${command_exit})" >&2
            stage_failed=1
        fi
    done

    if [[ ${stage_failed} -eq 0 ]]; then
        aggregate_attempt_dir="${BRANCH_MODE_DIR}/attempts/aggregate_${ATTEMPT_ID}"
        aggregate_history_dir="${BRANCH_MODE_DIR}/history/${ATTEMPT_ID}/aggregate"
        candidate_per_seed="${aggregate_attempt_dir}/${per_seed_name}"
        candidate_summary="${aggregate_attempt_dir}/${summary_name}"
        mkdir -p "${aggregate_attempt_dir}"
        merge_step_csvs "${candidate_per_seed}" \
            "${BRANCH_MODE_DIR}" "${per_seed_name}" "${BRANCH_MODE_STEPS}" || stage_failed=1
        merge_step_csvs "${candidate_summary}" \
            "${BRANCH_MODE_DIR}" "${summary_name}" "${BRANCH_MODE_STEPS}" || stage_failed=1
        if [[ ${stage_failed} -eq 0 ]] &&
            validate_result_csv "branch_per_seed" "${candidate_per_seed}" "${BRANCH_MODE_STEPS}" "${BRANCH_MODES}" &&
            validate_result_csv "branch_summary" "${candidate_summary}" "${BRANCH_MODE_STEPS}" "${BRANCH_MODES}"; then
            atomic_install_csv "${candidate_per_seed}" \
                "${BRANCH_MODE_DIR}/${per_seed_name}" "${aggregate_history_dir}" || stage_failed=1
            atomic_install_csv "${candidate_summary}" \
                "${BRANCH_MODE_DIR}/${summary_name}" "${aggregate_history_dir}" || stage_failed=1
            status_append "branch_aggregate_refreshed_from=${aggregate_attempt_dir} history=${aggregate_history_dir}"
        else
            stage_failed=1
        fi
    fi
    [[ ${stage_failed} -eq 0 ]] && branch_stage_complete
}

run_alpha_stage() {
    if alpha_stage_complete; then
        echo "Fixed-alpha step and aggregate CSVs are complete; skipping the stage."
        return 0
    fi

    local stage_failed=0
    local raw_step step step_dir attempt_dir attempt_log command_exit history_dir
    local aggregate_attempt_dir aggregate_history_dir candidate_per_seed candidate_summary
    local per_seed_name="ddfsd_${EXCLUDE_CLASS}_alpha_grid_per_seed.csv"
    local summary_name="ddfsd_${EXCLUDE_CLASS}_alpha_grid_summary.csv"
    local -a steps
    IFS=',' read -r -a steps <<< "${ALPHA_GRID_STEPS}"
    for raw_step in "${steps[@]}"; do
        step="${raw_step//[[:space:]]/}"
        [[ -n "${step}" ]] || continue
        step_dir="${ALPHA_GRID_DIR}/step_${step}"
        if validate_result_csv "alpha_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${ALPHA_MODES}" &&
            validate_result_csv "alpha_summary" "${step_dir}/${summary_name}" "${step}" "${ALPHA_MODES}"; then
            echo "fixed-alpha step ${step}: existing complete CSVs kept"
            continue
        fi

        attempt_dir="${ALPHA_GRID_DIR}/attempts/step_${step}_${ATTEMPT_ID}"
        attempt_log="${ALPHA_GRID_DIR}/logs/eval_${EXCLUDE_CLASS}_step_${step}.log"
        mkdir -p "${attempt_dir}"
        DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${attempt_dir}" EXCLUDE_CLASS="${EXCLUDE_CLASS}" \
            CKPT_DIR="${OUTPUT_PATH}/ckpt" CKPT_STEPS="${step}" FREQ_STATS_PATH="${FREQ_STATS_PATH}" \
            SUPPORT_SHOT="${SUPPORT_SHOT}" EVAL_SEEDS="${EVAL_SEEDS}" NUM_WORKERS="${NUM_WORKERS}" \
            EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" ALPHA_MODES="${ALPHA_MODES}" \
            BRANCH_MODES="${ALPHA_BRANCH_MODES}" MAX_EVAL_QUERY_PER_CLASS="${MAX_EVAL_QUERY_PER_CLASS}" \
            bash scripts/eval_ddfsd_alpha_grid.sh 2>&1 | tee -a "${attempt_log}"
        command_exit=${PIPESTATUS[0]}
        history_dir="${ALPHA_GRID_DIR}/history/${ATTEMPT_ID}/step_${step}"
        if [[ ${command_exit} -eq 0 ]] &&
            validate_result_csv "alpha_per_seed" "${attempt_dir}/${per_seed_name}" "${step}" "${ALPHA_MODES}" &&
            validate_result_csv "alpha_summary" "${attempt_dir}/${summary_name}" "${step}" "${ALPHA_MODES}"; then
            atomic_install_csv "${attempt_dir}/${per_seed_name}" \
                "${step_dir}/${per_seed_name}" "${history_dir}" || stage_failed=1
            atomic_install_csv "${attempt_dir}/${summary_name}" \
                "${step_dir}/${summary_name}" "${history_dir}" || stage_failed=1
            status_append "alpha_step_${step}_refreshed_from=${attempt_dir} history=${history_dir}"
        else
            stage_failed=1
        fi
        if ! validate_result_csv "alpha_per_seed" "${step_dir}/${per_seed_name}" "${step}" "${ALPHA_MODES}" ||
            ! validate_result_csv "alpha_summary" "${step_dir}/${summary_name}" "${step}" "${ALPHA_MODES}"; then
            echo "fixed-alpha step ${step}: FAILED (exit=${command_exit})" >&2
            stage_failed=1
        fi
    done

    if [[ ${stage_failed} -eq 0 ]]; then
        aggregate_attempt_dir="${ALPHA_GRID_DIR}/attempts/aggregate_${ATTEMPT_ID}"
        aggregate_history_dir="${ALPHA_GRID_DIR}/history/${ATTEMPT_ID}/aggregate"
        candidate_per_seed="${aggregate_attempt_dir}/${per_seed_name}"
        candidate_summary="${aggregate_attempt_dir}/${summary_name}"
        mkdir -p "${aggregate_attempt_dir}"
        merge_step_csvs "${candidate_per_seed}" \
            "${ALPHA_GRID_DIR}" "${per_seed_name}" "${ALPHA_GRID_STEPS}" || stage_failed=1
        merge_step_csvs "${candidate_summary}" \
            "${ALPHA_GRID_DIR}" "${summary_name}" "${ALPHA_GRID_STEPS}" || stage_failed=1
        if [[ ${stage_failed} -eq 0 ]] &&
            validate_result_csv "alpha_per_seed" "${candidate_per_seed}" "${ALPHA_GRID_STEPS}" "${ALPHA_MODES}" &&
            validate_result_csv "alpha_summary" "${candidate_summary}" "${ALPHA_GRID_STEPS}" "${ALPHA_MODES}"; then
            atomic_install_csv "${candidate_per_seed}" \
                "${ALPHA_GRID_DIR}/${per_seed_name}" "${aggregate_history_dir}" || stage_failed=1
            atomic_install_csv "${candidate_summary}" \
                "${ALPHA_GRID_DIR}/${summary_name}" "${aggregate_history_dir}" || stage_failed=1
            status_append "alpha_aggregate_refreshed_from=${aggregate_attempt_dir} history=${aggregate_history_dir}"
        else
            stage_failed=1
        fi
    fi
    [[ ${stage_failed} -eq 0 ]] && alpha_stage_complete
}

FAILURES=()

echo "############################################################"
echo "# [3/7] FORMAL DUAL 5-SEED 10-SHOT EVAL exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
if run_formal_stage; then
    status_append "formal_eval_status=OK"
else
    status_append "formal_eval_status=FAILED"
    FAILURES+=("formal_eval")
fi

echo "############################################################"
echo "# [4/7] BRANCH MODES (dual/rgb-only/freq-only) exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
if run_branch_stage; then
    status_append "branch_modes_status=OK"
else
    status_append "branch_modes_status=FAILED"
    FAILURES+=("branch_modes")
fi

echo "############################################################"
echo "# [5/7] FIXED ALPHA (adaptive,0.0,0.25,0.5,0.75,1.0) exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
if run_alpha_stage; then
    status_append "alpha_grid_status=OK"
else
    status_append "alpha_grid_status=FAILED"
    FAILURES+=("alpha_grid")
fi

echo "############################################################"
echo "# [6/7] TRAINING-TIME ALPHA/LOSS STATS exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
TRAIN_STATS_CSV="${CSV_DIR}/ddfsd_${EXCLUDE_CLASS}_train_alpha_loss_by_ckpt.csv"
if validate_result_csv "train" "${TRAIN_STATS_CSV}" "${TRAIN_STATS_STEPS}"; then
    echo "Training-statistics CSV already complete; keeping it."
    status_append "train_log_parse_status=OK_SKIPPED_EXISTING"
else
    TRAIN_STATS_ATTEMPT_DIR="${CSV_DIR}/attempts/${ATTEMPT_ID}"
    TRAIN_STATS_HISTORY_DIR="${CSV_DIR}/history/${ATTEMPT_ID}"
    TRAIN_STATS_ATTEMPT="${TRAIN_STATS_ATTEMPT_DIR}/ddfsd_${EXCLUDE_CLASS}_train_alpha_loss_by_ckpt.csv"
    mkdir -p "${TRAIN_STATS_ATTEMPT_DIR}"
    python tools/parse_ddfsd_train_alpha_loss.py \
        --exclude_class "${EXCLUDE_CLASS}" \
        --output_path "${OUTPUT_PATH}" \
        --ckpt_steps "${TRAIN_STATS_STEPS}" \
        --max_step_delta 0 \
        --out_csv "${TRAIN_STATS_ATTEMPT}" \
        2>&1 | tee -a "${LOGS_DIR}/parse_${EXCLUDE_CLASS}_train_alpha_loss.log"
    PARSE_EXIT=${PIPESTATUS[0]}
    if [[ ${PARSE_EXIT} -eq 0 ]] &&
        validate_result_csv "train" "${TRAIN_STATS_ATTEMPT}" "${TRAIN_STATS_STEPS}" &&
        atomic_install_csv "${TRAIN_STATS_ATTEMPT}" "${TRAIN_STATS_CSV}" "${TRAIN_STATS_HISTORY_DIR}" &&
        validate_result_csv "train" "${TRAIN_STATS_CSV}" "${TRAIN_STATS_STEPS}"; then
        status_append "train_log_parse_status=OK refreshed_from=${TRAIN_STATS_ATTEMPT} history=${TRAIN_STATS_HISTORY_DIR}"
    else
        status_append "train_log_parse_status=FAILED exit_code=${PARSE_EXIT}"
        FAILURES+=("train_log_parse")
    fi
fi

# Let the summary tool see the core pipeline state. If summary generation fails,
# a later PARTIAL marker is appended and becomes the authoritative last state.
if [[ ${#FAILURES[@]} -eq 0 ]]; then
    status_append "pipeline_state=SUMMARY_PENDING"
else
    status_append "pipeline_state=PARTIAL"
fi

echo "############################################################"
echo "# [7/7] PER-CLASS SUMMARY exclude_class=${EXCLUDE_CLASS}"
echo "############################################################"
SUMMARY_MD="${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_summary.md"
SUMMARY_CSV="${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_summary.csv"
SUMMARY_COMPLETENESS_CSV="${SUMMARY_DIR}/ddfsd_${EXCLUDE_CLASS}_20pct_completeness.csv"
SUMMARY_HISTORY_DIR="${SUMMARY_DIR}/history/${ATTEMPT_ID}"
if [[ -e "${SUMMARY_MD}" || -e "${SUMMARY_CSV}" || -e "${SUMMARY_COMPLETENESS_CSV}" ]]; then
    mkdir -p "${SUMMARY_HISTORY_DIR}"
    for SUMMARY_ARTIFACT in "${SUMMARY_MD}" "${SUMMARY_CSV}" "${SUMMARY_COMPLETENESS_CSV}"; do
        if [[ -e "${SUMMARY_ARTIFACT}" ]]; then
            cp -- "${SUMMARY_ARTIFACT}" "${SUMMARY_HISTORY_DIR}/$(basename "${SUMMARY_ARTIFACT}")"
        fi
    done
    status_append "summary_previous_outputs_preserved=${SUMMARY_HISTORY_DIR}"
fi

python tools/summarize_ddfsd_20pct.py per-class \
    --run-root "${EXPERIMENT_ROOT}" \
    --exclude-class "${EXCLUDE_CLASS}" \
    --output-path "${OUTPUT_PATH}" \
    --data-root "${DATA_ROOT}" \
    2>&1 | tee -a "${LOGS_DIR}/summarize_${EXCLUDE_CLASS}_20pct.log"
SUMMARY_EXIT=${PIPESTATUS[0]}
if [[ ${SUMMARY_EXIT} -eq 0 && -s "${SUMMARY_MD}" && -s "${SUMMARY_CSV}" ]] &&
    validate_completeness_csv "${SUMMARY_COMPLETENESS_CSV}"; then
    status_append "summary_status=OK"
else
    status_append "summary_status=FAILED exit_code=${SUMMARY_EXIT} reason=core_completeness_not_full_or_output_missing"
    FAILURES+=("summary")
fi

if [[ ${#FAILURES[@]} -eq 0 ]]; then
    status_append "pipeline_state=COMPLETE"
    status_append "pipeline_end=$(date -Iseconds)"
    echo "DDFSD 20pct pipeline COMPLETE for exclude_class=${EXCLUDE_CLASS}"
    exit 0
fi

FAILURE_LIST=$(IFS=,; printf '%s' "${FAILURES[*]}")
status_append "failure_stages=${FAILURE_LIST}"
status_append "pipeline_state=PARTIAL"
status_append "pipeline_end=$(date -Iseconds)"
echo "DDFSD 20pct pipeline PARTIAL for ${EXCLUDE_CLASS}; failed stages: ${FAILURE_LIST}" >&2
exit 1
