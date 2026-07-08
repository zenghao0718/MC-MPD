#!/usr/bin/env bash
# Loop the formal dual 5-seed/10-shot DDFSD evaluation (test_ddfsd.py) over multiple
# checkpoints, writing each checkpoint's output into its own step_<N> subdirectory
# (test_ddfsd.py always writes fixed filenames ddfsd_eval_{per_seed,summary}.csv into
# --output_dir, so per-step subdirectories are required to avoid overwriting), then
# merges all per-step CSVs into <FORMAL_EVAL_DIR>/ddfsd_eval_{per_seed,summary}_all_steps.csv.
set -euo pipefail

EXCLUDE_CLASS=${EXCLUDE_CLASS:?EXCLUDE_CLASS is required}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_10pct_steps15000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
CKPT_STEPS=${CKPT_STEPS:-"2500,5000,7500,10000,12500,15000"}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
NUM_WORKERS=${NUM_WORKERS:-8}
FORMAL_EVAL_DIR=${FORMAL_EVAL_DIR:-"${OUTPUT_PATH}/formal_eval"}

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${FREQ_STATS_PATH}" ]]; then
    echo "Frequency stats do not exist: ${FREQ_STATS_PATH}" >&2
    exit 1
fi

mkdir -p "${FORMAL_EVAL_DIR}"

IFS=',' read -ra STEPS_ARR <<< "${CKPT_STEPS}"
MISSING_STEPS=()
for STEP in "${STEPS_ARR[@]}"; do
    STEP_DIR="${FORMAL_EVAL_DIR}/step_${STEP}"
    CKPT_PATH="${OUTPUT_PATH}/ckpt/ddfsd_step[${STEP}].pth"
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "WARNING: checkpoint missing for step ${STEP}: ${CKPT_PATH} (skipped)" >&2
        MISSING_STEPS+=("${STEP}")
        continue
    fi
    mkdir -p "${STEP_DIR}"
    echo "=== formal eval: exclude_class=${EXCLUDE_CLASS} step=${STEP} ==="
    python test_ddfsd.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${STEP_DIR}" \
        --num_workers "${NUM_WORKERS}" \
        --seed 42 \
        --exclude_class "${EXCLUDE_CLASS}" \
        --ckpt_path "${CKPT_PATH}" \
        --ckpt_step "${STEP}" \
        --freq_stats_path "${FREQ_STATS_PATH}" \
        --num_support_test 10 \
        --eval_repeats 5 \
        --eval_seeds "${EVAL_SEEDS}" \
        --tau 0.2 \
        --tau_r 0.1 \
        --use_fp16 True \
        --pretrained False
done

python - "${FORMAL_EVAL_DIR}" "${CKPT_STEPS}" <<'PYEOF'
import csv
import os
import sys

formal_eval_dir, ckpt_steps = sys.argv[1], sys.argv[2]
steps = [s.strip() for s in ckpt_steps.split(",") if s.strip()]

per_seed_rows, per_seed_fields = [], None
summary_rows, summary_fields = [], None

for step in steps:
    step_dir = os.path.join(formal_eval_dir, f"step_{step}")
    per_seed_csv = os.path.join(step_dir, "ddfsd_eval_per_seed.csv")
    summary_csv = os.path.join(step_dir, "ddfsd_eval_summary.csv")
    if os.path.exists(per_seed_csv):
        with open(per_seed_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            per_seed_fields = per_seed_fields or reader.fieldnames
            per_seed_rows.extend(list(reader))
    if os.path.exists(summary_csv):
        with open(summary_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            summary_fields = summary_fields or reader.fieldnames
            summary_rows.extend(list(reader))

if per_seed_fields:
    out_path = os.path.join(formal_eval_dir, "ddfsd_eval_per_seed_all_steps.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"Merged per-seed CSV -> {out_path} ({len(per_seed_rows)} rows)")

if summary_fields:
    out_path = os.path.join(formal_eval_dir, "ddfsd_eval_summary_all_steps.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Merged summary CSV -> {out_path} ({len(summary_rows)} rows)")
PYEOF

if [[ ${#MISSING_STEPS[@]} -gt 0 ]]; then
    echo "MISSING_CKPT_STEPS: ${MISSING_STEPS[*]}" >&2
fi
