#!/usr/bin/env bash
# DDFSD alpha_mode x branch_mode diagnostic evaluation wrapper.
#
# Required env vars:
#   DATA_ROOT, OUTPUT_DIR, EXCLUDE_CLASS, CKPT_DIR, CKPT_STEPS, FREQ_STATS_PATH
#
# Optional env vars (with defaults):
#   SUPPORT_SHOT (10), EVAL_SEEDS (42,101,102,103,104), EVAL_BATCH_SIZE (128),
#   NUM_WORKERS (8), USE_FP16 (True), TAU (0.2), TAU_R (0.1),
#   ALPHA_MODES (adaptive), BRANCH_MODES (dual), MAX_EVAL_QUERY_PER_CLASS (0)
#
# Example:
#   export DATA_ROOT=/root/autodl-tmp/data
#   export OUTPUT_DIR=/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_10pct_steps15000/exclude_ADM/alpha_grid_adm
#   export EXCLUDE_CLASS=ADM
#   export CKPT_DIR=/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_10pct_steps15000/exclude_ADM/ckpt
#   export CKPT_STEPS=7500,12500,15000
#   export FREQ_STATS_PATH=/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_10pct_steps15000/exclude_ADM/freq_stats.pt
#   export ALPHA_MODES=adaptive,0.0,0.25,0.5,0.75,1.0
#   export BRANCH_MODES=dual
#   bash scripts/eval_ddfsd_alpha_grid.sh

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${EXCLUDE_CLASS:?EXCLUDE_CLASS is required}"
: "${CKPT_DIR:?CKPT_DIR is required}"
: "${CKPT_STEPS:?CKPT_STEPS is required}"
: "${FREQ_STATS_PATH:?FREQ_STATS_PATH is required}"

SUPPORT_SHOT="${SUPPORT_SHOT:-10}"
EVAL_SEEDS="${EVAL_SEEDS:-42,101,102,103,104}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_FP16="${USE_FP16:-True}"
TAU="${TAU:-0.2}"
TAU_R="${TAU_R:-0.1}"
ALPHA_MODES="${ALPHA_MODES:-adaptive}"
BRANCH_MODES="${BRANCH_MODES:-dual}"
MAX_EVAL_QUERY_PER_CLASS="${MAX_EVAL_QUERY_PER_CLASS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python tools/eval_ddfsd_alpha_grid.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --exclude_class "${EXCLUDE_CLASS}" \
  --ckpt_dir "${CKPT_DIR}" \
  --ckpt_steps "${CKPT_STEPS}" \
  --freq_stats_path "${FREQ_STATS_PATH}" \
  --support_shot "${SUPPORT_SHOT}" \
  --eval_seeds "${EVAL_SEEDS}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --use_fp16 "${USE_FP16}" \
  --tau "${TAU}" \
  --tau_r "${TAU_R}" \
  --alpha_modes "${ALPHA_MODES}" \
  --branch_modes "${BRANCH_MODES}" \
  --max_eval_query_per_class "${MAX_EVAL_QUERY_PER_CLASS}"
