#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH=${CKPT_PATH:-""}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-""}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-"./efficiency_experiments/ddfsd_dual_fp32_bs32_${TIMESTAMP}"}

BATCH_SIZE=${BATCH_SIZE:-32}
INPUT_SIZE=${INPUT_SIZE:-224}
WARMUP_ITERS=${WARMUP_ITERS:-50}
MEASURE_ITERS=${MEASURE_ITERS:-200}
REPEATS=${REPEATS:-5}
PRECISION=${PRECISION:-fp32}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Error: CKPT_PATH must point to a trained DDFSD dual checkpoint." >&2
  exit 2
fi
if [[ -z "${FREQ_STATS_PATH}" ]]; then
  echo "Error: FREQ_STATS_PATH must point to the checkpoint's freq_stats.pt." >&2
  exit 2
fi

# Smoke test:
# CKPT_PATH='/path/to/model_step[...].pth' FREQ_STATS_PATH='/path/to/freq_stats.pt' \
#   BATCH_SIZE=4 WARMUP_ITERS=5 MEASURE_ITERS=10 REPEATS=1 bash scripts/run_ddfsd_efficiency.sh
# Formal FP32 benchmark:
# CKPT_PATH='/path/to/model_step[...].pth' FREQ_STATS_PATH='/path/to/freq_stats.pt' \
#   BATCH_SIZE=32 WARMUP_ITERS=50 MEASURE_ITERS=200 REPEATS=5 PRECISION=fp32 \
#   bash scripts/run_ddfsd_efficiency.sh

python tools/benchmark_ddfsd_efficiency.py \
  --ckpt_path "${CKPT_PATH}" \
  --freq_stats_path "${FREQ_STATS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --input_size "${INPUT_SIZE}" \
  --warmup_iters "${WARMUP_ITERS}" \
  --measure_iters "${MEASURE_ITERS}" \
  --repeats "${REPEATS}" \
  --precision "${PRECISION}" \
  --seed "${SEED}" \
  --device "${DEVICE}"
