#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST_ROOT=${MANIFEST_ROOT:-"/root/autodl-tmp/MS_COCOAI/manifests/validation/fewshot"}
CKPT_PATH=${CKPT_PATH:-"/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/ckpt/ddfsd_step[15000].pth"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/freq_stats_allsource.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/runs/transfer_ms_cocoai/smoke/validation/dalle3/seed_42"}
NUM_WORKERS=${NUM_WORKERS:-8}

SUPPORT_MANIFEST="${MANIFEST_ROOT}/dalle3/seed_42/support.csv"
QUERY_MANIFEST="${MANIFEST_ROOT}/dalle3/seed_42/query.csv"
MANIFEST_LOCK="${MANIFEST_ROOT}/manifest_lock.json"
for required in "${SUPPORT_MANIFEST}" "${QUERY_MANIFEST}" "${MANIFEST_LOCK}" "${CKPT_PATH}" "${FREQ_STATS_PATH}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
python tools/build_ms_cocoai_fewshot_manifests.py --verify_lock "${MANIFEST_LOCK}"
mkdir -p "${OUTPUT_DIR}"
python test_ddfsd_transfer.py \
    --dataset_name MS_COCOAI \
    --target_generator dalle3 \
    --support_manifest "${SUPPORT_MANIFEST}" \
    --query_manifest "${QUERY_MANIFEST}" \
    --manifest_lock "${MANIFEST_LOCK}" \
    --ckpt_path "${CKPT_PATH}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num_workers "${NUM_WORKERS}" \
    --eval_batch_size 128 \
    --use_fp16 True \
    --tau 0.2 \
    --tau_r 0.1 \
    --branch_mode dual
