#!/usr/bin/env bash
set -euo pipefail

MANIFEST_ROOT=${MANIFEST_ROOT:-"/root/autodl-tmp/MS_COCOAI_extracted/manifests/test/fewshot"}
CKPT_PATH=${CKPT_PATH:-"/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/ckpt/ddfsd_step[15000].pth"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/freq_stats_allsource.pt"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/root/autodl-tmp/runs/transfer_ms_cocoai/formal/test"}
NUM_WORKERS=${NUM_WORKERS:-8}
GENERATORS=(sd21 sdxl sd3 dalle3 midjourney_v6)
SEEDS=(42 101 102 103 104)

[[ -f "${CKPT_PATH}" ]] || { echo "Missing checkpoint: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${FREQ_STATS_PATH}" ]] || { echo "Missing frequency stats: ${FREQ_STATS_PATH}" >&2; exit 1; }
for generator in "${GENERATORS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        task_root="${MANIFEST_ROOT}/${generator}/seed_${seed}"
        output_dir="${OUTPUT_ROOT}/${generator}/seed_${seed}"
        [[ -f "${task_root}/support.csv" ]] || { echo "Missing ${task_root}/support.csv" >&2; exit 1; }
        [[ -f "${task_root}/query.csv" ]] || { echo "Missing ${task_root}/query.csv" >&2; exit 1; }
        mkdir -p "${output_dir}"
        python test_ddfsd_transfer.py \
            --dataset_name MS_COCOAI \
            --target_generator "${generator}" \
            --support_manifest "${task_root}/support.csv" \
            --query_manifest "${task_root}/query.csv" \
            --ckpt_path "${CKPT_PATH}" \
            --freq_stats_path "${FREQ_STATS_PATH}" \
            --output_dir "${output_dir}" \
            --seed "${seed}" \
            --num_workers "${NUM_WORKERS}" \
            --eval_batch_size 128 \
            --use_fp16 True \
            --tau 0.2 \
            --tau_r 0.1 \
            --branch_mode dual
    done
done
python tools/summarize_ms_cocoai_ddfsd.py \
    --input_root "${OUTPUT_ROOT}" \
    --output_dir "${OUTPUT_ROOT}/summary"
