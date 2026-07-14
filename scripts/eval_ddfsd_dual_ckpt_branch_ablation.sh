#!/usr/bin/env bash
# Evaluate a 15000-step dual checkpoint with RGB-only and frequency-only inference.
# Required: DATA_ROOT and EXPERIMENT_ROOT. Optional: OUTPUT_ROOT, NUM_WORKERS.
# EXPERIMENT_ROOT must contain exclude_<class>/ckpt/ddfsd_step[15000].pth
# and exclude_<class>/freq_stats.pt for every class below.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the GenImage dataset root}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the dual-checkpoint experiment root}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${EXPERIMENT_ROOT}/eval_dual_ckpt_branch_ablation_step15000"}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_SEEDS=${EVAL_SEEDS:-"42,101,102,103,104"}
CLASSES=(ADM BigGAN glide Midjourney SD VQDM)
BRANCH_MODES=(rgb-only freq-only)

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }

for exclude_class in "${CLASSES[@]}"; do
    run_dir="${EXPERIMENT_ROOT}/exclude_${exclude_class}"
    ckpt_path="${run_dir}/ckpt/ddfsd_step[15000].pth"
    freq_stats_path="${run_dir}/freq_stats.pt"
    [[ -f "${ckpt_path}" ]] || { echo "Missing exact step-15000 checkpoint: ${ckpt_path}" >&2; exit 1; }
    [[ -f "${freq_stats_path}" ]] || { echo "Missing frequency stats: ${freq_stats_path}" >&2; exit 1; }

    for branch_mode in "${BRANCH_MODES[@]}"; do
        output_dir="${OUTPUT_ROOT}/${branch_mode}/exclude_${exclude_class}"
        mkdir -p "${output_dir}"
        command=(python test_ddfsd.py
            --data_root "${DATA_ROOT}" --output_dir "${output_dir}"
            --num_workers "${NUM_WORKERS}" --seed 42 --exclude_class "${exclude_class}"
            --ckpt_path "${ckpt_path}" --ckpt_step 15000 --model_mode dual
            --branch_mode "${branch_mode}" --freq_stats_path "${freq_stats_path}"
            --num_support_test 10 --eval_repeats 5 --eval_seeds "${EVAL_SEEDS}"
            --tau 0.2 --tau_r 0.1 --use_fp16 True --pretrained False)
        printf 'Executing:' | tee "${output_dir}/command.log"
        printf ' %q' "${command[@]}" | tee -a "${output_dir}/command.log"
        printf '\n' | tee -a "${output_dir}/command.log"
        "${command[@]}" 2>&1 | tee -a "${output_dir}/eval.log"
    done
done
