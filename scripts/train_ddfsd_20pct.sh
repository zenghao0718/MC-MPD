#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_20pct"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_20pct_steps30000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
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

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the real 1/5 GenImage root, for example: DATA_ROOT=/path/to/data_20pct ${0}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 --nnodes 1 train_ddfsd.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --exclude_class "${EXCLUDE_CLASS}" \
    --batch_size "${BATCH_SIZE}" \
    --total_training_steps "${TOTAL_STEPS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    --num_class_train 3 \
    --num_support_train 5 \
    --num_query_train 5 \
    --num_support_val 10 \
    --scheduler_type step \
    --lr_scheduler_step "${LR_STEP}" \
    --lr_scheduler_gamma "${LR_GAMMA}" \
    --rgb_backbone_lr "${RGB_BACKBONE_LR}" \
    --freq_backbone_lr "${FREQ_BACKBONE_LR}" \
    --rgb_head_lr "${RGB_HEAD_LR}" \
    --freq_head_lr "${FREQ_HEAD_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --tau 0.2 \
    --tau_r 0.1 \
    --m_rf 1.2 \
    --m_ff 0.6 \
    --lambda_ff 0.5 \
    --lambda_sep_target 0.03 \
    --lambda_sep_warmup_start 5000 \
    --lambda_sep_warmup_end 15000 \
    --branch_dropout_dual_prob 0.90 \
    --branch_dropout_rgb_prob 0.05 \
    --branch_dropout_freq_prob 0.05 \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --auto_compute_freq_stats True \
    --use_fp16 True \
    --pretrained True
