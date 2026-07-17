#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
OUTPUT_PATH=${OUTPUT_PATH:-"/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats_allsource.pt"}
TOTAL_STEPS=${TOTAL_STEPS:-15000}
GPU_NUM=${GPU_NUM:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-16}

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
if [[ -d "${OUTPUT_PATH}/ckpt" ]] && find "${OUTPUT_PATH}/ckpt" -maxdepth 1 -name '*.pth' -print -quit | grep -q .; then
    echo "Refusing to overwrite checkpoints in ${OUTPUT_PATH}/ckpt" >&2
    exit 1
fi
mkdir -p "${OUTPUT_PATH}"
printf '%s\n' \
    "training_scope=all-source" \
    "source_classes=real,ADM,BigGAN,glide,Midjourney,SD,VQDM" \
    "episode=real+2fake" \
    "DATA_ROOT=${DATA_ROOT}" \
    "OUTPUT_PATH=${OUTPUT_PATH}" \
    "FREQ_STATS_PATH=${FREQ_STATS_PATH}" \
    "TOTAL_STEPS=${TOTAL_STEPS}"

OMP_NUM_THREADS=1 torchrun --nproc_per_node "${GPU_NUM}" --nnodes 1 train_ddfsd.py \
    --training_scope all-source \
    --model_mode dual \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_PATH}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --auto_compute_freq_stats True \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_class_train 3 \
    --num_support_train 5 \
    --num_query_train 5 \
    --num_support_val 10 \
    --total_training_steps "${TOTAL_STEPS}" \
    --save_interval 2500 \
    --eval_interval 2500 \
    --log_interval 200 \
    --rgb_backbone_lr 3e-5 \
    --freq_backbone_lr 3e-5 \
    --rgb_head_lr 1e-4 \
    --freq_head_lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler_type step \
    --lr_scheduler_step 5000 \
    --lr_scheduler_gamma 0.5 \
    --tau 0.2 \
    --tau_r 0.1 \
    --m_rf 1.2 \
    --m_ff 0.6 \
    --lambda_ff 0.5 \
    --lambda_sep_target 0.03 \
    --lambda_sep_warmup_start 2500 \
    --lambda_sep_warmup_end 7500 \
    --branch_dropout_dual_prob 0.90 \
    --branch_dropout_rgb_prob 0.05 \
    --branch_dropout_freq_prob 0.05 \
    --use_fp16 True \
    --pretrained True
