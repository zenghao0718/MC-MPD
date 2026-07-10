#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_20pct/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
RUN_CONFIG=${RUN_CONFIG:-"ddfsd_20pct_steps30000"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/${RUN_CONFIG}/exclude_${EXCLUDE_CLASS}"}
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

# Keep the established 1/5 model defaults while allowing the pipeline to record
# and explicitly pass an overridden value when a controlled follow-up run needs it.
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

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the real 1/5 GenImage root (default: /root/autodl-tmp/data_fsd_20pct/GenImage)." >&2
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
    --tau "${TAU}" \
    --tau_r "${TAU_R}" \
    --m_rf "${M_RF}" \
    --m_ff "${M_FF}" \
    --lambda_ff "${LAMBDA_FF}" \
    --lambda_sep_target "${LAMBDA_SEP_TARGET}" \
    --lambda_sep_warmup_start "${LAMBDA_SEP_WARMUP_START}" \
    --lambda_sep_warmup_end "${LAMBDA_SEP_WARMUP_END}" \
    --branch_dropout_dual_prob "${BRANCH_DROPOUT_DUAL_PROB}" \
    --branch_dropout_rgb_prob "${BRANCH_DROPOUT_RGB_PROB}" \
    --branch_dropout_freq_prob "${BRANCH_DROPOUT_FREQ_PROB}" \
    --freq_stats_path "${FREQ_STATS_PATH}" \
    --auto_compute_freq_stats "${AUTO_COMPUTE_FREQ_STATS}" \
    --use_fp16 "${USE_FP16}" \
    --pretrained "${PRETRAINED}"
