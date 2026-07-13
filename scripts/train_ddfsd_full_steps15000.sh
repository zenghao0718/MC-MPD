#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_full"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-"ADM"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/ddfsd_full_steps15000/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}
TOTAL_STEPS=${TOTAL_STEPS:-15000}; SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
EVAL_INTERVAL=${EVAL_INTERVAL:-2500}; LOG_INTERVAL=${LOG_INTERVAL:-200}
SCHEDULER_TYPE=${SCHEDULER_TYPE:-"step"}; LR_SCHEDULER_STEP=${LR_SCHEDULER_STEP:-5000}
LR_SCHEDULER_GAMMA=${LR_SCHEDULER_GAMMA:-0.5}
NUM_WORKERS=${NUM_WORKERS:-8}; SEED=${SEED:-42}; BATCH_SIZE=${BATCH_SIZE:-16}

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
if [[ -d "${OUTPUT_PATH}/ckpt" ]] && find "${OUTPUT_PATH}/ckpt" -maxdepth 1 -name '*.pth' -print -quit | grep -q .; then
    echo "Refusing to overwrite checkpoints in ${OUTPUT_PATH}/ckpt" >&2; exit 1
fi
printf '%s\n' "DATA_ROOT=${DATA_ROOT}" "RUN_ROOT=${RUN_ROOT}" "OUTPUT_PATH=${OUTPUT_PATH}" \
  "EXCLUDE_CLASS=${EXCLUDE_CLASS}" "TOTAL_STEPS=${TOTAL_STEPS}" "SAVE_INTERVAL=${SAVE_INTERVAL}" \
  "EVAL_INTERVAL=${EVAL_INTERVAL}" "LOG_INTERVAL=${LOG_INTERVAL}" "SCHEDULER_TYPE=${SCHEDULER_TYPE}" \
  "LR_SCHEDULER_STEP=${LR_SCHEDULER_STEP}" "LR_SCHEDULER_GAMMA=${LR_SCHEDULER_GAMMA}" \
  "FREQ_STATS_PATH=${FREQ_STATS_PATH}"
mkdir -p "${OUTPUT_PATH}"
OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 --nnodes 1 train_ddfsd.py \
  --data_root "${DATA_ROOT}" --output_dir "${OUTPUT_PATH}" --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" --exclude_class "${EXCLUDE_CLASS}" --batch_size "${BATCH_SIZE}" \
  --total_training_steps "${TOTAL_STEPS}" --save_interval "${SAVE_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" --log_interval "${LOG_INTERVAL}" \
  --num_class_train 3 --num_support_train 5 --num_query_train 5 --num_support_val 10 \
  --scheduler_type "${SCHEDULER_TYPE}" --lr_scheduler_step "${LR_SCHEDULER_STEP}" \
  --lr_scheduler_gamma "${LR_SCHEDULER_GAMMA}" --rgb_backbone_lr 3e-5 --freq_backbone_lr 3e-5 \
  --rgb_head_lr 1e-4 --freq_head_lr 1e-4 --weight_decay 1e-4 --tau 0.2 --tau_r 0.1 \
  --m_rf 1.2 --m_ff 0.6 --lambda_ff 0.5 --lambda_sep_target 0.03 \
  --lambda_sep_warmup_start 2500 --lambda_sep_warmup_end 7500 \
  --branch_dropout_dual_prob 0.90 --branch_dropout_rgb_prob 0.05 --branch_dropout_freq_prob 0.05 \
  --freq_stats_path "${FREQ_STATS_PATH}" --auto_compute_freq_stats True --use_fp16 True --pretrained True
