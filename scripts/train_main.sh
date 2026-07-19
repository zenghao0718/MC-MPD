#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/Dual-Domain-Few-Shot-AIGI-Detector"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"main_full_steps15000"}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/${EXPERIMENT_NAME}/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${OUTPUT_PATH}/freq_stats.pt"}

TOTAL_STEPS=${TOTAL_STEPS:-15000}
BATCH_SIZE=${BATCH_SIZE:-16}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
EVAL_INTERVAL=${EVAL_INTERVAL:-2500}
LOG_INTERVAL=${LOG_INTERVAL:-200}
LR_STEP=${LR_STEP:-5000}
LR_GAMMA=${LR_GAMMA:-0.5}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

case "${EXCLUDE_CLASS}" in
  ADM|BigGAN|glide|Midjourney|SD|VQDM) ;;
  *) echo "Invalid EXCLUDE_CLASS: ${EXCLUDE_CLASS}" >&2; exit 2 ;;
esac

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$(realpath -m "${OUTPUT_PATH}")/" in
  "$(realpath -m "${REPO_ROOT}")/"*)
    echo "OUTPUT_PATH must be outside the repository: ${OUTPUT_PATH}" >&2
    exit 1
    ;;
esac

if [[ -d "${OUTPUT_PATH}/ckpt" ]] && find "${OUTPUT_PATH}/ckpt" -maxdepth 1 -type f -name '*.pth' -print -quit | grep -q .; then
  echo "Refusing to overwrite an output directory that already contains checkpoints: ${OUTPUT_PATH}" >&2
  echo "Set OUTPUT_PATH to a new directory explicitly." >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
cd "${REPO_ROOT}"
OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 --nnodes 1 train_ddfsd.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --model_mode dual \
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
  --rgb_backbone_lr 3e-5 \
  --freq_backbone_lr 3e-5 \
  --rgb_head_lr 1e-4 \
  --freq_head_lr 1e-4 \
  --weight_decay 1e-4 \
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
  --freq_stats_path "${FREQ_STATS_PATH}" \
  --auto_compute_freq_stats True \
  --use_fp16 True \
  --pretrained True 2>&1 | tee "${OUTPUT_PATH}/train.log"
