#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

MODEL_MODE=${MODEL_MODE:-}
if [[ "${MODEL_MODE}" != "rgb-only" && "${MODEL_MODE}" != "freq-only" ]]; then
    echo "MODEL_MODE must be explicitly set to rgb-only or freq-only." >&2
    echo "Example: MODEL_MODE=rgb-only EXCLUDE_CLASS=ADM bash scripts/train_ddfsd_ablation_full_10pct_schedule.sh" >&2
    exit 1
fi

NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
EXCLUDE_CLASS=${EXCLUDE_CLASS:-ADM}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"}
RUN_CONFIG=${RUN_CONFIG:-"ddfsd_ablation_full_steps15000_10pct_schedule"}
OUTPUT_PATH=${OUTPUT_PATH:-"${RUN_ROOT}/${RUN_CONFIG}/${MODEL_MODE}/exclude_${EXCLUDE_CLASS}"}
FREQ_STATS_PATH=${FREQ_STATS_PATH:-"${RUN_ROOT}/shared_full_freq_stats/exclude_${EXCLUDE_CLASS}/freq_stats.pt"}
BATCH_SIZE=${BATCH_SIZE:-16}

TOTAL_STEPS=${TOTAL_STEPS:-15000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
EVAL_INTERVAL=${EVAL_INTERVAL:-2500}
LOG_INTERVAL=${LOG_INTERVAL:-200}
LR_STEP=${LR_STEP:-5000}
LR_GAMMA=${LR_GAMMA:-0.5}

RGB_BACKBONE_LR=${RGB_BACKBONE_LR:-3e-5}
FREQ_BACKBONE_LR=${FREQ_BACKBONE_LR:-3e-5}
RGB_HEAD_LR=${RGB_HEAD_LR:-1e-4}
FREQ_HEAD_LR=${FREQ_HEAD_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
M_RF=${M_RF:-1.2}
M_FF=${M_FF:-0.6}

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the full GenImage root with the FSD layout." >&2
    echo "It must contain: real/ ADM/ BigGAN/ glide/ Midjourney/ SD/ VQDM/." >&2
    echo "Each class directory must contain train/ and val/." >&2
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

TRAIN_ARGS=(
    --data_root "${DATA_ROOT}"
    --output_dir "${OUTPUT_PATH}"
    --model_mode "${MODEL_MODE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --exclude_class "${EXCLUDE_CLASS}"
    --batch_size "${BATCH_SIZE}"
    --total_training_steps "${TOTAL_STEPS}"
    --save_interval "${SAVE_INTERVAL}"
    --eval_interval "${EVAL_INTERVAL}"
    --log_interval "${LOG_INTERVAL}"
    --num_class_train 3
    --num_support_train 5
    --num_query_train 5
    --num_support_val 10
    --scheduler_type step
    --lr_scheduler_step "${LR_STEP}"
    --lr_scheduler_gamma "${LR_GAMMA}"
    --rgb_backbone_lr "${RGB_BACKBONE_LR}"
    --freq_backbone_lr "${FREQ_BACKBONE_LR}"
    --rgb_head_lr "${RGB_HEAD_LR}"
    --freq_head_lr "${FREQ_HEAD_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --tau 0.2
    --tau_r 0.1
    --m_rf "${M_RF}"
    --m_ff "${M_FF}"
    --lambda_ff 0.5
    --lambda_sep_target 0.03
    --lambda_sep_warmup_start 2500
    --lambda_sep_warmup_end 7500
    --use_fp16 True
    --pretrained True
)

if [[ "${MODEL_MODE}" == "freq-only" ]]; then
    TRAIN_ARGS+=(
        --freq_stats_path "${FREQ_STATS_PATH}"
        --auto_compute_freq_stats True
    )
fi

OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 --nnodes 1 train_ddfsd.py "${TRAIN_ARGS[@]}"
