# Training script for Few-Shot AIGI Detector on 1/10 GenImage subset.
# - Single RTX 4090 (24G), FP16
# - EXCLUDE_CLASS / OUTPUT_PATH / TOTAL_STEPS can be overridden via env vars so
#   that an outer driver script (see scripts/run_all_excludes.sh) can loop over
#   the 6 fake generators for cross-generator (leave-one-out) evaluation.

# huggingface.co is blocked on autodl China nodes; use the public HF mirror so
# timm can fetch the ImageNet-pretrained ResNet50 weights.
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

GPU_NUM=${GPU_NUM:-1}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
"

# Root that directly contains ADM/ BigGAN/ glide/ Midjourney/ SD/ VQDM/ real/
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}

# Which fake generator to hold out (becomes the "unseen" test class).
EXCLUDE_CLASS=${EXCLUDE_CLASS:-"ADM"}

# Per-run output dir. Defaults to ./output_dir/exclude_<class> so every
# leave-one-out run keeps its own checkpoints and log.
OUTPUT_PATH=${OUTPUT_PATH:-"./output_dir/exclude_${EXCLUDE_CLASS}"}

# Hyper-parameters tuned for ~1/10 GenImage (16k/class train, 600/class val).
TOTAL_STEPS=${TOTAL_STEPS:-15000}
LR=${LR:-1e-4}
LR_STEP=${LR_STEP:-5000}
LR_GAMMA=${LR_GAMMA:-0.5}
BATCH_SIZE=${BATCH_SIZE:-16}
ACCUM_STEPS=${ACCUM_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
EVAL_INTERVAL=${EVAL_INTERVAL:-2500}
LOG_INTERVAL=${LOG_INTERVAL:-200}

mkdir -p "$OUTPUT_PATH"

OMP_NUM_THREADS=1 torchrun $DISTRIBUTED_ARGS train.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_PATH" \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_scheduler_gamma $LR_GAMMA \
    --lr_scheduler_step $LR_STEP \
    --exclude_class $EXCLUDE_CLASS \
    --total_training_steps $TOTAL_STEPS \
    --accumulation_steps $ACCUM_STEPS \
    --save_interval $SAVE_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --num_class_train 3 \
    --num_support_train 5 \
    --num_query_train 5 \
    --num_class_val 2 \
    --num_support_val 5 \
    --num_query_val 15 \
    --use_fp16 True
