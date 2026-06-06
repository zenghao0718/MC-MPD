# Evaluation script. Can be driven per-class by the outer loop in
# scripts/run_all_excludes.sh, or run standalone.
#
# IMPORTANT: util/utils.py saves checkpoints as
#   resnet50_step[<N>].pth
# (with literal square brackets). That differs from the old default in this
# script. Always quote CKPT_PATH so the shell does not glob the brackets.

# huggingface.co is blocked on autodl China nodes; use the public HF mirror so
# timm can fetch the ImageNet-pretrained ResNet50 weights when cache is missing.
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-8}

DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/data"}

# The held-out generator used for cross-generator evaluation.
TEST_CLASS=${TEST_CLASS:-"ADM"}

# Per-run output dir for eval logs.
OUTPUT_PATH=${OUTPUT_PATH:-"./output_dir/exclude_${TEST_CLASS}"}

# Default checkpoint path matches what save_model() writes: resnet50_step[N].pth
TOTAL_STEPS=${TOTAL_STEPS:-15000}
CKPT_PATH=${CKPT_PATH:-"${OUTPUT_PATH}/ckpt/resnet50_step[${TOTAL_STEPS}].pth"}
METRIC=${METRIC:-squared_euclidean}
INIT_SCALE=${INIT_SCALE:-10.0}
MAX_SCALE=${MAX_SCALE:-100.0}
SCALE_EPS=${SCALE_EPS:-1e-12}

mkdir -p "$OUTPUT_PATH"

python test.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_PATH" \
    --test_class "$TEST_CLASS" \
    --ckpt_path "$CKPT_PATH" \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --num_class_test 2 \
    --num_support_test 5 \
    --num_query_test 15 \
    --use_fp16 True \
    --metric "$METRIC" \
    --init_scale $INIT_SCALE \
    --max_scale $MAX_SCALE \
    --scale_eps $SCALE_EPS
