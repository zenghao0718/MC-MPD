#!/usr/bin/env bash
# Cross-generator (leave-one-out) driver.
#
# For each of the 6 fake generators in GenImage, train a model that excludes
# that generator from training, then evaluate on its val split. Produces 6
# independent checkpoints + logs under $RUNS_ROOT/exclude_<class>/.
#
# All checkpoints are kept (disk expanded to 70G, ~1.7G per run is fine).
#
# Usage:
#   bash scripts/run_all_excludes.sh                # train + eval all 6
#   MODE=train bash scripts/run_all_excludes.sh     # only train
#   MODE=eval  bash scripts/run_all_excludes.sh     # only eval (needs ckpts)
#   CLASSES="ADM VQDM" bash scripts/run_all_excludes.sh  # subset of classes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CLASSES=${CLASSES:-"ADM BigGAN glide Midjourney SD VQDM"}
RUNS_ROOT=${RUNS_ROOT:-"/root/autodl-tmp/runs"}
TOTAL_STEPS=${TOTAL_STEPS:-15000}
MODE=${MODE:-"all"}  # all | train | eval

mkdir -p "$RUNS_ROOT"

run_train() {
    local cls="$1"
    local out="$RUNS_ROOT/exclude_${cls}"
    mkdir -p "$out"
    echo "================================================================"
    echo "[TRAIN] exclude_class=${cls}  output=${out}  steps=${TOTAL_STEPS}"
    echo "================================================================"

    EXCLUDE_CLASS="$cls" \
    OUTPUT_PATH="$out" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    bash scripts/train.sh >>"$out/train.log" 2>&1
}

run_eval() {
    local cls="$1"
    local out="$RUNS_ROOT/exclude_${cls}"
    mkdir -p "$out"
    local ckpt="$out/ckpt/resnet50_step[${TOTAL_STEPS}].pth"
    echo "================================================================"
    echo "[EVAL]  test_class=${cls}  ckpt=${ckpt}"
    echo "================================================================"

    if [[ ! -f "$ckpt" ]]; then
        echo "[EVAL][ERROR] checkpoint not found: $ckpt"
        return 1
    fi

    TEST_CLASS="$cls" \
    OUTPUT_PATH="$out" \
    CKPT_PATH="$ckpt" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    bash scripts/eval.sh >>"$out/eval.log" 2>&1
}

for cls in $CLASSES; do
    case "$MODE" in
        train) run_train "$cls" ;;
        eval)  run_eval  "$cls" ;;
        all)   run_train "$cls"; run_eval "$cls" ;;
        *)     echo "Unknown MODE=$MODE"; exit 1 ;;
    esac
done

echo "All requested runs finished. Results under: $RUNS_ROOT"
