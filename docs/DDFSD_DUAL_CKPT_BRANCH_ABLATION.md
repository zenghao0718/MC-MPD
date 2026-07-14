# DDFSD dual checkpoint inference-time branch ablation

This evaluation does not retrain a single-branch model. It always instantiates and loads the
complete dual checkpoint; `--branch_mode` only selects the distance used for final classification.

- `dual`: preserves the existing adaptive-alpha fused distance.
- `rgb-only`: uses only the RGB distance.
- `freq-only`: uses only the frequency distance.

The formal ablation adds only `rgb-only` and `freq-only`; the existing step-15000 dual result can
be reused. Omitting `--branch_mode` preserves the old behavior by selecting the checkpoint mode.

Single-class smoke-test example (run on AutoDL with real paths):

```bash
python test_ddfsd.py --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR/rgb-only/exclude_ADM" \
  --exclude_class ADM --ckpt_path "$CKPT_PATH" --ckpt_step 15000 --model_mode dual \
  --branch_mode rgb-only --freq_stats_path "$FREQ_STATS_PATH" --num_support_test 10 \
  --eval_repeats 1 --eval_seeds 42 --max_eval_query_per_class 10 --pretrained False
```

Six-class, five-seed, 10-shot batch evaluation:

```bash
DATA_ROOT=/path/to/GenImage \
EXPERIMENT_ROOT=/path/to/dual/experiment \
OUTPUT_ROOT=/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/branch_ablation \
bash scripts/eval_ddfsd_dual_ckpt_branch_ablation.sh
```

The batch script requires the exact step-15000 checkpoint and the training run's frequency
statistics for every excluded class. It never falls back to another checkpoint step and writes
each branch/class combination to a separate directory.
