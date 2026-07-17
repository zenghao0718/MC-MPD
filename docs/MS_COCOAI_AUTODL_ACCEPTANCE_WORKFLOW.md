# MS COCOAI Transfer: AutoDL Acceptance and Execution

All commands in this document run on AutoDL, not on the local source-editing computer.

## 1. Pull the branch and run acceptance first

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
git fetch origin
git switch exp-ddfsd-ms-cocoai-transfer-allsource-v1
git pull --ff-only origin exp-ddfsd-ms-cocoai-transfer-allsource-v1
git rev-parse HEAD
bash scripts/validate_ms_cocoai_transfer_code.sh --code-only
```

The default mode is also `code-only`. It checks the branch/commit, required imports, Python compilation, shell syntax, and pytest. It does not touch datasets and never starts training or formal inference. `--data-smoke` additionally checks the seven GenImage train directories and all ten MS COCOAI Parquet shards, then processes the complete 9,000-row Validation split.

Optional data smoke (still no training/inference):

```bash
bash scripts/validate_ms_cocoai_transfer_code.sh --data-smoke
```

The data-smoke requires exactly 9,000 extracted rows, 1,500 valid groups, zero anomaly rows, a DALL-E 3 seed-42 manifest with 10+10 support and 100+100 query, and a valid manifest SHA lock. It never uses `--allow_anomalies`.

## 2. Build the formal image/manifests

```bash
EXTRACT_ROOT=/root/autodl-tmp/MS_COCOAI

python tools/extract_ms_cocoai_parquet.py \
  --input_root /root/autodl-tmp/MS_COCOAI \
  --output_root "$EXTRACT_ROOT" \
  --split validation \
  --verify_existing

python tools/extract_ms_cocoai_parquet.py \
  --input_root /root/autodl-tmp/MS_COCOAI \
  --output_root "$EXTRACT_ROOT" \
  --split test \
  --verify_existing

python tools/build_ms_cocoai_groups.py \
  --base_manifest "$EXTRACT_ROOT/manifests/validation/base_rows.csv" \
  --output_dir "$EXTRACT_ROOT/manifests/validation"

python tools/build_ms_cocoai_groups.py \
  --base_manifest "$EXTRACT_ROOT/manifests/test/base_rows.csv" \
  --output_dir "$EXTRACT_ROOT/manifests/test"

python tools/build_ms_cocoai_fewshot_manifests.py \
  --grouped_manifest "$EXTRACT_ROOT/manifests/validation/all_rows_grouped.csv" \
  --output_dir "$EXTRACT_ROOT/manifests/validation/fewshot" \
  --split validation

python tools/build_ms_cocoai_fewshot_manifests.py \
  --grouped_manifest "$EXTRACT_ROOT/manifests/test/all_rows_grouped.csv" \
  --output_dir "$EXTRACT_ROOT/manifests/test/fewshot" \
  --split test

python tools/build_ms_cocoai_fewshot_manifests.py \
  --verify_lock "$EXTRACT_ROOT/manifests/validation/fewshot/manifest_lock.json"

python tools/build_ms_cocoai_fewshot_manifests.py \
  --verify_lock "$EXTRACT_ROOT/manifests/test/fewshot/manifest_lock.json"
```

Acceptance targets, which must be checked rather than bypassed, are 9,000/45,000 rows and 1,500/7,500 valid groups for validation/test. `group_anomalies.csv` must contain no anomalies for full data. Preserve `manifest_lock.json` and `manifest_sha256.txt`; DDFSD and later baselines must validate the lock before running.

## 3. Prepare provenance-bound stats, then train in screen

Prepare or verify the all-source GenImage-train statistics before training:

```bash
mkdir -p /root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000
screen -S ms_cocoai_allsource_stats
python tools/prepare_ddfsd_allsource_freq_stats.py \
  --data_root /root/autodl-tmp/data_fsd_full/GenImage \
  --output_path /root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/freq_stats_allsource.pt \
  2>&1 | tee /root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/prepare_freq_stats.log
```

The stats file contains `mean`, `std`, and metadata for the fixed ordered classes `real, ADM, BigGAN, glide, Midjourney, SD, VQDM`; its `.sha256` sidecar is verified on reuse. The training script refuses to auto-compute or run without this file. Saved checkpoints bind the exact stats SHA and metadata.

```bash
screen -S ms_cocoai_allsource_train
bash scripts/train_ddfsd_allsource.sh 2>&1 | tee /root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/train.log
```

Detach with `Ctrl+A`, then `D`; resume with:

```bash
screen -r ms_cocoai_allsource_train
```

Start TensorBoard in a separate screen session with an explicit run name:

```bash
screen -S tb_ms_cocoai_allsource
tensorboard --logdir_spec "exp-ddfsd-ms-cocoai-transfer-allsource-v1_allsource_step15000:/root/autodl-tmp/runs/transfer_ms_cocoai/train/ddfsd_allsource_full_step15000/tb" --host 0.0.0.0 --port 6006
```

Detach from `tb_ms_cocoai_allsource` with `Ctrl+A`, then `D`. In the AutoDL port-forwarding/service panel, copy the externally accessible URL for port `6006`; that actual forwarded URL is the one to report and open. A raw `localhost:6006` address is not sufficient unless AutoDL explicitly exposes it.

Only `ckpt/ddfsd_step[15000].pth` is used for formal transfer.

## 4. Smoke and formal inference in screen

```bash
screen -S ms_cocoai_dalle3_smoke
bash scripts/run_ms_cocoai_ddfsd_smoke.sh 2>&1 | tee /root/autodl-tmp/runs/transfer_ms_cocoai/smoke/validation/dalle3_seed42.log
```

After smoke acceptance:

```bash
screen -S ms_cocoai_formal_test
bash scripts/run_ms_cocoai_ddfsd_formal.sh 2>&1 | tee /root/autodl-tmp/runs/transfer_ms_cocoai/formal/test/formal.log
```

The formal output is split by generator and seed. Do not mix per-task outputs in one directory. Each evaluator task validates the checkpoint/stats binding and its two manifests against the lock. A complete existing result with identical provenance is safely skipped; a mismatch fails. Use evaluator `--overwrite` only for an intentional replacement after reviewing the printed old/new provenance differences. Formal shell scripts never pass it by default.

## 5. Required result comparison

After experiments finish, create:

```text
/root/autodl-tmp/runs/transfer_ms_cocoai/paper_vs_reproduce_comparison.md
/root/autodl-tmp/runs/transfer_ms_cocoai/paper_vs_reproduce_comparison.csv
```

The table must include method/branch, training setting, test category, Accuracy, AP, checkpoint, log directory, and notes, comparing paper-reported data, reproduced baseline, and this experiment. If the paper does not report MS COCOAI transfer, mark it `N/A` rather than inventing a value.
