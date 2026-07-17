# DDFSD All-Source × MS COCOAI Transfer Design

## Scope

This change adds a DDFSD-only cross-dataset pipeline. It does not modify FSD, add MS COCOAI names to GenImage `FAKE_CLASSES`, or route external images through the GenImage loader.

## All-source training

`--training_scope all-source` changes only the candidate fake pool from five leave-one-out sources to all six GenImage fake sources:

```text
ADM, BigGAN, glide, Midjourney, SD, VQDM
```

Every training episode remains exactly `real + random two fake` with three classes, five support images and five query images per class. The default remains `--training_scope leave-one-out`, including its excluded-generator validation behavior.

Frequency statistics are computed only from the GenImage `train` directories for:

```text
real, ADM, BigGAN, glide, Midjourney, SD, VQDM
```

The all-source file is `freq_stats_allsource.pt`. MS COCOAI images never contribute to frequency statistics. The file retains backward-compatible `mean`/`std` tensors and adds metadata containing the all-source scope, `GenImage`, `train`, the fixed ordered classes, per-class counts, data root, UTC creation time, and code commit. Checkpoints bind the actual stats file with `freq_stats_sha256` and `freq_stats_metadata`.

Training-period validation treats all six fake generators as `val_seen`. All-source checkpoints record `training_scope`, `source_fake_classes`, and `source_classes`; strict resume refuses to reinterpret a legacy or leave-one-out checkpoint as all-source.

## Original MS COCOAI images and semantic groups

`tools/extract_ms_cocoai_parquet.py` iterates Parquet shards and row groups, writes `Image.bytes` directly, and uses PIL only to read/validate dimensions, format, and mode. Existing files are always SHA-256 checked; a mismatch or empty bytes is fatal. No JPEG is decoded and re-saved. Defaults keep Parquet, images, and manifests under `/root/autodl-tmp/MS_COCOAI` (`validation|test`, `images/validation|test`, and `manifests/validation|test`).

Grouping does not use `global_row_index // 6` and does not treat Caption alone as a unique group. For each Caption:

1. Split rows by `Label_B=0..5`.
2. Stable-sort each label by `(shard_index, row_index_in_shard, global_row_index)`.
3. Require all six label lists to have equal positive lengths.
4. Pair the kth row from every label list.
5. Set `occurrence_index=k` and `group_id=caption_sha256[:16] + "_" + k`.

Any anomaly is written to `group_anomalies.csv`; the full-data command fails by default. Each six-image group must also contain six distinct image SHA-256 values.

## Fixed few-shot protocol

Formal test seeds are `42,101,102,103,104`, with master seed `20260717`. Fifty complete groups are selected and partitioned into five disjoint ten-group support sets. All fifty are removed from the shared query pool. Every task uses ten real and ten target-generator support images. Query real/fake counts are balanced, query rows are identical across seeds for a generator, and all generators share the same real query rows.

Validation produces only a DALL-E 3, seed-42 smoke task with ten real + ten fake support and 100 real + 100 fake query images. All emitted file paths and image SHA-256 values are verified against the grouped base manifest. Support/query paths, groups, and image hashes are disjoint; neither side may contain duplicate hashes. Generation freezes the grouped input, index, provenance, support pool, query groups, and task manifests in `manifest_lock.json` plus `manifest_sha256.txt`.

## Transfer inference and reporting

`test_ddfsd_transfer.py` performs no-gradient prototype inference:

```text
10 real + 10 fake support
  -> RGB/frequency embeddings
  -> two prototypes
  -> support sigma
  -> adaptive alpha
  -> query probabilities
  -> threshold 0.5
```

There is no optimizer, backward pass, fine-tuning, target-statistics calculation, or target-side parameter selection. Formal mode requires a dual, all-source, step-15000 checkpoint, the exact six fake/source class lists, checkpoint and CLI `tau=0.2`/`tau_r=0.1`, a dual branch, valid GenImage-train all-source stats metadata, a matching checkpoint-bound stats SHA, and locked manifests. Outputs are:

- `per_image_scores.csv`
- `metrics.json`
- `run_config.json`
- `provenance.json`

The summary tool independently recalculates ACC/AP/AUC from per-image CSV files. Generator standard deviation uses `ddof=0`; Macro is the equal-weight average of the five generator means. Future baselines must reuse these exact query manifests.

Evaluator outputs are overwrite-safe. A complete result with an identical identity (checkpoint/stats/support/query hashes, generator, seed, tau/tau_r, and branch) is skipped. Any mismatch or partial non-empty directory fails unless `--overwrite` is explicit. Debug runs using `--allow_nonformal_checkpoint` always record `formal_checkpoint_validation=false` and `formal_result=false`.

## Out of scope

FSD, target-data tuning, checkpoint/threshold/tau/alpha selection on MS COCOAI, local training, and local real-data extraction are outside this task.
