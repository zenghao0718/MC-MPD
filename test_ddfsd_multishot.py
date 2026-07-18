# -*- coding: utf-8 -*-
"""Unified deterministic 0/1/2/5/10/20-shot DDFSD evaluation."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from test_ddfsd import (
    checkpoint_freq_stats_path, load_checkpoint, resolve_model_mode, str2bool, write_csv,
)
from util.ddfsd_multishot_logic import resolve_checkpoint_step


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDFSD at multiple support shots")
    parser.add_argument("--model", default="ddfsd")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--exclude_class", default="ADM")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--ckpt_step", type=int, default=0)
    parser.add_argument("--model_mode", default="auto", choices=["auto", "dual", "rgb-only", "freq-only"])
    parser.add_argument("--branch_mode", default=None, choices=["dual", "rgb-only", "freq-only"])
    parser.add_argument("--freq_stats_path", default="")
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--eval_seeds", default="42,101,102,103,104")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_eval_query_per_class", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    parser.add_argument("--shot_list", default="0,1,2,5,10,20")
    parser.add_argument("--zero_shot_metadata_per_class", type=int, default=1024)
    return parser.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _mean_std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _manifest_rows(exclude_class, seed, data_class, dataset, candidates, query):
    rows = []
    for rank, index in enumerate(candidates, 1):
        rows.append({"exclude_class": exclude_class, "seed": seed, "data_class": data_class,
                     "split": "val", "dataset_index": index, "filepath": dataset.paths[index],
                     "role": "support_candidate", "support_rank": rank})
    for index in query:
        rows.append({"exclude_class": exclude_class, "seed": seed, "data_class": data_class,
                     "split": "val", "dataset_index": index, "filepath": dataset.paths[index],
                     "role": "query", "support_rank": ""})
    return rows


def main():
    args = parse_args()
    # Heavy runtime dependencies remain lazy so py_compile works on lightweight machines.
    import torch
    import util.logger as logger
    from datasets.ddfsd_datasets import get_train_fake_classes, load_ddfsd_class_dataset, validate_generator_name
    from model.ddfsd import DDFSDDualDomainNet
    from util.ddfsd_frequency import load_frequency_stats
    from util.ddfsd_multishot_eval import (
        build_multishot_support_query_indices, encode_dataset_indices,
        evaluate_few_shot_embeddings, evaluate_zero_shot_embeddings,
        parse_shot_list, sample_metadata_indices,
    )
    from util.utils import set_seed

    validate_generator_name(args.exclude_class)
    shots = parse_shot_list(args.shot_list)
    positive_shots = [shot for shot in shots if shot > 0]
    max_shot = max(positive_shots, default=0)
    seeds = [int(item.strip()) for item in args.eval_seeds.split(",") if item.strip()]
    seeds = seeds[:args.eval_repeats] if args.eval_repeats > 0 else seeds
    if not seeds:
        raise ValueError("No eval seeds were provided.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate eval seeds are not allowed.")
    if args.max_eval_query_per_class < 0:
        raise ValueError("max_eval_query_per_class must be non-negative.")
    result_names = {
        "multishot_per_seed.csv", "multishot_summary.csv", "support_query_manifest.csv",
        "zero_shot_metadata_manifest.csv", "multishot_config.json",
    }
    existing = sorted(name for name in result_names if os.path.exists(os.path.join(args.output_dir, name)))
    if existing:
        raise FileExistsError(f"Output contains old multishot artifacts {existing}: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.setup(log_dir=args.output_dir, device=None)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.ckpt_path}")
    checkpoint = load_checkpoint(args.ckpt_path)
    checkpoint_step = int(checkpoint.get("step", 0))
    ckpt_step = resolve_checkpoint_step(args.ckpt_step, checkpoint_step, args.ckpt_path)
    if ckpt_step == 0:
        logger.warning(
            "Neither --ckpt_step nor checkpoint metadata records a positive step; results use ckpt_step=0."
        )
    checkpoint_mode = resolve_model_mode(args.model_mode, checkpoint)
    branch_mode = args.branch_mode or checkpoint_mode
    if checkpoint_mode != "dual" and branch_mode != checkpoint_mode:
        raise ValueError(f"A {checkpoint_mode} checkpoint can only use branch_mode={checkpoint_mode}.")
    stats = None
    if checkpoint_mode != "rgb-only":
        args.freq_stats_path = args.freq_stats_path or checkpoint_freq_stats_path(checkpoint)
        if not args.freq_stats_path or not os.path.isfile(args.freq_stats_path):
            raise FileNotFoundError(f"Missing checkpoint-specific freq_stats.pt: {args.freq_stats_path!r}")
        stats = load_frequency_stats(args.freq_stats_path)
    else:
        args.freq_stats_path = ""
    model = DDFSDDualDomainNet(pretrained=args.pretrained, model_mode=checkpoint_mode)
    if stats is not None:
        model.set_freq_stats(stats["mean"], stats["std"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    real_ds = load_ddfsd_class_dataset(args.data_root, "real", "val")
    fake_ds = load_ddfsd_class_dataset(args.data_root, args.exclude_class, "val")
    real_cache = encode_dataset_indices(model, real_ds, range(len(real_ds)), args.eval_batch_size,
                                        args.num_workers, device, args.use_fp16)
    fake_cache = encode_dataset_indices(model, fake_ds, range(len(fake_ds)), args.eval_batch_size,
                                        args.num_workers, device, args.use_fp16)

    metadata_names = ["real"] + get_train_fake_classes(args.exclude_class)
    metadata_datasets, metadata_seed_indices, metadata_caches, union_positions = {}, {}, {}, {}
    if 0 in shots:
        for class_offset, name in enumerate(metadata_names):
            ds = load_ddfsd_class_dataset(args.data_root, name, "train")
            metadata_datasets[name] = ds
            per_seed = {seed: sample_metadata_indices(len(ds), args.zero_shot_metadata_per_class,
                                                      seed + 1000003 * class_offset)
                        for seed in seeds}
            metadata_seed_indices[name] = per_seed
            union = sorted({index for selected in per_seed.values() for index in selected})
            position = {index: offset for offset, index in enumerate(union)}
            union_positions[name] = position
            metadata_caches[name] = encode_dataset_indices(model, ds, union, args.eval_batch_size,
                                                            args.num_workers, device, args.use_fp16)

    per_seed_rows, sq_rows, metadata_rows = [], [], []
    for seed in seeds:
        real_candidates, real_query = build_multishot_support_query_indices(len(real_ds), max_shot, seed)
        fake_candidates, fake_query = build_multishot_support_query_indices(len(fake_ds), max_shot, seed)
        query_count = min(len(real_query), len(fake_query))
        if args.max_eval_query_per_class > 0:
            query_count = min(query_count, args.max_eval_query_per_class)
        if query_count <= 0:
            raise ValueError(f"Seed {seed} has no balanced fixed query images.")
        real_query, fake_query = real_query[:query_count], fake_query[:query_count]
        sq_rows += _manifest_rows(args.exclude_class, seed, "real", real_ds, real_candidates, real_query)
        sq_rows += _manifest_rows(args.exclude_class, seed, args.exclude_class, fake_ds, fake_candidates, fake_query)
        selections = {}
        if 0 in shots:
            for name in metadata_names:
                selected = metadata_seed_indices[name][seed]
                selections[name] = [union_positions[name][index] for index in selected]
                for rank, index in enumerate(selected, 1):
                    metadata_rows.append({"exclude_class": args.exclude_class, "seed": seed,
                                          "metadata_class": name, "split": "train", "dataset_index": index,
                                          "filepath": metadata_datasets[name].paths[index], "metadata_rank": rank})
        for shot in shots:
            if shot == 0:
                metrics = evaluate_zero_shot_embeddings(
                    real_cache, fake_cache, real_query, fake_query, metadata_caches, selections,
                    device, checkpoint_mode, branch_mode, args.tau, args.tau_r)
            else:
                metrics = evaluate_few_shot_embeddings(
                    real_cache, fake_cache, real_candidates, fake_candidates, real_query, fake_query,
                    shot, device, checkpoint_mode, branch_mode, args.tau, args.tau_r)
            row = {"checkpoint_model_mode": checkpoint_mode, "model_mode": checkpoint_mode,
                   "branch_mode": branch_mode, "exclude_class": args.exclude_class, "seed": seed,
                   "shot": shot, "ckpt_step": ckpt_step, **metrics,
                   "num_real_support": shot, "num_fake_support": shot,
                   "num_real_query": query_count, "num_fake_query": query_count,
                   "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class if shot == 0 else "",
                   "freq_stats_path": args.freq_stats_path, "ckpt_path": args.ckpt_path}
            per_seed_rows.append(row)
            logger.info("seed=%d shot=%d ACC=%.6f AP=%.6f AUC=%.6f", seed, shot,
                        metrics["acc"], metrics["ap"], metrics["auc"])

    summary_rows = []
    for shot in shots:
        group = [row for row in per_seed_rows if row["shot"] == shot]
        row = {"checkpoint_model_mode": checkpoint_mode, "model_mode": checkpoint_mode,
               "branch_mode": branch_mode, "exclude_class": args.exclude_class, "shot": shot,
               "ckpt_step": ckpt_step, "eval_seeds": ",".join(map(str, seeds)),
               "freq_stats_path": args.freq_stats_path, "ckpt_path": args.ckpt_path}
        for metric in ("acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = _mean_std(group, metric)
        alpha = [r for r in group if r["alpha_mean"] != ""]
        row["alpha_mean"] = statistics.mean(float(r["alpha_mean"]) for r in alpha) if alpha else ""
        row["alpha_min"] = min(float(r["alpha_min"]) for r in alpha) if alpha else ""
        row["alpha_max"] = max(float(r["alpha_max"]) for r in alpha) if alpha else ""
        summary_rows.append(row)

    per_fields = ["checkpoint_model_mode", "model_mode", "branch_mode", "exclude_class", "seed", "shot",
                  "ckpt_step", "acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc", "num_real_support", "num_fake_support",
                  "num_real_query", "num_fake_query", "alpha_mean", "alpha_min", "alpha_max",
                  "zero_shot_metadata_per_class", "freq_stats_path", "ckpt_path"]
    summary_fields = ["checkpoint_model_mode", "model_mode", "branch_mode", "exclude_class", "shot",
                      "ckpt_step", "acc_mean", "acc_std", "real_acc_mean", "real_acc_std",
                      "fake_acc_mean", "fake_acc_std", "balanced_acc_mean", "balanced_acc_std",
                      "ap_mean", "ap_std", "auc_mean", "auc_std",
                      "alpha_mean", "alpha_min", "alpha_max", "eval_seeds", "freq_stats_path", "ckpt_path"]
    write_csv(os.path.join(args.output_dir, "multishot_per_seed.csv"), per_seed_rows, per_fields)
    write_csv(os.path.join(args.output_dir, "multishot_summary.csv"), summary_rows, summary_fields)
    write_csv(os.path.join(args.output_dir, "support_query_manifest.csv"), sq_rows,
              ["exclude_class", "seed", "data_class", "split", "dataset_index", "filepath", "role", "support_rank"])
    write_csv(os.path.join(args.output_dir, "zero_shot_metadata_manifest.csv"), metadata_rows,
              ["exclude_class", "seed", "metadata_class", "split", "dataset_index", "filepath", "metadata_rank"])
    config = {"exclude_class": args.exclude_class, "shot_list": shots, "seeds": seeds,
              "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class,
              "max_shot": max_shot, "max_eval_query_per_class": args.max_eval_query_per_class,
              "data_root": os.path.abspath(args.data_root), "ckpt_path": os.path.abspath(args.ckpt_path),
              "ckpt_step": ckpt_step, "freq_stats_path": os.path.abspath(args.freq_stats_path) if args.freq_stats_path else "",
              "checkpoint_model_mode": checkpoint_mode, "model_mode": checkpoint_mode, "branch_mode": branch_mode,
              "tau": args.tau, "tau_r": args.tau_r, "git_commit": _git_commit(),
              "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(args.output_dir, "multishot_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
