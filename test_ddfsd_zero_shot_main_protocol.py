# -*- coding: utf-8 -*-
"""Formal zero-shot DDFSD evaluation using main-experiment-compatible val queries."""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

from test_ddfsd import (
    checkpoint_freq_stats_path,
    git_commit,
    load_checkpoint,
    parse_seed_list,
    resolve_model_mode,
    str2bool,
    write_csv,
)
from util.ddfsd_main_protocol import (
    build_valid_image_indices,
    build_zero_shot_query_indices,
    sample_metadata_from_valid_indices,
    zero_shot_metadata_classes,
)
from util.ddfsd_multishot_logic import resolve_checkpoint_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DDFSD zero-shot main protocol"
    )
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
    parser.add_argument(
        "--model_mode",
        default="auto",
        choices=["auto", "dual", "rgb-only", "freq-only"],
    )
    parser.add_argument(
        "--branch_mode", default=None, choices=["dual", "rgb-only", "freq-only"]
    )
    parser.add_argument("--freq_stats_path", default="")
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--eval_seeds", default="42,101,102,103,104")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--zero_shot_metadata_per_class", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    return parser.parse_args()


def mean_std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.mean(values), (
        statistics.pstdev(values) if len(values) > 1 else 0.0
    )


def main():
    args = parse_args()

    # Runtime-heavy imports remain lazy so this file can be syntax-checked locally.
    import torch
    import util.logger as logger
    from datasets.ddfsd_datasets import (
        load_ddfsd_class_dataset,
        validate_generator_name,
    )
    from model.ddfsd import DDFSDDualDomainNet
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed
    from util.ddfsd_zero_shot_main_protocol_eval import (
        encode_dataset_indices,
        evaluate_zero_shot_embeddings,
    )

    validate_generator_name(args.exclude_class)
    metadata_names = zero_shot_metadata_classes(args.exclude_class)
    seeds = parse_seed_list(args.eval_seeds)
    seeds = seeds[: args.eval_repeats] if args.eval_repeats > 0 else seeds
    if not seeds:
        raise ValueError("No eval seeds were provided.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate eval seeds are not allowed.")
    if args.zero_shot_metadata_per_class <= 0:
        raise ValueError("zero_shot_metadata_per_class must be positive.")
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.ckpt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setup(log_dir=args.output_dir, device=None)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.ckpt_path)
    ckpt_step = resolve_checkpoint_step(
        args.ckpt_step, int(checkpoint.get("step", 0)), args.ckpt_path
    )
    checkpoint_mode = resolve_model_mode(args.model_mode, checkpoint)
    branch_mode = args.branch_mode or checkpoint_mode
    if checkpoint_mode != "dual" and branch_mode != checkpoint_mode:
        raise ValueError(
            f"A {checkpoint_mode} checkpoint can only use branch_mode={checkpoint_mode}."
        )

    stats = None
    if checkpoint_mode != "rgb-only":
        args.freq_stats_path = args.freq_stats_path or checkpoint_freq_stats_path(
            checkpoint
        )
        if not args.freq_stats_path or not os.path.isfile(args.freq_stats_path):
            raise FileNotFoundError(
                f"Missing checkpoint-specific freq_stats.pt: {args.freq_stats_path!r}"
            )
        stats = load_frequency_stats(args.freq_stats_path)
    else:
        args.freq_stats_path = ""

    model = DDFSDDualDomainNet(pretrained=args.pretrained, model_mode=checkpoint_mode)
    if stats is not None:
        model.set_freq_stats(stats["mean"], stats["std"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    real_dataset = load_ddfsd_class_dataset(args.data_root, "real", "val")
    fake_dataset = load_ddfsd_class_dataset(args.data_root, args.exclude_class, "val")
    real_query, fake_query = build_zero_shot_query_indices(
        len(real_dataset), len(fake_dataset)
    )
    real_cache = encode_dataset_indices(
        model,
        real_dataset,
        real_query,
        args.eval_batch_size,
        args.num_workers,
        device,
        args.use_fp16,
    )
    fake_cache = encode_dataset_indices(
        model,
        fake_dataset,
        fake_query,
        args.eval_batch_size,
        args.num_workers,
        device,
        args.use_fp16,
    )

    metadata_datasets = {}
    metadata_indices = {}
    metadata_positions = {}
    metadata_caches = {}
    metadata_manifest_rows = []
    invalid_rows = []
    invalid_fields = [
        "exclude_class",
        "metadata_class",
        "split",
        "dataset_index",
        "filepath",
        "error_type",
        "error",
    ]
    for name in metadata_names:
        dataset = load_ddfsd_class_dataset(args.data_root, name, "train")
        class_invalid = []
        valid_indices = build_valid_image_indices(dataset.paths, class_invalid)
        for record in class_invalid:
            invalid_rows.append(
                {
                    "exclude_class": args.exclude_class,
                    "metadata_class": name,
                    "split": "train",
                    **record,
                }
            )
        # Persist the audit incrementally so an insufficient-valid-images error
        # still leaves the paths that caused it available for diagnosis.
        write_csv(
            os.path.join(args.output_dir, "zero_shot_invalid_images.csv"),
            invalid_rows,
            invalid_fields,
        )
        per_seed = {
            seed: sample_metadata_from_valid_indices(
                valid_indices, args.zero_shot_metadata_per_class, seed
            )
            for seed in seeds
        }
        union = sorted({index for selected in per_seed.values() for index in selected})
        metadata_datasets[name] = dataset
        metadata_indices[name] = per_seed
        metadata_positions[name] = {
            index: position for position, index in enumerate(union)
        }
        metadata_caches[name] = encode_dataset_indices(
            model,
            dataset,
            union,
            args.eval_batch_size,
            args.num_workers,
            device,
            args.use_fp16,
        )

    per_seed_rows = []
    for seed in seeds:
        selections = {}
        for name in metadata_names:
            selected = metadata_indices[name][seed]
            selections[name] = [metadata_positions[name][index] for index in selected]
            for rank, index in enumerate(selected, 1):
                metadata_manifest_rows.append(
                    {
                        "exclude_class": args.exclude_class,
                        "seed": seed,
                        "metadata_class": name,
                        "split": "train",
                        "dataset_index": index,
                        "filepath": metadata_datasets[name].paths[index],
                        "metadata_rank": rank,
                    }
                )

        metrics = evaluate_zero_shot_embeddings(
            real_cache,
            fake_cache,
            real_query,
            fake_query,
            metadata_caches,
            selections,
            device,
            checkpoint_mode,
            branch_mode,
            args.tau,
            args.tau_r,
        )
        row = {
            "checkpoint_model_mode": checkpoint_mode,
            "model_mode": checkpoint_mode,
            "branch_mode": branch_mode,
            "exclude_class": args.exclude_class,
            "seed": seed,
            "shot": 0,
            "ckpt_step": ckpt_step,
            "acc": metrics["acc"],
            "real_acc": metrics["real_acc"],
            "fake_acc": metrics["fake_acc"],
            "balanced_acc": metrics["balanced_acc"],
            "ap": metrics["ap"],
            "auc": metrics["auc"],
            "num_real_support": 0,
            "num_fake_support": 0,
            "num_real_query": len(real_query),
            "num_fake_query": len(fake_query),
            "alpha_mean": metrics["alpha_mean"],
            "alpha_min": metrics["alpha_min"],
            "alpha_max": metrics["alpha_max"],
            "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class,
            "freq_stats_path": args.freq_stats_path,
            "ckpt_path": args.ckpt_path,
        }
        per_seed_rows.append(row)
        logger.info(
            "zero-shot seed=%d ACC=%.6f Real ACC=%.6f Fake ACC=%.6f "
            "Balanced ACC=%.6f AP=%.6f AUC=%.6f",
            seed,
            row["acc"],
            row["real_acc"],
            row["fake_acc"],
            row["balanced_acc"],
            row["ap"],
            row["auc"],
        )

    summary = {
        "checkpoint_model_mode": checkpoint_mode,
        "model_mode": checkpoint_mode,
        "branch_mode": branch_mode,
        "exclude_class": args.exclude_class,
        "shot": 0,
        "ckpt_step": ckpt_step,
        "eval_seeds": ",".join(map(str, seeds)),
        "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class,
        "freq_stats_path": args.freq_stats_path,
        "ckpt_path": args.ckpt_path,
    }
    for metric in ("acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc"):
        summary[f"{metric}_mean"], summary[f"{metric}_std"] = mean_std(
            per_seed_rows, metric
        )
    alpha_rows = [row for row in per_seed_rows if row["alpha_mean"] != ""]
    summary["alpha_mean"] = (
        statistics.mean(float(row["alpha_mean"]) for row in alpha_rows)
        if alpha_rows
        else ""
    )
    summary["alpha_min"] = (
        min(float(row["alpha_min"]) for row in alpha_rows) if alpha_rows else ""
    )
    summary["alpha_max"] = (
        max(float(row["alpha_max"]) for row in alpha_rows) if alpha_rows else ""
    )

    per_fields = [
        "checkpoint_model_mode",
        "model_mode",
        "branch_mode",
        "exclude_class",
        "seed",
        "shot",
        "ckpt_step",
        "acc",
        "real_acc",
        "fake_acc",
        "balanced_acc",
        "ap",
        "auc",
        "num_real_support",
        "num_fake_support",
        "num_real_query",
        "num_fake_query",
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "zero_shot_metadata_per_class",
        "freq_stats_path",
        "ckpt_path",
    ]
    summary_fields = [
        "checkpoint_model_mode",
        "model_mode",
        "branch_mode",
        "exclude_class",
        "shot",
        "ckpt_step",
        "acc_mean",
        "acc_std",
        "real_acc_mean",
        "real_acc_std",
        "fake_acc_mean",
        "fake_acc_std",
        "balanced_acc_mean",
        "balanced_acc_std",
        "ap_mean",
        "ap_std",
        "auc_mean",
        "auc_std",
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "eval_seeds",
        "zero_shot_metadata_per_class",
        "freq_stats_path",
        "ckpt_path",
    ]
    write_csv(
        os.path.join(args.output_dir, "ddfsd_zero_shot_per_seed.csv"),
        per_seed_rows,
        per_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "ddfsd_zero_shot_summary.csv"),
        [summary],
        summary_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "zero_shot_metadata_manifest.csv"),
        metadata_manifest_rows,
        [
            "exclude_class",
            "seed",
            "metadata_class",
            "split",
            "dataset_index",
            "filepath",
            "metadata_rank",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "zero_shot_invalid_images.csv"),
        invalid_rows,
        invalid_fields,
    )

    config = {
        "protocol": "main-protocol formal shot ablation",
        "git_commit": git_commit(),
        "command": sys.argv,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exclude_class": args.exclude_class,
        "shot": 0,
        "seeds": seeds,
        "metadata_classes": metadata_names,
        "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class,
        "data_root": os.path.abspath(args.data_root),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "ckpt_step": ckpt_step,
        "freq_stats_path": (
            os.path.abspath(args.freq_stats_path) if args.freq_stats_path else ""
        ),
        "checkpoint_model_mode": checkpoint_mode,
        "model_mode": checkpoint_mode,
        "branch_mode": branch_mode,
        "tau": args.tau,
        "tau_r": args.tau_r,
        "max_eval_query_per_class": 0,
        "num_real_query": len(real_query),
        "num_fake_query": len(fake_query),
    }
    with open(
        os.path.join(args.output_dir, "zero_shot_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
