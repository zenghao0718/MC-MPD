# -*- coding: utf-8 -*-
"""Formal DDFSD 10-shot / 5-seed evaluation."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from PIL import Image

FORMAL_PROTOCOL = "main_full_steps15000_10shot_5seed"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_seed_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDFSD v1")
    parser.add_argument("--model", type=str, default="ddfsd")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output_dir")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--exclude_class", type=str, default="ADM")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--ckpt_step", type=int, default=0)
    parser.add_argument("--model_mode", type=str, default="dual", choices=["dual"])
    parser.add_argument(
        "--branch_mode",
        type=str,
        default="dual",
        choices=["dual"],
        help="The published main evaluation always uses both branches.",
    )
    parser.add_argument("--freq_stats_path", type=str, default="")
    parser.add_argument("--num_support_test", type=int, default=10)
    parser.add_argument("--num_query_test", type=int, default=0)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--eval_seeds", type=str, default="42,101,102,103,104")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_eval_query_per_class", type=int, default=0)

    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    return parser.parse_args()


def load_runtime_dependencies():
    global torch, logger, validate_generator_name, DDFSDDualDomainNet
    global load_ddfsd_class_dataset
    global evaluate_binary_few_shot, load_frequency_stats, set_seed

    import torch

    import util.logger as logger
    from datasets.ddfsd_datasets import (
        load_ddfsd_class_dataset,
        validate_generator_name,
    )
    from model.ddfsd import DDFSDDualDomainNet
    from util.ddfsd_eval import evaluate_binary_few_shot
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed


def load_checkpoint(path: str):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"DDFSD checkpoint has no 'model' key: {path}")
    return checkpoint


def checkpoint_model_mode(checkpoint):
    model_mode = checkpoint.get("model_mode")
    if model_mode:
        return model_mode
    config = checkpoint.get("config", checkpoint.get("args", {}))
    if isinstance(config, dict):
        return config.get("model_mode", "dual")
    return getattr(config, "model_mode", "dual")


def resolve_model_mode(requested_mode, checkpoint):
    checkpoint_mode = checkpoint_model_mode(checkpoint)
    if checkpoint_mode not in {"dual", "rgb-only", "freq-only"}:
        raise ValueError(f"Checkpoint records invalid model_mode '{checkpoint_mode}'.")
    if requested_mode != "auto" and requested_mode != checkpoint_mode:
        raise ValueError(
            f"--model_mode {requested_mode} conflicts with checkpoint model_mode {checkpoint_mode}."
        )
    return checkpoint_mode


def checkpoint_freq_stats_path(checkpoint):
    """Return a checkpoint-recorded frequency-statistics path, if present."""

    freq_stats_path = checkpoint.get("freq_stats_path")
    if freq_stats_path:
        return freq_stats_path
    config = checkpoint.get("config", checkpoint.get("args", {}))
    if isinstance(config, dict):
        return config.get("freq_stats_path", "")
    return getattr(config, "freq_stats_path", "")


def write_csv(path: str, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_checkpoint_step(requested_step: int, checkpoint_step: int, checkpoint_path: str) -> int:
    """Validate the requested step against checkpoint metadata."""

    requested_step = int(requested_step or 0)
    checkpoint_step = int(checkpoint_step or 0)
    if requested_step < 0 or checkpoint_step < 0:
        raise ValueError("Checkpoint steps must be non-negative.")
    if requested_step and checkpoint_step and requested_step != checkpoint_step:
        raise ValueError(
            f"Requested step {requested_step} conflicts with checkpoint-recorded step "
            f"{checkpoint_step}: {checkpoint_path}"
        )
    return requested_step or checkpoint_step


def audit_image_paths(paths, data_class, split):
    """Strictly decode images and return structured failure records."""

    invalid = []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
        except (OSError, ValueError) as error:
            invalid.append(
                {
                    "data_class": data_class,
                    "split": split,
                    "dataset_index": index,
                    "filepath": path,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return invalid


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def audit_formal_val_images(data_root, exclude_class, output_dir):
    rows = []
    for data_class in ("real", exclude_class):
        dataset = load_ddfsd_class_dataset(
            data_root, data_class, "val", strict_images=True
        )
        rows.extend(audit_image_paths(dataset.paths, data_class, "val"))
    path = os.path.join(output_dir, "formal_val_invalid_images.csv")
    write_csv(
        path,
        rows,
        [
            "data_class",
            "split",
            "dataset_index",
            "filepath",
            "error_type",
            "error",
        ],
    )
    if rows:
        raise RuntimeError(
            f"Formal evaluation found {len(rows)} invalid val images; "
            f"audit saved to {path}. No evaluation was run."
        )
    return path


def main():
    args = parse_args()
    load_runtime_dependencies()
    validate_generator_name(args.exclude_class)
    if args.seed is not None:
        set_seed(args.seed)

    logger.setup(log_dir=args.output_dir, device=None)
    val_audit_path = audit_formal_val_images(
        args.data_root, args.exclude_class, args.output_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.ckpt_path)
    model_mode = resolve_model_mode(args.model_mode, checkpoint)
    branch_mode = args.branch_mode
    if model_mode != "dual" and branch_mode != model_mode:
        raise ValueError(
            f"A {model_mode} checkpoint can only be evaluated with --branch_mode {model_mode}; "
            f"requested {branch_mode}."
        )
    stats = None
    if model_mode != "rgb-only":
        if not args.freq_stats_path:
            args.freq_stats_path = checkpoint_freq_stats_path(checkpoint)
        if not args.freq_stats_path:
            args.freq_stats_path = os.path.join(args.output_dir, "freq_stats.pt")
        if not os.path.exists(args.freq_stats_path):
            raise FileNotFoundError(
                f"Missing freq_stats.pt for evaluation: {args.freq_stats_path}. "
                "Test must not auto-compute frequency stats."
            )
        stats = load_frequency_stats(args.freq_stats_path)
    else:
        args.freq_stats_path = ""

    logger.info(
        "DDFSD evaluation: checkpoint_model_mode=%s branch_mode=%s checkpoint=%s output_dir=%s",
        model_mode, branch_mode, args.ckpt_path, args.output_dir,
    )
    model = DDFSDDualDomainNet(pretrained=args.pretrained, model_mode=model_mode)
    if stats is not None:
        model.set_freq_stats(stats["mean"], stats["std"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    ckpt_step = resolve_checkpoint_step(
        args.ckpt_step, int(checkpoint.get("step", 0)), args.ckpt_path
    )
    seeds = parse_seed_list(args.eval_seeds)
    if args.eval_repeats > 0:
        seeds = seeds[: args.eval_repeats]
    if not seeds:
        raise ValueError("No eval seeds were provided.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate eval seeds are not allowed.")
    if args.num_support_test <= 0:
        raise ValueError("test_ddfsd.py main-protocol evaluation requires num_support_test > 0.")
    if args.max_eval_query_per_class < 0:
        raise ValueError("max_eval_query_per_class must be non-negative.")
    per_seed_rows = []
    for seed in seeds:
        metrics = evaluate_binary_few_shot(
            model=model,
            data_root=args.data_root,
            fake_class=args.exclude_class,
            support_shot=args.num_support_test,
            seed=seed,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            use_fp16=args.use_fp16,
            tau=args.tau,
            tau_r=args.tau_r,
            max_query_per_class=args.max_eval_query_per_class,
            model_mode=model_mode,
            branch_mode=branch_mode,
            return_indices=False,
            strict_images=True,
        )
        row = {
            "checkpoint_model_mode": model_mode,
            "model_mode": model_mode,
            "branch_mode": branch_mode,
            "exclude_class": args.exclude_class,
            "seed": seed,
            "support_shot": args.num_support_test,
            "split": "val",
            "ckpt_step": ckpt_step,
            "acc": metrics["acc"],
            "real_acc": metrics["real_acc"],
            "fake_acc": metrics["fake_acc"],
            "balanced_acc": metrics["balanced_acc"],
            "ap": metrics["ap"],
            "auc": metrics["auc"],
            "num_real_support": metrics["num_real_support"],
            "num_fake_support": metrics["num_fake_support"],
            "num_real_query": metrics["num_real_query"],
            "num_fake_query": metrics["num_fake_query"],
            "alpha_mean": metrics["alpha_mean"],
            "alpha_min": metrics["alpha_min"],
            "alpha_max": metrics["alpha_max"],
            "freq_stats_path": args.freq_stats_path,
            "ckpt_path": args.ckpt_path,
        }
        per_seed_rows.append(row)
        logger.info(
            "DDFSD eval checkpoint_model_mode=%s branch_mode=%s seed=%d: "
            "ACC %.6f Real ACC %.6f Fake ACC %.6f Balanced ACC %.6f AP %.6f AUC %.6f",
            model_mode,
            branch_mode,
            seed,
            metrics["acc"],
            metrics["real_acc"],
            metrics["fake_acc"],
            metrics["balanced_acc"],
            metrics["ap"],
            metrics["auc"],
        )

    def mean_std(key):
        values = [float(row[key]) for row in per_seed_rows]
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return statistics.mean(values), std

    acc_mean, acc_std = mean_std("acc")
    real_acc_mean, real_acc_std = mean_std("real_acc")
    fake_acc_mean, fake_acc_std = mean_std("fake_acc")
    balanced_acc_mean, balanced_acc_std = mean_std("balanced_acc")
    ap_mean, ap_std = mean_std("ap")
    auc_mean, auc_std = mean_std("auc")
    alpha_rows = [row for row in per_seed_rows if row["alpha_mean"] != ""]
    summary_rows = [
        {
            "checkpoint_model_mode": model_mode,
            "model_mode": model_mode,
            "branch_mode": branch_mode,
            "exclude_class": args.exclude_class,
            "support_shot": args.num_support_test,
            "ckpt_step": ckpt_step,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "real_acc_mean": real_acc_mean,
            "real_acc_std": real_acc_std,
            "fake_acc_mean": fake_acc_mean,
            "fake_acc_std": fake_acc_std,
            "balanced_acc_mean": balanced_acc_mean,
            "balanced_acc_std": balanced_acc_std,
            "ap_mean": ap_mean,
            "ap_std": ap_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "alpha_mean": (
                statistics.mean(float(row["alpha_mean"]) for row in alpha_rows)
                if alpha_rows
                else ""
            ),
            "alpha_min": (
                min(float(row["alpha_min"]) for row in alpha_rows) if alpha_rows else ""
            ),
            "alpha_max": (
                max(float(row["alpha_max"]) for row in alpha_rows) if alpha_rows else ""
            ),
            "eval_seeds": ",".join(str(seed) for seed in seeds),
            "freq_stats_path": args.freq_stats_path,
            "ckpt_path": args.ckpt_path,
        }
    ]

    per_seed_path = os.path.join(args.output_dir, "ddfsd_eval_per_seed.csv")
    summary_path = os.path.join(args.output_dir, "ddfsd_eval_summary.csv")
    write_csv(
        per_seed_path,
        per_seed_rows,
        [
            "checkpoint_model_mode",
            "model_mode",
            "branch_mode",
            "exclude_class",
            "seed",
            "support_shot",
            "split",
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
            "freq_stats_path",
            "ckpt_path",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "checkpoint_model_mode",
            "model_mode",
            "branch_mode",
            "exclude_class",
            "support_shot",
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
            "freq_stats_path",
            "ckpt_path",
        ],
    )
    config = {
        "protocol": FORMAL_PROTOCOL,
        "git_commit": git_commit(),
        "command": sys.argv,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exclude_class": args.exclude_class,
        "shot": args.num_support_test,
        "seeds": seeds,
        "data_root": os.path.abspath(args.data_root),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "ckpt_step": ckpt_step,
        "freq_stats_path": (
            os.path.abspath(args.freq_stats_path) if args.freq_stats_path else ""
        ),
        "checkpoint_model_mode": model_mode,
        "model_mode": model_mode,
        "branch_mode": branch_mode,
        "tau": args.tau,
        "tau_r": args.tau_r,
        "max_eval_query_per_class": args.max_eval_query_per_class,
        "strict_formal_eval_images": True,
        "formal_val_invalid_images_path": os.path.abspath(val_audit_path),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    logger.info(
        "DDFSD summary: ACC %.6f +/- %.6f AP %.6f +/- %.6f AUC %.6f +/- %.6f",
        acc_mean,
        acc_std,
        ap_mean,
        ap_std,
        auc_mean,
        auc_std,
    )
    logger.info("Saved per-seed CSV: %s", per_seed_path)
    logger.info("Saved summary CSV: %s", summary_path)


if __name__ == "__main__":
    main()
