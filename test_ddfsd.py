# -*- coding: utf-8 -*-
"""Formal DDFSD 10-shot / 5-seed evaluation."""

import argparse
import csv
import os
import statistics


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
    parser.add_argument("--model_mode", type=str, default="auto", choices=["auto", "dual", "rgb-only", "freq-only"])
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
    global evaluate_binary_few_shot, load_frequency_stats, set_seed

    import torch

    import util.logger as logger
    from datasets.ddfsd_datasets import validate_generator_name
    from model.ddfsd import DDFSDDualDomainNet
    from util.ddfsd_eval import evaluate_binary_few_shot
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed


def load_checkpoint(path: str):
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


def write_csv(path: str, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    load_runtime_dependencies()
    validate_generator_name(args.exclude_class)
    if args.seed is not None:
        set_seed(args.seed)

    logger.setup(log_dir=args.output_dir, device=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.ckpt_path)
    model_mode = resolve_model_mode(args.model_mode, checkpoint)
    stats = None
    if model_mode != "rgb-only":
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

    model = DDFSDDualDomainNet(pretrained=args.pretrained, model_mode=model_mode)
    if stats is not None:
        model.set_freq_stats(stats["mean"], stats["std"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    ckpt_step = args.ckpt_step or int(checkpoint.get("step", 0))
    seeds = parse_seed_list(args.eval_seeds)
    if args.eval_repeats > 0:
        seeds = seeds[: args.eval_repeats]
    if not seeds:
        raise ValueError("No eval seeds were provided.")

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
            branch_mode=model_mode,
        )
        row = {
            "model_mode": model_mode,
            "branch_mode": model_mode,
            "exclude_class": args.exclude_class,
            "seed": seed,
            "support_shot": args.num_support_test,
            "split": "val",
            "ckpt_step": ckpt_step,
            "acc": metrics["acc"],
            "ap": metrics["ap"],
            "auc": metrics["auc"],
            "num_real_support": metrics["num_real_support"],
            "num_fake_support": metrics["num_fake_support"],
            "num_real_query": metrics["num_real_query"],
            "num_fake_query": metrics["num_fake_query"],
            "freq_stats_path": args.freq_stats_path,
            "ckpt_path": args.ckpt_path,
        }
        per_seed_rows.append(row)
        logger.info(
            "DDFSD eval seed %d: ACC %.6f AP %.6f AUC %.6f",
            seed,
            metrics["acc"],
            metrics["ap"],
            metrics["auc"],
        )

    def mean_std(key):
        values = [float(row[key]) for row in per_seed_rows]
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return statistics.mean(values), std

    acc_mean, acc_std = mean_std("acc")
    ap_mean, ap_std = mean_std("ap")
    auc_mean, auc_std = mean_std("auc")
    summary_rows = [
        {
            "model_mode": model_mode,
            "branch_mode": model_mode,
            "exclude_class": args.exclude_class,
            "support_shot": args.num_support_test,
            "ckpt_step": ckpt_step,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "ap_mean": ap_mean,
            "ap_std": ap_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
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
            "model_mode",
            "branch_mode",
            "exclude_class",
            "seed",
            "support_shot",
            "split",
            "ckpt_step",
            "acc",
            "ap",
            "auc",
            "num_real_support",
            "num_fake_support",
            "num_real_query",
            "num_fake_query",
            "freq_stats_path",
            "ckpt_path",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "model_mode",
            "branch_mode",
            "exclude_class",
            "support_shot",
            "ckpt_step",
            "acc_mean",
            "acc_std",
            "ap_mean",
            "ap_std",
            "auc_mean",
            "auc_std",
            "eval_seeds",
            "freq_stats_path",
            "ckpt_path",
        ],
    )
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
