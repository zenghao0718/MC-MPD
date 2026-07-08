# -*- coding: utf-8 -*-
"""Generic diagnostic evaluation for DDFSD: dual / rgb-only / freq-only branch modes.

This is a standalone diagnostic script (generalization of tools/eval_ddfsd_branch_modes_adm.py
to any exclude_class). It does not modify train_ddfsd.py, test_ddfsd.py, or the shared
evaluation helpers in util/ddfsd_eval.py; it only reuses existing building blocks:

- datasets.ddfsd_datasets: load_ddfsd_class_dataset, make_subset_loader, sample_support_query_indices
- model.ddfsd_losses: compute_prototypes, compute_support_sigmas, compute_alpha, compute_query_logits
- util.ddfsd_eval: encode_batch, binary_metrics

For a fixed (checkpoint, seed) pair, the exact same support/query split (drawn once,
deterministically from the seed) is reused across all three branch modes so that
dual / rgb-only / freq-only results are directly comparable. Support-only sigma/alpha
statistics are computed once per (checkpoint, seed) and attached to every branch-mode row
(they do not depend on branch_mode, only on the support set).

No margin loss and no branch dropout are used here (this is eval-only code). freq_stats.pt
is required; it is never auto-computed by this script.
"""

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import util.logger as logger
from model.ddfsd import DDFSDDualDomainNet
from util.ddfsd_frequency import load_frequency_stats
from util.utils import set_seed
from datasets.ddfsd_datasets import (
    load_ddfsd_class_dataset,
    make_subset_loader,
    sample_support_query_indices,
    validate_generator_name,
)
from model.ddfsd_losses import (
    compute_prototypes,
    compute_support_sigmas,
    compute_alpha,
    compute_query_logits,
)
from util.ddfsd_eval import encode_batch, binary_metrics


BRANCH_MODES = ("dual", "rgb-only", "freq-only")
SPLIT = "val"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDFSD branch-mode (dual/rgb-only/freq-only) diagnostic evaluation on the excluded class."
    )
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--output_dir", type=str, required=True, help="Experiment output dir (contains ckpt/, freq_stats.pt)")
    parser.add_argument("--exclude_class", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="", help="Defaults to <output_dir>/ckpt")
    parser.add_argument("--ckpt_steps", type=str, default="2500,5000,7500,10000,12500,15000")
    parser.add_argument("--freq_stats_path", type=str, default="")
    parser.add_argument("--num_support_test", type=int, default=10)
    parser.add_argument("--max_eval_query_per_class", type=int, default=0)
    parser.add_argument("--eval_seeds", type=str, default="42,101,102,103,104")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for set_seed(), not the eval support seed.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to write CSVs and logs. Defaults to <output_dir>/branch_modes.",
    )
    return parser.parse_args()


def _stack_dataset_items(dataset, indices):
    return torch.stack([dataset[index][0] for index in indices], dim=0)


@torch.no_grad()
def evaluate_all_branch_modes(
    model,
    real_dataset,
    fake_dataset,
    support_shot: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device,
    use_fp16: bool,
    tau: float,
    tau_r: float,
    max_query_per_class: int = 0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, int]]:
    """Evaluate dual/rgb-only/freq-only on one fixed support/query split.

    Returns (metrics_by_branch_mode, sigma_alpha_stats, support_query_counts).
    """

    model.eval()
    max_query = max_query_per_class if max_query_per_class > 0 else None

    real_support_idx, real_query_idx = sample_support_query_indices(
        len(real_dataset), support_shot=support_shot, seed=seed, max_query=max_query
    )
    fake_support_idx, fake_query_idx = sample_support_query_indices(
        len(fake_dataset), support_shot=support_shot, seed=seed, max_query=max_query
    )

    support_images = torch.cat(
        [
            _stack_dataset_items(real_dataset, real_support_idx),
            _stack_dataset_items(fake_dataset, fake_support_idx),
        ],
        dim=0,
    )
    support_rgb_flat, support_freq_flat = encode_batch(model, support_images, device, use_fp16)
    support_rgb = support_rgb_flat.reshape(2, support_shot, -1).permute(1, 0, 2).unsqueeze(0)
    support_freq = support_freq_flat.reshape(2, support_shot, -1).permute(1, 0, 2).unsqueeze(0)

    proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq)
    sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
    alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)

    all_labels: List[int] = []
    query_rgb_chunks = []
    query_freq_chunks = []
    for label, dataset, query_idx in (
        (0, real_dataset, real_query_idx),
        (1, fake_dataset, fake_query_idx),
    ):
        loader = make_subset_loader(
            dataset,
            query_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for images, _ in loader:
            query_rgb, query_freq = encode_batch(model, images, device, use_fp16)
            query_rgb_chunks.append(query_rgb)
            query_freq_chunks.append(query_freq)
            all_labels.extend([label] * query_rgb.shape[0])

    query_rgb_all = torch.cat(query_rgb_chunks, dim=0).unsqueeze(0)
    query_freq_all = torch.cat(query_freq_chunks, dim=0).unsqueeze(0)

    metrics_by_branch_mode: Dict[str, Dict[str, float]] = {}
    for branch_mode in BRANCH_MODES:
        logits = compute_query_logits(
            query_rgb=query_rgb_all,
            query_freq=query_freq_all,
            proto_rgb=proto_rgb,
            proto_freq=proto_freq,
            alpha=alpha,
            tau=tau,
            branch_mode=branch_mode,
        )["logits"].squeeze(0)
        prob = logits.softmax(dim=-1)
        scores = prob[:, 1].detach().cpu().tolist()
        preds = prob.argmax(dim=-1).detach().cpu().tolist()
        metrics_by_branch_mode[branch_mode] = binary_metrics(all_labels, scores, preds)

    sigma_alpha_stats = {
        "sigma_rgb_mean": float(sigma_rgb.mean().item()),
        "sigma_rgb_min": float(sigma_rgb.min().item()),
        "sigma_rgb_max": float(sigma_rgb.max().item()),
        "sigma_freq_mean": float(sigma_freq.mean().item()),
        "sigma_freq_min": float(sigma_freq.min().item()),
        "sigma_freq_max": float(sigma_freq.max().item()),
        "adaptive_alpha_mean": float(alpha.mean().item()),
        "adaptive_alpha_min": float(alpha.min().item()),
        "adaptive_alpha_max": float(alpha.max().item()),
    }
    sigma_alpha_stats["sigma_diff_mean"] = sigma_alpha_stats["sigma_freq_mean"] - sigma_alpha_stats["sigma_rgb_mean"]

    counts = {
        "num_real_support": len(real_support_idx),
        "num_fake_support": len(fake_support_idx),
        "num_real_query": len(real_query_idx),
        "num_fake_query": len(fake_query_idx),
    }
    return metrics_by_branch_mode, sigma_alpha_stats, counts


def load_checkpoint(path: str, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"DDFSD checkpoint has no 'model' key: {path}")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def write_csv(path: str, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


PER_SEED_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "branch_mode",
    "seed",
    "support_shot",
    "split",
    "acc",
    "ap",
    "auc",
    "num_real_support",
    "num_fake_support",
    "num_real_query",
    "num_fake_query",
    "sigma_rgb_mean",
    "sigma_rgb_min",
    "sigma_rgb_max",
    "sigma_freq_mean",
    "sigma_freq_min",
    "sigma_freq_max",
    "sigma_diff_mean",
    "adaptive_alpha_mean",
    "adaptive_alpha_min",
    "adaptive_alpha_max",
    "ckpt_path",
    "freq_stats_path",
]

SUMMARY_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "branch_mode",
    "support_shot",
    "acc_mean",
    "acc_std",
    "ap_mean",
    "ap_std",
    "auc_mean",
    "auc_std",
    "sigma_rgb_mean",
    "sigma_freq_mean",
    "sigma_diff_mean",
    "adaptive_alpha_mean",
    "adaptive_alpha_min",
    "adaptive_alpha_max",
    "eval_seeds",
    "ckpt_path",
    "freq_stats_path",
]


def mean_std(values: List[float]) -> Tuple[float, float]:
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def main():
    args = parse_args()
    validate_generator_name(args.exclude_class)
    if args.seed is not None:
        set_seed(args.seed)

    ckpt_dir = args.ckpt_dir or os.path.join(args.output_dir, "ckpt")
    freq_stats_path = args.freq_stats_path or os.path.join(args.output_dir, "freq_stats.pt")
    out_dir = args.out_dir or os.path.join(args.output_dir, "branch_modes")
    os.makedirs(out_dir, exist_ok=True)

    logger.setup(log_dir=out_dir, device=None)

    if not os.path.exists(freq_stats_path):
        raise FileNotFoundError(f"Missing freq_stats.pt for evaluation: {freq_stats_path}")

    ckpt_steps = parse_int_list(args.ckpt_steps)
    seeds = parse_int_list(args.eval_seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_frequency_stats(freq_stats_path)
    model = DDFSDDualDomainNet(pretrained=args.pretrained)
    model.set_freq_stats(stats["mean"], stats["std"])
    model = model.to(device)
    model.eval()

    real_dataset = load_ddfsd_class_dataset(args.data_root, "real", SPLIT)
    fake_dataset = load_ddfsd_class_dataset(args.data_root, args.exclude_class, SPLIT)

    per_seed_rows = []
    missing_ckpt_steps = []
    evaluated_ckpt_steps = []

    for step in ckpt_steps:
        ckpt_path = os.path.join(ckpt_dir, f"ddfsd_step[{step}].pth")
        if not os.path.exists(ckpt_path):
            logger.error("Checkpoint missing for step %d: %s (skipped)", step, ckpt_path)
            missing_ckpt_steps.append(step)
            continue

        load_checkpoint(ckpt_path, model)
        model = model.to(device)
        model.eval()
        evaluated_ckpt_steps.append(step)

        for seed in seeds:
            metrics_by_branch_mode, sigma_alpha_stats, counts = evaluate_all_branch_modes(
                model=model,
                real_dataset=real_dataset,
                fake_dataset=fake_dataset,
                support_shot=args.num_support_test,
                seed=seed,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=device,
                use_fp16=args.use_fp16,
                tau=args.tau,
                tau_r=args.tau_r,
                max_query_per_class=args.max_eval_query_per_class,
            )
            for branch_mode in BRANCH_MODES:
                metrics = metrics_by_branch_mode[branch_mode]
                row = {
                    "exclude_class": args.exclude_class,
                    "ckpt_step": step,
                    "branch_mode": branch_mode,
                    "seed": seed,
                    "support_shot": args.num_support_test,
                    "split": SPLIT,
                    "acc": metrics["acc"],
                    "ap": metrics["ap"],
                    "auc": metrics["auc"],
                    **counts,
                    **sigma_alpha_stats,
                    "ckpt_path": ckpt_path,
                    "freq_stats_path": freq_stats_path,
                }
                per_seed_rows.append(row)
                logger.info(
                    "step=%d branch_mode=%-9s seed=%d ACC %.6f AP %.6f AUC %.6f "
                    "sigma_rgb_mean %.6f sigma_freq_mean %.6f adaptive_alpha_mean %.6f",
                    step,
                    branch_mode,
                    seed,
                    metrics["acc"],
                    metrics["ap"],
                    metrics["auc"],
                    sigma_alpha_stats["sigma_rgb_mean"],
                    sigma_alpha_stats["sigma_freq_mean"],
                    sigma_alpha_stats["adaptive_alpha_mean"],
                )

    if missing_ckpt_steps:
        logger.error("Missing checkpoints (skipped): %s", missing_ckpt_steps)

    summary_rows = []
    for step in evaluated_ckpt_steps:
        for branch_mode in BRANCH_MODES:
            rows = [r for r in per_seed_rows if r["ckpt_step"] == step and r["branch_mode"] == branch_mode]
            if not rows:
                continue
            acc_mean, acc_std = mean_std([r["acc"] for r in rows])
            ap_mean, ap_std = mean_std([r["ap"] for r in rows])
            auc_mean, auc_std = mean_std([r["auc"] for r in rows])
            sigma_rgb_mean = statistics.mean([r["sigma_rgb_mean"] for r in rows])
            sigma_freq_mean = statistics.mean([r["sigma_freq_mean"] for r in rows])
            sigma_diff_mean = statistics.mean([r["sigma_diff_mean"] for r in rows])
            adaptive_alpha_mean = statistics.mean([r["adaptive_alpha_mean"] for r in rows])
            adaptive_alpha_min = min([r["adaptive_alpha_min"] for r in rows])
            adaptive_alpha_max = max([r["adaptive_alpha_max"] for r in rows])
            summary_rows.append(
                {
                    "exclude_class": args.exclude_class,
                    "ckpt_step": step,
                    "branch_mode": branch_mode,
                    "support_shot": args.num_support_test,
                    "acc_mean": acc_mean,
                    "acc_std": acc_std,
                    "ap_mean": ap_mean,
                    "ap_std": ap_std,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                    "sigma_rgb_mean": sigma_rgb_mean,
                    "sigma_freq_mean": sigma_freq_mean,
                    "sigma_diff_mean": sigma_diff_mean,
                    "adaptive_alpha_mean": adaptive_alpha_mean,
                    "adaptive_alpha_min": adaptive_alpha_min,
                    "adaptive_alpha_max": adaptive_alpha_max,
                    "eval_seeds": ",".join(str(s) for s in seeds),
                    "ckpt_path": rows[0]["ckpt_path"],
                    "freq_stats_path": freq_stats_path,
                }
            )

    exclude_tag = args.exclude_class
    per_seed_path = os.path.join(out_dir, f"ddfsd_{exclude_tag}_branch_modes_per_seed.csv")
    summary_path = os.path.join(out_dir, f"ddfsd_{exclude_tag}_branch_modes_summary.csv")
    write_csv(per_seed_path, per_seed_rows, PER_SEED_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    logger.info("Saved per-seed CSV: %s", per_seed_path)
    logger.info("Saved summary CSV: %s", summary_path)

    print(f"\n=== DDFSD {exclude_tag} branch-mode diagnosis summary ===")
    header = f"{'step':>6} | {'mode':>9} | {'ACC mean±std':>16} | {'AP mean±std':>16} | {'AUC mean±std':>16} | {'sigma_rgb':>10} | {'sigma_freq':>10} | {'adaptive_alpha':>14}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['ckpt_step']:>6} | {row['branch_mode']:>9} | "
            f"{row['acc_mean']:.4f}±{row['acc_std']:.4f} | "
            f"{row['ap_mean']:.4f}±{row['ap_std']:.4f} | "
            f"{row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"{row['sigma_rgb_mean']:>10.4f} | {row['sigma_freq_mean']:>10.4f} | {row['adaptive_alpha_mean']:>14.4f}"
        )

    if missing_ckpt_steps:
        print(f"\n缺失 checkpoint（未评测）: {missing_ckpt_steps}")

    def get_metric(step, mode, key):
        for row in summary_rows:
            if row["ckpt_step"] == step and row["branch_mode"] == mode:
                return row[key]
        return None

    freq_weaker_count = 0
    dual_better_count = 0
    sigma_freq_gt_count = 0
    alpha_near_075_count = 0
    total_steps = len(evaluated_ckpt_steps)

    for step in evaluated_ckpt_steps:
        auc_rgb = get_metric(step, "rgb-only", "auc_mean")
        auc_freq = get_metric(step, "freq-only", "auc_mean")
        auc_dual = get_metric(step, "dual", "auc_mean")
        sigma_rgb = get_metric(step, "dual", "sigma_rgb_mean")
        sigma_freq = get_metric(step, "dual", "sigma_freq_mean")
        alpha_mean = get_metric(step, "dual", "adaptive_alpha_mean")
        if auc_freq is not None and auc_rgb is not None and auc_freq < auc_rgb:
            freq_weaker_count += 1
        if auc_dual is not None and auc_rgb is not None and auc_dual > auc_rgb:
            dual_better_count += 1
        if sigma_freq is not None and sigma_rgb is not None and sigma_freq > sigma_rgb:
            sigma_freq_gt_count += 1
        if alpha_mean is not None and abs(alpha_mean - 0.75) < 0.02:
            alpha_near_075_count += 1

    print("\n=== 关键判断 ===")
    print(
        f"1. freq-only 是否明显弱于 rgb-only（按AUC）: "
        f"{freq_weaker_count}/{total_steps} 个 checkpoint 上 freq-only < rgb-only"
    )
    print(
        f"2. dual 是否超过 rgb-only（按AUC）: "
        f"{dual_better_count}/{total_steps} 个 checkpoint 上 dual > rgb-only"
    )
    print(
        f"3. sigma_freq 是否长期大于 sigma_rgb: "
        f"{sigma_freq_gt_count}/{total_steps} 个 checkpoint 上 sigma_freq_mean > sigma_rgb_mean"
    )
    print(
        f"4. adaptive alpha 是否长期接近上限0.75（即长期偏向RGB）: "
        f"{alpha_near_075_count}/{total_steps} 个 checkpoint 上 adaptive_alpha_mean 在 [0.73, 0.77] 区间内"
    )


if __name__ == "__main__":
    main()
