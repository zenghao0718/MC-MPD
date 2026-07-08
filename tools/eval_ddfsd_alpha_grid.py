# -*- coding: utf-8 -*-
"""Generic DDFSD diagnostic evaluation over an (alpha_mode x branch_mode) grid.

This is a standalone diagnostic script. It does NOT modify train_ddfsd.py, test_ddfsd.py,
or the shared evaluation helpers in util/ddfsd_eval.py; it only reuses existing building
blocks:

- datasets.ddfsd_datasets: load_ddfsd_class_dataset, make_subset_loader, sample_support_query_indices
- model.ddfsd_losses: compute_prototypes, compute_support_sigmas, compute_alpha,
  ordinary_euclidean_distance
- util.ddfsd_eval: encode_batch, binary_metrics
- util.ddfsd_frequency: load_frequency_stats

Test split is always "val". Real label = 0, fake label = 1. AP/AUC use the fake score
(prob[:, 1]). For a fixed (checkpoint, seed), the support/query split is drawn once
(deterministically from the seed) and the resulting rgb/freq embeddings are reused across
every requested (branch_mode, alpha_mode) combination, so results are directly comparable.

No margin loss and no branch dropout are used here (this is eval-only code). freq_stats.pt
is required; it is never auto-computed by this script.
"""

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Tuple, Union

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
    ordinary_euclidean_distance,
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


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_alpha_mode(token: str) -> Union[str, float]:
    token = token.strip()
    if token.lower() == "adaptive":
        return "adaptive"
    return float(token)


def parse_alpha_modes(value: str) -> List[Union[str, float]]:
    return [parse_alpha_mode(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDFSD alpha_mode x branch_mode diagnostic evaluation on the excluded class (val split)."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write CSVs and logs into.")
    parser.add_argument("--exclude_class", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing ddfsd_step[N].pth files.")
    parser.add_argument("--ckpt_steps", type=str, required=True, help="Comma-separated checkpoint steps, e.g. 7500,12500,15000")
    parser.add_argument("--freq_stats_path", type=str, required=True)
    parser.add_argument("--support_shot", type=int, default=10)
    parser.add_argument("--eval_seeds", type=str, default="42,101,102,103,104")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    parser.add_argument(
        "--alpha_modes",
        type=str,
        default="adaptive",
        help="Comma-separated list, e.g. 'adaptive,0.0,0.25,0.5,0.75,1.0'",
    )
    parser.add_argument(
        "--branch_modes",
        type=str,
        default="dual",
        help="Comma-separated subset of {dual, rgb-only, freq-only}",
    )
    parser.add_argument("--max_eval_query_per_class", type=int, default=0)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for set_seed(), not the eval support seed.")
    return parser.parse_args()


def _stack_dataset_items(dataset, indices):
    return torch.stack([dataset[index][0] for index in indices], dim=0)


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


def mean_std(values: List[float]) -> Tuple[float, float]:
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


@torch.no_grad()
def evaluate_one_seed(
    model,
    real_dataset,
    fake_dataset,
    support_shot: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_fp16: bool,
    tau: float,
    tau_r: float,
    branch_modes: List[str],
    alpha_modes: List[Union[str, float]],
    max_query_per_class: int = 0,
) -> Tuple[Dict[Tuple[str, Union[str, float]], Dict[str, float]], Dict[str, float], Dict[str, int]]:
    """Evaluate every (branch_mode, alpha_mode) combo on one fixed support/query split.

    Returns (metrics_by_combo, support_stats, counts), where metrics_by_combo maps
    (branch_mode, alpha_mode) -> {"acc", "ap", "auc", "used_alpha_mean/min/max"}.
    support_stats holds sigma_rgb/sigma_freq/adaptive_alpha statistics that only depend
    on the support set (identical across branch_mode/alpha_mode for the same seed).
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
    adaptive_alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)

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

    rgb_dist = ordinary_euclidean_distance(query_rgb_all, proto_rgb)
    freq_dist = ordinary_euclidean_distance(query_freq_all, proto_freq)

    metrics_by_combo: Dict[Tuple[str, Union[str, float]], Dict[str, float]] = {}
    for branch_mode in branch_modes:
        if branch_mode == "rgb-only":
            final_dist = rgb_dist
            used_alpha = torch.ones_like(adaptive_alpha)
            logits = (-final_dist / tau).squeeze(0)
            prob = logits.softmax(dim=-1)
            scores = prob[:, 1].detach().cpu().tolist()
            preds = prob.argmax(dim=-1).detach().cpu().tolist()
            metrics = binary_metrics(all_labels, scores, preds)
            metrics.update(
                {
                    "used_alpha_mean": float(used_alpha.mean().item()),
                    "used_alpha_min": float(used_alpha.min().item()),
                    "used_alpha_max": float(used_alpha.max().item()),
                }
            )
            for alpha_mode in alpha_modes:
                metrics_by_combo[(branch_mode, alpha_mode)] = metrics
            continue

        if branch_mode == "freq-only":
            final_dist = freq_dist
            used_alpha = torch.zeros_like(adaptive_alpha)
            logits = (-final_dist / tau).squeeze(0)
            prob = logits.softmax(dim=-1)
            scores = prob[:, 1].detach().cpu().tolist()
            preds = prob.argmax(dim=-1).detach().cpu().tolist()
            metrics = binary_metrics(all_labels, scores, preds)
            metrics.update(
                {
                    "used_alpha_mean": float(used_alpha.mean().item()),
                    "used_alpha_min": float(used_alpha.min().item()),
                    "used_alpha_max": float(used_alpha.max().item()),
                }
            )
            for alpha_mode in alpha_modes:
                metrics_by_combo[(branch_mode, alpha_mode)] = metrics
            continue

        # branch_mode == "dual"
        for alpha_mode in alpha_modes:
            if alpha_mode == "adaptive":
                used_alpha = adaptive_alpha
            else:
                used_alpha = torch.full_like(adaptive_alpha, float(alpha_mode))
            alpha_for_dist = used_alpha[:, None, :]
            final_dist = alpha_for_dist * rgb_dist + (1.0 - alpha_for_dist) * freq_dist
            logits = (-final_dist / tau).squeeze(0)
            prob = logits.softmax(dim=-1)
            scores = prob[:, 1].detach().cpu().tolist()
            preds = prob.argmax(dim=-1).detach().cpu().tolist()
            metrics = binary_metrics(all_labels, scores, preds)
            metrics.update(
                {
                    "used_alpha_mean": float(used_alpha.mean().item()),
                    "used_alpha_min": float(used_alpha.min().item()),
                    "used_alpha_max": float(used_alpha.max().item()),
                }
            )
            metrics_by_combo[(branch_mode, alpha_mode)] = metrics

    support_stats = {
        "sigma_rgb_mean": float(sigma_rgb.mean().item()),
        "sigma_freq_mean": float(sigma_freq.mean().item()),
        "adaptive_alpha_mean": float(adaptive_alpha.mean().item()),
        "adaptive_alpha_min": float(adaptive_alpha.min().item()),
        "adaptive_alpha_max": float(adaptive_alpha.max().item()),
    }
    support_stats["sigma_diff_mean"] = support_stats["sigma_freq_mean"] - support_stats["sigma_rgb_mean"]

    counts = {
        "num_real_support": len(real_support_idx),
        "num_fake_support": len(fake_support_idx),
        "num_real_query": len(real_query_idx),
        "num_fake_query": len(fake_query_idx),
    }
    return metrics_by_combo, support_stats, counts


PER_SEED_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "seed",
    "support_shot",
    "split",
    "branch_mode",
    "alpha_mode",
    "fixed_alpha",
    "acc",
    "ap",
    "auc",
    "sigma_rgb_mean",
    "sigma_freq_mean",
    "sigma_diff_mean",
    "adaptive_alpha_mean",
    "adaptive_alpha_min",
    "adaptive_alpha_max",
    "used_alpha_mean",
    "used_alpha_min",
    "used_alpha_max",
    "num_real_support",
    "num_fake_support",
    "num_real_query",
    "num_fake_query",
    "ckpt_path",
    "freq_stats_path",
]

SUMMARY_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "support_shot",
    "branch_mode",
    "alpha_mode",
    "fixed_alpha",
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
    "used_alpha_mean",
    "used_alpha_min",
    "used_alpha_max",
    "eval_seeds",
    "ckpt_path",
    "freq_stats_path",
]


def alpha_mode_label(alpha_mode: Union[str, float]) -> str:
    return "adaptive" if alpha_mode == "adaptive" else f"{float(alpha_mode):g}"


def fixed_alpha_value(alpha_mode: Union[str, float]):
    return "" if alpha_mode == "adaptive" else float(alpha_mode)


def main():
    args = parse_args()
    validate_generator_name(args.exclude_class)
    for branch_mode in parse_str_list(args.branch_modes):
        if branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unknown branch_mode '{branch_mode}', expected one of {BRANCH_MODES}")
    if args.seed is not None:
        set_seed(args.seed)

    branch_modes = parse_str_list(args.branch_modes)
    alpha_modes = parse_alpha_modes(args.alpha_modes)
    ckpt_steps = parse_int_list(args.ckpt_steps)
    seeds = parse_int_list(args.eval_seeds)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setup(log_dir=args.output_dir, device=None)

    if not os.path.exists(args.freq_stats_path):
        raise FileNotFoundError(f"Missing freq_stats.pt for evaluation: {args.freq_stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_frequency_stats(args.freq_stats_path)
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
        ckpt_path = os.path.join(args.ckpt_dir, f"ddfsd_step[{step}].pth")
        if not os.path.exists(ckpt_path):
            logger.error("Checkpoint missing for step %d: %s (skipped)", step, ckpt_path)
            missing_ckpt_steps.append(step)
            continue

        load_checkpoint(ckpt_path, model)
        model = model.to(device)
        model.eval()
        evaluated_ckpt_steps.append(step)

        for seed in seeds:
            metrics_by_combo, support_stats, counts = evaluate_one_seed(
                model=model,
                real_dataset=real_dataset,
                fake_dataset=fake_dataset,
                support_shot=args.support_shot,
                seed=seed,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=device,
                use_fp16=args.use_fp16,
                tau=args.tau,
                tau_r=args.tau_r,
                branch_modes=branch_modes,
                alpha_modes=alpha_modes,
                max_query_per_class=args.max_eval_query_per_class,
            )
            for branch_mode in branch_modes:
                for alpha_mode in alpha_modes:
                    metrics = metrics_by_combo[(branch_mode, alpha_mode)]
                    row = {
                        "exclude_class": args.exclude_class,
                        "ckpt_step": step,
                        "seed": seed,
                        "support_shot": args.support_shot,
                        "split": SPLIT,
                        "branch_mode": branch_mode,
                        "alpha_mode": alpha_mode_label(alpha_mode),
                        "fixed_alpha": fixed_alpha_value(alpha_mode),
                        "acc": metrics["acc"],
                        "ap": metrics["ap"],
                        "auc": metrics["auc"],
                        **support_stats,
                        "used_alpha_mean": metrics["used_alpha_mean"],
                        "used_alpha_min": metrics["used_alpha_min"],
                        "used_alpha_max": metrics["used_alpha_max"],
                        **counts,
                        "ckpt_path": ckpt_path,
                        "freq_stats_path": args.freq_stats_path,
                    }
                    per_seed_rows.append(row)
                    logger.info(
                        "step=%d branch_mode=%-9s alpha_mode=%-8s seed=%d ACC %.6f AP %.6f AUC %.6f "
                        "used_alpha_mean %.6f adaptive_alpha_mean %.6f",
                        step,
                        branch_mode,
                        row["alpha_mode"],
                        seed,
                        metrics["acc"],
                        metrics["ap"],
                        metrics["auc"],
                        metrics["used_alpha_mean"],
                        support_stats["adaptive_alpha_mean"],
                    )

    if missing_ckpt_steps:
        logger.error("Missing checkpoints (skipped): %s", missing_ckpt_steps)

    summary_rows = []
    for step in evaluated_ckpt_steps:
        for branch_mode in branch_modes:
            for alpha_mode in alpha_modes:
                alpha_label = alpha_mode_label(alpha_mode)
                rows = [
                    r
                    for r in per_seed_rows
                    if r["ckpt_step"] == step and r["branch_mode"] == branch_mode and r["alpha_mode"] == alpha_label
                ]
                if not rows:
                    continue
                acc_mean, acc_std = mean_std([r["acc"] for r in rows])
                ap_mean, ap_std = mean_std([r["ap"] for r in rows])
                auc_mean, auc_std = mean_std([r["auc"] for r in rows])
                summary_rows.append(
                    {
                        "exclude_class": args.exclude_class,
                        "ckpt_step": step,
                        "support_shot": args.support_shot,
                        "branch_mode": branch_mode,
                        "alpha_mode": alpha_label,
                        "fixed_alpha": fixed_alpha_value(alpha_mode),
                        "acc_mean": acc_mean,
                        "acc_std": acc_std,
                        "ap_mean": ap_mean,
                        "ap_std": ap_std,
                        "auc_mean": auc_mean,
                        "auc_std": auc_std,
                        "sigma_rgb_mean": statistics.mean([r["sigma_rgb_mean"] for r in rows]),
                        "sigma_freq_mean": statistics.mean([r["sigma_freq_mean"] for r in rows]),
                        "sigma_diff_mean": statistics.mean([r["sigma_diff_mean"] for r in rows]),
                        "adaptive_alpha_mean": statistics.mean([r["adaptive_alpha_mean"] for r in rows]),
                        "adaptive_alpha_min": min([r["adaptive_alpha_min"] for r in rows]),
                        "adaptive_alpha_max": max([r["adaptive_alpha_max"] for r in rows]),
                        "used_alpha_mean": statistics.mean([r["used_alpha_mean"] for r in rows]),
                        "used_alpha_min": min([r["used_alpha_min"] for r in rows]),
                        "used_alpha_max": max([r["used_alpha_max"] for r in rows]),
                        "eval_seeds": ",".join(str(s) for s in seeds),
                        "ckpt_path": rows[0]["ckpt_path"],
                        "freq_stats_path": args.freq_stats_path,
                    }
                )

    exclude_tag = args.exclude_class
    per_seed_path = os.path.join(args.output_dir, f"ddfsd_{exclude_tag}_alpha_grid_per_seed.csv")
    summary_path = os.path.join(args.output_dir, f"ddfsd_{exclude_tag}_alpha_grid_summary.csv")
    write_csv(per_seed_path, per_seed_rows, PER_SEED_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    logger.info("Saved per-seed CSV: %s", per_seed_path)
    logger.info("Saved summary CSV: %s", summary_path)

    print("\n=== DDFSD alpha-grid diagnosis summary ===")
    header = (
        f"{'step':>6} | {'mode':>9} | {'alpha':>8} | {'ACC mean±std':>16} | "
        f"{'AP mean±std':>16} | {'AUC mean±std':>16} | {'used_alpha':>10} | {'adaptive_alpha':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['ckpt_step']:>6} | {row['branch_mode']:>9} | {row['alpha_mode']:>8} | "
            f"{row['acc_mean']:.4f}±{row['acc_std']:.4f} | "
            f"{row['ap_mean']:.4f}±{row['ap_std']:.4f} | "
            f"{row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"{row['used_alpha_mean']:>10.4f} | {row['adaptive_alpha_mean']:>14.4f}"
        )

    if missing_ckpt_steps:
        print(f"\n缺失 checkpoint（未评测）: {missing_ckpt_steps}")


if __name__ == "__main__":
    main()
