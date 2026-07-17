# -*- coding: utf-8 -*-
"""Train DDFSD dual-domain few-shot detector."""

import argparse
import math
import os
import random


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDFSD v1")

    parser.add_argument("--model", type=str, default="ddfsd")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output_dir")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--model_mode", type=str, default="dual", choices=["dual", "rgb-only", "freq-only"])
    parser.add_argument(
        "--training_scope",
        type=str,
        default="leave-one-out",
        choices=["leave-one-out", "all-source"],
    )
    parser.add_argument("--exclude_class", type=str, default="ADM")
    parser.add_argument("--batch_size", type=int, default=16, help="Episode batch size.")
    parser.add_argument("--num_class_train", type=int, default=3)
    parser.add_argument("--num_support_train", type=int, default=5)
    parser.add_argument("--num_query_train", type=int, default=5)
    parser.add_argument("--num_support_val", type=int, default=10)
    parser.add_argument("--num_query_val", type=int, default=0)
    parser.add_argument("--val_eval_repeats", type=int, default=1)
    parser.add_argument("--val_eval_seed", type=int, default=42)
    parser.add_argument("--num_support_test", type=int, default=10)
    parser.add_argument("--num_query_test", type=int, default=0)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--eval_seeds", type=str, default="42,101,102,103,104")
    parser.add_argument("--max_eval_query_per_class", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--total_training_steps", type=int, default=15000)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=2500)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)

    parser.add_argument("--rgb_backbone_lr", type=float, default=3e-5)
    parser.add_argument("--freq_backbone_lr", type=float, default=3e-5)
    parser.add_argument("--rgb_head_lr", type=float, default=1e-4)
    parser.add_argument("--freq_head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--lr_scheduler_step", type=int, default=5000)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_warmup_steps", type=int, default=0)

    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    parser.add_argument("--m_rf", type=float, default=1.2)
    parser.add_argument("--m_ff", type=float, default=0.6)
    parser.add_argument("--lambda_ff", type=float, default=0.5)
    parser.add_argument("--lambda_sep_target", type=float, default=0.03)
    parser.add_argument("--lambda_sep_warmup_start", type=int, default=2500)
    parser.add_argument("--lambda_sep_warmup_end", type=int, default=7500)
    parser.add_argument("--branch_dropout_dual_prob", type=float, default=0.90)
    parser.add_argument("--branch_dropout_rgb_prob", type=float, default=0.05)
    parser.add_argument("--branch_dropout_freq_prob", type=float, default=0.05)

    parser.add_argument("--freq_stats_path", type=str, default="")
    parser.add_argument("--auto_compute_freq_stats", type=str2bool, default=True)
    parser.add_argument("--freq_stats_batch_size", type=int, default=128)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Checkpoint path used to resume training.",
    )
    parser.add_argument(
        "--resume_strict_config",
        type=str2bool,
        default=True,
        help="Whether to strictly validate resume configuration.",
    )

    return parser.parse_args()


def load_runtime_dependencies():
    global np, torch, dist, GradScaler, autocast, LambdaLR, StepLR, SummaryWriter
    global logger, build_train_iterators, resolve_train_fake_classes, validate_generator_name
    global DDFSDDualDomainNet, compute_ddfsd_episode_loss, compute_lambda_sep, sample_branch_mode
    global evaluate_binary_few_shot, compute_frequency_stats, load_frequency_stats
    global sha256_file, validate_allsource_frequency_metadata, setup_dist

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.amp import GradScaler, autocast
    from torch.optim.lr_scheduler import LambdaLR, StepLR
    from torch.utils.tensorboard import SummaryWriter

    import util.logger as logger
    from datasets.ddfsd_datasets import (
        build_train_iterators,
        resolve_train_fake_classes,
        validate_generator_name,
    )
    from model.ddfsd import DDFSDDualDomainNet
    from model.ddfsd_losses import (
        compute_ddfsd_episode_loss,
        compute_lambda_sep,
        sample_branch_mode,
    )
    from util.ddfsd_eval import evaluate_binary_few_shot
    from util.ddfsd_frequency import (
        compute_frequency_stats,
        load_frequency_stats,
        sha256_file,
        validate_allsource_frequency_metadata,
    )
    from util.utils import setup_dist


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


_MARGIN_DIAG_EPS = 1e-12


def new_margin_diag_window(model_mode="dual"):
    """A fresh per-log-window accumulator of detached CPU numpy samples.

    Only ever fed with tensors already produced by compute_separation_loss
    via compute_ddfsd_episode_loss, which are detach()'d before being
    returned. Accumulating/aggregating these values does not affect
    training: no graph is retained and nothing here feeds back into the
    optimizer step.
    """

    if model_mode != "dual":
        return {"proto_rf_active": [], "proto_ff_active": []}
    return {
        "proto_rf_rgb": [],
        "proto_rf_freq": [],
        "proto_rf_fused": [],
        "alpha_rf_pair": [],
        "proto_ff_rgb": [],
        "proto_ff_freq": [],
        "proto_ff_fused": [],
        "alpha_ff_pair": [],
    }


def append_margin_diag_window(window, loss_out, model_mode="dual"):
    """Append one step's detached prototype-pair tensors (CPU numpy) to window."""

    if model_mode != "dual":
        window["proto_rf_active"].append(loss_out["proto_rf_active_dist"].detach().cpu().numpy().reshape(-1))
        window["proto_ff_active"].append(loss_out["proto_ff_active_dist"].detach().cpu().numpy().reshape(-1))
        return

    window["proto_rf_rgb"].append(loss_out["proto_rf_rgb_dist"].detach().cpu().numpy().reshape(-1))
    window["proto_rf_freq"].append(loss_out["proto_rf_freq_dist"].detach().cpu().numpy().reshape(-1))
    window["proto_rf_fused"].append(loss_out["proto_rf_fused_dist"].detach().cpu().numpy().reshape(-1))
    window["alpha_rf_pair"].append(loss_out["alpha_rf_pair"].detach().cpu().numpy().reshape(-1))
    window["proto_ff_rgb"].append(loss_out["proto_ff_rgb_dist"].detach().cpu().numpy().reshape(-1))
    window["proto_ff_freq"].append(loss_out["proto_ff_freq_dist"].detach().cpu().numpy().reshape(-1))
    window["proto_ff_fused"].append(loss_out["proto_ff_fused_dist"].detach().cpu().numpy().reshape(-1))
    window["alpha_ff_pair"].append(loss_out["alpha_ff_pair"].detach().cpu().numpy().reshape(-1))


def _concat_or_empty(list_of_arrays):
    if not list_of_arrays:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(list_of_arrays).astype(np.float64)


def _basic_stats(name, values):
    if values.size == 0:
        return {f"{name}_mean": 0.0, f"{name}_min": 0.0, f"{name}_max": 0.0}
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
    }


def _percentile_stats(name, values):
    out = {}
    for p in (10, 25, 50, 75, 90):
        key = f"{name}_p{p}"
        out[key] = float(np.percentile(values, p)) if values.size > 0 else 0.0
    return out


def _violation_stats(prefix, fused, margin):
    if fused.size == 0:
        return {
            f"{prefix}_violation_rate": 0.0,
            f"{prefix}_violation_gap_mean": 0.0,
            f"{prefix}_violation_gap_max": 0.0,
        }
    violation_mask = fused < margin
    violation_rate = float(np.mean(violation_mask))
    gaps = margin - fused[violation_mask]
    if gaps.size > 0:
        gap_mean = float(np.mean(gaps))
        gap_max = float(np.max(gaps))
    else:
        gap_mean = 0.0
        gap_max = 0.0
    return {
        f"{prefix}_violation_rate": violation_rate,
        f"{prefix}_violation_gap_mean": gap_mean,
        f"{prefix}_violation_gap_max": gap_max,
    }


def compute_margin_diag_window_stats(window, m_rf, m_ff, model_mode="dual"):
    """Pool all detached samples collected since the last log dump into a
    single dict of scalar diagnostics (mean/min/max/percentiles/violation
    rate/violation gap). Guaranteed no NaN: falls back to 0.0 when a window
    has zero samples (should not normally happen since every training step
    contributes exactly batch_size*2 RF and batch_size*1 FF samples).
    """

    if model_mode != "dual":
        branch_name = "rgb" if model_mode == "rgb-only" else "freq"
        rf_active = _concat_or_empty(window["proto_rf_active"])
        ff_active = _concat_or_empty(window["proto_ff_active"])
        out = {}
        out.update(_basic_stats(f"proto_rf_{branch_name}", rf_active))
        out.update(_percentile_stats(f"proto_rf_{branch_name}", rf_active))
        out.update(_basic_stats(f"proto_ff_{branch_name}", ff_active))
        out.update(_percentile_stats(f"proto_ff_{branch_name}", ff_active))
        out.update(_violation_stats("rf", rf_active, m_rf))
        out.update(_violation_stats("ff", ff_active, m_ff))
        return out

    rf_rgb = _concat_or_empty(window["proto_rf_rgb"])
    rf_freq = _concat_or_empty(window["proto_rf_freq"])
    rf_fused = _concat_or_empty(window["proto_rf_fused"])
    alpha_rf = _concat_or_empty(window["alpha_rf_pair"])
    ff_rgb = _concat_or_empty(window["proto_ff_rgb"])
    ff_freq = _concat_or_empty(window["proto_ff_freq"])
    ff_fused = _concat_or_empty(window["proto_ff_fused"])
    alpha_ff = _concat_or_empty(window["alpha_ff_pair"])

    out = {}
    out.update(_basic_stats("proto_rf_rgb", rf_rgb))
    out.update(_basic_stats("proto_rf_freq", rf_freq))
    out.update(_basic_stats("proto_rf_fused", rf_fused))
    out.update(_percentile_stats("proto_rf_fused", rf_fused))
    out.update(_basic_stats("alpha_rf_pair", alpha_rf))

    out.update(_basic_stats("proto_ff_rgb", ff_rgb))
    out.update(_basic_stats("proto_ff_freq", ff_freq))
    out.update(_basic_stats("proto_ff_fused", ff_fused))
    out.update(_percentile_stats("proto_ff_fused", ff_fused))
    out.update(_basic_stats("alpha_ff_pair", alpha_ff))

    out.update(_violation_stats("rf", rf_fused, m_rf))
    out.update(_violation_stats("ff", ff_fused, m_ff))

    for value in out.values():
        assert value == value, "NaN detected in margin diagnostics window stats"  # noqa: PLR0124

    return out


def create_scheduler(optimizer, args):
    if args.scheduler_type == "step":
        return StepLR(
            optimizer=optimizer,
            step_size=args.lr_scheduler_step,
            gamma=args.lr_scheduler_gamma,
        )

    def lr_lambda(step):
        if args.scheduler_warmup_steps > 0 and step < args.scheduler_warmup_steps:
            return max(float(step + 1) / float(args.scheduler_warmup_steps), 1e-8)
        decay_steps = max(args.total_training_steps - args.scheduler_warmup_steps, 1)
        progress = min(max((step - args.scheduler_warmup_steps) / float(decay_steps), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


RESUME_CONFIG_KEYS = (
    "model", "model_mode", "training_scope", "source_fake_classes", "exclude_class", "batch_size", "num_class_train",
    "num_support_train", "num_query_train", "accumulation_steps", "scheduler_type",
    "lr_scheduler_step", "lr_scheduler_gamma", "rgb_backbone_lr", "freq_backbone_lr",
    "rgb_head_lr", "freq_head_lr", "weight_decay", "tau", "tau_r", "m_rf", "m_ff",
    "lambda_ff", "lambda_sep_target", "lambda_sep_warmup_start", "lambda_sep_warmup_end",
    "branch_dropout_dual_prob", "branch_dropout_rgb_prob", "branch_dropout_freq_prob",
)


def checkpoint_config(checkpoint):
    config = checkpoint.get("config", checkpoint.get("args", {}))
    return config if isinstance(config, dict) else vars(config)


def validate_resume_config(args, checkpoint):
    config = checkpoint_config(checkpoint)
    if args.training_scope == "all-source":
        missing_scope_keys = [
            key for key in ("training_scope", "source_fake_classes") if key not in config
        ]
        if missing_scope_keys:
            raise ValueError(
                "An all-source strict resume requires checkpoint metadata for "
                f"{missing_scope_keys}; refusing to treat a legacy/leave-one-out checkpoint as all-source."
            )
    mismatches = []
    for key in RESUME_CONFIG_KEYS:
        if key not in config:
            # Legacy checkpoints saved before this key existed in the config schema
            # (e.g. model_mode) have nothing to compare against. Skip the strict
            # comparison for that key only, and log it instead of hard-failing, so
            # that formal resumes of pre-existing checkpoints remain possible.
            # Every key that IS present in the checkpoint is still strictly checked
            # below.
            logger.warn(
                "Resume config key '%s' is absent from the checkpoint config "
                "(legacy checkpoint format); skipping strict comparison for this "
                "key and using the current CLI value: %r",
                key, getattr(args, key),
            )
            continue
        old, current = config[key], getattr(args, key)
        equal = math.isclose(old, current, rel_tol=1e-7, abs_tol=1e-12) if (
            isinstance(old, float) and isinstance(current, (int, float))
        ) else old == current
        if not equal:
            mismatches.append(f"{key}: checkpoint={old!r}, current={current!r}")
    if mismatches:
        raise ValueError("Resume configuration mismatch:\n  " + "\n  ".join(mismatches))
    if args.training_scope == "all-source":
        checkpoint_stats_sha = checkpoint.get("freq_stats_sha256")
        if not checkpoint_stats_sha:
            raise ValueError("All-source strict resume checkpoint has no freq_stats_sha256.")
        current_stats_sha = sha256_file(args.freq_stats_path)
        if checkpoint_stats_sha != current_stats_sha:
            raise ValueError(
                "All-source strict resume frequency-stat SHA mismatch: "
                f"checkpoint={checkpoint_stats_sha}, current={current_stats_sha}"
            )


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_random_states(checkpoint):
    states = (
        ("python_random_state", random.setstate),
        ("numpy_random_state", np.random.set_state),
        ("torch_cpu_rng_state", torch.set_rng_state),
    )
    for key, restore in states:
        if checkpoint.get(key) is None:
            logger.warn("Resume checkpoint has no %s; continuing without restoring it.", key)
        else:
            restore(checkpoint[key])
    if checkpoint.get("torch_cuda_rng_state") is None:
        logger.warn("Resume checkpoint has no torch_cuda_rng_state; continuing without restoring it.")
    elif torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])


def save_ddfsd_checkpoint(output_dir, args, step, effective_step, model, optimizer, scheduler, scaler):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{args.model}_step[{step}].pth")
    freq_stats_sha256 = None
    freq_stats_metadata = None
    if args.model_mode != "rgb-only" or args.training_scope == "all-source":
        if not args.freq_stats_path or not os.path.isfile(args.freq_stats_path):
            raise FileNotFoundError(
                "Checkpoint provenance requires the actual frequency-statistics file; "
                f"missing: {args.freq_stats_path!r}"
            )
        stats = load_frequency_stats(args.freq_stats_path)
        freq_stats_sha256 = sha256_file(args.freq_stats_path)
        freq_stats_metadata = stats.get("metadata")
        bound_sha = getattr(args, "freq_stats_sha256", None)
        bound_metadata = getattr(args, "freq_stats_metadata", None)
        if bound_sha and bound_sha != freq_stats_sha256:
            raise RuntimeError(
                "Frequency-statistics file changed after model initialization: "
                f"bound={bound_sha}, current={freq_stats_sha256}"
            )
        if bound_metadata is not None and bound_metadata != freq_stats_metadata:
            raise RuntimeError("Frequency-statistics metadata changed after model initialization.")
        if args.training_scope == "all-source":
            validate_allsource_frequency_metadata(freq_stats_metadata)
    torch.save(
        {
            "step": step,
            "effective_step": effective_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": args,
            "config": vars(args),
            "model_mode": args.model_mode,
            "training_scope": args.training_scope,
            "source_fake_classes": list(args.source_fake_classes),
            "source_classes": list(args.source_classes),
            "exclude_class": args.exclude_class if args.training_scope == "leave-one-out" else None,
            "freq_stats_path": args.freq_stats_path if args.model_mode != "rgb-only" else None,
            "freq_stats_sha256": freq_stats_sha256,
            "freq_stats_metadata": freq_stats_metadata,
            "resume_from_checkpoint": args.resume_checkpoint or None,
            "resume_from_step": getattr(args, "resume_from_step", 0),
            "resume_target_total_steps": args.total_training_steps,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        save_path,
    )
    return save_path


def prepare_frequency_stats(args):
    def bind(stats):
        if args.training_scope == "all-source":
            validate_allsource_frequency_metadata(stats.get("metadata"))
        args.freq_stats_sha256 = sha256_file(args.freq_stats_path)
        args.freq_stats_metadata = stats.get("metadata")
        return stats

    if not args.freq_stats_path:
        filename = "freq_stats_allsource.pt" if args.training_scope == "all-source" else "freq_stats.pt"
        args.freq_stats_path = os.path.join(args.output_dir, filename)

    if os.path.exists(args.freq_stats_path):
        stats = load_frequency_stats(args.freq_stats_path)
        return bind(stats)

    if not args.auto_compute_freq_stats:
        raise FileNotFoundError(
            f"Missing freq_stats.pt at {args.freq_stats_path}; "
            "training requires --auto_compute_freq_stats True or a valid --freq_stats_path."
        )

    if is_main_process():
        logger.info("Frequency stats not found. Computing train-split stats at %s", args.freq_stats_path)
        compute_frequency_stats(
            data_root=args.data_root,
            output_path=args.freq_stats_path,
            classes=args.source_classes if args.training_scope == "all-source" else None,
            exclude_class=args.exclude_class if args.training_scope == "leave-one-out" else None,
            batch_size=args.freq_stats_batch_size,
            num_workers=args.num_workers,
            device=torch.device("cuda", args.local_rank),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    stats = load_frequency_stats(args.freq_stats_path)
    return bind(stats)


def validate_args(args):
    if args.training_scope == "leave-one-out":
        validate_generator_name(args.exclude_class)
    args.source_fake_classes = resolve_train_fake_classes(
        training_scope=args.training_scope,
        exclude_class=args.exclude_class,
    )
    args.source_classes = ["real"] + list(args.source_fake_classes)
    if args.training_scope == "all-source":
        args.exclude_class = None
    if args.num_class_train != 3:
        raise ValueError("DDFSD v1 training is fixed to real + 2 fake classes, so num_class_train must be 3.")
    if args.num_support_train != 5 or args.num_query_train != 5:
        raise ValueError("DDFSD v1 training uses exactly 5 support and 5 query samples per class.")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1.")
    if args.model_mode == "dual":
        prob_sum = (
            args.branch_dropout_dual_prob
            + args.branch_dropout_rgb_prob
            + args.branch_dropout_freq_prob
        )
        if prob_sum <= 0:
            raise ValueError("Branch dropout probabilities must sum to a positive value.")


def run_validation(model, args, step, tb_writer):
    fake_classes = list(args.source_fake_classes)
    if args.training_scope == "leave-one-out":
        fake_classes.append(args.exclude_class)
    for fake_class in fake_classes:
        metrics_per_repeat = []
        for repeat_idx in range(args.val_eval_repeats):
            seed = args.val_eval_seed + repeat_idx
            metrics = evaluate_binary_few_shot(
                model=model,
                data_root=args.data_root,
                fake_class=fake_class,
                support_shot=args.num_support_val,
                seed=seed,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=args.device,
                use_fp16=args.use_fp16,
                tau=args.tau,
                tau_r=args.tau_r,
                max_query_per_class=args.max_eval_query_per_class,
                model_mode=args.model_mode,
                branch_mode=args.model_mode,
            )
            metrics_per_repeat.append(metrics)

        acc = float(np.mean([item["acc"] for item in metrics_per_repeat]))
        ap = float(np.mean([item["ap"] for item in metrics_per_repeat]))
        auc = float(np.mean([item["auc"] for item in metrics_per_repeat]))
        split = (
            "val_unseen"
            if args.training_scope == "leave-one-out" and fake_class == args.exclude_class
            else "val_seen"
        )
        logger.info(
            "Validation %s/%s at step %d: ACC %.6f AP %.6f AUC %.6f",
            split,
            fake_class,
            step,
            acc,
            ap,
            auc,
        )
        tb_writer.add_scalar(f"{split}/{fake_class}/ACC", acc, step)
        tb_writer.add_scalar(f"{split}/{fake_class}/AP", ap, step)
        tb_writer.add_scalar(f"{split}/{fake_class}/AUC", auc, step)


def main():
    args = parse_args()
    load_runtime_dependencies()
    validate_args(args)
    setup_dist(args)
    if args.world_size != 1:
        raise NotImplementedError("DDFSD v1 only supports single-GPU training; use --nproc_per_node 1.")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setup(log_dir=args.output_dir, device=args.device)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb")) if is_main_process() else None

    stats = None
    if args.model_mode != "rgb-only":
        stats = prepare_frequency_stats(args)
        logger.info("Loaded frequency stats from %s", args.freq_stats_path)
        logger.info("Frequency mean shape: %s std shape: %s", tuple(stats["mean"].shape), tuple(stats["std"].shape))
    else:
        logger.info("model_mode=rgb-only: skipping frequency-stat loading and computation.")

    images_per_class = (args.num_support_train + args.num_query_train) * args.batch_size
    train_fake_classes = list(args.source_fake_classes)
    train_classes = ["real"] + train_fake_classes
    train_iters = build_train_iterators(
        data_root=args.data_root,
        classes=train_classes,
        images_per_class_per_step=images_per_class,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_pretrained = args.pretrained and not bool(args.resume_checkpoint)
    model = DDFSDDualDomainNet(pretrained=model_pretrained, model_mode=args.model_mode)
    if stats is not None:
        model.set_freq_stats(stats["mean"], stats["std"])
    model = model.to(args.device)

    if args.model_mode == "dual":
        optimizer_groups = [
            {"params": model.rgb_backbone.parameters(), "lr": args.rgb_backbone_lr},
            {"params": model.freq_backbone.parameters(), "lr": args.freq_backbone_lr},
            {"params": model.rgb_projector.parameters(), "lr": args.rgb_head_lr},
            {"params": model.freq_projector.parameters(), "lr": args.freq_head_lr},
        ]
    elif args.model_mode == "rgb-only":
        optimizer_groups = [
            {"params": model.rgb_backbone.parameters(), "lr": args.rgb_backbone_lr},
            {"params": model.rgb_projector.parameters(), "lr": args.rgb_head_lr},
        ]
    else:
        optimizer_groups = [
            {"params": model.freq_backbone.parameters(), "lr": args.freq_backbone_lr},
            {"params": model.freq_projector.parameters(), "lr": args.freq_head_lr},
        ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.use_fp16)

    start_step = 1
    effective_step = 0
    args.resume_from_step = 0
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        required = ("model", "optimizer", "scheduler", "scaler", "step", "effective_step")
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise KeyError(f"Resume checkpoint is missing required keys: {missing}")
        if args.resume_strict_config:
            validate_resume_config(args, checkpoint)
        resume_step = int(checkpoint["step"])
        if args.total_training_steps <= resume_step:
            raise ValueError(
                f"total_training_steps ({args.total_training_steps}) must be greater than "
                f"resume checkpoint step ({resume_step})."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, args.device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        effective_step = int(checkpoint.get("effective_step", resume_step))
        start_step = resume_step + 1
        args.resume_from_step = resume_step
        restore_random_states(checkpoint)
        old_config = checkpoint_config(checkpoint)
        logger.info("Resume checkpoint: %s", args.resume_checkpoint)
        logger.info("Resume step: %d; effective step: %d; start step: %d; target total step: %d",
                    resume_step, effective_step, start_step, args.total_training_steps)
        logger.info("Data root: checkpoint=%r current=%r", old_config.get("data_root"), args.data_root)
        logger.info("Frequency stats path: checkpoint=%r current=%r",
                    old_config.get("freq_stats_path"), args.freq_stats_path)
        logger.info("Resume does not restore the dataloader iterator; sample order may differ after restart.")

    logger.info("Scheduler type: %s; last_epoch: %s; step_size: %s; gamma: %s",
                args.scheduler_type, scheduler.last_epoch, getattr(scheduler, "step_size", None),
                getattr(scheduler, "gamma", None))
    for group_idx, group in enumerate(optimizer.param_groups):
        logger.info("Optimizer param group %d current lr: %.12g", group_idx, group["lr"])
    optimizer.zero_grad(set_to_none=True)
    logger.info("Start DDFSD %s training for %d steps.", args.model_mode, args.total_training_steps)
    logger.info("Training scope: %s", args.training_scope)
    logger.info("Source fake classes: %s", ", ".join(args.source_fake_classes))
    logger.info("Margin config for this run: m_rf=%.4f m_ff=%.4f lambda_ff=%.4f", args.m_rf, args.m_ff, args.lambda_ff)
    margin_diag_window = new_margin_diag_window(args.model_mode)

    for step in range(start_step, args.total_training_steps + 1):
        model.train()
        selected_fake_classes = random.sample(train_fake_classes, 2)
        selected_classes = ["real"] + selected_fake_classes
        class_batches = [next(train_iters[class_name])[0] for class_name in selected_classes]
        raw_rgb = torch.stack(class_batches, dim=0).to(args.device, non_blocking=True)
        raw_rgb = raw_rgb.reshape(-1, raw_rgb.shape[-3], raw_rgb.shape[-2], raw_rgb.shape[-1])

        lambda_sep_current = compute_lambda_sep(
            step=step,
            target=args.lambda_sep_target,
            warmup_start=args.lambda_sep_warmup_start,
            warmup_end=args.lambda_sep_warmup_end,
        )
        if args.model_mode == "dual":
            branch_mode = sample_branch_mode(
                args.branch_dropout_dual_prob,
                args.branch_dropout_rgb_prob,
                args.branch_dropout_freq_prob,
            )
        else:
            branch_mode = args.model_mode

        with autocast(device_type="cuda", enabled=args.use_fp16):
            outputs = model(raw_rgb)

        loss_out = compute_ddfsd_episode_loss(
            z_rgb_flat=outputs.get("z_rgb"),
            z_freq_flat=outputs.get("z_freq"),
            episode_batch_size=args.batch_size,
            num_classes=args.num_class_train,
            num_support=args.num_support_train,
            num_query=args.num_query_train,
            tau=args.tau,
            tau_r=args.tau_r,
            m_rf=args.m_rf,
            m_ff=args.m_ff,
            lambda_ff=args.lambda_ff,
            lambda_sep_current=lambda_sep_current,
            branch_mode=branch_mode,
            model_mode=args.model_mode,
        )
        loss = loss_out["loss_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DDFSD loss at step {step}: {loss.item()}")

        scaler.scale(loss / args.accumulation_steps).backward()

        logger.logkv_mean("loss_total", loss_out["loss_total"].item())
        logger.logkv_mean("loss_cls", loss_out["loss_cls"].item())
        logger.logkv_mean("loss_dual", loss_out["loss_dual"].item())
        logger.logkv_mean("loss_sep", loss_out["loss_sep"].item())
        logger.logkv_mean("loss_rf", loss_out["loss_rf"].item())
        logger.logkv_mean("loss_ff", loss_out["loss_ff"].item())
        if loss_out["rgb_dist"] is not None:
            logger.logkv_mean("rgb_dist_mean", loss_out["rgb_dist"].mean().item())
        if loss_out["freq_dist"] is not None:
            logger.logkv_mean("freq_dist_mean", loss_out["freq_dist"].mean().item())
        if args.model_mode == "dual":
            logger.logkv_mean("alpha_mean", loss_out["alpha"].mean().item())
            logger.logkv_mean("alpha_min", loss_out["alpha"].min().item())
            logger.logkv_mean("alpha_max", loss_out["alpha"].max().item())
            logger.logkv_mean("fused_dist_mean", loss_out["fused_dist"].mean().item())
            logger.logkv_mean("branch_mode_dual_ratio", 1.0 if branch_mode == "dual" else 0.0)
            logger.logkv_mean("branch_mode_rgb_ratio", 1.0 if branch_mode == "rgb-only" else 0.0)
            logger.logkv_mean("branch_mode_freq_ratio", 1.0 if branch_mode == "freq-only" else 0.0)
        logger.logkv("model_mode", args.model_mode)
        logger.logkv("lambda_sep_current", lambda_sep_current)
        logger.logkv("m_rf", args.m_rf)
        logger.logkv("m_ff", args.m_ff)
        logger.logkv("lambda_ff", args.lambda_ff)

        # --- Margin diagnostics (read-only; does not affect loss/backward) ---
        append_margin_diag_window(margin_diag_window, loss_out, args.model_mode)
        weighted_loss_rf = lambda_sep_current * loss_out["loss_rf"].item()
        weighted_loss_ff = lambda_sep_current * args.lambda_ff * loss_out["loss_ff"].item()
        weighted_loss_sep = lambda_sep_current * loss_out["loss_sep"].item()
        weighted_sep_to_dual_ratio = weighted_loss_sep / max(loss_out["loss_dual"].item(), _MARGIN_DIAG_EPS)
        logger.logkv_mean("weighted_loss_rf", weighted_loss_rf)
        logger.logkv_mean("weighted_loss_ff", weighted_loss_ff)
        logger.logkv_mean("weighted_loss_sep", weighted_loss_sep)
        logger.logkv_mean("weighted_sep_to_dual_ratio", weighted_sep_to_dual_ratio)

        if step % args.accumulation_steps == 0:
            effective_step += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if is_main_process() and step % args.log_interval == 0:
            margin_diag_stats = compute_margin_diag_window_stats(
                margin_diag_window, args.m_rf, args.m_ff, args.model_mode
            )
            for key, value in margin_diag_stats.items():
                logger.logkv(key, value)
            margin_diag_window = new_margin_diag_window(args.model_mode)

            logger.logkv("step", step)
            logger.logkv("effective_step", effective_step)
            kvs = logger.dumpkvs()
            for key, value in kvs.items():
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(f"train/{key}", value, step)
            for idx, lr in enumerate(scheduler.get_last_lr()):
                tb_writer.add_scalar(f"train/lr_group_{idx}", lr, step)

        if is_main_process() and step % args.save_interval == 0:
            logger.info("Save DDFSD checkpoint at step: %d", step)
            save_path = save_ddfsd_checkpoint(
                os.path.join(args.output_dir, "ckpt"),
                args,
                step,
                effective_step,
                model,
                optimizer,
                scheduler,
                scaler,
            )
            logger.info("Saved checkpoint: %s", save_path)
            torch.cuda.empty_cache()

        if is_main_process() and step % args.eval_interval == 0:
            logger.info("Evaluating DDFSD at step: %d", step)
            run_validation(model, args, step, tb_writer)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if "dist" in globals() and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
