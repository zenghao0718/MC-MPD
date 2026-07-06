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

    return parser.parse_args()


def load_runtime_dependencies():
    global np, torch, dist, GradScaler, autocast, LambdaLR, StepLR, SummaryWriter
    global logger, build_train_iterators, get_train_fake_classes, validate_generator_name
    global DDFSDDualDomainNet, compute_ddfsd_episode_loss, compute_lambda_sep, sample_branch_mode
    global evaluate_binary_few_shot, compute_frequency_stats, load_frequency_stats, setup_dist

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.amp import GradScaler, autocast
    from torch.optim.lr_scheduler import LambdaLR, StepLR
    from torch.utils.tensorboard import SummaryWriter

    import util.logger as logger
    from datasets.ddfsd_datasets import (
        build_train_iterators,
        get_train_fake_classes,
        validate_generator_name,
    )
    from model.ddfsd import DDFSDDualDomainNet
    from model.ddfsd_losses import (
        compute_ddfsd_episode_loss,
        compute_lambda_sep,
        sample_branch_mode,
    )
    from util.ddfsd_eval import evaluate_binary_few_shot
    from util.ddfsd_frequency import compute_frequency_stats, load_frequency_stats
    from util.utils import setup_dist


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


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


def save_ddfsd_checkpoint(output_dir, args, step, effective_step, model, optimizer, scheduler, scaler):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{args.model}_step[{step}].pth")
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
            "exclude_class": args.exclude_class,
            "freq_stats_path": args.freq_stats_path,
        },
        save_path,
    )
    return save_path


def prepare_frequency_stats(args):
    if not args.freq_stats_path:
        args.freq_stats_path = os.path.join(args.output_dir, "freq_stats.pt")

    if os.path.exists(args.freq_stats_path):
        return load_frequency_stats(args.freq_stats_path)

    if not args.auto_compute_freq_stats:
        raise FileNotFoundError(
            f"Missing freq_stats.pt at {args.freq_stats_path}; "
            "training requires --auto_compute_freq_stats True or a valid --freq_stats_path."
        )

    if is_main_process():
        logger.info("Frequency stats not found. Computing train-split stats at %s", args.freq_stats_path)
        compute_frequency_stats(
            data_root=args.data_root,
            exclude_class=args.exclude_class,
            output_path=args.freq_stats_path,
            batch_size=args.freq_stats_batch_size,
            num_workers=args.num_workers,
            device=torch.device("cuda", args.local_rank),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return load_frequency_stats(args.freq_stats_path)


def validate_args(args):
    validate_generator_name(args.exclude_class)
    if args.num_class_train != 3:
        raise ValueError("DDFSD v1 training is fixed to real + 2 fake classes, so num_class_train must be 3.")
    if args.num_support_train != 5 or args.num_query_train != 5:
        raise ValueError("DDFSD v1 training uses exactly 5 support and 5 query samples per class.")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1.")
    prob_sum = (
        args.branch_dropout_dual_prob
        + args.branch_dropout_rgb_prob
        + args.branch_dropout_freq_prob
    )
    if prob_sum <= 0:
        raise ValueError("Branch dropout probabilities must sum to a positive value.")


def run_validation(model, args, step, tb_writer):
    fake_classes = get_train_fake_classes(args.exclude_class) + [args.exclude_class]
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
            )
            metrics_per_repeat.append(metrics)

        acc = float(np.mean([item["acc"] for item in metrics_per_repeat]))
        ap = float(np.mean([item["ap"] for item in metrics_per_repeat]))
        auc = float(np.mean([item["auc"] for item in metrics_per_repeat]))
        split = "val_unseen" if fake_class == args.exclude_class else "val_seen"
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

    stats = prepare_frequency_stats(args)
    logger.info("Loaded frequency stats from %s", args.freq_stats_path)
    logger.info("Frequency mean shape: %s std shape: %s", tuple(stats["mean"].shape), tuple(stats["std"].shape))

    images_per_class = (args.num_support_train + args.num_query_train) * args.batch_size
    train_fake_classes = get_train_fake_classes(args.exclude_class)
    train_classes = ["real"] + train_fake_classes
    train_iters = build_train_iterators(
        data_root=args.data_root,
        classes=train_classes,
        images_per_class_per_step=images_per_class,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DDFSDDualDomainNet(pretrained=args.pretrained)
    model.set_freq_stats(stats["mean"], stats["std"])
    model = model.to(args.device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.rgb_backbone.parameters(), "lr": args.rgb_backbone_lr},
            {"params": model.freq_backbone.parameters(), "lr": args.freq_backbone_lr},
            {"params": model.rgb_projector.parameters(), "lr": args.rgb_head_lr},
            {"params": model.freq_projector.parameters(), "lr": args.freq_head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.use_fp16)

    effective_step = 0
    optimizer.zero_grad(set_to_none=True)
    logger.info("Start DDFSD training for %d steps.", args.total_training_steps)

    for step in range(1, args.total_training_steps + 1):
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
        branch_mode = sample_branch_mode(
            args.branch_dropout_dual_prob,
            args.branch_dropout_rgb_prob,
            args.branch_dropout_freq_prob,
        )

        with autocast(device_type="cuda", enabled=args.use_fp16):
            outputs = model(raw_rgb)

        loss_out = compute_ddfsd_episode_loss(
            z_rgb_flat=outputs["z_rgb"],
            z_freq_flat=outputs["z_freq"],
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
        )
        loss = loss_out["loss_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DDFSD loss at step {step}: {loss.item()}")

        scaler.scale(loss / args.accumulation_steps).backward()

        logger.logkv_mean("loss_total", loss_out["loss_total"].item())
        logger.logkv_mean("loss_dual", loss_out["loss_dual"].item())
        logger.logkv_mean("loss_sep", loss_out["loss_sep"].item())
        logger.logkv_mean("loss_rf", loss_out["loss_rf"].item())
        logger.logkv_mean("loss_ff", loss_out["loss_ff"].item())
        logger.logkv_mean("alpha_mean", loss_out["alpha"].mean().item())
        logger.logkv_mean("alpha_min", loss_out["alpha"].min().item())
        logger.logkv_mean("alpha_max", loss_out["alpha"].max().item())
        logger.logkv_mean("rgb_dist_mean", loss_out["rgb_dist"].mean().item())
        logger.logkv_mean("freq_dist_mean", loss_out["freq_dist"].mean().item())
        logger.logkv_mean("fused_dist_mean", loss_out["fused_dist"].mean().item())
        logger.logkv_mean("branch_mode_dual_ratio", 1.0 if branch_mode == "dual" else 0.0)
        logger.logkv_mean("branch_mode_rgb_ratio", 1.0 if branch_mode == "rgb-only" else 0.0)
        logger.logkv_mean("branch_mode_freq_ratio", 1.0 if branch_mode == "freq-only" else 0.0)
        logger.logkv("lambda_sep_current", lambda_sep_current)

        if step % args.accumulation_steps == 0:
            effective_step += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if is_main_process() and step % args.log_interval == 0:
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
