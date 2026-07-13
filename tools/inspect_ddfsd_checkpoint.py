#!/usr/bin/env python3
"""Read-only inspection and validation for DDFSD training checkpoints."""

import argparse
import math
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_step", type=int)
    parser.add_argument("--expected_exclude_class")
    parser.add_argument("--expected_scheduler_type")
    parser.add_argument("--expected_lr_scheduler_step", type=int)
    parser.add_argument("--expected_lr_scheduler_gamma", type=float)
    return parser.parse_args()


def config_dict(checkpoint):
    config = checkpoint.get("config", checkpoint.get("args", {}))
    return config if isinstance(config, dict) else vars(config)


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing = [key for key in ("model", "optimizer", "scheduler", "step") if key not in checkpoint]
    if missing:
        raise KeyError(f"Missing required checkpoint keys: {missing}")

    config = config_dict(checkpoint)
    scheduler = checkpoint["scheduler"] or {}
    optimizer = checkpoint["optimizer"] or {}
    values = {
        "step": int(checkpoint["step"]),
        "exclude_class": checkpoint.get("exclude_class", config.get("exclude_class")),
        "scheduler_type": config.get("scheduler_type"),
        "lr_scheduler_step": config.get("lr_scheduler_step"),
        "lr_scheduler_gamma": config.get("lr_scheduler_gamma"),
    }
    print(f"checkpoint path: {os.path.abspath(args.checkpoint)}")
    print(f"file size: {os.path.getsize(args.checkpoint)} bytes")
    for key in ("step", "effective_step", "model_mode", "exclude_class", "data_root",
                "freq_stats_path", "total_training_steps", "scheduler_type",
                "lr_scheduler_step", "lr_scheduler_gamma"):
        print(f"{key}: {checkpoint.get(key, config.get(key))}")
    print(f"scheduler last_epoch: {scheduler.get('last_epoch')}")
    print(f"scheduler _step_count: {scheduler.get('_step_count')}")
    print(f"scheduler _last_lr: {scheduler.get('_last_lr')}")
    groups = optimizer.get("param_groups", [])
    print(f"optimizer param group count: {len(groups)}")
    for index, group in enumerate(groups):
        print(f"optimizer param group {index} lr: {group.get('lr')}")
    print(f"contains scaler: {checkpoint.get('scaler') is not None}")
    rng_keys = ("python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state")
    print(f"contains random states: {all(key in checkpoint for key in rng_keys)}")
    resume_keys = ("resume_from_checkpoint", "resume_from_step", "resume_target_total_steps")
    print(f"contains resume metadata: {all(key in checkpoint for key in resume_keys)}")
    for key in resume_keys:
        print(f"{key}: {checkpoint.get(key)}")

    expected = {
        "step": args.expected_step,
        "exclude_class": args.expected_exclude_class,
        "scheduler_type": args.expected_scheduler_type,
        "lr_scheduler_step": args.expected_lr_scheduler_step,
        "lr_scheduler_gamma": args.expected_lr_scheduler_gamma,
    }
    errors = []
    for key, wanted in expected.items():
        if wanted is None:
            continue
        actual = values[key]
        equal = math.isclose(actual, wanted, rel_tol=1e-7, abs_tol=1e-12) if isinstance(wanted, float) else actual == wanted
        if not equal:
            errors.append(f"{key}: checkpoint={actual!r}, expected={wanted!r}")
    if errors:
        print("Checkpoint assertions failed:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    print("Checkpoint inspection passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
