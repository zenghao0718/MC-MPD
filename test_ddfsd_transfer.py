#!/usr/bin/env python3
"""No-gradient DDFSD prototype transfer evaluation on manifest-defined tasks."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="MS_COCOAI")
    parser.add_argument("--target_generator", required=True)
    parser.add_argument("--support_manifest", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--freq_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau_r", type=float, default=0.1)
    parser.add_argument("--branch_mode", choices=["dual", "rgb-only", "freq-only"], default="dual")
    parser.add_argument("--allow_nonformal_checkpoint", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config(checkpoint):
    config = checkpoint.get("config", checkpoint.get("args", {}))
    return config if isinstance(config, dict) else vars(config)


def checkpoint_value(checkpoint, key, default=None):
    if key in checkpoint:
        return checkpoint[key]
    return checkpoint_config(checkpoint).get(key, default)


def validate_formal_checkpoint(checkpoint, branch_mode, allow_nonformal):
    actual = {
        "model_mode": checkpoint_value(checkpoint, "model_mode", "dual"),
        "training_scope": checkpoint_value(checkpoint, "training_scope"),
        "source_fake_classes": checkpoint_value(checkpoint, "source_fake_classes"),
        "source_classes": checkpoint_value(checkpoint, "source_classes"),
        "step": int(checkpoint.get("step", 0)),
    }
    required = {"model_mode": "dual", "training_scope": "all-source", "step": 15000}
    mismatches = [f"{key}={actual[key]!r} (required {value!r})" for key, value in required.items() if actual[key] != value]
    if branch_mode != "dual":
        mismatches.append(f"branch_mode={branch_mode!r} (required 'dual')")
    if mismatches and not allow_nonformal:
        raise ValueError("Formal transfer checkpoint validation failed: " + "; ".join(mismatches))
    return actual, mismatches


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    args = parse_args()
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("eval_batch_size must be positive and num_workers must be >= 0")

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    from datasets.transfer_manifest_dataset import TransferManifestDataset, transfer_collate_fn
    from model.ddfsd import DDFSDDualDomainNet
    from model.ddfsd_losses import compute_alpha, compute_prototypes, compute_query_logits
    from util.ddfsd_eval import encode_batch
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed

    set_seed(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for path in (args.support_manifest, args.query_manifest, args.ckpt_path, args.freq_stats_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no model state: {args.ckpt_path}")
    checkpoint_meta, formal_mismatches = validate_formal_checkpoint(
        checkpoint, args.branch_mode, args.allow_nonformal_checkpoint
    )
    model_mode = checkpoint_meta["model_mode"]
    if model_mode != "dual" and args.branch_mode != model_mode:
        raise ValueError(f"A {model_mode} checkpoint cannot use branch_mode={args.branch_mode}")

    stats = load_frequency_stats(args.freq_stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDFSDDualDomainNet(pretrained=False, model_mode=model_mode)
    model.load_state_dict(checkpoint["model"])
    if model_mode != "rgb-only":
        model.set_freq_stats(stats["mean"], stats["std"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    support_dataset = TransferManifestDataset(args.support_manifest)
    query_dataset = TransferManifestDataset(args.query_manifest)
    for role, dataset in (("support", support_dataset), ("query", query_dataset)):
        generator_values = {row.get("target_generator") for row in dataset.rows}
        seed_values = {int(row.get("seed", -1)) for row in dataset.rows}
        role_values = {row.get("task_role") for row in dataset.rows}
        if generator_values != {args.target_generator}:
            raise ValueError(f"{role} manifest target_generator mismatch: {generator_values}")
        if seed_values != {args.seed}:
            raise ValueError(f"{role} manifest seed mismatch: {seed_values}")
        if role_values != {role}:
            raise ValueError(f"{role} manifest task_role mismatch: {role_values}")
    support_paths = {row["image_path"] for row in support_dataset.rows}
    query_paths = {row["image_path"] for row in query_dataset.rows}
    support_groups = {row.get("group_id") for row in support_dataset.rows}
    query_groups = {row.get("group_id") for row in query_dataset.rows}
    if support_paths & query_paths or support_groups & query_groups:
        raise ValueError("Support and query manifests overlap by image path or group_id.")
    by_label = {0: [], 1: []}
    for index, row in enumerate(support_dataset.rows):
        by_label[int(row["label"])].append(index)
    if len(by_label[0]) != 10 or len(by_label[1]) != 10:
        raise ValueError(
            f"Formal support must contain 10 real + 10 fake images, got "
            f"{len(by_label[0])} + {len(by_label[1])}"
        )
    query_label_counts = {
        label: sum(int(row["label"]) == label for row in query_dataset.rows) for label in (0, 1)
    }
    if query_label_counts[0] != query_label_counts[1]:
        raise ValueError(f"Query real/fake counts are not balanced: {query_label_counts}")
    support_images = torch.stack(
        [support_dataset[index][0] for label in (0, 1) for index in by_label[label]], dim=0
    )

    with torch.no_grad():
        support_rgb_flat, support_freq_flat = encode_batch(model, support_images, device, args.use_fp16)
        support_rgb = (
            support_rgb_flat.reshape(2, 10, -1).permute(1, 0, 2).unsqueeze(0)
            if support_rgb_flat is not None else None
        )
        support_freq = (
            support_freq_flat.reshape(2, 10, -1).permute(1, 0, 2).unsqueeze(0)
            if support_freq_flat is not None else None
        )
        proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode=model_mode)
        alpha = None
        if model_mode == "dual":
            from model.ddfsd_losses import compute_support_sigmas
            sigma_rgb, sigma_freq = compute_support_sigmas(
                support_rgb, support_freq, proto_rgb, proto_freq
            )
            alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=args.tau_r)

        loader = DataLoader(
            query_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=transfer_collate_fn,
        )
        result_rows = []
        all_labels, all_scores, all_predictions = [], [], []
        ckpt_sha = sha256_file(args.ckpt_path)
        stats_sha = sha256_file(args.freq_stats_path)
        ckpt_step = int(checkpoint.get("step", 0))
        for images, labels, metadata in loader:
            query_rgb, query_freq = encode_batch(model, images, device, args.use_fp16)
            logits = compute_query_logits(
                query_rgb=query_rgb.unsqueeze(0) if query_rgb is not None else None,
                query_freq=query_freq.unsqueeze(0) if query_freq is not None else None,
                proto_rgb=proto_rgb,
                proto_freq=proto_freq,
                alpha=alpha,
                tau=args.tau,
                branch_mode=args.branch_mode,
                model_mode=model_mode,
            )["logits"].squeeze(0)
            scores = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
            label_values = labels.numpy()
            predictions = (scores >= 0.5).astype(np.int64)
            for metadata_row, label, score, prediction in zip(
                metadata, label_values, scores, predictions
            ):
                result_rows.append({
                    "dataset": args.dataset_name,
                    "split": metadata_row.get("split", ""),
                    "target_generator": args.target_generator,
                    "seed": args.seed,
                    "group_id": metadata_row.get("group_id", ""),
                    "image_path": metadata_row["image_path"],
                    "image_sha256": metadata_row.get("image_sha256", ""),
                    "label": int(label),
                    "fake_score": float(score),
                    "prediction": int(prediction),
                    "correct": int(prediction == label),
                    "branch_mode": args.branch_mode,
                    "ckpt_step": ckpt_step,
                    "ckpt_sha256": ckpt_sha,
                    "freq_stats_sha256": stats_sha,
                })
            all_labels.extend(label_values.tolist())
            all_scores.extend(scores.tolist())
            all_predictions.extend(predictions.tolist())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    scores_np = np.asarray(all_scores, dtype=np.float64)
    predictions_np = np.asarray(all_predictions, dtype=np.int64)
    if set(labels_np.tolist()) != {0, 1}:
        raise ValueError("Query manifest must contain both binary classes.")
    real_mask, fake_mask = labels_np == 0, labels_np == 1
    acc = float((predictions_np == labels_np).mean())
    ap = float(average_precision_score(labels_np, scores_np))
    auc = float(roc_auc_score(labels_np, scores_np))
    metrics = {
        "ACC": acc,
        "AP": ap,
        "AUC": auc,
        "acc": acc,
        "ap": ap,
        "auc": auc,
        "real_acc": float((predictions_np[real_mask] == 0).mean()),
        "fake_acc": float((predictions_np[fake_mask] == 1).mean()),
        "balanced_acc": float(
            0.5 * ((predictions_np[real_mask] == 0).mean() + (predictions_np[fake_mask] == 1).mean())
        ),
        "mean_real_score": float(scores_np[real_mask].mean()),
        "mean_fake_score": float(scores_np[fake_mask].mean()),
        "num_real_query": int(real_mask.sum()),
        "num_fake_query": int(fake_mask.sum()),
    }

    score_path = os.path.join(output_dir, "per_image_scores.csv")
    with open(score_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    run_config = vars(args).copy()
    run_config.update({"device": str(device), "checkpoint_metadata": checkpoint_meta})
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "support_manifest_sha256": sha256_file(args.support_manifest),
        "query_manifest_sha256": sha256_file(args.query_manifest),
        "checkpoint_sha256": ckpt_sha,
        "frequency_stats_sha256": stats_sha,
        "checkpoint_metadata": checkpoint_meta,
        "formal_checkpoint_validation": not formal_mismatches,
        "formal_mismatches_allowed": formal_mismatches,
        "gradient_updates": False,
        "target_frequency_statistics_computed": False,
    }
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_json(os.path.join(output_dir, "run_config.json"), run_config)
    write_json(os.path.join(output_dir, "provenance.json"), provenance)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
