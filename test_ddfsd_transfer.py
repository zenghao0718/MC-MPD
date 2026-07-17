#!/usr/bin/env python3
"""No-gradient DDFSD prototype transfer evaluation on manifest-defined tasks."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FAKE_CLASSES = ["ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
EXPECTED_SOURCE_CLASSES = ["real"] + EXPECTED_FAKE_CLASSES
RESULT_FILES = ("per_image_scores.csv", "metrics.json", "run_config.json", "provenance.json")
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
    parser.add_argument("--manifest_lock", default="")
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
    parser.add_argument("--overwrite", action="store_true")
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


def _same_float(left, right):
    return isinstance(left, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=1e-9, abs_tol=1e-12
    )


def validate_formal_artifacts(checkpoint, stats_metadata, stats_sha, args):
    actual = {
        "model_mode": checkpoint_value(checkpoint, "model_mode", "dual"),
        "training_scope": checkpoint_value(checkpoint, "training_scope"),
        "source_fake_classes": checkpoint_value(checkpoint, "source_fake_classes"),
        "source_classes": checkpoint_value(checkpoint, "source_classes"),
        "step": int(checkpoint.get("step", 0)),
        "tau": checkpoint_value(checkpoint, "tau"),
        "tau_r": checkpoint_value(checkpoint, "tau_r"),
        "freq_stats_sha256": checkpoint.get("freq_stats_sha256"),
        "freq_stats_metadata": checkpoint.get("freq_stats_metadata"),
    }
    expected = {
        "model_mode": "dual",
        "training_scope": "all-source",
        "source_fake_classes": EXPECTED_FAKE_CLASSES,
        "source_classes": EXPECTED_SOURCE_CLASSES,
        "step": 15000,
    }
    mismatches = [
        f"checkpoint.{key}={actual[key]!r} (required {value!r})"
        for key, value in expected.items() if actual[key] != value
    ]
    if args.branch_mode != "dual":
        mismatches.append(f"branch_mode={args.branch_mode!r} (required 'dual')")
    for name, cli_value, required in (("tau", args.tau, 0.2), ("tau_r", args.tau_r, 0.1)):
        if not _same_float(cli_value, required):
            mismatches.append(f"CLI {name}={cli_value!r} (required {required!r})")
        if not _same_float(actual[name], required):
            mismatches.append(f"checkpoint {name}={actual[name]!r} (required {required!r})")
        if not _same_float(actual[name], cli_value):
            mismatches.append(f"CLI/checkpoint {name} mismatch: {cli_value!r} vs {actual[name]!r}")
    expected_stats = {
        "training_scope": "all-source",
        "dataset": "GenImage",
        "split": "train",
        "classes": EXPECTED_SOURCE_CLASSES,
    }
    if not isinstance(stats_metadata, dict):
        mismatches.append("frequency stats metadata is missing")
    else:
        mismatches.extend(
            f"stats.metadata.{key}={stats_metadata.get(key)!r} (required {value!r})"
            for key, value in expected_stats.items() if stats_metadata.get(key) != value
        )
    if actual["freq_stats_sha256"] != stats_sha:
        mismatches.append(
            f"stats SHA is not bound to checkpoint: checkpoint={actual['freq_stats_sha256']!r}, actual={stats_sha!r}"
        )
    if actual["freq_stats_metadata"] != stats_metadata:
        mismatches.append("checkpoint.freq_stats_metadata does not exactly match the loaded stats metadata")
    return actual, mismatches


def validate_manifest_rows(support_rows, query_rows):
    for role, rows in (("support", support_rows), ("query", query_rows)):
        paths = [row["image_path"] for row in rows]
        hashes = [row["image_sha256"] for row in rows]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{role} contains duplicate image paths.")
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{role} contains duplicate image SHA-256 values.")
        labels = [int(row["label"]) for row in rows]
        if labels.count(0) != labels.count(1):
            raise ValueError(f"{role} real/fake counts are not balanced.")
    support_labels = [int(row["label"]) for row in support_rows]
    if support_labels.count(0) != 10 or support_labels.count(1) != 10:
        raise ValueError("Support must contain exactly 10 real + 10 fake images.")
    dimensions = (
        ("image path", {row["image_path"] for row in support_rows}, {row["image_path"] for row in query_rows}),
        ("group_id", {row["group_id"] for row in support_rows}, {row["group_id"] for row in query_rows}),
        ("image SHA-256", {row["image_sha256"] for row in support_rows}, {row["image_sha256"] for row in query_rows}),
    )
    for label, support_values, query_values in dimensions:
        if support_values & query_values:
            raise ValueError(f"Support/query {label} overlap.")


def validate_task_lock(lock_path, support_manifest, query_manifest):
    if not lock_path:
        raise ValueError("Formal transfer evaluation requires --manifest_lock.")
    resolved_lock = Path(lock_path).resolve()
    lock = json.loads(resolved_lock.read_text(encoding="utf-8"))
    locked = {str(Path(record["path"]).resolve()): record["sha256"] for record in lock.get("files", [])}
    for path in (support_manifest, query_manifest):
        resolved = str(Path(path).resolve())
        if resolved not in locked:
            raise ValueError(f"Task manifest is not present in manifest lock: {resolved}")
        actual = sha256_file(resolved)
        if locked[resolved] != actual:
            raise ValueError(f"Task manifest SHA does not match lock: {resolved}")
    return lock, sha256_file(resolved_lock)


def result_identity(args, ckpt_sha, stats_sha, support_sha, query_sha, lock_sha, formal_result):
    return {
        "checkpoint_sha256": ckpt_sha,
        "frequency_stats_sha256": stats_sha,
        "support_manifest_sha256": support_sha,
        "query_manifest_sha256": query_sha,
        "manifest_lock_sha256": lock_sha,
        "dataset_name": args.dataset_name,
        "target_generator": args.target_generator,
        "seed": args.seed,
        "tau": args.tau,
        "tau_r": args.tau_r,
        "branch_mode": args.branch_mode,
        "formal_result": formal_result,
    }


def prepare_output_directory(output_dir, identity, overwrite):
    output_path = Path(output_dir).resolve()
    if not output_path.exists():
        output_path.mkdir(parents=True)
        return False
    existing_entries = list(output_path.iterdir())
    if not existing_entries:
        return False
    result_paths = [output_path / name for name in RESULT_FILES]
    complete = all(path.is_file() and path.stat().st_size > 0 for path in result_paths)
    old_provenance = {}
    provenance_path = output_path / "provenance.json"
    if complete:
        try:
            for name in ("metrics.json", "run_config.json", "provenance.json"):
                json.loads((output_path / name).read_text(encoding="utf-8"))
            with (output_path / "per_image_scores.csv").open(encoding="utf-8") as handle:
                first_two_lines = [handle.readline(), handle.readline()]
                if not all(line.strip() for line in first_two_lines):
                    complete = False
        except (OSError, json.JSONDecodeError):
            complete = False
    if provenance_path.is_file():
        try:
            old_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_provenance = {"_error": "existing provenance.json is unreadable"}
    differences = {
        key: {"old": old_provenance.get(key), "new": identity.get(key)}
        for key in identity if old_provenance.get(key) != identity.get(key)
    }
    if complete and not differences and not overwrite:
        print(f"Result already exists with matching provenance; safely skipping: {output_path}")
        return True
    if not overwrite:
        state = "complete but provenance-mismatched" if complete else "non-empty/incomplete"
        raise FileExistsError(
            f"Refusing to overwrite {state} output directory {output_path}. "
            f"Differences: {json.dumps(differences, ensure_ascii=False, sort_keys=True)}"
        )
    print("--overwrite requested; replacing known result files.")
    print("Old/new provenance differences:")
    print(json.dumps(differences or {"status": "identical provenance"}, indent=2, ensure_ascii=False))
    return False


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
    required_paths = [args.support_manifest, args.query_manifest, args.ckpt_path, args.freq_stats_path]
    if args.manifest_lock:
        required_paths.append(args.manifest_lock)
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    from datasets.transfer_manifest_dataset import TransferManifestDataset, transfer_collate_fn
    from model.ddfsd import DDFSDDualDomainNet
    from model.ddfsd_losses import (
        compute_alpha, compute_prototypes, compute_query_logits, compute_support_sigmas,
    )
    from util.ddfsd_eval import encode_batch
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no model state: {args.ckpt_path}")
    stats = load_frequency_stats(args.freq_stats_path)
    ckpt_sha = sha256_file(args.ckpt_path)
    stats_sha = sha256_file(args.freq_stats_path)
    support_sha = sha256_file(args.support_manifest)
    query_sha = sha256_file(args.query_manifest)
    lock_mismatches, lock_sha = [], None
    try:
        _, lock_sha = validate_task_lock(args.manifest_lock, args.support_manifest, args.query_manifest)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        lock_mismatches.append(str(exc))
    checkpoint_meta, formal_mismatches = validate_formal_artifacts(
        checkpoint, stats.get("metadata"), stats_sha, args
    )
    formal_mismatches.extend(lock_mismatches)
    if formal_mismatches and not args.allow_nonformal_checkpoint:
        raise ValueError("Formal transfer validation failed: " + "; ".join(formal_mismatches))
    formal_result = not args.allow_nonformal_checkpoint and not formal_mismatches
    identity = result_identity(
        args, ckpt_sha, stats_sha, support_sha, query_sha, lock_sha, formal_result
    )
    if prepare_output_directory(args.output_dir, identity, args.overwrite):
        return

    model_mode = checkpoint_meta["model_mode"]
    if model_mode != "dual" and args.branch_mode != model_mode:
        raise ValueError(f"A {model_mode} checkpoint cannot use branch_mode={args.branch_mode}")
    set_seed(args.seed)
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
        if {row.get("target_generator") for row in dataset.rows} != {args.target_generator}:
            raise ValueError(f"{role} manifest target_generator mismatch.")
        if {int(row.get("seed", -1)) for row in dataset.rows} != {args.seed}:
            raise ValueError(f"{role} manifest seed mismatch.")
        if {row.get("task_role") for row in dataset.rows} != {role}:
            raise ValueError(f"{role} manifest task_role mismatch.")
    validate_manifest_rows(support_dataset.rows, query_dataset.rows)
    by_label = {0: [], 1: []}
    for index, row in enumerate(support_dataset.rows):
        by_label[int(row["label"])].append(index)
    query_label_counts = {
        label: sum(int(row["label"]) == label for row in query_dataset.rows) for label in (0, 1)
    }
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
            sigma_rgb, sigma_freq = compute_support_sigmas(
                support_rgb, support_freq, proto_rgb, proto_freq
            )
            alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=args.tau_r)
        loader = DataLoader(
            query_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=transfer_collate_fn,
        )
        result_rows, all_labels, all_scores, all_predictions = [], [], [], []
        ckpt_step = int(checkpoint.get("step", 0))
        for images, labels, metadata in loader:
            query_rgb, query_freq = encode_batch(model, images, device, args.use_fp16)
            logits = compute_query_logits(
                query_rgb=query_rgb.unsqueeze(0) if query_rgb is not None else None,
                query_freq=query_freq.unsqueeze(0) if query_freq is not None else None,
                proto_rgb=proto_rgb, proto_freq=proto_freq, alpha=alpha, tau=args.tau,
                branch_mode=args.branch_mode, model_mode=model_mode,
            )["logits"].squeeze(0)
            scores = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
            label_values = labels.numpy()
            predictions = (scores >= 0.5).astype(np.int64)
            for metadata_row, label, score, prediction in zip(metadata, label_values, scores, predictions):
                result_rows.append({
                    "dataset": args.dataset_name, "split": metadata_row.get("split", ""),
                    "target_generator": args.target_generator, "seed": args.seed,
                    "group_id": metadata_row.get("group_id", ""),
                    "image_path": metadata_row["image_path"],
                    "image_sha256": metadata_row["image_sha256"], "label": int(label),
                    "fake_score": float(score), "prediction": int(prediction),
                    "correct": int(prediction == label), "branch_mode": args.branch_mode,
                    "ckpt_step": ckpt_step, "ckpt_sha256": ckpt_sha,
                    "freq_stats_sha256": stats_sha, "formal_result": formal_result,
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
        "ACC": acc, "AP": ap, "AUC": auc, "acc": acc, "ap": ap, "auc": auc,
        "real_acc": float((predictions_np[real_mask] == 0).mean()),
        "fake_acc": float((predictions_np[fake_mask] == 1).mean()),
        "balanced_acc": float(
            0.5 * ((predictions_np[real_mask] == 0).mean() + (predictions_np[fake_mask] == 1).mean())
        ),
        "mean_real_score": float(scores_np[real_mask].mean()),
        "mean_fake_score": float(scores_np[fake_mask].mean()),
        "num_real_support": 10, "num_fake_support": 10,
        "num_real_query": int(real_mask.sum()), "num_fake_query": int(fake_mask.sum()),
        "formal_result": formal_result,
    }
    output_dir = Path(args.output_dir).resolve()
    with (output_dir / "per_image_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    run_config = vars(args).copy()
    run_config.update({"device": str(device), "checkpoint_metadata": checkpoint_meta})
    provenance = {
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "checkpoint_metadata": checkpoint_meta,
        "frequency_stats_metadata": stats.get("metadata"),
        "formal_checkpoint_validation": formal_result,
        "formal_result": formal_result,
        "formal_validation_mismatches": formal_mismatches,
        "allow_nonformal_checkpoint": args.allow_nonformal_checkpoint,
        "gradient_updates": False,
        "target_frequency_statistics_computed": False,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
