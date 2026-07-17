#!/usr/bin/env python3
"""Create fixed MS COCOAI 10-shot support/query manifests."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


SEEDS = (42, 101, 102, 103, 104)
GENERATORS = {"sd21": 1, "sdxl": 2, "sd3": 3, "dalle3": 4, "midjourney_v6": 5}
DISPLAY_NAMES = {
    "sd21": "SD2.1", "sdxl": "SDXL", "sd3": "SD3",
    "dalle3": "DALL-E 3", "midjourney_v6": "Midjourney v6",
}


def choose_support_groups(group_ids, master_seed, count):
    ordered = sorted(group_ids)
    if len(ordered) < count:
        raise ValueError(f"Need at least {count} complete groups, got {len(ordered)}")
    random.Random(master_seed).shuffle(ordered)
    return ordered[:count], ordered[count:]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_rows(rows):
    verified = {}
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {path}")
        actual_sha = verified.get(str(path))
        if actual_sha is None:
            actual_sha = sha256_file(path)
            verified[str(path)] = actual_sha
        if actual_sha != row["image_sha256"]:
            raise RuntimeError(f"Image SHA mismatch: {path}: {actual_sha} != {row['image_sha256']}")


def task_rows(groups, selected_ids, fake_label, role, target_generator, seed):
    output = []
    for group_id in selected_ids:
        for label in (0, fake_label):
            row = dict(groups[group_id][label])
            row.update({
                "label": 0 if label == 0 else 1,
                "task_role": role,
                "target_generator": target_generator,
                "seed": seed,
            })
            output.append(row)
    return output


def write_manifest(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_complete_groups(path):
    groups = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label_b"])
            if label in groups[row["group_id"]]:
                raise ValueError(f"Duplicate Label_B={label} in group {row['group_id']}")
            groups[row["group_id"]][label] = row
    invalid = [group_id for group_id, items in groups.items() if set(items) != set(range(6))]
    if invalid:
        raise ValueError(f"Grouped manifest contains incomplete groups, first={invalid[:5]}")
    return dict(groups)


def validate_task(support, query):
    support_paths = {row["image_path"] for row in support}
    query_paths = {row["image_path"] for row in query}
    if support_paths & query_paths:
        raise ValueError("Support/query image paths overlap.")
    support_groups = {row["group_id"] for row in support}
    query_groups = {row["group_id"] for row in query}
    if support_groups & query_groups:
        raise ValueError("Support/query group IDs overlap.")
    for name, rows in (("support", support), ("query", query)):
        labels = [int(row["label"]) for row in rows]
        if labels.count(0) != labels.count(1):
            raise ValueError(f"{name} real/fake counts are not balanced.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--master_seed", type=int, default=20260717)
    args = parser.parse_args()
    groups = load_complete_groups(args.grouped_manifest)
    output_dir = Path(args.output_dir)
    index_rows = []

    if args.split == "test":
        support_pool, query_group_ids = choose_support_groups(groups, args.master_seed, 50)
        support_by_seed = {
            seed: support_pool[index * 10:(index + 1) * 10]
            for index, seed in enumerate(SEEDS)
        }
        if len(set().union(*(set(value) for value in support_by_seed.values()))) != 50:
            raise AssertionError("The five seed support-group sets are not disjoint.")
        query_signatures = {}
        real_query_signature = None
        for generator, fake_label in GENERATORS.items():
            for seed in SEEDS:
                support = task_rows(groups, support_by_seed[seed], fake_label, "support", generator, seed)
                query = task_rows(groups, query_group_ids, fake_label, "query", generator, seed)
                validate_task(support, query)
                signature = tuple((row["group_id"], row["label_b"], row["image_sha256"]) for row in query)
                previous = query_signatures.setdefault(generator, signature)
                if previous != signature:
                    raise AssertionError(f"Query differs across seeds for {generator}")
                current_real = tuple(row["image_sha256"] for row in query if int(row["label"]) == 0)
                if real_query_signature is None:
                    real_query_signature = current_real
                elif real_query_signature != current_real:
                    raise AssertionError("Generators do not share the same real query set.")
                task_dir = output_dir / generator / f"seed_{seed}"
                support_path, query_path = task_dir / "support.csv", task_dir / "query.csv"
                write_manifest(support_path, support)
                write_manifest(query_path, query)
                index_rows.append({
                    "split": args.split, "generator": generator, "display_name": DISPLAY_NAMES[generator],
                    "seed": seed, "support_manifest": str(support_path.resolve()),
                    "query_manifest": str(query_path.resolve()), "support_groups": 10,
                    "real_query": len(query) // 2, "fake_query": len(query) // 2,
                })
    else:
        shuffled, remaining = choose_support_groups(groups, args.master_seed, 10)
        if len(remaining) < 100:
            raise ValueError(f"Validation smoke needs 100 query groups, got {len(remaining)}")
        query_group_ids = remaining[:100]
        generator, seed = "dalle3", 42
        support = task_rows(groups, shuffled, GENERATORS[generator], "support", generator, seed)
        query = task_rows(groups, query_group_ids, GENERATORS[generator], "query", generator, seed)
        validate_task(support, query)
        task_dir = output_dir / generator / f"seed_{seed}"
        support_path, query_path = task_dir / "support.csv", task_dir / "query.csv"
        write_manifest(support_path, support)
        write_manifest(query_path, query)
        index_rows.append({
            "split": args.split, "generator": generator, "display_name": DISPLAY_NAMES[generator],
            "seed": seed, "support_manifest": str(support_path.resolve()),
            "query_manifest": str(query_path.resolve()), "support_groups": 10,
            "real_query": 100, "fake_query": 100,
        })

    if args.split == "test":
        verify_rows(
            row for group_rows in groups.values() for row in group_rows.values()
        )
    else:
        verify_rows(support + query)
    write_manifest(output_dir / "manifest_index.csv", index_rows)
    provenance = {
        "grouped_manifest": str(Path(args.grouped_manifest).resolve()),
        "grouped_manifest_sha256": sha256_file(args.grouped_manifest),
        "split": args.split,
        "master_seed": args.master_seed,
        "seeds": list(SEEDS) if args.split == "test" else [42],
        "support_shot_per_class": 10,
        "file_sha_verification": True,
        "num_tasks": len(index_rows),
    }
    with (output_dir / "manifest_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)
    print(f"Wrote and verified {len(index_rows)} tasks under {output_dir}")


if __name__ == "__main__":
    main()
