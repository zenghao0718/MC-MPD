#!/usr/bin/env python3
"""Create or verify frozen MS COCOAI 10-shot support/query manifests."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
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


def partition_support_groups(support_pool, seeds=SEEDS, groups_per_seed=10):
    required = len(seeds) * groups_per_seed
    if len(support_pool) != required or len(set(support_pool)) != required:
        raise ValueError(f"Support pool must contain exactly {required} unique groups.")
    partition = {
        seed: list(support_pool[index * groups_per_seed:(index + 1) * groups_per_seed])
        for index, seed in enumerate(seeds)
    }
    if len(set().union(*(set(value) for value in partition.values()))) != required:
        raise ValueError("Seed support-group sets are not mutually disjoint.")
    return partition


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    invalid = []
    for group_id, items in groups.items():
        hashes = [row["image_sha256"] for row in items.values()]
        if set(items) != set(range(6)) or len(set(hashes)) != 6:
            invalid.append(group_id)
    if invalid:
        raise ValueError(f"Grouped manifest contains incomplete/SHA-duplicate groups, first={invalid[:5]}")
    return dict(groups)


def validate_task(support, query):
    for name, rows in (("support", support), ("query", query)):
        paths = [row["image_path"] for row in rows]
        hashes = [row["image_sha256"] for row in rows]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{name} contains duplicate image paths.")
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"{name} contains duplicate image SHA-256 values.")
        labels = [int(row["label"]) for row in rows]
        if labels.count(0) != labels.count(1):
            raise ValueError(f"{name} real/fake counts are not balanced.")
    support_labels = [int(row["label"]) for row in support]
    if support_labels.count(0) != 10 or support_labels.count(1) != 10:
        raise ValueError("Support must contain exactly 10 real + 10 fake images.")
    checks = (
        ("image paths", {row["image_path"] for row in support}, {row["image_path"] for row in query}),
        ("group IDs", {row["group_id"] for row in support}, {row["group_id"] for row in query}),
        ("image SHA-256", {row["image_sha256"] for row in support}, {row["image_sha256"] for row in query}),
    )
    for label, support_values, query_values in checks:
        if support_values & query_values:
            raise ValueError(f"Support/query {label} overlap.")


def _file_record(path):
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def verify_manifest_lock(lock_path):
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for record in lock.get("files", []):
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Locked manifest file is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"Manifest lock SHA mismatch: {path}: {actual} != {record['sha256']}")
    freeze_path = lock_path.parent / "manifest_sha256.txt"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Missing manifest SHA freeze file: {freeze_path}")
    frozen = {}
    for line in freeze_path.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        frozen[str(Path(path).resolve())] = digest
    expected = {record["path"]: record["sha256"] for record in lock["files"]}
    expected[str(lock_path)] = sha256_file(lock_path)
    if frozen != expected:
        raise RuntimeError("manifest_sha256.txt does not exactly match manifest_lock.json and its files.")
    return lock


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped_manifest")
    parser.add_argument("--output_dir")
    parser.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--master_seed", type=int, default=20260717)
    parser.add_argument("--verify_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_lock:
        verify_manifest_lock(args.verify_lock)
        print(f"Manifest lock verified: {Path(args.verify_lock).resolve()}")
        return
    if not args.grouped_manifest or not args.output_dir or not args.split:
        raise ValueError("Generation requires --grouped_manifest, --output_dir, and --split.")

    groups = load_complete_groups(args.grouped_manifest)
    output_dir = Path(args.output_dir).resolve()
    index_rows, manifest_paths = [], []
    created_at = datetime.now(timezone.utc).isoformat()

    if args.split == "test":
        support_pool, query_group_ids = choose_support_groups(groups, args.master_seed, 50)
        support_by_seed = partition_support_groups(support_pool)
        query_signatures, real_query_signature = {}, None
        for generator, fake_label in GENERATORS.items():
            for seed in SEEDS:
                support = task_rows(groups, support_by_seed[seed], fake_label, "support", generator, seed)
                query = task_rows(groups, query_group_ids, fake_label, "query", generator, seed)
                validate_task(support, query)
                signature = tuple((row["group_id"], row["label_b"], row["image_sha256"]) for row in query)
                if query_signatures.setdefault(generator, signature) != signature:
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
                manifest_paths.extend([support_path, query_path])
                index_rows.append({
                    "split": args.split, "generator": generator,
                    "display_name": DISPLAY_NAMES[generator], "seed": seed,
                    "support_manifest": str(support_path), "query_manifest": str(query_path),
                    "support_groups": 10, "real_query": len(query) // 2,
                    "fake_query": len(query) // 2,
                })
        verify_rows(row for group_rows in groups.values() for row in group_rows.values())
    else:
        support_pool, remaining = choose_support_groups(groups, args.master_seed, 10)
        support_by_seed = {42: support_pool}
        if len(remaining) < 100:
            raise ValueError(f"Validation smoke needs 100 query groups, got {len(remaining)}")
        query_group_ids = remaining[:100]
        generator, seed = "dalle3", 42
        support = task_rows(groups, support_pool, GENERATORS[generator], "support", generator, seed)
        query = task_rows(groups, query_group_ids, GENERATORS[generator], "query", generator, seed)
        validate_task(support, query)
        task_dir = output_dir / generator / f"seed_{seed}"
        support_path, query_path = task_dir / "support.csv", task_dir / "query.csv"
        write_manifest(support_path, support)
        write_manifest(query_path, query)
        manifest_paths.extend([support_path, query_path])
        index_rows.append({
            "split": args.split, "generator": generator,
            "display_name": DISPLAY_NAMES[generator], "seed": seed,
            "support_manifest": str(support_path), "query_manifest": str(query_path),
            "support_groups": 10, "real_query": 100, "fake_query": 100,
        })
        verify_rows(support + query)

    index_path = output_dir / "manifest_index.csv"
    support_groups_path = output_dir / "support_pool_groups.csv"
    query_groups_path = output_dir / "query_groups.csv"
    provenance_path = output_dir / "manifest_provenance.json"
    write_manifest(index_path, index_rows)
    write_manifest(support_groups_path, [
        {"seed": seed, "support_rank": rank, "group_id": group_id}
        for seed, group_ids in support_by_seed.items()
        for rank, group_id in enumerate(group_ids)
    ])
    write_manifest(query_groups_path, [
        {"query_rank": rank, "group_id": group_id}
        for rank, group_id in enumerate(query_group_ids)
    ])
    provenance = {
        "created_at_utc": created_at,
        "grouped_manifest": str(Path(args.grouped_manifest).resolve()),
        "grouped_manifest_sha256": sha256_file(args.grouped_manifest),
        "split": args.split,
        "master_seed": args.master_seed,
        "seeds": list(SEEDS) if args.split == "test" else [42],
        "generators": list(GENERATORS) if args.split == "test" else ["dalle3"],
        "support_shot_per_class": 10,
        "file_sha_verification": True,
        "num_tasks": len(index_rows),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    locked_paths = [
        Path(args.grouped_manifest), index_path, provenance_path,
        support_groups_path, query_groups_path, *manifest_paths,
    ]
    lock = {
        "split": args.split,
        "master_seed": args.master_seed,
        "seeds": list(SEEDS) if args.split == "test" else [42],
        "generators": list(GENERATORS) if args.split == "test" else ["dalle3"],
        "support_groups": {str(seed): values for seed, values in support_by_seed.items()},
        "query_groups": query_group_ids,
        "created_at_utc": created_at,
        "files": [_file_record(path) for path in locked_paths],
    }
    lock_path = output_dir / "manifest_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    freeze_records = list(lock["files"]) + [_file_record(lock_path)]
    freeze_path = output_dir / "manifest_sha256.txt"
    freeze_path.write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in freeze_records),
        encoding="utf-8",
    )
    verify_manifest_lock(lock_path)
    print(f"Wrote and locked {len(index_rows)} tasks under {output_dir}")


if __name__ == "__main__":
    main()
