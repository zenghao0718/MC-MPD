#!/usr/bin/env python3
"""Build stable six-image semantic groups from an MS COCOAI base manifest."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


LABELS = tuple(range(6))
LABEL_NAMES = ("real", "sd21", "sdxl", "sd3", "dalle3", "midjourney_v6")


def stable_key(row):
    return tuple(int(row[name]) for name in ("shard_index", "row_index_in_shard", "global_row_index"))


def build_groups(rows):
    """Return (grouped_rows, group_rows, anomalies) using caption occurrence pairing."""

    by_caption = defaultdict(list)
    for row in rows:
        by_caption[row["caption"]].append(dict(row))
    grouped_rows, group_rows, anomalies = [], [], []
    seen_ids = {}
    for caption in sorted(by_caption):
        caption_rows = by_caption[caption]
        expected_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        recorded_shas = {row["caption_sha256"] for row in caption_rows}
        by_label = {label: [] for label in LABELS}
        invalid_labels = []
        for row in caption_rows:
            try:
                label = int(row["label_b"])
            except (TypeError, ValueError):
                invalid_labels.append(row.get("label_b"))
                continue
            if label not in by_label:
                invalid_labels.append(label)
            else:
                by_label[label].append(row)
        counts = {label: len(by_label[label]) for label in LABELS}
        reasons = []
        if recorded_shas != {expected_sha}:
            reasons.append(f"caption_sha256 mismatch: {sorted(recorded_shas)} expected {expected_sha}")
        if invalid_labels:
            reasons.append(f"invalid Label_B values: {invalid_labels}")
        if len(set(counts.values())) != 1 or not counts[0]:
            reasons.append(f"Label_B counts are not equal and positive: {counts}")
        if reasons:
            anomalies.append({
                "caption_sha256": expected_sha,
                "caption": caption,
                "reason": "; ".join(reasons),
                "label_counts": ";".join(f"{key}:{value}" for key, value in counts.items()),
            })
            continue
        for label in LABELS:
            by_label[label].sort(key=stable_key)
        for occurrence_index in range(counts[0]):
            group_id = f"{expected_sha[:16]}_{occurrence_index}"
            previous_caption = seen_ids.setdefault(group_id, caption)
            if previous_caption != caption:
                anomalies.append({
                    "caption_sha256": expected_sha,
                    "caption": caption,
                    "reason": f"group_id collision with caption {previous_caption!r}",
                    "label_counts": ";".join(f"{key}:{value}" for key, value in counts.items()),
                })
                continue
            aligned = [dict(by_label[label][occurrence_index]) for label in LABELS]
            if {int(row["label_b"]) for row in aligned} != set(LABELS):
                anomalies.append({
                    "caption_sha256": expected_sha,
                    "caption": caption,
                    "reason": f"group {group_id} does not contain Label_B 0..5",
                    "label_counts": ";".join(f"{key}:{value}" for key, value in counts.items()),
                })
                continue
            group_record = {
                "group_id": group_id,
                "caption_sha256": expected_sha,
                "caption": caption,
                "occurrence_index": occurrence_index,
                "num_rows": 6,
            }
            for label, row in zip(LABELS, aligned):
                row["occurrence_index"] = occurrence_index
                row["group_id"] = group_id
                grouped_rows.append(row)
                class_name = LABEL_NAMES[label]
                group_record[f"{class_name}_image_path"] = row["image_path"]
                group_record[f"{class_name}_image_sha256"] = row["image_sha256"]
            group_rows.append(group_record)
    return grouped_rows, group_rows, anomalies


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_anomalies", action="store_true")
    args = parser.parse_args()
    with open(args.base_manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Base manifest is empty.")
    grouped, groups, anomalies = build_groups(rows)
    output_dir = Path(args.output_dir)
    grouped_fields = list(rows[0]) + ["occurrence_index", "group_id"]
    group_fields = ["group_id", "caption_sha256", "caption", "occurrence_index", "num_rows"]
    for name in LABEL_NAMES:
        group_fields.extend([f"{name}_image_path", f"{name}_image_sha256"])
    anomaly_fields = ["caption_sha256", "caption", "reason", "label_counts"]
    write_csv(output_dir / "all_rows_grouped.csv", grouped, grouped_fields)
    write_csv(output_dir / "groups.csv", groups, group_fields)
    write_csv(output_dir / "group_anomalies.csv", anomalies, anomaly_fields)
    provenance = {
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "pairing_algorithm": "caption -> Label_B -> stable(shard,row,global) -> kth occurrence",
        "group_id": "caption_sha256[:16] + '_' + zero_based_occurrence_index",
        "valid_groups": len(groups),
        "anomalies": len(anomalies),
        "allow_anomalies": args.allow_anomalies,
    }
    (output_dir / "grouping_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"valid_groups={len(groups)} anomalies={len(anomalies)} output={output_dir}")
    if anomalies and not args.allow_anomalies:
        raise RuntimeError(
            f"Found {len(anomalies)} grouping anomalies; see {output_dir / 'group_anomalies.csv'}"
        )


if __name__ == "__main__":
    main()
