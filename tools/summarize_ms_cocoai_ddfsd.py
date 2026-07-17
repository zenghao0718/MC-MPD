#!/usr/bin/env python3
"""Independently recompute and summarize DDFSD MS COCOAI per-image results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


GENERATOR_ORDER = ("sd21", "sdxl", "sd3", "dalle3", "midjourney_v6")
DISPLAY = {
    "sd21": "SD2.1", "sdxl": "SDXL", "sd3": "SD3",
    "dalle3": "DALL-E 3", "midjourney_v6": "Midjourney v6",
}
METHOD = "DDFSD-AllSource-15000"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    score_files = sorted(Path(args.input_root).rglob("per_image_scores.csv"))
    if not score_files:
        raise FileNotFoundError(f"No per_image_scores.csv found under {args.input_root}")
    grouped = defaultdict(list)
    for score_file in score_files:
        with score_file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty score file: {score_file}")
        formal_values = {str(row.get("formal_result", "")).lower() for row in rows}
        if formal_values != {"true"}:
            raise ValueError(
                f"Refusing to summarize non-formal or unlabeled results: {score_file}: {formal_values}"
            )
        keys = {(row["target_generator"], int(row["seed"])) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"Score file mixes tasks: {score_file}: {keys}")
        task_key = next(iter(keys))
        if task_key in grouped:
            raise ValueError(f"Duplicate per-image score file for task {task_key}: {score_file}")
        grouped[task_key] = rows

    expected = {(generator, seed) for generator in GENERATOR_ORDER for seed in (42, 101, 102, 103, 104)}
    if set(grouped) != expected:
        missing, extra = sorted(expected - set(grouped)), sorted(set(grouped) - expected)
        raise ValueError(f"Expected 25 formal tasks; missing={missing}, extra={extra}")
    checkpoint_hashes = {
        row["ckpt_sha256"] for rows in grouped.values() for row in rows
    }
    frequency_hashes = {
        row["freq_stats_sha256"] for rows in grouped.values() for row in rows
    }
    if len(checkpoint_hashes) != 1 or len(frequency_hashes) != 1:
        raise ValueError(
            "Formal tasks must share one checkpoint and one frequency-statistics file; "
            f"checkpoint_hashes={checkpoint_hashes}, frequency_hashes={frequency_hashes}"
        )

    per_seed = []
    for (generator, seed), rows in sorted(grouped.items()):
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        scores = np.asarray([float(row["fake_score"]) for row in rows], dtype=np.float64)
        predictions = (scores >= 0.5).astype(np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"Task {generator}/{seed} lacks one binary class")
        per_seed.append({
            "method": METHOD, "target_generator": generator, "seed": seed,
            "acc": float((predictions == labels).mean()),
            "ap": float(average_precision_score(labels, scores)),
            "auc": float(roc_auc_score(labels, scores)),
            "num_images": len(rows),
        })

    by_generator = []
    for generator in GENERATOR_ORDER:
        rows = [row for row in per_seed if row["target_generator"] == generator]
        record = {"method": METHOD, "target_generator": generator}
        for metric in ("acc", "ap", "auc"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=0))
        record["num_seeds"] = len(rows)
        by_generator.append(record)

    macro = {"method": METHOD}
    for metric in ("acc", "ap", "auc"):
        macro[metric] = float(np.mean([row[f"{metric}_mean"] for row in by_generator]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ddfsd_per_seed_per_generator.csv", per_seed, list(per_seed[0]))
    write_csv(output_dir / "ddfsd_per_generator_mean_std.csv", by_generator, list(by_generator[0]))
    write_csv(output_dir / "ddfsd_macro.csv", [macro], list(macro))

    for metric in ("acc", "ap", "auc"):
        cells = []
        for generator in GENERATOR_ORDER:
            row = next(item for item in by_generator if item["target_generator"] == generator)
            cells.append(f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}")
        cells.append(f"{macro[metric]:.4f}")
        header = "| Method | " + " | ".join(DISPLAY[name] for name in GENERATOR_ORDER) + " | Macro |\n"
        divider = "|---|" + "---:|" * 6 + "\n"
        table = header + divider + "| " + METHOD + " | " + " | ".join(cells) + " |\n"
        (output_dir / f"ddfsd_{metric}_table.md").write_text(table, encoding="utf-8")
    provenance = {
        "input_root": str(Path(args.input_root).resolve()),
        "score_files": [str(path.resolve()) for path in score_files],
        "metrics_recomputed_from_per_image_csv": ["acc", "ap", "auc"],
        "std_ddof": 0,
        "macro": "equal-weight mean over five generator means",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "frequency_stats_sha256": next(iter(frequency_hashes)),
        "formal_results_only": True,
    }
    (output_dir / "summary_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
