#!/usr/bin/env python3
"""Extract original MS COCOAI JPEG bytes and write an auditable base manifest."""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path


LABEL_B_TO_CLASS = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney_v6",
}
MANIFEST_FIELDS = [
    "split", "shard_index", "row_index_in_shard", "global_row_index",
    "caption", "caption_sha256", "label_a", "label_b", "class_name",
    "image_path", "original_path_field", "image_sha256", "image_num_bytes",
    "width", "height", "format", "mode",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", default="/root/autodl-tmp/MS_COCOAI")
    parser.add_argument("--output_root", default="/root/autodl-tmp/MS_COCOAI")
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_existing", action="store_true")
    return parser.parse_args()


def inspect_jpeg(image_bytes: bytes):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            image_format = image.format
            mode = image.mode
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Image bytes cannot be decoded: {exc}") from exc
    if image_format != "JPEG":
        raise ValueError(f"Expected original JPEG bytes, got format={image_format!r}")
    return width, height, image_format, mode


def write_original_bytes(path: Path, image_bytes: bytes, overwrite: bool, verify_existing: bool):
    expected_sha = sha256_bytes(image_bytes)
    if path.exists():
        existing_bytes = path.read_bytes()
        existing_sha = sha256_bytes(existing_bytes)
        if existing_sha != expected_sha:
            raise RuntimeError(
                f"Existing image SHA mismatch for {path}: {existing_sha} != {expected_sha}"
            )
        if verify_existing:
            inspect_jpeg(existing_bytes)
        print(f"Verified existing image: {path}")
        if not overwrite:
            return expected_sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    if sha256_bytes(path.read_bytes()) != expected_sha:
        raise IOError(f"Post-write SHA verification failed: {path}")
    return expected_sha


def main():
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max_rows must be >= 0")

    import pyarrow.parquet as pq

    split_root = Path(args.input_root) / args.split
    shards = sorted(split_root.glob(f"{args.split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {split_root}")

    output_root = Path(args.output_root).resolve()
    manifest_path = output_root / "manifests" / args.split / "base_rows.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for shard_index, shard_path in enumerate(shards):
            parquet = pq.ParquetFile(shard_path)
            row_index_in_shard = 0
            for row_group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(
                    row_group_index,
                    columns=["Caption", "Image", "Label_A", "Label_B"],
                )
                for record in table.to_pylist():
                    if args.max_rows and processed >= args.max_rows:
                        break
                    caption = str(record["Caption"])
                    label_a = int(record["Label_A"])
                    label_b = int(record["Label_B"])
                    if label_b not in LABEL_B_TO_CLASS:
                        raise ValueError(f"Unknown Label_B={label_b} at {shard_path}:{row_index_in_shard}")
                    expected_label_a = 0 if label_b == 0 else 1
                    if label_a != expected_label_a:
                        raise ValueError(
                            f"Label_A/Label_B mismatch at {shard_path}:{row_index_in_shard}: "
                            f"{label_a}/{label_b}"
                        )
                    image_struct = record["Image"]
                    if not isinstance(image_struct, dict):
                        raise TypeError(f"Image field is not a struct at {shard_path}:{row_index_in_shard}")
                    image_bytes = image_struct.get("bytes")
                    if not image_bytes:
                        raise ValueError(f"Empty Image.bytes at {shard_path}:{row_index_in_shard}")
                    image_bytes = bytes(image_bytes)
                    image_sha = sha256_bytes(image_bytes)
                    class_name = LABEL_B_TO_CLASS[label_b]
                    filename = f"{args.split}_{processed:08d}_{image_sha[:12]}.jpg"
                    image_path = output_root / "images" / args.split / class_name / filename
                    write_original_bytes(image_path, image_bytes, args.overwrite, args.verify_existing)
                    width, height, image_format, mode = inspect_jpeg(image_bytes)
                    writer.writerow({
                        "split": args.split,
                        "shard_index": shard_index,
                        "row_index_in_shard": row_index_in_shard,
                        "global_row_index": processed,
                        "caption": caption,
                        "caption_sha256": sha256_bytes(caption.encode("utf-8")),
                        "label_a": label_a,
                        "label_b": label_b,
                        "class_name": class_name,
                        "image_path": str(image_path),
                        "original_path_field": image_struct.get("path") or "",
                        "image_sha256": image_sha,
                        "image_num_bytes": len(image_bytes),
                        "width": width,
                        "height": height,
                        "format": image_format,
                        "mode": mode,
                    })
                    processed += 1
                    row_index_in_shard += 1
                if args.max_rows and processed >= args.max_rows:
                    break
            if args.max_rows and processed >= args.max_rows:
                break
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(Path(args.input_root).resolve()),
        "output_root": str(output_root),
        "split": args.split,
        "source_parquet_shards": [str(path.resolve()) for path in shards],
        "max_rows": args.max_rows,
        "rows_written": processed,
        "image_write_mode": "original Image.bytes; no image re-encoding",
        "existing_file_policy": "always SHA-256 verify; mismatch is fatal",
        "base_manifest": str(manifest_path),
    }
    provenance_path = manifest_path.parent / "extraction_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Extracted/verified {processed} rows; manifest={manifest_path}")


if __name__ == "__main__":
    main()
