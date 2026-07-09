#!/usr/bin/env python
"""Build a Resize(256) copy of GenImage without touching source files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DEFAULT_SRC = "/root/autodl-tmp/GenImage"
DEFAULT_DST = "/root/autodl-tmp/GenImage_resize256"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LOG_NAME = "preprocess_resize256_log.json"


@dataclass(frozen=True)
class WorkItem:
    src: Path
    dst: Path


@dataclass(frozen=True)
class Result:
    status: str
    src: str
    dst: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize GenImage images so the shorter edge equals --size."
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"Source dataset root. Default: {DEFAULT_SRC}")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"Output dataset root. Default: {DEFAULT_DST}")
    parser.add_argument("--size", type=int, default=256, help="Target shorter-edge size. Default: 256")
    parser.add_argument("--workers", type=int, default=8, help="Thread workers. Default: 8")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output image files.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report only; do not write files.")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def normalize_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def validate_paths(src: Path, dst: Path, size: int, workers: int) -> tuple[Path, Path, int]:
    src_abs = Path(os.path.abspath(src))
    dst_abs = Path(os.path.abspath(dst))
    src_norm = normalize_path(src_abs)
    dst_norm = normalize_path(dst_abs)

    if size <= 0:
        raise ValueError(f"--size must be positive, got {size}")
    if workers <= 0:
        raise ValueError(f"--workers must be positive, got {workers}")
    if not src_abs.exists():
        raise ValueError(f"Source directory does not exist: {src_abs}")
    if not src_abs.is_dir():
        raise ValueError(f"Source path is not a directory: {src_abs}")
    if src_norm == dst_norm:
        raise ValueError("--src and --dst point to the same directory; refusing to continue.")

    common = os.path.commonpath([src_norm, dst_norm])
    if common == src_norm:
        raise ValueError("--dst is inside --src; refusing to avoid recursive self-processing.")

    return src_abs, dst_abs, workers


def scan_images(src: Path, dst: Path) -> list[WorkItem]:
    items: list[WorkItem] = []
    for source_path in src.rglob("*"):
        if not is_image_file(source_path):
            continue
        relative_path = source_path.relative_to(src)
        items.append(WorkItem(src=source_path, dst=dst / relative_path))
    return items


def resized_size(width: int, height: int, target_short_edge: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")

    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge == target_short_edge:
        return width, height

    new_long_edge = int(target_short_edge * long_edge / short_edge)
    if width <= height:
        return target_short_edge, new_long_edge
    return new_long_edge, target_short_edge


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    return image.convert("RGB")


def save_image(image: Image.Image, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, format="JPEG", quality=95)
    elif suffix == ".png":
        image.save(output_path, format="PNG")
    elif suffix == ".bmp":
        image.save(output_path, format="BMP")
    elif suffix == ".webp":
        image.save(output_path, format="WEBP", quality=95)
    elif suffix in {".tif", ".tiff"}:
        image.save(output_path, format="TIFF")
    else:
        image.save(output_path)


def process_one(item: WorkItem, size: int, overwrite: bool) -> Result:
    try:
        if item.dst.exists() and not overwrite:
            return Result(status="skipped", src=str(item.src), dst=str(item.dst))

        item.dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(item.src) as image:
            image.load()
            image = to_rgb(image)
            new_size = resized_size(image.width, image.height, size)
            if new_size != image.size:
                image = image.resize(new_size, Image.Resampling.BILINEAR)
            save_image(image, item.dst)

        return Result(status="success", src=str(item.src), dst=str(item.dst))
    except Exception as exc:  # Keep one bad image from stopping the whole run.
        return Result(status="failed", src=str(item.src), dst=str(item.dst), error=repr(exc))


def print_dry_run(items: list[WorkItem], overwrite: bool) -> None:
    existing_outputs = sum(1 for item in items if item.dst.exists())
    would_skip = 0 if overwrite else existing_outputs
    would_process = len(items) - would_skip

    print("=" * 72)
    print("Resize(256) dry run")
    print("=" * 72)
    print(f"Total source images: {len(items)}")
    print(f"Existing output files: {existing_outputs}")
    print(f"Would process: {would_process}")
    print(f"Would skip: {would_skip}")
    print()
    print("Output path examples:")
    for item in items[:10]:
        print(f"  {item.src} -> {item.dst}")
    if len(items) > 10:
        print(f"  ... {len(items) - 10} more")
    print()
    print("Dry-run mode: no directories, images, or log files were written.")


def write_log(dst: Path, payload: dict[str, Any]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    log_path = dst / LOG_NAME
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    start_time = datetime.now()
    start_counter = time.perf_counter()

    try:
        src, dst, workers = validate_paths(Path(args.src), Path(args.dst), args.size, args.workers)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    items = scan_images(src, dst)

    print(f"Source: {src}")
    print(f"Destination: {dst}")
    print(f"Target shorter edge: {args.size}")
    print(f"Workers: {workers}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Scanned images: {len(items)}")
    print()

    if args.dry_run:
        print_dry_run(items, args.overwrite)
        return 0

    successes = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, item, args.size, args.overwrite) for item in items]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Resize({args.size})",
        ):
            result = future.result()
            if result.status == "success":
                successes += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                failures.append({"src": result.src, "dst": result.dst, "error": result.error or ""})

    end_time = datetime.now()
    elapsed_seconds = time.perf_counter() - start_counter
    log_payload: dict[str, Any] = {
        "src": str(src),
        "dst": str(dst),
        "size": args.size,
        "workers": workers,
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "total_images": len(items),
        "success_count": successes,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "failed_files": failures,
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    write_log(dst, log_payload)

    print()
    print("Done.")
    print(f"  Total images: {len(items)}")
    print(f"  Success: {successes}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {len(failures)}")
    print(f"  Log: {dst / LOG_NAME}")

    if failures:
        print()
        print("Failed files:")
        for failure in failures:
            print(f"  - {failure['src']}: {failure['error']}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
