#!/usr/bin/env python
"""Read-only layout checker for the GenImage dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = r"D:\data\GenImage\GenImage"
AI_CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
REAL_CLASS = "real"
ALL_CLASSES = (*AI_CLASSES, REAL_CLASS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def count_images(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("*") if is_image_file(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the expected GenImage directory layout without modifying files."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"Dataset root. Default: {DEFAULT_ROOT}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    errors: list[str] = []
    warnings: list[str] = []
    total_images = 0

    print("=" * 72)
    print("GenImage layout check")
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Supported suffixes: {', '.join(sorted(IMAGE_SUFFIXES))}")
    print()

    if not root.exists():
        errors.append(f"Root directory is missing: {root}")
    elif not root.is_dir():
        errors.append(f"Root path is not a directory: {root}")

    print("Class directories:")
    for class_name in ALL_CLASSES:
        class_dir = root / class_name
        if class_dir.is_dir():
            print(f"  [OK]      {class_dir}")
        else:
            print(f"  [MISSING] {class_dir}")
            errors.append(f"Missing class directory: {class_dir}")
    print()

    print("Expected image directories:")

    for class_name in AI_CLASSES:
        for split in ("train", "val"):
            directory = root / class_name / split / "ai"
            if directory.is_dir():
                image_count = count_images(directory)
                total_images += image_count
                print(f"  [OK]      {directory}  images={image_count}")
            else:
                print(f"  [MISSING] {directory}  images=0")
                errors.append(f"Missing AI directory: {directory}")

    for split in ("train", "val"):
        directory = root / REAL_CLASS / split / "nature"
        if directory.is_dir():
            image_count = count_images(directory)
            total_images += image_count
            print(f"  [OK]      {directory}  images={image_count}")
        else:
            print(f"  [MISSING] {directory}  images=0")
            message = f"Missing real directory: {directory}"
            if split == "val":
                warnings.append(message)
            else:
                errors.append(message)

    print()
    print(f"Total images in expected directories: {total_images}")
    print()

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Preprocessing prerequisite: NOT SATISFIED")
        return 1

    print("Preprocessing prerequisite: SATISFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
