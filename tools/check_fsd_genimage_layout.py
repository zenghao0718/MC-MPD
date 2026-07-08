# -*- coding: utf-8 -*-
"""Check the final 7-class FSD GenImage layout."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_ROOT = "/root/autodl-tmp/data_fsd_full/GenImage"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
SOURCE_TAGS = ("sdv14__", "sdv15__", "wukong__", "adm__", "biggan__", "glide__", "midjourney__", "vqdm__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the final FSD GenImage directory layout.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="FSD GenImage root.")
    parser.add_argument("--json_out", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--sample_open",
        type=int,
        default=5,
        help="Open this many sample images per leaf directory with PIL. Use 0 to skip.",
    )
    parser.add_argument(
        "--require_source_tag",
        action="store_true",
        help="Fail if image filenames do not contain a known source_tag__ prefix.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def expected_leaf(class_name: str) -> str:
    return "nature" if class_name == "real" else "ai"


def collect_dir_items(directory: Path) -> Tuple[List[Path], List[str], List[str]]:
    images: List[Path] = []
    non_images: List[str] = []
    broken_links: List[str] = []
    if not directory.exists():
        return images, non_images, broken_links

    for path in directory.rglob("*"):
        if path.is_symlink() and not path.exists():
            broken_links.append(str(path))
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in IMG_EXTENSIONS:
            images.append(path)
        else:
            non_images.append(str(path))
    images.sort(key=lambda path: str(path).lower())
    return images, non_images, broken_links


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_unexpected_split_files(split_root: Path, expected_dir: Path) -> Tuple[List[str], List[str]]:
    unexpected_images: List[str] = []
    unexpected_non_images: List[str] = []
    if not split_root.exists():
        return unexpected_images, unexpected_non_images

    for path in split_root.rglob("*"):
        if not path.is_file():
            continue
        if is_under(path, expected_dir):
            continue
        if path.suffix.lower() in IMG_EXTENSIONS:
            unexpected_images.append(str(path))
        else:
            unexpected_non_images.append(str(path))
    return unexpected_images, unexpected_non_images


def sample_open_images(paths: Sequence[Path], sample_count: int) -> List[str]:
    if sample_count <= 0:
        return []
    try:
        from PIL import Image
    except ImportError as exc:
        return [f"PIL is required for --sample_open > 0 but is not installed: {exc}"]

    bad: List[str] = []
    for path in list(paths)[:sample_count]:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            bad.append(f"{path}: {exc}")
    return bad


def has_source_tag(path: Path) -> bool:
    return path.name.startswith(SOURCE_TAGS)


def check_class(root: Path, class_name: str, sample_open: int, require_source_tag: bool) -> Dict[str, object]:
    leaf = expected_leaf(class_name)
    class_report: Dict[str, object] = {
        "class": class_name,
        "leaf": leaf,
        "train_count": 0,
        "val_count": 0,
        "status": "OK",
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    errors: List[str] = []
    warnings: List[str] = []

    class_dir = root / class_name
    if not class_dir.is_dir():
        errors.append(f"Missing class directory: {class_dir}")

    for split in ("train", "val"):
        directory = class_dir / split / leaf
        split_report: Dict[str, object] = {
            "path": str(directory),
            "exists": directory.is_dir(),
            "image_count": 0,
            "non_image_files": [],
            "broken_links": [],
            "bad_images": [],
            "missing_source_tag_examples": [],
            "unexpected_image_files": [],
            "unexpected_non_image_files": [],
        }
        if not directory.is_dir():
            errors.append(f"Missing required directory: {directory}")
            class_report["splits"][split] = split_report
            continue

        images, non_images, broken_links = collect_dir_items(directory)
        unexpected_images, unexpected_non_images = collect_unexpected_split_files(class_dir / split, directory)
        bad_images = sample_open_images(images, sample_open)
        missing_tags = [str(path) for path in images if not has_source_tag(path)]
        split_report["image_count"] = len(images)
        split_report["non_image_files"] = non_images[:20]
        split_report["broken_links"] = broken_links[:20]
        split_report["bad_images"] = bad_images[:20]
        split_report["missing_source_tag_examples"] = missing_tags[:20]
        split_report["unexpected_image_files"] = unexpected_images[:20]
        split_report["unexpected_non_image_files"] = unexpected_non_images[:20]
        class_report[f"{split}_count"] = len(images)

        if not images:
            errors.append(f"No images found in required directory: {directory}")
        if non_images:
            errors.append(f"Non-image files found in {directory}: {len(non_images)}")
        if unexpected_images:
            errors.append(
                f"Unexpected image files outside {directory.name}/ under {class_dir / split}: "
                f"{len(unexpected_images)}"
            )
        if unexpected_non_images:
            errors.append(
                f"Unexpected non-image files outside {directory.name}/ under {class_dir / split}: "
                f"{len(unexpected_non_images)}"
            )
        if broken_links:
            errors.append(f"Broken links found in {directory}: {len(broken_links)}")
        if bad_images:
            errors.append(f"Bad image samples found in {directory}: {len(bad_images)}")
        if missing_tags:
            message = f"Files without source_tag__ prefix in {directory}: {len(missing_tags)}"
            if require_source_tag:
                errors.append(message)
            else:
                warnings.append(message)

        class_report["splits"][split] = split_report

    class_report["errors"] = errors
    class_report["warnings"] = warnings
    class_report["status"] = "OK" if not errors else "ERROR"
    return class_report


def print_table(reports: Sequence[Dict[str, object]]) -> None:
    header = f"{'class':<12} {'train_count':>12} {'val_count':>12} {'status'}"
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report['class']:<12} {report['train_count']:>12} "
            f"{report['val_count']:>12} {report['status']}"
        )


def main() -> int:
    args = parse_args()
    setup_logging()

    root = Path(args.root).expanduser()
    if not root.exists():
        logging.error("FSD GenImage root does not exist: %s", root)
        return 2
    if not root.is_dir():
        logging.error("FSD GenImage root is not a directory: %s", root)
        return 2

    reports = [
        check_class(root, class_name, args.sample_open, args.require_source_tag)
        for class_name in EXPECTED_CLASSES
    ]
    errors: List[str] = []
    warnings: List[str] = []
    for report in reports:
        errors.extend(str(item) for item in report["errors"])
        warnings.extend(str(item) for item in report["warnings"])

    print()
    print_table(reports)
    print()
    for warning in warnings[:50]:
        logging.warning("%s", warning)
    if len(warnings) > 50:
        logging.warning("... %d more warnings omitted", len(warnings) - 50)

    output = {
        "root": str(root),
        "classes": reports,
        "errors": errors,
        "warnings": warnings,
        "status": "OK" if not errors else "ERROR",
    }
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("JSON report written: %s", json_path)

    if errors:
        logging.error("FSD GenImage layout check failed:")
        for error in errors[:100]:
            logging.error("  %s", error)
        if len(errors) > 100:
            logging.error("  ... %d more errors omitted", len(errors) - 100)
        return 1

    logging.info("FSD GenImage layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
