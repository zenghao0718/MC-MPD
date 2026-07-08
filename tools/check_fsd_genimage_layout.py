# -*- coding: utf-8 -*-
"""Check a 7-class FSD GenImage layout and optional manifests/samples."""

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


DEFAULT_ROOT = "/root/autodl-tmp/data_fsd_full/GenImage"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
SOURCE_TAGS = ("sdv14__", "sdv15__", "wukong__", "adm__", "biggan__", "glide__", "midjourney__", "vqdm__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the final FSD GenImage directory layout.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="FSD GenImage root to check.")
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
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional _build_manifest.json or _sample_manifest.json to compare leaf counts against.",
    )
    parser.add_argument(
        "--compare_root",
        default=None,
        help="Optional full FSD root used to verify that --root is a deterministic sampled subset.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Sampling ratio used with --compare_root. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed used with --compare_root. Default: 42.",
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


def leaf_key(class_name: str, split: str, leaf: str) -> str:
    return f"{class_name}/{split}/{leaf}"


def expected_leaves() -> List[Tuple[str, str, str]]:
    leaves: List[Tuple[str, str, str]] = []
    for class_name in EXPECTED_CLASSES:
        leaf = expected_leaf(class_name)
        for split in ("train", "val"):
            leaves.append((class_name, split, leaf))
    return leaves


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
        key = leaf_key(class_name, split, leaf)
        split_report: Dict[str, object] = {
            "leaf_key": key,
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


def leaf_counts_from_reports(reports: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for report in reports:
        splits = report.get("splits", {})
        for split in ("train", "val"):
            item = splits.get(split, {})
            key = item.get("leaf_key")
            if key:
                counts[str(key)] = int(item.get("image_count", 0))
    return counts


def load_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_leaf_counts(manifest: Dict[str, object]) -> Dict[str, int]:
    leaf_counts = manifest.get("leaf_counts")
    if not isinstance(leaf_counts, dict):
        raise ValueError("Manifest does not contain a leaf_counts object.")
    result: Dict[str, int] = {}
    for key, item in leaf_counts.items():
        if isinstance(item, dict):
            result[str(key)] = int(item.get("count", 0))
        else:
            result[str(key)] = int(item)
    return result


def compare_manifest_counts(
    manifest_path: Path,
    reports: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], List[str]]:
    manifest = load_manifest(manifest_path)
    expected = manifest_leaf_counts(manifest)
    actual = leaf_counts_from_reports(reports)
    errors: List[str] = []

    for key, expected_count in expected.items():
        actual_count = actual.get(key)
        if actual_count != expected_count:
            errors.append(f"Manifest count mismatch for {key}: actual={actual_count}, manifest={expected_count}")
    for key in actual:
        if key not in expected:
            errors.append(f"Leaf exists in dataset but not in manifest: {key}")

    return manifest, errors


def stable_leaf_seed(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sample_size(total: int, ratio: float) -> int:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if total <= 0:
        return 0
    return max(1, int(total * ratio))


def selected_relative_paths(full_leaf_dir: Path, ratio: float, seed: int, key: str) -> Set[str]:
    images, _, _ = collect_dir_items(full_leaf_dir)
    rng = random.Random(stable_leaf_seed(seed, key))
    shuffled = list(images)
    rng.shuffle(shuffled)
    selected = shuffled[: sample_size(len(shuffled), ratio)]
    return {str(path.relative_to(full_leaf_dir)).replace("\\", "/") for path in selected}


def actual_relative_paths(sample_leaf_dir: Path) -> Set[str]:
    images, _, _ = collect_dir_items(sample_leaf_dir)
    return {str(path.relative_to(sample_leaf_dir)).replace("\\", "/") for path in images}


def same_hardlink(a: Path, b: Path) -> bool:
    try:
        a_stat = os.stat(a)
        b_stat = os.stat(b)
    except OSError:
        return False
    return a_stat.st_dev == b_stat.st_dev and a_stat.st_ino == b_stat.st_ino


def compare_sample_to_full(sample_root: Path, full_root: Path, ratio: float, seed: int) -> Tuple[Dict[str, object], List[str]]:
    errors: List[str] = []
    report: Dict[str, object] = {
        "compare_root": str(full_root),
        "ratio": ratio,
        "seed": seed,
        "leaves": {},
    }

    for class_name, split, leaf in expected_leaves():
        key = leaf_key(class_name, split, leaf)
        full_leaf_dir = full_root / class_name / split / leaf
        sample_leaf_dir = sample_root / class_name / split / leaf
        if not full_leaf_dir.is_dir():
            errors.append(f"Missing compare_root leaf: {full_leaf_dir}")
            continue
        if not sample_leaf_dir.is_dir():
            errors.append(f"Missing sample root leaf: {sample_leaf_dir}")
            continue

        expected_rel = selected_relative_paths(full_leaf_dir, ratio, seed, key)
        actual_rel = actual_relative_paths(sample_leaf_dir)
        missing = sorted(expected_rel - actual_rel)
        extra = sorted(actual_rel - expected_rel)
        hardlink_mismatches: List[str] = []
        for rel in sorted(expected_rel & actual_rel):
            if not same_hardlink(full_leaf_dir / rel, sample_leaf_dir / rel):
                hardlink_mismatches.append(rel)

        leaf_report = {
            "expected_count": len(expected_rel),
            "actual_count": len(actual_rel),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
            "hardlink_mismatch_examples": hardlink_mismatches[:20],
        }
        report["leaves"][key] = leaf_report

        if missing:
            errors.append(f"Sample leaf is missing expected files for {key}: {len(missing)}")
        if extra:
            errors.append(f"Sample leaf has unexpected files for {key}: {len(extra)}")
        if hardlink_mismatches:
            errors.append(f"Sample leaf files are not hardlinks to compare_root for {key}: {len(hardlink_mismatches)}")

    return report, errors


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

    manifest_report = None
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        try:
            manifest_report, manifest_errors = compare_manifest_counts(manifest_path, reports)
            errors.extend(manifest_errors)
            logging.info("Compared dataset counts against manifest: %s", manifest_path)
        except Exception as exc:
            errors.append(f"Manifest comparison failed: {exc}")

    sample_compare_report = None
    if args.compare_root:
        compare_root = Path(args.compare_root).expanduser()
        if not compare_root.is_dir():
            errors.append(f"compare_root does not exist or is not a directory: {compare_root}")
        else:
            try:
                sample_compare_report, sample_errors = compare_sample_to_full(root, compare_root, args.ratio, args.seed)
                errors.extend(sample_errors)
                logging.info("Compared sampled root against full root: %s", compare_root)
            except Exception as exc:
                errors.append(f"Sample comparison failed: {exc}")

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
        "leaf_counts": leaf_counts_from_reports(reports),
        "manifest": manifest_report,
        "sample_compare": sample_compare_report,
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
