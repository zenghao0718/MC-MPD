# -*- coding: utf-8 -*-
"""Create a deterministic sampled FSD GenImage layout with hardlinks only."""

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_FULL_ROOT = "/root/autodl-tmp/data_fsd_full/GenImage"
DEFAULT_OUT_ROOT = "/root/autodl-tmp/data_fsd_20pct/GenImage"
LEGACY_DATA_ROOT = "/root/autodl-tmp/data"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample an FSD GenImage layout leaf-by-leaf using hardlinks."
    )
    parser.add_argument(
        "--full_root",
        default=DEFAULT_FULL_ROOT,
        help="Full FSD GenImage root. Default: /root/autodl-tmp/data_fsd_full/GenImage.",
    )
    parser.add_argument(
        "--out_root",
        default=DEFAULT_OUT_ROOT,
        help="Sampled FSD GenImage root. Default: /root/autodl-tmp/data_fsd_20pct/GenImage.",
    )
    parser.add_argument("--ratio", type=float, default=0.2, help="Per-leaf sample ratio. Default: 0.2.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed. Default: 42.")
    parser.add_argument(
        "--link_mode",
        default="hardlink",
        choices=["hardlink"],
        help="Only hardlink is supported; copy/symlink fallback is intentionally unavailable.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without creating links.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow continuing in an existing out_root without overwriting conflicting files.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


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


def commonpath_is_parent(child: Path, parent: Path) -> bool:
    try:
        common = os.path.commonpath([str(child), str(parent)])
    except ValueError:
        return False
    return common == str(parent)


def protect_output_path(out_root: Path, full_root: Path) -> None:
    out_resolved = out_root.resolve(strict=False)
    legacy = Path(LEGACY_DATA_ROOT).resolve(strict=False)
    full_resolved = full_root.resolve(strict=False)
    if out_resolved == legacy or commonpath_is_parent(out_resolved, legacy):
        raise ValueError(f"Refusing to write inside legacy data root: {LEGACY_DATA_ROOT}")
    if out_resolved == full_resolved or commonpath_is_parent(out_resolved, full_resolved):
        raise ValueError(f"Refusing to write sampled output inside full_root: {full_root}")


def validate_ratio(ratio: float) -> None:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")


def collect_image_files(directory: Path) -> List[Path]:
    images: List[Path] = []
    if not directory.is_dir():
        return images
    for current, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in IMG_EXTENSIONS:
                images.append(Path(current) / filename)
    images.sort(key=lambda path: str(path).lower())
    return images


def total_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            logging.warning("Could not stat image file: %s", path)
    return total


def stable_leaf_seed(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sample_size(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    return max(1, int(total * ratio))


def select_sample(paths: Sequence[Path], ratio: float, seed: int, key: str) -> List[Path]:
    shuffled = list(paths)
    rng = random.Random(stable_leaf_seed(seed, key))
    rng.shuffle(shuffled)
    selected = shuffled[: sample_size(len(shuffled), ratio)]
    selected.sort(key=lambda path: str(path).lower())
    return selected


def is_same_file(src: Path, dst: Path) -> bool:
    src_stat = os.stat(src)
    dst_stat = os.stat(dst)
    return src_stat.st_dev == dst_stat.st_dev and src_stat.st_ino == dst_stat.st_ino


def make_plan(full_root: Path, out_root: Path, ratio: float, seed: int) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, object]]]:
    plan: List[Dict[str, str]] = []
    leaf_counts: Dict[str, Dict[str, object]] = {}
    seen_targets: Dict[Path, Path] = {}

    for class_name, split, leaf in expected_leaves():
        key = leaf_key(class_name, split, leaf)
        src_leaf_dir = full_root / class_name / split / leaf
        dst_leaf_dir = out_root / class_name / split / leaf
        if not src_leaf_dir.is_dir():
            raise FileNotFoundError(f"Missing full FSD leaf directory: {src_leaf_dir}")

        all_images = collect_image_files(src_leaf_dir)
        if not all_images:
            raise ValueError(f"No images found in full FSD leaf directory: {src_leaf_dir}")
        selected = select_sample(all_images, ratio, seed, key)
        selected_bytes = total_bytes(selected)
        leaf_counts[key] = {
            "class": class_name,
            "split": split,
            "leaf": leaf,
            "source_dir": str(src_leaf_dir),
            "target_dir": str(dst_leaf_dir),
            "source_count": len(all_images),
            "count": len(selected),
            "ratio": ratio,
            "seed": seed,
            "bytes": selected_bytes,
            "size": format_bytes(selected_bytes),
        }

        for src in selected:
            rel_path = src.relative_to(src_leaf_dir)
            dst = dst_leaf_dir / rel_path
            if dst in seen_targets and seen_targets[dst] != src:
                raise ValueError(
                    "Target filename collision in sampled layout:\n"
                    f"  target: {dst}\n"
                    f"  first source: {seen_targets[dst]}\n"
                    f"  second source: {src}"
                )
            seen_targets[dst] = src
            plan.append(
                {
                    "source": str(src),
                    "target": str(dst),
                    "leaf_key": key,
                    "class": class_name,
                    "split": split,
                    "leaf": leaf,
                }
            )

    return plan, leaf_counts


def check_existing_targets(plan: Sequence[Dict[str, str]]) -> Tuple[int, List[str]]:
    existing_same = 0
    errors: List[str] = []
    for item in plan:
        src = Path(item["source"])
        dst = Path(item["target"])
        if not dst.exists():
            continue
        try:
            if is_same_file(src, dst):
                existing_same += 1
            else:
                errors.append(f"Target exists and is not the same hardlink: {dst}")
        except OSError as exc:
            errors.append(f"Could not compare existing target {dst}: {exc}")
    return existing_same, errors


def create_hardlinks(plan: Sequence[Dict[str, str]]) -> Tuple[int, int]:
    created = 0
    already_present = 0
    for item in plan:
        src = Path(item["source"])
        dst = Path(item["target"])
        if dst.exists():
            if is_same_file(src, dst):
                already_present += 1
                continue
            raise FileExistsError(f"Target exists and is not the same hardlink: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError as exc:
            raise OSError(f"Hardlink failed:\n  source: {src}\n  target: {dst}\n  reason: {exc}") from exc
        created += 1
    return created, already_present


def class_summary_from_leaf_counts(leaf_counts: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    summary = {class_name: {"train_count": 0, "val_count": 0} for class_name in EXPECTED_CLASSES}
    for item in leaf_counts.values():
        class_name = str(item["class"])
        split = str(item["split"])
        key = "train_count" if split == "train" else "val_count"
        summary[class_name][key] = int(summary[class_name][key]) + int(item["count"])
    return summary


def build_manifest(
    full_root: Path,
    out_root: Path,
    ratio: float,
    seed: int,
    leaf_counts: Dict[str, Dict[str, object]],
    planned_links: int,
    existing_matching_links: int,
    created_links: int,
    already_present_links: int,
    status: str,
) -> Dict[str, object]:
    return {
        "manifest_type": "fsd_genimage_sample",
        "version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "full_root": str(full_root),
        "out_root": str(out_root),
        "ratio": ratio,
        "seed": seed,
        "link_mode": "hardlink",
        "leaf_counts": leaf_counts,
        "class_summary": class_summary_from_leaf_counts(leaf_counts),
        "planned_links": planned_links,
        "existing_matching_links_before_sample": existing_matching_links,
        "created_links": created_links,
        "already_present_links": already_present_links,
        "status": status,
    }


def summary_text(manifest: Dict[str, object]) -> str:
    lines = [
        "FSD GenImage sample summary",
        f"created_at: {manifest['created_at']}",
        f"full_root: {manifest['full_root']}",
        f"out_root: {manifest['out_root']}",
        f"ratio: {manifest['ratio']}",
        f"seed: {manifest['seed']}",
        f"link_mode: {manifest['link_mode']}",
        f"planned_links: {manifest['planned_links']}",
        f"created_links: {manifest['created_links']}",
        f"already_present_links: {manifest['already_present_links']}",
        "",
        "class summary",
        f"{'class':<12} {'train_count':>12} {'val_count':>12}",
        "-" * 40,
    ]
    for class_name in EXPECTED_CLASSES:
        item = manifest["class_summary"][class_name]
        lines.append(f"{class_name:<12} {item['train_count']:>12} {item['val_count']:>12}")

    lines.extend(
        [
            "",
            "leaf sample counts",
            f"{'leaf':<28} {'source':>12} {'sample':>12} {'size':>12}",
            "-" * 72,
        ]
    )
    for key in sorted(manifest["leaf_counts"]):
        item = manifest["leaf_counts"][key]
        lines.append(
            f"{key:<28} {item['source_count']:>12} {item['count']:>12} {item['size']:>12}"
        )
    lines.append("")
    return "\n".join(lines)


def write_text_checked(path: Path, text: str, allow_overwrite: bool) -> None:
    if path.exists():
        old_text = path.read_text(encoding="utf-8")
        if old_text == text:
            logging.info("Metadata already up to date: %s", path)
            return
        if not allow_overwrite:
            raise FileExistsError(f"Metadata file already exists and differs: {path}")
        logging.warning("Overwriting metadata file during --resume: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_metadata(out_root: Path, manifest: Dict[str, object], allow_overwrite: bool) -> None:
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    summary = summary_text(manifest)
    manifest_path = out_root / "_sample_manifest.json"
    summary_path = out_root / "_sample_summary.txt"
    write_text_checked(manifest_path, manifest_text, allow_overwrite)
    write_text_checked(summary_path, summary, allow_overwrite)
    logging.info("Sample manifest written: %s", manifest_path)
    logging.info("Sample summary written: %s", summary_path)


def print_summary(leaf_counts: Dict[str, Dict[str, object]]) -> None:
    summary = class_summary_from_leaf_counts(leaf_counts)
    header = f"{'class':<12} {'train_count':>12} {'val_count':>12}"
    print(header)
    print("-" * len(header))
    for class_name in EXPECTED_CLASSES:
        item = summary[class_name]
        print(f"{class_name:<12} {item['train_count']:>12} {item['val_count']:>12}")


def main() -> int:
    args = parse_args()
    setup_logging()

    full_root = Path(args.full_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    try:
        validate_ratio(args.ratio)
        protect_output_path(out_root, full_root)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    if not full_root.is_dir():
        logging.error("Full FSD GenImage root does not exist or is not a directory: %s", full_root)
        return 2
    if out_root.exists() and not args.resume:
        logging.error("Output root already exists: %s", out_root)
        logging.error("Pass --resume to continue without overwriting conflicting files.")
        return 2

    try:
        plan, leaf_counts = make_plan(full_root, out_root, args.ratio, args.seed)
        existing_same, existing_errors = check_existing_targets(plan)
        if existing_errors:
            for error in existing_errors:
                logging.error("%s", error)
            return 1
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Planned sampled hardlinks: %d", len(plan))
    logging.info("Existing matching hardlinks: %d", existing_same)
    print()
    print_summary(leaf_counts)
    print()

    if args.dry_run:
        logging.info("dry_run enabled; no directories, hardlinks, or metadata files were created.")
        for item in plan[:20]:
            logging.info("[dry_run] %s -> %s", item["source"], item["target"])
        if len(plan) > 20:
            logging.info("[dry_run] ... %d more planned links omitted", len(plan) - 20)
        return 0

    try:
        created, already_present = create_hardlinks(plan)
        manifest = build_manifest(
            full_root=full_root,
            out_root=out_root,
            ratio=args.ratio,
            seed=args.seed,
            leaf_counts=leaf_counts,
            planned_links=len(plan),
            existing_matching_links=existing_same,
            created_links=created,
            already_present_links=already_present,
            status="OK",
        )
        write_metadata(out_root, manifest, allow_overwrite=args.resume)
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Created sampled hardlinks: %d", created)
    logging.info("Already-present matching hardlinks: %d", already_present)
    logging.info("Sampled FSD GenImage layout built at: %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
