# -*- coding: utf-8 -*-
"""Build the 7-class FSD GenImage layout with hardlinks only."""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_RAW_ROOT = "/root/autodl-tmp/GenImage"
DEFAULT_OUT_ROOT = "/root/autodl-tmp/data_fsd_full/GenImage"
LEGACY_DATA_ROOT = "/root/autodl-tmp/data"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISCOVERY_SKIP_NAMES = {"train", "val", "ai", "nature", "_extract_logs", "_extract_done", "__pycache__"}
EXPECTED_CLASSES = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]

SOURCE_SPECS = {
    "ADM": {"aliases": ["ADM"], "keywords": ["adm"]},
    "BigGAN": {"aliases": ["BigGAN"], "keywords": ["biggan"]},
    "glide": {"aliases": ["glide", "GLIDE"], "keywords": ["glide"]},
    "Midjourney": {"aliases": ["Midjourney"], "keywords": ["midjourney"]},
    "stable_diffusion_v_1_4": {
        "aliases": ["stable_diffusion_v_1_4"],
        "keywords": ["stable_diffusion_v_1_4", "stable_diffusion_v1_4", "sdv4", "sd14", "sd_1_4"],
    },
    "stable_diffusion_v_1_5": {
        "aliases": ["stable_diffusion_v_1_5"],
        "keywords": ["stable_diffusion_v_1_5", "stable_diffusion_v1_5", "sdv5", "sd15", "sd_1_5"],
    },
    "wukong": {"aliases": ["wukong", "Wukong"], "keywords": ["wukong"]},
    "VQDM": {"aliases": ["VQDM"], "keywords": ["vqdm"]},
}

BUILD_RULES = [
    ("stable_diffusion_v_1_4", "train", "nature", "real", "train", "nature", "sdv14"),
    ("stable_diffusion_v_1_5", "train", "nature", "real", "train", "nature", "sdv15"),
    ("stable_diffusion_v_1_4", "val", "nature", "real", "val", "nature", "sdv14"),
    ("stable_diffusion_v_1_5", "val", "nature", "real", "val", "nature", "sdv15"),
    ("ADM", "train", "ai", "ADM", "train", "ai", "adm"),
    ("ADM", "val", "ai", "ADM", "val", "ai", "adm"),
    ("BigGAN", "train", "ai", "BigGAN", "train", "ai", "biggan"),
    ("BigGAN", "val", "ai", "BigGAN", "val", "ai", "biggan"),
    ("glide", "train", "ai", "glide", "train", "ai", "glide"),
    ("glide", "val", "ai", "glide", "val", "ai", "glide"),
    ("Midjourney", "train", "ai", "Midjourney", "train", "ai", "midjourney"),
    ("Midjourney", "val", "ai", "Midjourney", "val", "ai", "midjourney"),
    ("stable_diffusion_v_1_4", "train", "ai", "SD", "train", "ai", "sdv14"),
    ("stable_diffusion_v_1_5", "train", "ai", "SD", "train", "ai", "sdv15"),
    ("wukong", "train", "ai", "SD", "train", "ai", "wukong"),
    ("stable_diffusion_v_1_4", "val", "ai", "SD", "val", "ai", "sdv14"),
    ("stable_diffusion_v_1_5", "val", "ai", "SD", "val", "ai", "sdv15"),
    ("wukong", "val", "ai", "SD", "val", "ai", "wukong"),
    ("VQDM", "train", "ai", "VQDM", "train", "ai", "vqdm"),
    ("VQDM", "val", "ai", "VQDM", "val", "ai", "vqdm"),
]

CLASS_SOURCES = {
    "real": "sdv14 nature + sdv15 nature",
    "ADM": "ADM ai",
    "BigGAN": "BigGAN ai",
    "glide": "glide ai",
    "Midjourney": "Midjourney ai",
    "SD": "sdv14 ai + sdv15 ai + wukong ai",
    "VQDM": "VQDM ai",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build /root/autodl-tmp/data_fsd_full/GenImage style FSD layout with hardlinks."
    )
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT, help="Raw extracted GenImage root.")
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Output FSD GenImage root.")
    parser.add_argument(
        "--link_mode",
        default="hardlink",
        choices=["hardlink"],
        help="Only hardlink is supported; copy/symlink fallback is intentionally unavailable.",
    )
    parser.add_argument(
        "--name_policy",
        default="source_prefix",
        choices=["source_prefix"],
        help="Only source_prefix naming is supported: source_tag__original_filename.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without creating links.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow continuing in an existing out_root without overwriting conflicting files.",
    )
    parser.add_argument("--json_out", default=None, help="Optional extra JSON build summary path.")
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


def leaf_key(class_name: str, split: str, leaf: str) -> str:
    return f"{class_name}/{split}/{leaf}"


def is_dataset_root(path: Path) -> bool:
    return any((path / split / leaf).is_dir() for split in ("train", "val") for leaf in ("ai", "nature"))


def iter_candidate_dirs(root: Path, max_depth: int = 4) -> Iterable[Path]:
    yield root
    stack: List[Tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name.lower())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name in DISCOVERY_SKIP_NAMES:
                continue
            child = Path(entry.path)
            yield child
            stack.append((child, depth + 1))


def rel_parts_lower(root: Path, path: Path) -> List[str]:
    try:
        rel = path.relative_to(root)
        return [part.lower() for part in rel.parts]
    except ValueError:
        return [part.lower() for part in path.parts]


def matches_source(root: Path, path: Path, spec: Dict[str, List[str]]) -> bool:
    aliases = [alias.lower() for alias in spec["aliases"]]
    keywords = [keyword.lower() for keyword in spec["keywords"]]
    parts = rel_parts_lower(root, path)
    if any(part in aliases for part in parts):
        return True
    name = path.name.lower()
    return any(keyword in name for keyword in keywords)


def find_source_root(raw_root: Path, source_name: str) -> Path:
    spec = SOURCE_SPECS[source_name]
    candidates = [
        directory
        for directory in iter_candidate_dirs(raw_root)
        if is_dataset_root(directory) and matches_source(raw_root, directory, spec)
    ]
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path).lower()))
    if not candidates:
        raise FileNotFoundError(f"Missing raw source for {source_name} under {raw_root}")
    if len(candidates) > 1:
        logging.warning("%s has multiple candidates; using %s", source_name, candidates[0])
        for candidate in candidates:
            logging.warning("  candidate: %s", candidate)
    else:
        logging.info("%s source root: %s", source_name, candidates[0])
    return candidates[0]


def collect_image_files(directory: Path) -> List[Path]:
    files: List[Path] = []
    if not directory.is_dir():
        return files
    for current, _, filenames in os.walk(directory):
        for filename in filenames:
            if Path(filename).suffix.lower() in IMG_EXTENSIONS:
                files.append(Path(current) / filename)
    files.sort(key=lambda path: str(path).lower())
    return files


def total_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            logging.warning("Could not stat image file: %s", path)
    return total


def commonpath_is_parent(child: Path, parent: Path) -> bool:
    try:
        common = os.path.commonpath([str(child), str(parent)])
    except ValueError:
        return False
    return common == str(parent)


def protect_output_path(out_root: Path) -> None:
    out_resolved = out_root.resolve(strict=False)
    legacy = Path(LEGACY_DATA_ROOT).resolve(strict=False)
    if out_resolved == legacy or commonpath_is_parent(out_resolved, legacy):
        raise ValueError(f"Refusing to write inside legacy data root: {LEGACY_DATA_ROOT}")


def is_same_file(src: Path, dst: Path) -> bool:
    src_stat = os.stat(src)
    dst_stat = os.stat(dst)
    return src_stat.st_dev == dst_stat.st_dev and src_stat.st_ino == dst_stat.st_ino


def init_summary() -> Dict[str, Dict[str, object]]:
    return {
        class_name: {"train_count": 0, "val_count": 0, "sources": CLASS_SOURCES[class_name]}
        for class_name in EXPECTED_CLASSES
    }


def make_plan(
    raw_root: Path,
    out_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, object]], Dict[str, str], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    needed_sources = sorted({rule[0] for rule in BUILD_RULES})
    source_roots = {source: find_source_root(raw_root, source) for source in needed_sources}
    summary = init_summary()
    target_seen: Dict[Path, Path] = {}
    plan: List[Dict[str, str]] = []
    source_dir_counts: List[Dict[str, object]] = []
    leaf_counts: Dict[str, Dict[str, object]] = {}

    for source, src_split, src_leaf, dst_class, dst_split, dst_leaf, tag in BUILD_RULES:
        src_dir = source_roots[source] / src_split / src_leaf
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing required source directory: {src_dir}")
        images = collect_image_files(src_dir)
        if not images:
            raise ValueError(f"No images found in required source directory: {src_dir}")

        dst_dir = out_root / dst_class / dst_split / dst_leaf
        key = leaf_key(dst_class, dst_split, dst_leaf)
        image_bytes = total_bytes(images)
        source_dir_counts.append(
            {
                "source": source,
                "source_dir": str(src_dir),
                "source_split": src_split,
                "source_leaf": src_leaf,
                "target_class": dst_class,
                "target_split": dst_split,
                "target_leaf": dst_leaf,
                "target_key": key,
                "target_dir": str(dst_dir),
                "source_tag": tag,
                "image_count": len(images),
                "image_bytes": image_bytes,
                "image_size": format_bytes(image_bytes),
            }
        )
        leaf_entry = leaf_counts.setdefault(
            key,
            {
                "class": dst_class,
                "split": dst_split,
                "leaf": dst_leaf,
                "path": str(dst_dir),
                "count": 0,
                "bytes": 0,
                "size": "0.00 B",
            },
        )
        leaf_entry["count"] = int(leaf_entry["count"]) + len(images)
        leaf_entry["bytes"] = int(leaf_entry["bytes"]) + image_bytes
        leaf_entry["size"] = format_bytes(int(leaf_entry["bytes"]))

        for src in images:
            dst = dst_dir / f"{tag}__{src.name}"
            if dst in target_seen and target_seen[dst] != src:
                raise ValueError(
                    "Target filename collision after source_prefix naming:\n"
                    f"  target: {dst}\n"
                    f"  first source: {target_seen[dst]}\n"
                    f"  second source: {src}"
                )
            target_seen[dst] = src
            plan.append(
                {
                    "source": str(src),
                    "target": str(dst),
                    "class": dst_class,
                    "split": dst_split,
                    "leaf": dst_leaf,
                    "leaf_key": key,
                    "tag": tag,
                    "source_dir": str(src_dir),
                    "target_dir": str(dst_dir),
                }
            )
            summary_key = "train_count" if dst_split == "train" else "val_count"
            summary[dst_class][summary_key] = int(summary[dst_class][summary_key]) + 1

    source_root_report = {source: str(path) for source, path in source_roots.items()}
    return plan, summary, source_root_report, source_dir_counts, leaf_counts


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


def print_summary(summary: Dict[str, Dict[str, object]]) -> None:
    header = f"{'class':<12} {'train_count':>12} {'val_count':>12} {'sources'}"
    print(header)
    print("-" * len(header))
    for class_name in EXPECTED_CLASSES:
        item = summary[class_name]
        print(
            f"{class_name:<12} {item['train_count']:>12} {item['val_count']:>12} "
            f"{item['sources']}"
        )


def build_manifest(
    raw_root: Path,
    out_root: Path,
    source_roots: Dict[str, str],
    summary: Dict[str, Dict[str, object]],
    source_dir_counts: Sequence[Dict[str, object]],
    leaf_counts: Dict[str, Dict[str, object]],
    planned_links: int,
    existing_matching_links: int,
    created_links: int,
    already_present_links: int,
    status: str,
) -> Dict[str, object]:
    return {
        "manifest_type": "fsd_genimage_build",
        "version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "link_mode": "hardlink",
        "name_policy": "source_prefix",
        "source_roots": source_roots,
        "class_summary": summary,
        "source_dir_counts": list(source_dir_counts),
        "leaf_counts": leaf_counts,
        "planned_links": planned_links,
        "existing_matching_links_before_build": existing_matching_links,
        "created_links": created_links,
        "already_present_links": already_present_links,
        "status": status,
    }


def build_summary_text(manifest: Dict[str, object]) -> str:
    lines = [
        "FSD GenImage build summary",
        f"created_at: {manifest['created_at']}",
        f"raw_root: {manifest['raw_root']}",
        f"out_root: {manifest['out_root']}",
        f"link_mode: {manifest['link_mode']}",
        f"name_policy: {manifest['name_policy']}",
        f"planned_links: {manifest['planned_links']}",
        f"created_links: {manifest['created_links']}",
        f"already_present_links: {manifest['already_present_links']}",
        "",
        "class summary",
        f"{'class':<12} {'train_count':>12} {'val_count':>12} sources",
        "-" * 72,
    ]
    class_summary = manifest["class_summary"]
    for class_name in EXPECTED_CLASSES:
        item = class_summary[class_name]
        lines.append(
            f"{class_name:<12} {item['train_count']:>12} {item['val_count']:>12} {item['sources']}"
        )

    lines.extend(
        [
            "",
            "source directory hardlink counts",
            f"{'source':<24} {'target':<24} {'count':>12} {'size':>12} source_dir",
            "-" * 120,
        ]
    )
    for item in manifest["source_dir_counts"]:
        lines.append(
            f"{item['source']:<24} {item['target_key']:<24} {item['image_count']:>12} "
            f"{item['image_size']:>12} {item['source_dir']}"
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


def write_metadata(
    out_root: Path,
    manifest: Dict[str, object],
    allow_overwrite: bool,
    extra_json_out: str,
) -> None:
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    summary_text = build_summary_text(manifest)
    manifest_path = out_root / "_build_manifest.json"
    summary_path = out_root / "_build_summary.txt"
    write_text_checked(manifest_path, manifest_text + "\n", allow_overwrite)
    write_text_checked(summary_path, summary_text, allow_overwrite)
    logging.info("Build manifest written: %s", manifest_path)
    logging.info("Build summary written: %s", summary_path)

    if extra_json_out:
        json_path = Path(extra_json_out).expanduser()
        write_text_checked(json_path, manifest_text + "\n", allow_overwrite)
        logging.info("Extra JSON summary written: %s", json_path)


def main() -> int:
    args = parse_args()
    setup_logging()

    raw_root = Path(args.raw_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    if not raw_root.is_dir():
        logging.error("Raw GenImage root does not exist or is not a directory: %s", raw_root)
        return 2

    try:
        protect_output_path(out_root)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    if out_root.exists() and not args.resume:
        logging.error("Output root already exists: %s", out_root)
        logging.error("Pass --resume to continue without overwriting conflicting files.")
        return 2

    try:
        plan, summary, source_roots, source_dir_counts, leaf_counts = make_plan(raw_root, out_root)
        existing_same, existing_errors = check_existing_targets(plan)
        if existing_errors:
            for error in existing_errors:
                logging.error("%s", error)
            return 1
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Planned hardlinks: %d", len(plan))
    logging.info("Existing matching hardlinks: %d", existing_same)
    print()
    print_summary(summary)
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
            raw_root=raw_root,
            out_root=out_root,
            source_roots=source_roots,
            summary=summary,
            source_dir_counts=source_dir_counts,
            leaf_counts=leaf_counts,
            planned_links=len(plan),
            existing_matching_links=existing_same,
            created_links=created,
            already_present_links=already_present,
            status="OK",
        )
        write_metadata(out_root, manifest, allow_overwrite=args.resume, extra_json_out=args.json_out)
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Created hardlinks: %d", created)
    logging.info("Already-present matching hardlinks: %d", already_present)
    logging.info("FSD GenImage layout built at: %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
