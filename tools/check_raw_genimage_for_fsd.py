# -*- coding: utf-8 -*-
"""Check whether raw GenImage data is ready for FSD layout construction."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ROOT = "/root/autodl-tmp/GenImage"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_VOLUME_RE = re.compile(r"\.z\d+$", re.IGNORECASE)
DISCOVERY_SKIP_NAMES = {"train", "val", "ai", "nature", "_extract_logs", "_extract_done", "__pycache__"}

SOURCE_SPECS = [
    {
        "name": "ADM",
        "aliases": ["ADM"],
        "keywords": ["adm"],
        "required": [("train", "ai"), ("val", "ai")],
    },
    {
        "name": "BigGAN",
        "aliases": ["BigGAN"],
        "keywords": ["biggan"],
        "required": [("train", "ai"), ("val", "ai")],
    },
    {
        "name": "glide",
        "aliases": ["glide", "GLIDE"],
        "keywords": ["glide"],
        "required": [("train", "ai"), ("val", "ai")],
    },
    {
        "name": "Midjourney",
        "aliases": ["Midjourney"],
        "keywords": ["midjourney"],
        "required": [("train", "ai"), ("val", "ai")],
    },
    {
        "name": "stable_diffusion_v_1_4",
        "aliases": ["stable_diffusion_v_1_4"],
        "keywords": ["stable_diffusion_v_1_4", "stable_diffusion_v1_4", "sdv4", "sd14", "sd_1_4"],
        "required": [("train", "ai"), ("val", "ai"), ("train", "nature"), ("val", "nature")],
    },
    {
        "name": "stable_diffusion_v_1_5",
        "aliases": ["stable_diffusion_v_1_5"],
        "keywords": ["stable_diffusion_v_1_5", "stable_diffusion_v1_5", "sdv5", "sd15", "sd_1_5"],
        "required": [("train", "ai"), ("val", "ai"), ("train", "nature"), ("val", "nature")],
    },
    {
        "name": "wukong",
        "aliases": ["wukong", "Wukong"],
        "keywords": ["wukong"],
        "required": [("train", "ai"), ("val", "ai")],
    },
    {
        "name": "VQDM",
        "aliases": ["VQDM"],
        "keywords": ["vqdm"],
        "required": [("train", "ai"), ("val", "ai")],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only check for raw GenImage folders required by the FSD hardlink builder."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Raw GenImage root.")
    parser.add_argument("--json_out", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--max_bad_examples",
        type=int,
        default=20,
        help="Maximum number of abnormal paths to print per category.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def matches_source(root: Path, path: Path, spec: Dict[str, object]) -> bool:
    aliases = [str(alias).lower() for alias in spec["aliases"]]
    keywords = [str(keyword).lower() for keyword in spec["keywords"]]
    parts = rel_parts_lower(root, path)
    if any(part in aliases for part in parts):
        return True
    name = path.name.lower()
    return any(keyword in name for keyword in keywords)


def find_source_candidates(root: Path, spec: Dict[str, object]) -> List[Path]:
    candidates = []
    for directory in iter_candidate_dirs(root):
        if is_dataset_root(directory) and matches_source(root, directory, spec):
            candidates.append(directory)
    unique = sorted(set(candidates), key=lambda path: (len(path.parts), str(path).lower()))
    return unique


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for current, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in IMG_EXTENSIONS:
                count += 1
    return count


def collect_limited_paths(
    root: Path,
    predicate,
    max_examples: int,
) -> List[str]:
    examples: List[str] = []
    if not root.exists():
        return examples
    for current, _, files in os.walk(root):
        for filename in files:
            path = Path(current) / filename
            if predicate(path):
                examples.append(str(path))
                if len(examples) >= max_examples:
                    return examples
    return examples


def collect_empty_dirs(root: Path, max_examples: int) -> List[str]:
    examples: List[str] = []
    if not root.exists():
        return examples
    for current, dirs, files in os.walk(root):
        if not dirs and not files:
            examples.append(str(Path(current)))
            if len(examples) >= max_examples:
                return examples
    return examples


def report_source(root: Path, spec: Dict[str, object], max_bad_examples: int) -> Tuple[Dict[str, object], List[str]]:
    name = str(spec["name"])
    required = set(tuple(item) for item in spec["required"])
    candidates = find_source_candidates(root, spec)
    errors: List[str] = []

    report: Dict[str, object] = {
        "source": name,
        "selected_root": None,
        "candidates": [str(path) for path in candidates],
        "splits": {},
        "tmp_files": [],
        "archive_files": [],
        "empty_dirs": [],
    }

    if not candidates:
        errors.append(f"{name}: source directory not found")
        return report, errors

    selected = candidates[0]
    report["selected_root"] = str(selected)
    if len(candidates) > 1:
        logging.warning("%s has multiple candidates; using %s", name, selected)
        for candidate in candidates:
            logging.warning("  candidate: %s", candidate)
    else:
        logging.info("%s source root: %s", name, selected)

    tmp_files = collect_limited_paths(
        selected,
        lambda path: path.name.lower().endswith(".tmp"),
        max_bad_examples,
    )
    archive_files = collect_limited_paths(
        selected,
        lambda path: path.suffix.lower() == ".zip" or SPLIT_VOLUME_RE.search(path.name.lower()) is not None,
        max_bad_examples,
    )
    empty_dirs = collect_empty_dirs(selected, max_bad_examples)
    report["tmp_files"] = tmp_files
    report["archive_files"] = archive_files
    report["empty_dirs"] = empty_dirs
    if tmp_files:
        errors.append(f"{name}: found .tmp files, download may be incomplete")

    split_report: Dict[str, Dict[str, Dict[str, object]]] = {}
    for split in ("train", "val"):
        split_report[split] = {}
        for leaf in ("ai", "nature"):
            directory = selected / split / leaf
            exists = directory.is_dir()
            count = count_images(directory) if exists else 0
            is_required = (split, leaf) in required
            status = "OK"
            if is_required and not exists:
                status = "MISSING"
                errors.append(f"{name}: missing required directory {split}/{leaf}")
            elif is_required and count == 0:
                status = "EMPTY"
                errors.append(f"{name}: required directory has no images {split}/{leaf}")
            elif not exists:
                status = "not_present"
            elif count == 0:
                status = "empty_optional"
            split_report[split][leaf] = {
                "path": str(directory),
                "exists": exists,
                "image_count": count,
                "required": is_required,
                "status": status,
            }
    report["splits"] = split_report
    return report, errors


def print_table(reports: Sequence[Dict[str, object]]) -> None:
    header = f"{'source':<24} {'split':<6} {'leaf':<7} {'exists':<7} {'images':>12} {'required':<8} {'status'}"
    print(header)
    print("-" * len(header))
    for report in reports:
        source = str(report["source"])
        splits = report.get("splits", {})
        if not splits:
            print(f"{source:<24} {'-':<6} {'-':<7} {'no':<7} {0:>12} {'yes':<8} MISSING_SOURCE")
            continue
        for split in ("train", "val"):
            for leaf in ("ai", "nature"):
                item = splits[split][leaf]
                exists = "yes" if item["exists"] else "no"
                required = "yes" if item["required"] else "no"
                print(
                    f"{source:<24} {split:<6} {leaf:<7} {exists:<7} "
                    f"{item['image_count']:>12} {required:<8} {item['status']}"
                )


def main() -> int:
    args = parse_args()
    setup_logging()

    root = Path(args.root).expanduser()
    if not root.exists():
        logging.error("Raw GenImage root does not exist: %s", root)
        return 2
    if not root.is_dir():
        logging.error("Raw GenImage root is not a directory: %s", root)
        return 2

    reports: List[Dict[str, object]] = []
    errors: List[str] = []
    for spec in SOURCE_SPECS:
        report, source_errors = report_source(root, spec, args.max_bad_examples)
        reports.append(report)
        errors.extend(source_errors)

    global_archives = collect_limited_paths(
        root,
        lambda path: path.suffix.lower() == ".zip" or SPLIT_VOLUME_RE.search(path.name.lower()) is not None,
        args.max_bad_examples,
    )
    global_tmp = collect_limited_paths(
        root,
        lambda path: path.name.lower().endswith(".tmp"),
        args.max_bad_examples,
    )
    if global_tmp:
        errors.append("Raw root contains .tmp files; downloads may be incomplete")

    print()
    print_table(reports)
    print()
    if global_archives:
        logging.info("Archive/split-volume examples still present under raw root:")
        for path in global_archives:
            logging.info("  %s", path)
    if global_tmp:
        logging.error(".tmp examples under raw root:")
        for path in global_tmp:
            logging.error("  %s", path)

    output = {
        "root": str(root),
        "sources": reports,
        "global_archive_examples": global_archives,
        "global_tmp_examples": global_tmp,
        "errors": errors,
        "status": "OK" if not errors else "ERROR",
    }
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("JSON report written: %s", json_path)

    if errors:
        logging.error("Raw GenImage check failed:")
        for error in errors:
            logging.error("  %s", error)
        return 1

    logging.info("Raw GenImage check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
