# -*- coding: utf-8 -*-
"""Sequentially extract GenImage zip archives with 7z-compatible tools.

This script is intentionally conservative:

- only ``.zip`` main files are extracted;
- split volumes such as ``.z01`` are never treated as standalone archives;
- ``.tmp`` files are never extracted, and zip files in those directories are skipped;
- completed archives are tracked with marker files;
- no archive is deleted, moved, or modified.
"""

import argparse
import datetime as dt
import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_ROOT = "/root/autodl-tmp/GenImage"
TOOL_CANDIDATES = ("7zz", "7z", "7za")
SPLIT_VOLUME_RE = re.compile(r"\.z\d+$", re.IGNORECASE)
SKIP_DIR_NAMES = {"_extract_logs", "_extract_done"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially extract GenImage .zip archives with 7zz/7z/7za."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Raw GenImage archive root.")
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Extraction log directory. Default: {root}/_extract_logs.",
    )
    parser.add_argument(
        "--marker_dir",
        default=None,
        help="Completion marker directory. Default: {root}/_extract_done.",
    )
    parser.add_argument(
        "--install_if_missing",
        dest="install_if_missing",
        action="store_true",
        default=True,
        help="Try to install 7z tools when missing. Enabled by default.",
    )
    parser.add_argument(
        "--no_install_if_missing",
        dest="install_if_missing",
        action="store_false",
        help="Disable automatic install attempts when 7z tools are missing.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without extracting.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing completion markers and try extraction again.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_7z_tool() -> Optional[str]:
    for name in TOOL_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def command_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def run_install_command(command: Sequence[str]) -> bool:
    logging.info("Running install command: %s", command_text(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        return True
    logging.warning("Install command failed with return code %s", completed.returncode)
    return False


def try_install_7z_tools() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "7zz/7z/7za was not found. Automatic install is only attempted on Linux; "
            f"current platform is {platform.platform()}."
        )

    attempted = False
    apt_get = shutil.which("apt-get")
    geteuid = getattr(os, "geteuid", None)
    is_root = geteuid is not None and geteuid() == 0
    if apt_get and is_root:
        attempted = True
        logging.info("7z tools are missing. Will try apt-get installation first.")
        if run_install_command([apt_get, "update"]):
            run_install_command([apt_get, "install", "-y", "7zip", "p7zip-full", "unzip"])
    elif apt_get:
        logging.warning("apt-get exists but current user is not root; skipping apt-get install.")

    if find_7z_tool():
        return

    conda = shutil.which("conda")
    if conda:
        attempted = True
        logging.info("Trying conda installation for p7zip/unzip.")
        run_install_command([conda, "install", "-c", "conda-forge", "p7zip", "unzip", "-y"])

    if not attempted:
        raise RuntimeError(
            "7zz/7z/7za was not found, and neither a usable root apt-get nor conda was available. "
            "Please install 7zip/p7zip manually."
        )


def ensure_7z_tool(install_if_missing: bool, dry_run: bool) -> str:
    tool = find_7z_tool()
    if tool:
        logging.info("Using extraction tool: %s", tool)
        return tool

    logging.warning("No 7z-compatible tool found in PATH: %s", ", ".join(TOOL_CANDIDATES))
    logging.info("Suggested apt command: apt-get update && apt-get install -y 7zip p7zip-full unzip")
    logging.info("Suggested conda command: conda install -c conda-forge p7zip unzip -y")

    if dry_run:
        logging.info("dry_run is enabled; install commands will not be executed.")
        return "7z"

    if not install_if_missing:
        raise RuntimeError("Missing 7zz/7z/7za and --no_install_if_missing was set.")

    try_install_7z_tools()
    tool = find_7z_tool()
    if not tool:
        raise RuntimeError(
            "7zz/7z/7za is still missing after install attempts. Please install it manually."
        )
    logging.info("Using extraction tool after installation: %s", tool)
    return tool


def iter_archive_tree(root: Path) -> Tuple[List[Path], Dict[Path, List[Path]], List[Path]]:
    zip_files: List[Path] = []
    tmp_by_dir: Dict[Path, List[Path]] = {}
    split_volumes: List[Path] = []

    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
        current_path = Path(current)
        for filename in files:
            path = current_path / filename
            lower_name = filename.lower()
            if lower_name.endswith(".tmp"):
                tmp_by_dir.setdefault(current_path, []).append(path)
                continue
            if lower_name.endswith(".zip"):
                zip_files.append(path)
                continue
            if SPLIT_VOLUME_RE.search(lower_name):
                split_volumes.append(path)

    zip_files.sort(key=lambda p: str(p).lower())
    split_volumes.sort(key=lambda p: str(p).lower())
    return zip_files, tmp_by_dir, split_volumes


def marker_path(marker_dir: Path, zip_path: Path) -> Path:
    resolved = str(zip_path.resolve(strict=False))
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    return marker_dir / f"{digest}.done"


def log_path(log_dir: Path, zip_path: Path) -> Path:
    resolved = str(zip_path.resolve(strict=False))
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    return log_dir / f"{digest}.log"


def write_marker(marker: Path, zip_path: Path, output_dir: Path, tool: str, return_code: int) -> None:
    finish_time = dt.datetime.now(dt.timezone.utc).isoformat()
    content = (
        f"zip_path={zip_path.resolve(strict=False)}\n"
        f"output_dir={output_dir.resolve(strict=False)}\n"
        f"tool={tool}\n"
        f"return_code={return_code}\n"
        f"finish_time={finish_time}\n"
    )
    marker.write_text(content, encoding="utf-8")


def filter_complete_zip_files(
    zip_files: Sequence[Path],
    tmp_by_dir: Dict[Path, List[Path]],
) -> Tuple[List[Path], List[Tuple[Path, List[Path]]]]:
    complete_zip_files = []
    skipped = []
    for zip_path in zip_files:
        tmp_files = tmp_by_dir.get(zip_path.parent)
        if tmp_files:
            skipped.append((zip_path, tmp_files))
            continue
        complete_zip_files.append(zip_path)
    return complete_zip_files, skipped


def extract_one(tool: str, zip_path: Path, output_dir: Path, log_file: Path) -> int:
    command = [tool, "x", str(zip_path), f"-o{output_dir}", "-mmt=on", "-aos"]
    logging.info("Extracting: %s", zip_path)
    logging.info("Command: %s", command_text(command))
    logging.info("Writing extraction log: %s", log_file)

    with log_file.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"command={command_text(command)}\n")
        handle.write(f"start_time={dt.datetime.now(dt.timezone.utc).isoformat()}\n\n")
        handle.flush()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(f"\nreturn_code={completed.returncode}\n")
        handle.write(f"finish_time={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        return completed.returncode


def main() -> int:
    args = parse_args()
    setup_logging()

    root = Path(args.root).expanduser()
    if not root.exists():
        logging.error("GenImage archive root does not exist: %s", root)
        return 2
    if not root.is_dir():
        logging.error("GenImage archive root is not a directory: %s", root)
        return 2

    log_dir = Path(args.log_dir).expanduser() if args.log_dir else root / "_extract_logs"
    marker_dir = Path(args.marker_dir).expanduser() if args.marker_dir else root / "_extract_done"

    zip_files, tmp_by_dir, split_volumes = iter_archive_tree(root)
    if split_volumes:
        logging.info("Found %d split volume files (.z01 etc.); they will be used only by 7z via .zip files.", len(split_volumes))
    zip_files, skipped_due_tmp = filter_complete_zip_files(zip_files, tmp_by_dir)
    if skipped_due_tmp:
        logging.warning(
            "Skipping %d .zip file(s) because their directory contains .tmp files; "
            "rerun the same command after downloads finish.",
            len(skipped_due_tmp),
        )
        for zip_path, tmp_files in skipped_due_tmp[:20]:
            examples = ", ".join(str(path) for path in tmp_files[:5])
            logging.warning("  skipped zip: %s", zip_path)
            logging.warning("  tmp files: %s", examples)
        if len(skipped_due_tmp) > 20:
            logging.warning("  ... %d more skipped zip files omitted", len(skipped_due_tmp) - 20)

    if not zip_files:
        logging.warning("No complete .zip archive files found under: %s", root)
        return 0

    tool = ensure_7z_tool(args.install_if_missing, args.dry_run)

    logging.info("Found %d .zip archive(s).", len(zip_files))
    for zip_path in zip_files:
        output_dir = zip_path.parent
        marker = marker_path(marker_dir, zip_path)
        per_zip_log = log_path(log_dir, zip_path)
        command = [tool, "x", str(zip_path), f"-o{output_dir}", "-mmt=on", "-aos"]

        if marker.exists() and not args.force:
            logging.info("Skipping completed archive: %s", zip_path)
            logging.info("Completion marker: %s", marker)
            continue

        if args.dry_run:
            if marker.exists() and args.force:
                logging.info("Would ignore existing marker because --force is set: %s", marker)
            logging.info("[dry_run] Would run: %s", command_text(command))
            logging.info("[dry_run] Would write log: %s", per_zip_log)
            logging.info("[dry_run] Would write marker on success: %s", marker)
            continue

        log_dir.mkdir(parents=True, exist_ok=True)
        marker_dir.mkdir(parents=True, exist_ok=True)
        return_code = extract_one(tool, zip_path, output_dir, per_zip_log)
        if return_code != 0:
            logging.error("Extraction failed for %s with return code %s", zip_path, return_code)
            logging.error("See log: %s", per_zip_log)
            return return_code
        write_marker(marker, zip_path, output_dir, tool, return_code)
        logging.info("Extraction finished; marker written: %s", marker)

    logging.info("All planned archives have been processed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
