# -*- coding: utf-8 -*-
"""Parse DDFSD training logs to extract per-checkpoint training-time alpha/loss stats.

This is a read-only, standalone analysis script. It does not touch train_ddfsd.py or any
training code; it only parses the DEBUG dict lines that train_ddfsd.py already writes via
util.logger (logger.logkv_mean(...) -> logger.dumpkvs() -> logging.debug(dict(...))) every
``--log_interval`` steps.

Two kinds of log files may exist for one training run:
  1. The tee'd stdout/stderr transcript (e.g. logs/train_<CLASS>_10pct.log). Since
     util.logger.setup() uses console_level=logging.INFO, the per-step DEBUG dict lines are
     NOT written to stdout, so this file normally does not contain them.
  2. The internal timestamped file written by util.logger.setup() at
     <output_dir>/logs/<timestamp>_log.txt, which is opened at file_level=logging.DEBUG and
     therefore does contain the DEBUG dict lines used here.

This script auto-discovers whichever file in <output_dir>/logs/ actually contains DEBUG dict
lines (largest count of matches) unless --log_path is given explicitly.

Fields that the training log does NOT record (e.g. sigma_rgb_mean, sigma_freq_mean,
sigma_diff_mean) are filled with "NA" -- they are never fabricated.
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

DEBUG_LINE_RE = re.compile(r"DEBUG (\{.*\})\s*$")

TRAIN_KEYS = (
    "loss_total",
    "loss_dual",
    "loss_sep",
    "loss_rf",
    "loss_ff",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "branch_mode_dual_ratio",
    "branch_mode_rgb_ratio",
    "branch_mode_freq_ratio",
    "lambda_sep_current",
    "lr_group_0",
    "lr_group_1",
    "lr_group_2",
    "lr_group_3",
    "step",
)

OUT_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "matched_train_step",
    "step_delta",
    "loss_total",
    "loss_dual",
    "loss_sep",
    "loss_rf",
    "loss_ff",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "sigma_rgb_mean",
    "sigma_freq_mean",
    "sigma_diff_mean",
    "lambda_sep_current",
    "learning_rate",
    "lr_group_0",
    "lr_group_1",
    "lr_group_2",
    "lr_group_3",
    "branch_mode",
    "log_source",
]

LR_GROUP_LABELS = (
    ("lr_group_0", "rgb_backbone"),
    ("lr_group_1", "freq_backbone"),
    ("lr_group_2", "rgb_head"),
    ("lr_group_3", "freq_head"),
)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse DDFSD training logs for per-checkpoint training-time alpha/loss stats."
    )
    parser.add_argument("--exclude_class", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="<RUN_ROOT>/.../exclude_<CLASS> directory")
    parser.add_argument("--log_path", type=str, default="", help="Explicit log file. Auto-discovered under <output_path>/logs/ if omitted.")
    parser.add_argument("--ckpt_steps", type=str, default="2500,5000,7500,10000,12500,15000")
    parser.add_argument(
        "--max_step_delta",
        type=int,
        default=-1,
        help=(
            "Maximum allowed distance between a checkpoint and its matched log step. "
            "Use 0 to require an exact record; negative keeps the legacy unlimited-nearest behavior."
        ),
    )
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def extract_debug_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = DEBUG_LINE_RE.search(line.strip())
            if not match:
                continue
            try:
                record = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(record, dict):
                continue
            if "step" in record and "loss_total" in record:
                records.append(record)
    return records


def discover_log_file(output_path: str, exclude_class: str) -> Tuple[str, List[Dict]]:
    candidates = []

    tee_log = os.path.join(output_path, "logs", f"train_{exclude_class}_10pct.log")
    if os.path.exists(tee_log):
        candidates.append(tee_log)

    # Keep the legacy exact candidate above, while also accepting experiment
    # suffixes such as ``20pct`` without hard-coding every future variant.
    candidates.extend(sorted(glob.glob(os.path.join(output_path, "logs", f"train_{exclude_class}_*.log"))))

    candidates.extend(sorted(glob.glob(os.path.join(output_path, "logs", "*_log.txt"))))
    candidates.extend(sorted(glob.glob(os.path.join(output_path, "*.log"))))

    best_path: Optional[str] = None
    best_records: List[Dict] = []
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        records = extract_debug_records(candidate)
        if len(records) > len(best_records):
            best_records = records
            best_path = candidate

    if best_path is None:
        raise FileNotFoundError(
            f"No training log with DEBUG step records found under {output_path}/logs/ "
            f"(checked {len(candidates)} candidate file(s))."
        )
    return best_path, best_records


def branch_mode_label(record: Dict) -> str:
    dual_ratio = record.get("branch_mode_dual_ratio")
    rgb_ratio = record.get("branch_mode_rgb_ratio")
    freq_ratio = record.get("branch_mode_freq_ratio")
    if dual_ratio is None or rgb_ratio is None or freq_ratio is None:
        return "NA"
    return f"train_mixed(dual={dual_ratio:.4f},rgb={rgb_ratio:.4f},freq={freq_ratio:.4f})"


def learning_rate_label(record: Dict) -> str:
    """Return exact logged group LRs in optimizer parameter-group order."""
    parts = []
    for key, label in LR_GROUP_LABELS:
        value = record.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return ";".join(parts) if parts else "NA"


def find_nearest_record(
    records: List[Dict], target_step: int, max_step_delta: int = -1
) -> Optional[Dict]:
    if not records:
        return None
    best_record = None
    best_delta = None
    for record in records:
        step = record["step"]
        delta = abs(step - target_step)
        if best_delta is None or delta < best_delta or (delta == best_delta and step < best_record["step"]):
            best_delta = delta
            best_record = record
    if max_step_delta >= 0 and best_delta is not None and best_delta > max_step_delta:
        return None
    return best_record


def main():
    args = parse_args()
    ckpt_steps = parse_int_list(args.ckpt_steps)

    if args.log_path:
        log_path = args.log_path
        records = extract_debug_records(log_path)
        if not records:
            raise ValueError(f"No DEBUG step records found in explicit --log_path: {log_path}")
    else:
        log_path, records = discover_log_file(args.output_path, args.exclude_class)

    records.sort(key=lambda r: r["step"])

    rows = []
    for ckpt_step in ckpt_steps:
        record = find_nearest_record(records, ckpt_step, args.max_step_delta)
        if record is None:
            rows.append(
                {
                    "exclude_class": args.exclude_class,
                    "ckpt_step": ckpt_step,
                    "matched_train_step": "NA",
                    "step_delta": "NA",
                    "loss_total": "NA",
                    "loss_dual": "NA",
                    "loss_sep": "NA",
                    "loss_rf": "NA",
                    "loss_ff": "NA",
                    "alpha_mean": "NA",
                    "alpha_min": "NA",
                    "alpha_max": "NA",
                    "sigma_rgb_mean": "NA",
                    "sigma_freq_mean": "NA",
                    "sigma_diff_mean": "NA",
                    "lambda_sep_current": "NA",
                    "learning_rate": "NA",
                    "lr_group_0": "NA",
                    "lr_group_1": "NA",
                    "lr_group_2": "NA",
                    "lr_group_3": "NA",
                    "branch_mode": "NA",
                    "log_source": log_path,
                }
            )
            continue

        matched_train_step = record["step"]
        rows.append(
            {
                "exclude_class": args.exclude_class,
                "ckpt_step": ckpt_step,
                "matched_train_step": matched_train_step,
                "step_delta": matched_train_step - ckpt_step,
                "loss_total": record.get("loss_total", "NA"),
                "loss_dual": record.get("loss_dual", "NA"),
                "loss_sep": record.get("loss_sep", "NA"),
                "loss_rf": record.get("loss_rf", "NA"),
                "loss_ff": record.get("loss_ff", "NA"),
                "alpha_mean": record.get("alpha_mean", "NA"),
                "alpha_min": record.get("alpha_min", "NA"),
                "alpha_max": record.get("alpha_max", "NA"),
                "sigma_rgb_mean": "NA",
                "sigma_freq_mean": "NA",
                "sigma_diff_mean": "NA",
                "lambda_sep_current": record.get("lambda_sep_current", "NA"),
                "learning_rate": learning_rate_label(record),
                "lr_group_0": record.get("lr_group_0", "NA"),
                "lr_group_1": record.get("lr_group_1", "NA"),
                "lr_group_2": record.get("lr_group_2", "NA"),
                "lr_group_3": record.get("lr_group_3", "NA"),
                "branch_mode": branch_mode_label(record),
                "log_source": log_path,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(records)} DEBUG step record(s) from: {log_path}")
    print(f"Wrote {len(rows)} row(s) to: {args.out_csv}")
    print(
        "NOTE: sigma_rgb_mean / sigma_freq_mean / sigma_diff_mean are not recorded by "
        "train_ddfsd.py's training-loop logging and are filled with 'NA' (not fabricated)."
    )
    for row in rows:
        print(
            f"ckpt_step={row['ckpt_step']:>6} matched_train_step={row['matched_train_step']:>6} "
            f"loss_total={row['loss_total']} alpha_mean={row['alpha_mean']} "
            f"loss_rf={row['loss_rf']} loss_ff={row['loss_ff']}"
        )


if __name__ == "__main__":
    main()
