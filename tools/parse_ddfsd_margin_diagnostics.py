# -*- coding: utf-8 -*-
"""Parse DDFSD training logs into margin-diagnostics CSVs.

Read-only, standalone analysis script. It does not touch train_ddfsd.py or
model/ddfsd_losses.py; it only parses the DEBUG dict lines that
train_ddfsd.py already writes via util.logger every ``--log_interval``
steps (logger.logkv/logkv_mean(...) -> logger.dumpkvs() ->
logging.debug(dict(...))).

Produces two CSVs per exclude_class:
  1. ``*_margin_diagnostics_by_log_interval.csv`` -- one row per log
     window (every ``--log_interval`` training steps).
  2. ``*_margin_diagnostics_by_ckpt.csv`` -- one row per requested
     checkpoint step, using the nearest available log window
     (``matched_train_step`` / ``step_delta`` record the mismatch, 0 when
     exact).

Fields that a given training run did not log (e.g. because it predates
this diagnostics patch) are filled with "NA" -- never fabricated.
"""

import argparse
import ast
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

DEBUG_LINE_RE = re.compile(r"DEBUG (\{.*\})\s*$")

# Fields copied verbatim from a DEBUG record when present.
DIRECT_FIELDS = [
    "m_rf",
    "m_ff",
    "lambda_ff",
    "lambda_sep_current",
    "loss_total",
    "loss_dual",
    "loss_sep",
    "loss_rf",
    "loss_ff",
    "weighted_loss_rf",
    "weighted_loss_ff",
    "weighted_loss_sep",
    "weighted_sep_to_dual_ratio",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "alpha_rf_pair_mean",
    "alpha_rf_pair_min",
    "alpha_rf_pair_max",
    "alpha_ff_pair_mean",
    "alpha_ff_pair_min",
    "alpha_ff_pair_max",
    "proto_rf_rgb_mean",
    "proto_rf_rgb_min",
    "proto_rf_rgb_max",
    "proto_rf_freq_mean",
    "proto_rf_freq_min",
    "proto_rf_freq_max",
    "proto_rf_fused_mean",
    "proto_rf_fused_min",
    "proto_rf_fused_max",
    "proto_rf_fused_p10",
    "proto_rf_fused_p25",
    "proto_rf_fused_p50",
    "proto_rf_fused_p75",
    "proto_rf_fused_p90",
    "proto_ff_rgb_mean",
    "proto_ff_rgb_min",
    "proto_ff_rgb_max",
    "proto_ff_freq_mean",
    "proto_ff_freq_min",
    "proto_ff_freq_max",
    "proto_ff_fused_mean",
    "proto_ff_fused_min",
    "proto_ff_fused_max",
    "proto_ff_fused_p10",
    "proto_ff_fused_p25",
    "proto_ff_fused_p50",
    "proto_ff_fused_p75",
    "proto_ff_fused_p90",
    "rf_violation_rate",
    "ff_violation_rate",
    "rf_violation_gap_mean",
    "rf_violation_gap_max",
    "ff_violation_gap_mean",
    "ff_violation_gap_max",
    "branch_mode_dual_ratio",
    "branch_mode_rgb_ratio",
    "branch_mode_freq_ratio",
]

INTERVAL_FIELDS = [
    "exclude_class",
    "step_start",
    "step_end",
    "matched_step",
] + DIRECT_FIELDS + ["log_source"]

CKPT_FIELDS = [
    "exclude_class",
    "ckpt_step",
    "matched_train_step",
    "step_delta",
] + DIRECT_FIELDS + ["log_source"]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse DDFSD training logs into margin-diagnostics CSVs (by log interval and by checkpoint)."
    )
    parser.add_argument("--exclude_class", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="<RUN_ROOT>/.../exclude_<CLASS> directory")
    parser.add_argument("--log_path", type=str, default="", help="Explicit log file. Auto-discovered under <output_path>/logs/ if omitted.")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--ckpt_steps", type=str, default="2500,5000,7500,10000,12500,15000")
    parser.add_argument("--out_csv_by_interval", type=str, required=True)
    parser.add_argument("--out_csv_by_ckpt", type=str, required=True)
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

    for pattern in (f"train_{exclude_class}_10pct.log", f"train_{exclude_class}*.log"):
        candidates.extend(sorted(glob.glob(os.path.join(output_path, "logs", pattern))))

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


def find_nearest_record(records: List[Dict], target_step: int) -> Optional[Dict]:
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
    return best_record


def record_to_row(record: Dict, log_path: str) -> Dict:
    row = {}
    for field in DIRECT_FIELDS:
        row[field] = record.get(field, "NA")
    row["log_source"] = log_path
    return row


def na_row(fields: List[str], log_path: str) -> Dict:
    row = {field: "NA" for field in fields}
    row["log_source"] = log_path
    return row


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

    # --- by_log_interval: one row per logged window ---
    interval_rows = []
    for record in records:
        step_end = record["step"]
        step_start = step_end - args.log_interval + 1
        row = {
            "exclude_class": args.exclude_class,
            "step_start": step_start,
            "step_end": step_end,
            "matched_step": step_end,
        }
        row.update(record_to_row(record, log_path))
        interval_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv_by_interval)), exist_ok=True)
    with open(args.out_csv_by_interval, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERVAL_FIELDS)
        writer.writeheader()
        writer.writerows(interval_rows)

    # --- by_ckpt: nearest available window for each requested checkpoint ---
    ckpt_rows = []
    for ckpt_step in ckpt_steps:
        record = find_nearest_record(records, ckpt_step)
        if record is None:
            row = {
                "exclude_class": args.exclude_class,
                "ckpt_step": ckpt_step,
                "matched_train_step": "NA",
                "step_delta": "NA",
            }
            row.update(na_row(DIRECT_FIELDS, log_path))
            ckpt_rows.append(row)
            continue

        matched_train_step = record["step"]
        row = {
            "exclude_class": args.exclude_class,
            "ckpt_step": ckpt_step,
            "matched_train_step": matched_train_step,
            "step_delta": matched_train_step - ckpt_step,
        }
        row.update(record_to_row(record, log_path))
        ckpt_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv_by_ckpt)), exist_ok=True)
    with open(args.out_csv_by_ckpt, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CKPT_FIELDS)
        writer.writeheader()
        writer.writerows(ckpt_rows)

    print(f"Parsed {len(records)} DEBUG step record(s) from: {log_path}")
    print(f"Wrote {len(interval_rows)} row(s) to: {args.out_csv_by_interval}")
    print(f"Wrote {len(ckpt_rows)} row(s) to: {args.out_csv_by_ckpt}")
    missing_margin_fields = [
        field
        for field in ("proto_rf_fused_mean", "rf_violation_rate", "weighted_loss_sep")
        if records and field not in records[-1]
    ]
    if missing_margin_fields:
        print(
            "WARNING: the parsed log does not contain margin diagnostics fields "
            f"{missing_margin_fields}; this run likely predates the diagnostics patch. "
            "Those columns are filled with 'NA'."
        )


if __name__ == "__main__":
    main()
