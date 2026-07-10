# -*- coding: utf-8 -*-
"""Summarize the DDFSD 20% GenImage experiment without running evaluation.

The tool is intentionally scoped to one caller-supplied 20% experiment root.  It
does not search other run directories, legacy 10% experiments, or baseline data.
All result values and conclusions come from CSV/status/config files below that root.

Commands
--------
per-class
    Build Markdown, a long-form result CSV, and a completeness CSV for one
    leave-one-out class.
all-classes (aliases: all, all6)
    Build the corresponding report for all requested classes (six by default),
    including an explicit missing-file/failed-stage CSV.

The evaluator wrappers may be interrupted before their merged CSV is produced.
For formal, branch-mode, and alpha-grid results, this script therefore reads each
``step_<N>`` CSV first and uses the merged CSV only as a per-step fallback.  Rows
are deterministically de-duplicated and no recursive globbing is used.
"""

import argparse
import csv
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RUN_CONFIG = "ddfsd_20pct_steps30000"
ALL_CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
CHECKPOINT_STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
FINAL_CANDIDATES = (25000, 30000)
ALPHA_STEPS = (15000, 25000, 30000)
EVAL_SEEDS = (42, 101, 102, 103, 104)
BRANCH_MODES = ("dual", "rgb-only", "freq-only")
ALPHA_MODES = ("adaptive", "0", "0.25", "0.5", "0.75", "1")
MISSING = "缺失"

DEFAULT_RUN_ROOT = (
    "/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/"
    "ddfsd_20pct_steps30000"
)
DEFAULT_DATA_ROOT = "/root/autodl-tmp/data_fsd_20pct/GenImage"

# Expected defaults are labels for audit only.  They must never be rendered as
# values that were actually used: actual values come exclusively from each
# class's PIPELINE_CONFIG.txt snapshot.
CONFIG_DEFAULTS = (
    ("DATA_ROOT", DEFAULT_DATA_ROOT),
    ("NUM_WORKERS", "8"),
    ("SEED", "42"),
    ("BATCH_SIZE", "16"),
    ("TOTAL_STEPS", "30000"),
    ("SAVE_INTERVAL", "5000"),
    ("EVAL_INTERVAL", "5000"),
    ("LOG_INTERVAL", "200"),
    ("LR_STEP", "10000"),
    ("LR_GAMMA", "0.5"),
    ("EXPECTED_CKPT_STEPS", "5000,10000,15000,20000,25000,30000"),
    ("FORMAL_EVAL_STEPS", "5000,10000,15000,20000,25000,30000"),
    ("BRANCH_MODE_STEPS", "5000,10000,15000,20000,25000,30000"),
    ("ALPHA_GRID_STEPS", "15000,25000,30000"),
    ("TRAIN_STATS_STEPS", "5000,10000,15000,20000,25000,30000"),
    ("EVAL_SEEDS", "42,101,102,103,104"),
    ("SUPPORT_SHOT", "10"),
    ("BRANCH_MODES", "dual,rgb-only,freq-only"),
    ("ALPHA_MODES", "adaptive,0.0,0.25,0.5,0.75,1.0"),
    ("ALPHA_BRANCH_MODES", "dual"),
    ("RGB_BACKBONE_LR", "3e-5"),
    ("FREQ_BACKBONE_LR", "3e-5"),
    ("RGB_HEAD_LR", "1e-4"),
    ("FREQ_HEAD_LR", "1e-4"),
    ("WEIGHT_DECAY", "1e-4"),
    ("TAU", "0.2"),
    ("TAU_R", "0.1"),
    ("M_RF", "1.2"),
    ("M_FF", "0.6"),
    ("LAMBDA_FF", "0.5"),
    ("LAMBDA_SEP_TARGET", "0.03"),
    ("LAMBDA_SEP_WARMUP_START", "5000"),
    ("LAMBDA_SEP_WARMUP_END", "15000"),
    ("BRANCH_DROPOUT_DUAL_PROB", "0.90"),
    ("BRANCH_DROPOUT_RGB_PROB", "0.05"),
    ("BRANCH_DROPOUT_FREQ_PROB", "0.05"),
    ("AUTO_COMPUTE_FREQ_STATS", "True"),
    ("USE_FP16", "True"),
    ("PRETRAINED", "True"),
    ("TRAIN_EPISODE", "3-way,5-shot,5-query"),
    ("VALIDATION_SHOT", "10"),
)

CONFIG_CRITICAL_KEYS = {
    "RUN_CONFIG",
    "RUN_ROOT",
    "EXCLUDE_CLASS",
    "OUTPUT_PATH",
    "FREQ_STATS_PATH",
    "SEED",
    "TOTAL_STEPS",
    "SAVE_INTERVAL",
    "EVAL_INTERVAL",
    "LOG_INTERVAL",
    "LR_STEP",
    "LR_GAMMA",
    "BATCH_SIZE",
    "EXPECTED_CKPT_STEPS",
    "FORMAL_EVAL_STEPS",
    "BRANCH_MODE_STEPS",
    "ALPHA_GRID_STEPS",
    "TRAIN_STATS_STEPS",
    "EVAL_SEEDS",
    "SUPPORT_SHOT",
    "BRANCH_MODES",
    "ALPHA_MODES",
    "ALPHA_BRANCH_MODES",
    "RGB_BACKBONE_LR",
    "FREQ_BACKBONE_LR",
    "RGB_HEAD_LR",
    "FREQ_HEAD_LR",
    "WEIGHT_DECAY",
    "TAU",
    "TAU_R",
    "M_RF",
    "M_FF",
    "LAMBDA_FF",
    "LAMBDA_SEP_TARGET",
    "LAMBDA_SEP_WARMUP_START",
    "LAMBDA_SEP_WARMUP_END",
    "BRANCH_DROPOUT_DUAL_PROB",
    "BRANCH_DROPOUT_RGB_PROB",
    "BRANCH_DROPOUT_FREQ_PROB",
    "AUTO_COMPUTE_FREQ_STATS",
    "USE_FP16",
    "PRETRAINED",
    "TRAIN_EPISODE",
    "VALIDATION_SHOT",
}

CONFIG_CONSISTENCY_KEYS = tuple(key for key, _default in CONFIG_DEFAULTS) + (
    "RUN_CONFIG",
    "RUN_ROOT",
)

SUMMARY_FIELDS = [
    "record_type",
    "exclude_class",
    "config_key",
    "config_value",
    "expected_value",
    "checkpoint",
    "seed",
    "branch_mode",
    "alpha_mode",
    "alpha_mode_raw",
    "fixed_alpha",
    "acc",
    "ap",
    "auc",
    "acc_mean",
    "acc_std",
    "ap_mean",
    "ap_std",
    "auc_mean",
    "auc_std",
    "eval_seeds",
    "adaptive_alpha_mean",
    "adaptive_alpha_min",
    "adaptive_alpha_max",
    "used_alpha_mean",
    "used_alpha_min",
    "used_alpha_max",
    "matched_train_step",
    "step_delta",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "loss_rf",
    "loss_ff",
    "loss_sep",
    "lambda_sep_current",
    "loss_dual",
    "loss_total",
    "selection",
    "status",
    "detail",
    "classes_available",
    "classes_expected",
    "source_path",
]

COMPLETENESS_FIELDS = [
    "exclude_class",
    "component",
    "expected",
    "observed",
    "status",
    "path",
    "detail",
]

MISSING_FAILURE_FIELDS = [
    "exclude_class",
    "kind",
    "component_or_stage",
    "status",
    "path",
    "detail",
]


def normalize_class(value: str) -> str:
    mapping = {name.lower(): name for name in ALL_CLASSES}
    normalized = mapping.get(value.strip().lower())
    if normalized is None:
        raise argparse.ArgumentTypeError(
            "exclude class must be one of: " + ", ".join(ALL_CLASSES)
        )
    return normalized


def parse_classes(value: str) -> List[str]:
    classes: List[str] = []
    for token in value.split(","):
        if not token.strip():
            continue
        name = normalize_class(token)
        if name not in classes:
            classes.append(name)
    if not classes:
        raise argparse.ArgumentTypeError("--classes must contain at least one class")
    return classes


def real_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def is_within(path: str, root: str) -> bool:
    path_real = os.path.normcase(real_path(path))
    root_real = os.path.normcase(real_path(root))
    try:
        return os.path.commonpath([path_real, root_real]) == root_real
    except ValueError:
        return False


def require_within(path: str, root: str, label: str) -> str:
    resolved = real_path(path)
    if not is_within(resolved, root):
        raise ValueError(
            f"{label} must stay inside the supplied 20% run_root: "
            f"{resolved!r} is outside {real_path(root)!r}"
        )
    return resolved


def resolve_run_root(value: str) -> str:
    """Resolve either the config root itself or its branch-level parent.

    A branch-level parent is mapped only to the exact RUN_CONFIG child; no other
    experiment directory is scanned.  This keeps every subsequent read under one
    unambiguous 20% root.
    """

    supplied = real_path(value)
    if os.path.basename(supplied.rstrip(os.sep)) == RUN_CONFIG:
        return supplied

    direct_class_paths = [
        os.path.join(supplied, f"exclude_{name}") for name in ALL_CLASSES
    ]
    if any(os.path.isdir(path) for path in direct_class_paths):
        return supplied
    return real_path(os.path.join(supplied, RUN_CONFIG))


def safe_join(root: str, *parts: str) -> str:
    return require_within(os.path.join(root, *parts), root, "result path")


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "NULL", MISSING}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def to_int(value) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def fmt(value, digits: int = 4) -> str:
    number = to_float(value)
    return MISSING if number is None else f"{number:.{digits}f}"


def fmt_mean_std(mean_value, std_value, digits: int = 4) -> str:
    mean_text = fmt(mean_value, digits)
    std_text = fmt(std_value, digits)
    if mean_text == MISSING:
        return MISSING
    if std_text == MISSING:
        return f"{mean_text} ± {MISSING}"
    return f"{mean_text} ± {std_text}"


def display_path(path: str) -> str:
    return path if path else MISSING


def normalize_alpha_mode(value) -> str:
    text = str(value or "").strip().lower()
    if text == "adaptive":
        return "adaptive"
    number = to_float(text)
    if number is None:
        return text
    return f"{number:g}"


def adaptive_display(row: Dict[str, str]) -> str:
    value = to_float(row.get("adaptive_alpha_mean"))
    if value is None:
        return "adaptive（adaptive具体数值缺失）"
    return f"adaptive（{value:.4f}）"


def read_csv_file(path: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not os.path.isfile(path):
        return [], None
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], "CSV 缺少表头"
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"CSV 读取失败: {exc}"
    for row in rows:
        row["_source_path"] = path
    return rows, None


def filter_class_rows(
    rows: Iterable[Dict[str, str]], exclude_class: str
) -> Tuple[List[Dict[str, str]], int]:
    kept: List[Dict[str, str]] = []
    mismatched = 0
    for row in rows:
        row_class = str(row.get("exclude_class", "")).strip()
        if row_class and row_class.lower() != exclude_class.lower():
            mismatched += 1
            continue
        kept.append(row)
    return kept, mismatched


def dedupe_rows(
    rows: Iterable[Dict[str, str]], key_fields: Sequence[str]
) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def row_step(row: Dict[str, str]) -> Optional[int]:
    return to_int(row.get("ckpt_step", row.get("checkpoint")))


def load_step_preferred_csvs(
    *,
    run_root: str,
    base_dir: str,
    expected_steps: Sequence[int],
    step_filename,
    merged_filename: str,
    key_fields: Sequence[str],
    exclude_class: str,
) -> Tuple[List[Dict[str, str]], Dict[int, str], List[str], List[str]]:
    """Read step files first and a single fixed merged file as fallback.

    Returns rows, chosen source per step, read errors, and class-mismatch notes.
    Every constructed path is checked against run_root before opening.
    """

    base_dir = require_within(base_dir, run_root, "result directory")
    merged_path = require_within(
        os.path.join(base_dir, merged_filename), run_root, "merged CSV"
    )
    merged_rows, merged_error = read_csv_file(merged_path)
    merged_rows, merged_mismatch = filter_class_rows(merged_rows, exclude_class)
    errors: List[str] = []
    mismatch_notes: List[str] = []
    if merged_error:
        errors.append(f"{merged_path}: {merged_error}")
    if merged_mismatch:
        mismatch_notes.append(
            f"{merged_path}: 忽略 {merged_mismatch} 行 exclude_class 不匹配数据"
        )
    merged_by_step: Dict[int, List[Dict[str, str]]] = {}
    for row in merged_rows:
        step = row_step(row)
        if step in expected_steps:
            merged_by_step.setdefault(step, []).append(row)

    chosen_rows: List[Dict[str, str]] = []
    source_by_step: Dict[int, str] = {}
    for step in expected_steps:
        step_path = require_within(
            os.path.join(base_dir, f"step_{step}", step_filename(step)),
            run_root,
            "per-step CSV",
        )
        step_rows, step_error = read_csv_file(step_path)
        step_rows, step_mismatch = filter_class_rows(step_rows, exclude_class)
        step_rows = [row for row in step_rows if row_step(row) == step]
        if step_error:
            errors.append(f"{step_path}: {step_error}")
        if step_mismatch:
            mismatch_notes.append(
                f"{step_path}: 忽略 {step_mismatch} 行 exclude_class 不匹配数据"
            )
        # Prefer each key from the per-step file, but fill keys that are absent
        # there from the merged CSV.  This matters when a process was interrupted
        # after writing only part of one step's mode/seed grid.
        selected = list(step_rows)
        step_keys = {
            tuple(str(row.get(field, "")).strip() for field in key_fields)
            for row in selected
        }
        for merged_row in merged_by_step.get(step, []):
            merged_key = tuple(
                str(merged_row.get(field, "")).strip() for field in key_fields
            )
            if merged_key not in step_keys:
                selected.append(merged_row)
                step_keys.add(merged_key)
        if step_rows:
            source_by_step[step] = step_path
        elif selected:
            source_by_step[step] = merged_path
        chosen_rows.extend(selected)

    return (
        dedupe_rows(chosen_rows, key_fields),
        source_by_step,
        errors,
        mismatch_notes,
    )


def parse_status_file(path: str) -> Tuple[Dict[str, str], List[str], Optional[str]]:
    if not os.path.isfile(path):
        return {}, [], None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        return {}, [], str(exc)

    # Pipeline status files are append-only so earlier failed attempts remain
    # available for audit.  Current completeness must be based on the latest
    # attempt, otherwise a successfully recovered class would be reported as
    # failed forever because of historical FAILED/MISSING lines.
    attempt_starts = [
        index for index, line in enumerate(lines) if line.startswith("pipeline_attempt_start=")
    ]
    if attempt_starts:
        lines = lines[attempt_starts[-1] :]

    values: Dict[str, str] = {}
    failed: List[str] = []
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)\s*=\s*(.*)$")
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        values[key] = value
        upper = value.upper()
        if "FAIL" in upper or "ERROR" in upper or "MISSING" in upper:
            failed.append(f"{key}={value}")
    return values, failed, None


def read_config_snapshot(path: str) -> Tuple[Dict[str, str], List[str], Optional[str]]:
    """Read one fixed PIPELINE_CONFIG.txt without following any other path."""

    if not os.path.isfile(path):
        return {}, [], None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\r\n") for line in handle]
    except OSError as exc:
        return {}, [], str(exc)
    values: Dict[str, str] = {}
    malformed: List[str] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            malformed.append(f"line {line_number}: {stripped}")
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            malformed.append(f"line {line_number}: {stripped}")
            continue
        values[key] = value.strip()
    return values, malformed, None


def config_expectations(data) -> List[Tuple[str, str]]:
    run_parent = os.path.dirname(data.run_root.rstrip(os.sep))
    dynamic = [
        ("EXCLUDE_CLASS", data.exclude_class),
        ("RUN_ROOT", run_parent),
        ("RUN_CONFIG", RUN_CONFIG),
        ("OUTPUT_PATH", data.output_path),
        ("FREQ_STATS_PATH", data.paths["freq_stats"]),
    ]
    return dynamic + list(CONFIG_DEFAULTS)


def config_values_equal(key: str, actual: str, expected: str) -> bool:
    actual, expected = str(actual).strip(), str(expected).strip()
    if key.endswith("_PATH") or key in {"RUN_ROOT", "DATA_ROOT"}:
        return os.path.normpath(actual) == os.path.normpath(expected)
    if "," in actual or "," in expected:
        left = [item.strip() for item in actual.split(",")]
        right = [item.strip() for item in expected.split(",")]
        return left == right
    left_number, right_number = to_float(actual), to_float(expected)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return actual.lower() == expected.lower()


def config_snapshot_state(data) -> Tuple[str, List[str], List[Tuple[str, str, str]]]:
    if data.config_error:
        return "失败", [f"读取失败: {data.config_error}"], []
    if not os.path.isfile(data.paths["config"]):
        return MISSING, ["PIPELINE_CONFIG.txt 不存在"], []
    expected = config_expectations(data)
    missing = [key for key, _default in expected if not data.config_values.get(key, "").strip()]
    differences = [
        (key, data.config_values[key], default)
        for key, default in expected
        if key in CONFIG_CRITICAL_KEYS
        and data.config_values.get(key, "").strip()
        and not config_values_equal(key, data.config_values[key], default)
    ]
    details = []
    if missing:
        details.append("缺失字段=" + ",".join(missing))
    if data.config_malformed:
        details.append("无法解析=" + "; ".join(data.config_malformed))
    if differences:
        details.append(
            "关键配置差异="
            + "; ".join(
                f"{key}:实际={actual},预期默认={default}"
                for key, actual, default in differences
            )
        )
    if differences:
        status = "配置差异"
    elif missing or data.config_malformed:
        status = "部分缺失"
    else:
        status = "完整"
    return status, details, differences


@dataclass
class ClassData:
    exclude_class: str
    run_root: str
    output_path: str
    paths: Dict[str, str]
    formal_rows: List[Dict[str, str]] = field(default_factory=list)
    formal_seed_rows: List[Dict[str, str]] = field(default_factory=list)
    branch_rows: List[Dict[str, str]] = field(default_factory=list)
    branch_seed_rows: List[Dict[str, str]] = field(default_factory=list)
    alpha_rows: List[Dict[str, str]] = field(default_factory=list)
    alpha_seed_rows: List[Dict[str, str]] = field(default_factory=list)
    train_rows: List[Dict[str, str]] = field(default_factory=list)
    source_by_component: Dict[str, Dict[int, str]] = field(default_factory=dict)
    read_errors: List[str] = field(default_factory=list)
    mismatch_notes: List[str] = field(default_factory=list)
    status_values: Dict[str, str] = field(default_factory=dict)
    failed_statuses: List[str] = field(default_factory=list)
    config_values: Dict[str, str] = field(default_factory=dict)
    config_malformed: List[str] = field(default_factory=list)
    config_error: Optional[str] = None
    completeness: List[Dict[str, str]] = field(default_factory=list)


def build_class_paths(run_root: str, exclude_class: str, output_path: str) -> Dict[str, str]:
    output_path = require_within(output_path, run_root, "output_path")
    return {
        "output_path": output_path,
        "status": safe_join(output_path, "PIPELINE_STATUS.txt"),
        "config": safe_join(output_path, "PIPELINE_CONFIG.txt"),
        "freq_stats": safe_join(output_path, "freq_stats.pt"),
        "ckpt_dir": safe_join(output_path, "ckpt"),
        "formal_dir": safe_join(output_path, "formal_eval"),
        "branch_dir": safe_join(output_path, "branch_modes"),
        "alpha_dir": safe_join(output_path, "alpha_grid"),
        "train_csv": safe_join(
            output_path,
            "csv",
            f"ddfsd_{exclude_class}_train_alpha_loss_by_ckpt.csv",
        ),
    }


def load_class_data(run_root: str, exclude_class: str, output_path: str) -> ClassData:
    paths = build_class_paths(run_root, exclude_class, output_path)
    data = ClassData(exclude_class, run_root, paths["output_path"], paths)

    formal_rows, formal_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["formal_dir"],
        expected_steps=CHECKPOINT_STEPS,
        step_filename=lambda _step: "ddfsd_eval_summary.csv",
        merged_filename="ddfsd_eval_summary_all_steps.csv",
        key_fields=("ckpt_step",),
        exclude_class=exclude_class,
    )
    data.formal_rows = formal_rows
    data.source_by_component["formal_summary"] = formal_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    seed_rows, seed_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["formal_dir"],
        expected_steps=CHECKPOINT_STEPS,
        step_filename=lambda _step: "ddfsd_eval_per_seed.csv",
        merged_filename="ddfsd_eval_per_seed_all_steps.csv",
        key_fields=("ckpt_step", "seed"),
        exclude_class=exclude_class,
    )
    data.formal_seed_rows = seed_rows
    data.source_by_component["formal_per_seed"] = seed_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    branch_rows, branch_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["branch_dir"],
        expected_steps=CHECKPOINT_STEPS,
        step_filename=lambda _step: f"ddfsd_{exclude_class}_branch_modes_summary.csv",
        merged_filename=f"ddfsd_{exclude_class}_branch_modes_summary.csv",
        key_fields=("ckpt_step", "branch_mode"),
        exclude_class=exclude_class,
    )
    data.branch_rows = branch_rows
    data.source_by_component["branch_summary"] = branch_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    branch_seed_rows, branch_seed_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["branch_dir"],
        expected_steps=CHECKPOINT_STEPS,
        step_filename=lambda _step: f"ddfsd_{exclude_class}_branch_modes_per_seed.csv",
        merged_filename=f"ddfsd_{exclude_class}_branch_modes_per_seed.csv",
        key_fields=("ckpt_step", "branch_mode", "seed"),
        exclude_class=exclude_class,
    )
    data.branch_seed_rows = branch_seed_rows
    data.source_by_component["branch_per_seed"] = branch_seed_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    alpha_rows, alpha_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["alpha_dir"],
        expected_steps=ALPHA_STEPS,
        step_filename=lambda _step: f"ddfsd_{exclude_class}_alpha_grid_summary.csv",
        merged_filename=f"ddfsd_{exclude_class}_alpha_grid_summary.csv",
        key_fields=("ckpt_step", "branch_mode", "alpha_mode"),
        exclude_class=exclude_class,
    )
    data.alpha_rows = [
        row for row in alpha_rows if str(row.get("branch_mode", "dual")) == "dual"
    ]
    data.source_by_component["alpha_summary"] = alpha_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    alpha_seed_rows, alpha_seed_sources, errors, notes = load_step_preferred_csvs(
        run_root=run_root,
        base_dir=paths["alpha_dir"],
        expected_steps=ALPHA_STEPS,
        step_filename=lambda _step: f"ddfsd_{exclude_class}_alpha_grid_per_seed.csv",
        merged_filename=f"ddfsd_{exclude_class}_alpha_grid_per_seed.csv",
        key_fields=("ckpt_step", "branch_mode", "alpha_mode", "seed"),
        exclude_class=exclude_class,
    )
    data.alpha_seed_rows = [
        row
        for row in alpha_seed_rows
        if str(row.get("branch_mode", "dual")) == "dual"
    ]
    data.source_by_component["alpha_per_seed"] = alpha_seed_sources
    data.read_errors.extend(errors)
    data.mismatch_notes.extend(notes)

    train_rows, train_error = read_csv_file(paths["train_csv"])
    train_rows, train_mismatch = filter_class_rows(train_rows, exclude_class)
    data.train_rows = dedupe_rows(
        [row for row in train_rows if row_step(row) in CHECKPOINT_STEPS],
        ("ckpt_step",),
    )
    if train_error:
        data.read_errors.append(f"{paths['train_csv']}: {train_error}")
    if train_mismatch:
        data.mismatch_notes.append(
            f"{paths['train_csv']}: 忽略 {train_mismatch} 行 exclude_class 不匹配数据"
        )

    values, failures, status_error = parse_status_file(paths["status"])
    data.status_values = values
    data.failed_statuses = failures
    if status_error:
        data.read_errors.append(f"{paths['status']}: 状态文件读取失败: {status_error}")

    config_values, config_malformed, config_error = read_config_snapshot(paths["config"])
    data.config_values = config_values
    data.config_malformed = config_malformed
    data.config_error = config_error
    if config_error:
        data.read_errors.append(f"{paths['config']}: 配置快照读取失败: {config_error}")

    data.completeness = build_completeness(data)
    return data


def combo_set(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> set:
    combos = set()
    for row in rows:
        values = []
        valid = True
        for field_name in fields:
            if field_name == "ckpt_step":
                value = row_step(row)
            elif field_name == "seed":
                value = to_int(row.get(field_name))
            elif field_name == "alpha_mode":
                value = normalize_alpha_mode(row.get(field_name))
            else:
                value = str(row.get(field_name, "")).strip()
            if value is None or value == "":
                valid = False
                break
            values.append(value)
        if valid:
            combos.add(tuple(values))
    return combos


def completeness_row(
    exclude_class: str,
    component: str,
    expected: str,
    observed: str,
    status: str,
    path: str,
    detail: str = "",
) -> Dict[str, str]:
    return {
        "exclude_class": exclude_class,
        "component": component,
        "expected": expected,
        "observed": observed,
        "status": status,
        "path": path,
        "detail": detail,
    }


def set_status(observed: int, expected: int, exists: bool = True) -> str:
    if observed >= expected:
        return "完整"
    if observed > 0:
        return "部分缺失"
    return MISSING if not exists or observed == 0 else "部分缺失"


def build_completeness(data: ClassData) -> List[Dict[str, str]]:
    cls = data.exclude_class
    rows: List[Dict[str, str]] = []
    rows.append(
        completeness_row(
            cls,
            "class_directory",
            "目录存在",
            "存在" if os.path.isdir(data.output_path) else MISSING,
            "完整" if os.path.isdir(data.output_path) else MISSING,
            data.output_path,
        )
    )
    status_exists = os.path.isfile(data.paths["status"])
    status_detail = (
        "; ".join(data.failed_statuses)
        if data.failed_statuses
        else ("未记录失败状态" if status_exists else "状态文件不存在")
    )
    rows.append(
        completeness_row(
            cls,
            "pipeline_status",
            "PIPELINE_STATUS.txt 且无 FAILED/MISSING/ERROR",
            "存在" if status_exists else MISSING,
            "失败" if data.failed_statuses else ("完整" if status_exists else MISSING),
            data.paths["status"],
            status_detail,
        )
    )
    config_status, config_details, _config_differences = config_snapshot_state(data)
    rows.append(
        completeness_row(
            cls,
            "config_snapshot",
            "PIPELINE_CONFIG.txt，字段完整且关键配置符合 20% 实验约定",
            "存在" if os.path.isfile(data.paths["config"]) else MISSING,
            config_status,
            data.paths["config"],
            "; ".join(config_details) if config_details else "实际配置快照完整",
        )
    )
    rows.append(
        completeness_row(
            cls,
            "freq_stats",
            "freq_stats.pt",
            "存在" if os.path.isfile(data.paths["freq_stats"]) else MISSING,
            "完整" if os.path.isfile(data.paths["freq_stats"]) else MISSING,
            data.paths["freq_stats"],
        )
    )

    ckpt_paths = [
        safe_join(data.paths["ckpt_dir"], f"ddfsd_step[{step}].pth")
        for step in CHECKPOINT_STEPS
    ]
    existing_ckpts = [step for step, path in zip(CHECKPOINT_STEPS, ckpt_paths) if os.path.isfile(path)]
    missing_ckpts = [step for step in CHECKPOINT_STEPS if step not in existing_ckpts]
    rows.append(
        completeness_row(
            cls,
            "checkpoints",
            str(len(CHECKPOINT_STEPS)),
            str(len(existing_ckpts)),
            set_status(len(existing_ckpts), len(CHECKPOINT_STEPS)),
            data.paths["ckpt_dir"],
            f"缺失 checkpoint: {missing_ckpts}" if missing_ckpts else "",
        )
    )

    formal_expected = {(step,) for step in CHECKPOINT_STEPS}
    formal_observed = combo_set(data.formal_rows, ("ckpt_step",))
    rows.append(
        completeness_row(
            cls,
            "formal_summary",
            str(len(formal_expected)),
            str(len(formal_observed & formal_expected)),
            set_status(len(formal_observed & formal_expected), len(formal_expected)),
            data.paths["formal_dir"],
            f"缺失 step: {sorted(step[0] for step in formal_expected - formal_observed)}",
        )
    )
    formal_seed_expected = {(step, seed) for step in CHECKPOINT_STEPS for seed in EVAL_SEEDS}
    formal_seed_observed = combo_set(data.formal_seed_rows, ("ckpt_step", "seed"))
    rows.append(
        completeness_row(
            cls,
            "formal_per_seed",
            str(len(formal_seed_expected)),
            str(len(formal_seed_observed & formal_seed_expected)),
            set_status(
                len(formal_seed_observed & formal_seed_expected), len(formal_seed_expected)
            ),
            data.paths["formal_dir"],
            "每个 checkpoint 应包含 seeds=" + ",".join(map(str, EVAL_SEEDS)),
        )
    )

    branch_expected = {
        (step, mode) for step in CHECKPOINT_STEPS for mode in BRANCH_MODES
    }
    branch_observed = combo_set(data.branch_rows, ("ckpt_step", "branch_mode"))
    rows.append(
        completeness_row(
            cls,
            "branch_modes_summary",
            str(len(branch_expected)),
            str(len(branch_observed & branch_expected)),
            set_status(len(branch_observed & branch_expected), len(branch_expected)),
            data.paths["branch_dir"],
            "缺失组合数=" + str(len(branch_expected - branch_observed)),
        )
    )
    branch_seed_expected = {
        (step, mode, seed)
        for step in CHECKPOINT_STEPS
        for mode in BRANCH_MODES
        for seed in EVAL_SEEDS
    }
    branch_seed_observed = combo_set(
        data.branch_seed_rows, ("ckpt_step", "branch_mode", "seed")
    )
    rows.append(
        completeness_row(
            cls,
            "branch_modes_per_seed",
            str(len(branch_seed_expected)),
            str(len(branch_seed_observed & branch_seed_expected)),
            set_status(
                len(branch_seed_observed & branch_seed_expected),
                len(branch_seed_expected),
            ),
            data.paths["branch_dir"],
            "缺失组合数=" + str(len(branch_seed_expected - branch_seed_observed)),
        )
    )

    alpha_expected = {(step, mode) for step in ALPHA_STEPS for mode in ALPHA_MODES}
    alpha_observed = combo_set(data.alpha_rows, ("ckpt_step", "alpha_mode"))
    rows.append(
        completeness_row(
            cls,
            "alpha_grid_summary",
            str(len(alpha_expected)),
            str(len(alpha_observed & alpha_expected)),
            set_status(len(alpha_observed & alpha_expected), len(alpha_expected)),
            data.paths["alpha_dir"],
            "缺失组合数=" + str(len(alpha_expected - alpha_observed)),
        )
    )
    alpha_seed_expected = {
        (step, mode, seed)
        for step in ALPHA_STEPS
        for mode in ALPHA_MODES
        for seed in EVAL_SEEDS
    }
    alpha_seed_observed = combo_set(
        data.alpha_seed_rows, ("ckpt_step", "alpha_mode", "seed")
    )
    rows.append(
        completeness_row(
            cls,
            "alpha_grid_per_seed",
            str(len(alpha_seed_expected)),
            str(len(alpha_seed_observed & alpha_seed_expected)),
            set_status(
                len(alpha_seed_observed & alpha_seed_expected), len(alpha_seed_expected)
            ),
            data.paths["alpha_dir"],
            "缺失组合数=" + str(len(alpha_seed_expected - alpha_seed_observed)),
        )
    )

    train_expected = {(step,) for step in CHECKPOINT_STEPS}
    train_observed = combo_set(data.train_rows, ("ckpt_step",))
    required_train_fields = (
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "loss_rf",
        "loss_ff",
        "loss_sep",
        "lambda_sep_current",
    )
    missing_values = []
    for row in data.train_rows:
        step = row_step(row)
        for key in required_train_fields:
            if to_float(row.get(key)) is None:
                missing_values.append(f"step{step}:{key}")
    train_count = len(train_observed & train_expected)
    train_status = set_status(train_count, len(train_expected))
    if train_status == "完整" and missing_values:
        train_status = "部分缺失"
    rows.append(
        completeness_row(
            cls,
            "train_alpha_loss",
            str(len(train_expected)),
            str(train_count),
            train_status,
            data.paths["train_csv"],
            (
                "缺失字段: " + ", ".join(missing_values)
                if missing_values
                else "训练统计字段完整"
            ),
        )
    )

    for index, message in enumerate(data.read_errors, start=1):
        rows.append(
            completeness_row(
                cls,
                f"read_error_{index}",
                "可读取",
                "读取失败",
                "失败",
                message.split(": ", 1)[0],
                message,
            )
        )
    for index, message in enumerate(data.mismatch_notes, start=1):
        rows.append(
            completeness_row(
                cls,
                f"class_mismatch_{index}",
                f"exclude_class={cls}",
                "存在不匹配行",
                "部分缺失",
                message.split(": ", 1)[0],
                message,
            )
        )
    return rows


def class_overall_status(data: ClassData) -> str:
    core = {
        "config_snapshot",
        "checkpoints",
        "formal_summary",
        "formal_per_seed",
        "branch_modes_summary",
        "branch_modes_per_seed",
        "alpha_grid_summary",
        "alpha_grid_per_seed",
        "train_alpha_loss",
    }
    statuses = [row["status"] for row in data.completeness if row["component"] in core]
    if statuses and all(status == "完整" for status in statuses) and not data.failed_statuses:
        return "完整"
    if any(
        (
            data.formal_rows,
            data.formal_seed_rows,
            data.branch_rows,
            data.alpha_rows,
            data.train_rows,
        )
    ):
        return "部分完成"
    if data.failed_statuses:
        return "失败（无可汇总结果）"
    return MISSING


def find_row(
    rows: Iterable[Dict[str, str]],
    step: int,
    **conditions: str,
) -> Optional[Dict[str, str]]:
    for row in rows:
        if row_step(row) != step:
            continue
        matches = True
        for key, expected in conditions.items():
            actual = row.get(key, "")
            if key == "alpha_mode":
                actual = normalize_alpha_mode(actual)
            if str(actual) != str(expected):
                matches = False
                break
        if matches:
            return row
    return None


def dominates(left: Dict[str, str], right: Dict[str, str]) -> Optional[bool]:
    left_ap, left_acc = to_float(left.get("ap_mean")), to_float(left.get("acc_mean"))
    right_ap, right_acc = to_float(right.get("ap_mean")), to_float(right.get("acc_mean"))
    if None in (left_ap, left_acc, right_ap, right_acc):
        return None
    return bool(
        left_ap >= right_ap
        and left_acc >= right_acc
        and (left_ap > right_ap or left_acc > right_acc)
    )


def compare_final_candidates(data: ClassData) -> Dict[str, str]:
    row_25 = find_row(data.formal_rows, 25000)
    row_30 = find_row(data.formal_rows, 30000)
    if row_25 is None and row_30 is None:
        return {
            "selection": MISSING,
            "status": MISSING,
            "detail": "25000 与 30000 正式 dual 结果均缺失",
        }
    if row_25 is None or row_30 is None:
        available = 30000 if row_30 is not None else 25000
        return {
            "selection": str(available),
            "status": "仅单候选，不判定最佳",
            "detail": f"仅 step{available} 有正式结果；另一最终候选缺失",
        }
    if None in (
        to_float(row_25.get("ap_mean")),
        to_float(row_25.get("acc_mean")),
        to_float(row_30.get("ap_mean")),
        to_float(row_30.get("acc_mean")),
    ):
        return {
            "selection": "无单一最佳",
            "status": "指标缺失",
            "detail": "两个候选至少一个缺少 AP 或 ACC，不能做双指标判定",
        }
    if dominates(row_25, row_30):
        return {
            "selection": "25000",
            "status": "单一最佳",
            "detail": "step25000 的 AP、ACC 均不差于 step30000，且至少一项更高",
        }
    if dominates(row_30, row_25):
        return {
            "selection": "30000",
            "status": "单一最佳",
            "detail": "step30000 的 AP、ACC 均不差于 step25000，且至少一项更高",
        }
    ap_25, acc_25 = to_float(row_25["ap_mean"]), to_float(row_25["acc_mean"])
    ap_30, acc_30 = to_float(row_30["ap_mean"]), to_float(row_30["acc_mean"])
    if ap_25 == ap_30 and acc_25 == acc_30:
        return {
            "selection": "25000/30000 并列",
            "status": "并列",
            "detail": "AP 与 ACC 均相同",
        }
    return {
        "selection": "无单一最佳",
        "status": "AP/ACC 冲突",
        "detail": (
            "AP 与 ACC 分别支持不同 checkpoint；保留两行原始结果，不使用 AP+ACC 求和"
        ),
    }


def mode_comparison(data: ClassData, step: int) -> str:
    rows = {
        mode: find_row(data.branch_rows, step, branch_mode=mode) for mode in BRANCH_MODES
    }
    if any(row is None for row in rows.values()):
        return f"step{step}: 三模式数据缺失，不能比较"
    winners = []
    for mode, row in rows.items():
        if all(
            other_mode == mode or dominates(row, other_row) is True
            for other_mode, other_row in rows.items()
        ):
            winners.append(mode)
    if len(winners) == 1:
        return f"step{step}: {winners[0]} 在 AP、ACC 两项上支配其余模式"
    return f"step{step}: AP/ACC 存在权衡或并列，无单一支配模式"


def alpha_comparison(data: ClassData, step: int) -> str:
    adaptive = find_row(data.alpha_rows, step, alpha_mode="adaptive")
    if adaptive is None:
        return f"step{step}: adaptive 结果缺失，不能比较"
    adaptive_label = adaptive_display(adaptive)
    fixed_rows = [
        (mode, find_row(data.alpha_rows, step, alpha_mode=mode))
        for mode in ALPHA_MODES
        if mode != "adaptive"
    ]
    fixed_rows = [(mode, row) for mode, row in fixed_rows if row is not None]
    if not fixed_rows:
        return f"step{step}: fixed alpha 结果缺失；{adaptive_label}"
    beats = [mode for mode, row in fixed_rows if dominates(row, adaptive) is True]
    adaptive_beats = [mode for mode, row in fixed_rows if dominates(adaptive, row) is True]
    incomparable = [
        mode
        for mode, row in fixed_rows
        if dominates(row, adaptive) is not True and dominates(adaptive, row) is not True
    ]
    if beats:
        return (
            f"step{step}: fixed alpha {','.join(beats)} 在 AP、ACC 上支配 "
            f"{adaptive_label}；其余不可单判={','.join(incomparable) or '无'}"
        )
    if len(adaptive_beats) == len(fixed_rows):
        return f"step{step}: {adaptive_label} 在 AP、ACC 上支配全部 fixed alpha"
    return (
        f"step{step}: {adaptive_label} 与 fixed alpha 存在 AP/ACC 权衡或并列，"
        "无单一结论"
    )


def trend_text(data: ClassData, key: str, label: str, digits: int = 6) -> str:
    points = []
    for row in sorted(data.train_rows, key=lambda item: row_step(item) or 0):
        value = to_float(row.get(key))
        step = row_step(row)
        if value is not None and step is not None:
            points.append((step, value))
    if len(points) < 2:
        return f"{label}: 缺失（不足两个训练 checkpoint 的实际数值）"
    first_step, first_value = points[0]
    last_step, last_value = points[-1]
    if last_value > first_value:
        direction = "上升"
    elif last_value < first_value:
        direction = "下降"
    else:
        direction = "持平"
    return (
        f"{label}: step{first_step}={first_value:.{digits}f}，"
        f"step{last_step}={last_value:.{digits}f}（首末值{direction}）"
    )


def md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    escaped_headers = [str(value).replace("|", "\\|") for value in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "|" + "|".join("---" for _ in escaped_headers) + "|",
    ]
    count = 0
    for row in rows:
        count += 1
        values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    if count == 0:
        values = [MISSING] + [""] * (len(headers) - 1)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def formal_table(data: ClassData) -> str:
    by_step = {row_step(row): row for row in data.formal_rows}
    rows = []
    for step in CHECKPOINT_STEPS:
        row = by_step.get(step)
        if row is None:
            rows.append((step, MISSING, MISSING, MISSING, "0/5", MISSING))
            continue
        seed_count = len(
            {
                to_int(seed_row.get("seed"))
                for seed_row in data.formal_seed_rows
                if row_step(seed_row) == step
                and to_int(seed_row.get("seed")) in EVAL_SEEDS
            }
        )
        rows.append(
            (
                step,
                fmt_mean_std(row.get("acc_mean"), row.get("acc_std")),
                fmt_mean_std(row.get("ap_mean"), row.get("ap_std")),
                fmt_mean_std(row.get("auc_mean"), row.get("auc_std")),
                f"{seed_count}/5",
                display_path(row.get("_source_path", "")),
            )
        )
    return md_table(
        ("checkpoint", "ACC mean ± std", "AP mean ± std", "AUC mean ± std", "seed 完整性", "来源"),
        rows,
    )


def formal_seed_table(data: ClassData) -> str:
    by_key = {
        (row_step(row), to_int(row.get("seed"))): row for row in data.formal_seed_rows
    }
    rows = []
    for step in CHECKPOINT_STEPS:
        for seed in EVAL_SEEDS:
            row = by_key.get((step, seed))
            if row is None:
                rows.append((step, seed, MISSING, MISSING, MISSING))
            else:
                rows.append(
                    (step, seed, fmt(row.get("acc")), fmt(row.get("ap")), fmt(row.get("auc")))
                )
    return md_table(("checkpoint", "seed", "ACC", "AP", "AUC"), rows)


def branch_table(data: ClassData) -> str:
    by_key = {
        (row_step(row), str(row.get("branch_mode", ""))): row
        for row in data.branch_rows
    }
    rows = []
    for step in CHECKPOINT_STEPS:
        for mode in BRANCH_MODES:
            row = by_key.get((step, mode))
            if row is None:
                rows.append((step, mode, MISSING, MISSING, MISSING, MISSING))
            else:
                adaptive = adaptive_display(row)
                rows.append(
                    (
                        step,
                        mode,
                        fmt_mean_std(row.get("acc_mean"), row.get("acc_std")),
                        fmt_mean_std(row.get("ap_mean"), row.get("ap_std")),
                        fmt_mean_std(row.get("auc_mean"), row.get("auc_std")),
                        adaptive,
                    )
                )
    return md_table(
        ("checkpoint", "模式", "ACC mean ± std", "AP mean ± std", "AUC mean ± std", "测试期 adaptive alpha"),
        rows,
    )


def alpha_table(data: ClassData) -> str:
    by_key = {
        (row_step(row), normalize_alpha_mode(row.get("alpha_mode"))): row
        for row in data.alpha_rows
    }
    rows = []
    for step in ALPHA_STEPS:
        for mode in ALPHA_MODES:
            row = by_key.get((step, mode))
            if row is None:
                label = "adaptive（adaptive具体数值缺失）" if mode == "adaptive" else mode
                rows.append((step, label, MISSING, MISSING, MISSING, MISSING, MISSING))
                continue
            label = adaptive_display(row) if mode == "adaptive" else mode
            rows.append(
                (
                    step,
                    label,
                    fmt_mean_std(row.get("acc_mean"), row.get("acc_std")),
                    fmt_mean_std(row.get("ap_mean"), row.get("ap_std")),
                    fmt_mean_std(row.get("auc_mean"), row.get("auc_std")),
                    fmt(row.get("used_alpha_mean")),
                    fmt(row.get("adaptive_alpha_mean")),
                )
            )
    return md_table(
        (
            "checkpoint",
            "alpha 模式（adaptive 含实测值）",
            "ACC mean ± std",
            "AP mean ± std",
            "AUC mean ± std",
            "used_alpha_mean",
            "测试期 adaptive_alpha_mean",
        ),
        rows,
    )


def train_table(data: ClassData) -> str:
    by_step = {row_step(row): row for row in data.train_rows}
    rows = []
    for step in CHECKPOINT_STEPS:
        row = by_step.get(step)
        if row is None:
            rows.append((step,) + (MISSING,) * 10)
            continue
        rows.append(
            (
                step,
                row.get("matched_train_step", MISSING) or MISSING,
                fmt(row.get("alpha_mean")),
                fmt(row.get("alpha_min")),
                fmt(row.get("alpha_max")),
                fmt(row.get("loss_rf"), 6),
                fmt(row.get("loss_ff"), 6),
                fmt(row.get("loss_sep"), 6),
                fmt(row.get("lambda_sep_current"), 6),
                fmt(row.get("loss_dual"), 6),
                fmt(row.get("loss_total"), 6),
            )
        )
    return md_table(
        (
            "checkpoint",
            "matched_train_step",
            "alpha_mean",
            "alpha_min",
            "alpha_max",
            "loss_rf",
            "loss_ff",
            "loss_sep",
            "lambda_sep_current",
            "loss_dual",
            "loss_total",
        ),
        rows,
    )


def final_comparison_table(data: ClassData) -> str:
    rows = []
    for step in FINAL_CANDIDATES:
        row = find_row(data.formal_rows, step)
        if row is None:
            rows.append((step, MISSING, MISSING, MISSING))
        else:
            rows.append(
                (
                    step,
                    fmt_mean_std(row.get("acc_mean"), row.get("acc_std")),
                    fmt_mean_std(row.get("ap_mean"), row.get("ap_std")),
                    fmt_mean_std(row.get("auc_mean"), row.get("auc_std")),
                )
            )
    return md_table(("checkpoint", "ACC mean ± std", "AP mean ± std", "AUC mean ± std"), rows)


def completeness_table(data: ClassData) -> str:
    return md_table(
        ("组件", "预期", "实际", "状态", "路径", "说明"),
        (
            (
                row["component"],
                row["expected"],
                row["observed"],
                row["status"],
                row["path"],
                row["detail"],
            )
            for row in data.completeness
        ),
    )


def class_phenomena(data: ClassData) -> List[str]:
    final = compare_final_candidates(data)
    lines = [
        f"最终候选：{final['status']}；选择={final['selection']}。{final['detail']}。",
        mode_comparison(data, 25000),
        mode_comparison(data, 30000),
        alpha_comparison(data, 25000),
        alpha_comparison(data, 30000),
        trend_text(data, "alpha_mean", "训练期 alpha_mean", 4),
        trend_text(data, "loss_rf", "训练期 loss_rf"),
        trend_text(data, "loss_ff", "训练期 loss_ff"),
        trend_text(data, "loss_sep", "训练期 loss_sep"),
    ]
    return lines


def config_snapshot_table(data: ClassData, data_root_hint: str = "") -> str:
    rows = []
    expected_items = config_expectations(data)
    expected_keys = {key for key, _default in expected_items}
    for key, default in expected_items:
        actual = data.config_values.get(key, "").strip()
        if not actual:
            display_actual = f"缺失（预期默认={default}）"
            status = MISSING
        elif config_values_equal(key, actual, default):
            display_actual = actual
            status = "与预期默认一致"
        else:
            display_actual = actual
            status = "实际快照覆盖默认" if key not in CONFIG_CRITICAL_KEYS else "关键配置差异"
        rows.append((key, display_actual, default, status))
    for key in sorted(set(data.config_values) - expected_keys):
        rows.append((key, data.config_values[key], "未设预期默认", "快照附加字段"))
    rows.extend(
        [
            ("REPORT_RUN_ROOT", data.run_root, "不适用", "汇总读取边界（非实验快照）"),
            ("REPORT_DATA_ROOT_HINT", data_root_hint, "不适用", "CLI 展示参数（非实验快照）"),
        ]
    )
    return md_table(("字段", "实际快照值", "预期默认", "状态"), rows)


def build_per_class_markdown(data: ClassData, data_root: str) -> str:
    final = compare_final_candidates(data)
    phenomena = "\n".join(
        f"{index}. {line}" for index, line in enumerate(class_phenomena(data), start=1)
    )
    return f"""# DDFSD 20% GenImage：exclude_{data.exclude_class} 结果汇总

> 读取边界：本报告只读取 `{data.run_root}` 内的固定相对路径；未读取 1/10、baseline 或其他旧实验目录。`缺失` 表示对应实际文件/行/字段不存在，未做推测或补值。

## 1. 实验配置

配置来源：`{data.paths['config']}`。只有“实际快照值”列代表本次运行实际记录；“预期默认”仅用于审计，快照缺失时不会被当作实际值。

{config_snapshot_table(data, data_root)}

## 2. 正式 dual 10-shot、5-seed 结果（6 checkpoints）

{formal_table(data)}

## 3. 正式测试逐 seed 结果

{formal_seed_table(data)}

## 4. dual / rgb-only / freq-only 三模式诊断

{branch_table(data)}

表中的 adaptive alpha 是测试期由 support set 实际测得的值，不等同于第 6 节训练期 alpha。

## 5. fixed alpha 诊断

{alpha_table(data)}

`adaptive` 行始终同时显示该 checkpoint 的实测 `adaptive_alpha_mean`；源 CSV 不含该值时明确写为 `adaptive具体数值缺失`。

## 6. 训练期 alpha / loss 统计

{train_table(data)}

## 7. step25000 与 step30000 最终候选比较

{final_comparison_table(data)}

- 判定状态：**{final['status']}**
- 结果：**{final['selection']}**
- 依据：{final['detail']}。

仅当一个 checkpoint 的 AP、ACC 均不差且至少一项更高时，才判为单一最佳；若 AP 与 ACC 各支持一个 checkpoint，则保留原始两行并标记冲突，不使用 AP+ACC 求和掩盖冲突。

## 8. 数据完整性检查

{completeness_table(data)}

## 9. 现象总结（只描述当前实际结果）

{phenomena}

以上均是当前一次实验文件支持的描述，不构成因果证明；数据不足处明确标为缺失或不能比较。
"""


def summary_row(record_type: str, data: ClassData, source: Dict[str, str]) -> Dict[str, str]:
    row = {field_name: "" for field_name in SUMMARY_FIELDS}
    row["record_type"] = record_type
    row["exclude_class"] = data.exclude_class
    row["checkpoint"] = str(row_step(source) or "")
    row["source_path"] = source.get("_source_path", "")
    mapping = {
        "seed": "seed",
        "branch_mode": "branch_mode",
        "fixed_alpha": "fixed_alpha",
        "acc": "acc",
        "ap": "ap",
        "auc": "auc",
        "acc_mean": "acc_mean",
        "acc_std": "acc_std",
        "ap_mean": "ap_mean",
        "ap_std": "ap_std",
        "auc_mean": "auc_mean",
        "auc_std": "auc_std",
        "eval_seeds": "eval_seeds",
        "adaptive_alpha_mean": "adaptive_alpha_mean",
        "adaptive_alpha_min": "adaptive_alpha_min",
        "adaptive_alpha_max": "adaptive_alpha_max",
        "used_alpha_mean": "used_alpha_mean",
        "used_alpha_min": "used_alpha_min",
        "used_alpha_max": "used_alpha_max",
        "matched_train_step": "matched_train_step",
        "step_delta": "step_delta",
        "alpha_mean": "alpha_mean",
        "alpha_min": "alpha_min",
        "alpha_max": "alpha_max",
        "loss_rf": "loss_rf",
        "loss_ff": "loss_ff",
        "loss_sep": "loss_sep",
        "lambda_sep_current": "lambda_sep_current",
        "loss_dual": "loss_dual",
        "loss_total": "loss_total",
    }
    for output_key, source_key in mapping.items():
        row[output_key] = str(source.get(source_key, ""))
    if "alpha_mode" in source:
        raw_mode = normalize_alpha_mode(source.get("alpha_mode"))
        row["alpha_mode_raw"] = raw_mode
        row["alpha_mode"] = adaptive_display(source) if raw_mode == "adaptive" else raw_mode
    return row


def class_summary_rows(data: ClassData) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for key, expected in config_expectations(data):
        actual = data.config_values.get(key, "").strip()
        config_row = {field_name: "" for field_name in SUMMARY_FIELDS}
        config_row.update(
            {
                "record_type": "config_snapshot",
                "exclude_class": data.exclude_class,
                "config_key": key,
                "config_value": actual if actual else MISSING,
                "expected_value": expected,
                "status": (
                    MISSING
                    if not actual
                    else (
                        "与预期默认一致"
                        if config_values_equal(key, actual, expected)
                        else "实际快照覆盖默认"
                    )
                ),
                "source_path": data.paths["config"],
            }
        )
        rows.append(config_row)
    rows.extend(summary_row("formal_dual_summary", data, row) for row in data.formal_rows)
    rows.extend(summary_row("formal_dual_per_seed", data, row) for row in data.formal_seed_rows)
    rows.extend(summary_row("branch_modes_summary", data, row) for row in data.branch_rows)
    rows.extend(summary_row("alpha_grid_summary", data, row) for row in data.alpha_rows)
    rows.extend(summary_row("train_alpha_loss", data, row) for row in data.train_rows)
    final = compare_final_candidates(data)
    final_row = {field_name: "" for field_name in SUMMARY_FIELDS}
    final_row.update(
        {
            "record_type": "final_candidate_decision",
            "exclude_class": data.exclude_class,
            "selection": final["selection"],
            "status": final["status"],
            "detail": final["detail"],
        }
    )
    rows.append(final_row)
    status_row = {field_name: "" for field_name in SUMMARY_FIELDS}
    status_row.update(
        {
            "record_type": "class_completeness",
            "exclude_class": data.exclude_class,
            "status": class_overall_status(data),
            "detail": "; ".join(data.failed_statuses),
            "source_path": data.paths["status"],
        }
    )
    rows.append(status_row)
    return rows


def macro_rows(class_data: Sequence[ClassData]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for step in CHECKPOINT_STEPS:
        complete_rows = []
        for data in class_data:
            row = find_row(data.formal_rows, step)
            observed_seeds = {
                to_int(seed_row.get("seed"))
                for seed_row in data.formal_seed_rows
                if row_step(seed_row) == step
                and to_int(seed_row.get("seed")) in EVAL_SEEDS
            }
            if (
                row is not None
                and observed_seeds == set(EVAL_SEEDS)
                and to_float(row.get("acc_mean")) is not None
                and to_float(row.get("ap_mean")) is not None
            ):
                complete_rows.append(row)
        out = {field_name: "" for field_name in SUMMARY_FIELDS}
        out.update(
            {
                "record_type": "formal_dual_macro_mean",
                "exclude_class": "ALL",
                "checkpoint": str(step),
                "classes_available": str(len(complete_rows)),
                "classes_expected": str(len(class_data)),
                "status": "完整" if len(complete_rows) == len(class_data) else "部分均值（非六类完整均值）",
            }
        )
        if complete_rows:
            out["acc_mean"] = str(statistics.mean(to_float(row["acc_mean"]) for row in complete_rows))
            out["ap_mean"] = str(statistics.mean(to_float(row["ap_mean"]) for row in complete_rows))
            auc_values = [to_float(row.get("auc_mean")) for row in complete_rows]
            auc_values = [value for value in auc_values if value is not None]
            if auc_values:
                out["auc_mean"] = str(statistics.mean(auc_values))
        else:
            out["status"] = MISSING
        output.append(out)
    return output


def component_status(data: ClassData, component: str) -> str:
    for row in data.completeness:
        if row["component"] == component:
            return row["status"]
    return MISSING


def all_config_snapshot_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        values = data.config_values
        rows.append(
            (
                data.exclude_class,
                component_status(data, "config_snapshot"),
                values.get("DATA_ROOT", MISSING) or MISSING,
                values.get("RUN_CONFIG", MISSING) or MISSING,
                values.get("TOTAL_STEPS", MISSING) or MISSING,
                values.get("EXPECTED_CKPT_STEPS", MISSING) or MISSING,
                values.get("EVAL_SEEDS", MISSING) or MISSING,
                values.get("SUPPORT_SHOT", MISSING) or MISSING,
                values.get("ALPHA_GRID_STEPS", MISSING) or MISSING,
            )
        )
    return md_table(
        (
            "exclude_class",
            "快照状态",
            "DATA_ROOT",
            "RUN_CONFIG",
            "TOTAL_STEPS",
            "checkpoint steps",
            "eval seeds",
            "support shot",
            "alpha steps",
        ),
        rows,
    )


def cross_class_config_issues(class_data: Sequence[ClassData]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for key in CONFIG_CONSISTENCY_KEYS:
        values = {
            data.exclude_class: data.config_values.get(key, "").strip()
            for data in class_data
        }
        present = [value for value in values.values() if value]
        missing_classes = [name for name, value in values.items() if not value]
        semantically_consistent = not present or all(
            config_values_equal(key, value, present[0]) for value in present[1:]
        )
        if semantically_consistent and not missing_classes:
            continue
        if semantically_consistent and missing_classes:
            status = "无法确认（部分快照字段缺失）"
        else:
            status = "配置不一致"
        issues.append(
            {
                "key": key,
                "status": status,
                "detail": "; ".join(
                    f"{name}={value if value else MISSING}" for name, value in values.items()
                ),
            }
        )
    return issues


def config_consistency_table(class_data: Sequence[ClassData]) -> str:
    issues_by_key = {item["key"]: item for item in cross_class_config_issues(class_data)}
    rows = []
    for key in CONFIG_CONSISTENCY_KEYS:
        values = {
            data.exclude_class: data.config_values.get(key, "").strip()
            for data in class_data
        }
        issue = issues_by_key.get(key)
        rows.append(
            (
                key,
                "一致" if issue is None else issue["status"],
                "; ".join(
                    f"{name}={value if value else MISSING}" for name, value in values.items()
                ),
            )
        )
    return md_table(("配置字段", "跨类一致性", "各类实际快照值"), rows)


def all_completeness_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        missing_count = sum(row["status"] != "完整" for row in data.completeness)
        rows.append(
            (
                data.exclude_class,
                class_overall_status(data),
                component_status(data, "config_snapshot"),
                len(data.formal_rows),
                len(data.formal_seed_rows),
                len(data.branch_rows),
                len(data.alpha_rows),
                len(data.train_rows),
                missing_count,
                "; ".join(data.failed_statuses) or "无已记录失败",
            )
        )
    return md_table(
        (
            "exclude_class",
            "完整性状态",
            "config snapshot",
            "formal rows",
            "formal seed rows",
            "branch rows",
            "alpha rows",
            "train rows",
            "非完整项数",
            "失败状态",
        ),
        rows,
    )


def all_formal_table(class_data: Sequence[ClassData], steps: Sequence[int]) -> str:
    rows = []
    for data in class_data:
        for step in steps:
            row = find_row(data.formal_rows, step)
            if row is None:
                rows.append((data.exclude_class, step, MISSING, MISSING, MISSING))
            else:
                rows.append(
                    (
                        data.exclude_class,
                        step,
                        fmt_mean_std(row.get("acc_mean"), row.get("acc_std")),
                        fmt_mean_std(row.get("ap_mean"), row.get("ap_std")),
                        fmt_mean_std(row.get("auc_mean"), row.get("auc_std")),
                    )
                )
    return md_table(
        ("exclude_class", "checkpoint", "ACC mean ± std", "AP mean ± std", "AUC mean ± std"),
        rows,
    )


def all_final_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        final = compare_final_candidates(data)
        row25 = find_row(data.formal_rows, 25000) or {}
        row30 = find_row(data.formal_rows, 30000) or {}
        rows.append(
            (
                data.exclude_class,
                fmt(row25.get("acc_mean")),
                fmt(row25.get("ap_mean")),
                fmt(row30.get("acc_mean")),
                fmt(row30.get("ap_mean")),
                final["selection"],
                final["status"],
                final["detail"],
            )
        )
    return md_table(
        (
            "exclude_class",
            "25000 ACC",
            "25000 AP",
            "30000 ACC",
            "30000 AP",
            "选择",
            "判定",
            "说明",
        ),
        rows,
    )


def all_branch_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        for step in CHECKPOINT_STEPS:
            for mode in BRANCH_MODES:
                row = find_row(data.branch_rows, step, branch_mode=mode)
                if row is None:
                    rows.append((data.exclude_class, step, mode, MISSING, MISSING, MISSING, MISSING))
                else:
                    rows.append(
                        (
                            data.exclude_class,
                            step,
                            mode,
                            fmt(row.get("acc_mean")),
                            fmt(row.get("ap_mean")),
                            fmt(row.get("auc_mean")),
                            adaptive_display(row),
                        )
                    )
    return md_table(
        ("exclude_class", "checkpoint", "模式", "ACC", "AP", "AUC", "测试期 adaptive alpha"),
        rows,
    )


def all_alpha_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        for step in ALPHA_STEPS:
            for mode in ALPHA_MODES:
                row = find_row(data.alpha_rows, step, alpha_mode=mode)
                if row is None:
                    label = "adaptive（adaptive具体数值缺失）" if mode == "adaptive" else mode
                    rows.append((data.exclude_class, step, label, MISSING, MISSING, MISSING))
                else:
                    label = adaptive_display(row) if mode == "adaptive" else mode
                    rows.append(
                        (
                            data.exclude_class,
                            step,
                            label,
                            fmt(row.get("acc_mean")),
                            fmt(row.get("ap_mean")),
                            fmt(row.get("used_alpha_mean")),
                        )
                    )
    return md_table(
        ("exclude_class", "checkpoint", "alpha 模式", "ACC", "AP", "used_alpha_mean"),
        rows,
    )


def all_train_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for data in class_data:
        by_step = {row_step(row): row for row in data.train_rows}
        for step in CHECKPOINT_STEPS:
            row = by_step.get(step)
            if row is None:
                rows.append((data.exclude_class, step) + (MISSING,) * 8)
            else:
                rows.append(
                    (
                        data.exclude_class,
                        step,
                        fmt(row.get("alpha_mean")),
                        fmt(row.get("alpha_min")),
                        fmt(row.get("alpha_max")),
                        fmt(row.get("loss_rf"), 6),
                        fmt(row.get("loss_ff"), 6),
                        fmt(row.get("loss_sep"), 6),
                        fmt(row.get("lambda_sep_current"), 6),
                        row.get("matched_train_step", MISSING) or MISSING,
                    )
                )
    return md_table(
        (
            "exclude_class",
            "checkpoint",
            "alpha_mean",
            "alpha_min",
            "alpha_max",
            "loss_rf",
            "loss_ff",
            "loss_sep",
            "lambda_sep_current",
            "matched_train_step",
        ),
        rows,
    )


def macro_table(class_data: Sequence[ClassData]) -> str:
    rows = []
    for row in macro_rows(class_data):
        rows.append(
            (
                row["checkpoint"],
                fmt(row.get("acc_mean")),
                fmt(row.get("ap_mean")),
                fmt(row.get("auc_mean")),
                f"{row['classes_available']}/{row['classes_expected']}",
                row["status"],
            )
        )
    return md_table(
        ("checkpoint", "ACC 宏均值", "AP 宏均值", "AUC 宏均值", "类别覆盖", "状态"),
        rows,
    )


def missing_failure_rows(class_data: Sequence[ClassData]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for data in class_data:
        for item in data.completeness:
            if item["status"] == "完整":
                continue
            rows.append(
                {
                    "exclude_class": data.exclude_class,
                    "kind": "missing_or_incomplete",
                    "component_or_stage": item["component"],
                    "status": item["status"],
                    "path": item["path"],
                    "detail": item["detail"],
                }
            )
        for failure in data.failed_statuses:
            stage = failure.split("=", 1)[0]
            rows.append(
                {
                    "exclude_class": data.exclude_class,
                    "kind": "failed_stage",
                    "component_or_stage": stage,
                    "status": "失败",
                    "path": data.paths["status"],
                    "detail": failure,
                }
            )
        _config_status, _config_details, config_differences = config_snapshot_state(data)
        for key, actual, expected in config_differences:
            rows.append(
                {
                    "exclude_class": data.exclude_class,
                    "kind": "config_expected_difference",
                    "component_or_stage": f"config:{key}",
                    "status": "配置差异",
                    "path": data.paths["config"],
                    "detail": f"实际={actual}; 预期默认={expected}",
                }
            )
    for issue in cross_class_config_issues(class_data):
        rows.append(
            {
                "exclude_class": "ALL",
                "kind": "cross_class_config_difference",
                "component_or_stage": f"config:{issue['key']}",
                "status": issue["status"],
                "path": ";".join(data.paths["config"] for data in class_data),
                "detail": issue["detail"],
            }
        )
    return rows


def missing_failure_table(class_data: Sequence[ClassData]) -> str:
    return md_table(
        ("exclude_class", "类型", "组件/阶段", "状态", "路径", "说明"),
        (
            (
                row["exclude_class"],
                row["kind"],
                row["component_or_stage"],
                row["status"],
                row["path"],
                row["detail"],
            )
            for row in missing_failure_rows(class_data)
        ),
    )


def build_all_markdown(class_data: Sequence[ClassData], run_root: str) -> str:
    class_names = ", ".join(data.exclude_class for data in class_data)
    return f"""# DDFSD 20% GenImage 全类结果汇总

> 读取边界：本报告只读取 `{run_root}` 内各 `exclude_<CLASS>` 的固定相对路径；未读取 1/10、baseline 或其他旧实验。覆盖类别：{class_names}。

## 1. 每类实际配置快照与跨类一致性

以下均来自各类固定路径 `exclude_<CLASS>/PIPELINE_CONFIG.txt`；缺失字段不会用硬编码默认值冒充实际值。

{all_config_snapshot_table(class_data)}

### 跨类关键配置一致性

{config_consistency_table(class_data)}

## 2. 六类数据完整性状态

{all_completeness_table(class_data)}

## 3. 每类正式 dual 结果（全部 6 checkpoints）

{all_formal_table(class_data, CHECKPOINT_STEPS)}

## 4. 每类 step25000 结果

{all_formal_table(class_data, (25000,))}

## 5. 每类 step30000 结果

{all_formal_table(class_data, (30000,))}

## 6. 每类 25000 / 30000 最终候选判定

{all_final_table(class_data)}

判定只采用双指标支配关系：一方 AP、ACC 均不差且至少一项更高才是单一最佳；AP/ACC 各胜一项时标为冲突，不以求和强行排序。

## 7. 每类三模式诊断

{all_branch_table(class_data)}

## 8. 每类 adaptive 与 fixed alpha 诊断

{all_alpha_table(class_data)}

adaptive 行均显示测试期实测 alpha；若源 CSV 缺少该值则明确写 `adaptive具体数值缺失`。

## 9. 每类训练期 alpha / loss 变化

{all_train_table(class_data)}

训练期 alpha 与测试期 adaptive alpha 是不同阶段的数据，不混用。

## 10. 类别总体均值

{macro_table(class_data)}

只有类别覆盖为 `{len(class_data)}/{len(class_data)}` 时才是所请求类别的完整总体均值；覆盖不足的行明确标为部分均值，不冒充六类均值。

## 11. 缺失文件、配置差异、数据与失败阶段清单

{missing_failure_table(class_data)}

以上结论仅来自当前 20% 实验根目录中的实际结果文件；所有缺失项均保留为 `缺失`，未读取或拼接 baseline/旧实验数据。
"""


def write_csv(path: str, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def check_outputs(paths: Sequence[str], no_clobber: bool) -> None:
    if not no_clobber:
        return
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "--no-clobber was set and output already exists: " + ", ".join(existing)
        )


def cmd_per_class(args) -> int:
    run_root = resolve_run_root(args.run_root)
    output_path = args.output_path or safe_join(run_root, f"exclude_{args.exclude_class}")
    output_path = require_within(output_path, run_root, "output_path")
    output_dir = args.output_dir or safe_join(output_path, "summary")
    output_dir = require_within(output_dir, run_root, "summary output directory")

    data = load_class_data(run_root, args.exclude_class, output_path)
    stem = f"ddfsd_{args.exclude_class}_20pct"
    markdown_path = safe_join(output_dir, f"{stem}_summary.md")
    summary_csv_path = safe_join(output_dir, f"{stem}_summary.csv")
    completeness_path = safe_join(output_dir, f"{stem}_completeness.csv")
    outputs = (markdown_path, summary_csv_path, completeness_path)
    check_outputs(outputs, args.no_clobber)

    write_text(markdown_path, build_per_class_markdown(data, args.data_root))
    write_csv(summary_csv_path, class_summary_rows(data), SUMMARY_FIELDS)
    write_csv(completeness_path, data.completeness, COMPLETENESS_FIELDS)
    for path in outputs:
        print(f"Wrote {path}")
    print(f"exclude_class={data.exclude_class} completeness={class_overall_status(data)}")
    return 0


def cmd_all_classes(args) -> int:
    run_root = resolve_run_root(args.run_root)
    classes = parse_classes(args.classes)
    class_data = [
        load_class_data(run_root, name, safe_join(run_root, f"exclude_{name}"))
        for name in classes
    ]
    output_dir = args.output_dir or safe_join(run_root, "summary")
    output_dir = require_within(output_dir, run_root, "summary output directory")

    markdown_path = safe_join(output_dir, "ddfsd_20pct_all_classes_summary.md")
    summary_csv_path = safe_join(output_dir, "ddfsd_20pct_all_classes_summary.csv")
    completeness_path = safe_join(output_dir, "ddfsd_20pct_all_classes_completeness.csv")
    missing_path = safe_join(output_dir, "ddfsd_20pct_missing_files_and_failures.csv")
    outputs = (markdown_path, summary_csv_path, completeness_path, missing_path)
    check_outputs(outputs, args.no_clobber)

    all_rows: List[Dict[str, str]] = []
    all_completeness: List[Dict[str, str]] = []
    for data in class_data:
        all_rows.extend(class_summary_rows(data))
        all_completeness.extend(data.completeness)
    all_rows.extend(macro_rows(class_data))

    write_text(markdown_path, build_all_markdown(class_data, run_root))
    write_csv(summary_csv_path, all_rows, SUMMARY_FIELDS)
    write_csv(completeness_path, all_completeness, COMPLETENESS_FIELDS)
    write_csv(missing_path, missing_failure_rows(class_data), MISSING_FAILURE_FIELDS)
    for path in outputs:
        print(f"Wrote {path}")
    for data in class_data:
        print(f"exclude_class={data.exclude_class} completeness={class_overall_status(data)}")
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-root",
        "--run_root",
        default=DEFAULT_RUN_ROOT,
        help="DDFSD 20%% experiment config root (ddfsd_20pct_steps30000)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        help="Report output directory; must remain inside run_root",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Fail instead of replacing an existing generated report",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize only the caller-supplied DDFSD 20% run root; "
            "never read baseline or legacy experiment results."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    per_class = subparsers.add_parser("per-class", help="summarize one exclude class")
    add_common_arguments(per_class)
    per_class.add_argument(
        "--exclude-class",
        "--exclude_class",
        required=True,
        type=normalize_class,
    )
    per_class.add_argument(
        "--output-path",
        "--output_path",
        default="",
        help="Class result directory; must remain inside run_root",
    )
    per_class.add_argument(
        "--data-root",
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="Displayed as configuration only; this tool never reads it",
    )
    per_class.set_defaults(func=cmd_per_class)

    all_classes = subparsers.add_parser(
        "all-classes",
        aliases=["all", "all6"],
        help="summarize every requested class (all six by default)",
    )
    add_common_arguments(all_classes)
    all_classes.add_argument(
        "--classes",
        default=",".join(ALL_CLASSES),
        help="Comma-separated subset; defaults to all six leave-one-out classes",
    )
    all_classes.set_defaults(func=cmd_all_classes)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
