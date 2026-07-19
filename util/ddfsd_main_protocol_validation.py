"""Configuration validation for formal DDFSD main-protocol shot ablations."""

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence

from util.ddfsd_main_protocol import FAKE_CLASSES


FORMAL_PROTOCOL = "main-protocol formal shot ablation"
PATH_FIELDS = {"data_root", "ckpt_path", "freq_stats_path"}
FLOAT_FIELDS = {"tau", "tau_r"}
COMPLETION_FIELDS = (
    "protocol",
    "git_commit",
    "exclude_class",
    "shot",
    "seeds",
    "data_root",
    "ckpt_path",
    "ckpt_step",
    "freq_stats_path",
    "checkpoint_model_mode",
    "model_mode",
    "branch_mode",
    "tau",
    "tau_r",
    "max_eval_query_per_class",
    "zero_shot_metadata_per_class",
    "strict_formal_eval_images",
)
CONSISTENT_AGGREGATE_FIELDS = (
    "git_commit",
    "seeds",
    "data_root",
    "ckpt_step",
    "checkpoint_model_mode",
    "model_mode",
    "branch_mode",
    "tau",
    "tau_r",
    "max_eval_query_per_class",
    "zero_shot_metadata_per_class",
    "strict_formal_eval_images",
)
ZERO_SHOT_COMPLETION_FIELDS = (
    "held_out_class",
    "metadata_samples_per_class",
    "metadata_sampling_mode",
    "full_train_audit",
)
METRIC_FIELDS = ("acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc")


def normalize_path(value: object) -> str:
    if value in (None, ""):
        return ""
    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))
    )


def resolve_reference_csv(explicit_path, candidates: Sequence[str]) -> str:
    """Resolve an explicit parity CSV or require exactly one default candidate."""

    if explicit_path is not None:
        resolved = normalize_path(explicit_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"Explicit REFERENCE_CSV does not exist: {resolved}"
            )
        return resolved

    resolved_candidates = []
    for candidate in candidates:
        resolved = normalize_path(candidate)
        if resolved not in resolved_candidates:
            resolved_candidates.append(resolved)
    existing = [path for path in resolved_candidates if os.path.isfile(path)]
    if len(existing) == 1:
        return existing[0]
    listed = "\n- ".join(resolved_candidates)
    if not existing:
        raise FileNotFoundError(
            "No default reference CSV exists. Checked:\n- " + listed
        )
    raise ValueError(
        "Ambiguous reference CSV: multiple default candidates exist. "
        "Set REFERENCE_CSV explicitly. Existing:\n- " + "\n- ".join(existing)
    )


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def config_filename(shot: int) -> str:
    return "zero_shot_config.json" if int(shot) == 0 else "config.json"


def required_result_files(shot: int) -> List[str]:
    if int(shot) == 0:
        return [
            "ddfsd_zero_shot_per_seed.csv",
            "ddfsd_zero_shot_summary.csv",
            "zero_shot_metadata_manifest.csv",
            "zero_shot_invalid_images.csv",
            "formal_val_invalid_images.csv",
            "zero_shot_config.json",
            "eval.log",
        ]
    return [
        "ddfsd_eval_per_seed.csv",
        "ddfsd_eval_summary.csv",
        "support_query_manifest.csv",
        "formal_val_invalid_images.csv",
        "config.json",
        "eval.log",
    ]


def completion_fields(shot: int):
    if int(shot) == 0:
        return COMPLETION_FIELDS + ZERO_SHOT_COMPLETION_FIELDS
    return COMPLETION_FIELDS


def _same_value(field: str, actual: object, expected: object) -> bool:
    if field in PATH_FIELDS:
        return normalize_path(actual) == normalize_path(expected)
    if field in FLOAT_FIELDS:
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    if field in {
        "shot",
        "ckpt_step",
        "max_eval_query_per_class",
        "zero_shot_metadata_per_class",
        "metadata_samples_per_class",
    }:
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return False
    if field == "seeds":
        try:
            return [int(value) for value in actual] == [
                int(value) for value in expected
            ]
        except (TypeError, ValueError):
            return False
    if field == "strict_formal_eval_images":
        return actual is True and expected is True
    if field == "full_train_audit":
        return actual is False and expected is False
    return actual == expected


def compare_config(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    fields: Iterable[str] = COMPLETION_FIELDS,
) -> List[str]:
    errors = []
    for field in fields:
        if field not in actual:
            errors.append(f"{field}: missing (expected {expected.get(field)!r})")
            continue
        if field not in expected:
            errors.append(f"{field}: validator has no expected value")
            continue
        if not _same_value(field, actual[field], expected[field]):
            actual_value = (
                normalize_path(actual[field]) if field in PATH_FIELDS else actual[field]
            )
            expected_value = (
                normalize_path(expected[field])
                if field in PATH_FIELDS
                else expected[field]
            )
            errors.append(
                f"{field}: actual={actual_value!r}, expected={expected_value!r}"
            )
    return errors


def _read_csv_rows(path, required_fields, allow_empty=False):
    errors = []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if len(header) != len(set(header)):
                errors.append(f"{path}: CSV contains duplicate header columns")
            missing = [field for field in required_fields if field not in header]
            if missing:
                errors.append(f"{path}: missing CSV columns {missing}")
            for line_number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    errors.append(f"{path}:{line_number}: malformed CSV row")
                    continue
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"{path}: cannot parse CSV: {exc}")
        return [], errors
    if not allow_empty and not rows:
        errors.append(f"{path}: CSV has no data rows")
    return rows, errors


def _int_value(row, field, context, errors):
    try:
        return int(row.get(field, ""))
    except (TypeError, ValueError):
        errors.append(f"{context}: {field} is not an integer: {row.get(field)!r}")
        return None


def _finite_value(row, field, context, errors):
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        errors.append(f"{context}: {field} is not a float: {row.get(field)!r}")
        return None
    if not math.isfinite(value):
        errors.append(f"{context}: {field} is not finite: {row.get(field)!r}")
    return value


def _validate_audit_csvs(output_dir, shot):
    errors = []
    audit_path = os.path.join(output_dir, "formal_val_invalid_images.csv")
    audit_rows, audit_errors = _read_csv_rows(
        audit_path,
        ("data_class", "split", "dataset_index", "filepath", "error_type", "error"),
        allow_empty=True,
    )
    errors.extend(audit_errors)
    if audit_rows:
        errors.append(
            f"{audit_path}: formal val audit must contain only the header for a completed result"
        )
    if int(shot) == 0:
        _, invalid_errors = _read_csv_rows(
            os.path.join(output_dir, "zero_shot_invalid_images.csv"),
            (
                "held_out_class",
                "metadata_class",
                "seed",
                "image_path",
                "error_type",
                "error_message",
            ),
            allow_empty=True,
        )
        errors.extend(invalid_errors)
    return errors


def _validate_metric_csvs(output_dir, config):
    errors = []
    shot = int(config["shot"])
    class_name = str(config["exclude_class"])
    ckpt_step = int(config["ckpt_step"])
    expected_seeds = [int(seed) for seed in config["seeds"]]
    per_filename = (
        "ddfsd_zero_shot_per_seed.csv" if shot == 0 else "ddfsd_eval_per_seed.csv"
    )
    summary_filename = (
        "ddfsd_zero_shot_summary.csv" if shot == 0 else "ddfsd_eval_summary.csv"
    )
    shot_field = "shot" if shot == 0 else "support_shot"
    per_required = (
        "exclude_class",
        "seed",
        shot_field,
        "ckpt_step",
        *METRIC_FIELDS,
        "num_real_support",
        "num_fake_support",
        "num_real_query",
        "num_fake_query",
    )
    per_path = os.path.join(output_dir, per_filename)
    rows, csv_errors = _read_csv_rows(per_path, per_required)
    errors.extend(csv_errors)
    seen_seeds = set()
    per_seed_counts = {}
    for line_number, row in enumerate(rows, 2):
        context = f"{per_path}:{line_number}"
        seed = _int_value(row, "seed", context, errors)
        row_shot = _int_value(row, shot_field, context, errors)
        row_step = _int_value(row, "ckpt_step", context, errors)
        if row.get("exclude_class") != class_name:
            errors.append(
                f"{context}: exclude_class={row.get('exclude_class')!r}, expected {class_name!r}"
            )
        if row_shot is not None and row_shot != shot:
            errors.append(f"{context}: {shot_field}={row_shot}, expected {shot}")
        if row_step is not None and row_step != ckpt_step:
            errors.append(f"{context}: ckpt_step={row_step}, expected {ckpt_step}")
        if seed is not None:
            if seed in seen_seeds:
                errors.append(f"{per_path}: duplicate seed {seed}")
            seen_seeds.add(seed)
        for field in METRIC_FIELDS:
            _finite_value(row, field, context, errors)
        counts = {}
        for field in (
            "num_real_support",
            "num_fake_support",
            "num_real_query",
            "num_fake_query",
        ):
            counts[field] = _int_value(row, field, context, errors)
        if shot == 0:
            if counts["num_real_support"] not in (None, 0):
                errors.append(f"{context}: num_real_support must be 0")
            if counts["num_fake_support"] not in (None, 0):
                errors.append(f"{context}: num_fake_support must be 0")
        else:
            for field in ("num_real_support", "num_fake_support"):
                if counts[field] is not None and counts[field] != shot:
                    errors.append(f"{context}: {field}={counts[field]}, expected {shot}")
        for field in ("num_real_query", "num_fake_query"):
            if counts[field] is not None and counts[field] <= 0:
                errors.append(f"{context}: {field} must be positive")
        if seed is not None:
            per_seed_counts[seed] = counts
    expected_seed_set = set(expected_seeds)
    if seen_seeds != expected_seed_set:
        errors.append(
            f"{per_path}: seed set mismatch; missing={sorted(expected_seed_set - seen_seeds)}, "
            f"unexpected={sorted(seen_seeds - expected_seed_set)}"
        )

    summary_required = (
        "exclude_class",
        shot_field,
        "ckpt_step",
        "eval_seeds",
        *(f"{field}_{suffix}" for field in METRIC_FIELDS for suffix in ("mean", "std")),
    )
    summary_path = os.path.join(output_dir, summary_filename)
    summary_rows, summary_errors = _read_csv_rows(summary_path, summary_required)
    errors.extend(summary_errors)
    if len(summary_rows) != 1:
        errors.append(f"{summary_path}: summary must contain exactly one row")
    for line_number, row in enumerate(summary_rows, 2):
        context = f"{summary_path}:{line_number}"
        if row.get("exclude_class") != class_name:
            errors.append(f"{context}: exclude_class mismatch")
        row_shot = _int_value(row, shot_field, context, errors)
        row_step = _int_value(row, "ckpt_step", context, errors)
        if row_shot is not None and row_shot != shot:
            errors.append(f"{context}: {shot_field}={row_shot}, expected {shot}")
        if row_step is not None and row_step != ckpt_step:
            errors.append(f"{context}: ckpt_step={row_step}, expected {ckpt_step}")
        try:
            summary_seeds = [
                int(value.strip())
                for value in row.get("eval_seeds", "").split(",")
                if value.strip()
            ]
        except ValueError:
            summary_seeds = []
        if summary_seeds != expected_seeds:
            errors.append(
                f"{context}: eval_seeds={summary_seeds}, expected {expected_seeds}"
            )
        for field in METRIC_FIELDS:
            for suffix in ("mean", "std"):
                _finite_value(row, f"{field}_{suffix}", context, errors)
    return per_seed_counts, errors


def _validate_zero_shot_manifest(output_dir, config):
    errors = []
    path = os.path.join(output_dir, "zero_shot_metadata_manifest.csv")
    required = (
        "held_out_class",
        "metadata_class",
        "seed",
        "sample_rank",
        "dataset_index",
        "image_path",
    )
    rows, csv_errors = _read_csv_rows(path, required)
    errors.extend(csv_errors)
    held_out = str(config["exclude_class"])
    seeds = [int(seed) for seed in config["seeds"]]
    metadata_classes = list(config.get("metadata_classes", []))
    expected_metadata_classes = {"real", *[name for name in FAKE_CLASSES if name != held_out]}
    if len(metadata_classes) != 6 or len(metadata_classes) != len(set(metadata_classes)):
        errors.append(f"{path}: config metadata_classes must contain 6 unique classes")
    if set(metadata_classes) != expected_metadata_classes:
        errors.append(
            f"{path}: metadata_classes={sorted(set(metadata_classes))}, expected "
            f"{sorted(expected_metadata_classes)}"
        )
    if held_out in metadata_classes:
        errors.append(f"{path}: held-out class {held_out} appears in metadata_classes")
    try:
        count = int(config["metadata_samples_per_class"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{path}: invalid metadata_samples_per_class in config")
        count = 0
    expected_groups = {(name, seed) for name in metadata_classes for seed in seeds}
    groups = {}
    for line_number, row in enumerate(rows, 2):
        context = f"{path}:{line_number}"
        metadata_class = row.get("metadata_class")
        seed = _int_value(row, "seed", context, errors)
        rank = _int_value(row, "sample_rank", context, errors)
        index = _int_value(row, "dataset_index", context, errors)
        if row.get("held_out_class") != held_out:
            errors.append(f"{context}: held_out_class mismatch")
        group_key = (metadata_class, seed)
        if metadata_class not in metadata_classes or seed not in seeds:
            errors.append(f"{context}: unexpected metadata_class/seed {group_key}")
            continue
        group = groups.setdefault(
            group_key, {"count": 0, "ranks": set(), "indices": set(), "paths": set()}
        )
        group["count"] += 1
        if rank is not None:
            if rank in group["ranks"]:
                errors.append(f"{context}: duplicate sample_rank {rank} in {group_key}")
            group["ranks"].add(rank)
        if index is not None:
            if index in group["indices"]:
                errors.append(f"{context}: duplicate dataset_index {index} in {group_key}")
            group["indices"].add(index)
        image_path = normalize_path(row.get("image_path", ""))
        if not image_path:
            errors.append(f"{context}: image_path is empty")
        elif image_path in group["paths"]:
            errors.append(f"{context}: duplicate image_path in {group_key}: {image_path}")
        group["paths"].add(image_path)
    if set(groups) != expected_groups:
        errors.append(
            f"{path}: metadata class/seed groups mismatch; "
            f"missing={sorted(expected_groups - set(groups))}, "
            f"unexpected={sorted(set(groups) - expected_groups)}"
        )
    expected_ranks = set(range(1, count + 1))
    for group_key in sorted(expected_groups):
        group = groups.get(group_key)
        if group is None:
            continue
        if group["count"] != count:
            errors.append(
                f"{path}: group {group_key} has {group['count']} rows, expected {count}"
            )
        if group["ranks"] != expected_ranks:
            errors.append(f"{path}: group {group_key} sample_rank is not 1..{count}")
    return errors


def _validate_positive_manifest(output_dir, config, per_seed_counts):
    errors = []
    path = os.path.join(output_dir, "support_query_manifest.csv")
    required = (
        "exclude_class",
        "seed",
        "shot",
        "data_class",
        "split",
        "dataset_index",
        "filepath",
        "role",
        "support_rank",
    )
    rows, csv_errors = _read_csv_rows(path, required)
    errors.extend(csv_errors)
    held_out = str(config["exclude_class"])
    shot = int(config["shot"])
    seeds = [int(seed) for seed in config["seeds"]]
    legal_classes = ("real", held_out)
    groups = {}
    for line_number, row in enumerate(rows, 2):
        context = f"{path}:{line_number}"
        seed = _int_value(row, "seed", context, errors)
        row_shot = _int_value(row, "shot", context, errors)
        index = _int_value(row, "dataset_index", context, errors)
        data_class = row.get("data_class")
        role = row.get("role")
        if row.get("exclude_class") != held_out:
            errors.append(f"{context}: exclude_class mismatch")
        if row_shot is not None and row_shot != shot:
            errors.append(f"{context}: shot={row_shot}, expected {shot}")
        if seed not in seeds:
            errors.append(f"{context}: unexpected seed {seed}")
        if data_class not in legal_classes:
            errors.append(f"{context}: invalid data_class {data_class!r}")
        if role not in ("support", "query"):
            errors.append(f"{context}: invalid role {role!r}")
        if row.get("split") != "val":
            errors.append(f"{context}: split must be 'val'")
        if seed not in seeds or data_class not in legal_classes or role not in (
            "support",
            "query",
        ):
            continue
        group = groups.setdefault(
            (seed, data_class),
            {"support": set(), "query": set(), "support_ranks": set()},
        )
        if index is not None:
            if index in group[role]:
                errors.append(
                    f"{context}: duplicate {role} dataset_index {index} for {data_class}"
                )
            group[role].add(index)
        if role == "support":
            rank = _int_value(row, "support_rank", context, errors)
            if rank is not None:
                group["support_ranks"].add(rank)
    for seed in seeds:
        counts = per_seed_counts.get(seed, {})
        for data_class, count_field in (
            ("real", "num_real_query"),
            (held_out, "num_fake_query"),
        ):
            group = groups.get((seed, data_class))
            if group is None:
                errors.append(f"{path}: missing seed={seed} data_class={data_class}")
                continue
            if len(group["support"]) != shot:
                errors.append(
                    f"{path}: seed={seed} {data_class} support count "
                    f"{len(group['support'])}, expected {shot}"
                )
            if group["support_ranks"] != set(range(1, shot + 1)):
                errors.append(
                    f"{path}: seed={seed} {data_class} support_rank is not 1..{shot}"
                )
            overlap = group["support"] & group["query"]
            if overlap:
                errors.append(
                    f"{path}: seed={seed} {data_class} support/query overlap {sorted(overlap)}"
                )
            expected_query = counts.get(count_field)
            if expected_query is not None and len(group["query"]) != expected_query:
                errors.append(
                    f"{path}: seed={seed} {data_class} query count "
                    f"{len(group['query'])}, expected {expected_query} from per-seed CSV"
                )
    return errors


def validate_result_contents(output_dir, config):
    """Validate that formal result CSVs are structurally and semantically complete."""

    shot = int(config["shot"])
    errors = _validate_audit_csvs(output_dir, shot)
    per_seed_counts, metric_errors = _validate_metric_csvs(output_dir, config)
    errors.extend(metric_errors)
    if shot == 0:
        errors.extend(_validate_zero_shot_manifest(output_dir, config))
    else:
        errors.extend(
            _validate_positive_manifest(output_dir, config, per_seed_counts)
        )
    return errors


def validate_completed_result(
    output_dir: str, expected: Mapping[str, object]
) -> List[str]:
    """Return every reason an existing result cannot be safely skipped."""

    errors = []
    shot = int(expected["shot"])
    if not os.path.isdir(output_dir):
        return [f"result directory does not exist: {normalize_path(output_dir)}"]
    for filename in required_result_files(shot):
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            errors.append(f"required result file is missing or empty: {path}")
    config_path = os.path.join(output_dir, config_filename(shot))
    if not os.path.isfile(config_path):
        return errors
    try:
        config = load_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read configuration {config_path}: {exc}")
        return errors
    errors.extend(compare_config(config, expected, fields=completion_fields(shot)))
    content_files_present = all(
        os.path.isfile(os.path.join(output_dir, filename))
        for filename in required_result_files(shot)
    )
    if content_files_present:
        try:
            errors.extend(validate_result_contents(output_dir, config))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot validate result contents from configuration: {exc}")
    return errors


def load_result_configs(
    input_root: str,
    result_dir_template: str,
    classes: Sequence[str],
    shots: Sequence[int],
    ckpt_step: int,
) -> List[dict]:
    records = []
    for class_name in classes:
        for shot in shots:
            result_dir = normalize_path(
                result_dir_template.format(
                    input_root=input_root,
                    class_name=class_name,
                    **{"class": class_name},
                    shot=shot,
                    step=ckpt_step,
                )
            )
            path = os.path.join(result_dir, config_filename(shot))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing formal configuration for {class_name} shot={shot}: {path}"
                )
            records.append(
                {
                    "exclude_class": class_name,
                    "shot": int(shot),
                    "result_dir": result_dir,
                    "config_path": path,
                    "config": load_json(path),
                }
            )
    return records


def _path_has_class_marker(path: object, class_name: str) -> bool:
    normalized = normalize_path(path).replace("\\", "/").lower()
    marker = f"exclude_{class_name}".lower()
    return marker in normalized.split("/")


def validate_aggregate_configs(
    records: Sequence[Mapping[str, object]],
    classes: Sequence[str],
    shots: Sequence[int],
    seeds: Sequence[int],
    ckpt_step: int,
) -> List[str]:
    """Validate provenance and resource consistency before aggregation writes anything."""

    errors = []
    expected_keys = {(name, int(shot)) for name in classes for shot in shots}
    actual_keys = [
        (str(record["exclude_class"]), int(record["shot"])) for record in records
    ]
    if len(actual_keys) != len(set(actual_keys)):
        errors.append("duplicate exclude_class/shot configuration records")
    if set(actual_keys) != expected_keys:
        errors.append(
            f"configuration matrix mismatch: missing={sorted(expected_keys - set(actual_keys))}, "
            f"unexpected={sorted(set(actual_keys) - expected_keys)}"
        )

    common_values: Dict[str, object] = {}
    zero_shot_values: Dict[str, object] = {}
    resources: Dict[str, Dict[str, set]] = {
        name: {"ckpt_path": set(), "freq_stats_path": set()} for name in classes
    }
    for record in records:
        class_name = str(record["exclude_class"])
        shot = int(record["shot"])
        config = record["config"]
        label = f"{class_name}/shot_{shot} ({record['config_path']})"
        result_parts = (
            normalize_path(record["result_dir"]).replace("\\", "/").split("/")
        )
        if f"exclude_{class_name}".lower() not in [
            part.lower() for part in result_parts
        ]:
            errors.append(
                f"{label}: result directory lacks exclude_{class_name} component"
            )
        if f"shot_{shot}".lower() not in [part.lower() for part in result_parts]:
            errors.append(f"{label}: result directory lacks shot_{shot} component")
        expected = {
            "protocol": FORMAL_PROTOCOL,
            "exclude_class": class_name,
            "shot": shot,
            "seeds": list(seeds),
            "ckpt_step": ckpt_step,
            "max_eval_query_per_class": 0,
            "strict_formal_eval_images": True,
        }
        if shot == 0:
            expected.update(
                {
                    "held_out_class": class_name,
                    "metadata_samples_per_class": config.get(
                        "zero_shot_metadata_per_class"
                    ),
                    "metadata_sampling_mode": (
                        "deterministic_lazy_strict_until_full"
                    ),
                    "full_train_audit": False,
                }
            )
        errors.extend(
            f"{label}: {error}"
            for error in compare_config(
                config,
                expected,
                fields=expected.keys(),
            )
        )
        if normalize_path(record["result_dir"]) != normalize_path(
            os.path.dirname(str(record["config_path"]))
        ):
            errors.append(f"{label}: config path is outside its result directory")

        for field in CONSISTENT_AGGREGATE_FIELDS:
            if field not in config:
                errors.append(f"{label}: missing {field}")
                continue
            value = (
                normalize_path(config[field]) if field in PATH_FIELDS else config[field]
            )
            if field not in common_values:
                common_values[field] = value
            elif not _same_value(field, value, common_values[field]):
                errors.append(
                    f"{label}: {field}={value!r} differs from {common_values[field]!r}"
                )

        if shot == 0:
            if not _same_value(
                "metadata_samples_per_class",
                config.get("metadata_samples_per_class"),
                config.get("zero_shot_metadata_per_class"),
            ):
                errors.append(
                    f"{label}: metadata_samples_per_class must equal "
                    "zero_shot_metadata_per_class"
                )
            for field in (
                "metadata_samples_per_class",
                "metadata_sampling_mode",
                "full_train_audit",
            ):
                if field not in config:
                    errors.append(f"{label}: missing {field}")
                    continue
                value = config[field]
                if field not in zero_shot_values:
                    zero_shot_values[field] = value
                elif not _same_value(field, value, zero_shot_values[field]):
                    errors.append(
                        f"{label}: {field}={value!r} differs from "
                        f"{zero_shot_values[field]!r}"
                    )

        if class_name not in resources:
            continue
        for field in ("ckpt_path", "freq_stats_path"):
            if field not in config or not config[field]:
                errors.append(f"{label}: missing {field}")
                continue
            path = normalize_path(config[field])
            resources[class_name][field].add(path)
            if not _path_has_class_marker(path, class_name):
                errors.append(
                    f"{label}: {field} does not contain class marker exclude_{class_name}: {path}"
                )

    if common_values.get("max_eval_query_per_class") != 0:
        errors.append("max_eval_query_per_class must be 0 for formal aggregation")
    if common_values.get("strict_formal_eval_images") is not True:
        errors.append("strict_formal_eval_images must be true for formal aggregation")
    try:
        if int(common_values.get("zero_shot_metadata_per_class", 0)) <= 0:
            errors.append(
                "zero_shot_metadata_per_class must be a positive consistent value"
            )
    except (TypeError, ValueError):
        errors.append("zero_shot_metadata_per_class must be an integer")
    if not common_values.get("git_commit"):
        errors.append("git_commit must be non-empty and consistent")
    if zero_shot_values:
        if zero_shot_values.get("metadata_sampling_mode") != (
            "deterministic_lazy_strict_until_full"
        ):
            errors.append(
                "metadata_sampling_mode must be "
                "deterministic_lazy_strict_until_full for zero-shot aggregation"
            )
        if zero_shot_values.get("full_train_audit") is not False:
            errors.append("full_train_audit must be false for zero-shot aggregation")
        try:
            if int(zero_shot_values.get("metadata_samples_per_class", 0)) != 1024:
                errors.append(
                    "metadata_samples_per_class must be 1024 for formal zero-shot aggregation"
                )
        except (TypeError, ValueError):
            errors.append("metadata_samples_per_class must be an integer")

    for field in ("ckpt_path", "freq_stats_path"):
        owner_by_path = {}
        for class_name in classes:
            values = resources[class_name][field]
            if len(values) != 1:
                errors.append(
                    f"exclude_{class_name}: expected one consistent {field}, got {sorted(values)}"
                )
            for value in values:
                previous = owner_by_path.get(value)
                if previous is not None and previous != class_name:
                    errors.append(
                        f"{field} is shared by exclude_{previous} and exclude_{class_name}: {value}"
                    )
                owner_by_path[value] = class_name
    return errors
