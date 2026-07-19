"""Configuration validation for formal DDFSD main-protocol shot ablations."""

import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence


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
            "No default ADM parity reference CSV exists. Checked:\n- " + listed
        )
    raise ValueError(
        "Ambiguous ADM parity reference CSV: multiple default candidates exist. "
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
