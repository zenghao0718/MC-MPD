"""Dependency-free protocol and validation helpers for DDFSD multi-shot evaluation."""

import random
from typing import Dict, List, Sequence, Tuple


def parse_shot_list(value: str) -> List[int]:
    try:
        shots = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"Invalid --shot_list {value!r}; expected comma-separated integers.") from exc
    if not shots:
        raise ValueError("--shot_list must contain at least one shot.")
    if shots[0] < 0:
        raise ValueError("Shots must be non-negative.")
    return shots


def build_multishot_support_query_indices(
    dataset_size: int,
    max_shot: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Shuffle once, reserving max_shot candidates and a fixed query suffix."""
    if max_shot < 0:
        raise ValueError("max_shot must be non-negative.")
    if dataset_size <= max_shot:
        raise ValueError(
            f"Need more than max_shot={max_shot} val images to retain a query, got {dataset_size}."
        )
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    return indices[:max_shot], indices[max_shot:]


def nested_support_indices(candidates: Sequence[int], shot: int) -> List[int]:
    if shot <= 0:
        raise ValueError("Nested support requires a positive shot.")
    if shot > len(candidates):
        raise ValueError(f"shot={shot} exceeds {len(candidates)} support candidates.")
    return list(candidates[:shot])


def sample_metadata_indices(dataset_size: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        raise ValueError("zero_shot_metadata_per_class must be positive.")
    if dataset_size < count:
        raise ValueError(f"Metadata class has {dataset_size} images, fewer than requested {count}.")
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def select_real_and_nearest_fake_logits(logits):
    """Return real logits and the nearest-fake (largest negative-distance) logits."""
    if hasattr(logits, "ndim"):
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("Metadata logits must have shape [N, 1 + num_fake].")
        return logits[:, 0], logits[:, 1:].max(dim=1).values
    rows = [list(row) for row in logits]
    if not rows or any(len(row) < 2 for row in rows):
        raise ValueError("Metadata logits must contain real and at least one fake logit.")
    return [row[0] for row in rows], [max(row[1:]) for row in rows]


def resolve_checkpoint_step(requested_step: int, checkpoint_step: int, checkpoint_path: str) -> int:
    requested_step = int(requested_step or 0)
    checkpoint_step = int(checkpoint_step or 0)
    if requested_step < 0 or checkpoint_step < 0:
        raise ValueError("Checkpoint steps must be non-negative.")
    if requested_step > 0 and checkpoint_step > 0 and requested_step != checkpoint_step:
        raise ValueError(
            f"Requested step {requested_step} conflicts with checkpoint-recorded step "
            f"{checkpoint_step}: {checkpoint_path}"
        )
    return requested_step or checkpoint_step


CONSISTENT_CONFIG_FIELDS = (
    "shot_list", "seeds", "zero_shot_metadata_per_class", "max_shot",
    "max_eval_query_per_class", "ckpt_step", "checkpoint_model_mode",
    "model_mode", "branch_mode", "tau", "tau_r",
)


def validate_config_consistency(configs: Dict[str, dict]) -> None:
    if not configs:
        raise ValueError("No multi-shot configs were provided.")
    missing = []
    for class_name, config in configs.items():
        for field in CONSISTENT_CONFIG_FIELDS:
            if field not in config:
                missing.append(f"{class_name}.{field}")
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    mismatches = []
    for field in CONSISTENT_CONFIG_FIELDS:
        values = {name: config[field] for name, config in configs.items()}
        first = next(iter(values.values()))
        if any(value != first for value in values.values()):
            rendered = ", ".join(f"{name}={value!r}" for name, value in values.items())
            mismatches.append(f"{field}: {rendered}")
    if mismatches:
        raise ValueError("Inconsistent multi-shot configs:\n" + "\n".join(mismatches))
