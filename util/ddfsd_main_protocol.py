"""Dependency-light helpers for the formal DDFSD main-protocol shot ablation."""

import random
from typing import Dict, List, MutableSequence, Optional, Sequence, Tuple


FAKE_CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
FORMAL_SHOTS = (0, 5, 10, 20, 30, 50)
FORMAL_SEEDS = (42, 101, 102, 103, 104)


def zero_shot_metadata_classes(exclude_class: str) -> List[str]:
    if exclude_class not in FAKE_CLASSES:
        raise ValueError(
            f"Unknown held-out class {exclude_class!r}; expected one of {FAKE_CLASSES}."
        )
    return ["real"] + [name for name in FAKE_CLASSES if name != exclude_class]


def build_zero_shot_query_indices(
    real_dataset_size: int, fake_dataset_size: int
) -> Tuple[List[int], List[int]]:
    """Use every validation item independently; never balance the two classes."""

    if real_dataset_size <= 0 or fake_dataset_size <= 0:
        raise ValueError(
            "Zero-shot evaluation requires non-empty real and held-out fake val sets."
        )
    return list(range(real_dataset_size)), list(range(fake_dataset_size))


def build_valid_image_indices(
    paths: Sequence[str],
    invalid_records: Optional[MutableSequence[Dict[str, object]]] = None,
) -> List[int]:
    """Return deterministically ordered indices whose images fully decode as RGB.

    ``invalid_records`` is optional so the helper remains convenient in tests while
    production callers can retain a complete audit trail.
    """

    from PIL import Image

    valid = []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                decoded = image.convert("RGB")
                decoded.load()
        except Exception as exc:  # noqa: BLE001 - audit arbitrary decode failures
            if invalid_records is not None:
                invalid_records.append(
                    {
                        "dataset_index": index,
                        "filepath": path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue
        valid.append(index)
    return valid


def audit_image_paths(
    paths: Sequence[str], data_class: str, split: str
) -> List[Dict[str, object]]:
    invalid = []
    build_valid_image_indices(paths, invalid)
    return [{"data_class": data_class, "split": split, **record} for record in invalid]


def sample_metadata_from_valid_indices(
    valid_indices: Sequence[int], count: int, seed: int
) -> List[int]:
    if count <= 0:
        raise ValueError("zero_shot_metadata_per_class must be positive.")
    if len(valid_indices) < count:
        raise ValueError(
            f"Metadata class has only {len(valid_indices)} valid images, fewer than requested {count}."
        )
    selected = list(valid_indices)
    random.Random(seed).shuffle(selected)
    return selected[:count]
