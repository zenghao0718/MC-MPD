"""Dependency-light helpers for the formal DDFSD main-protocol shot ablation."""

import hashlib
import os
import random
from typing import Callable, Dict, List, MutableSequence, Optional, Sequence, Tuple


FAKE_CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
FORMAL_SHOTS = (0, 5, 10, 20, 30, 50)
FORMAL_SEEDS = (42, 101, 102, 103, 104)
ZERO_SHOT_METADATA_SAMPLING_MODE = "deterministic_lazy_strict_until_full"


class InsufficientValidMetadataImages(ValueError):
    """Raised after a lazy strict scan exhausts all metadata candidates."""

    def __init__(self, message, invalid_records, visited_count):
        super().__init__(message)
        self.invalid_records = list(invalid_records)
        self.visited_count = int(visited_count)


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


def _normalized_sampling_root(data_root: str) -> str:
    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.expanduser(str(data_root))))
    )


def metadata_sampling_seed(
    data_root: str, held_out_class: str, metadata_class: str, seed: int
) -> int:
    """Derive a stable seed without relying on Python's randomized ``hash``."""

    components = (
        _normalized_sampling_root(data_root),
        str(held_out_class),
        str(metadata_class),
        str(int(seed)),
    )
    digest = hashlib.sha256("\0".join(components).encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def _strict_decode_rgb(path: str) -> None:
    from PIL import Image

    with Image.open(path) as image:
        decoded = image.convert("RGB")
        decoded.load()


def sample_metadata_paths_lazy_strict(
    paths: Sequence[str],
    count: int,
    data_root: str,
    held_out_class: str,
    metadata_class: str,
    seed: int,
    decode_image: Optional[Callable[[str], None]] = None,
) -> Tuple[List[int], List[Dict[str, object]], int]:
    """Strictly decode a seeded path permutation only until ``count`` are valid.

    The returned indices always address the caller's original ``paths`` sequence.
    Invalid records cover only candidates visited for this seed; unvisited train
    images are intentionally not audited.
    """

    if count <= 0:
        raise ValueError("zero_shot_metadata_per_class must be positive.")
    decode = decode_image or _strict_decode_rgb
    candidates = sorted(
        enumerate(paths),
        key=lambda item: os.path.normcase(os.path.abspath(str(item[1]))),
    )
    random.Random(
        metadata_sampling_seed(
            data_root=data_root,
            held_out_class=held_out_class,
            metadata_class=metadata_class,
            seed=seed,
        )
    ).shuffle(candidates)

    selected = []
    invalid = []
    visited_count = 0
    for index, path in candidates:
        visited_count += 1
        try:
            decode(path)
        except Exception as exc:  # noqa: BLE001 - retain arbitrary decode errors
            invalid.append(
                {
                    "dataset_index": index,
                    "filepath": path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        selected.append(index)
        if len(selected) == count:
            return selected, invalid, visited_count

    raise InsufficientValidMetadataImages(
        f"Metadata class {metadata_class!r} for held-out {held_out_class!r} "
        f"and seed {seed} has only {len(selected)} strictly decodable images, "
        f"fewer than requested {count}; visited all {visited_count} candidates.",
        invalid_records=invalid,
        visited_count=visited_count,
    )


def zero_shot_metadata_manifest_rows(
    paths: Sequence[str],
    selected_indices: Sequence[int],
    held_out_class: str,
    metadata_class: str,
    seed: int,
) -> List[Dict[str, object]]:
    """Build manifest rows directly from the strict sampled dataset indices."""

    rows = []
    for rank, index in enumerate(selected_indices, 1):
        path = paths[index]
        rows.append(
            {
                "held_out_class": held_out_class,
                "exclude_class": held_out_class,
                "seed": seed,
                "metadata_class": metadata_class,
                "sample_rank": rank,
                "split": "train",
                "dataset_index": index,
                "image_path": path,
                "filepath": path,
                "metadata_rank": rank,
            }
        )
    return rows
