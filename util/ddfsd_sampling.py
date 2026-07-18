"""Dependency-free sampling shared by DDFSD main-protocol evaluation."""

import random
from typing import List, Optional, Tuple


def sample_support_query_indices(
    dataset_size: int,
    support_shot: int,
    seed: int,
    max_query: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    if dataset_size <= support_shot:
        raise ValueError(
            f"Need more than {support_shot} images to build support/query, got {dataset_size}."
        )

    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    support = indices[:support_shot]
    query = indices[support_shot:]
    if max_query is not None and max_query > 0:
        query = query[:max_query]
    return support, query
