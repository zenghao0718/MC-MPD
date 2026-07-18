"""Zero-shot embedding evaluation for the formal DDFSD main protocol."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from datasets.ddfsd_datasets import make_subset_loader
from model.ddfsd_losses import (
    compute_alpha,
    compute_prototypes,
    compute_query_logits,
    compute_support_sigmas,
)
from util.ddfsd_eval import binary_metrics, encode_batch


@dataclass
class EmbeddingCache:
    dataset_indices: List[int]
    rgb: Optional[torch.Tensor]
    freq: Optional[torch.Tensor]


@torch.no_grad()
def encode_dataset_indices(
    model, dataset, indices, batch_size, num_workers, device, use_fp16
):
    """Encode requested dataset indices in the exact supplied order."""

    ordered = list(indices)
    loader = make_subset_loader(
        dataset,
        ordered,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    rgb_parts, freq_parts = [], []
    model.eval()
    for images, _ in loader:
        rgb, freq = encode_batch(model, images, device, use_fp16)
        if rgb is not None:
            rgb_parts.append(rgb.cpu())
        if freq is not None:
            freq_parts.append(freq.cpu())
    return EmbeddingCache(
        dataset_indices=ordered,
        rgb=torch.cat(rgb_parts) if rgb_parts else None,
        freq=torch.cat(freq_parts) if freq_parts else None,
    )


def _select(cache: EmbeddingCache, positions: Sequence[int]):
    index = torch.as_tensor(positions, dtype=torch.long)
    return (
        cache.rgb.index_select(0, index) if cache.rgb is not None else None,
        cache.freq.index_select(0, index) if cache.freq is not None else None,
    )


def _binary_metrics_from_logits(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    logits = torch.cat([real_logits, fake_logits], dim=0).cpu()
    labels = [0] * len(real_logits) + [1] * len(fake_logits)
    probabilities = logits.softmax(dim=-1)
    return binary_metrics(
        labels,
        probabilities[:, 1].tolist(),
        probabilities.argmax(dim=-1).tolist(),
    )


def _metadata_prototypes(caches, selections, device, model_mode, tau_r):
    rgb_means, freq_means = [], []
    for name in caches:
        rgb, freq = _select(caches[name], selections[name])
        if rgb is not None:
            rgb_means.append(rgb.mean(0))
        if freq is not None:
            freq_means.append(freq.mean(0))
    rgb = (
        torch.stack(rgb_means).unsqueeze(0).unsqueeze(1).to(device)
        if rgb_means
        else None
    )
    freq = (
        torch.stack(freq_means).unsqueeze(0).unsqueeze(1).to(device)
        if freq_means
        else None
    )
    proto_rgb, proto_freq = compute_prototypes(rgb, freq, model_mode=model_mode)
    alpha = None
    if model_mode == "dual":
        rgb_full = (
            torch.stack(
                [_select(caches[name], selections[name])[0] for name in caches], dim=1
            )
            .unsqueeze(0)
            .to(device)
        )
        freq_full = (
            torch.stack(
                [_select(caches[name], selections[name])[1] for name in caches], dim=1
            )
            .unsqueeze(0)
            .to(device)
        )
        sigma_rgb, sigma_freq = compute_support_sigmas(
            rgb_full, freq_full, proto_rgb, proto_freq
        )
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)
    return proto_rgb, proto_freq, alpha


def _real_and_nearest_fake_logits(logits: torch.Tensor):
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            "Metadata logits must contain real and at least one visible fake class."
        )
    return logits[:, 0], logits[:, 1:].max(dim=1).values


def evaluate_zero_shot_embeddings(
    real_cache,
    fake_cache,
    real_query_positions,
    fake_query_positions,
    metadata_caches,
    metadata_selections,
    device,
    model_mode,
    branch_mode,
    tau,
    tau_r,
) -> Dict[str, float]:
    """Classify all supplied val queries with visible-train metadata prototypes."""

    proto_rgb, proto_freq, alpha = _metadata_prototypes(
        metadata_caches, metadata_selections, device, model_mode, tau_r
    )

    def binary_logits(cache, positions):
        rgb, freq = _select(cache, positions)
        logits = compute_query_logits(
            query_rgb=rgb.to(device).unsqueeze(0) if rgb is not None else None,
            query_freq=freq.to(device).unsqueeze(0) if freq is not None else None,
            proto_rgb=proto_rgb,
            proto_freq=proto_freq,
            alpha=alpha,
            tau=tau,
            branch_mode=branch_mode,
            model_mode=model_mode,
        )["logits"].squeeze(0)
        real_logits, fake_logits = _real_and_nearest_fake_logits(logits)
        return torch.stack([real_logits, fake_logits], dim=-1)

    metrics = _binary_metrics_from_logits(
        binary_logits(real_cache, real_query_positions),
        binary_logits(fake_cache, fake_query_positions),
    )
    values = alpha.detach().cpu().flatten() if alpha is not None else None
    metrics.update(
        {
            "alpha_mean": float(values.mean()) if values is not None else "",
            "alpha_min": float(values.min()) if values is not None else "",
            "alpha_max": float(values.max()) if values is not None else "",
        }
    )
    return metrics
