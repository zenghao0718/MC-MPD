"""Deterministic, cached 0/multi-shot evaluation helpers for DDFSD."""

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
from util.ddfsd_multishot_logic import (
    build_multishot_support_query_indices,
    nested_support_indices,
    parse_shot_list,
    sample_metadata_indices,
    select_real_and_nearest_fake_logits,
)


@dataclass
class EmbeddingCache:
    paths: List[str]
    rgb: Optional[torch.Tensor]
    freq: Optional[torch.Tensor]


@torch.no_grad()
def encode_dataset_indices(model, dataset, indices, batch_size, num_workers, device, use_fp16):
    """Encode each requested dataset index once and retain ordered CPU tensors."""
    ordered = list(indices)
    loader = make_subset_loader(
        dataset, ordered, batch_size=batch_size, num_workers=num_workers,
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
        paths=[dataset.paths[index] for index in ordered],
        rgb=torch.cat(rgb_parts) if rgb_parts else None,
        freq=torch.cat(freq_parts) if freq_parts else None,
    )


def _select(cache: EmbeddingCache, indices: Sequence[int]):
    idx = torch.as_tensor(indices, dtype=torch.long)
    return (
        cache.rgb.index_select(0, idx) if cache.rgb is not None else None,
        cache.freq.index_select(0, idx) if cache.freq is not None else None,
    )


def _support_tensor(real, fake, shot, branch):
    if real is None or fake is None:
        return None
    flat = torch.cat([real[:shot], fake[:shot]], dim=0)
    return flat.reshape(2, shot, -1).permute(1, 0, 2).unsqueeze(0).to(branch)


def _metrics_from_logits(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    logits = torch.cat([real_logits, fake_logits], dim=0).cpu()
    labels = [0] * len(real_logits) + [1] * len(fake_logits)
    probs = logits.softmax(dim=-1)
    return binary_metrics(labels, probs[:, 1].tolist(), probs.argmax(dim=-1).tolist())


def evaluate_few_shot_embeddings(
    real_cache: EmbeddingCache,
    fake_cache: EmbeddingCache,
    real_support: Sequence[int],
    fake_support: Sequence[int],
    real_query: Sequence[int],
    fake_query: Sequence[int],
    shot: int,
    device: torch.device,
    model_mode: str,
    branch_mode: str,
    tau: float,
    tau_r: float,
) -> Dict[str, float]:
    if shot <= 0:
        raise ValueError("Few-shot embedding evaluation requires shot > 0.")
    rs_rgb, rs_freq = _select(real_cache, nested_support_indices(real_support, shot))
    fs_rgb, fs_freq = _select(fake_cache, nested_support_indices(fake_support, shot))
    support_rgb = _support_tensor(rs_rgb, fs_rgb, shot, device)
    support_freq = _support_tensor(rs_freq, fs_freq, shot, device)
    proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode=model_mode)
    alpha = None
    if model_mode == "dual":
        sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)

    def logits(cache, query):
        rgb, freq = _select(cache, query)
        out = compute_query_logits(
            query_rgb=rgb.to(device).unsqueeze(0) if rgb is not None else None,
            query_freq=freq.to(device).unsqueeze(0) if freq is not None else None,
            proto_rgb=proto_rgb, proto_freq=proto_freq, alpha=alpha, tau=tau,
            branch_mode=branch_mode, model_mode=model_mode,
        )
        return out["logits"].squeeze(0)

    metrics = _metrics_from_logits(logits(real_cache, real_query), logits(fake_cache, fake_query))
    alpha_values = alpha.detach().cpu().flatten() if alpha is not None else None
    metrics.update({
        "alpha_mean": float(alpha_values.mean()) if alpha_values is not None else "",
        "alpha_min": float(alpha_values.min()) if alpha_values is not None else "",
        "alpha_max": float(alpha_values.max()) if alpha_values is not None else "",
    })
    return metrics


def build_metadata_prototypes(caches, selections, device, model_mode, tau_r):
    rgb_support, freq_support = [], []
    for name in caches:
        rgb, freq = _select(caches[name], selections[name])
        if rgb is not None:
            rgb_support.append(rgb.mean(0))
        if freq is not None:
            freq_support.append(freq.mean(0))
    rgb = torch.stack(rgb_support).unsqueeze(0).unsqueeze(1).to(device) if rgb_support else None
    freq = torch.stack(freq_support).unsqueeze(0).unsqueeze(1).to(device) if freq_support else None
    # compute_prototypes normalizes the class means, matching ordinary support inference.
    proto_rgb, proto_freq = compute_prototypes(rgb, freq, model_mode=model_mode)
    alpha = None
    if model_mode == "dual":
        # Sigmas must use every selected metadata embedding, not the class means above.
        rgb_full = torch.stack([_select(caches[n], selections[n])[0] for n in caches], dim=1).unsqueeze(0).to(device)
        freq_full = torch.stack([_select(caches[n], selections[n])[1] for n in caches], dim=1).unsqueeze(0).to(device)
        sigma_rgb, sigma_freq = compute_support_sigmas(rgb_full, freq_full, proto_rgb, proto_freq)
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)
    return proto_rgb, proto_freq, alpha


def evaluate_zero_shot_embeddings(
    real_cache, fake_cache, real_query, fake_query, metadata_caches, metadata_selections,
    device, model_mode, branch_mode, tau, tau_r,
):
    proto_rgb, proto_freq, alpha = build_metadata_prototypes(
        metadata_caches, metadata_selections, device, model_mode, tau_r
    )

    def binary_logits(cache, query):
        rgb, freq = _select(cache, query)
        logits = compute_query_logits(
            query_rgb=rgb.to(device).unsqueeze(0) if rgb is not None else None,
            query_freq=freq.to(device).unsqueeze(0) if freq is not None else None,
            proto_rgb=proto_rgb, proto_freq=proto_freq, alpha=alpha, tau=tau,
            branch_mode=branch_mode, model_mode=model_mode,
        )["logits"].squeeze(0)
        real_logits, fake_logits = select_real_and_nearest_fake_logits(logits)
        return torch.stack([real_logits, fake_logits], dim=-1)

    metrics = _metrics_from_logits(binary_logits(real_cache, real_query), binary_logits(fake_cache, fake_query))
    values = alpha.detach().cpu().flatten() if alpha is not None else None
    metrics.update({
        "alpha_mean": float(values.mean()) if values is not None else "",
        "alpha_min": float(values.min()) if values is not None else "",
        "alpha_max": float(values.max()) if values is not None else "",
    })
    return metrics
