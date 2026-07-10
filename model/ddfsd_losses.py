"""DDFSD prototype losses and distance utilities."""

import random
from typing import Dict

import torch
import torch.nn.functional as F


VALID_BRANCH_MODES = {"dual", "rgb-only", "freq-only"}


def ordinary_euclidean_distance(query: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """Pairwise ordinary Euclidean distance on the last dimension."""

    diff = query[:, :, None, :] - proto[:, None, :, :]
    return torch.sqrt(diff.pow(2).sum(dim=-1).clamp_min(1e-12))


def reshape_flat_embeddings(
    embeddings: torch.Tensor,
    episode_batch_size: int,
    num_classes: int,
    samples_per_class: int,
) -> torch.Tensor:
    """Convert class-major flat embeddings to [B, T, C, D]."""

    return (
        embeddings.float()
        .reshape(num_classes, episode_batch_size, samples_per_class, -1)
        .permute(1, 2, 0, 3)
        .contiguous()
    )


def make_query_labels(
    episode_batch_size: int,
    num_classes: int,
    num_query: int,
    device: torch.device,
) -> torch.Tensor:
    labels_one_episode = torch.arange(num_classes, device=device).repeat(num_query)
    return labels_one_episode.unsqueeze(0).expand(episode_batch_size, -1).reshape(-1)


def compute_prototypes(support_rgb: torch.Tensor, support_freq: torch.Tensor):
    support_rgb = support_rgb.float()
    support_freq = support_freq.float()
    proto_rgb = F.normalize(support_rgb.mean(dim=1), p=2, dim=-1)
    proto_freq = F.normalize(support_freq.mean(dim=1), p=2, dim=-1)
    return proto_rgb, proto_freq


def compute_support_sigmas(
    support_rgb: torch.Tensor,
    support_freq: torch.Tensor,
    proto_rgb: torch.Tensor,
    proto_freq: torch.Tensor,
):
    support_rgb = support_rgb.float()
    support_freq = support_freq.float()
    proto_rgb = proto_rgb.float()
    proto_freq = proto_freq.float()

    sigma_rgb = (support_rgb - proto_rgb[:, None, :, :]).pow(2).sum(dim=-1).mean(dim=1)
    sigma_freq = (support_freq - proto_freq[:, None, :, :]).pow(2).sum(dim=-1).mean(dim=1)
    return sigma_rgb, sigma_freq


def compute_alpha(sigma_rgb: torch.Tensor, sigma_freq: torch.Tensor, tau_r: float) -> torch.Tensor:
    sigma_rgb = sigma_rgb.float()
    sigma_freq = sigma_freq.float()
    alpha = torch.sigmoid((sigma_freq - sigma_rgb) / tau_r)
    return alpha.clamp(0.25, 0.75).detach()


def fuse_distances(
    rgb_dist: torch.Tensor,
    freq_dist: torch.Tensor,
    alpha: torch.Tensor,
    branch_mode: str,
) -> torch.Tensor:
    if branch_mode not in VALID_BRANCH_MODES:
        raise ValueError(f"Unknown branch mode '{branch_mode}', expected {VALID_BRANCH_MODES}")
    if branch_mode == "rgb-only":
        return rgb_dist
    if branch_mode == "freq-only":
        return freq_dist
    alpha_for_dist = alpha[:, None, :]
    return alpha_for_dist * rgb_dist + (1.0 - alpha_for_dist) * freq_dist


def compute_query_logits(
    query_rgb: torch.Tensor,
    query_freq: torch.Tensor,
    proto_rgb: torch.Tensor,
    proto_freq: torch.Tensor,
    alpha: torch.Tensor,
    tau: float,
    branch_mode: str = "dual",
) -> Dict[str, torch.Tensor]:
    query_rgb = query_rgb.float()
    query_freq = query_freq.float()
    proto_rgb = proto_rgb.float()
    proto_freq = proto_freq.float()
    rgb_dist = ordinary_euclidean_distance(query_rgb, proto_rgb)
    freq_dist = ordinary_euclidean_distance(query_freq, proto_freq)
    fused_dist = fuse_distances(rgb_dist, freq_dist, alpha, branch_mode)
    logits = -fused_dist / tau
    return {
        "logits": logits,
        "rgb_dist": rgb_dist,
        "freq_dist": freq_dist,
        "fused_dist": fused_dist,
    }


def compute_separation_loss(
    proto_rgb: torch.Tensor,
    proto_freq: torch.Tensor,
    alpha: torch.Tensor,
    m_rf: float,
    m_ff: float,
    lambda_ff: float,
) -> Dict[str, torch.Tensor]:
    proto_rgb = proto_rgb.float()
    proto_freq = proto_freq.float()
    alpha = alpha.float().detach()

    batch_size, num_classes, _ = proto_rgb.shape
    device = proto_rgb.device
    loss_rf = torch.zeros(batch_size, dtype=torch.float32, device=device)
    loss_ff = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # Diagnostics-only accumulators (detached immediately; never enter the
    # autograd graph and never influence loss_rf/loss_ff/loss_sep below).
    proto_rf_rgb_list = []
    proto_rf_freq_list = []
    proto_rf_fused_list = []
    alpha_rf_pair_list = []

    for fake_idx in range(1, num_classes):
        rgb_dist = torch.sqrt((proto_rgb[:, 0] - proto_rgb[:, fake_idx]).pow(2).sum(dim=-1).clamp_min(1e-12))
        freq_dist = torch.sqrt((proto_freq[:, 0] - proto_freq[:, fake_idx]).pow(2).sum(dim=-1).clamp_min(1e-12))
        alpha_ij = 0.5 * (alpha[:, 0] + alpha[:, fake_idx])
        dist = alpha_ij * rgb_dist + (1.0 - alpha_ij) * freq_dist
        loss_rf = loss_rf + F.relu(m_rf - dist).pow(2)

        proto_rf_rgb_list.append(rgb_dist.detach())
        proto_rf_freq_list.append(freq_dist.detach())
        proto_rf_fused_list.append(dist.detach())
        alpha_rf_pair_list.append(alpha_ij.detach())

    proto_ff_rgb_list = []
    proto_ff_freq_list = []
    proto_ff_fused_list = []
    alpha_ff_pair_list = []

    for first_idx in range(1, num_classes):
        for second_idx in range(first_idx + 1, num_classes):
            rgb_dist = torch.sqrt(
                (proto_rgb[:, first_idx] - proto_rgb[:, second_idx]).pow(2).sum(dim=-1).clamp_min(1e-12)
            )
            freq_dist = torch.sqrt(
                (proto_freq[:, first_idx] - proto_freq[:, second_idx]).pow(2).sum(dim=-1).clamp_min(1e-12)
            )
            alpha_ij = 0.5 * (alpha[:, first_idx] + alpha[:, second_idx])
            dist = alpha_ij * rgb_dist + (1.0 - alpha_ij) * freq_dist
            loss_ff = loss_ff + F.relu(m_ff - dist).pow(2)

            proto_ff_rgb_list.append(rgb_dist.detach())
            proto_ff_freq_list.append(freq_dist.detach())
            proto_ff_fused_list.append(dist.detach())
            alpha_ff_pair_list.append(alpha_ij.detach())

    loss_rf_mean = loss_rf.mean()
    loss_ff_mean = loss_ff.mean()
    loss_sep = (loss_rf + lambda_ff * loss_ff).mean()

    def _stack_or_empty(tensor_list):
        if tensor_list:
            return torch.stack(tensor_list, dim=1)
        return torch.zeros(batch_size, 0, dtype=torch.float32, device=device)

    return {
        "loss_sep": loss_sep,
        "loss_rf": loss_rf_mean,
        "loss_ff": loss_ff_mean,
        # Prototype-to-prototype distances used by the margin loss above,
        # shape [batch_size, num_pairs]. num_pairs is 2 for RF and 1 for FF
        # when num_classes == 3 (Real, Fake-A, Fake-B).
        "proto_rf_rgb_dist": _stack_or_empty(proto_rf_rgb_list),
        "proto_rf_freq_dist": _stack_or_empty(proto_rf_freq_list),
        "proto_rf_fused_dist": _stack_or_empty(proto_rf_fused_list),
        "alpha_rf_pair": _stack_or_empty(alpha_rf_pair_list),
        "proto_ff_rgb_dist": _stack_or_empty(proto_ff_rgb_list),
        "proto_ff_freq_dist": _stack_or_empty(proto_ff_freq_list),
        "proto_ff_fused_dist": _stack_or_empty(proto_ff_fused_list),
        "alpha_ff_pair": _stack_or_empty(alpha_ff_pair_list),
    }


def compute_ddfsd_episode_loss(
    z_rgb_flat: torch.Tensor,
    z_freq_flat: torch.Tensor,
    episode_batch_size: int,
    num_classes: int,
    num_support: int,
    num_query: int,
    tau: float,
    tau_r: float,
    m_rf: float,
    m_ff: float,
    lambda_ff: float,
    lambda_sep_current: float,
    branch_mode: str = "dual",
) -> Dict[str, torch.Tensor]:
    samples_per_class = num_support + num_query
    z_rgb = reshape_flat_embeddings(z_rgb_flat, episode_batch_size, num_classes, samples_per_class)
    z_freq = reshape_flat_embeddings(z_freq_flat, episode_batch_size, num_classes, samples_per_class)

    support_rgb = z_rgb[:, :num_support, :, :]
    support_freq = z_freq[:, :num_support, :, :]
    query_rgb = z_rgb[:, num_support:, :, :].reshape(episode_batch_size, num_query * num_classes, -1)
    query_freq = z_freq[:, num_support:, :, :].reshape(episode_batch_size, num_query * num_classes, -1)

    proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq)
    sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
    alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)

    query_out = compute_query_logits(
        query_rgb=query_rgb,
        query_freq=query_freq,
        proto_rgb=proto_rgb,
        proto_freq=proto_freq,
        alpha=alpha,
        tau=tau,
        branch_mode=branch_mode,
    )
    labels = make_query_labels(episode_batch_size, num_classes, num_query, z_rgb_flat.device)
    logits = query_out["logits"].reshape(-1, num_classes)
    loss_dual = F.cross_entropy(logits, labels)

    sep_out = compute_separation_loss(
        proto_rgb=proto_rgb,
        proto_freq=proto_freq,
        alpha=alpha,
        m_rf=m_rf,
        m_ff=m_ff,
        lambda_ff=lambda_ff,
    )
    loss_total = loss_dual + float(lambda_sep_current) * sep_out["loss_sep"]

    return {
        "loss_total": loss_total.float(),
        "loss_dual": loss_dual.float(),
        "loss_sep": sep_out["loss_sep"].float(),
        "loss_rf": sep_out["loss_rf"].float(),
        "loss_ff": sep_out["loss_ff"].float(),
        "alpha": alpha,
        "labels": labels,
        "logits": logits,
        "rgb_dist": query_out["rgb_dist"],
        "freq_dist": query_out["freq_dist"],
        "fused_dist": query_out["fused_dist"],
        "proto_rgb": proto_rgb,
        "proto_freq": proto_freq,
        # Diagnostics-only prototype-pair distances/weights (all detached in
        # compute_separation_loss; do not participate in backward and do not
        # change loss_total/loss_dual/loss_sep/loss_rf/loss_ff above).
        "proto_rf_rgb_dist": sep_out["proto_rf_rgb_dist"],
        "proto_rf_freq_dist": sep_out["proto_rf_freq_dist"],
        "proto_rf_fused_dist": sep_out["proto_rf_fused_dist"],
        "alpha_rf_pair": sep_out["alpha_rf_pair"],
        "proto_ff_rgb_dist": sep_out["proto_ff_rgb_dist"],
        "proto_ff_freq_dist": sep_out["proto_ff_freq_dist"],
        "proto_ff_fused_dist": sep_out["proto_ff_fused_dist"],
        "alpha_ff_pair": sep_out["alpha_ff_pair"],
    }


def compute_lambda_sep(step: int, target: float, warmup_start: int, warmup_end: int) -> float:
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return float(target)
    if warmup_end <= warmup_start:
        return float(target)
    progress = (step - warmup_start) / float(warmup_end - warmup_start)
    return float(target) * progress


def sample_branch_mode(
    dual_prob: float,
    rgb_prob: float,
    freq_prob: float,
    rng=random,
) -> str:
    total = dual_prob + rgb_prob + freq_prob
    if total <= 0:
        raise ValueError("Branch dropout probabilities must sum to a positive value.")
    draw = rng.random() * total
    if draw < dual_prob:
        return "dual"
    if draw < dual_prob + rgb_prob:
        return "rgb-only"
    return "freq-only"
