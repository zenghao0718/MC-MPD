"""Frequency-domain utilities for DDFSD.

The first DDFSD version uses signed Haar high-frequency coefficients from the
raw RGB tensor before ImageNet normalization.
"""

import os
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from datasets.ddfsd_datasets import (
    FAKE_CLASSES,
    ImagePathDataset,
    make_stats_transform,
)


def rgb_to_y(raw_rgb: torch.Tensor) -> torch.Tensor:
    """Convert raw RGB tensors in [0, 1] to a single Y channel."""

    if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3:
        raise ValueError(f"raw_rgb must have shape [B, 3, H, W], got {tuple(raw_rgb.shape)}")

    raw_rgb = raw_rgb.float()
    red = raw_rgb[:, 0:1, :, :]
    green = raw_rgb[:, 1:2, :, :]
    blue = raw_rgb[:, 2:3, :, :]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def haar_dwt_high_frequency(y: torch.Tensor) -> torch.Tensor:
    """Return signed LH/HL/HH Haar-DWT bands as [B, 3, H/2, W/2]."""

    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError(f"y must have shape [B, 1, H, W], got {tuple(y.shape)}")
    if y.shape[-2] % 2 != 0 or y.shape[-1] % 2 != 0:
        raise ValueError(f"Haar-DWT requires even spatial size, got {tuple(y.shape[-2:])}")

    x00 = y[:, :, 0::2, 0::2]
    x01 = y[:, :, 0::2, 1::2]
    x10 = y[:, :, 1::2, 0::2]
    x11 = y[:, :, 1::2, 1::2]

    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.cat([lh, hl, hh], dim=1)


def raw_rgb_to_frequency(raw_rgb: torch.Tensor) -> torch.Tensor:
    """Compute signed Y-channel Haar high-frequency maps from raw RGB."""

    return haar_dwt_high_frequency(rgb_to_y(raw_rgb))


def normalize_frequency(
    freq: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize frequency maps with [3] or broadcastable mean/std tensors."""

    mean = mean.to(device=freq.device, dtype=freq.dtype).view(1, 3, 1, 1)
    std = std.to(device=freq.device, dtype=freq.dtype).view(1, 3, 1, 1)
    return (freq - mean) / (std + eps)


def _validate_stats_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu().float()
    if tuple(value.shape) != (3,):
        raise ValueError(f"{name} must have shape [3], got {tuple(value.shape)}")
    return value


def save_frequency_stats(path: str, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Save DDFSD frequency statistics with mean/std shape exactly [3]."""

    stats = {
        "mean": _validate_stats_tensor(mean, "mean"),
        "std": _validate_stats_tensor(std, "std"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(stats, path)


def load_frequency_stats(path: str, map_location: Optional[str] = "cpu") -> Dict[str, torch.Tensor]:
    """Load and validate DDFSD frequency statistics."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Frequency stats not found: {path}")
    stats = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError(f"Invalid frequency stats file: {path}")
    return {
        "mean": _validate_stats_tensor(stats["mean"], "mean"),
        "std": _validate_stats_tensor(stats["std"], "std"),
    }


def compute_frequency_stats_from_loader(loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Compute channel-wise mean/std for signed LH/HL/HH maps."""

    total_sum = torch.zeros(3, dtype=torch.float64, device=device)
    total_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
    total_count = 0

    for raw_rgb, _ in loader:
        raw_rgb = raw_rgb.to(device=device, non_blocking=True)
        freq = raw_rgb_to_frequency(raw_rgb).double()
        total_sum += freq.sum(dim=(0, 2, 3))
        total_sq_sum += freq.pow(2).sum(dim=(0, 2, 3))
        total_count += freq.shape[0] * freq.shape[2] * freq.shape[3]

    if total_count == 0:
        raise ValueError("Cannot compute frequency stats from an empty loader.")

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - mean.pow(2)
    std = var.clamp_min(0.0).sqrt()
    return {"mean": mean.float().cpu(), "std": std.float().cpu()}


def compute_frequency_stats(
    data_root: str,
    exclude_class: str,
    output_path: str,
    batch_size: int = 128,
    num_workers: int = 8,
    device: Optional[torch.device] = None,
    fake_classes: Iterable[str] = FAKE_CLASSES,
) -> Dict[str, torch.Tensor]:
    """Compute and save train-split stats for real + non-excluded fake classes."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ["real"] + [name for name in fake_classes if name != exclude_class]
    transform = make_stats_transform()
    total_sum = torch.zeros(3, dtype=torch.float64, device=device)
    total_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
    total_count = 0

    for class_name in classes:
        class_root = os.path.join(data_root, class_name, "train")
        dataset = ImagePathDataset(class_root, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        stats = compute_frequency_stats_from_loader(loader, device=device)
        class_count = len(dataset) * 112 * 112
        total_sum += stats["mean"].to(device=device, dtype=torch.float64) * class_count
        total_sq_sum += (
            stats["std"].to(device=device, dtype=torch.float64).pow(2)
            + stats["mean"].to(device=device, dtype=torch.float64).pow(2)
        ) * class_count
        total_count += class_count

    if total_count == 0:
        raise ValueError("No images found while computing frequency stats.")

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - mean.pow(2)
    std = var.clamp_min(0.0).sqrt()
    save_frequency_stats(output_path, mean.float().cpu(), std.float().cpu())
    return load_frequency_stats(output_path)
