"""Frequency-domain helpers for the DWT dual-branch experiment."""

import json
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def haar_dwt_highfreq_rgb_mean(
    x: torch.Tensor,
    use_abs: bool = True,
    use_log1p: bool = True,
    resize_to: Optional[Union[int, Tuple[int, int]]] = 224,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    input_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a 3-channel Haar-DWT high-frequency tensor from raw RGB input.

    Args:
        x: Raw RGB tensor after ToTensor(), shape [B, 3, H, W], values in [0, 1].
        use_abs: Whether to use high-frequency magnitudes.
        use_log1p: Whether to apply log(1 + x) compression after RGB averaging.
        resize_to: Output size. Use None when collecting pre-resize statistics.
        mean, std: Train-set DWT stats with shape [3] or [1, 3, 1, 1].
        input_scale: Optional scale before DWT. Default keeps baseline [0, 1] scale.
        eps: Numerical epsilon for mean/std normalization.

    Returns:
        Tensor with shape [B, 3, resize_to, resize_to] when resize_to is set.
    """

    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B, 3, H, W], got {tuple(x.shape)}.")
    if x.size(1) != 3:
        raise ValueError(f"Expected 3 RGB channels, got {x.size(1)}.")

    x = x.float()
    if input_scale != 1.0:
        x = x * input_scale

    height, width = x.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError(f"DWT input is too small: {height}x{width}.")
    if height % 2 != 0:
        x = x[..., :-1, :]
    if width % 2 != 0:
        x = x[..., :, :-1]

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    # One-level Haar high-frequency subbands. Names are kept consistent here.
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    if use_abs:
        lh = lh.abs()
        hl = hl.abs()
        hh = hh.abs()

    freq = torch.cat(
        [
            lh.mean(dim=1, keepdim=True),
            hl.mean(dim=1, keepdim=True),
            hh.mean(dim=1, keepdim=True),
        ],
        dim=1,
    )

    if use_log1p:
        freq = torch.log1p(freq)

    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("mean and std must be provided together for DWT normalization.")
        mean = _reshape_stats(mean, freq)
        std = _reshape_stats(std, freq)
        freq = (freq - mean) / (std + eps)

    if resize_to is not None:
        size = (resize_to, resize_to) if isinstance(resize_to, int) else resize_to
        freq = F.interpolate(freq, size=size, mode="bilinear", align_corners=False)

    return freq


def load_dwt_stats(stats_path: str, device=None, dtype=torch.float32):
    """Load DWT mean/std JSON and return tensors with shape [3]."""

    if not stats_path:
        raise ValueError(
            "freq_stats_path is required when use_dual_branch=True and freq_input_type='dwt'. "
            "Please run tools/compute_dwt_stats.py on the training split first."
        )
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"DWT stats file not found: {stats_path}. "
            "Please run tools/compute_dwt_stats.py on the training split first."
        )

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "mean" not in data or "std" not in data:
        raise ValueError(f"DWT stats file {stats_path} must contain 'mean' and 'std' fields.")
    if len(data["mean"]) != 3 or len(data["std"]) != 3:
        raise ValueError(f"DWT stats file {stats_path} must contain 3-channel mean/std values.")

    mean = torch.tensor(data["mean"], dtype=dtype, device=device)
    std = torch.tensor(data["std"], dtype=dtype, device=device)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"DWT stats file {stats_path} contains NaN or Inf.")
    if (std <= 0).any():
        raise ValueError(f"DWT stats file {stats_path} contains non-positive std values.")
    return mean, std


def _reshape_stats(stats: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(stats):
        stats = torch.tensor(stats)
    stats = stats.to(device=target.device, dtype=target.dtype)
    if stats.dim() == 1:
        if stats.numel() != 3:
            raise ValueError(f"Expected 3 stats values, got {stats.numel()}.")
        return stats.view(1, 3, 1, 1)
    if stats.shape == (1, 3, 1, 1):
        return stats
    raise ValueError(f"Expected stats shape [3] or [1, 3, 1, 1], got {tuple(stats.shape)}.")
