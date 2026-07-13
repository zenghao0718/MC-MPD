"""Dual-domain model for DDFSD."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from util.ddfsd_frequency import load_frequency_stats, normalize_frequency, raw_rgb_to_frequency


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


VALID_MODEL_MODES = {"dual", "rgb-only", "freq-only"}


class DDFSDDualDomainNet(nn.Module):
    """DDFSD encoder with dual, RGB-only, and frequency-only architectures."""

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 512,
        freq_stats_path: Optional[str] = None,
        model_mode: str = "dual",
    ):
        super().__init__()
        if model_mode not in VALID_MODEL_MODES:
            raise ValueError(f"Unknown model_mode '{model_mode}', expected one of {sorted(VALID_MODEL_MODES)}")
        self.model_mode = model_mode
        self.uses_rgb = model_mode in {"dual", "rgb-only"}
        self.uses_frequency = model_mode in {"dual", "freq-only"}

        if self.uses_rgb:
            self.rgb_backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            self.rgb_projector = ProjectionHead(2048, 1024, embedding_dim)
        if self.uses_frequency:
            self.freq_backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
            self.freq_projector = ProjectionHead(512, 512, embedding_dim)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        if self.uses_frequency:
            self.register_buffer("freq_mean", torch.zeros(3, dtype=torch.float32))
            self.register_buffer("freq_std", torch.ones(3, dtype=torch.float32))

        if freq_stats_path and self.uses_frequency:
            self.load_freq_stats(freq_stats_path)
        elif freq_stats_path:
            raise ValueError("freq_stats_path is not used by an rgb-only DDFSD model.")

    def load_freq_stats(self, path: str) -> None:
        stats = load_frequency_stats(path)
        self.set_freq_stats(stats["mean"], stats["std"])

    def set_freq_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.uses_frequency:
            raise RuntimeError("rgb-only DDFSD models do not have frequency statistics.")
        mean = torch.as_tensor(mean, dtype=torch.float32).view(3)
        std = torch.as_tensor(std, dtype=torch.float32).view(3)
        self.freq_mean.copy_(mean)
        self.freq_std.copy_(std)

    def forward(self, raw_rgb: torch.Tensor):
        if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3:
            raise ValueError(f"raw_rgb must have shape [B, 3, H, W], got {tuple(raw_rgb.shape)}")

        raw_rgb = raw_rgb.float()
        outputs = {}
        if self.uses_rgb:
            rgb_input = (raw_rgb - self.imagenet_mean) / self.imagenet_std
            rgb_feat = self.rgb_backbone(rgb_input)
            outputs["z_rgb"] = F.normalize(self.rgb_projector(rgb_feat).float(), p=2, dim=-1)

        if self.uses_frequency:
            freq = raw_rgb_to_frequency(raw_rgb)
            freq = normalize_frequency(freq, self.freq_mean, self.freq_std)
            freq_feat = self.freq_backbone(freq)
            outputs["z_freq"] = F.normalize(self.freq_projector(freq_feat).float(), p=2, dim=-1)

        return outputs
