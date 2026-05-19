"""Dual-branch FSD model wrapper."""

import torch.nn as nn
import timm


class DualBranchFSD(nn.Module):
    """Two independent ResNet50 encoders for RGB and DWT-frequency inputs."""

    def __init__(
        self,
        rgb_backbone: str = "resnet50",
        freq_backbone: str = "resnet50",
        rgb_pretrained: bool = True,
        freq_pretrained: bool = True,
        embedding_dim: int = 1024,
    ):
        super().__init__()
        self.rgb_encoder = timm.create_model(
            rgb_backbone,
            pretrained=rgb_pretrained,
            num_classes=embedding_dim,
        )
        self.freq_encoder = timm.create_model(
            freq_backbone,
            pretrained=freq_pretrained,
            num_classes=embedding_dim,
        )

    def forward(self, x_rgb, x_freq):
        z_rgb = self.rgb_encoder(x_rgb)
        z_freq = self.freq_encoder(x_freq)
        return z_rgb, z_freq
