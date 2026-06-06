"""FSD model wrapper for L2-normalized cosine prototypical classification."""

import math

import torch
import torch.nn as nn
import timm


class CosineMetricFSD(nn.Module):
    """Single-branch FSD model with a learnable positive cosine logit scale."""

    def __init__(
        self,
        pretrained=True,
        embedding_dim=1024,
        init_scale=10.0,
        max_scale=100.0,
    ):
        super().__init__()

        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}.")
        if max_scale <= 0:
            raise ValueError(f"max_scale must be positive, got {max_scale}.")
        if init_scale > max_scale:
            raise ValueError(
                f"init_scale should not be larger than max_scale, got "
                f"init_scale={init_scale}, max_scale={max_scale}."
            )

        self.encoder = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=embedding_dim,
        )
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale), dtype=torch.float32)
        )
        self.max_scale = float(max_scale)

    def forward(self, x):
        return self.encoder(x)

    def get_scale(self):
        scale = self.log_scale.exp()
        scale = torch.clamp(scale, max=self.max_scale)
        return scale
