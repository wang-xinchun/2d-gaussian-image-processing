from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAffineModulator(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.scale_basis = nn.Parameter(torch.zeros(rank, channels))
        self.bias_basis = nn.Parameter(torch.zeros(rank, channels))
        nn.init.normal_(self.scale_basis, std=0.02)
        nn.init.normal_(self.bias_basis, std=0.02)

    def forward(self, features: torch.Tensor, coeffs_lowres: torch.Tensor) -> torch.Tensor:
        coeffs = F.interpolate(
            coeffs_lowres,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        scale = torch.einsum("brhw,rc->bchw", coeffs, self.scale_basis)
        bias = torch.einsum("brhw,rc->bchw", coeffs, self.bias_basis)
        return features * (1.0 + scale) + bias
