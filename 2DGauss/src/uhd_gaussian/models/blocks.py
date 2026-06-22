from __future__ import annotations

import torch
import torch.nn as nn


def make_group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dw = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            groups=channels * 2,
        )
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + x


class StackedResidualBlocks(nn.Module):
    def __init__(self, channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualDWBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
