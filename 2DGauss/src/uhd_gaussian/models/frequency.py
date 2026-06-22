from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class ExplicitFrequencySplitter(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        highres_high_band_strength: float = 0.50,
        gaussian_feedback_strength: float = 0.10,
        gaussian_feedback_attention_source: str = "main",
        detail_blocks: int = 1,
        use_structure_hint: bool = False,
        structure_hint_strength: float = 0.25,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if gaussian_feedback_attention_source not in {"main", "control"}:
            raise ValueError("gaussian_feedback_attention_source must be main or control")
        self.kernel_size = kernel_size
        self.highres_high_band_strength = highres_high_band_strength
        self.gaussian_feedback_strength = gaussian_feedback_strength
        self.gaussian_feedback_attention_source = gaussian_feedback_attention_source
        self.use_structure_hint = use_structure_hint
        self.structure_hint_strength = structure_hint_strength
        self.high_band_proj = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[ResidualDWBlock(channels) for _ in range(detail_blocks)],
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def split(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.kernel_size // 2
        padded = F.pad(rgb, (pad, pad, pad, pad), mode="reflect")
        low_band = F.avg_pool2d(padded, kernel_size=self.kernel_size, stride=1)
        high_band = rgb - low_band
        return low_band, high_band

    def forward(
        self,
        rgb: torch.Tensor,
        highres_feature: torch.Tensor,
        attention_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_band, high_band = self.split(rgb)
        high_detail = self.high_band_proj(high_band)
        if self.use_structure_hint:
            structure_hint = F.avg_pool2d(low_band, kernel_size=3, stride=1, padding=1)
            structure_hint = F.interpolate(
                structure_hint,
                size=high_detail.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            structure_hint = structure_hint.mean(dim=1, keepdim=True)
        else:
            structure_hint = None
        if attention_map is not None:
            if attention_map.shape[-2:] != high_detail.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map,
                    size=high_detail.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if structure_hint is not None:
                attention_map = attention_map + self.structure_hint_strength * structure_hint
            guided_high_detail = high_detail * (1.0 + self.gaussian_feedback_strength * attention_map)
        else:
            guided_high_detail = high_detail
        highres_feature = highres_feature + self.highres_high_band_strength * guided_high_detail
        return low_band, high_band, highres_feature, guided_high_detail


class TextureFrequencyExtractor(nn.Module):
    def __init__(self, feature_channels: int, out_channels: int) -> None:
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.proj = nn.Conv2d(feature_channels + 5, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            ResidualDWBlock(out_channels),
            ResidualDWBlock(out_channels),
        )

    def forward(self, lowres_rgb: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        gray = lowres_rgb.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(torch.clamp(grad_x.square() + grad_y.square(), min=1e-8))
        lap = F.conv2d(gray, self.laplacian, padding=1)

        freq_descriptor = torch.cat(
            [
                gray,
                grad_x.abs(),
                grad_y.abs(),
                grad_mag,
                lap.abs(),
            ],
            dim=1,
        )
        x = torch.cat([lowres_feature, freq_descriptor], dim=1)
        x = self.proj(x)
        return self.refine(x)
