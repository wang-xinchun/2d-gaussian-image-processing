#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("smoke_test_model_framework.py")

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uhd_gaussian import CrossResolutionGaussianRestorer, ModelConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(7)

    config = ModelConfig()
    model = CrossResolutionGaussianRestorer(config)
    model.eval()

    x = torch.randn(1, 3, 721, 1283)
    with torch.no_grad():
        outputs = model(x, return_aux=True)

    restored = outputs["restored"]
    attention = outputs["attention_map"]
    gaussian_params = outputs["gaussian_params"]
    affine_coeffs = outputs["affine_coeffs"]

    assert restored.shape == x.shape, f"restored shape mismatch: {restored.shape} vs {x.shape}"
    assert attention.shape[-2:] == x.shape[-2:], "attention map size must match the input"
    assert gaussian_params.dim() == 5, "gaussian parameter tensor should be [B, H, W, K, P]"
    assert affine_coeffs.dim() == 4, "affine coefficient tensor should be [B, R, H, W]"

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("smoke_test: ok")
    print(f"input_shape: {tuple(x.shape)}")
    print(f"restored_shape: {tuple(restored.shape)}")
    print(f"attention_shape: {tuple(attention.shape)}")
    print(f"gaussian_params_shape: {tuple(gaussian_params.shape)}")
    print(f"affine_coeffs_shape: {tuple(affine_coeffs.shape)}")
    print(f"total_params: {total_params}")


if __name__ == "__main__":
    main()
