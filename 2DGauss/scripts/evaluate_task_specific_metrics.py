#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("evaluate_task_specific_metrics.py")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config, compute_auxiliary_loss, compute_loss  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import (  # noqa: E402
    chroma_l1_loss,
    dark_channel_l1_loss,
    edge_magnitude_l1_loss,
    high_frequency_l1_loss,
    local_contrast_l1_loss,
    luminance_l1_loss,
    psnr,
    saturation_l1_loss,
    ssim,
)


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_dataset(config: Dict[str, Any], split: str, crop_size: int, limit: int) -> PairedImageDataset:
    dataset_cfg = config["dataset"]
    if split == "val":
        return PairedImageDataset(
            dataset_cfg["val_input"],
            dataset_cfg["val_target"],
            crop_size=crop_size,
            training=False,
            augment_flip=False,
            offset=int(dataset_cfg.get("val_offset", 0)),
            limit=limit if limit > 0 else int(dataset_cfg.get("val_limit", 0)),
        )
    if split == "test":
        return PairedImageDataset(
            dataset_cfg["test_input"],
            dataset_cfg["test_target"],
            crop_size=crop_size,
            training=False,
            augment_flip=False,
            limit=limit,
        )
    raise ValueError(f"unsupported split: {split}")


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate task-specific restoration diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=0, help="0 means full image.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use the split default/all samples.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(config, args.split, args.crop_size, args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    start = time.time()

    losses: list[float] = []
    auxiliary_losses: list[float] = []
    total_losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    edge_mag: list[float] = []
    high_freq: list[float] = []
    local_contrast: list[float] = []
    luminance: list[float] = []
    chroma: list[float] = []
    saturation: list[float] = []
    dark_channel: list[float] = []
    per_sample: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            aux = model(inputs, return_aux="loss")
            predictions = aux["restored"].clamp(0.0, 1.0)
            loss, _ = compute_loss(predictions, targets, config, input_image=inputs)
            aux_loss, _ = compute_auxiliary_loss(aux, targets, config, inputs=inputs)

            sample = {
                "name": str(batch["name"][0]),
                "loss": float(loss.cpu()),
                "auxiliary_loss": float(aux_loss.cpu()),
                "total_loss": float((loss + aux_loss).cpu()),
                "psnr": psnr(predictions, targets),
                "ssim": ssim(predictions, targets),
                "edge_magnitude_l1": float(edge_magnitude_l1_loss(predictions, targets).cpu()),
                "high_frequency_l1": float(high_frequency_l1_loss(predictions, targets).cpu()),
                "local_contrast_l1": float(local_contrast_l1_loss(predictions, targets).cpu()),
                "luminance_l1": float(luminance_l1_loss(predictions, targets).cpu()),
                "chroma_l1": float(chroma_l1_loss(predictions, targets).cpu()),
                "saturation_l1": float(saturation_l1_loss(predictions, targets).cpu()),
                "dark_channel_l1": float(dark_channel_l1_loss(predictions, targets).cpu()),
            }
            losses.append(sample["loss"])
            auxiliary_losses.append(sample["auxiliary_loss"])
            total_losses.append(sample["total_loss"])
            psnrs.append(sample["psnr"])
            ssims.append(sample["ssim"])
            edge_mag.append(sample["edge_magnitude_l1"])
            high_freq.append(sample["high_frequency_l1"])
            local_contrast.append(sample["local_contrast_l1"])
            luminance.append(sample["luminance_l1"])
            chroma.append(sample["chroma_l1"])
            saturation.append(sample["saturation_l1"])
            dark_channel.append(sample["dark_channel_l1"])
            per_sample.append(sample)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(dataset),
        "loss_mean": mean(losses),
        "auxiliary_loss_mean": mean(auxiliary_losses),
        "total_loss_mean": mean(total_losses),
        "psnr_mean": mean(psnrs),
        "ssim_mean": mean(ssims),
        "edge_magnitude_l1_mean": mean(edge_mag),
        "high_frequency_l1_mean": mean(high_freq),
        "local_contrast_l1_mean": mean(local_contrast),
        "luminance_l1_mean": mean(luminance),
        "chroma_l1_mean": mean(chroma),
        "saturation_l1_mean": mean(saturation),
        "dark_channel_l1_mean": mean(dark_channel),
        "elapsed_seconds": time.time() - start,
        "notes": [
            "PSNR/SSIM remain paired fidelity anchors.",
            "The L1 diagnostics are differentiable task-aware proxies, not replacements for FADE/NIQE/BRISQUE/LPIPS/DISTS.",
            "For dehazing claims, pair these diagnostics with haze-specific/no-reference metrics and visual panels.",
        ],
        "samples_detail": per_sample,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
