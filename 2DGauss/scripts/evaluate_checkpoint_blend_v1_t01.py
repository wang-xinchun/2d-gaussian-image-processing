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

apply_cuda_visible_devices_policy("evaluate_checkpoint_blend_v1_t01.py")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import psnr  # noqa: E402


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


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    config = load_toml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("step", -1))


def parse_alphas(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one alpha is required")
    return values


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed output blend between two V1/T01 checkpoints.")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=0, help="0 means full image.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use the split default/all samples.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--alphas",
        default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0",
        help="Comma-separated weights for checkpoint B: prediction=(1-alpha)*A + alpha*B.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.dataset_config)
    device = torch.device(args.device)
    if device.type == "cuda":
        fraction_raw = None
        import os

        fraction_raw = os.environ.get("TWODGAUSS_CUDA_MEMORY_FRACTION")
        if fraction_raw:
            memory_device = device if device.index is not None else torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(fraction_raw), device=memory_device)

    alphas = parse_alphas(args.alphas)
    model_a, step_a = load_model(args.config_a, args.checkpoint_a, device)
    model_b, step_b = load_model(args.config_b, args.checkpoint_b, device)

    dataset = build_dataset(config, args.split, args.crop_size, args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    alpha_psnrs = {alpha: [] for alpha in alphas}
    alpha_mses = {alpha: [] for alpha in alphas}
    per_sample: list[dict[str, Any]] = []
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            pred_a = model_a(inputs, return_aux="loss")["restored"].clamp(0.0, 1.0)
            pred_b = model_b(inputs, return_aux="loss")["restored"].clamp(0.0, 1.0)

            sample_scores: list[dict[str, float]] = []
            for alpha in alphas:
                prediction = ((1.0 - alpha) * pred_a + alpha * pred_b).clamp(0.0, 1.0)
                mse = torch.mean((prediction - targets) ** 2).item()
                score = psnr(prediction, targets)
                alpha_psnrs[alpha].append(score)
                alpha_mses[alpha].append(mse)
                sample_scores.append({"alpha": alpha, "psnr": score, "mse": mse})
            best_sample = max(sample_scores, key=lambda item: item["psnr"])
            per_sample.append(
                {
                    "name": str(batch["name"][0]),
                    "best_alpha": best_sample["alpha"],
                    "best_psnr": best_sample["psnr"],
                    "scores": sample_scores,
                }
            )

    alpha_summary = [
        {
            "alpha": alpha,
            "psnr_mean": mean(alpha_psnrs[alpha]),
            "mse_mean": mean(alpha_mses[alpha]),
        }
        for alpha in alphas
    ]
    best_fixed = max(alpha_summary, key=lambda item: item["psnr_mean"])
    oracle_psnr_mean = mean([sample["best_psnr"] for sample in per_sample])
    oracle_alpha_histogram: dict[str, int] = {}
    for sample in per_sample:
        key = f"{sample['best_alpha']:.4f}"
        oracle_alpha_histogram[key] = oracle_alpha_histogram.get(key, 0) + 1

    result = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "checkpoint_step_a": step_a,
        "checkpoint_step_b": step_b,
        "split": args.split,
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(dataset),
        "alphas": alphas,
        "alpha_summary": alpha_summary,
        "best_fixed_alpha": best_fixed["alpha"],
        "best_fixed_psnr_mean": best_fixed["psnr_mean"],
        "best_fixed_mse_mean": best_fixed["mse_mean"],
        "oracle_psnr_mean": oracle_psnr_mean,
        "oracle_alpha_histogram": oracle_alpha_histogram,
        "elapsed_seconds": time.time() - start,
        "samples_detail": per_sample,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
