#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("evaluate_dual_backbone_router_v1_t01.py")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config, compute_loss  # noqa: E402
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
            offset=int(dataset_cfg.get("test_offset", 0)),
            limit=limit if limit > 0 else int(dataset_cfg.get("test_limit", 0)),
        )
    raise ValueError(f"unsupported split: {split}")


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def load_checkpoint_model(
    checkpoint: dict[str, Any],
    *,
    key: str,
    config_key: str,
    device: torch.device,
) -> CrossResolutionGaussianRestorer:
    model = CrossResolutionGaussianRestorer(build_model_config(checkpoint[config_key])).to(device)
    model.load_state_dict(checkpoint[key])
    model.eval()
    return model


def route_high(input_image: torch.Tensor, threshold: float, direction: str) -> torch.Tensor:
    input_std = input_image.detach().flatten(start_dim=1).std(dim=1, unbiased=False)
    if direction in {"gt", "gte"}:
        return input_std > threshold if direction == "gt" else input_std >= threshold
    if direction in {"lt", "lte"}:
        return input_std < threshold if direction == "lt" else input_std <= threshold
    raise ValueError(f"unsupported direction: {direction}")


def load_records(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {record["name"]: record for record in data["test_records"]}, dict(data.get("selected_rule", {}))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for metric in (
        "psnr",
        "loss",
        "ssim",
        "edge_magnitude_l1",
        "high_frequency_l1",
        "local_contrast_l1",
        "luminance_l1",
        "chroma_l1",
        "saturation_l1",
        "dark_channel_l1",
        "input_std",
    ):
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            result[f"{metric}_mean"] = mean(values)
    if rows and "gap_router" in rows[0]:
        result["gap_router_mean"] = mean([float(row["gap_router"]) for row in rows])
        result["gap_oracle_mean"] = mean([float(row["gap_oracle"]) for row in rows])
        result["router_psnr_mean"] = mean([float(row["router_psnr"]) for row in rows])
        result["oracle_psnr_mean"] = mean([float(row["oracle_psnr"]) for row in rows])
    return result


def group_summary(rows: list[dict[str, Any]], *, key: str, value: Any) -> dict[str, Any]:
    selected = [row for row in rows if row[key] == value]
    return summarize_rows(selected)


def rule_choose_b(input_std: float, threshold: float, direction: str) -> bool:
    if direction == "lt":
        return input_std < threshold
    if direction == "lte":
        return input_std <= threshold
    if direction == "gt":
        return input_std > threshold
    if direction == "gte":
        return input_std >= threshold
    raise ValueError(f"unsupported selected rule direction: {direction}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one packed dual-backbone V1/T01 input_std router checkpoint.")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset_config = load_toml(args.dataset_config)
    device = torch.device(args.device)
    if device.type == "cuda":
        fraction_raw = os.environ.get("TWODGAUSS_CUDA_MEMORY_FRACTION")
        if fraction_raw:
            memory_device = device if device.index is not None else torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(fraction_raw), device=memory_device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("architecture") != "dual_backbone_input_std_router_v1_t01":
        raise ValueError(f"unsupported checkpoint architecture: {checkpoint.get('architecture')}")
    model_high = load_checkpoint_model(checkpoint, key="model_high", config_key="config_high", device=device)
    model_low = load_checkpoint_model(checkpoint, key="model_low", config_key="config_low", device=device)
    router = checkpoint["router"]
    threshold = float(router["threshold"])
    direction = str(router["direction"])
    records, selected_rule = load_records(args.records)

    dataset = build_dataset(dataset_config, args.split, args.crop_size, args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    rows: list[dict[str, Any]] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            high_mask = route_high(inputs, threshold, direction)
            pred_high = model_high(inputs).clamp(0.0, 1.0)
            pred_low = model_low(inputs).clamp(0.0, 1.0)
            mask = high_mask.view(-1, 1, 1, 1).to(dtype=pred_high.dtype)
            predictions = mask * pred_high + (1.0 - mask) * pred_low
            loss, _ = compute_loss(predictions, targets, dataset_config, input_image=inputs)
            input_std = float(inputs.detach().flatten(start_dim=1).std(dim=1, unbiased=False).mean().cpu())
            row = {
                "name": str(batch["name"][0]),
                "route": "high" if bool(high_mask.item()) else "low",
                "input_std": input_std,
                "loss": float(loss.cpu()),
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
            record = records.get(row["name"])
            if record is not None:
                psnr_a = float(record["psnr_a"])
                psnr_b = float(record["psnr_b"])
                router_threshold = float(selected_rule.get("threshold", threshold))
                router_direction = str(selected_rule.get("direction", "lte"))
                choose_low = rule_choose_b(input_std, router_threshold, router_direction)
                router_psnr = psnr_b if choose_low else psnr_a
                oracle_psnr = max(psnr_a, psnr_b)
                row |= {
                    "psnr_a": psnr_a,
                    "psnr_b": psnr_b,
                    "router_psnr": router_psnr,
                    "oracle_psnr": oracle_psnr,
                    "gap_router": row["psnr"] - router_psnr,
                    "gap_oracle": row["psnr"] - oracle_psnr,
                    "a08_oracle": psnr_a >= psnr_b,
                }
            rows.append(row)

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "architecture": str(checkpoint["architecture"]),
        "checkpoint_high": str(checkpoint["checkpoint_high"]),
        "checkpoint_low": str(checkpoint["checkpoint_low"]),
        "checkpoint_step_high": int(checkpoint["checkpoint_step_high"]),
        "checkpoint_step_low": int(checkpoint["checkpoint_step_low"]),
        "label_high": str(checkpoint["label_high"]),
        "label_low": str(checkpoint["label_low"]),
        "split": args.split,
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(rows),
        "threshold": threshold,
        "direction": direction,
        "records": str(args.records) if args.records else None,
        "records_selected_rule": selected_rule,
        "elapsed_seconds": time.time() - start,
        "overall": summarize_rows(rows),
        "high_route": group_summary(rows, key="route", value="high"),
        "low_route": group_summary(rows, key="route", value="low"),
        "samples_detail": rows,
    }
    for metric in (
        "psnr",
        "loss",
        "ssim",
        "edge_magnitude_l1",
        "high_frequency_l1",
        "local_contrast_l1",
        "luminance_l1",
        "chroma_l1",
        "saturation_l1",
        "dark_channel_l1",
        "input_std",
    ):
        result[f"{metric}_mean"] = mean([float(row[metric]) for row in rows])
    if rows and "gap_router" in rows[0]:
        result["gap_router_mean"] = mean([float(row["gap_router"]) for row in rows])
        result["gap_oracle_mean"] = mean([float(row["gap_oracle"]) for row in rows])
        result["a08_oracle"] = group_summary(rows, key="a08_oracle", value=True)
        result["a15_oracle"] = group_summary(rows, key="a08_oracle", value=False)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
