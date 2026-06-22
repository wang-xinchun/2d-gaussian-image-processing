#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterable

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("export_mechanism_panels.py")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config, compute_control_canvas_target, compute_loss  # noqa: E402
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


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def tensor_stats(tensor: torch.Tensor | None) -> dict[str, float] | None:
    if tensor is None:
        return None
    values = tensor.detach().float().cpu()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "mean_abs": float(values.abs().mean().item()),
    }


def _as_chw(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"expected CHW-like tensor, got shape={tuple(tensor.shape)}")
    return tensor


def rgb_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _as_chw(tensor)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] > 3:
        tensor = tensor[:3]
    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def heatmap_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _as_chw(tensor)
    if tensor.shape[0] > 1:
        tensor = tensor.abs().mean(dim=0, keepdim=True)
    array = tensor[0].numpy()
    lo = float(np.percentile(array, 1.0))
    hi = float(np.percentile(array, 99.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(array.min())
        hi = float(array.max())
    norm = np.clip((array - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    red = np.clip(1.75 * norm, 0.0, 1.0)
    green = np.clip(1.75 * norm - 0.45, 0.0, 1.0)
    blue = np.clip(1.25 * (1.0 - norm), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8))


def signed_rgb_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _as_chw(tensor)
    if tensor.shape[0] != 3:
        return heatmap_to_pil(tensor.abs().mean(dim=0, keepdim=True))
    array = tensor.permute(1, 2, 0).numpy()
    scale = float(np.percentile(np.abs(array), 99.0))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.abs(array).max())
    scale = max(scale, 1e-8)
    visible = np.clip(0.5 + 0.5 * array / scale, 0.0, 1.0)
    return Image.fromarray((visible * 255.0).round().astype(np.uint8))


def feature_energy_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _as_chw(tensor)
    energy = tensor.abs().mean(dim=0, keepdim=True)
    return heatmap_to_pil(energy)


def resize_for_panel(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_panel(columns: Iterable[tuple[str, Image.Image]], output_path: Path, metrics_line: str) -> None:
    columns = list(columns)
    panel_h = 220
    label_h = 30
    footer_h = 30
    gap = 8
    font = ImageFont.load_default()
    prepared = [(label, resize_for_panel(image, panel_h)) for label, image in columns]
    width = sum(image.width for _, image in prepared) + gap * (len(prepared) + 1)
    height = label_h + panel_h + footer_h + gap * 3
    panel = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    x = gap
    for label, image in prepared:
        draw.text((x, 9), label, fill=(20, 20, 20), font=font)
        panel.paste(image, (x, label_h + gap))
        x += image.width + gap
    draw.text((gap, label_h + panel_h + gap * 2), metrics_line, fill=(35, 35, 35), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export restoration mechanism panels with Gaussian/frequency aux maps.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=512, help="0 means full image.")
    parser.add_argument("--limit", type=int, default=4, help="0 means use the split default/all samples.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = load_toml(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(config, args.split, args.crop_size, args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_root = args.output_root
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows: list[dict[str, Any]] = []
    aux_records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            name = str(batch["name"][0])
            sample_dir = image_root / safe_name(name)
            sample_dir.mkdir(parents=True, exist_ok=True)

            aux = model(inputs, return_aux=True)
            restored = aux["restored"].clamp(0.0, 1.0)
            attention = aux["attention_map"]
            raw_attention = aux.get("raw_attention_map")
            control_attention = aux.get("control_attention_map")
            attention_bridge = aux.get("attention_bridge_map")
            low_band = aux.get("explicit_low_band")
            high_band = aux.get("explicit_high_band")
            guided_high_detail = aux.get("guided_high_detail")
            control_canvas = aux.get("control_canvas")
            control_canvas_target = aux.get("control_canvas_target")
            if control_canvas is not None:
                control_canvas_target = compute_control_canvas_target(targets, config, inputs=inputs)
            control_feedback_map = aux.get("control_feedback_map")
            decoupled_selectivity_map = aux.get("decoupled_selectivity_map")
            decoupled_amplitude_map = aux.get("decoupled_amplitude_map")
            decoupled_correction_map = aux.get("decoupled_correction_map")

            error = (restored - targets).abs().mean(dim=1, keepdim=True)
            residual = restored - inputs

            images: dict[str, Image.Image] = {
                "input": rgb_to_pil(inputs),
                "restored": rgb_to_pil(restored),
                "target": rgb_to_pil(targets),
                "error_heatmap": heatmap_to_pil(error),
                "gaussian_attention": heatmap_to_pil(attention),
                "restoration_residual_signed": signed_rgb_to_pil(residual),
                "restoration_residual_energy": feature_energy_to_pil(residual),
            }
            if raw_attention is not None:
                images["raw_gaussian_attention"] = heatmap_to_pil(raw_attention)
            if attention_bridge is not None:
                images["attention_demand_bridge"] = heatmap_to_pil(attention_bridge)
            if control_attention is not None:
                images["control_attention"] = heatmap_to_pil(control_attention)
            if low_band is not None:
                images["explicit_low_band"] = rgb_to_pil(low_band)
            if high_band is not None:
                images["explicit_high_band_signed"] = signed_rgb_to_pil(high_band)
                images["explicit_high_band_energy"] = feature_energy_to_pil(high_band)
            if guided_high_detail is not None:
                images["guided_high_detail_energy"] = feature_energy_to_pil(guided_high_detail)
            if control_canvas is not None:
                images["control_canvas"] = heatmap_to_pil(control_canvas)
            if control_canvas_target is not None:
                images["control_canvas_target"] = heatmap_to_pil(control_canvas_target)
            if control_feedback_map is not None:
                images["control_feedback_map"] = heatmap_to_pil(control_feedback_map)
            if decoupled_selectivity_map is not None:
                images["decoupled_selectivity_map"] = heatmap_to_pil(decoupled_selectivity_map)
            if decoupled_amplitude_map is not None:
                images["decoupled_amplitude_map"] = feature_energy_to_pil(decoupled_amplitude_map)
            if decoupled_correction_map is not None:
                images["decoupled_correction_map"] = signed_rgb_to_pil(decoupled_correction_map)

            for image_name, image in images.items():
                image.save(sample_dir / f"{image_name}.png")

            columns: list[tuple[str, Image.Image]] = [("Input", images["input"])]
            if "explicit_low_band" in images:
                columns.append(("Low band", images["explicit_low_band"]))
            if "explicit_high_band_energy" in images:
                columns.append(("High band", images["explicit_high_band_energy"]))
            if "raw_gaussian_attention" in images:
                columns.append(("Raw Gaussian", images["raw_gaussian_attention"]))
            if "attention_demand_bridge" in images:
                columns.append(("Demand bridge", images["attention_demand_bridge"]))
            columns.append(("Gaussian control", images["gaussian_attention"]))
            if "control_attention" in images:
                columns.append(("Control attention", images["control_attention"]))
            if "control_canvas" in images:
                columns.append(("Canvas", images["control_canvas"]))
            if "control_canvas_target" in images:
                columns.append(("Canvas target", images["control_canvas_target"]))
            if "control_feedback_map" in images:
                columns.append(("Feedback", images["control_feedback_map"]))
            if "decoupled_selectivity_map" in images:
                columns.append(("Selectivity", images["decoupled_selectivity_map"]))
            if "decoupled_amplitude_map" in images:
                columns.append(("Amplitude", images["decoupled_amplitude_map"]))
            if "decoupled_correction_map" in images:
                columns.append(("Decoupled corr", images["decoupled_correction_map"]))
            if "guided_high_detail_energy" in images:
                columns.append(("Guided detail", images["guided_high_detail_energy"]))
            columns.extend(
                [
                    ("Restored", images["restored"]),
                    ("GT", images["target"]),
                    ("Error", images["error_heatmap"]),
                ]
            )

            sample_psnr = psnr(restored, targets)
            sample_ssim = ssim(restored, targets)
            input_psnr = psnr(inputs, targets)
            input_ssim = ssim(inputs, targets)
            loss, loss_parts = compute_loss(restored, targets, config, input_image=inputs)
            metrics_line = (
                f"{name} | PSNR {input_psnr:.2f}->{sample_psnr:.2f} dB | "
                f"SSIM {input_ssim:.4f}->{sample_ssim:.4f}"
            )
            make_panel(columns, sample_dir / "mechanism_panel.png", metrics_line)

            row = {
                "name": name,
                "height": int(inputs.shape[-2]),
                "width": int(inputs.shape[-1]),
                "input_psnr": round(float(input_psnr), 6),
                "restored_psnr": round(float(sample_psnr), 6),
                "delta_psnr": round(float(sample_psnr - input_psnr), 6),
                "input_ssim": round(float(input_ssim), 6),
                "restored_ssim": round(float(sample_ssim), 6),
                "delta_ssim": round(float(sample_ssim - input_ssim), 6),
                "loss_total": round(float(loss.cpu()), 8),
                "edge_magnitude_l1": round(float(edge_magnitude_l1_loss(restored, targets).cpu()), 8),
                "high_frequency_l1": round(float(high_frequency_l1_loss(restored, targets).cpu()), 8),
                "local_contrast_l1": round(float(local_contrast_l1_loss(restored, targets).cpu()), 8),
                "luminance_l1": round(float(luminance_l1_loss(restored, targets).cpu()), 8),
                "chroma_l1": round(float(chroma_l1_loss(restored, targets).cpu()), 8),
                "saturation_l1": round(float(saturation_l1_loss(restored, targets).cpu()), 8),
                "dark_channel_l1": round(float(dark_channel_l1_loss(restored, targets).cpu()), 8),
                "panel_path": str(sample_dir / "mechanism_panel.png"),
            }
            for part_name, part_value in loss_parts.items():
                row[f"train_{part_name}"] = round(float(part_value), 8)
            rows.append(row)

            aux_records.append(
                {
                    "name": name,
                    "attention_map": tensor_stats(attention),
                    "raw_attention_map": tensor_stats(raw_attention),
                    "control_attention_map": tensor_stats(control_attention),
                    "attention_bridge_map": tensor_stats(attention_bridge),
                    "explicit_low_band": tensor_stats(low_band),
                    "explicit_high_band": tensor_stats(high_band),
                    "guided_high_detail": tensor_stats(guided_high_detail),
                    "control_canvas": tensor_stats(control_canvas),
                    "control_canvas_target": tensor_stats(control_canvas_target),
                    "control_feedback_map": tensor_stats(control_feedback_map),
                    "decoupled_selectivity_map": tensor_stats(decoupled_selectivity_map),
                    "decoupled_amplitude_map": tensor_stats(decoupled_amplitude_map),
                    "decoupled_correction_map": tensor_stats(decoupled_correction_map),
                    "restoration_residual": tensor_stats(residual),
                }
            )

            del inputs, targets, aux, restored, attention, raw_attention, control_attention, attention_bridge, low_band, high_band, guided_high_detail, control_canvas, control_canvas_target, control_feedback_map, decoupled_selectivity_map, decoupled_amplitude_map, decoupled_correction_map, error, residual
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": str(args.config),
        "split": args.split,
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(rows),
        "elapsed_seconds": time.time() - started,
        "mean": {
            "input_psnr": mean([row["input_psnr"] for row in rows]),
            "restored_psnr": mean([row["restored_psnr"] for row in rows]),
            "delta_psnr": mean([row["delta_psnr"] for row in rows]),
            "input_ssim": mean([row["input_ssim"] for row in rows]),
            "restored_ssim": mean([row["restored_ssim"] for row in rows]),
            "delta_ssim": mean([row["delta_ssim"] for row in rows]),
            "edge_magnitude_l1": mean([row["edge_magnitude_l1"] for row in rows]),
            "high_frequency_l1": mean([row["high_frequency_l1"] for row in rows]),
            "local_contrast_l1": mean([row["local_contrast_l1"] for row in rows]),
        },
        "notes": [
            "Mechanism panels visualize explicit low/high split, Gaussian control attention, optional control canvas/target/feedback maps, guided high-detail energy, output, GT, and error.",
            "The panel is diagnostic evidence only; promotion still requires same-protocol official metrics and visual failure-mode inspection.",
        ],
        "samples": rows,
        "aux_stats": aux_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "sample_metrics.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
