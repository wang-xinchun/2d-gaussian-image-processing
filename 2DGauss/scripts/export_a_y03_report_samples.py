#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("export_a_y03_report_samples.py")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.training import psnr, ssim  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_CONFIG = (
    ROOT
    / "experiments/V1/runs/A_y03/configs/"
    "a_y03_medium_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h_confirmation.toml"
)
DEFAULT_CHECKPOINT = ROOT / "experiments/V1/runs/A_y03/checkpoints/final.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "reports/report_package_20260519_a_y03_fast"
DEFAULT_OFFICIAL_METRICS = ROOT / "experiments/V1/runs/A_y03/metrics/official_test_final.json"
DATA_ROOT = Path("/root/autodl-tmp/2D_Gaussian_Dataset")


@dataclass(frozen=True)
class SamplePair:
    dataset: str
    name: str
    input_path: Path
    target_path: Path


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path.relative_to(root)),
    )


def select_evenly(items: list[SamplePair], limit: int) -> list[SamplePair]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    selected: list[SamplePair] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(items[index])
            seen.add(index)
    cursor = 0
    while len(selected) < limit and cursor < len(items):
        if cursor not in seen:
            selected.append(items[cursor])
            seen.add(cursor)
        cursor += 1
    return selected


def exact_name_pairs(dataset: str, input_dir: Path, target_dir: Path) -> list[SamplePair]:
    targets = image_files(target_dir)
    by_name = {path.name: path for path in targets}
    by_stem = {path.stem: path for path in targets}
    pairs: list[SamplePair] = []
    for input_path in image_files(input_dir):
        target_path = by_name.get(input_path.name) or by_stem.get(input_path.stem)
        if target_path is None:
            continue
        pairs.append(SamplePair(dataset, input_path.stem, input_path, target_path))
    return pairs


def sots_pairs(dataset: str, input_dir: Path, target_dir: Path) -> list[SamplePair]:
    targets = image_files(target_dir)
    by_stem = {path.stem: path for path in targets}
    pairs: list[SamplePair] = []
    for input_path in image_files(input_dir):
        target_key = input_path.stem.split("_", 1)[0]
        target_path = by_stem.get(target_key)
        if target_path is None:
            continue
        pairs.append(SamplePair(dataset, input_path.stem, input_path, target_path))
    return pairs


def gopro_pairs(root: Path) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    if not root.is_dir():
        return pairs
    for sequence_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name):
        blur_dir = sequence_dir / "blur"
        sharp_dir = sequence_dir / "sharp"
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        sharp_by_name = {path.name: path for path in image_files(sharp_dir)}
        for input_path in image_files(blur_dir):
            target_path = sharp_by_name.get(input_path.name)
            if target_path is None:
                continue
            name = f"{sequence_dir.name}_{input_path.stem}"
            pairs.append(SamplePair("gopro", name, input_path, target_path))
    return pairs


def build_sample_catalog(samples_per_dataset: int) -> dict[str, list[SamplePair]]:
    catalog = {
        "uhd_deblur": exact_name_pairs(
            "uhd_deblur",
            DATA_ROOT / "UHD_deblur/test/input300",
            DATA_ROOT / "UHD_deblur/test/gt300",
        ),
        "uhd_haze": exact_name_pairs(
            "uhd_haze",
            DATA_ROOT / "UHD_haze/test/input",
            DATA_ROOT / "UHD_haze/test/gt",
        ),
        "gopro": gopro_pairs(DATA_ROOT / "GOPRO_Large/test"),
        "sots_indoor": sots_pairs(
            "sots_indoor",
            DATA_ROOT / "SOTS_extracted/inner/nyuhaze500/hazy",
            DATA_ROOT / "SOTS_extracted/inner/nyuhaze500/gt",
        ),
        "sots_outdoor": sots_pairs(
            "sots_outdoor",
            DATA_ROOT / "SOTS_extracted/inner/outdoor/hazy",
            DATA_ROOT / "SOTS_extracted/inner/outdoor/gt",
        ),
    }
    return {name: select_evenly(pairs, samples_per_dataset) for name, pairs in catalog.items()}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def center_crop_pair(input_tensor: torch.Tensor, target_tensor: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height = min(input_tensor.shape[-2], target_tensor.shape[-2])
    width = min(input_tensor.shape[-1], target_tensor.shape[-1])
    input_tensor = input_tensor[..., :height, :width]
    target_tensor = target_tensor[..., :height, :width]
    if crop_size <= 0:
        return input_tensor, target_tensor
    crop_h = min(height, crop_size)
    crop_w = min(width, crop_size)
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return (
        input_tensor[..., top : top + crop_h, left : left + crop_w],
        target_tensor[..., top : top + crop_h, left : left + crop_w],
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def heatmap_to_pil(map_tensor: torch.Tensor) -> Image.Image:
    tensor = map_tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=0)
    array = tensor.numpy()
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


def resize_for_panel(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_panel(columns: Iterable[tuple[str, Image.Image]], output_path: Path, metrics_line: str) -> None:
    columns = list(columns)
    panel_h = 260
    label_h = 34
    footer_h = 28
    gap = 8
    font = ImageFont.load_default()
    prepared = [(label, resize_for_panel(image, panel_h)) for label, image in columns]
    width = sum(image.width for _, image in prepared) + gap * (len(prepared) + 1)
    height = label_h + panel_h + footer_h + gap * 3
    panel = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    x = gap
    for label, image in prepared:
        draw.text((x, 10), label, fill=(20, 20, 20), font=font)
        panel.paste(image, (x, label_h + gap))
        x += image.width + gap
    draw.text((gap, label_h + panel_h + gap * 2), metrics_line, fill=(35, 35, 35), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fast A_y03 report samples across local paired datasets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    config = load_toml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_root = args.output_root
    image_root = output_root / "images"
    metrics_root = output_root / "metrics"
    script_root = output_root / "scripts"
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    script_root.mkdir(parents=True, exist_ok=True)

    if DEFAULT_OFFICIAL_METRICS.is_file():
        shutil.copy2(DEFAULT_OFFICIAL_METRICS, metrics_root / "official_a_y03_uhd_deblur.json")
    shutil.copy2(Path(__file__), script_root / Path(__file__).name)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    catalog = build_sample_catalog(args.samples_per_dataset)
    rows: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    started = time.time()

    with torch.inference_mode():
        for dataset_name, pairs in catalog.items():
            dataset_rows: list[dict[str, Any]] = []
            for pair in pairs:
                sample_name = safe_name(pair.name)
                sample_dir = image_root / dataset_name / sample_name
                input_tensor = read_rgb(pair.input_path)
                target_tensor = read_rgb(pair.target_path)
                input_tensor, target_tensor = center_crop_pair(input_tensor, target_tensor, args.crop_size)
                inputs = input_tensor.unsqueeze(0).to(device)
                targets = target_tensor.unsqueeze(0).to(device)

                aux = model(inputs, return_aux=True)
                restored = aux["restored"].clamp(0.0, 1.0)
                attention = aux["attention_map"]
                error = (restored - targets).abs().mean(dim=1, keepdim=True)

                input_psnr = psnr(inputs, targets)
                output_psnr = psnr(restored, targets)
                input_ssim = ssim(inputs, targets)
                output_ssim = ssim(restored, targets)

                input_image = tensor_to_pil(inputs)
                restored_image = tensor_to_pil(restored)
                target_image = tensor_to_pil(targets)
                error_image = heatmap_to_pil(error)
                attention_image = heatmap_to_pil(attention)

                sample_dir.mkdir(parents=True, exist_ok=True)
                input_image.save(sample_dir / "input.png")
                restored_image.save(sample_dir / "ours_a_y03.png")
                target_image.save(sample_dir / "gt.png")
                error_image.save(sample_dir / "error_heatmap.png")
                attention_image.save(sample_dir / "gaussian_attention.png")
                make_panel(
                    [
                        ("Input", input_image),
                        ("Ours", restored_image),
                        ("GT", target_image),
                        ("Error", error_image),
                        ("Gaussian attention", attention_image),
                    ],
                    sample_dir / "panel.png",
                    (
                        f"{dataset_name} / {pair.name} | "
                        f"PSNR {input_psnr:.2f}->{output_psnr:.2f} dB | "
                        f"SSIM {input_ssim:.4f}->{output_ssim:.4f}"
                    ),
                )

                row = {
                    "dataset": dataset_name,
                    "sample": pair.name,
                    "height": int(inputs.shape[-2]),
                    "width": int(inputs.shape[-1]),
                    "input_psnr": round(float(input_psnr), 6),
                    "ours_psnr": round(float(output_psnr), 6),
                    "delta_psnr": round(float(output_psnr - input_psnr), 6),
                    "input_ssim": round(float(input_ssim), 6),
                    "ours_ssim": round(float(output_ssim), 6),
                    "delta_ssim": round(float(output_ssim - input_ssim), 6),
                    "input_path": str(pair.input_path),
                    "target_path": str(pair.target_path),
                    "panel_path": str(sample_dir / "panel.png"),
                }
                rows.append(row)
                dataset_rows.append(row)

                del inputs, targets, aux, restored, attention, error
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if dataset_rows:
                dataset_summary[dataset_name] = {
                    "samples": len(dataset_rows),
                    "input_psnr_mean": sum(row["input_psnr"] for row in dataset_rows) / len(dataset_rows),
                    "ours_psnr_mean": sum(row["ours_psnr"] for row in dataset_rows) / len(dataset_rows),
                    "delta_psnr_mean": sum(row["delta_psnr"] for row in dataset_rows) / len(dataset_rows),
                    "input_ssim_mean": sum(row["input_ssim"] for row in dataset_rows) / len(dataset_rows),
                    "ours_ssim_mean": sum(row["ours_ssim"] for row in dataset_rows) / len(dataset_rows),
                    "delta_ssim_mean": sum(row["delta_ssim"] for row in dataset_rows) / len(dataset_rows),
                }
            else:
                dataset_summary[dataset_name] = {"samples": 0}

    manifest = {
        "created_by": "scripts/export_a_y03_report_samples.py",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "crop_size": int(args.crop_size),
        "samples_per_dataset": int(args.samples_per_dataset),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "dataset_summary": dataset_summary,
        "elapsed_seconds": time.time() - started,
    }
    save_json(metrics_root / "sample_metrics.json", {"manifest": manifest, "samples": rows})
    save_json(output_root / "manifest.json", manifest)
    write_csv(metrics_root / "sample_metrics.csv", rows)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
