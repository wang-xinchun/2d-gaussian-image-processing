#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("run_v1_frame_enhancer.py")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_v1_t01 import build_model_config, load_toml  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def infer_tensor(model: CrossResolutionGaussianRestorer, tensor: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    if tile_size <= 0:
        return model(tensor).clamp(0.0, 1.0)
    _, _, height, width = tensor.shape
    output = torch.zeros_like(tensor)
    weight = torch.zeros((1, 1, height, width), device=tensor.device, dtype=tensor.dtype)
    for y in tile_starts(height, tile_size, overlap):
        for x in tile_starts(width, tile_size, overlap):
            patch = tensor[..., y : y + tile_size, x : x + tile_size]
            pred = model(patch).clamp(0.0, 1.0)
            output[..., y : y + pred.shape[-2], x : x + pred.shape[-1]] += pred
            weight[..., y : y + pred.shape[-2], x : x + pred.shape[-1]] += 1.0
    return output / weight.clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing V1/T01 single-frame restorer over a frame directory. "
            "This is a frame-enhancement baseline, not a full video deshaking model."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    paths = image_files(args.input_dir)
    if args.limit > 0:
        paths = paths[: args.limit]
    start = time.time()
    with torch.no_grad():
        for index, path in enumerate(paths, 1):
            rel = path.relative_to(args.input_dir)
            output_path = args.output_dir / rel
            tensor = read_rgb(path).to(device)
            restored = infer_tensor(model, tensor, args.tile_size, args.tile_overlap)
            save_rgb(restored, output_path)
            if index % 20 == 0:
                print(f"processed={index}/{len(paths)}")

    result = {
        "classification": "frame_enhancement_baseline_not_full_video_deshaking",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "processed_frames": len(paths),
        "elapsed_seconds": time.time() - start,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
