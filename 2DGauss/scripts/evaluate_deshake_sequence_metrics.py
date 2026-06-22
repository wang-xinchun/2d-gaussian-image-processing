#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_gray(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if max_side > 0:
            scale = min(1.0, max_side / max(image.width, image.height))
            if scale < 1.0:
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def center_crop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    def crop(x: np.ndarray) -> np.ndarray:
        y0 = max((x.shape[0] - height) // 2, 0)
        x0 = max((x.shape[1] - width) // 2, 0)
        return x[y0 : y0 + height, x0 : x0 + width]
    return crop(a), crop(b)


def edge_map(gray: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gx[:, :-1] = np.abs(gray[:, 1:] - gray[:, :-1])
    return gx + gy


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = center_crop_pair(a, b)
    a = a - float(a.mean())
    b = b - float(b.mean())
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    denom = np.abs(cross)
    cross = cross / np.maximum(denom, 1e-8)
    corr = np.fft.ifft2(cross).real
    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if y > corr.shape[0] // 2:
        y -= corr.shape[0]
    if x > corr.shape[1] // 2:
        x -= corr.shape[1]
    return float(y), float(x)


def sequence_metrics(paths: list[Path], max_side: int) -> dict[str, float | int]:
    if len(paths) < 2:
        return {"frame_count": len(paths)}
    shifts: list[float] = []
    frame_l1: list[float] = []
    edge_l1: list[float] = []
    prev = read_gray(paths[0], max_side)
    prev_edge = edge_map(prev)
    for path in paths[1:]:
        cur = read_gray(path, max_side)
        cur_edge = edge_map(cur)
        a, b = center_crop_pair(prev, cur)
        ea, eb = center_crop_pair(prev_edge, cur_edge)
        dy, dx = phase_shift(prev, cur)
        shifts.append(math.sqrt(dx * dx + dy * dy))
        frame_l1.append(float(np.mean(np.abs(a - b))))
        edge_l1.append(float(np.mean(np.abs(ea - eb))))
        prev = cur
        prev_edge = cur_edge
    return {
        "frame_count": len(paths),
        "adjacent_pairs": len(paths) - 1,
        "phase_shift_abs_mean": float(np.mean(shifts)),
        "phase_shift_abs_std": float(np.std(shifts)),
        "frame_l1_mean": float(np.mean(frame_l1)),
        "frame_l1_std": float(np.std(frame_l1)),
        "edge_l1_mean": float(np.mean(edge_l1)),
        "edge_l1_std": float(np.std(edge_l1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute lightweight no-reference temporal stability proxies for a frame sequence."
    )
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sequence = image_files(args.sequence_dir)
    result: dict[str, object] = {
        "sequence_dir": str(args.sequence_dir),
        "metrics": sequence_metrics(sequence, args.max_side),
        "note": (
            "These are deployment-gate proxy metrics, not final video-stabilization quality. "
            "Scene motion can affect them; pair with visual panels and task-specific metrics."
        ),
    }
    if args.reference_dir is not None:
        reference = image_files(args.reference_dir)
        pair_count = min(len(sequence), len(reference))
        l1_values: list[float] = []
        for pred_path, ref_path in zip(sequence[:pair_count], reference[:pair_count]):
            pred = read_gray(pred_path, args.max_side)
            ref = read_gray(ref_path, args.max_side)
            pred, ref = center_crop_pair(pred, ref)
            l1_values.append(float(np.mean(np.abs(pred - ref))))
        result["reference_dir"] = str(args.reference_dir)
        result["reference_pair_count"] = pair_count
        result["reference_l1_mean"] = float(np.mean(l1_values)) if l1_values else None

    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
