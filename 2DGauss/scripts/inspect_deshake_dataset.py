#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".npy"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".y4m", ".yuv"}
PAIR_CANDIDATES = (
    ("input", "target"),
    ("inputs", "targets"),
    ("shaky", "stable"),
    ("unstable", "stable"),
    ("jitter", "stable"),
    ("jittery", "stable"),
    ("raw", "stable"),
    ("source", "target"),
    ("blur", "sharp"),
)


def collect_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def count_suffixes(files: Iterable[Path]) -> dict[str, int]:
    return dict(sorted(Counter(path.suffix.lower() for path in files).items()))


def image_info(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path, mmap_mode="r")
            return {
                "path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        except Exception as exc:  # pragma: no cover - used for field diagnostics
            return {"path": str(path), "error": str(exc)}
    try:
        with Image.open(path) as image:
            return {
                "path": str(path),
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
    except Exception as exc:  # pragma: no cover - used for field diagnostics
        return {"path": str(path), "error": str(exc)}


def relative_index(root: Path, files: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in files:
        result[str(path.relative_to(root))] = path
    return result


def stem_index(files: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in files:
        result.setdefault(path.stem, path)
    return result


def pair_files(input_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    input_files = collect_files(input_dir, FRAME_SUFFIXES)
    target_files = collect_files(target_dir, FRAME_SUFFIXES)
    targets_by_rel = relative_index(target_dir, target_files)
    targets_by_stem = stem_index(target_files)
    pairs: list[tuple[Path, Path]] = []
    for input_path in input_files:
        rel = str(input_path.relative_to(input_dir))
        target_path = targets_by_rel.get(rel) or targets_by_stem.get(input_path.stem)
        if target_path is not None:
            pairs.append((input_path, target_path))
    return pairs


def find_pair_candidates(root: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for left_name, right_name in PAIR_CANDIDATES:
        for left_dir in sorted(root.rglob(left_name)):
            if not left_dir.is_dir():
                continue
            right_dir = left_dir.parent / right_name
            if not right_dir.is_dir():
                continue
            pairs = pair_files(left_dir, right_dir)
            candidates.append(
                {
                    "input_dir": str(left_dir),
                    "target_dir": str(right_dir),
                    "pair_count": len(pairs),
                    "input_frame_count": len(collect_files(left_dir, FRAME_SUFFIXES)),
                    "target_frame_count": len(collect_files(right_dir, FRAME_SUFFIXES)),
                }
            )
    return sorted(candidates, key=lambda item: int(item["pair_count"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an incoming UHD video deshaking/stabilization dataset without modifying it."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root to scan.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Optional shaky/input frame directory.")
    parser.add_argument("--target-dir", type=Path, default=None, help="Optional stable/target frame directory.")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--allow-unpaired", action="store_true", help="Return success even if no paired frame view exists yet.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    frame_files = collect_files(root, FRAME_SUFFIXES)
    video_files = collect_files(root, VIDEO_SUFFIXES)
    pair_candidates = find_pair_candidates(root)
    explicit_pairs: list[tuple[Path, Path]] = []
    if args.input_dir is not None and args.target_dir is not None:
        explicit_pairs = pair_files(args.input_dir, args.target_dir)

    sampled_frames = [image_info(path) for path in frame_files[: max(args.sample_count, 0)]]
    best_candidate = pair_candidates[0] if pair_candidates else None
    pair_count = len(explicit_pairs)
    if pair_count == 0 and best_candidate is not None:
        pair_count = int(best_candidate["pair_count"])

    status = "ok" if pair_count >= args.min_pairs else "pending"
    result = {
        "root": str(root),
        "root_exists": root.exists(),
        "frame_count": len(frame_files),
        "video_count": len(video_files),
        "frame_suffixes": count_suffixes(frame_files),
        "video_suffixes": count_suffixes(video_files),
        "sampled_frames": sampled_frames,
        "explicit_input_dir": str(args.input_dir) if args.input_dir is not None else None,
        "explicit_target_dir": str(args.target_dir) if args.target_dir is not None else None,
        "explicit_pair_count": len(explicit_pairs),
        "pair_candidates": pair_candidates[:20],
        "best_pair_candidate": best_candidate,
        "status": status,
        "notes": [
            "This script only inspects files; it does not validate visual stabilization quality.",
            "A valid deshaking training view needs shaky/input frames and stable/target frames or an equivalent supervision file.",
        ],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if status != "ok" and not args.allow_unpaired:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
