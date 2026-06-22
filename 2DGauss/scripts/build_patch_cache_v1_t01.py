#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairedImageSample:
    name: str
    input_path: Path
    target_path: Path


def _image_files(root: Path) -> List[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def _index_by_name(paths: Sequence[Path]) -> Dict[str, Path]:
    return {path.name: path for path in paths}


def _index_by_stem(paths: Sequence[Path]) -> Dict[str, Path]:
    return {path.stem: path for path in paths}


def collect_pairs(input_dir: Path, target_dir: Path) -> List[PairedImageSample]:
    input_paths = _image_files(input_dir)
    target_paths = _image_files(target_dir)
    targets_by_name = _index_by_name(target_paths)
    targets_by_stem = _index_by_stem(target_paths)

    samples: List[PairedImageSample] = []
    for input_path in input_paths:
        target_path = targets_by_name.get(input_path.name) or targets_by_stem.get(input_path.stem)
        if target_path is not None:
            samples.append(PairedImageSample(input_path.name, input_path, target_path))
    if not samples:
        raise RuntimeError(f"no paired images found between {input_dir} and {target_dir}")
    return samples


def stable_seed(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    crop_w = min(width, crop_size)
    crop_h = min(height, crop_size)
    left = max((width - crop_w) // 2, 0)
    top = max((height - crop_h) // 2, 0)
    return left, top, left + crop_w, top + crop_h


def train_crop_boxes(width: int, height: int, crop_size: int, count: int, key: str) -> List[tuple[int, int, int, int]]:
    crop_w = min(width, crop_size)
    crop_h = min(height, crop_size)
    if width <= crop_w or height <= crop_h:
        return [(0, 0, crop_w, crop_h)]

    max_left = width - crop_w
    max_top = height - crop_h
    target_count = max(count, 1)
    rng = random.Random(stable_seed(key))
    boxes: List[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    attempts = 0
    max_attempts = max(target_count * 32, 64)
    while len(boxes) < target_count and attempts < max_attempts:
        left = rng.randint(0, max_left)
        top = rng.randint(0, max_top)
        box = (left, top, left + crop_w, top + crop_h)
        if box not in seen:
            boxes.append(box)
            seen.add(box)
        attempts += 1

    if len(boxes) < target_count:
        anchors = [
            (0, 0),
            (max_left, 0),
            (0, max_top),
            (max_left, max_top),
            (max_left // 2, max_top // 2),
        ]
        rng.shuffle(anchors)
        for left, top in anchors:
            box = (left, top, left + crop_w, top + crop_h)
            if box not in seen:
                boxes.append(box)
                seen.add(box)
            if len(boxes) >= target_count:
                break

    return boxes[:target_count]


def write_png(image: Image.Image, path: Path, compress_level: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=compress_level)
    return path.stat().st_size


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_root already exists: {output_root}")
        shutil.rmtree(output_root)

    train_pairs = collect_pairs(args.train_input.resolve(), args.train_target.resolve())
    test_pairs = collect_pairs(args.test_input.resolve(), args.test_target.resolve())

    val_pairs = train_pairs[: args.val_limit]
    effective_train_pairs = train_pairs[args.train_offset :]
    if not effective_train_pairs:
        raise RuntimeError("effective training split is empty after applying train_offset")
    if args.train_source_limit > 0:
        effective_train_pairs = effective_train_pairs[: args.train_source_limit]
    if args.val_source_limit > 0:
        val_pairs = val_pairs[: args.val_source_limit]
    if args.test_source_limit > 0:
        test_pairs = test_pairs[: args.test_source_limit]

    records_path = output_root / "cache_records.jsonl"
    bytes_by_split = {"train": 0, "val": 0, "test": 0}
    counts_by_split = {"train": 0, "val": 0, "test": 0}

    def append_record(record: dict[str, object]) -> None:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def process_split(
        split: str,
        samples: Sequence[PairedImageSample],
        patches_per_image: int,
        crop_mode: str,
    ) -> None:
        for sample_index, sample in enumerate(samples, start=1):
            with Image.open(sample.input_path) as input_image:
                input_image = input_image.convert("RGB")
                input_width, input_height = input_image.size
                with Image.open(sample.target_path) as target_image:
                    target_image = target_image.convert("RGB")
                    target_width, target_height = target_image.size
                    width = min(input_width, target_width)
                    height = min(input_height, target_height)
                    base_input = input_image.crop((0, 0, width, height))
                    base_target = target_image.crop((0, 0, width, height))

                    if crop_mode == "center":
                        boxes = [center_crop_box(width, height, args.patch_size)]
                    else:
                        key = f"{args.seed}:{split}:{sample.name}"
                        boxes = train_crop_boxes(width, height, args.patch_size, patches_per_image, key)

                    for patch_index, box in enumerate(boxes):
                        left, top, right, bottom = box
                        input_patch = base_input.crop((left, top, right, bottom))
                        target_patch = base_target.crop((left, top, right, bottom))

                        stem = Path(sample.name).stem
                        if split == "train":
                            cache_name = f"{stem}__p{patch_index:03d}.png"
                        else:
                            cache_name = f"{stem}.png"

                        input_bytes = write_png(
                            input_patch,
                            output_root / split / "input" / cache_name,
                            compress_level=args.png_compress_level,
                        )
                        target_bytes = write_png(
                            target_patch,
                            output_root / split / "gt" / cache_name,
                            compress_level=args.png_compress_level,
                        )

                        counts_by_split[split] += 1
                        bytes_by_split[split] += input_bytes + target_bytes
                        append_record(
                            {
                                "split": split,
                                "source_name": sample.name,
                                "cache_name": cache_name,
                                "box": [left, top, right, bottom],
                                "width": width,
                                "height": height,
                            }
                        )

            if sample_index % args.progress_interval == 0 or sample_index == len(samples):
                print(f"[{split}] processed {sample_index}/{len(samples)} source images; cached_pairs={counts_by_split[split]}")

    process_split("train", effective_train_pairs, args.train_patches_per_image, crop_mode="train")
    process_split("val", val_pairs, patches_per_image=1, crop_mode="center")
    process_split("test", test_pairs, patches_per_image=1, crop_mode="center")

    summary = {
        "cache_root": str(output_root),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "png_compress_level": args.png_compress_level,
        "train_offset": args.train_offset,
        "val_limit": args.val_limit,
        "train_patches_per_image": args.train_patches_per_image,
        "train_source_limit": args.train_source_limit,
        "val_source_limit": args.val_source_limit,
        "test_source_limit": args.test_source_limit,
        "source_counts": {
            "train_pairs_total": len(train_pairs),
            "train_pairs_effective": len(effective_train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
        },
        "cache_counts": counts_by_split,
        "cache_bytes": bytes_by_split,
    }
    summary_path = output_root / "cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cache of fixed paired image patches for V1/T01 experiments.")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--test-target", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--train-offset", type=int, default=100)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--train-patches-per-image", type=int, default=4)
    parser.add_argument("--train-source-limit", type=int, default=0)
    parser.add_argument("--val-source-limit", type=int, default=0)
    parser.add_argument("--test-source-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--png-compress-level", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_cache(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
