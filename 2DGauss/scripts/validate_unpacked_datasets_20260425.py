#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASET_ROOT = Path("/root/autodl-tmp/2D_Gaussian_Dataset")


def image_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def all_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file())


def rel(path: Path) -> str:
    return str(path.relative_to(DATASET_ROOT))


def sample_paths(paths: list[Path], count: int = 3) -> list[Path]:
    if len(paths) <= count:
        return paths
    idxs = sorted({0, len(paths) // 2, len(paths) - 1})
    return [paths[i] for i in idxs]


def inspect_samples(paths: Iterable[Path]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for path in paths:
        entry: dict[str, object] = {"path": rel(path), "ok": False}
        try:
            with Image.open(path) as img:
                entry.update(
                    {
                        "ok": True,
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format,
                    }
                )
        except Exception as exc:  # pragma: no cover - diagnostic script
            entry["error"] = repr(exc)
        samples.append(entry)
    return samples


def dir_report(path: Path) -> dict[str, object]:
    files = all_files(path)
    images = image_files(path)
    non_images = [p for p in files if p.suffix.lower() not in IMAGE_SUFFIXES]
    return {
        "path": str(path),
        "file_count": len(files),
        "image_count": len(images),
        "non_image_count": len(non_images),
        "non_image_examples": [rel(p) for p in non_images[:20]],
        "samples": inspect_samples(sample_paths(images)),
    }


def pair_report(left: Path, right: Path) -> dict[str, object]:
    left_names = {p.name for p in image_files(left)}
    right_names = {p.name for p in image_files(right)}
    only_left = sorted(left_names - right_names)
    only_right = sorted(right_names - left_names)
    return {
        "left": str(left),
        "right": str(right),
        "left_image_count": len(left_names),
        "right_image_count": len(right_names),
        "matched_count": len(left_names & right_names),
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left_examples": only_left[:20],
        "only_right_examples": only_right[:20],
        "pair_ok": not only_left and not only_right,
    }


def expected_check(actual: int, expected: int | None) -> dict[str, object]:
    return {
        "expected": expected,
        "actual": actual,
        "matches_expected": expected is None or actual == expected,
    }


def validate_uhd_haze() -> dict[str, object]:
    root = DATASET_ROOT / "UHD_haze"
    train_input = root / "train" / "input"
    train_gt = root / "train" / "gt"
    test_input = root / "test" / "input"
    test_gt = root / "test" / "gt"
    train_pair = pair_report(train_input, train_gt)
    test_pair = pair_report(test_input, test_gt)
    return {
        "name": "UHD-Haze",
        "root": str(root),
        "task": "uhd_dehazing",
        "paper_expected": {"train_pairs": 2290, "test_pairs": 230},
        "directories": {
            "train/input": dir_report(train_input),
            "train/gt": dir_report(train_gt),
            "test/input": dir_report(test_input),
            "test/gt": dir_report(test_gt),
        },
        "pairing": {"train": train_pair, "test": test_pair},
        "expected_checks": {
            "train_pairs": expected_check(train_pair["matched_count"], 2290),
            "test_pairs": expected_check(test_pair["matched_count"], 230),
        },
        "acceptance_note": (
            "train split matches paper inventory; local test split has 231 paired images, "
            "one more than the paper inventory count of 230, but input/gt pairing is complete."
        ),
    }


def validate_uhd_deblur() -> dict[str, object]:
    root = DATASET_ROOT / "UHD_deblur"
    train_input = root / "train" / "input_new"
    train_gt = root / "train" / "gt_new"
    test_input = root / "test" / "input300"
    test_gt = root / "test" / "gt300"
    train_pair = pair_report(train_input, train_gt)
    test_pair = pair_report(test_input, test_gt)
    return {
        "name": "UHD-Blur",
        "root": str(root),
        "task": "uhd_deblurring",
        "paper_expected": {"train_pairs": 1964, "test_pairs": 300},
        "directories": {
            "train/input_new": dir_report(train_input),
            "train/gt_new": dir_report(train_gt),
            "test/input300": dir_report(test_input),
            "test/gt300": dir_report(test_gt),
        },
        "pairing": {"train": train_pair, "test": test_pair},
        "expected_checks": {
            "train_pairs": expected_check(train_pair["matched_count"], 1964),
            "test_pairs": expected_check(test_pair["matched_count"], 300),
        },
        "acceptance_note": (
            "image counts and pair names match the paper inventory; extra files in gt300 are "
            "non-image metadata files and are ignored by image-count acceptance."
        ),
    }


def validate_4k() -> dict[str, object]:
    root = DATASET_ROOT / "4K"
    lr_dirs = ["LR_X2", "LR_X3", "LR_X3_BD", "LR_X3_DN", "LR_X4", "LR_X8", "LR_X16"]
    directories: dict[str, object] = {}
    pairing: dict[str, object] = {}
    for split in ["train", "test"]:
        hr = root / split / "HR"
        directories[f"{split}/HR"] = dir_report(hr)
        split_pairs = {}
        for lr_dir in lr_dirs:
            lr = root / split / lr_dir
            directories[f"{split}/{lr_dir}"] = dir_report(lr)
            split_pairs[lr_dir] = pair_report(hr, lr)
        pairing[split] = split_pairs
    train_hr_count = len(image_files(root / "train" / "HR"))
    test_hr_count = len(image_files(root / "test" / "HR"))
    all_pair_ok = all(
        report["pair_ok"]
        for split_pairs in pairing.values()
        for report in split_pairs.values()
    )
    return {
        "name": "4K",
        "root": str(root),
        "task": "super_resolution",
        "paper_expected": None,
        "directories": directories,
        "pairing": pairing,
        "summary_counts": {
            "train_hr": train_hr_count,
            "test_hr": test_hr_count,
            "train_per_lr": train_hr_count,
            "test_per_lr": test_hr_count,
            "total_images": sum(
                report["image_count"] for report in directories.values()
            ),
        },
        "all_hr_lr_pairing_ok": all_pair_ok,
        "acceptance_note": (
            "HR and all LR variants have identical filename sets within each split."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "dataset_root": str(DATASET_ROOT),
        "datasets": {
            "UHD-Haze": validate_uhd_haze(),
            "UHD-Blur": validate_uhd_deblur(),
            "4K": validate_4k(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for name, dataset in report["datasets"].items():
        print(f"## {name}")
        if name == "4K":
            print(json.dumps(dataset["summary_counts"], ensure_ascii=False))
            print(f"pairing_ok={dataset['all_hr_lr_pairing_ok']}")
        else:
            for split, pair in dataset["pairing"].items():
                print(
                    f"{split}: matched={pair['matched_count']} "
                    f"left={pair['left_image_count']} right={pair['right_image_count']} "
                    f"pair_ok={pair['pair_ok']}"
                )
            print(json.dumps(dataset["expected_checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
