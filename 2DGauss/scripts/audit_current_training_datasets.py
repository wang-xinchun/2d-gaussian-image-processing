#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
from pathlib import Path


DATASET_ROOT = Path(os.environ.get("TWODGAUSS_DATASET_ROOT", "/home/ldj/wxc/baiduwangpan/2D高斯数据集"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "datasets" / "current_training_ready"
SPLIT_ROOT = OUT_DIR / "views"
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in FRAME_EXTENSIONS],
        key=lambda p: p.name,
    )


def pair_count(input_dir: Path, target_dir: Path) -> tuple[int, int, int, list[str]]:
    inputs = image_files(input_dir)
    targets = image_files(target_dir)
    by_name = {p.name: p for p in targets}
    by_stem = {p.stem: p for p in targets}
    pairs = []
    for inp in inputs:
        if inp.name in by_name or inp.stem in by_stem:
            pairs.append(inp.name)
    return len(pairs), len(inputs), len(targets), pairs


def rel(path: Path) -> str:
    return str(path)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, start=dst.parent))


def build_4kdehaze_split() -> dict[str, dict[str, str | int]]:
    input_dir = DATASET_ROOT / "4KDehaze" / "4KDehazing" / "inputs"
    target_dir = DATASET_ROOT / "4KDehaze" / "4KDehazing" / "groundtrues"
    pair_total, _, _, pair_names = pair_count(input_dir, target_dir)
    if pair_total == 0:
        return {}

    split_dir = SPLIT_ROOT / "4KDehaze_local_90_5_5"
    reset_dir(split_dir)
    train_n = int(pair_total * 0.90)
    val_n = int(pair_total * 0.05)
    splits = {
        "train": pair_names[:train_n],
        "val": pair_names[train_n : train_n + val_n],
        "test": pair_names[train_n + val_n :],
    }
    result: dict[str, dict[str, str | int]] = {}
    for split, names in splits.items():
        for name in names:
            symlink_file(input_dir / name, split_dir / split / "input" / name)
            symlink_file(target_dir / name, split_dir / split / "gt" / name)
        result[split] = {
            "input_dir": rel(split_dir / split / "input"),
            "target_dir": rel(split_dir / split / "gt"),
            "pairs": len(names),
        }
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    def add_pair(
        dataset: str,
        task: str,
        split: str,
        input_dir: Path,
        target_dir: Path,
        training_usable: str,
        note: str,
    ) -> None:
        pairs, inputs, targets, _ = pair_count(input_dir, target_dir)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "split": split,
                "pairs": pairs,
                "input_images": inputs,
                "target_images": targets,
                "training_usable": training_usable,
                "input_dir": rel(input_dir),
                "target_dir": rel(target_dir),
                "note": note,
            }
        )

    add_pair(
        "LOL-v1",
        "low_light",
        "train",
        DATASET_ROOT / "LOL-v1" / "our485" / "input",
        DATASET_ROOT / "LOL-v1" / "our485" / "gt",
        "yes_t01",
        "official train split",
    )
    add_pair(
        "LOL-v1",
        "low_light",
        "test",
        DATASET_ROOT / "LOL-v1" / "eval15" / "input",
        DATASET_ROOT / "LOL-v1" / "eval15" / "gt",
        "eval_only",
        "official eval split",
    )
    add_pair(
        "LOL-v2",
        "low_light",
        "train",
        DATASET_ROOT / "LOL-v2" / "Train" / "input",
        DATASET_ROOT / "LOL-v2" / "Train" / "gt",
        "yes_t01",
        "official train split",
    )
    add_pair(
        "LOL-v2",
        "low_light",
        "test",
        DATASET_ROOT / "LOL-v2" / "Test" / "input",
        DATASET_ROOT / "LOL-v2" / "Test" / "gt",
        "eval_only",
        "official test split",
    )
    for name in ("UHD-LL", "UHD-LL_down4"):
        add_pair(
            name,
            "low_light",
            "train",
            DATASET_ROOT / name / "training_set" / "input",
            DATASET_ROOT / name / "training_set" / "gt",
            "yes_t01",
            "official training_set",
        )
        add_pair(
            name,
            "low_light",
            "test",
            DATASET_ROOT / name / "testing_set" / "input",
            DATASET_ROOT / name / "testing_set" / "gt",
            "eval_only",
            "official testing_set",
        )
    add_pair(
        "4K-Rain13k",
        "deraining",
        "train",
        DATASET_ROOT / "4K-Rain13k" / "train" / "input",
        DATASET_ROOT / "4K-Rain13k" / "train" / "target",
        "yes_t01",
        "official train split",
    )
    add_pair(
        "4K-Rain13k",
        "deraining",
        "test",
        DATASET_ROOT / "4K-Rain13k" / "test" / "input",
        DATASET_ROOT / "4K-Rain13k" / "test" / "target",
        "eval_only",
        "official test split",
    )
    add_pair(
        "4KDehaze",
        "dehazing",
        "all_raw",
        DATASET_ROOT / "4KDehaze" / "4KDehazing" / "inputs",
        DATASET_ROOT / "4KDehaze" / "4KDehazing" / "groundtrues",
        "split_required",
        "raw paired files, deterministic local split generated below",
    )

    dehaze_split = build_4kdehaze_split()
    for split, info in dehaze_split.items():
        add_pair(
            "4KDehaze_local_90_5_5",
            "dehazing",
            split,
            Path(str(info["input_dir"])),
            Path(str(info["target_dir"])),
            "yes_t01" if split == "train" else "eval_only",
            "project deterministic local split from filename order",
        )

    sr_dirs = [
        ("UHD4K (SR)", DATASET_ROOT / "UHD4K (SR)" / "4K"),
        ("8K", DATASET_ROOT / "8K"),
    ]
    for dataset, root in sr_dirs:
        for split in ("train", "test"):
            hr = image_files(root / split / "HR")
            for scale in ("LR_X2", "LR_X3", "LR_X3_BD", "LR_X3_DN", "LR_X4", "LR_X8", "LR_X16"):
                lr = image_files(root / split / scale)
                common = len(set(p.name for p in hr) & set(p.name for p in lr))
                rows.append(
                    {
                        "dataset": dataset,
                        "task": "super_resolution",
                        "split": f"{split}_{scale}",
                        "pairs": common,
                        "input_images": len(lr),
                        "target_images": len(hr),
                        "training_usable": "sr_template_required",
                        "input_dir": rel(root / split / scale),
                        "target_dir": rel(root / split / "HR"),
                        "note": "HR/LR names aligned; current T01 paired-restoration template is not SR-specific",
                    }
                )

    tsv = OUT_DIR / "training_dataset_audit.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        fields = [
            "dataset",
            "task",
            "split",
            "pairs",
            "input_images",
            "target_images",
            "training_usable",
            "input_dir",
            "target_dir",
            "note",
        ]
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")

    t01_train_pairs = sum(
        int(row["pairs"])
        for row in rows
        if row["split"] == "train" and row["training_usable"] == "yes_t01"
    )
    t01_train_datasets = sorted(
        {str(row["dataset"]) for row in rows if row["split"] == "train" and row["training_usable"] == "yes_t01"}
    )
    sr_train_pairs = sum(
        int(row["pairs"])
        for row in rows
        if str(row["split"]).startswith("train_") and row["training_usable"] == "sr_template_required"
    )

    md = OUT_DIR / "TRAINING_DATASETS_READY.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Current Training Dataset Audit\n\n")
        f.write(f"Dataset root: `{DATASET_ROOT}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- T01 paired-restoration trainable datasets: `{len(t01_train_datasets)}`\n")
        f.write(f"- T01 paired-restoration train pairs: `{t01_train_pairs}`\n")
        f.write(f"- SR aligned train pairs across all LR variants: `{sr_train_pairs}` (requires SR-specific template)\n")
        f.write("- Missing from current Baidu copy: `UHD_haze` and `UHD_deblur` historical datasets are not present under this root.\n\n")
        f.write("## T01-Ready Train Splits\n\n")
        f.write("| Dataset | Task | Train Pairs | Input | Target |\n")
        f.write("| --- | --- | ---: | --- | --- |\n")
        for row in rows:
            if row["split"] == "train" and row["training_usable"] == "yes_t01":
                f.write(
                    f"| `{row['dataset']}` | `{row['task']}` | {row['pairs']} | `{row['input_dir']}` | `{row['target_dir']}` |\n"
                )
        f.write("\n## Full Audit TSV\n\n")
        f.write(f"- `{tsv}`\n")
        f.write(f"- 4KDehaze local split root: `{SPLIT_ROOT / '4KDehaze_local_90_5_5'}`\n")

    print(f"wrote {tsv}")
    print(f"wrote {md}")
    print(f"t01_train_datasets={len(t01_train_datasets)}")
    print(f"t01_train_pairs={t01_train_pairs}")
    print(f"sr_train_pairs_all_variants={sr_train_pairs}")


if __name__ == "__main__":
    main()
