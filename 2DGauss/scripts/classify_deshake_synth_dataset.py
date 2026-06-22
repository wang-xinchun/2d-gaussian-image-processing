#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a symlink classification index for synthesized deshake data.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.strip() or "unknown"
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", value)
    return value.strip("._") or "unknown"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def infer_source_family(source_video: str, plan_entry: dict[str, Any] | None = None) -> str:
    if plan_entry and plan_entry.get("source_family"):
        return str(plan_entry["source_family"])
    text = source_video.lower()
    name = Path(source_video).name.lower()
    if "bvi-aom" in text:
        return "bvi_aom"
    if name.startswith("netflix_") or "netflix" in text:
        return "netflix"
    if name.endswith(".7z") or "uvg" in text:
        return "uvg"
    return "unknown"


def infer_dataset(source_video: str, source_family: str) -> str:
    path = Path(source_video)
    text = source_video.lower()
    if source_family == "bvi_aom":
        if "bvi-aom_extracted" in text:
            return "bvi_aom_extracted_dense_v1"
        return "bvi_aom_dense_v1"
    if source_family == "netflix":
        if "netflix_completed_sources" in text:
            return "netflix_completed_dense_v1"
        return "netflix_dense_v1"
    if source_family == "uvg":
        return "uvg_dense_v1"
    return safe_name(path.parent.name)


def parse_source_technical(source_video: str, motion: dict[str, Any] | None = None) -> dict[str, str]:
    name = Path(source_video).name
    suffix = Path(source_video).suffix.lower().lstrip(".") or "unknown"
    width = None
    height = None
    fps = None
    bit_depth = None

    match = re.search(r"(?P<w>\d{3,5})x(?P<h>\d{3,5})", name)
    if match:
        width = match.group("w")
        height = match.group("h")
    elif motion:
        width = str(motion.get("output_width", "unknown"))
        height = str(motion.get("output_height", "unknown"))

    match = re.search(r"(?P<fps>\d+(?:\.\d+)?)fps", name, flags=re.IGNORECASE)
    if match:
        fps = match.group("fps")
        if "." in fps:
            fps = fps.rstrip("0").rstrip(".")

    match = re.search(r"(?P<bit>8|10|12)bit", name, flags=re.IGNORECASE)
    if match:
        bit_depth = match.group("bit")

    return {
        "container": suffix,
        "resolution": f"{width or 'unknown'}x{height or 'unknown'}",
        "fps": fps or "unknown",
        "bit_depth": bit_depth or "unknown",
    }


def load_plan_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plan_path = manifest.get("clip_plan")
    if not plan_path:
        return {}
    path = Path(plan_path)
    if not path.exists():
        return {}
    plan = load_json(path)
    return {entry["clip_id"]: entry for entry in plan.get("clips", [])}


def manifest_clip_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("clips"):
        return list(manifest["clips"])
    if manifest.get("pass_clips"):
        rows = list(manifest["pass_clips"])
        rows.extend(manifest.get("failed_source_clips_excluded_from_paired_frames", []))
        return rows
    return []


def split_for_clip(row: dict[str, Any], dataset_root: Path) -> str:
    if row.get("split") in {"train", "val"}:
        return str(row["split"])
    clip_id = row["clip_id"]
    for split in ("train", "val"):
        pattern = dataset_root / "paired_frames" / split / "input" / f"{clip_id}__*.npy"
        if any(pattern.parent.glob(pattern.name)):
            return split
    return "unknown"


def build_clip_record(row: dict[str, Any], dataset_root: Path, plan_entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    clip_id = row["clip_id"]
    clip_dir = Path(row.get("clip_dir") or dataset_root / "clips" / clip_id)
    motion_path = clip_dir / "motion.json"
    motion = load_json(motion_path) if motion_path.exists() else {}
    source_video = str(row.get("source_video") or motion.get("source_video") or "")
    plan_entry = plan_entries.get(clip_id)
    source_family = infer_source_family(source_video, plan_entry)
    dataset = infer_dataset(source_video, source_family)
    technical = parse_source_technical(source_video, motion)
    frame_count = len(list((clip_dir / "stable").glob("*.npy"))) if clip_dir.exists() else 0
    motion_preset = str(row.get("motion_preset") or motion.get("motion_preset") or "unknown")
    return {
        "clip_key": clip_id,
        "split": split_for_clip(row, dataset_root),
        "frame_count": frame_count,
        "source_family": source_family,
        "dataset": dataset,
        "motion_preset": motion_preset,
        "source_name": Path(source_video).name,
        "container": technical["container"],
        "resolution": technical["resolution"],
        "fps": technical["fps"],
        "bit_depth": technical["bit_depth"],
        "source_video": source_video,
        "status": row.get("status", "unknown"),
        "clip_dir": str(clip_dir),
        "start_frame": row.get("start_frame", motion.get("clip_start_frame", "")),
        "frame_stride": row.get("frame_stride", motion.get("frame_stride", "")),
        "clip_length": row.get("clip_length", motion.get("clip_length", "")),
        "valid_area_ratio_min": row.get("valid_area_ratio_min", motion.get("valid_area_ratio_min", "")),
        "valid_area_ratio_mean": row.get("valid_area_ratio_mean", motion.get("valid_area_ratio_mean", "")),
        "error": row.get("error", ""),
    }


def pair_rows_for_clip(record: dict[str, Any]) -> list[dict[str, str]]:
    if record["status"] != "pass" or record["split"] not in {"train", "val"}:
        return []
    clip_dir = Path(record["clip_dir"])
    rows: list[dict[str, str]] = []
    for stable_path in sorted((clip_dir / "stable").glob("*.npy")):
        shaky_path = clip_dir / "shaky" / stable_path.name
        pair_name = f"{record['clip_key']}__{stable_path.name}"
        rows.append(
            {
                "pair_key": pair_name,
                "clip_key": record["clip_key"],
                "split": record["split"],
                "frame_index": stable_path.stem,
                "input": str(shaky_path),
                "target": str(stable_path),
                "source_family": record["source_family"],
                "dataset": record["dataset"],
                "motion_preset": record["motion_preset"],
                "source_video": record["source_video"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_pair_list(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("input\ttarget\tclip_key\tframe_index\tsource_family\tsource_dataset\tmotion_preset\n")
        for row in rows:
            handle.write(
                f"{row['input']}\t{row['target']}\t{row['clip_key']}\t{row['frame_index']}\t"
                f"{row['source_family']}\t{row['dataset']}\t{row['motion_preset']}\n"
            )


def add_clip_links(output_root: Path, record: dict[str, Any]) -> None:
    clip_dir = Path(record["clip_dir"])
    if not clip_dir.exists():
        return
    clip_key = record["clip_key"]
    categories = {
        "by_split": [record["split"]],
        "by_source_family": [record["source_family"]],
        "by_source_dataset": [record["dataset"]],
        "by_motion_preset": [record["motion_preset"]],
        "by_source_family_motion": [f"{record['source_family']}/{record['motion_preset']}"],
        "by_container": [record["container"]],
        "by_resolution": [record["resolution"]],
        "by_fps": [record["fps"]],
        "by_bit_depth": [record["bit_depth"]],
        "clips": [f"{record['source_family']}/{record['motion_preset']}/{record['split']}"],
    }
    for root_name, values in categories.items():
        for value in values:
            parts = [safe_name(part) for part in value.split("/") if part]
            link_or_copy(clip_dir.resolve(), output_root / root_name / Path(*parts) / clip_key)


def ensure_no_broken_symlinks(output_root: Path) -> list[str]:
    broken: list[str] = []
    for path in output_root.rglob("*"):
        if path.is_symlink() and not path.exists():
            broken.append(str(path))
    return broken


def build_readme(
    output_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    classification_manifest: dict[str, Any],
) -> None:
    clip_counts_json = json.dumps(classification_manifest["clip_counts"], ensure_ascii=False, indent=2, sort_keys=True)
    frame_counts_json = json.dumps(classification_manifest["frame_counts"], ensure_ascii=False, indent=2, sort_keys=True)
    text = f"""# {output_root.name}

Classification index for `{dataset_root}`. This directory is made of symlinks and metadata tables; it does not duplicate the `.npy` frame payloads.

## Main Training View

Use the base synthesized data root for training:

- Data root: `{dataset_root}`
- Manifest: `{manifest_path}`
- Train pairs: `{classification_manifest['pair_counts'].get('train', 0)}`
- Val pairs: `{classification_manifest['pair_counts'].get('val', 0)}`

## Classification Directories

- `by_source_family/`: source family groups
- `by_source_dataset/`: source package groups
- `by_motion_preset/`: synthetic motion category
- `by_source_family_motion/`: family and motion cross-category
- `by_container/`, `by_resolution/`, `by_fps/`, `by_bit_depth/`: source technical categories
- `clips/`: clip-level symlinks grouped by source, motion, and split
- `tables/`: CSV tables for clips, frames, and failed clips
- `lists/`: train/val TSV pair lists

## Clip Counts

```json
{clip_counts_json}
```

## Frame Counts

```json
{frame_counts_json}
```
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = (args.manifest or dataset_root / "manifest.json").expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists; use --force to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    manifest = load_json(manifest_path)
    plan_entries = load_plan_entries(manifest)
    records = [build_clip_record(row, dataset_root, plan_entries) for row in manifest_clip_rows(manifest)]
    pass_records = [record for record in records if record["status"] == "pass"]
    failed_records = [record for record in records if record["status"] != "pass"]
    frame_rows: list[dict[str, str]] = []
    for record in pass_records:
        add_clip_links(output_root, record)
        frame_rows.extend(pair_rows_for_clip(record))

    train_rows = [row for row in frame_rows if row["split"] == "train"]
    val_rows = [row for row in frame_rows if row["split"] == "val"]

    clip_fieldnames = [
        "clip_key",
        "split",
        "frame_count",
        "source_family",
        "dataset",
        "motion_preset",
        "source_name",
        "container",
        "resolution",
        "fps",
        "bit_depth",
        "source_video",
        "status",
        "start_frame",
        "frame_stride",
        "clip_length",
        "valid_area_ratio_min",
        "valid_area_ratio_mean",
    ]
    frame_fieldnames = [
        "pair_key",
        "clip_key",
        "split",
        "frame_index",
        "input",
        "target",
        "source_family",
        "dataset",
        "motion_preset",
        "source_video",
    ]
    failed_fieldnames = [
        "clip_key",
        "split",
        "source_family",
        "dataset",
        "motion_preset",
        "source_name",
        "source_video",
        "status",
        "error",
    ]

    write_csv(output_root / "tables" / "clips.csv", pass_records, clip_fieldnames)
    write_csv(output_root / "tables" / "frames.csv", frame_rows, frame_fieldnames)
    write_csv(output_root / "tables" / "failed_source_candidates.csv", failed_records, failed_fieldnames)
    write_pair_list(output_root / "lists" / "train_pairs.tsv", train_rows)
    write_pair_list(output_root / "lists" / "val_pairs.tsv", val_rows)

    clip_counts = {
        "split": Counter(record["split"] for record in pass_records),
        "source_family": Counter(record["source_family"] for record in pass_records),
        "source_dataset": Counter(record["dataset"] for record in pass_records),
        "motion_preset": Counter(record["motion_preset"] for record in pass_records),
        "container": Counter(record["container"] for record in pass_records),
        "resolution": Counter(record["resolution"] for record in pass_records),
        "fps": Counter(record["fps"] for record in pass_records),
        "bit_depth": Counter(record["bit_depth"] for record in pass_records),
    }
    frame_counts = {
        "source_family": Counter(row["source_family"] for row in frame_rows),
        "source_dataset": Counter(row["dataset"] for row in frame_rows),
        "motion_preset": Counter(row["motion_preset"] for row in frame_rows),
        "split_source_family": Counter(f"{row['split']}/{row['source_family']}" for row in frame_rows),
        "split_motion_preset": Counter(f"{row['split']}/{row['motion_preset']}" for row in frame_rows),
    }

    source_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in pass_records:
        source_to_splits[record["source_video"]].add(record["split"])
    cross_split_sources = sorted(source for source, splits in source_to_splits.items() if len(splits) > 1)
    broken_symlinks = ensure_no_broken_symlinks(output_root)

    classification_manifest = {
        "base_dataset_root": str(dataset_root),
        "base_manifest": str(manifest_path),
        "classification_root": str(output_root),
        "directories": {
            name: str(output_root / name)
            for name in [
                "by_split",
                "by_source_family",
                "by_source_dataset",
                "by_motion_preset",
                "by_source_family_motion",
                "by_container",
                "by_resolution",
                "by_fps",
                "by_bit_depth",
                "clips",
                "tables",
                "lists",
            ]
        },
        "pair_counts": {"train": len(train_rows), "val": len(val_rows)},
        "clip_counts": {key: dict(counter) for key, counter in clip_counts.items()},
        "frame_counts": {key: dict(counter) for key, counter in frame_counts.items()},
        "failed_source_candidates_excluded": failed_records,
        "validation": {
            "pass_clips": len(pass_records),
            "failed_clips": len(failed_records),
            "frame_rows": len(frame_rows),
            "broken_symlinks": broken_symlinks,
            "source_split_leak_count": len(cross_split_sources),
            "source_split_leak_sources": cross_split_sources,
        },
        "training_view": manifest.get("training_view", {}),
    }
    write_json(output_root / "classification_manifest.json", classification_manifest)
    build_readme(output_root, dataset_root, manifest_path, classification_manifest)

    if broken_symlinks:
        raise RuntimeError(f"broken symlinks found: {len(broken_symlinks)}")
    if cross_split_sources:
        raise RuntimeError(f"sources cross train/val split: {len(cross_split_sources)}")
    print(
        json.dumps(
            {
                "classification_root": str(output_root),
                "clips": len(pass_records),
                "failed": len(failed_records),
                "train_pairs": len(train_rows),
                "val_pairs": len(val_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
