#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".ts",
    ".webm",
    ".y4m",
    ".yuv",
}
FRAME_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".npy"}
REQUIRED_TASK_KEYWORDS = ("deshake", "stabil", "jitter", "shake", "video")
BANNED_TASK_KEYWORDS = ("uhd_haze", "uhdhaze", "dehaze", "hazy", "haze")
BANNED_PATH_HINTS = ("uhd_haze", "uhdhaze", "haze_patch", "任务四")
DEFAULT_MIN_DATA_BYTES = 300 * 1024**3
REQUIRED_DATASET_PATH_KEYS = ("train_input", "train_target", "val_input", "val_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight gate for launching the real UHD video deshaking DDP run "
            "on physical GPUs 0 and 3."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--physical-gpus", default="0,3")
    parser.add_argument(
        "--allow-single-gpu0-smoke",
        action="store_true",
        help="Permit the explicit physical GPU0-only deshake smoke profile.",
    )
    parser.add_argument(
        "--allow-single-gpu3-train",
        action="store_true",
        help="Permit an explicit physical GPU3-only deshake train profile when GPU0 is busy.",
    )
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument(
        "--gpu-utilization-policy",
        choices=("block", "warn"),
        default="block",
        help=(
            "Whether high GPU utilization blocks launch or is recorded as a warning. "
            "Use warn only for an explicit shared-GPU smoke with tight memory caps."
        ),
    )
    parser.add_argument("--max-per-rank-batch-size", type=int, default=2)
    parser.add_argument("--max-global-batch-size", type=int, default=4)
    parser.add_argument("--max-num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-crop-size",
        type=int,
        default=0,
        help="Optional safety cap for dataset.train_crop_size; 0 disables the check.",
    )
    parser.add_argument(
        "--max-val-crop-size",
        type=int,
        default=0,
        help="Optional safety cap for dataset.val_crop_size; 0 disables the check.",
    )
    parser.add_argument("--min-data-bytes", type=int, default=DEFAULT_MIN_DATA_BYTES)
    parser.add_argument("--min-video-files", type=int, default=1)
    parser.add_argument("--min-frame-files", type=int, default=0)
    parser.add_argument("--min-train-pairs", type=int, default=1)
    parser.add_argument("--min-val-pairs", type=int, default=1)
    parser.add_argument("--max-scan-files", type=int, default=500000)
    parser.add_argument("--min-file-age-seconds", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def query_gpu(gpu_id: str) -> dict[str, Any]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            gpu_id,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"unexpected nvidia-smi GPU row for GPU{gpu_id}: {result.stdout!r}")
    return {
        "index": int(parts[0]),
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
    }


def query_compute_processes() -> list[dict[str, Any]]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    processes: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        processes.append(
            {
                "gpu_bus_id": parts[0],
                "pid": int(parts[1]) if parts[1].isdigit() else parts[1],
                "process_name": parts[2],
                "used_memory_mib": int(parts[3]) if parts[3].isdigit() else parts[3],
            }
        )
    return processes


def collect_dataset_evidence(
    root: Path,
    max_scan_files: int,
    min_file_age_seconds: int,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "bytes_scanned": 0,
        "files_scanned": 0,
        "video_files": 0,
        "frame_files": 0,
        "scan_truncated": False,
        "largest_files": [],
        "newest_files": [],
        "recent_files": [],
        "recent_file_count": 0,
        "min_file_age_seconds": min_file_age_seconds,
        "latest_mtime_epoch": None,
        "latest_mtime_age_seconds": None,
        "elapsed_seconds": 0.0,
    }
    if not root.exists() or not root.is_dir():
        return evidence

    start = time.monotonic()
    now_epoch = time.time()
    largest: list[tuple[int, str]] = []
    newest: list[tuple[float, float, int, str]] = []
    recent: list[tuple[float, float, int, str]] = []

    def finalize_evidence() -> None:
        if newest:
            latest_mtime = newest[0][0]
            latest_age = max(0.0, now_epoch - latest_mtime)
            evidence["latest_mtime_epoch"] = round(latest_mtime, 3)
            evidence["latest_mtime_age_seconds"] = round(latest_age, 3)
        evidence["largest_files"] = [{"bytes": size, "path": item_path} for size, item_path in largest]
        evidence["newest_files"] = [
            {
                "age_seconds": round(age_seconds, 3),
                "mtime_epoch": round(mtime_epoch, 3),
                "bytes": size,
                "path": item_path,
            }
            for mtime_epoch, age_seconds, size, item_path in newest
        ]
        evidence["recent_files"] = [
            {
                "age_seconds": round(age_seconds, 3),
                "mtime_epoch": round(mtime_epoch, 3),
                "bytes": size,
                "path": item_path,
            }
            for age_seconds, mtime_epoch, size, item_path in recent
        ]

    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(current_root) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            suffix = path.suffix.lower()
            age_seconds = max(0.0, now_epoch - stat.st_mtime)
            evidence["files_scanned"] += 1
            evidence["bytes_scanned"] += stat.st_size
            if suffix in VIDEO_EXTENSIONS:
                evidence["video_files"] += 1
            if suffix in FRAME_EXTENSIONS:
                evidence["frame_files"] += 1
            largest.append((stat.st_size, str(path)))
            largest = sorted(largest, reverse=True)[:10]
            newest.append((stat.st_mtime, age_seconds, stat.st_size, str(path)))
            newest = sorted(newest, key=lambda item: item[0], reverse=True)[:10]
            if min_file_age_seconds > 0 and age_seconds < min_file_age_seconds:
                evidence["recent_file_count"] += 1
                recent.append((age_seconds, stat.st_mtime, stat.st_size, str(path)))
                recent = sorted(recent, key=lambda item: item[0])[:10]
            if evidence["files_scanned"] >= max_scan_files:
                evidence["scan_truncated"] = True
                evidence["elapsed_seconds"] = round(time.monotonic() - start, 3)
                finalize_evidence()
                return evidence
    evidence["elapsed_seconds"] = round(time.monotonic() - start, 3)
    finalize_evidence()
    return evidence


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS],
        key=lambda item: item.name,
    )


def paired_count(input_dir: Path, target_dir: Path) -> int:
    inputs = image_files(input_dir)
    targets = image_files(target_dir)
    targets_by_name = {path.name for path in targets}
    targets_by_stem = {path.stem for path in targets}
    return sum(1 for path in inputs if path.name in targets_by_name or path.stem in targets_by_stem)


def collect_config_evidence(
    config_path: Path | None,
    data_root: Path,
    *,
    min_train_pairs: int,
    min_val_pairs: int,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": str(config_path) if config_path else "",
        "provided": config_path is not None,
        "exists": False,
        "parse_ok": False,
        "contains_required_task_keyword": False,
        "contains_banned_task_keyword": False,
        "dataset_paths": {},
        "pair_counts": {},
        "training": {},
    }
    if config_path is None:
        return evidence
    evidence["exists"] = config_path.exists()
    if not config_path.exists():
        return evidence
    try:
        with config_path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = str(exc)
        return evidence
    evidence["parse_ok"] = True
    text = f"{config_path} {flatten_text(parsed)}".lower()
    evidence["contains_required_task_keyword"] = any(keyword in text for keyword in REQUIRED_TASK_KEYWORDS)
    evidence["contains_banned_task_keyword"] = any(keyword in text for keyword in BANNED_TASK_KEYWORDS)
    training_cfg = parsed.get("training", {})
    dataset_cfg = parsed.get("dataset", {})
    if isinstance(training_cfg, dict):
        raw_batch_size = training_cfg.get("batch_size")
        batch_size = int_or_none(raw_batch_size)
        raw_max_steps = training_cfg.get("max_steps")
        evidence["training"] = {
            "batch_size_provided": raw_batch_size is not None,
            "batch_size": batch_size,
            "batch_size_raw": raw_batch_size,
            "max_steps": int_or_none(raw_max_steps),
        }
    else:
        evidence["training"] = {
            "batch_size_provided": False,
            "batch_size": None,
            "batch_size_raw": None,
            "training_error": "[training] section is missing or invalid",
        }
    dataset_cfg = parsed.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        evidence["dataset_error"] = "[dataset] section is missing or invalid"
        return evidence
    raw_num_workers = dataset_cfg.get("num_workers", 2)
    evidence["training"]["num_workers"] = int_or_none(raw_num_workers)
    evidence["training"]["num_workers_raw"] = raw_num_workers
    for crop_key in ("train_crop_size", "val_crop_size"):
        raw_crop_size = dataset_cfg.get(crop_key)
        evidence["training"][crop_key] = int_or_none(raw_crop_size)
        evidence["training"][f"{crop_key}_raw"] = raw_crop_size

    dataset_paths: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_DATASET_PATH_KEYS:
        raw_path = dataset_cfg.get(key, "")
        if raw_path:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
        else:
            path = Path()
        path_info = {
            "path": str(path) if raw_path else "",
            "provided": bool(raw_path),
            "exists": path.exists() if raw_path else False,
            "is_dir": path.is_dir() if raw_path else False,
            "under_data_root": is_relative_to(path, data_root) if raw_path else False,
            "contains_banned_path_hint": (
                any(hint in str(path).lower() for hint in BANNED_PATH_HINTS) if raw_path else False
            ),
            "frame_files": len(image_files(path)) if raw_path and path.is_dir() else 0,
        }
        dataset_paths[key] = path_info
    evidence["dataset_paths"] = dataset_paths

    all_dataset_paths_provided = all(
        dataset_paths[key]["provided"] for key in REQUIRED_DATASET_PATH_KEYS
    )
    train_input = Path(dataset_paths["train_input"]["path"]) if all_dataset_paths_provided else Path()
    train_target = Path(dataset_paths["train_target"]["path"]) if all_dataset_paths_provided else Path()
    val_input = Path(dataset_paths["val_input"]["path"]) if all_dataset_paths_provided else Path()
    val_target = Path(dataset_paths["val_target"]["path"]) if all_dataset_paths_provided else Path()
    evidence["pair_counts"] = {
        "train": paired_count(train_input, train_target) if all_dataset_paths_provided else 0,
        "val": paired_count(val_input, val_target) if all_dataset_paths_provided else 0,
        "min_train_pairs": min_train_pairs,
        "min_val_pairs": min_val_pairs,
    }
    return evidence


def main() -> int:
    args = parse_args()
    issues: list[str] = []
    warnings: list[str] = []

    physical_gpus = [item.strip() for item in args.physical_gpus.split(",") if item.strip()]
    gpu0_smoke_layout = args.allow_single_gpu0_smoke and physical_gpus == ["0"]
    gpu3_train_layout = args.allow_single_gpu3_train and physical_gpus == ["3"]
    if physical_gpus != ["0", "3"] and not gpu0_smoke_layout and not gpu3_train_layout:
        issues.append(
            "physical_gpus must be exactly 0,3"
            + (
                ", exactly 0 with --allow-single-gpu0-smoke, or exactly 3 with --allow-single-gpu3-train"
                if args.allow_single_gpu0_smoke or args.allow_single_gpu3_train
                else ""
            )
            + f"; got {args.physical_gpus!r}"
        )

    gpu_evidence: list[dict[str, Any]] = []
    for gpu_id in physical_gpus:
        try:
            gpu_row = query_gpu(gpu_id)
            gpu_evidence.append(gpu_row)
            if gpu_row["memory_free_mib"] < args.min_free_mib:
                issues.append(
                    f"GPU{gpu_id} free memory {gpu_row['memory_free_mib']}MiB < "
                    f"{args.min_free_mib}MiB"
                )
            if gpu_row["utilization_percent"] > args.max_gpu_utilization:
                message = (
                    f"GPU{gpu_id} utilization {gpu_row['utilization_percent']}% > "
                    f"{args.max_gpu_utilization}%"
                )
                if args.gpu_utilization_policy == "warn":
                    warnings.append(message)
                else:
                    issues.append(message)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"failed to query GPU{gpu_id}: {exc}")

    data_root = args.data_root.resolve()
    data_evidence = collect_dataset_evidence(
        data_root,
        args.max_scan_files,
        args.min_file_age_seconds,
    )
    root_text = str(data_root).lower()
    if not data_evidence["exists"]:
        issues.append(f"data_root does not exist: {data_root}")
    elif not data_evidence["is_dir"]:
        issues.append(f"data_root is not a directory: {data_root}")
    if any(hint in root_text for hint in BANNED_PATH_HINTS):
        issues.append(f"data_root looks like a known non-deshaking path: {data_root}")
    if data_evidence["bytes_scanned"] < args.min_data_bytes:
        issues.append(
            f"data_root scanned size {data_evidence['bytes_scanned']} bytes < "
            f"{args.min_data_bytes} bytes"
        )
    if data_evidence["video_files"] < args.min_video_files:
        issues.append(
            f"video file count {data_evidence['video_files']} < required {args.min_video_files}"
        )
    if data_evidence["frame_files"] < args.min_frame_files:
        issues.append(
            f"frame file count {data_evidence['frame_files']} < required {args.min_frame_files}"
        )
    if data_evidence["recent_file_count"] > 0:
        issues.append(
            f"data_root has {data_evidence['recent_file_count']} files modified within "
            f"{args.min_file_age_seconds} seconds; latest file age is "
            f"{data_evidence['latest_mtime_age_seconds']} seconds"
        )
    if data_evidence["scan_truncated"]:
        warnings.append(
            "dataset scan hit max_scan_files; raise --max-scan-files for final launch evidence"
        )

    config_evidence = collect_config_evidence(
        args.config.resolve() if args.config else None,
        data_root,
        min_train_pairs=args.min_train_pairs,
        min_val_pairs=args.min_val_pairs,
    )
    if not config_evidence["provided"]:
        issues.append("deshaking-specific config was not provided")
    elif not config_evidence["exists"]:
        issues.append(f"config does not exist: {config_evidence['path']}")
    elif not config_evidence["parse_ok"]:
        issues.append(f"config could not be parsed: {config_evidence.get('error', 'unknown error')}")
    else:
        if not config_evidence["contains_required_task_keyword"]:
            issues.append("config does not contain deshake/stabilization/video task keywords")
        if config_evidence["contains_banned_task_keyword"]:
            issues.append("config contains UHD_haze/dehaze/haze task keywords")
        training_evidence = config_evidence.get("training", {})
        training_error = training_evidence.get("training_error")
        if training_error:
            issues.append(str(training_error))
        batch_size = training_evidence.get("batch_size")
        if not training_evidence.get("batch_size_provided", False):
            issues.append("config training.batch_size is missing")
        elif batch_size is None:
            issues.append(
                f"config training.batch_size is not an integer: "
                f"{training_evidence.get('batch_size_raw')!r}"
            )
        elif batch_size < 1:
            issues.append(f"config training.batch_size must be >= 1; got {batch_size}")
        elif batch_size > args.max_per_rank_batch_size:
            issues.append(
                f"per-rank batch_size {batch_size} > "
                f"safe cap {args.max_per_rank_batch_size}"
            )
        if isinstance(batch_size, int):
            global_batch_size = batch_size * len(physical_gpus)
            training_evidence["world_size"] = len(physical_gpus)
            training_evidence["global_batch_size"] = global_batch_size
            training_evidence["max_per_rank_batch_size"] = args.max_per_rank_batch_size
            training_evidence["max_global_batch_size"] = args.max_global_batch_size
            if global_batch_size > args.max_global_batch_size:
                issues.append(
                    f"global batch_size {global_batch_size} > "
                    f"safe cap {args.max_global_batch_size}"
                )
        num_workers = training_evidence.get("num_workers")
        training_evidence["max_num_workers"] = args.max_num_workers
        if num_workers is None:
            issues.append(
                f"config dataset.num_workers is not an integer: "
                f"{training_evidence.get('num_workers_raw')!r}"
            )
        elif num_workers > args.max_num_workers:
            issues.append(
                f"config dataset.num_workers {num_workers} > "
                f"safe cap {args.max_num_workers}"
            )
        crop_caps = {
            "train_crop_size": args.max_train_crop_size,
            "val_crop_size": args.max_val_crop_size,
        }
        for crop_key, max_crop_size in crop_caps.items():
            training_evidence[f"max_{crop_key}"] = max_crop_size
            if max_crop_size <= 0:
                continue
            crop_size = training_evidence.get(crop_key)
            if crop_size is None:
                issues.append(
                    f"config dataset.{crop_key} is missing or not an integer while "
                    f"safe cap {max_crop_size} is active"
                )
            elif crop_size < 1:
                issues.append(f"config dataset.{crop_key} must be >= 1; got {crop_size}")
            elif crop_size > max_crop_size:
                issues.append(
                    f"config dataset.{crop_key} {crop_size} > safe cap {max_crop_size}"
                )
        dataset_error = config_evidence.get("dataset_error")
        if dataset_error:
            issues.append(str(dataset_error))
        for key, path_info in config_evidence.get("dataset_paths", {}).items():
            if not path_info.get("provided", False):
                issues.append(f"config dataset.{key} is missing")
                continue
            if not path_info.get("exists", False):
                issues.append(f"config dataset.{key} does not exist: {path_info.get('path')}")
            elif not path_info.get("is_dir", False):
                issues.append(f"config dataset.{key} is not a directory: {path_info.get('path')}")
            if not path_info.get("under_data_root", False):
                issues.append(f"config dataset.{key} is outside data_root: {path_info.get('path')}")
            if path_info.get("contains_banned_path_hint", False):
                issues.append(f"config dataset.{key} contains a known non-deshaking path hint")
        pair_counts = config_evidence.get("pair_counts", {})
        if pair_counts.get("train", 0) < args.min_train_pairs:
            issues.append(
                f"train paired frame count {pair_counts.get('train', 0)} < required {args.min_train_pairs}"
            )
        if pair_counts.get("val", 0) < args.min_val_pairs:
            issues.append(
                f"val paired frame count {pair_counts.get('val', 0)} < required {args.min_val_pairs}"
            )

    evidence = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "gpu": {
            "physical_gpus": physical_gpus,
            "allow_single_gpu0_smoke": args.allow_single_gpu0_smoke,
            "allow_single_gpu3_train": args.allow_single_gpu3_train,
            "min_free_mib": args.min_free_mib,
            "max_gpu_utilization": args.max_gpu_utilization,
            "gpu_utilization_policy": args.gpu_utilization_policy,
            "rows": gpu_evidence,
            "compute_processes": query_compute_processes(),
        },
        "dataset": data_evidence,
        "config": config_evidence,
    }

    if args.json:
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
    else:
        print(f"ok={str(evidence['ok']).lower()}")
        for issue in issues:
            print(f"issue={issue}")
        for warning in warnings:
            print(f"warning={warning}")
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0 if evidence["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
