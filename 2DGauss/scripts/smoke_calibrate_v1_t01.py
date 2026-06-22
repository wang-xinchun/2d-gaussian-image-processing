#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("smoke_calibrate_v1_t01.py")

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import (  # noqa: E402
    build_model_config,
    compute_auxiliary_loss,
    compute_loss,
    seed_everything,
    select_training_supervision,
)
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def clone_with_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    copied: Dict[str, Any] = {section: dict(values) if isinstance(values, dict) else values for section, values in config.items()}
    copied["training"]["batch_size"] = args.batch_size
    copied["dataset"]["train_crop_size"] = args.crop_size
    copied["dataset"]["num_workers"] = args.num_workers
    copied["dataset"]["train_limit"] = max(args.batch_size * max(args.steps + 1, 2), args.batch_size)
    if args.wall_clock_minutes is not None:
        copied["training"]["wall_clock_minutes"] = args.wall_clock_minutes
    if args.target_memory_gb_min is not None:
        copied["training"]["target_reserved_memory_gb_min"] = args.target_memory_gb_min
    if args.target_memory_gb_max is not None:
        copied["training"]["target_reserved_memory_gb_max"] = args.target_memory_gb_max
    model_cfg = copied.get("model", {})
    if bool(model_cfg.get("use_train_inference_consistency", False)):
        model_cfg["train_context_size"] = args.crop_size
        if int(model_cfg.get("train_supervision_crop_size", 0)) <= 0:
            model_cfg["train_supervision_crop_size"] = args.crop_size
    return copied


def make_loader(config: Dict[str, Any]) -> torch.utils.data.DataLoader:
    dataset_cfg = config["dataset"]
    model_cfg = config.get("model", {})
    consistency_enabled = bool(model_cfg.get("use_train_inference_consistency", False))
    dataset = PairedImageDataset(
        dataset_cfg["train_input"],
        dataset_cfg["train_target"],
        crop_size=int(dataset_cfg["train_crop_size"]),
        training=True,
        augment_flip=bool(dataset_cfg.get("augment_flip", True)),
        offset=int(dataset_cfg.get("train_offset", 0)),
        limit=int(dataset_cfg.get("train_limit", 0)),
        context_crop_size=int(model_cfg.get("train_context_size", 0)) if consistency_enabled else 0,
        supervision_crop_size=int(model_cfg.get("train_supervision_crop_size", 0)) if consistency_enabled else 0,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def memory_gb(value: int) -> float:
    return value / 1024**3


def build_memory_window_report(config: Dict[str, Any], reserved_gb: float) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    target_min = training_cfg.get("target_reserved_memory_gb_min")
    target_max = training_cfg.get("target_reserved_memory_gb_max")
    if target_min is None and target_max is None:
        return {}
    report: Dict[str, Any] = {}
    if target_min is not None:
        report["target_reserved_memory_gb_min"] = float(target_min)
    if target_max is not None:
        report["target_reserved_memory_gb_max"] = float(target_max)
    within_window = True
    if target_min is not None:
        within_window = within_window and reserved_gb >= float(target_min)
    if target_max is not None:
        within_window = within_window and reserved_gb <= float(target_max)
    report["within_target_reserved_memory_window"] = within_window
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke calibrate V1/T01 memory and step time on real batches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--crop-size", type=int, default=0, help="0 means full image.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wall-clock-minutes", type=float, default=None)
    parser.add_argument("--target-memory-gb-min", type=float, default=None)
    parser.add_argument("--target-memory-gb-max", type=float, default=None)
    args = parser.parse_args()

    config = clone_with_overrides(load_toml(args.config), args)
    seed_everything(int(config["training"].get("seed", 20260424)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("calibration requires CUDA")

    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.set_training_progress(1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"].get("lr", 2e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )
    loader = make_loader(config)
    iterator = iter(loader)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    fetch_times: list[float] = []
    compute_times: list[float] = []
    total_times: list[float] = []
    last_loss = 0.0
    batch_shape: tuple[int, ...] | None = None

    try:
        for step in range(args.steps):
            total_start = time.perf_counter()
            fetch_start = total_start
            batch = next(iterator)
            fetch_times.append(time.perf_counter() - fetch_start)
            compute_start = time.perf_counter()
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            batch_shape = tuple(inputs.shape)
            optimizer.zero_grad(set_to_none=True)
            aux = model(inputs, return_aux="loss")
            predictions = aux["restored"].clamp(0.0, 1.0)
            effective_predictions, effective_targets, _ = select_training_supervision(predictions, targets, config)
            loss, _ = compute_loss(effective_predictions, effective_targets, config, input_image=inputs)
            aux_loss, _ = compute_auxiliary_loss(aux, targets, config, inputs=inputs)
            loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            compute_times.append(time.perf_counter() - compute_start)
            total_times.append(time.perf_counter() - total_start)
            last_loss = float(loss.detach().cpu())
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        result = {
            "status": "oom",
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "steps_completed": len(total_times),
            "error": str(exc).splitlines()[0],
            "max_memory_allocated_gb": memory_gb(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_gb": memory_gb(torch.cuda.max_memory_reserved(device)),
        }
        result.update(build_memory_window_report(config, float(result["max_memory_reserved_gb"])))
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    avg_fetch_seconds = sum(fetch_times) / max(len(fetch_times), 1)
    avg_compute_seconds = sum(compute_times) / max(len(compute_times), 1)
    avg_total_step_seconds = sum(total_times) / max(len(total_times), 1)
    wall_clock_minutes = float(config["training"].get("wall_clock_minutes", 55.0))
    result = {
        "status": "ok",
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "steps": args.steps,
        "batch_shape": batch_shape,
        "loss": last_loss,
        "avg_fetch_seconds": avg_fetch_seconds,
        "avg_compute_seconds": avg_compute_seconds,
        "avg_total_step_seconds": avg_total_step_seconds,
        "avg_step_seconds": avg_total_step_seconds,
        "estimated_steps_55min": int((55 * 60) / avg_total_step_seconds) if avg_total_step_seconds > 0 else 0,
        "estimated_steps_60min": int((60 * 60) / avg_total_step_seconds) if avg_total_step_seconds > 0 else 0,
        "target_wall_clock_minutes": wall_clock_minutes,
        "estimated_steps_target_wall_clock": int((wall_clock_minutes * 60) / avg_total_step_seconds) if avg_total_step_seconds > 0 else 0,
        "max_memory_allocated_gb": memory_gb(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_gb": memory_gb(torch.cuda.max_memory_reserved(device)),
    }
    result.update(build_memory_window_report(config, float(result["max_memory_reserved_gb"])))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
