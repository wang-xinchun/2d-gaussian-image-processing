#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("build_input_std_route_oracle_hard_cases_v1_t01.py")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import psnr  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(raw_path: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base / path


def build_dataset(config: Dict[str, Any], *, split: str, crop_size: int, limit: int) -> PairedImageDataset:
    dataset_cfg = config["dataset"]
    if split == "train":
        return PairedImageDataset(
            dataset_cfg["train_input"],
            dataset_cfg["train_target"],
            crop_size=crop_size,
            training=False,
            augment_flip=False,
            offset=int(dataset_cfg.get("train_offset", 0)),
            limit=limit if limit > 0 else int(dataset_cfg.get("train_limit", 0)),
        )
    if split == "val":
        return PairedImageDataset(
            dataset_cfg["val_input"],
            dataset_cfg["val_target"],
            crop_size=crop_size,
            training=False,
            augment_flip=False,
            offset=int(dataset_cfg.get("val_offset", 0)),
            limit=limit if limit > 0 else int(dataset_cfg.get("val_limit", 0)),
        )
    raise ValueError(f"unsupported split: {split}")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    config = load_toml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict load produced incompatible keys for {checkpoint_path}: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(checkpoint.get("step", -1))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().mean().cpu())


def collect_records(
    *,
    config: Dict[str, Any],
    split: str,
    crop_size: int,
    limit: int,
    num_workers: int,
    device: torch.device,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    threshold: float,
    advantage_margin: float,
    progress_interval: int,
) -> list[dict[str, Any]]:
    dataset = build_dataset(config, split=split, crop_size=crop_size, limit=limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            aux_a = model_a(inputs, return_aux="loss")
            aux_b = model_b(inputs, return_aux="loss")
            pred_a = aux_a["restored"].clamp(0.0, 1.0)
            pred_b = aux_b["restored"].clamp(0.0, 1.0)
            teacher_a_error = torch.sqrt((pred_a - targets).square() + 1e-6).flatten(1).mean(1)
            teacher_b_error = torch.sqrt((pred_b - targets).square() + 1e-6).flatten(1).mean(1)
            advantage = teacher_b_error - teacher_a_error
            input_std = inputs.flatten(1).std(dim=1, unbiased=False)
            fixed_low = input_std < threshold
            oracle_positive = advantage > advantage_margin
            records.append(
                {
                    "index": index,
                    "name": str(batch["name"][0]),
                    "input_std": scalar(input_std),
                    "fixed_low_route": bool(fixed_low.item()),
                    "psnr_a": psnr(pred_a, targets),
                    "psnr_b": psnr(pred_b, targets),
                    "teacher_a_error": scalar(teacher_a_error),
                    "teacher_b_error": scalar(teacher_b_error),
                    "teacher_a_advantage": scalar(advantage),
                    "oracle_positive": bool(oracle_positive.item()),
                    "hard_positive": bool((fixed_low & oracle_positive).item()),
                }
            )
            if progress_interval > 0 and (index + 1) % progress_interval == 0:
                hard_count = sum(1 for record in records if record["hard_positive"])
                positive_count = sum(1 for record in records if record["oracle_positive"])
                print(
                    json.dumps(
                        {
                            "event": "hard_case_cache_progress",
                            "processed": index + 1,
                            "hard_positive_samples": hard_count,
                            "oracle_positive_samples": positive_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an offline low-route A08-positive hard-case cache for V1/T01 input-std route training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-a", type=Path, default=None)
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--config-b", type=Path, default=None)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--max-hard-cases", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_toml(args.config)
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})
    config_a = args.config_a or resolve_path(loss_cfg["router_distillation_config_a"])
    config_b = args.config_b or resolve_path(loss_cfg["router_distillation_config_b"])
    checkpoint_a = args.checkpoint_a or resolve_path(loss_cfg["router_distillation_checkpoint_a"])
    checkpoint_b = args.checkpoint_b or resolve_path(loss_cfg["router_distillation_checkpoint_b"])
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(
            config.get("training", {}).get(
                "input_std_route_balanced_threshold",
                model_cfg.get("input_std_route_threshold", 0.07026463374495506),
            )
        )
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        fraction_raw = os.environ.get("TWODGAUSS_CUDA_MEMORY_FRACTION")
        if fraction_raw:
            memory_device = device if device.index is not None else torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(fraction_raw), device=memory_device)

    model_a, step_a = load_model(config_a, checkpoint_a, device)
    model_b, step_b = load_model(config_b, checkpoint_b, device)

    start = time.time()
    records = collect_records(
        config=config,
        split=args.split,
        crop_size=args.crop_size,
        limit=args.limit,
        num_workers=args.num_workers,
        device=device,
        model_a=model_a,
        model_b=model_b,
        threshold=threshold,
        advantage_margin=float(args.advantage_margin),
        progress_interval=max(int(args.progress_interval), 0),
    )
    hard_records = [record for record in records if record["hard_positive"]]
    hard_records.sort(key=lambda record: float(record["teacher_a_advantage"]), reverse=True)
    if args.max_hard_cases > 0:
        hard_records = hard_records[: args.max_hard_cases]
    low_records = [record for record in records if record["fixed_low_route"]]
    positive_records = [record for record in records if record["oracle_positive"]]
    advantages = [float(record["teacher_a_advantage"]) for record in records]
    low_advantages = [float(record["teacher_a_advantage"]) for record in low_records]
    hard_advantages = [float(record["teacher_a_advantage"]) for record in hard_records]

    result = {
        "metadata": {
            "script": "build_input_std_route_oracle_hard_cases_v1_t01.py",
            "config": str(args.config),
            "split": args.split,
            "crop_size": args.crop_size,
            "limit": args.limit,
            "threshold": threshold,
            "advantage_margin": float(args.advantage_margin),
            "label_a": args.label_a,
            "label_b": args.label_b,
            "checkpoint_a": str(checkpoint_a),
            "checkpoint_b": str(checkpoint_b),
            "config_a": str(config_a),
            "config_b": str(config_b),
            "checkpoint_step_a": step_a,
            "checkpoint_step_b": step_b,
            "device": str(device),
            "elapsed_seconds": time.time() - start,
        },
        "summary": {
            "samples": len(records),
            "low_route_samples": len(low_records),
            "oracle_positive_samples": len(positive_records),
            "hard_positive_samples": len(hard_records),
            "low_route_fraction": len(low_records) / max(len(records), 1),
            "oracle_positive_fraction": len(positive_records) / max(len(records), 1),
            "hard_positive_fraction": len(hard_records) / max(len(records), 1),
            "advantage_mean": mean(advantages),
            "advantage_p10": percentile(advantages, 0.10),
            "advantage_p50": percentile(advantages, 0.50),
            "advantage_p90": percentile(advantages, 0.90),
            "low_advantage_mean": mean(low_advantages),
            "hard_advantage_mean": mean(hard_advantages),
            "hard_advantage_min": min(hard_advantages) if hard_advantages else 0.0,
            "hard_advantage_max": max(hard_advantages) if hard_advantages else 0.0,
        },
        "hard_case_names": [str(record["name"]) for record in hard_records],
        "hard_case_indices": [int(record["index"]) for record in hard_records],
        "hard_records": hard_records,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "hard_case_cache_built", **result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
