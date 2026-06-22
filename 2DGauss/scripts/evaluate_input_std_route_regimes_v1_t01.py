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

apply_cuda_visible_devices_policy("evaluate_input_std_route_regimes_v1_t01.py")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import (  # noqa: E402
    build_model_config,
    center_crop_like,
    compute_loss,
    load_router_distillation_teachers_if_requested,
)
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import psnr  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def build_dataset(config: Dict[str, Any], split: str, crop_size: int, limit: int) -> PairedImageDataset:
    dataset_cfg = config["dataset"]
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
    if split == "test":
        return PairedImageDataset(
            dataset_cfg["test_input"],
            dataset_cfg["test_target"],
            crop_size=crop_size,
            training=False,
            augment_flip=False,
            limit=limit,
        )
    raise ValueError(f"unsupported split: {split}")


def route_target(
    inputs: torch.Tensor,
    prediction: torch.Tensor,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})
    threshold = float(
        loss_cfg.get(
            "input_std_route_threshold",
            model_cfg.get("input_std_route_threshold", 0.07026463374495506),
        )
    )
    direction = str(
        loss_cfg.get(
            "input_std_route_direction",
            model_cfg.get("input_std_route_direction", "gt"),
        )
    )
    if direction not in {"gt", "gte", "lt", "lte"}:
        raise ValueError("input_std_route_direction must be gt, gte, lt, or lte")
    effective_input = inputs
    if effective_input.shape[-2:] != prediction.shape[-2:]:
        effective_input = center_crop_like(effective_input, prediction.shape[-2:])
    input_std = effective_input.detach().flatten(start_dim=1).std(dim=1, unbiased=False)
    if direction in {"gt", "gte"}:
        high_hard = input_std >= threshold
    else:
        high_hard = input_std <= threshold
    return input_std, high_hard


def masked_charbonnier(prediction: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> float:
    if reference.shape[-2:] != prediction.shape[-2:]:
        reference = center_crop_like(reference, prediction.shape[-2:])
    diff = torch.sqrt((prediction - reference).square() + 1e-6)
    mask4 = mask.view(-1, 1, 1, 1).float()
    denom = (mask4.sum() * diff.shape[1] * diff.shape[2] * diff.shape[3]).clamp_min(1.0)
    return float((diff * mask4).sum().detach().cpu() / denom.detach().cpu())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric_keys = [
        "input_std",
        "psnr",
        "loss",
        "gate_mean",
        "gate_min",
        "gate_max",
        "prior_mean",
        "high_delta_abs_mean",
        "low_delta_abs_mean",
        "selected_delta_abs_mean",
        "high_branch_l1_vs_teacher_a",
        "low_branch_l1_vs_teacher_b",
        "selected_branch_l1_vs_teacher",
    ]
    return {
        "count": len(rows),
        **{f"{key}_mean": mean([float(row[key]) for row in rows]) for key in numeric_keys if key in rows[0]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate input-std route regimes and terminal branch diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    teacher_a, teacher_b, teacher_info = load_router_distillation_teachers_if_requested(
        config,
        args.config,
        device,
    )

    dataset = build_dataset(config, args.split, args.crop_size, args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    rows: list[dict[str, Any]] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            aux = model(inputs, return_aux="loss")
            predictions = aux["restored"].clamp(0.0, 1.0)
            loss, _ = compute_loss(predictions, targets, config, input_image=inputs)
            input_std, high_hard = route_target(inputs, predictions, config)
            gate = aux.get("input_std_route_gate")
            prior = aux.get("input_std_route_prior")
            high_delta = aux.get("input_std_route_high_delta")
            low_delta = aux.get("input_std_route_low_delta")
            high_restored = aux.get("input_std_route_high_restored")
            low_restored = aux.get("input_std_route_low_restored")
            teacher_a_pred = teacher_a(inputs).clamp(0.0, 1.0) if teacher_a is not None else None
            teacher_b_pred = teacher_b(inputs).clamp(0.0, 1.0) if teacher_b is not None else None
            is_high = bool(high_hard.item())
            row: dict[str, Any] = {
                "name": str(batch["name"][0]),
                "route": "high" if is_high else "low",
                "input_std": float(input_std.item()),
                "psnr": psnr(predictions, targets),
                "loss": float(loss.detach().cpu()),
            }
            if gate is not None:
                gate_sample = gate.detach().float()
                row.update(
                    {
                        "gate_mean": float(gate_sample.mean().cpu()),
                        "gate_min": float(gate_sample.amin().cpu()),
                        "gate_max": float(gate_sample.amax().cpu()),
                    }
                )
            if prior is not None:
                row["prior_mean"] = float(prior.detach().float().mean().cpu())
            if high_delta is not None:
                row["high_delta_abs_mean"] = float(high_delta.detach().float().abs().mean().cpu())
            if low_delta is not None:
                row["low_delta_abs_mean"] = float(low_delta.detach().float().abs().mean().cpu())
            if high_delta is not None and low_delta is not None:
                row["selected_delta_abs_mean"] = row["high_delta_abs_mean"] if is_high else row["low_delta_abs_mean"]
            if high_restored is not None and teacher_a_pred is not None:
                row["high_branch_l1_vs_teacher_a"] = masked_charbonnier(
                    high_restored,
                    teacher_a_pred,
                    torch.ones_like(high_hard, dtype=torch.float32),
                )
            if low_restored is not None and teacher_b_pred is not None:
                row["low_branch_l1_vs_teacher_b"] = masked_charbonnier(
                    low_restored,
                    teacher_b_pred,
                    torch.ones_like(high_hard, dtype=torch.float32),
                )
            if high_restored is not None and low_restored is not None:
                if is_high and teacher_a_pred is not None:
                    row["selected_branch_l1_vs_teacher"] = row["high_branch_l1_vs_teacher_a"]
                elif (not is_high) and teacher_b_pred is not None:
                    row["selected_branch_l1_vs_teacher"] = row["low_branch_l1_vs_teacher_b"]
            rows.append(row)

    low_rows = [row for row in rows if row["route"] == "low"]
    high_rows = [row for row in rows if row["route"] == "high"]
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": str(args.config),
        "split": args.split,
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(rows),
        "teacher_info": teacher_info,
        "elapsed_seconds": time.time() - start,
        "overall": summarize(rows),
        "low_route": summarize(low_rows),
        "high_route": summarize(high_rows),
        "worst_psnr": sorted(rows, key=lambda row: float(row["psnr"]))[:10],
        "best_psnr": sorted(rows, key=lambda row: float(row["psnr"]))[-10:],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
