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

apply_cuda_visible_devices_policy("evaluate_checkpoint_router_v1_t01.py")

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
from uhd_gaussian.training import (  # noqa: E402
    chroma_l1_loss,
    dark_channel_l1_loss,
    edge_magnitude_l1_loss,
    high_frequency_l1_loss,
    local_contrast_l1_loss,
    luminance_l1_loss,
    psnr,
    saturation_l1_loss,
    ssim,
)


METRIC_NAMES = [
    "psnr",
    "ssim",
    "edge_magnitude_l1",
    "high_frequency_l1",
    "local_contrast_l1",
    "luminance_l1",
    "chroma_l1",
    "saturation_l1",
    "dark_channel_l1",
]


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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
            offset=int(dataset_cfg.get("test_offset", 0)),
            limit=limit if limit > 0 else int(dataset_cfg.get("test_limit", 0)),
        )
    raise ValueError(f"unsupported split: {split}")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    config = load_toml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("step", -1))


def scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().mean().cpu())


def std_scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().std().cpu())


def aux_mean(aux: dict[str, Any], key: str) -> float:
    value = aux.get(key)
    if torch.is_tensor(value):
        return scalar(value)
    return 0.0


def aux_std(aux: dict[str, Any], key: str) -> float:
    value = aux.get(key)
    if torch.is_tensor(value):
        return std_scalar(value)
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "edge_magnitude_l1": float(edge_magnitude_l1_loss(prediction, target).cpu()),
        "high_frequency_l1": float(high_frequency_l1_loss(prediction, target).cpu()),
        "local_contrast_l1": float(local_contrast_l1_loss(prediction, target).cpu()),
        "luminance_l1": float(luminance_l1_loss(prediction, target).cpu()),
        "chroma_l1": float(chroma_l1_loss(prediction, target).cpu()),
        "saturation_l1": float(saturation_l1_loss(prediction, target).cpu()),
        "dark_channel_l1": float(dark_channel_l1_loss(prediction, target).cpu()),
    }


def collect_records(
    *,
    dataset_config: dict[str, Any],
    split: str,
    crop_size: int,
    limit: int,
    num_workers: int,
    device: torch.device,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
) -> list[dict[str, Any]]:
    dataset = build_dataset(dataset_config, split, crop_size, limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            aux_a = model_a(inputs, return_aux="loss")
            aux_b = model_b(inputs, return_aux="loss")
            pred_a = aux_a["restored"].clamp(0.0, 1.0)
            pred_b = aux_b["restored"].clamp(0.0, 1.0)

            features = {
                "input_mean": scalar(inputs),
                "input_std": std_scalar(inputs),
                "pred_a_mean": scalar(pred_a),
                "pred_a_std": std_scalar(pred_a),
                "pred_b_mean": scalar(pred_b),
                "pred_b_std": std_scalar(pred_b),
                "pred_ab_l1": scalar(torch.abs(pred_b - pred_a)),
                "pred_ab_mse": scalar((pred_b - pred_a) ** 2),
                "input_a_l1": scalar(torch.abs(pred_a - inputs)),
                "input_b_l1": scalar(torch.abs(pred_b - inputs)),
                "input_a_mse": scalar((pred_a - inputs) ** 2),
                "input_b_mse": scalar((pred_b - inputs) ** 2),
                "input_residual_l1_delta_b_minus_a": scalar(torch.abs(pred_b - inputs))
                - scalar(torch.abs(pred_a - inputs)),
                "a_control_canvas_mean": aux_mean(aux_a, "control_canvas"),
                "a_control_canvas_std": aux_std(aux_a, "control_canvas"),
                "a_control_feedback_mean": aux_mean(aux_a, "control_feedback_map"),
                "a_control_feedback_std": aux_std(aux_a, "control_feedback_map"),
                "a_guidance_blend_mean": aux_mean(aux_a, "control_guidance_blend_map"),
                "a_guidance_blend_std": aux_std(aux_a, "control_guidance_blend_map"),
                "b_control_canvas_mean": aux_mean(aux_b, "control_canvas"),
                "b_control_canvas_std": aux_std(aux_b, "control_canvas"),
                "b_control_feedback_mean": aux_mean(aux_b, "control_feedback_map"),
                "b_control_feedback_std": aux_std(aux_b, "control_feedback_map"),
                "b_guidance_blend_mean": aux_mean(aux_b, "control_guidance_blend_map"),
                "b_guidance_blend_std": aux_std(aux_b, "control_guidance_blend_map"),
            }
            records.append(
                {
                    "name": str(batch["name"][0]),
                    "psnr_a": psnr(pred_a, targets),
                    "psnr_b": psnr(pred_b, targets),
                    "metrics_a": prediction_metrics(pred_a, targets),
                    "metrics_b": prediction_metrics(pred_b, targets),
                    "features": features,
                }
            )
    return records


def evaluate_rule(records: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    choices: list[str] = []
    scores: list[float] = []
    metric_values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    if rule["type"] == "constant":
        choose_b = rule["choice"] == "b"
        for record in records:
            choices.append("b" if choose_b else "a")
            scores.append(record["psnr_b"] if choose_b else record["psnr_a"])
            metrics = record["metrics_b"] if choose_b else record["metrics_a"]
            for name in METRIC_NAMES:
                metric_values[name].append(metrics[name])
    else:
        feature = rule["feature"]
        threshold = float(rule["threshold"])
        direction = rule["direction"]
        for record in records:
            value = float(record["features"][feature])
            choose_b = value >= threshold if direction == "gte" else value <= threshold
            choices.append("b" if choose_b else "a")
            scores.append(record["psnr_b"] if choose_b else record["psnr_a"])
            metrics = record["metrics_b"] if choose_b else record["metrics_a"]
            for name in METRIC_NAMES:
                metric_values[name].append(metrics[name])
    result = {
        "psnr_mean": mean(scores),
        "choose_b_count": choices.count("b"),
        "samples": len(records),
    }
    for name in METRIC_NAMES:
        result[f"{name}_mean"] = mean(metric_values[name])
    return result


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return [unique[0]]
    thresholds = [unique[0] - 1e-12]
    thresholds.extend((a + b) * 0.5 for a, b in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1e-12)
    return thresholds


def tune_router(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    feature_names = sorted(records[0]["features"])
    candidates: list[dict[str, Any]] = [
        {"type": "constant", "choice": "a"},
        {"type": "constant", "choice": "b"},
    ]
    for feature in feature_names:
        values = [float(record["features"][feature]) for record in records]
        for threshold in candidate_thresholds(values):
            candidates.append({"type": "threshold", "feature": feature, "direction": "gte", "threshold": threshold})
            candidates.append({"type": "threshold", "feature": feature, "direction": "lte", "threshold": threshold})

    scored: list[dict[str, Any]] = []
    for rule in candidates:
        score = evaluate_rule(records, rule)
        scored.append({"rule": rule, **score})
    scored.sort(key=lambda item: item["psnr_mean"], reverse=True)
    best = scored[0]
    return best["rule"], best, scored[:20]


def oracle(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [max(record["psnr_a"], record["psnr_b"]) for record in records]
    choose_b_count = sum(record["psnr_b"] > record["psnr_a"] for record in records)
    return {"psnr_mean": mean(scores), "choose_b_count": choose_b_count, "samples": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a simple val-threshold router between two V1/T01 checkpoints.")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fixed-rule-feature", default=None)
    parser.add_argument("--fixed-rule-direction", choices=("gte", "lte"), default=None)
    parser.add_argument("--fixed-rule-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    fixed_rule_args = (args.fixed_rule_feature, args.fixed_rule_direction, args.fixed_rule_threshold)
    if any(value is not None for value in fixed_rule_args) and not all(
        value is not None for value in fixed_rule_args
    ):
        parser.error("--fixed-rule-feature, --fixed-rule-direction, and --fixed-rule-threshold must be provided together")

    config = load_toml(args.dataset_config)
    device = torch.device(args.device)
    if device.type == "cuda":
        import os

        fraction_raw = os.environ.get("TWODGAUSS_CUDA_MEMORY_FRACTION")
        if fraction_raw:
            memory_device = device if device.index is not None else torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(fraction_raw), device=memory_device)

    model_a, step_a = load_model(args.config_a, args.checkpoint_a, device)
    model_b, step_b = load_model(args.config_b, args.checkpoint_b, device)
    start = time.time()
    val_records = collect_records(
        dataset_config=config,
        split="val",
        crop_size=args.crop_size,
        limit=args.val_limit,
        num_workers=args.num_workers,
        device=device,
        model_a=model_a,
        model_b=model_b,
    )
    test_records = collect_records(
        dataset_config=config,
        split="test",
        crop_size=args.crop_size,
        limit=args.test_limit,
        num_workers=args.num_workers,
        device=device,
        model_a=model_a,
        model_b=model_b,
    )
    rule, val_score, top_val_rules = tune_router(val_records)
    result = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "checkpoint_step_a": step_a,
        "checkpoint_step_b": step_b,
        "device": str(device),
        "crop_size": args.crop_size,
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "baseline_val_a": evaluate_rule(val_records, {"type": "constant", "choice": "a"}),
        "baseline_val_b": evaluate_rule(val_records, {"type": "constant", "choice": "b"}),
        "baseline_test_a": evaluate_rule(test_records, {"type": "constant", "choice": "a"}),
        "baseline_test_b": evaluate_rule(test_records, {"type": "constant", "choice": "b"}),
        "binary_oracle_val": oracle(val_records),
        "binary_oracle_test": oracle(test_records),
        "selected_rule": rule,
        "selected_rule_val": val_score,
        "selected_rule_test": evaluate_rule(test_records, rule),
        "top_val_rules": top_val_rules,
        "elapsed_seconds": time.time() - start,
        "val_records": val_records,
        "test_records": test_records,
    }
    if args.fixed_rule_feature is not None:
        fixed_rule = {
            "type": "threshold",
            "feature": args.fixed_rule_feature,
            "direction": args.fixed_rule_direction,
            "threshold": args.fixed_rule_threshold,
        }
        result["fixed_rule"] = fixed_rule
        result["fixed_rule_val"] = evaluate_rule(val_records, fixed_rule)
        result["fixed_rule_test"] = evaluate_rule(test_records, fixed_rule)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
