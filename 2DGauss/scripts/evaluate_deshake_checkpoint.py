#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpu_safety import apply_cuda_visible_devices_policy


apply_cuda_visible_devices_policy("evaluate_deshake_checkpoint.py")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config, compute_auxiliary_loss, compute_loss  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training.metrics import psnr, ssim  # noqa: E402


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_val_dataset(config: dict[str, Any], crop_size: int | None) -> PairedImageDataset:
    dataset_cfg = config["dataset"]
    if crop_size is None:
        crop_size = int(dataset_cfg.get("val_crop_size", 0))
    return PairedImageDataset(
        dataset_cfg["val_input"],
        dataset_cfg["val_target"],
        crop_size=crop_size,
        training=False,
        augment_flip=False,
        offset=int(dataset_cfg.get("val_offset", 0)),
        limit=int(dataset_cfg.get("val_limit", 0)),
    )


def parse_sample_name(name: str) -> tuple[str, int | None]:
    stem = Path(name).stem
    if "__" not in stem:
        return stem, None
    clip_id, frame_text = stem.rsplit("__", 1)
    try:
        frame_index = int(frame_text)
    except ValueError:
        frame_index = None
    return clip_id, frame_index


def manifest_clip_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("clips", []):
        clip_id = item.get("clip_id")
        if isinstance(clip_id, str):
            result[clip_id] = item
    return result


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    value = psnr(prediction, target)
    if math.isinf(value):
        return 99.0
    return float(value)


def edge_map(tensor: torch.Tensor) -> torch.Tensor:
    gray = tensor.mean(dim=0, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, :-1] = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
    gy[:, :-1, :] = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
    return gx + gy


def adjacent_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        [record for record in records if record["frame_index"] is not None],
        key=lambda item: int(item["frame_index"]),
    )
    if len(ordered) < 2:
        return {"adjacent_pairs": 0}
    domains = ("input", "prediction", "target")
    values: dict[str, list[float]] = defaultdict(list)
    for left, right in zip(ordered, ordered[1:]):
        for domain in domains:
            left_tensor = left[f"{domain}_tensor"]
            right_tensor = right[f"{domain}_tensor"]
            values[f"{domain}_adjacent_l1"].append(float(F.l1_loss(left_tensor, right_tensor).item()))
            values[f"{domain}_adjacent_edge_l1"].append(
                float(F.l1_loss(edge_map(left_tensor), edge_map(right_tensor)).item())
            )
    result: dict[str, Any] = {"adjacent_pairs": len(ordered) - 1}
    for key, series in values.items():
        result[f"{key}_mean"] = mean(series)
    target_l1 = result.get("target_adjacent_l1_mean")
    target_edge = result.get("target_adjacent_edge_l1_mean")
    if target_l1 is not None:
        result["input_adjacent_l1_delta_to_target"] = result["input_adjacent_l1_mean"] - target_l1
        result["prediction_adjacent_l1_delta_to_target"] = result["prediction_adjacent_l1_mean"] - target_l1
    if target_edge is not None:
        result["input_adjacent_edge_l1_delta_to_target"] = result["input_adjacent_edge_l1_mean"] - target_edge
        result["prediction_adjacent_edge_l1_delta_to_target"] = (
            result["prediction_adjacent_edge_l1_mean"] - target_edge
        )
    return result


def summarize(records: list[dict[str, Any]], key_prefix: str = "") -> dict[str, Any]:
    return {
        f"{key_prefix}sample_count": len(records),
        f"{key_prefix}input_psnr_mean": mean([record["input_psnr"] for record in records]),
        f"{key_prefix}prediction_psnr_mean": mean([record["prediction_psnr"] for record in records]),
        f"{key_prefix}psnr_gain_mean": mean([record["psnr_gain"] for record in records]),
        f"{key_prefix}input_l1_mean": mean([record["input_l1"] for record in records]),
        f"{key_prefix}prediction_l1_mean": mean([record["prediction_l1"] for record in records]),
        f"{key_prefix}l1_delta_mean": mean([record["prediction_l1"] - record["input_l1"] for record in records]),
        f"{key_prefix}input_ssim_mean": mean([record["input_ssim"] for record in records]),
        f"{key_prefix}prediction_ssim_mean": mean([record["prediction_ssim"] for record in records]),
        f"{key_prefix}ssim_gain_mean": mean([record["ssim_gain"] for record in records]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a deshake checkpoint on paired val frames with frame and sequence proxies."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    manifest = load_manifest(args.manifest)
    clips_by_id = manifest_clip_map(manifest)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and device.type != "cuda":
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_val_dataset(config, args.crop_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    start = time.time()
    records: list[dict[str, Any]] = []
    tensor_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            aux = model(inputs, return_aux="loss")
            predictions = aux["restored"].clamp(0.0, 1.0)
            loss, _ = compute_loss(predictions, targets, config, input_image=inputs)
            aux_loss, _ = compute_auxiliary_loss(aux, targets, config, inputs=inputs)

            name = str(batch["name"][0])
            clip_id, frame_index = parse_sample_name(name)
            clip_info = clips_by_id.get(clip_id, {})
            input_tensor = inputs[0].detach().cpu()
            prediction_tensor = predictions[0].detach().cpu()
            target_tensor = targets[0].detach().cpu()
            input_l1 = float(F.l1_loss(input_tensor, target_tensor).item())
            prediction_l1 = float(F.l1_loss(prediction_tensor, target_tensor).item())
            input_psnr = safe_psnr(input_tensor, target_tensor)
            prediction_psnr = safe_psnr(prediction_tensor, target_tensor)
            input_ssim = float(ssim(input_tensor, target_tensor))
            prediction_ssim = float(ssim(prediction_tensor, target_tensor))
            record = {
                "name": name,
                "clip_id": clip_id,
                "frame_index": frame_index,
                "motion_preset": clip_info.get("motion_preset"),
                "strength": clip_info.get("strength"),
                "input_l1": input_l1,
                "prediction_l1": prediction_l1,
                "input_psnr": input_psnr,
                "prediction_psnr": prediction_psnr,
                "psnr_gain": prediction_psnr - input_psnr,
                "input_ssim": input_ssim,
                "prediction_ssim": prediction_ssim,
                "ssim_gain": prediction_ssim - input_ssim,
                "loss": float(loss.detach().cpu()),
                "auxiliary_loss": float(aux_loss.detach().cpu()),
                "total_loss": float((loss + aux_loss).detach().cpu()),
            }
            records.append(record)
            tensor_records.append(
                {
                    **record,
                    "input_tensor": input_tensor,
                    "prediction_tensor": prediction_tensor,
                    "target_tensor": target_tensor,
                }
            )

    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_motion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in tensor_records:
        by_clip[str(record["clip_id"])].append(record)
        by_motion[str(record.get("motion_preset") or "unknown")].append(record)

    clip_summary: dict[str, Any] = {}
    for clip_id, clip_records in sorted(by_clip.items()):
        plain_records = [{k: v for k, v in item.items() if not k.endswith("_tensor")} for item in clip_records]
        clip_summary[clip_id] = {
            **summarize(plain_records),
            **adjacent_metrics(clip_records),
        }

    motion_summary: dict[str, Any] = {}
    for motion, motion_records in sorted(by_motion.items()):
        plain_records = [{k: v for k, v in item.items() if not k.endswith("_tensor")} for item in motion_records]
        motion_summary[motion] = summarize(plain_records)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": str(args.config),
        "manifest": str(args.manifest) if args.manifest else None,
        "device": str(device),
        "crop_size": args.crop_size if args.crop_size is not None else int(config["dataset"].get("val_crop_size", 0)),
        "elapsed_seconds": time.time() - start,
        "overall": summarize(records),
        "clip_summary": clip_summary,
        "motion_summary": motion_summary,
        "note": (
            "Sequence metrics are lightweight adjacent-frame proxies on validation crops. "
            "They indicate temporal consistency trends but are not final full-video stabilization metrics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_sample_output is not None:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
