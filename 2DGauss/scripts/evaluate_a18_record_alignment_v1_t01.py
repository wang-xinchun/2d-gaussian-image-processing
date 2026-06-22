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

apply_cuda_visible_devices_policy("evaluate_a18_record_alignment_v1_t01.py")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config, compute_loss  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import psnr  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "psnr_mean": mean([float(row["psnr"]) for row in rows]),
        "router_psnr_mean": mean([float(row["router_psnr"]) for row in rows]),
        "oracle_psnr_mean": mean([float(row["oracle_psnr"]) for row in rows]),
        "gap_router_mean": mean([float(row["gap_router"]) for row in rows]),
        "gap_oracle_mean": mean([float(row["gap_oracle"]) for row in rows]),
        key: mean([float(row[key]) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a checkpoint with A18 A08/A15 router records.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    records_data = json.loads(args.records.read_text(encoding="utf-8"))
    selected_rule = records_data["selected_rule"]
    threshold = float(selected_rule["threshold"])
    direction = str(selected_rule["direction"])
    records = {record["name"]: record for record in records_data["test_records"]}

    dataset_cfg = config["dataset"]
    dataset = PairedImageDataset(
        dataset_cfg["test_input"],
        dataset_cfg["test_target"],
        crop_size=args.crop_size,
        training=False,
        augment_flip=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            name = str(batch["name"][0])
            record = records.get(name)
            if record is None:
                continue
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = model(inputs).clamp(0.0, 1.0)
            loss, _ = compute_loss(predictions, targets, config, input_image=inputs)
            pred_psnr = psnr(predictions, targets)
            input_std = float(record["features"]["input_std"])
            psnr_a = float(record["psnr_a"])
            psnr_b = float(record["psnr_b"])
            choose_b = input_std <= threshold if direction == "lte" else input_std >= threshold
            router_psnr = psnr_b if choose_b else psnr_a
            oracle_psnr = max(psnr_a, psnr_b)
            row = {
                "name": name,
                "input_std": input_std,
                "route": "low_b" if choose_b else "high_a",
                "psnr": pred_psnr,
                "loss": float(loss.cpu()),
                "psnr_a": psnr_a,
                "psnr_b": psnr_b,
                "router_psnr": router_psnr,
                "oracle_psnr": oracle_psnr,
                "gap_router": pred_psnr - router_psnr,
                "gap_oracle": pred_psnr - oracle_psnr,
                "a15_minus_a08": psnr_b - psnr_a,
            }
            rows.append(row)

    low_rows = [row for row in rows if row["route"] == "low_b"]
    high_rows = [row for row in rows if row["route"] == "high_a"]
    a15_oracle_rows = [row for row in rows if float(row["a15_minus_a08"]) > 0.0]
    a08_oracle_rows = [row for row in rows if float(row["a15_minus_a08"]) <= 0.0]
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": str(args.config),
        "records": str(args.records),
        "device": str(device),
        "crop_size": args.crop_size,
        "samples": len(rows),
        "selected_rule": selected_rule,
        "elapsed_seconds": time.time() - start,
        "overall": summarize(rows, "input_std"),
        "low_route": summarize(low_rows, "input_std"),
        "high_route": summarize(high_rows, "input_std"),
        "a15_oracle": summarize(a15_oracle_rows, "a15_minus_a08"),
        "a08_oracle": summarize(a08_oracle_rows, "a15_minus_a08"),
        "worst_gap_router": sorted(rows, key=lambda row: float(row["gap_router"]))[:15],
        "best_gap_router": sorted(rows, key=lambda row: float(row["gap_router"]))[-10:],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
