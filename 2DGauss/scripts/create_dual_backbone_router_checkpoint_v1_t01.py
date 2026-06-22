#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_v1_t01 import build_model_config  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_model_state(config_path: Path, checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], int]:
    config = load_toml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CrossResolutionGaussianRestorer(build_model_config(config))
    model.load_state_dict(checkpoint["model"])
    return model.state_dict(), int(checkpoint.get("step", -1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack two V1/T01 backbones into one fixed input_std router checkpoint.")
    parser.add_argument("--config-high", type=Path, required=True)
    parser.add_argument("--checkpoint-high", type=Path, required=True)
    parser.add_argument("--label-high", default="high")
    parser.add_argument("--config-low", type=Path, required=True)
    parser.add_argument("--checkpoint-low", type=Path, required=True)
    parser.add_argument("--label-low", default="low")
    parser.add_argument("--threshold", type=float, default=0.07026463374495506)
    parser.add_argument("--direction", choices=("gt", "gte", "lt", "lte"), default="gt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    high_config = load_toml(args.config_high)
    low_config = load_toml(args.config_low)
    high_state, high_step = load_model_state(args.config_high, args.checkpoint_high)
    low_state, low_step = load_model_state(args.config_low, args.checkpoint_low)
    if set(high_state) != set(low_state):
        high_only = sorted(set(high_state) - set(low_state))
        low_only = sorted(set(low_state) - set(high_state))
        raise ValueError(
            "high and low backbones must have matching state keys; "
            f"high_only={high_only[:8]}, low_only={low_only[:8]}"
        )

    checkpoint = {
        "architecture": "dual_backbone_input_std_router_v1_t01",
        "model_high": high_state,
        "model_low": low_state,
        "config_high": high_config,
        "config_low": low_config,
        "checkpoint_high": str(args.checkpoint_high),
        "checkpoint_low": str(args.checkpoint_low),
        "checkpoint_step_high": high_step,
        "checkpoint_step_low": low_step,
        "label_high": str(args.label_high),
        "label_low": str(args.label_low),
        "router": {
            "feature": "input_std",
            "threshold": float(args.threshold),
            "direction": str(args.direction),
            "high_label": str(args.label_high),
            "low_label": str(args.label_low),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    summary = {
        "output": str(args.output),
        "architecture": checkpoint["architecture"],
        "checkpoint_high": str(args.checkpoint_high),
        "checkpoint_low": str(args.checkpoint_low),
        "checkpoint_step_high": high_step,
        "checkpoint_step_low": low_step,
        "label_high": str(args.label_high),
        "label_low": str(args.label_low),
        "state_key_count": len(high_state),
        "threshold": float(args.threshold),
        "direction": str(args.direction),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
