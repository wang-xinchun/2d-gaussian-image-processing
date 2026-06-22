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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v1_t01 import build_model_config, load_init_checkpoint_if_requested, load_model_state_from_checkpoint  # noqa: E402
from uhd_gaussian import CrossResolutionGaussianRestorer  # noqa: E402


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    route = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("gaussian_route_control")
    )
    return {
        "total_parameter_count": int(total),
        "gaussian_route_control_parameter_count": int(route),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a37-config",
        type=Path,
        default=ROOT
        / "experiments/V1/runs/A_4kd37/configs/a_4kd37_fast_4kdehaze_a33_high_route_a36_teacher_compress.toml",
    )
    parser.add_argument(
        "--g-config",
        type=Path,
        default=ROOT
        / "experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_grscn_route_specific_gaussian_control.template.toml",
    )
    parser.add_argument(
        "--a37-checkpoint",
        type=Path,
        default=ROOT / "experiments/V1/runs/A_4kd37/checkpoints/best_val.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/G_RSCN/runs/G_4kd01/metrics/a37_zero_init_equivalence_cpu.json",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--gate-prior-tolerance", type=float, default=1.1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    a37_config = load_toml(args.a37_config)
    g_config = load_toml(args.g_config)

    a37_model = CrossResolutionGaussianRestorer(build_model_config(a37_config)).to(device)
    g_model = CrossResolutionGaussianRestorer(build_model_config(g_config)).to(device)
    a37_init = load_model_state_from_checkpoint(a37_model, args.a37_checkpoint, device, strict=True)
    g_init = load_init_checkpoint_if_requested(g_model, g_config, args.g_config, device)
    a37_model.eval()
    g_model.eval()

    sample = torch.rand(2, 3, args.height, args.width, device=device)
    with torch.no_grad():
        a37_aux = a37_model(sample, return_aux=True)
        g_aux = g_model(sample, return_aux=True)

    output_diff = tensor_diff(a37_aux["restored"], g_aux["restored"])
    route_delta = g_aux.get("gaussian_route_control_delta")
    route_gate = g_aux.get("gaussian_route_control_gate")
    route_prior = g_aux.get("gaussian_route_control_prior")
    route_confidence = g_aux.get("gaussian_route_control_confidence")
    route_delta_stats = (
        {
            "max_abs": float(route_delta.abs().max().item()),
            "mean_abs": float(route_delta.abs().mean().item()),
        }
        if route_delta is not None
        else None
    )
    gate_prior_diff = tensor_diff(route_gate, route_prior) if route_gate is not None and route_prior is not None else None
    confidence_stats = (
        {
            "mean": float(route_confidence.mean().item()),
            "min": float(route_confidence.min().item()),
            "max": float(route_confidence.max().item()),
        }
        if route_confidence is not None
        else None
    )

    a37_params = count_params(a37_model)
    g_params = count_params(g_model)
    expected_missing_prefixes = ["gaussian_route_control."]
    unexpected_keys = list(g_init.get("init_checkpoint_unexpected_keys", []))
    missing_keys = list(g_init.get("init_checkpoint_missing_keys", []))
    missing_ok = bool(missing_keys) and all(
        any(key.startswith(prefix) for prefix in expected_missing_prefixes)
        for key in missing_keys
    )
    pass_equivalence = (
        output_diff["max_abs"] <= args.tolerance
        and route_delta_stats is not None
        and route_delta_stats["max_abs"] <= args.tolerance
        and gate_prior_diff is not None
        and gate_prior_diff["max_abs"] <= args.gate_prior_tolerance
        and missing_ok
        and not unexpected_keys
    )
    record: Dict[str, Any] = {
        "check": "G_4kd01_A37_zero_init_equivalence_cpu",
        "seed": args.seed,
        "sample_shape": [2, 3, args.height, args.width],
        "tolerance": args.tolerance,
        "gate_prior_tolerance": args.gate_prior_tolerance,
        "pass": pass_equivalence,
        "a37_config": str(args.a37_config),
        "a37_checkpoint": str(args.a37_checkpoint),
        "g_config": str(args.g_config),
        "a37_init": a37_init,
        "g_init": g_init,
        "a37_params": a37_params,
        "g_params": g_params,
        "parameter_increase_vs_a37": int(g_params["total_parameter_count"] - a37_params["total_parameter_count"]),
        "output_diff": output_diff,
        "gaussian_route_control_delta": route_delta_stats,
        "gaussian_route_control_gate_prior_diff": gate_prior_diff,
        "gaussian_route_control_confidence": confidence_stats,
        "missing_keys_ok": missing_ok,
        "unexpected_keys_ok": not unexpected_keys,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not pass_equivalence:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
