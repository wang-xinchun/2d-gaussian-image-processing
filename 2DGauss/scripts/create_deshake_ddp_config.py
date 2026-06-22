#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FRAME_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".npy"}
BANNED_PATH_HINTS = ("uhd_haze", "uhdhaze", "haze_patch", "任务四")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a conservative GPU0/GPU3 DDP config, or an explicit GPU0-only "
            "large-batch smoke config, for the corrected UHD video deshaking task "
            "after paired frame directories exist."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--val-target", type=Path, required=True)
    parser.add_argument(
        "--single-gpu0-smoke",
        action="store_true",
        help="Generate the explicit physical GPU0-only large-batch smoke profile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-ddp-batch-size",
        type=int,
        default=2,
        help="Upper safety cap for the GPU0/GPU3 DDP per-rank smoke profile.",
    )
    parser.add_argument(
        "--max-single-gpu0-batch-size",
        type=int,
        default=24,
        help="Upper safety cap for the explicit physical GPU0-only smoke profile.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--wall-clock-minutes", type=float, default=20.0)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--val-limit", type=int, default=8)
    parser.add_argument("--train-crop-size", type=int, default=512)
    parser.add_argument("--val-crop-size", type=int, default=512)
    parser.add_argument("--min-train-pairs", type=int, default=1)
    parser.add_argument("--min-val-pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_existing_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not an existing directory: {resolved}")
    return resolved


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def reject_banned_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    if any(hint in lowered for hint in BANNED_PATH_HINTS):
        raise ValueError(f"{label} contains a known non-deshaking path hint: {path}")


def image_files(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS],
        key=lambda item: item.name,
    )


def paired_samples(input_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    targets = image_files(target_dir)
    targets_by_name = {path.name: path for path in targets}
    targets_by_stem = {path.stem: path for path in targets}
    pairs: list[tuple[Path, Path]] = []
    for input_path in image_files(input_dir):
        target_path = targets_by_name.get(input_path.name) or targets_by_stem.get(input_path.stem)
        if target_path is not None:
            pairs.append((input_path, target_path))
    return pairs


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def toml_section(name: str, values: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
    return "\n".join(lines)


def build_config(args: argparse.Namespace, paths: dict[str, Path], counts: dict[str, int]) -> str:
    run_id = args.run_id
    profile = "gpu0_only_large_bs_smoke" if args.single_gpu0_smoke else "gpu0_3_ddp_smoke"
    classification = (
        "current_uhd_video_deshaking_gpu0_large_bs_smoke_candidate"
        if args.single_gpu0_smoke
        else "current_uhd_video_deshaking_ddp_smoke_candidate"
    )
    sections = [
        toml_section(
            "identity",
            {
                "architecture": "V1",
                "template": "T01",
                "run": run_id,
                "full_id": f"V1/T01/{run_id}",
            },
        ),
        toml_section(
            "task",
            {
                "name": "UHD_video_deshaking",
                "mode": f"paired_video_stabilization_frame_{profile}",
                "classification": classification,
                "notes": (
                    "Generated only after paired train/val frame directories were "
                    "verified under the deshaking data root."
                ),
            },
        ),
        toml_section(
            "dataset",
            {
                "name": "UHD_video_deshaking_paired_frames",
                "task": "paired_uhd_video_deshake_stabilization_frames",
                "data_root": str(paths["data_root"]),
                "train_input": str(paths["train_input"]),
                "train_target": str(paths["train_target"]),
                "val_input": str(paths["val_input"]),
                "val_target": str(paths["val_target"]),
                "train_offset": 0,
                "train_limit": args.train_limit,
                "val_offset": 0,
                "val_limit": args.val_limit,
                "preflight_val_limit": min(max(args.val_limit, 1), 20),
                "train_crop_size": args.train_crop_size,
                "val_crop_size": args.val_crop_size,
                "augment_flip": True,
                "num_workers": args.num_workers,
                "verified_train_pairs": counts["train"],
                "verified_val_pairs": counts["val"],
            },
        ),
        toml_section(
            "model",
            {
                "in_channels": 3,
                "out_channels": 3,
                "width": 32,
                "lowres_scale": 4,
                "highres_blocks": 2,
                "lowres_blocks": 4,
                "decoder_blocks": 2,
                "affine_rank": 4,
                "gaussian_atoms": 4,
                "gaussian_param_dim": 10,
                "max_frequency": 4.0,
                "max_texture_gain": 1.0,
            },
        ),
        toml_section(
            "loss",
            {
                "charbonnier_weight": 1.0,
                "gradient_weight": 0.05,
            },
        ),
        toml_section(
            "optimizer",
            {
                "name": "AdamW",
                "lr": 0.0002,
                "weight_decay": 0.0001,
            },
        ),
        toml_section(
            "training",
            {
                "tier": "smoke",
                "wall_clock_minutes": args.wall_clock_minutes,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "grad_accum": 1,
                "seed": args.seed,
                "device": "auto",
                "amp": False,
                "log_interval": 1,
                "val_interval": max(1, min(args.max_steps, 10)),
                "checkpoint_interval": max(1, min(args.max_steps, 10)),
                "max_val_batches": max(1, min(args.val_limit, 8)),
                "ddp_find_unused_parameters": True,
            },
        ),
        toml_section(
            "archive",
            {
                "output_root": "experiments/V1/runs",
                "snapshot_on_start": True,
            },
        ),
    ]
    return "\n\n".join(sections) + "\n"


def validate_args(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, int]]:
    if args.max_ddp_batch_size < 1:
        raise ValueError("max_ddp_batch_size must be >= 1")
    if args.max_single_gpu0_batch_size < 1:
        raise ValueError("max_single_gpu0_batch_size must be >= 1")
    max_batch_size = args.max_single_gpu0_batch_size if args.single_gpu0_smoke else args.max_ddp_batch_size
    if args.batch_size < 1 or args.batch_size > max_batch_size:
        profile = "GPU0-only smoke" if args.single_gpu0_smoke else "GPU0/GPU3 DDP"
        raise ValueError(
            f"batch_size must be in the safe {profile} per-GPU range 1..{max_batch_size}"
        )
    if args.num_workers < 0 or args.num_workers > 2:
        raise ValueError("num_workers must be in the safe per-rank range 0..2")
    if args.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if args.wall_clock_minutes <= 0:
        raise ValueError("wall_clock_minutes must be > 0")
    for name in ("train_limit", "val_limit", "train_crop_size", "val_crop_size"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be >= 0")

    data_root = resolve_existing_dir(args.data_root, "data_root")
    paths = {
        "data_root": data_root,
        "train_input": resolve_existing_dir(args.train_input, "train_input"),
        "train_target": resolve_existing_dir(args.train_target, "train_target"),
        "val_input": resolve_existing_dir(args.val_input, "val_input"),
        "val_target": resolve_existing_dir(args.val_target, "val_target"),
    }
    for label, path in paths.items():
        reject_banned_path(path, label)
        if label != "data_root" and not is_relative_to(path, data_root):
            raise ValueError(f"{label} is outside data_root: {path}")

    train_pairs = paired_samples(paths["train_input"], paths["train_target"])
    val_pairs = paired_samples(paths["val_input"], paths["val_target"])
    if len(train_pairs) < args.min_train_pairs:
        raise ValueError(
            f"train paired frame count {len(train_pairs)} < required {args.min_train_pairs}"
        )
    if len(val_pairs) < args.min_val_pairs:
        raise ValueError(f"val paired frame count {len(val_pairs)} < required {args.min_val_pairs}")
    return paths, {"train": len(train_pairs), "val": len(val_pairs)}


def main() -> int:
    args = parse_args()
    try:
        if args.run_id is None:
            args.run_id = (
                "A_dsh03_gpu0_bs8_smoke"
                if args.single_gpu0_smoke
                else "A_dsh02_ddp_smoke"
            )
        if args.batch_size is None:
            args.batch_size = 8 if args.single_gpu0_smoke else 2
        if args.output is None:
            args.output = Path(
                "experiments/V1/runs/deshake_ddp_0_3_pending/configs/"
                + (
                    "a_dsh03_gpu0_bs8_smoke_uhd_video_deshake.toml"
                    if args.single_gpu0_smoke
                    else "a_dsh02_ddp_smoke_uhd_video_deshake.toml"
                )
            )
        paths, counts = validate_args(args)
        output = args.output.expanduser()
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        if output.exists() and not args.force:
            raise ValueError(f"output already exists; pass --force to overwrite: {output}")
        config_text = build_config(args, paths, counts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(config_text, encoding="utf-8")
        result = {
            "ok": True,
            "output": str(output),
            "data_root": str(paths["data_root"]),
            "train_pairs": counts["train"],
            "val_pairs": counts["val"],
            "profile": "gpu0_only_large_bs_smoke" if args.single_gpu0_smoke else "gpu0_3_ddp_smoke",
            "intended_physical_gpus": "0" if args.single_gpu0_smoke else "0,3",
            "batch_size_per_gpu": args.batch_size,
            "effective_global_batch_size": args.batch_size * (1 if args.single_gpu0_smoke else 2),
            "num_workers_per_gpu": args.num_workers,
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            print(f"ok=true output={output}")
            print(f"train_pairs={counts['train']} val_pairs={counts['val']}")
            print(
                f"profile={result['profile']} batch_size_per_gpu={args.batch_size} "
                f"effective_global_batch_size={result['effective_global_batch_size']}"
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            print(f"ok=false error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
