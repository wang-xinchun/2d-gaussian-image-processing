#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".y4m"}
FRAME_EXTENSIONS = {".png"}
BANNED_PATH_HINTS = ("uhd_haze", "uhdhaze", "haze_patch", "任务四")
Y4M_HEADER_PATTERN = re.compile(rb"\bW(?P<width>\d+)\b.*\bH(?P<height>\d+)\b")
Y4M_CHROMA_PATTERN = re.compile(rb"\bC(?P<chroma>[^\s]+)")


@dataclass(frozen=True)
class VideoPair:
    input_video: Path
    target_video: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small paired-frame view for the UHD video deshaking smoke "
            "path from explicit shaky/input and stable/target video pairs. The "
            "script refuses unpaired videos and same-file input/target pairs."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-pair",
        action="append",
        default=[],
        metavar="INPUT_VIDEO:TARGET_VIDEO",
        help="Explicit train video pair. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--val-pair",
        action="append",
        default=[],
        metavar="INPUT_VIDEO:TARGET_VIDEO",
        help="Explicit validation video pair. Repeat for multiple pairs.",
    )
    parser.add_argument("--train-frames-per-pair", type=int, default=16)
    parser.add_argument("--val-frames-per-pair", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write PNG frames. Without this flag the script only validates and prints a plan.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


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


def resolve_video(path: Path, data_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not an existing file: {resolved}")
    if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"{label} is not a supported video file: {resolved}")
    if not is_relative_to(resolved, data_root):
        raise ValueError(f"{label} is outside data_root: {resolved}")
    reject_banned_path(resolved, label)
    return resolved


def parse_pair(raw: str, data_root: Path, label: str) -> VideoPair:
    if ":" not in raw:
        raise ValueError(f"{label} must be INPUT_VIDEO:TARGET_VIDEO; got {raw!r}")
    left, right = raw.split(":", 1)
    input_video = resolve_video(Path(left), data_root, f"{label}.input")
    target_video = resolve_video(Path(right), data_root, f"{label}.target")
    if input_video == target_video:
        raise ValueError(f"{label} input and target must be different files: {input_video}")
    return VideoPair(input_video=input_video, target_video=target_video)


def output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "train_input": output_root / "train" / "input",
        "train_target": output_root / "train" / "target",
        "val_input": output_root / "val" / "input",
        "val_target": output_root / "val" / "target",
    }


def ensure_empty_output(output_root: Path, *, force: bool, write: bool) -> None:
    if not write:
        return
    if output_root.exists():
        existing_files = [path for path in output_root.rglob("*") if path.is_file()]
        if existing_files and not force:
            raise ValueError(f"output_root already contains files; pass --force to overwrite: {output_root}")
        if force:
            shutil.rmtree(output_root)


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def run_ffmpeg_extract(
    ffmpeg: str,
    video: Path,
    output_pattern: Path,
    *,
    frames: int,
    start_frame: int,
    stride: int,
    jpeg_quality: int,
) -> None:
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    select_expr = f"gte(n\\,{start_frame})*not(mod(n-{start_frame}\\,{stride}))"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        "-frames:v",
        str(frames),
        "-q:v",
        str(jpeg_quality),
        str(output_pattern),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def parse_y4m_header(handle) -> tuple[int, int, int]:
    header = handle.readline()
    if not header.startswith(b"YUV4MPEG2"):
        raise ValueError("not a Y4M stream")
    match = Y4M_HEADER_PATTERN.search(header)
    if match is None:
        raise ValueError(f"could not parse Y4M width/height from header: {header[:200]!r}")
    width = int(match.group("width"))
    height = int(match.group("height"))
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("Python Y4M fallback supports only even-sized 4:2:0 frames")
    chroma_match = Y4M_CHROMA_PATTERN.search(header)
    chroma = chroma_match.group("chroma").decode("ascii", errors="ignore").lower() if chroma_match else "420jpeg"
    if any(marker in chroma for marker in ("p10", "p12", "p14", "p16")):
        raise ValueError(
            "Python Y4M fallback supports only 8-bit Y4M; install ffmpeg for high-bit-depth Y4M"
        )
    if not chroma.startswith(("420", "420jpeg", "420mpeg2", "420paldv")):
        raise ValueError(f"Python Y4M fallback supports only 4:2:0 Y4M; got C{chroma}")
    chroma_420_bytes = width * height * 3 // 2
    return width, height, chroma_420_bytes


def y_plane_to_rgb(path: Path, frame_index: int) -> Image.Image:
    with path.open("rb") as handle:
        width, height, frame_bytes = parse_y4m_header(handle)
        current_frame = 0
        while True:
            frame_header = handle.readline()
            if not frame_header:
                raise EOFError(f"frame {frame_index} not available in {path}")
            if not frame_header.startswith(b"FRAME"):
                raise ValueError(f"unexpected Y4M frame marker in {path}: {frame_header[:80]!r}")
            payload = handle.read(frame_bytes)
            if len(payload) != frame_bytes:
                raise EOFError(f"truncated frame {current_frame} in {path}")
            if current_frame == frame_index:
                y_plane = np.frombuffer(payload[: width * height], dtype=np.uint8).reshape(height, width)
                return Image.fromarray(y_plane, mode="L").convert("RGB")
            current_frame += 1


def fallback_y4m_extract(video: Path, destination_dir: Path, *, prefix: str, frames: int, start_frame: int, stride: int) -> None:
    if video.suffix.lower() != ".y4m":
        raise ValueError(f"ffmpeg is unavailable and Python fallback supports only Y4M: {video}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for output_index in range(frames):
        frame_index = start_frame + output_index * stride
        image = y_plane_to_rgb(video, frame_index)
        image.save(destination_dir / f"{prefix}_{output_index:06d}.png")


def extract_video(
    video: Path,
    destination_dir: Path,
    *,
    prefix: str,
    frames: int,
    start_frame: int,
    stride: int,
    jpeg_quality: int,
    write: bool,
) -> dict[str, object]:
    planned = {
        "video": str(video),
        "destination_dir": str(destination_dir),
        "prefix": prefix,
        "frames": frames,
        "start_frame": start_frame,
        "stride": stride,
        "wrote": False,
        "method": "plan_only",
    }
    if not write:
        return planned
    ffmpeg = ffmpeg_path()
    if ffmpeg is not None:
        pattern = destination_dir / f"{prefix}_%06d.png"
        run_ffmpeg_extract(
            ffmpeg,
            video,
            pattern,
            frames=frames,
            start_frame=start_frame,
            stride=stride,
            jpeg_quality=jpeg_quality,
        )
        planned["method"] = "ffmpeg"
    else:
        fallback_y4m_extract(
            video,
            destination_dir,
            prefix=prefix,
            frames=frames,
            start_frame=start_frame,
            stride=stride,
        )
        planned["method"] = "python_y4m_luma_fallback"
    planned["wrote"] = True
    planned["actual_png_count"] = len(list(destination_dir.glob(f"{prefix}_*.png")))
    return planned


def collect_png_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len([item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in FRAME_EXTENSIONS])


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, list[VideoPair], list[VideoPair]]:
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"data_root is not an existing directory: {data_root}")
    reject_banned_path(data_root, "data_root")
    if not is_relative_to(output_root, data_root):
        raise ValueError(f"output_root must be under data_root for DDP preflight compatibility: {output_root}")
    if args.train_frames_per_pair < 1:
        raise ValueError("train_frames_per_pair must be >= 1")
    if args.val_frames_per_pair < 1:
        raise ValueError("val_frames_per_pair must be >= 1")
    if args.start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if args.stride < 1:
        raise ValueError("stride must be >= 1")
    if args.jpeg_quality < 1 or args.jpeg_quality > 31:
        raise ValueError("jpeg_quality must be in ffmpeg q:v range 1..31")
    if not args.train_pair:
        raise ValueError("at least one --train-pair is required")
    if not args.val_pair:
        raise ValueError("at least one --val-pair is required")
    train_pairs = [parse_pair(raw, data_root, f"train_pair[{idx}]") for idx, raw in enumerate(args.train_pair)]
    val_pairs = [parse_pair(raw, data_root, f"val_pair[{idx}]") for idx, raw in enumerate(args.val_pair)]
    return data_root, output_root, train_pairs, val_pairs


def prepare_split(
    split: str,
    pairs: list[VideoPair],
    dirs: dict[str, Path],
    *,
    frames_per_pair: int,
    start_frame: int,
    stride: int,
    jpeg_quality: int,
    write: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        prefix = f"{split}_p{pair_index:03d}"
        records.append(
            extract_video(
                pair.input_video,
                dirs[f"{split}_input"],
                prefix=prefix,
                frames=frames_per_pair,
                start_frame=start_frame,
                stride=stride,
                jpeg_quality=jpeg_quality,
                write=write,
            )
        )
        records.append(
            extract_video(
                pair.target_video,
                dirs[f"{split}_target"],
                prefix=prefix,
                frames=frames_per_pair,
                start_frame=start_frame,
                stride=stride,
                jpeg_quality=jpeg_quality,
                write=write,
            )
        )
    return records


def main() -> int:
    args = parse_args()
    try:
        data_root, output_root, train_pairs, val_pairs = validate_args(args)
        ensure_empty_output(output_root, force=args.force, write=args.write)
        dirs = output_dirs(output_root)
        records = []
        records.extend(
            prepare_split(
                "train",
                train_pairs,
                dirs,
                frames_per_pair=args.train_frames_per_pair,
                start_frame=args.start_frame,
                stride=args.stride,
                jpeg_quality=args.jpeg_quality,
                write=args.write,
            )
        )
        records.extend(
            prepare_split(
                "val",
                val_pairs,
                dirs,
                frames_per_pair=args.val_frames_per_pair,
                start_frame=args.start_frame,
                stride=args.stride,
                jpeg_quality=args.jpeg_quality,
                write=args.write,
            )
        )
        counts = {name: collect_png_count(path) for name, path in dirs.items()}
        result = {
            "ok": True,
            "write": bool(args.write),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_frames_per_pair": args.train_frames_per_pair,
            "val_frames_per_pair": args.val_frames_per_pair,
            "start_frame": args.start_frame,
            "stride": args.stride,
            "directories": {name: str(path) for name, path in dirs.items()},
            "png_counts": counts,
            "records": records,
            "next_config_command": (
                "python scripts/create_deshake_ddp_config.py "
                f"--data-root {data_root} "
                f"--train-input {dirs['train_input']} "
                f"--train-target {dirs['train_target']} "
                f"--val-input {dirs['val_input']} "
                f"--val-target {dirs['val_target']}"
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": str(exc)}
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
