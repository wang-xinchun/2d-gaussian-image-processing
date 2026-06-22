#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SYNTH_ROOT = ROOT.parent / "4K_8K" / "uhd_shake_synthesis_algorithm"
SYNTH_SCRIPTS = SYNTH_ROOT / "scripts"
if str(SYNTH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SYNTH_SCRIPTS))

from common import base_view_transform, homography_from_params, write_json  # noqa: E402
from synthesize_shaky_clip import (  # noqa: E402
    generate_trajectory,
    make_output_grid,
    map_grid,
    sample_rgb16_bilinear,
)
from validate_clip_quality import validate_clip  # noqa: E402


VIDEO_EXTENSIONS = {".7z", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".y4m", ".yuv"}
PRESET_CYCLE = [
    "portrait_handheld",
    "walking_diagonal_smooth_detail",
    "bicycle_smooth",
    "quick_rotation",
    "impact_shake",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize high-quality shaky/stable paired clips from BVI-AOM videos "
            "using the UHD shake synthesis algorithm."
        )
    )
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "4K_8K" / "BVI-AOM_extracted")
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "4K_8K" / "BVI-AOM_synth_deshake_v1")
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="File basename to exclude from source discovery. Repeat for multiple files.",
    )
    parser.add_argument(
        "--min-file-age-seconds",
        type=int,
        default=0,
        help="Skip source files modified more recently than this many seconds.",
    )
    parser.add_argument(
        "--source-selection",
        choices=("sorted", "round_robin_parts"),
        default="sorted",
        help="Select source videos either by sorted path order or by round-robin over parent part folders.",
    )
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=24)
    parser.add_argument("--decode-width", type=int, default=960)
    parser.add_argument("--decode-height", type=int, default=544)
    parser.add_argument("--base-view-scale", type=float, default=1.18)
    parser.add_argument("--strength", default="medium")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--skip-failed-clips", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument(
        "--clip-plan",
        type=Path,
        default=None,
        help="JSON plan with explicit source/start/stride/split entries. Overrides source discovery.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def find_ffmpeg(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    try:
        import imageio_ffmpeg

        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:  # noqa: BLE001
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ffmpeg was not found; install ffmpeg or imageio-ffmpeg")


def video_files(root: Path, exclude_names: set[str], min_file_age_seconds: int) -> list[Path]:
    now = time.time()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.name in exclude_names:
            continue
        if min_file_age_seconds > 0 and now - path.stat().st_mtime < min_file_age_seconds:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.as_posix())


def select_source_videos(paths: list[Path], max_videos: int, mode: str) -> list[Path]:
    if max_videos <= 0:
        return paths
    if mode == "sorted":
        return paths[:max_videos]
    if mode != "round_robin_parts":
        raise ValueError(f"unknown source selection mode: {mode}")

    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.parent.name, []).append(path)
    selected: list[Path] = []
    keys = sorted(grouped)
    while len(selected) < max_videos and any(grouped[key] for key in keys):
        for key in keys:
            group = grouped[key]
            if not group:
                continue
            selected.append(group.pop(0))
            if len(selected) >= max_videos:
                break
    return selected


def safe_clip_id(path: Path, clip_index: int, preset: str) -> str:
    stem = "".join(ch if ch.isalnum() else "_" for ch in path.stem)
    stem = "_".join(part for part in stem.split("_") if part)
    return f"{stem[:80]}_c{clip_index:02d}_{preset}"


def raw_yuv_metadata(video: Path) -> tuple[int, int, str, str]:
    match = re.search(
        r"(?P<width>\d{3,5})x(?P<height>\d{3,5})_(?P<fps>\d+)fps_.*?(?P<subsampling>420|444)_(?P<bitdepth>8|10)bit",
        video.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            "raw .yuv source names must include geometry/fps/subsampling/bitdepth, "
            f"for example 3840x2160_120fps_420_10bit: {video}"
        )
    width = int(match.group("width"))
    height = int(match.group("height"))
    fps = match.group("fps")
    subsampling = match.group("subsampling")
    bitdepth = match.group("bitdepth")
    if subsampling != "420":
        raise ValueError(f"unsupported raw YUV subsampling {subsampling!r}: {video}")
    pix_fmt = "yuv420p10le" if bitdepth == "10" else "yuv420p"
    return width, height, fps, pix_fmt


def archive_yuv_member(archive: Path) -> str:
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise FileNotFoundError("7z was not found; install p7zip to stream .7z raw YUV sources")
    result = subprocess.run(
        [seven_zip, "l", "-slt", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"7z failed to inspect {archive}: {result.stderr.strip()}")
    members = [
        line.removeprefix("Path = ")
        for line in result.stdout.splitlines()
        if line.startswith("Path = ") and line.lower().endswith(".yuv")
    ]
    if len(members) != 1:
        raise RuntimeError(f"expected exactly one .yuv member in {archive}, found {len(members)}")
    return members[0]


def ffmpeg_input_args(video: Path) -> list[str]:
    if video.suffix.lower() not in {".7z", ".yuv"}:
        return []
    width, height, fps, pix_fmt = raw_yuv_metadata(video)
    return [
        "-f",
        "rawvideo",
        "-pixel_format",
        pix_fmt,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        fps,
    ]


def source_type_for_video(video: Path) -> str:
    suffix = video.suffix.lower()
    if suffix == ".7z":
        return "raw_yuv_streamed_from_7z_with_ffmpeg"
    if suffix == ".yuv":
        return "raw_yuv_decoded_with_ffmpeg"
    if suffix == ".y4m":
        return "y4m_decoded_with_ffmpeg"
    return "video_decoded_with_ffmpeg"


def decode_frame(
    ffmpeg: Path,
    video: Path,
    frame_index: int,
    width: int,
    height: int,
    tmp_dir: Path,
) -> np.ndarray:
    out_path = tmp_dir / f"frame_{frame_index:08d}.png"
    vf = f"select=eq(n\\,{frame_index}),scale={width}:{height}:flags=lanczos"
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_input_args(video),
        "-i",
        str(video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"ffmpeg failed for {video} frame {frame_index}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    with Image.open(out_path) as image:
        arr8 = np.asarray(image.convert("RGB"), dtype=np.uint16)
    return arr8 * 257


def decode_clip_frames(
    ffmpeg: Path,
    video: Path,
    start_frame: int,
    frame_stride: int,
    length: int,
    width: int,
    height: int,
    tmp_dir: Path,
) -> list[np.ndarray]:
    frames_expr = "+".join(f"eq(n\\,{start_frame + idx * frame_stride})" for idx in range(length))
    vf = f"select='{frames_expr}',scale={width}:{height}:flags=lanczos,setpts=N/FRAME_RATE/TB"
    out_pattern = tmp_dir / "clip_frame_%06d.png"
    extractor: subprocess.Popen[bytes] | None = None
    stdin = None
    input_source = str(video)
    if video.suffix.lower() == ".7z":
        member = archive_yuv_member(video)
        seven_zip = shutil.which("7z")
        if seven_zip is None:
            raise FileNotFoundError("7z was not found; install p7zip to stream .7z raw YUV sources")
        extractor = subprocess.Popen(
            [seven_zip, "x", "-so", str(video), member],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdin = extractor.stdout
        input_source = "pipe:0"
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_input_args(video),
        "-i",
        input_source,
        "-vf",
        vf,
        "-vsync",
        "0",
        "-frames:v",
        str(length),
        str(out_pattern),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, stdin=stdin)
    if stdin is not None:
        stdin.close()
    if extractor is not None:
        try:
            extractor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            extractor.terminate()
            extractor.wait(timeout=5)
    paths = sorted(tmp_dir.glob("clip_frame_*.png"))
    if result.returncode != 0 or len(paths) != length:
        raise RuntimeError(
            f"ffmpeg failed for {video} frames {start_frame}:{frame_stride}:{length}: "
            f"{result.stderr.strip() or result.stdout.strip()} decoded={len(paths)}"
        )
    frames: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            arr8 = np.asarray(image.convert("RGB"), dtype=np.uint16)
        frames.append(arr8 * 257)
    return frames


def decode_selected_frames(
    ffmpeg: Path,
    video: Path,
    frame_indices: list[int],
    width: int,
    height: int,
    tmp_dir: Path,
) -> dict[int, np.ndarray]:
    unique_indices = sorted(set(frame_indices))
    frames_expr = "+".join(f"eq(n\\,{frame_index})" for frame_index in unique_indices)
    vf = f"select='{frames_expr}',scale={width}:{height}:flags=lanczos,setpts=N/FRAME_RATE/TB"
    out_pattern = tmp_dir / "selected_frame_%06d.png"
    extractor: subprocess.Popen[bytes] | None = None
    stdin = None
    input_source = str(video)
    if video.suffix.lower() == ".7z":
        member = archive_yuv_member(video)
        seven_zip = shutil.which("7z")
        if seven_zip is None:
            raise FileNotFoundError("7z was not found; install p7zip to stream .7z raw YUV sources")
        extractor = subprocess.Popen(
            [seven_zip, "x", "-so", str(video), member],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdin = extractor.stdout
        input_source = "pipe:0"
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_input_args(video),
        "-i",
        input_source,
        "-vf",
        vf,
        "-vsync",
        "0",
        "-frames:v",
        str(len(unique_indices)),
        str(out_pattern),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, stdin=stdin)
    if stdin is not None:
        stdin.close()
    if extractor is not None:
        try:
            extractor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            extractor.terminate()
            extractor.wait(timeout=5)
    paths = sorted(tmp_dir.glob("selected_frame_*.png"))
    if result.returncode != 0 or len(paths) != len(unique_indices):
        raise RuntimeError(
            f"ffmpeg failed for {video} selected frames count={len(unique_indices)}: "
            f"{result.stderr.strip() or result.stdout.strip()} decoded={len(paths)}"
        )
    frames: dict[int, np.ndarray] = {}
    for frame_index, path in zip(unique_indices, paths, strict=True):
        with Image.open(path) as image:
            arr8 = np.asarray(image.convert("RGB"), dtype=np.uint16)
        frames[frame_index] = arr8 * 257
    return frames


def save_rgb_npy(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image.astype(np.uint16, copy=False))


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image / 257).clip(0, 255).astype(np.uint8), mode="RGB").save(path)


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def contact_sheet(clip_dir: Path, max_frames: int = 4) -> Path:
    stable_files = sorted((clip_dir / "stable").glob("*.npy"))[:max_frames]
    shaky_files = sorted((clip_dir / "shaky").glob("*.npy"))[:max_frames]
    rows: list[Image.Image] = []
    for idx, (stable_path, shaky_path) in enumerate(zip(stable_files, shaky_files)):
        stable = preview_image(stable_path)
        shaky = preview_image(shaky_path)
        row = Image.new("RGB", (stable.width + shaky.width, stable.height + 28), (20, 20, 20))
        row.paste(stable, (0, 28))
        row.paste(shaky, (stable.width, 28))
        draw = ImageDraw.Draw(row)
        draw.text((8, 8), f"frame {idx:06d} stable", fill=(255, 255, 255))
        draw.text((stable.width + 8, 8), f"frame {idx:06d} shaky", fill=(255, 255, 255))
        rows.append(row)
    if not rows:
        raise ValueError(f"no frames for contact sheet: {clip_dir}")
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (0, 0, 0))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    out = clip_dir / "preview_contact_sheet.png"
    sheet.save(out)
    return out


def preview_image(path: Path, max_width: int = 640) -> Image.Image:
    arr = np.load(path)
    image = Image.fromarray((arr / 257).clip(0, 255).astype(np.uint8), mode="RGB")
    scale = max_width / image.width
    return image.resize((max_width, max(1, int(image.height * scale))), Image.Resampling.BILINEAR)


def config_for_validation(width: int, height: int, clip_length: int) -> dict[str, Any]:
    return {
        "poc": {
            "output_width": width,
            "output_height": height,
            "clip_length": clip_length,
        },
        "quality": {
            "valid_area_ratio_min": 1.0,
            "black_border_allowed": False,
            "require_preview": True,
        },
    }


def synthesize_clip(
    *,
    ffmpeg: Path,
    video: Path,
    output_root: Path,
    clip_id: str,
    start_frame: int,
    frame_stride: int,
    length: int,
    width: int,
    height: int,
    preset: str,
    strength: str,
    base_view_scale: float,
    seed: int,
    save_png: bool,
) -> dict[str, Any]:
    clip_dir = output_root / "clips" / clip_id
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    stable_dir = clip_dir / "stable"
    shaky_dir = clip_dir / "shaky"
    mask_dir = clip_dir / "valid_mask"
    stable_dir.mkdir(parents=True, exist_ok=True)
    shaky_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    grid = make_output_grid(width, height)
    b = base_view_transform(width, height, base_view_scale)
    trajectory = generate_trajectory(preset, strength, length, width, height, seed)
    motion_frames: list[dict[str, Any]] = []
    valid_ratios: list[float] = []

    with tempfile.TemporaryDirectory(prefix="bvi_synth_decode_") as tmp:
        tmp_dir = Path(tmp)
        raw_frames = decode_clip_frames(
            ffmpeg,
            video,
            start_frame,
            frame_stride,
            length,
            width,
            height,
            tmp_dir,
        )
        for local_idx, (params, raw) in enumerate(zip(trajectory, raw_frames, strict=True)):
            source_frame_idx = start_frame + local_idx * frame_stride
            h = homography_from_params(
                width,
                height,
                params["dx"],
                params["dy"],
                params["rotation_deg"],
                params["scale"],
                params["perspective_x"],
                params["perspective_y"],
            )
            stable_xy = map_grid(grid, b)
            shaky_xy = map_grid(grid, b @ np.linalg.inv(h))
            stable, stable_valid = sample_rgb16_bilinear(raw, stable_xy)
            shaky, shaky_valid = sample_rgb16_bilinear(raw, shaky_xy)
            valid = stable_valid & shaky_valid
            valid_ratio = float(valid.mean())
            valid_ratios.append(valid_ratio)
            stem = f"{local_idx:06d}"
            save_rgb_npy(stable_dir / f"{stem}.npy", stable)
            save_rgb_npy(shaky_dir / f"{stem}.npy", shaky)
            save_mask_png(mask_dir / f"{stem}.png", valid)
            if save_png:
                save_rgb_png(clip_dir / "stable_png" / f"{stem}.png", stable)
                save_rgb_png(clip_dir / "shaky_png" / f"{stem}.png", shaky)

            frame_meta = dict(params)
            frame_meta.update(
                {
                    "source_frame_index": source_frame_idx,
                    "homography": h.tolist(),
                    "valid_area_ratio": valid_ratio,
                    "blur_kernel": 0,
                    "blur_angle_deg": 0.0,
                }
            )
            motion_frames.append(frame_meta)

    motion = {
        "clip_id": clip_id,
        "source_video": str(video),
        "source_type": source_type_for_video(video),
        "clip_start_frame": start_frame,
        "frame_stride": frame_stride,
        "clip_length": length,
        "output_width": width,
        "output_height": height,
        "degradation_type": "clean_shake",
        "motion_preset": preset,
        "scene_difficulty": "normal",
        "strength": strength,
        "seed": seed,
        "base_view_transform": {
            "type": "zoom_in_internal_viewport",
            "scale": base_view_scale,
            "matrix": b.tolist(),
        },
        "valid_area_ratio_min": min(valid_ratios),
        "valid_area_ratio_mean": float(np.mean(valid_ratios)),
        "frames": motion_frames,
    }
    write_json(clip_dir / "motion.json", motion)
    contact_sheet(clip_dir)
    report = validate_clip(config_for_validation(width, height, length), clip_dir)
    return {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "source_video": str(video),
        "start_frame": start_frame,
        "frame_stride": frame_stride,
        "clip_length": length,
        "motion_preset": preset,
        "strength": strength,
        "seed": seed,
        "valid_area_ratio_min": report["valid_area_ratio_min"],
        "valid_area_ratio_mean": report["valid_area_ratio_mean"],
        "black_border_detected": report["black_border_detected"],
        "status": report["status"],
        "quality_report_json": str(clip_dir / "quality_report.json"),
    }


def synthesize_clip_from_frames(
    *,
    video: Path,
    output_root: Path,
    clip_id: str,
    start_frame: int,
    frame_stride: int,
    length: int,
    width: int,
    height: int,
    preset: str,
    strength: str,
    base_view_scale: float,
    seed: int,
    save_png: bool,
    decoded_frames: dict[int, np.ndarray],
) -> dict[str, Any]:
    clip_dir = output_root / "clips" / clip_id
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    stable_dir = clip_dir / "stable"
    shaky_dir = clip_dir / "shaky"
    mask_dir = clip_dir / "valid_mask"
    stable_dir.mkdir(parents=True, exist_ok=True)
    shaky_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    grid = make_output_grid(width, height)
    b = base_view_transform(width, height, base_view_scale)
    trajectory = generate_trajectory(preset, strength, length, width, height, seed)
    motion_frames: list[dict[str, Any]] = []
    valid_ratios: list[float] = []

    for local_idx, params in enumerate(trajectory):
        source_frame_idx = start_frame + local_idx * frame_stride
        raw = decoded_frames[source_frame_idx]
        h = homography_from_params(
            width,
            height,
            params["dx"],
            params["dy"],
            params["rotation_deg"],
            params["scale"],
            params["perspective_x"],
            params["perspective_y"],
        )
        stable_xy = map_grid(grid, b)
        shaky_xy = map_grid(grid, b @ np.linalg.inv(h))
        stable, stable_valid = sample_rgb16_bilinear(raw, stable_xy)
        shaky, shaky_valid = sample_rgb16_bilinear(raw, shaky_xy)
        valid = stable_valid & shaky_valid
        valid_ratio = float(valid.mean())
        valid_ratios.append(valid_ratio)
        stem = f"{local_idx:06d}"
        save_rgb_npy(stable_dir / f"{stem}.npy", stable)
        save_rgb_npy(shaky_dir / f"{stem}.npy", shaky)
        save_mask_png(mask_dir / f"{stem}.png", valid)
        if save_png:
            save_rgb_png(clip_dir / "stable_png" / f"{stem}.png", stable)
            save_rgb_png(clip_dir / "shaky_png" / f"{stem}.png", shaky)

        frame_meta = dict(params)
        frame_meta.update(
            {
                "source_frame_index": source_frame_idx,
                "homography": h.tolist(),
                "valid_area_ratio": valid_ratio,
                "blur_kernel": 0,
                "blur_angle_deg": 0.0,
            }
        )
        motion_frames.append(frame_meta)

    motion = {
        "clip_id": clip_id,
        "source_video": str(video),
        "source_type": source_type_for_video(video),
        "clip_start_frame": start_frame,
        "frame_stride": frame_stride,
        "clip_length": length,
        "output_width": width,
        "output_height": height,
        "degradation_type": "clean_shake",
        "motion_preset": preset,
        "scene_difficulty": "normal",
        "strength": strength,
        "seed": seed,
        "base_view_transform": {
            "type": "zoom_in_internal_viewport",
            "scale": base_view_scale,
            "matrix": b.tolist(),
        },
        "valid_area_ratio_min": min(valid_ratios),
        "valid_area_ratio_mean": float(np.mean(valid_ratios)),
        "frames": motion_frames,
    }
    write_json(clip_dir / "motion.json", motion)
    contact_sheet(clip_dir)
    report = validate_clip(config_for_validation(width, height, length), clip_dir)
    return {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "source_video": str(video),
        "start_frame": start_frame,
        "frame_stride": frame_stride,
        "clip_length": length,
        "motion_preset": preset,
        "strength": strength,
        "seed": seed,
        "valid_area_ratio_min": report["valid_area_ratio_min"],
        "valid_area_ratio_mean": report["valid_area_ratio_mean"],
        "black_border_detected": report["black_border_detected"],
        "status": report["status"],
        "quality_report_json": str(clip_dir / "quality_report.json"),
    }


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_training_view(output_root: Path, rows: list[dict[str, Any]], val_ratio: float) -> dict[str, Any]:
    passed = [row for row in rows if row["status"] == "pass"]
    if not passed:
        raise RuntimeError("no passing clips were generated")
    val_count = max(1, int(round(len(passed) * val_ratio))) if len(passed) > 1 else 1
    val_ids = {row["clip_id"] for row in passed[-val_count:]}
    counts = {"train": 0, "val": 0}
    for row in passed:
        split = "val" if row["clip_id"] in val_ids else "train"
        if split == "val" and len(passed) == 1:
            split = "train"
        clip_dir = Path(row["clip_dir"])
        for stable_path in sorted((clip_dir / "stable").glob("*.npy")):
            shaky_path = clip_dir / "shaky" / stable_path.name
            rel_name = f"{row['clip_id']}__{stable_path.name}"
            link_or_copy(shaky_path.resolve(), output_root / "paired_frames" / split / "input" / rel_name)
            link_or_copy(stable_path.resolve(), output_root / "paired_frames" / split / "target" / rel_name)
            counts[split] += 1
    if counts["val"] == 0 and counts["train"] > 1:
        input_dir = output_root / "paired_frames" / "train" / "input"
        target_dir = output_root / "paired_frames" / "train" / "target"
        val_input = output_root / "paired_frames" / "val" / "input"
        val_target = output_root / "paired_frames" / "val" / "target"
        last_input = sorted(input_dir.glob("*.npy"))[-1]
        last_target = target_dir / last_input.name
        link_or_copy(last_input.resolve(), val_input / last_input.name)
        link_or_copy(last_target.resolve(), val_target / last_target.name)
        counts["val"] += 1
    return {
        "train_input": str(output_root / "paired_frames" / "train" / "input"),
        "train_target": str(output_root / "paired_frames" / "train" / "target"),
        "val_input": str(output_root / "paired_frames" / "val" / "input"),
        "val_target": str(output_root / "paired_frames" / "val" / "target"),
        "train_pairs": counts["train"],
        "val_pairs": counts["val"],
    }


def build_training_view_from_plan(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"train": 0, "val": 0}
    for row in rows:
        if row["status"] != "pass":
            continue
        split = row.get("split", "train")
        if split not in counts:
            raise ValueError(f"unknown split {split!r} for {row['clip_id']}")
        clip_dir = Path(row["clip_dir"])
        for stable_path in sorted((clip_dir / "stable").glob("*.npy")):
            shaky_path = clip_dir / "shaky" / stable_path.name
            rel_name = f"{row['clip_id']}__{stable_path.name}"
            link_or_copy(shaky_path.resolve(), output_root / "paired_frames" / split / "input" / rel_name)
            link_or_copy(stable_path.resolve(), output_root / "paired_frames" / split / "target" / rel_name)
            counts[split] += 1
    return {
        "train_input": str(output_root / "paired_frames" / "train" / "input"),
        "train_target": str(output_root / "paired_frames" / "train" / "target"),
        "val_input": str(output_root / "paired_frames" / "val" / "input"),
        "val_target": str(output_root / "paired_frames" / "val" / "target"),
        "train_pairs": counts["train"],
        "val_pairs": counts["val"],
    }


def planned_row_from_exception(entry: dict[str, Any], exc: Exception, output_root: Path, seed: int) -> dict[str, Any]:
    return {
        "clip_id": entry["clip_id"],
        "clip_dir": str(output_root / "clips" / entry["clip_id"]),
        "source_video": entry["source_video"],
        "start_frame": entry["start_frame"],
        "frame_stride": entry["frame_stride"],
        "clip_length": entry["clip_length"],
        "motion_preset": entry["motion_preset"],
        "strength": entry.get("strength"),
        "seed": seed,
        "split": entry.get("split", "train"),
        "valid_area_ratio_min": 0.0,
        "valid_area_ratio_mean": 0.0,
        "black_border_detected": True,
        "status": "fail",
        "error": str(exc),
    }


def run_clip_plan(args: argparse.Namespace, ffmpeg: Path, output_root: Path, start_time: float) -> int:
    plan_path = args.clip_plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan["clips"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(Path(entry["source_video"]).expanduser().resolve()), []).append(entry)

    rows: list[dict[str, Any]] = []
    for source_index, (source_video, source_entries) in enumerate(grouped.items()):
        video = Path(source_video)
        needed_frames: list[int] = []
        for entry in source_entries:
            needed_frames.extend(
                entry["start_frame"] + local_idx * entry["frame_stride"]
                for local_idx in range(entry["clip_length"])
            )
        try:
            with tempfile.TemporaryDirectory(prefix="bvi_synth_decode_plan_") as tmp:
                decoded_frames = decode_selected_frames(
                    ffmpeg,
                    video,
                    needed_frames,
                    args.decode_width,
                    args.decode_height,
                    Path(tmp),
                )
                for entry in source_entries:
                    seed = int(entry.get("seed", args.seed + len(rows)))
                    row = synthesize_clip_from_frames(
                        video=video,
                        output_root=output_root,
                        clip_id=entry["clip_id"],
                        start_frame=entry["start_frame"],
                        frame_stride=entry["frame_stride"],
                        length=entry["clip_length"],
                        width=args.decode_width,
                        height=args.decode_height,
                        preset=entry["motion_preset"],
                        strength=entry.get("strength", args.strength),
                        base_view_scale=float(entry.get("base_view_scale", args.base_view_scale)),
                        seed=seed,
                        save_png=args.save_png,
                        decoded_frames=decoded_frames,
                    )
                    row["split"] = entry.get("split", "train")
                    row["plan_source_index"] = source_index
                    rows.append(row)
                    print(f"{row['clip_id']}: status={row['status']} valid_min={row['valid_area_ratio_min']:.6f}")
        except Exception as exc:  # noqa: BLE001
            if not args.skip_failed_clips:
                raise
            for entry in source_entries:
                row = planned_row_from_exception(entry, exc, output_root, args.seed + len(rows))
                rows.append(row)
                print(f"{row['clip_id']}: status=fail valid_min=0.000000")

    training_view = build_training_view_from_plan(output_root, rows)
    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start_time,
        "ffmpeg": str(ffmpeg),
        "source_root": str(args.source_root.expanduser().resolve()),
        "output_root": str(output_root),
        "clip_plan": str(plan_path),
        "plan_summary": plan.get("summary", {}),
        "settings": {
            "clip_length": args.clip_length,
            "decode_width": args.decode_width,
            "decode_height": args.decode_height,
            "base_view_scale": args.base_view_scale,
            "strength": args.strength,
            "seed": args.seed,
        },
        "clip_count": len(rows),
        "pass_count": sum(1 for row in rows if row["status"] == "pass"),
        "fail_count": sum(1 for row in rows if row["status"] != "pass"),
        "clips": rows,
        "training_view": training_view,
    }
    write_json(output_root / args.manifest_name, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["fail_count"] == 0 else 2


def main() -> int:
    args = parse_args()
    start = time.time()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.clip_plan is not None:
        return run_clip_plan(args, ffmpeg, output_root, start)
    sources = select_source_videos(
        video_files(source_root, set(args.exclude_name), args.min_file_age_seconds),
        args.max_videos,
        args.source_selection,
    )
    if not sources:
        raise FileNotFoundError(f"no BVI-AOM video files found under {source_root}")

    rows: list[dict[str, Any]] = []
    for video_idx, video in enumerate(sources):
        for clip_idx in range(args.clips_per_video):
            preset = PRESET_CYCLE[(video_idx + clip_idx) % len(PRESET_CYCLE)]
            start_frame = args.start_frame + clip_idx * args.clip_length * args.frame_stride
            clip_id = safe_clip_id(video, clip_idx, preset)
            try:
                row = synthesize_clip(
                    ffmpeg=ffmpeg,
                    video=video,
                    output_root=output_root,
                    clip_id=clip_id,
                    start_frame=start_frame,
                    frame_stride=args.frame_stride,
                    length=args.clip_length,
                    width=args.decode_width,
                    height=args.decode_height,
                    preset=preset,
                    strength=args.strength,
                    base_view_scale=args.base_view_scale,
                    seed=args.seed + len(rows),
                    save_png=args.save_png,
                )
            except Exception as exc:  # noqa: BLE001
                if not args.skip_failed_clips:
                    raise
                row = {
                    "clip_id": clip_id,
                    "clip_dir": str(output_root / "clips" / clip_id),
                    "source_video": str(video),
                    "start_frame": start_frame,
                    "frame_stride": args.frame_stride,
                    "clip_length": args.clip_length,
                    "motion_preset": preset,
                    "strength": args.strength,
                    "seed": args.seed + len(rows),
                    "valid_area_ratio_min": 0.0,
                    "valid_area_ratio_mean": 0.0,
                    "black_border_detected": True,
                    "status": "fail",
                    "error": str(exc),
                }
            rows.append(row)
            print(f"{clip_id}: status={row['status']} valid_min={row['valid_area_ratio_min']:.6f}")

    training_view = build_training_view(output_root, rows, args.val_ratio)
    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "ffmpeg": str(ffmpeg),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "settings": {
            "max_videos": args.max_videos,
            "source_selection": args.source_selection,
            "clips_per_video": args.clips_per_video,
            "clip_length": args.clip_length,
            "start_frame": args.start_frame,
            "frame_stride": args.frame_stride,
            "decode_width": args.decode_width,
            "decode_height": args.decode_height,
            "base_view_scale": args.base_view_scale,
            "strength": args.strength,
            "seed": args.seed,
        },
        "clip_count": len(rows),
        "pass_count": sum(1 for row in rows if row["status"] == "pass"),
        "fail_count": sum(1 for row in rows if row["status"] != "pass"),
        "clips": rows,
        "training_view": training_view,
    }
    write_json(output_root / args.manifest_name, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["fail_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
