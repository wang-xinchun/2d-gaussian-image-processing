from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


@dataclass(frozen=True)
class PairedImageSample:
    name: str
    input_path: Path
    target_path: Path


def _image_files(root: Path) -> List[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def _index_by_name(paths: Sequence[Path]) -> Dict[str, Path]:
    return {path.name: path for path in paths}


def _index_by_stem(paths: Sequence[Path]) -> Dict[str, Path]:
    return {path.stem: path for path in paths}


def _read_rgb(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB npy frame, got shape {loaded.shape}: {path}")
        if loaded.dtype == np.uint16:
            array = loaded.astype(np.float32) / 65535.0
        elif loaded.dtype == np.uint8:
            array = loaded.astype(np.float32) / 255.0
        else:
            array = loaded.astype(np.float32)
            if array.max(initial=0.0) > 1.0:
                array /= 65535.0
        array = np.clip(array, 0.0, 1.0)
    else:
        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _paired_random_crop(input_tensor: torch.Tensor, target_tensor: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0:
        return input_tensor, target_tensor

    height = min(input_tensor.shape[-2], target_tensor.shape[-2])
    width = min(input_tensor.shape[-1], target_tensor.shape[-1])
    input_tensor = input_tensor[..., :height, :width]
    target_tensor = target_tensor[..., :height, :width]

    if height <= crop_size or width <= crop_size:
        return input_tensor, target_tensor

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        input_tensor[..., top : top + crop_size, left : left + crop_size],
        target_tensor[..., top : top + crop_size, left : left + crop_size],
    )


def _paired_random_crop_for_input_std_route(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    crop_size: int,
    *,
    threshold: float,
    target_high: bool,
    attempts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0 or attempts <= 1:
        return _paired_random_crop(input_tensor, target_tensor, crop_size)

    height = min(input_tensor.shape[-2], target_tensor.shape[-2])
    width = min(input_tensor.shape[-1], target_tensor.shape[-1])
    input_tensor = input_tensor[..., :height, :width]
    target_tensor = target_tensor[..., :height, :width]

    if height <= crop_size or width <= crop_size:
        return input_tensor, target_tensor

    best: tuple[torch.Tensor, torch.Tensor] | None = None
    best_margin = float("inf")
    for _ in range(attempts):
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)
        input_crop = input_tensor[..., top : top + crop_size, left : left + crop_size]
        target_crop = target_tensor[..., top : top + crop_size, left : left + crop_size]
        crop_std = float(input_crop.std(unbiased=False))
        if (target_high and crop_std >= threshold) or (not target_high and crop_std < threshold):
            return input_crop, target_crop
        margin = (threshold - crop_std) if target_high else (crop_std - threshold)
        if margin < best_margin:
            best_margin = margin
            best = (input_crop, target_crop)
    if best is None:
        return _paired_random_crop(input_tensor, target_tensor, crop_size)
    return best


def _paired_center_crop(input_tensor: torch.Tensor, target_tensor: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0:
        return input_tensor, target_tensor

    height = min(input_tensor.shape[-2], target_tensor.shape[-2])
    width = min(input_tensor.shape[-1], target_tensor.shape[-1])
    input_tensor = input_tensor[..., :height, :width]
    target_tensor = target_tensor[..., :height, :width]

    crop_h = min(height, crop_size)
    crop_w = min(width, crop_size)
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return (
        input_tensor[..., top : top + crop_h, left : left + crop_w],
        target_tensor[..., top : top + crop_h, left : left + crop_w],
    )


def _paired_random_crop_with_context(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    supervision_crop_size: int,
    context_crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if supervision_crop_size <= 0 or context_crop_size <= 0:
        return _paired_random_crop(input_tensor, target_tensor, max(supervision_crop_size, context_crop_size))

    height = min(input_tensor.shape[-2], target_tensor.shape[-2])
    width = min(input_tensor.shape[-1], target_tensor.shape[-1])
    input_tensor = input_tensor[..., :height, :width]
    target_tensor = target_tensor[..., :height, :width]

    if height <= context_crop_size or width <= context_crop_size:
        input_crop = input_tensor
        target_crop = target_tensor
        return input_crop, target_crop

    top = random.randint(0, height - context_crop_size)
    left = random.randint(0, width - context_crop_size)
    input_crop = input_tensor[..., top : top + context_crop_size, left : left + context_crop_size]
    target_crop = target_tensor[..., top : top + context_crop_size, left : left + context_crop_size]
    return input_crop, target_crop


def _paired_flips(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        input_tensor = torch.flip(input_tensor, dims=(-1,))
        target_tensor = torch.flip(target_tensor, dims=(-1,))
    if random.random() < 0.5:
        input_tensor = torch.flip(input_tensor, dims=(-2,))
        target_tensor = torch.flip(target_tensor, dims=(-2,))
    return input_tensor, target_tensor


class PairedImageDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        crop_size: int = 0,
        training: bool = True,
        augment_flip: bool = False,
        offset: int = 0,
        limit: int = 0,
        context_crop_size: int = 0,
        supervision_crop_size: int = 0,
        route_crop_threshold: float | None = None,
        route_crop_attempts: int = 0,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.crop_size = crop_size
        self.training = training
        self.augment_flip = augment_flip
        self.context_crop_size = context_crop_size
        self.supervision_crop_size = supervision_crop_size
        self.route_crop_threshold = route_crop_threshold
        self.route_crop_attempts = route_crop_attempts

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")
        if not self.target_dir.is_dir():
            raise FileNotFoundError(f"target_dir does not exist: {self.target_dir}")

        input_paths = _image_files(self.input_dir)
        target_paths = _image_files(self.target_dir)
        targets_by_name = _index_by_name(target_paths)
        targets_by_stem = _index_by_stem(target_paths)

        samples: List[PairedImageSample] = []
        for input_path in input_paths:
            target_path = targets_by_name.get(input_path.name) or targets_by_stem.get(input_path.stem)
            if target_path is not None:
                samples.append(PairedImageSample(input_path.name, input_path, target_path))

        if offset > 0:
            samples = samples[offset:]
        if limit > 0:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"no paired images found between {self.input_dir} and {self.target_dir}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, torch.Tensor | str]:
        route_target: int | None = None
        if isinstance(index, tuple):
            index, route_target = index
        sample = self.samples[index]
        input_tensor = _read_rgb(sample.input_path)
        target_tensor = _read_rgb(sample.target_path)

        if self.training:
            if self.context_crop_size > 0 and self.supervision_crop_size > 0:
                input_tensor, target_tensor = _paired_random_crop_with_context(
                    input_tensor,
                    target_tensor,
                    self.supervision_crop_size,
                    self.context_crop_size,
                )
            elif (
                route_target is not None
                and self.route_crop_threshold is not None
                and self.route_crop_attempts > 1
            ):
                input_tensor, target_tensor = _paired_random_crop_for_input_std_route(
                    input_tensor,
                    target_tensor,
                    self.crop_size,
                    threshold=self.route_crop_threshold,
                    target_high=bool(route_target),
                    attempts=self.route_crop_attempts,
                )
            else:
                input_tensor, target_tensor = _paired_random_crop(input_tensor, target_tensor, self.crop_size)
            if self.augment_flip:
                input_tensor, target_tensor = _paired_flips(input_tensor, target_tensor)
        else:
            input_tensor, target_tensor = _paired_center_crop(input_tensor, target_tensor, self.crop_size)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": sample.name,
        }
