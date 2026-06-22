from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    if height > 1 and width > 1:
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


def crop_to_size(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    height, width = size
    return x[..., :height, :width]
