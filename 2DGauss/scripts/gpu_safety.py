#!/usr/bin/env python3

from __future__ import annotations

import os


DEFAULT_ALLOWED_PHYSICAL_GPUS = "0,3"
DEFAULT_VISIBLE_PHYSICAL_GPUS = "3"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def apply_cuda_visible_devices_policy(entrypoint: str) -> None:
    """Restrict CUDA entrypoints to the physical GPUs approved for this server."""

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    allowed = set(
        _split_csv(os.environ.get("TWODGAUSS_ALLOWED_PHYSICAL_GPUS", DEFAULT_ALLOWED_PHYSICAL_GPUS))
    )
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "TWODGAUSS_DEFAULT_CUDA_VISIBLE_DEVICES",
            DEFAULT_VISIBLE_PHYSICAL_GPUS,
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip() in {"", "-1"}:
        return

    requested = _split_csv(visible)
    non_numeric = [item for item in requested if not item.isdigit()]
    disallowed = [item for item in requested if item.isdigit() and item not in allowed]
    if non_numeric or disallowed:
        allowed_text = ",".join(sorted(allowed, key=int))
        raise RuntimeError(
            f"{entrypoint} may only use physical GPU ids {allowed_text}; "
            f"got CUDA_VISIBLE_DEVICES={visible!r}. "
            "Leave it unset to default to physical GPU 3, or set it explicitly to 0 or 3."
        )
    if len(requested) > 1 and os.environ.get("TWODGAUSS_ALLOW_MULTI_GPU_VISIBLE") != "1":
        raise RuntimeError(
            f"{entrypoint} is configured for conservative single-GPU use; "
            f"got CUDA_VISIBLE_DEVICES={visible!r}. "
            "Set it to a single physical GPU id, preferably 3 on this shared server."
        )
