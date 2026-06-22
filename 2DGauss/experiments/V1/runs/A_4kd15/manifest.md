# Run Manifest: V1/T01/A_4kd15

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd15`
- Full identifier: `V1/T01/A_4kd15`

## Purpose

`A_4kd15` tests the main hypothesis from the A_4kd14 review: the EMA
continuation helps hard samples but may over-correct easy samples. This run keeps
the A_4kd14 high-EMA setup and removes the haze-side guard losses to check
whether those guards are responsible for the easy-sample PSNR tail.

## Narrow Variable

Compared with `A_4kd14`, this run changes only:

- `loss.dark_channel_weight`: `0.010 -> 0.0`
- `loss.saturation_weight`: `0.002 -> 0.0`

It keeps the same strict `A_4kd08` best-val initialization, medium detail gate,
low LR, LR decay, grad clipping, `ema_decay=0.995`, and EMA validation.

## Candidate Rule

The run can supersede `A_4kd08` only if CPU fast-probe test PSNR reaches or
exceeds `17.789133503142946`. The specific diagnostic target is to improve the
easy-sample tail that hurt `A_4kd14` while preserving most of A_4kd14's hard-sample
gains and SSIM/loss improvements.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.
