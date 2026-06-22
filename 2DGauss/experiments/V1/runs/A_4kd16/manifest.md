# Run Manifest: V1/T01/A_4kd16

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd16`
- Full identifier: `V1/T01/A_4kd16`

## Purpose

`A_4kd16` tests the mechanism route selected after `A_4kd15`: stop sweeping haze
guards and add PSNR-aligned/easy-sample protection. The goal is to keep A15's
hard-sample and loss/proxy gains while preventing overcorrection on low-demand,
already-good regions.

## Narrow Variable

Compared with `A_4kd15`, this run adds:

- `loss.mse_weight=0.25`
- `loss.low_demand_identity_weight=0.03`
- `loss.low_demand_identity_threshold=0.06`
- `loss.low_demand_identity_softness=0.02`

It keeps `A_4kd15`'s no-haze-guard setup, strict `A_4kd08` best-val
initialization, low LR, LR decay, grad clipping, `ema_decay=0.995`, and EMA
validation.

## Candidate Rule

The run can supersede `A_4kd08` only if CPU fast-probe test PSNR reaches or
exceeds `17.789133503142946`. The diagnostic target is watched easy-tail repair:
`94_94000077/80/73/84/9` and neighboring high-PSNR samples should not remain
below the A14/A15 tail.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.
