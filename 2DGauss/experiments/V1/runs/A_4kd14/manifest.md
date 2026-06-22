# Run Manifest: V1/T01/A_4kd14

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd14`
- Full identifier: `V1/T01/A_4kd14`

## Purpose

`A_4kd14` tests whether a slower EMA can recover the remaining PSNR gap to
`A_4kd08`. `A_4kd13` showed that relaxing haze-side guards only gives a tiny
PSNR gain over `A_4kd12`, so this run keeps the relaxed guards and changes only
the EMA decay.

## Narrow Variable

Compared with `A_4kd13`, this run changes only:

- `training.ema_decay`: `0.99 -> 0.995`

It keeps the same seed, same strict `A_4kd08` best-val initialization, same
medium detail gate, same low LR, same LR decay, same grad clipping, and same
relaxed haze guards (`dark_channel_weight=0.010`, `saturation_weight=0.002`).

## Candidate Rule

This run can supersede `A_4kd08` only if CPU fast-probe test PSNR is at or above
`17.789133503142946` without losing the main `A_4kd12` benefits in SSIM and
task proxies. If it remains below `A_4kd08` but improves the Pareto tradeoff,
it is a diagnostic for the next guard/EMA setting.

## Result

`A_4kd14` completed the full fast-probe loop. Best EMA validation was step `250`
with `val_psnr=18.223114904672048` and `val_loss=0.12495512887835503`. CPU
test PSNR was `17.774995041050754`, which is the closest EMA-continuation result
to `A_4kd08` so far but remains below `A_4kd08` by about `0.014 dB`. It is
therefore retained as a high-EMA diagnostic, not a strict PSNR replacement.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.
