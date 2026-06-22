# Run Manifest: V1/T01/A_4kd13

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd13`
- Full identifier: `V1/T01/A_4kd13`

## Purpose

`A_4kd13` tests whether the `A_4kd12` EMA-stabilized continuation can recover
the remaining PSNR gap to `A_4kd08` by relaxing haze-side guards. `A_4kd12`
nearly matched `A_4kd08` PSNR and improved SSIM/task proxies, but still missed
the strict PSNR candidate by about `0.0594 dB`.

## Narrow Variable

Compared with `A_4kd12`, this run changes only the guard weights:

- `loss.dark_channel_weight`: `0.012 -> 0.010`
- `loss.saturation_weight`: `0.004 -> 0.002`

It keeps the same seed, same strict `A_4kd08` best-val initialization, same
medium detail gate, same low LR, same LR decay, same grad clipping, and same
EMA validation/checkpointing (`ema_decay=0.99`, `ema_eval=true`).

## Candidate Rule

This run can supersede `A_4kd08` only if CPU fast-probe test PSNR is at or above
`17.789133503142946` without losing the main `A_4kd12` benefits in SSIM and
task proxies. If it remains below `A_4kd08` but improves the Pareto tradeoff,
it is a diagnostic for the next guard/EMA setting.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.

## Result Summary

- Formal preflight and `300`-step GPU3 training completed with `train_exit_code=0`.
- Best EMA validation: step `250`, `val_psnr=18.222453652339333`, `val_loss=0.12469094827771186`.
- CPU test: `psnr_mean=17.73016814214326`, `loss_mean=0.12258190430700779`, `ssim_mean=0.6996025824546814`.
- Conclusion: marginally better than `A_4kd12` on PSNR/loss/SSIM, but not a strict PSNR breakthrough over `A_4kd08`.
