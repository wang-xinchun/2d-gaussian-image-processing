# Run Manifest: V1/T01/A_4kd12

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd12`
- Full identifier: `V1/T01/A_4kd12`

## Purpose

`A_4kd12` isolates whether EMA validation/checkpointing can turn the useful
`A_4kd11` tradeoff into a stronger fast-probe candidate. `A_4kd11` improved
SSIM and most task-aware diagnostics over `A_4kd08`, but its best-val checkpoint
lost test PSNR. This run keeps the same raw training path and uses EMA weights
only for validation and `best_val.pt`.

## Narrow Variable

Compared with `A_4kd11`, this run changes only:

- `training.ema_decay = 0.99`
- `training.ema_eval = true`

It keeps the same seed, same low-LR continuation from `A_4kd08` best-val,
same light haze-side guards, same linear LR decay, and same grad clipping.

## Candidate Rule

This run can supersede `A_4kd08` only if CPU test PSNR is at or above
`17.789133503142946` while preserving the task-diagnostic improvements seen in
`A_4kd11`, especially dark-channel, saturation, edge/high-frequency, and local
contrast. Otherwise it is an EMA diagnostic.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.

## Result Summary

- Formal GPU3 preflight and `300`-step training completed with `train_exit_code=0`.
- Best EMA validation: step `250`, `val_psnr=18.222970922288727`, `val_loss=0.12504795756191014`.
- CPU test: `psnr_mean=17.729771630807832`, `loss_mean=0.12298042900860309`, `ssim_mean=0.6995948380231858`.
- Conclusion: not a strict PSNR supersede of `A_4kd08`, but a balanced EMA/Pareto candidate. EMA keeps most `A_4kd11` task-proxy gains while recovering test PSNR close to `A_4kd08`.
