# Result Analysis: V1/T01/A_4kd15

## Run Status

- Status: completed no-haze-guard EMA995 diagnostic; improves over `A_4kd14` but does not strictly supersede `A_4kd08`.
- Formal preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/logs/train_20260611_205622.log`.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/logs/train_20260611_210055.log`.
- Checkpoint used for evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`.

## Protocol

- Started from `A_4kd08` best-val checkpoint at step `700`.
- Same raw path as `A_4kd14`: low LR `3e-5`, linear LR decay, grad clipping, `ema_decay=0.995`, EMA validation, medium detail gate, and shared demand-control.
- Narrow change relative to `A_4kd14`: `dark_channel_weight 0.010 -> 0.0`, `saturation_weight 0.002 -> 0.0`.
- Same `4KDehaze_local_90_5_5_fast_probe_512` protocol: `512` train / `100` val / `100` test, crop size `512`.
- Physical GPU3 only; `CUDA_VISIBLE_DEVICES=3`; `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- CPU-only standard test eval and task-specific diagnostics after training.

## Training Result

- Preflight from `A_4kd08` best-val: `val_psnr=18.54360101346333`, `val_loss=0.11857988886535167`.
- EMA validation curve:
  - step50: `18.112978635302806`
  - step100: `18.10483285648261`
  - step150: `18.118200110438508`
  - step200: `18.16291624609318`
  - step250: `18.197813098283962`
  - step300: `18.24311035106881`
- Best validation: step `300`, `val_psnr=18.24311035106881`, `val_loss=0.12273863352835178`.
- Training completed with `train_exit_code=0`.

## CPU Test Result

- Standard test: `psnr_mean=17.781017099305146`, `loss_mean=0.12060526199638844`, `total_loss_mean=0.12383682064712048`.
- Task diagnostics:
  - `ssim_mean=0.6972798517346382`
  - `dark_channel_l1_mean=0.14506316002458333`
  - `saturation_l1_mean=0.055948681086301806`
  - `edge_magnitude_l1_mean=0.16086940564215182`
  - `high_frequency_l1_mean=0.021652142368257046`
  - `local_contrast_l1_mean=0.02542852663435042`
  - `luminance_l1_mean=0.1051734821870923`
  - `chroma_l1_mean=0.030280336271971464`

## Comparisons

- Versus `A_4kd08`: `A_4kd15` is still lower on CPU test PSNR by about `0.00812 dB` (`17.781017099305146` vs `17.789133503142946`), but improves loss by about `0.00221`, SSIM by about `0.00252`, dark-channel L1 by about `0.00412`, edge L1 by about `0.00275`, and almost ties chroma.
- Versus `A_4kd14`: `A_4kd15` improves CPU test PSNR by about `0.00602 dB` and loss by about `0.00184`; it also improves dark-channel, edge, high-frequency, local-contrast, and chroma means. SSIM is slightly lower.
- Per-sample PSNR versus `A_4kd08`: `A_4kd15` wins `54/100` samples, with median delta `+0.019215287 dB`, but mean delta remains `-0.008116404 dB`.
- Per-sample PSNR versus `A_4kd14`: `A_4kd15` wins `52/100` samples, with median delta `+0.004699126 dB`.

## Tail Finding

The original A14 review suspected haze guards might be responsible for easy-sample tail damage. `A_4kd15` disproves that as a complete explanation. On the watched A14-worst easy samples, `A_4kd15` is even lower than `A_4kd14`:

- `94_94000077.jpg`: A08 `22.050700589490596`, A14 `21.651874503081974`, A15 `21.529487766937443`
- `94_94000080.jpg`: A08 `21.907239927075185`, A14 `21.51134276506039`, A15 `21.389679075552067`
- `94_94000073.jpg`: A08 `22.331647965008198`, A14 `21.94180384391253`, A15 `21.820913857239628`
- `94_94000084.jpg`: A08 `22.485268367393026`, A14 `22.09922006818685`, A15 `21.98071229821109`
- `94_9400009.jpg`: A08 `21.098742065710418`, A14 `20.714536270981515`, A15 `20.597875619859472`

## Interpretation

Removing haze guards is a small positive step for average test PSNR and loss, but not the strict breakthrough. The gain comes from harder samples and broader L1/proxy improvements, not from repairing the easy-sample PSNR tail. Therefore the next mechanism should not keep only sweeping haze-guard weights. The remaining bottleneck is now more sharply identified as PSNR-aligned easy-sample/low-demand protection: preserve already-good samples while allowing the EMA continuation to help hard samples.
