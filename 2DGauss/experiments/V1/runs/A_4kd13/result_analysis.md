# Result Analysis: V1/T01/A_4kd13

## Run Status

- Status: completed relaxed haze-guard EMA diagnostic; not a strict PSNR breakthrough.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd13/logs/train_20260611_185453.log`.
- Checkpoint used for evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd13/checkpoints/best_val.pt`.

## Protocol

- Started from `A_4kd08` best-val checkpoint at step `700`.
- Same raw path as `A_4kd12`, same seed, same low LR, same LR decay, same grad clipping, same EMA validation/checkpointing.
- Only changed guard weights relative to `A_4kd12`: `dark_channel_weight 0.012 -> 0.010`, `saturation_weight 0.004 -> 0.002`.
- Same `4KDehaze_local_90_5_5_fast_probe_512` protocol: `512` train / `100` val / `100` test, crop size `512`.
- Physical GPU3 only; `CUDA_VISIBLE_DEVICES=3`; `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- CPU-only standard test eval and task-specific diagnostics after training.

## Training Result

- Preflight from `A_4kd08` best-val: `val_psnr=17.68881486831974`, `val_loss=0.13449759554117918`.
- EMA validation curve:
  - step50: `18.17918788025536`
  - step100: `18.075806715469955`
  - step150: `18.061406732308868`
  - step200: `18.17622578566669`
  - step250: `18.222453652339333`
  - step300: `18.16303874035927`
- Best validation: step `250`, `val_psnr=18.222453652339333`, `val_loss=0.12469094827771186`.
- Peak CUDA reserved memory: about `11.2265625GiB`.

## CPU Test Result

- Standard test: `psnr_mean=17.73016814214326`, `loss_mean=0.12258190430700779`, `total_loss_mean=0.1258132006973028`.
- Task diagnostics:
  - `ssim_mean=0.6996025824546814`
  - `dark_channel_l1_mean=0.1437134078145027`
  - `saturation_l1_mean=0.05548523813486099`
  - `edge_magnitude_l1_mean=0.15986251048743724`
  - `high_frequency_l1_mean=0.021618296522647143`
  - `local_contrast_l1_mean=0.02512654896825552`
  - `luminance_l1_mean=0.1050608617067337`
  - `chroma_l1_mean=0.031774937603622676`

## Interpretation

`A_4kd13` gives a tiny improvement over `A_4kd12` on CPU test PSNR, loss, SSIM, luminance, and chroma, but the change is far too small to count as a breakthrough. It still remains below `A_4kd08` on strict PSNR (`17.73016814214326` vs `17.789133503142946`).

The useful lesson is that the remaining PSNR gap is not primarily controlled by the small haze-guard weights tested here. The next promising knob is EMA behavior: use a slower EMA decay so the selected checkpoint stays closer to the `A_4kd08` initialization while still absorbing the useful low-LR continuation.
