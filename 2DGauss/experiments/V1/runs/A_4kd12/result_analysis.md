# Result Analysis: V1/T01/A_4kd12

## Run Status

- Status: completed EMA diagnostic and balanced fast-probe candidate; not a strict PSNR supersede of `A_4kd08`.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd12/logs/train_20260611_181125.log`.
- Checkpoint used for evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd12/checkpoints/best_val.pt`.

## Protocol

- Started from `A_4kd08` best-val checkpoint at step `700`.
- Same raw training path as `A_4kd11`, with `training.ema_decay=0.99` and `training.ema_eval=true`.
- EMA weights are used for validation and `best_val.pt`; raw model training remains unchanged.
- Same `4KDehaze_local_90_5_5_fast_probe_512` protocol as `A_4kd08`: `512` train / `100` val / `100` test, crop size `512`.
- Physical GPU3 only; `CUDA_VISIBLE_DEVICES=3`; `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- CPU-only standard test eval and task-specific diagnostics after training.

## Training Result

- Preflight from `A_4kd08` best-val: `val_psnr=17.68866416700685`.
- EMA validation curve:
  - step50: `18.179347230366535`
  - step100: `18.075476685031347`
  - step150: `18.060962691761816`
  - step200: `18.17683881390156`
  - step250: `18.222970922288727`
  - step300: `18.164094433925065`
- Best validation: step `250`, `val_psnr=18.222970922288727`, `val_loss=0.12504795756191014`.

## CPU Test Result

- Standard test: `psnr_mean=17.729771630807832`, `loss_mean=0.12298042900860309`, `total_loss_mean=0.12621172420680524`.
- Task diagnostics:
  - `ssim_mean=0.6995948380231858`
  - `dark_channel_l1_mean=0.143662343993783`
  - `saturation_l1_mean=0.0554562546685338`
  - `edge_l1_mean=0.1598578527569771`
  - `high_frequency_l1_mean=0.02161801033653319`
  - `local_contrast_l1_mean=0.025125211514532567`
  - `luminance_l1_mean=0.10506133791059255`
  - `chroma_l1_mean=0.03179603109136224`

## Interpretation

`A_4kd12` stabilizes the raw `A_4kd11` continuation: validation no longer spikes and collapses, and CPU test PSNR recovers from `17.36895205643582` to `17.729771630807832`. It still does not strictly supersede `A_4kd08` on PSNR (`17.789133503142946`), but it improves SSIM and most task proxies while nearly matching the current PSNR candidate.

Current status: keep `A_4kd08` as the strict fast-probe PSNR candidate; keep `A_4kd12` as the balanced EMA/Pareto candidate and the strongest evidence that low-LR haze-guard continuation should be paired with averaging.
