# Result Analysis: V1/T01/A_4kd14

## Run Status

- Status: completed high-EMA relaxed haze-guard diagnostic; validation improved slightly, CPU test nearly reaches but does not supersede `A_4kd08`.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd14/logs/train_20260611_192115.log`.
- Checkpoint used for evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd14/checkpoints/best_val.pt`.

## Protocol

- Started from `A_4kd08` best-val checkpoint at step `700`.
- Same raw path as `A_4kd13`: same seed, same low LR `3e-5`, same LR decay, same grad clipping, same relaxed haze guards, same medium detail gate, and same shared demand-control.
- Narrow change relative to `A_4kd13`: `ema_decay 0.99 -> 0.995`.
- Same `4KDehaze_local_90_5_5_fast_probe_512` protocol: `512` train / `100` val / `100` test, crop size `512`.
- Physical GPU3 only; `CUDA_VISIBLE_DEVICES=3`; `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- CPU-only standard test eval and task-specific diagnostics after training.

## Training Result

- Preflight from `A_4kd08` best-val: `val_psnr=17.688815785102005`, `val_loss=0.13449757874011994`.
- EMA validation curve:
  - step50: `18.170936188104953`
  - step100: `18.123144210570537`
  - step150: `18.12215815276933`
  - step200: `18.18555240425511`
  - step250: `18.223114904672048`
  - step300: `18.220425418418163`
- Best validation: step `250`, `val_psnr=18.223114904672048`, `val_loss=0.12495512887835503`.
- Training completed with `train_exit_code=0`; peak CUDA reserved memory was about `11.224609375GiB`.

## CPU Test Result

- Standard test: `psnr_mean=17.774995041050754`, `loss_mean=0.12244070835411548`, `total_loss_mean=0.12567226648330687`.
- Task diagnostics:
  - `ssim_mean=0.6976614135503769`
  - `dark_channel_l1_mean=0.14586309671401979`
  - `saturation_l1_mean=0.055857277773320675`
  - `edge_magnitude_l1_mean=0.16153054259717464`
  - `high_frequency_l1_mean=0.021660706866532563`
  - `local_contrast_l1_mean=0.025637199571356177`
  - `luminance_l1_mean=0.10508509073406458`
  - `chroma_l1_mean=0.03105577051639557`

## Interpretation

`A_4kd14` is the best PSNR recovery among the EMA continuation branch so far. It improves over `A_4kd13` by about `+0.04483 dB` CPU test PSNR and narrows the gap to `A_4kd08` from about `0.059 dB` to about `0.014 dB`.

It is still not a strict breakthrough because `A_4kd08` remains higher on CPU test PSNR (`17.789133503142946` vs `17.774995041050754`). It also does not replace `A_4kd13` as the balanced/task candidate: `A_4kd13` keeps higher SSIM and better dark-channel, saturation, edge, high-frequency, local-contrast, and luminance diagnostics, while `A_4kd14` mainly improves PSNR and chroma.

The useful lesson is that EMA decay is the main remaining knob in this low-LR continuation family. Slower EMA (`0.995`) preserves more of the `A_4kd08` fidelity than `0.99`, but the relaxed haze guards still trade away some task proxies and do not cross the strict PSNR boundary. The next viable fast-probe move should either tune around `ema_decay=0.995` with weaker/zero haze guards, or stop squeezing this small fast-probe and move `A_4kd08` plus the best EMA variant to a larger/current split.
