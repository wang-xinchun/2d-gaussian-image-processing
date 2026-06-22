# Result Analysis: V1/T01/A_4kd11

## Run Status

- Status: completed diagnostic; not promoted.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd11/logs/train_20260611_154017.log`.
- Earlier failed launch retained for audit: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd11/logs/train_20260611_153858.log`.
- Checkpoint used for evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd11/checkpoints/best_val.pt`.

## Protocol

- Started from `A_4kd08` best-val checkpoint at step `700`.
- Same `4KDehaze_local_90_5_5_fast_probe_512` protocol as `A_4kd08`: `512` train / `100` val / `100` test, crop size `512`.
- Physical GPU3 only; `CUDA_VISIBLE_DEVICES=3`; `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- CPU-only standard test eval and task-specific diagnostics after training.

## Training Result

- Preflight from `A_4kd08` best-val: `val_psnr=17.688662408469433`, `val_loss=0.13490331169217826`.
- Validation curve:
  - step50: `17.311713320301973`
  - step100: `16.33763964020725`
  - step150: `18.20259826088991`
  - step200: `18.456089733874084`
  - step250: `16.583666035177327`
  - step300: `16.319475088266117`
- Best validation: step `200`, `val_psnr=18.456089733874084`, `val_loss=0.11976054381579161`.

## CPU Test Result

- Standard test: `psnr_mean=17.36895205643582`, `loss_mean=0.12583659052848817`, `total_loss_mean=0.12906743220984937`.
- Task diagnostics:
  - `ssim_mean=0.6967320647835732`
  - `dark_channel_l1_mean=0.13161039672791958`
  - `saturation_l1_mean=0.0539228655770421`
  - `edge_l1_mean=0.15448181137442588`
  - `high_frequency_l1_mean=0.021592021659016608`
  - `local_contrast_l1_mean=0.02341419655829668`
  - `luminance_l1_mean=0.10826870784163475`
  - `chroma_l1_mean=0.03478097889572382`

## Interpretation

`A_4kd11` proves that a low-LR haze-guard continuation can improve validation peak and many task-specific proxies, especially dark-channel, saturation, edge, high-frequency, and local contrast. It does not supersede `A_4kd08` because the best-val checkpoint loses CPU test PSNR (`17.36895205643582` vs `17.789133503142946`) and has a highly volatile validation trajectory.

The useful lesson is not to use raw `A_4kd11` as the next candidate. The useful lesson is that the haze-guard continuation needs averaging or a more robust selection rule before it can be trusted.
