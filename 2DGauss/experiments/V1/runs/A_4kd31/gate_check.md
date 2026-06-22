# A_4kd31 Gate Check

Updated: 2026-06-14 18:53:44 CST

## Static And Smoke Checks

- `py_compile` passed for:
  - `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- TOML parse passed for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/configs/a_4kd31_fast_4kdehaze_a08_a15_transplanted_route_heads_route_only.toml`.
- Checkpoint paths exist:
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Launcher syntax passed:
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/launch_tmux.sh`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/wait_and_launch_a_4kd31_memory_gated_gpu_safe.sh`
- GPU3 dry-run memory gate passed with `free_mib=19686`, `util=100`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.

## Transplant Smoke

- High route head copied from A08 `decoder.2`: max weight diff `0.0`, max bias diff `0.0`.
- Low route head copied from A15 `decoder.2`: max weight diff `0.0`, max bias diff `0.0`.
- Route-only trainable policy enabled: `19718/168406` parameters trainable.

## Runtime

- GPU3 preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183403.log`, exit code `0`.
- GPU3 formal training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183538.log`, exit code `0`.
- Best validation: step `50`, `val_psnr=18.25040839715078`, `val_loss=0.12270757514983416`.

## Evaluation Gates

- Standard test: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_eval_best_val_gpu.json`
  - `psnr_mean=17.770243709681385`
  - `loss_mean=0.12073371022939682`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_task_metrics_best_val_gpu.json`
  - `ssim_mean=0.7060290277004242`
  - `dark_channel_l1_mean=0.14252242375165225`
  - `edge_magnitude_l1_mean=0.15756582029163838`
  - `local_contrast_l1_mean=0.024481982365250586`
  - `chroma_l1_mean=0.03137393491342664`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/a18_record_alignment_best_val_gpu.json`
  - overall gap `-0.07669066613303403 dB`
  - low-route gap `-0.02801579592015501 dB`
  - high-route gap `-0.14121316850824578 dB`
- Route-regime diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/input_std_route_regime_best_val_gpu.json`
  - selected delta `0.012584832096472383`
  - high selected delta `0.019272921736850294`
  - low selected delta `0.007539431139696063`

## Gate Result

Not promoted. A31 improves the high-route gap materially versus A24/A30, but it does not beat A24 aggregate PSNR and low route falls below the required floor. The next gate should preserve low route from A24 while constraining high-route specialization.
