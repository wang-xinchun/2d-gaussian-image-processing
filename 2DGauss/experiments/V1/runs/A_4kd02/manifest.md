# Run Manifest: V1/T01/A_4kd02

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd02`
- Full identifier: `V1/T01/A_4kd02`

## Current Status

Formal low-footprint training and CPU fast-probe test evaluation are complete.
The run used physical GPU3 only, did not use GPU1/GPU2, and did not overwrite
`A_4kd01`.

## Purpose

`A_4kd02` tests a narrow dehaze-prior variant before scaling the current
server old-task line. Compared with `A_4kd01`, it lowers the runtime envelope
for shared GPU safety and adds small task-aware losses that may improve
dehazing/restoration detail without changing the overall V1/T01 training
entrypoint.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Root data asset: `/home/ldj/wxc/baiduwangpan/2D高斯数据集/4KDehaze/4KDehazing`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Split rule: deterministic filename order subset from the local `90/5/5`
  split. This fast-probe view avoids full-split symlink scan stalls during the
  first shared-GPU dehaze-prior probe.

## Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `800`
- Planned batch size: `4`
- Gradient accumulation: `2`
- Training crop: `512`
- Validation crop: `512`
- Validation subset for training-time checks: first `100` validation pairs
- DataLoader workers: `1`
- CUDA memory fraction: `0.30`
- Launch gate: `>=8192MiB` free memory; high utilization is allowed but memory
  stays self-limited.
- Allowed physical GPUs: `0` or `3` only
- Default preferred physical GPU: `3`

## Launch State

The run-local config exists at:

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/configs/a_4kd02_short_4kdehaze_dehaze_prior_safe_probe.toml`

Formal GPU preflight, formal training, and CPU test evaluation have completed.
GPU1/GPU2 were not used for this run.

## 2026-06-10 Completion Summary

- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232733.log`.
- Exit code: `train_exit_code=0`.
- Completed steps: `800`.
- Training wall time: about `33.43` minutes.
- Peak CUDA reserved memory observed in metrics: about `10.21875GiB`.
- Best/final training-time validation: step `800`,
  `val_loss=0.12707203164696693`, `val_psnr=17.529067811582326`,
  `selection_metric=val_loss`.
- Best summary:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/best_val_summary.json`.
- Checkpoints present: `best_val.pt`, `final.pt`, `latest.pt`, and
  `step_000100.pt` through `step_000800.pt`; the final/best/latest checkpoint
  step is `800`.
- CPU test eval on `best_val.pt`:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/test_eval_best_val_cpu.json`.
  It covers `100` test samples at `crop_size=512` with
  `psnr_mean=15.909636046480303` and `loss_mean=0.14773030210286378`.
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/test_task_metrics_best_val_cpu.json`.
  Summary metrics include `ssim_mean=0.6678106400370598`,
  `edge_magnitude_l1_mean=0.14476336620748043`,
  `high_frequency_l1_mean=0.02070328968577087`,
  `dark_channel_l1_mean=0.138143544010818`.
- Cleanup check: no `A_4kd02`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, or `evaluate_task_specific_metrics.py` process
  remained, and no project tmux session remained.
- Boundary: this is a `512/100/100` fast-probe subset with `512` crop
  evaluation. It is not an official/full split benchmark and must not be
  compared directly to historical full-image `UHD_deblur` or accepted
  `UHD_haze` references.

## 2026-06-09 Low-Footprint Launch State

- User clarified that we should not wait for large free VRAM on the shared box;
  instead, fit into GPU0/3 with low memory pressure while never killing or
  displacing external processes.
- The old high-free-memory wait session `2dgauss_wait_a_4kd02_gpu_safe` was
  stopped.
- Fast-probe dataset construction passed: `construct 0.021s`, first sample read
  `7.198s`, tensor shapes `(3,512,512)`.
- Formal preflight passed on physical GPU3:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232309.log`.
  It used `train_samples=8`, `val_samples=100`, batch shape `(4,3,512,512)`,
  and exited with code `0`.
- Formal training session:
  `2dgauss_a_4kd02_4kdehaze_dehaze_prior_safe_probe`.
- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232733.log`.
- Later completion superseded this live snapshot: the run reached step `800`
  with `train_exit_code=0`.
- First training-time validation completed at step `100`: `val_loss=0.19756620861589907`,
  `val_psnr=14.261986201609318`, `val_batches=100`.
- Checkpoints now present: `best_val.pt`, `latest.pt`, and `step_000100.pt`.
