# Gate Check: V1/T01/A_4kd02

Status: formal preflight, low-footprint GPU3 training, and CPU fast-probe test evaluation completed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd02`.
- [x] Full run identifier: `V1/T01/A_4kd02`.
- [x] Run folder: `experiments/V1/runs/A_4kd02`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd01` and does not overwrite it.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [x] Dataset comes from a deterministic fast-probe subset of the current-server `90/5/5` split view.
- [x] Formal preflight data load.

## Runtime Check

- [x] Planned batch size: `4`, lowered from `A_4kd01` batch `16` for shared GPU safety.
- [x] Planned gradient accumulation: `2`.
- [x] Planned workers: `1`, lowered from `A_4kd01` workers `8`.
- [x] Planned train crop: `512`.
- [x] Planned validation crop: `512`.
- [x] Planned max steps: `800`.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Only physical GPU `0` or `3` may be used.
- [x] Default launch preference is physical GPU `3`.
- [x] GPU1/GPU2 are forbidden.
- [x] Launcher must refuse too-low-free-memory states while allowing high utilization for this low-footprint squeeze run.
- [x] Formal GPU preflight.
- [x] Formal launch.
- [x] Formal training completion.
- [x] CPU test evaluation.
- [x] Cleanup check.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Enables conservative Gaussian feature refinement.
- [x] Adds small dehaze/restoration priors: high-frequency, local-contrast, and dark-channel losses.
- [x] Task-aware loss weights remain below the conservative caps enforced by `train_v1_t01.py`.

## Current Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this probe, but the project must
keep its own CUDA memory fraction low and refuse launch below the configured
free-memory floor.

## 2026-06-10 Completion Gate

- Formal training completed with `train_exit_code=0` in
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232733.log`.
- Final step: `800`.
- Peak CUDA reserved memory recorded in metrics: about `10.21875GiB`.
- Best/final validation at step `800`: `val_loss=0.12707203164696693`,
  `val_psnr=17.529067811582326`, selected by `val_loss`.
- Checkpoint gate passed: `best_val.pt`, `final.pt`, `latest.pt`, and
  `step_000800.pt` all exist with checkpoint step `800`.
- CPU test eval on `best_val.pt` completed for `100` samples at `crop_size=512`:
  `psnr_mean=15.909636046480303`, `loss_mean=0.14773030210286378`.
- CPU task-specific diagnostics completed:
  `ssim_mean=0.6678106400370598`,
  `edge_magnitude_l1_mean=0.14476336620748043`,
  `high_frequency_l1_mean=0.02070328968577087`,
  `dark_channel_l1_mean=0.138143544010818`.
- Safety gate held through cleanup: project compute only used physical GPU3
  during training; GPU1/GPU2 were not used; no external process was killed or
  displaced; no `A_4kd02`/train/eval process or tmux session remained after
  completion.
- Interpretation gate: this result is a low-footprint fast-probe subset result,
  not an official/full split benchmark.

## 2026-06-09 Low-Footprint Live Gate And Launch Snapshot

- Initial high-free-memory queue policy was superseded after the user clarified that large free VRAM should not be expected on this shared server.
- Active launch threshold: `free >= 8192MiB`, `util <= 100%`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- Fast-probe data read passed before launch: dataset construction `0.021s`, first sample read `7.198s`, tensors `(3,512,512)`.
- Formal preflight passed on physical GPU3 in
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232309.log`
  with `train_exit_code=0`, `val_samples=100`, `val_loss=0.3574191193282604`, and
  `val_psnr=9.269584141729382`.
- Formal training was launched in tmux session `2dgauss_a_4kd02_4kdehaze_dehaze_prior_safe_probe`.
- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232733.log`.
- Later completion superseded this live snapshot: the run reached step `800`
  with `train_exit_code=0`.
- First validation/checkpoint gate passed at step `100`: `val_loss=0.19756620861589907`,
  `val_psnr=14.261986201609318`, and checkpoints `best_val.pt`, `latest.pt`,
  `step_000100.pt` were created.
- No GPU1/GPU2 use and no external process kill/displacement.
