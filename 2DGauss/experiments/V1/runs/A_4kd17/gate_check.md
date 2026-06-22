# Gate Check: V1/T01/A_4kd17

Status: static checks passed; formal GPU preflight passed; formal training queued behind the elevated GPU0/GPU3 safety gate.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd17`.
- [x] Full run identifier: `V1/T01/A_4kd17`.
- [x] Run folder: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd17`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`, `A_4kd15`, `A_4kd16`, and earlier diagnostics.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [x] Formal preflight data load.

## Runtime Check

- [x] Planned batch size: `2`.
- [x] Planned gradient accumulation: `4`.
- [x] Planned workers: `1`.
- [x] Planned train crop: `512`.
- [x] Planned validation crop: `512`.
- [x] Planned max steps: `300`.
- [x] Planned wall-clock cap: `45` minutes.
- [x] Formal launch free-memory floor: `12288MiB`.
- [x] Best checkpoint selection metric: `val_psnr`.
- [x] Loads `A_4kd08` best-val checkpoint strictly as init.
- [x] Loads `A_4kd08` best-val checkpoint strictly as frozen teacher.
- [x] Uses low LR `3e-5`, linear decay to `0.25x`, and `grad_clip_norm=1.0`.
- [x] Uses EMA validation/checkpointing with `ema_decay=0.995`.
- [x] Only physical GPU `0` or `3` may be used.
- [x] GPU3 is preferred; GPU0 is an allowed fallback when it passes the same safety gate.
- [x] GPU1/GPU2 are forbidden.
- [x] Formal GPU preflight.
- [ ] Formal launch queued behind `24576MiB` free-memory gate.
- [ ] CPU standard test eval.
- [ ] CPU task-specific diagnostics.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd08` medium detail gate and blended shared demand-control.
- [x] Does not enable reliability gating.
- [x] Keeps no haze guards: `dark_channel_weight=0.0`, `saturation_weight=0.0`.
- [x] Adds A08 teacher consistency: `weight=0.05`, `threshold=0.08`, `softness=0.02`.
- [x] Does not enable input identity protection.
- [x] Does not enable MSE in this run.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3, but use GPU0 if it passes the
same free-memory gate. Do not kill or displace external processes. High
utilization is acceptable for this low-footprint probe only if free memory
stays above the configured floor and the project self-limits CUDA memory.

## Static Check Result

- TOML parse and key checks passed for template/run-local configs.
- `scripts/train_v1_t01.py`, `scripts/evaluate_v1_t01.py`, `scripts/evaluate_task_specific_metrics.py`, and `scripts/gpu_safety.py` compile under the project venv.
- Teacher consistency random-tensor smoke check passed.
- `launch_tmux.sh` and `wait_and_launch_a_4kd17_first_available_gpu_safe.sh` pass `bash -n`.
- GPU0 dry-run passed at the standard `12288MiB` prelaunch floor; GPU3 dry-run correctly refused because free memory was below the floor.

## Preflight Result

- Formal preflight ran on physical GPU0 with `CUDA_VISIBLE_DEVICES=0`, `TWODGAUSS_ALLOWED_PHYSICAL_GPUS=0,3`, and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd17/logs/train_20260611_222116.log`.
- `train_exit_code=0`.
- Strict init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`, step `700`.
- Strict teacher checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`, step `700`.
- Teacher diagnostics: `teacher_consistency_mask_mean=0.07771717011928558`, `teacher_consistency_error_mean=0.1831231415271759`, `loss_teacher_consistency=0.0010173976188525558`, weighted `5.086988312541507e-05`.
- Validation: `val_psnr=17.899931849961654`, `val_loss=0.12849370319396258`, `val_batches=100`.

## Formal Queue

- Formal training is queued in tmux session `2dgauss_wait_a_4kd17_formal_gpu_safe`.
- Queue log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd17/logs/wait_formal_20260611_222336.log`.
- The formal queue uses elevated `TWODGAUSS_MIN_FREE_MIB=24576` because preflight showed GPU0 batch2 leaves too little post-launch memory margin when external processes are present.
- Initial queue refusal: GPU3 free `9304MiB < 24576MiB`; GPU0 free `17118MiB < 24576MiB`.
