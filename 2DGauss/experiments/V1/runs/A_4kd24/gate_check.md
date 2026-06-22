# Gate Check: V1/T01/A_4kd24

## Deployment Gate

- Created: `2026-06-13 CST`
- Target GPU policy: physical GPU0/GPU3 only.
- Default GPU: physical GPU3.
- Memory gate: `TWODGAUSS_MIN_FREE_MIB=12288`.
- Utilization gate: `TWODGAUSS_MAX_GPU_UTIL=100`; high utilization alone does
  not block launch under the current memory-first rule.
- CUDA memory fraction: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- GPU1/GPU2 remain forbidden.

## Code Gate

- Training entrypoint changed: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- Model changed: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- Config API changed: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- Public exports changed: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`
- New default-off model keys:
  - `model.use_input_std_route_residual`
  - `model.input_std_route_residual_blocks`
  - `model.input_std_route_residual_strength`
  - `model.input_std_route_threshold`
  - `model.input_std_route_softness`
  - `model.input_std_route_direction`
  - `model.input_std_route_learned_gate_weight`
  - `model.input_std_route_detach_gate`
- New default-off loss keys:
  - `loss.input_std_route_gate_weight`
  - `loss.input_std_route_branch_weight`
  - `loss.input_std_route_threshold`
  - `loss.input_std_route_softness`
  - `loss.input_std_route_direction`

## Planned Checks

- `python -m py_compile` for changed Python entrypoints.
- venv tensor smoke for route forward and zero-init behavior.
- TOML parse for the A24 config.
- Checkpoint existence checks for A15 init and A08/A15 teachers.
- `bash -n` for A24 launchers.
- GPU memory-gate dry run on GPU3 or GPU0.
- GPU preflight with `TWODGAUSS_PREFLIGHT_ONLY=1` before full training launch.

## Completed Checks

- `python -m py_compile` for changed Python files: passed.
- venv tensor smoke for route forward keys/shape/zero-init behavior: passed.
- A24 TOML parse and checkpoint/dataset path existence checks: passed.
- `bash -n` for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/launch_tmux.sh`: passed.
- `bash -n` for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/wait_and_launch_a_4kd24_memory_gated_gpu_safe.sh`: passed.
- GPU3 memory-gate dry run: passed with `48516MiB` free memory.
- Formal preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/logs/train_20260613_191253.log`, exit code `0`.
- Formal training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/logs/train_20260613_191503.log`, exit code `0`, completed step `400`.
- Best-val GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_eval_best_val_gpu.json`, exit code `0`.
- Best-val GPU task eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_task_metrics_best_val_gpu.json`, exit code `0`.
- Final GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_eval_final_gpu.json`, exit code `0`.

## Result Gate

- Best validation: step `400`, `val_psnr=18.349088096641417`,
  `val_loss=0.1207316406071186`.
- Best-val fast-test: `psnr_mean=17.790202882742463`,
  `ssim_mean=0.7000000587105751`, `loss_mean=0.12000417917966842`,
  `total_loss_mean=0.1232352689653635`.
- Final fast-test: `psnr_mean=17.558779176036488`,
  `loss_mean=0.12264519166201353`,
  `total_loss_mean=0.1258754152804613`.
- Decision: gate passes as the new best single-checkpoint fast-probe candidate
  and first positive structural conversion result. It does not yet pass as a
  full replacement for the A18 two-checkpoint router on PSNR.
