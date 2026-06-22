# Gate Check: V1/T01/A_4kd23

## Deployment Gate

- Created: `2026-06-12 CST`
- Target GPU policy: physical GPU0/GPU3 only.
- Default GPU: physical GPU3.
- Memory gate: `TWODGAUSS_MIN_FREE_MIB=12288`.
- Utilization gate: `TWODGAUSS_MAX_GPU_UTIL=100`; high utilization alone does
  not block launch under the user's current rule.
- CUDA memory fraction: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- GPU1/GPU2 remain forbidden.

## Code Gate

- Training entrypoint changed: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- New default-off loss keys:
  - `loss.router_distillation_weight`
  - `loss.router_distillation_checkpoint_a`
  - `loss.router_distillation_checkpoint_b`
  - `loss.router_distillation_feature`
  - `loss.router_distillation_direction`
  - `loss.router_distillation_threshold`
  - `loss.router_distillation_softness`
- Backward compatibility: old configs keep `router_distillation_weight=0.0` by
  default and do not load extra teachers.

## Planned Checks

- `python -m py_compile scripts/train_v1_t01.py`
- TOML parse for the A23 config.
- CPU tensor smoke for `compute_router_distillation_loss`.
- `launch_tmux.sh` dry-run under GPU3 memory gate.
- GPU preflight with `TWODGAUSS_PREFLIGHT_ONLY=1` before full training launch.

## Completed Checks

- `python -m py_compile scripts/train_v1_t01.py`: passed.
- Run-local TOML parse and checkpoint existence checks: passed.
- CPU tensor smoke for `compute_router_distillation_loss`: passed.
- `bash -n` for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/launch_tmux.sh`: passed.
- `bash -n` for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/wait_and_launch_a_4kd23_memory_gated_gpu_safe.sh`: passed.
- GPU3 memory-gate dry run: passed under `MAX_GPU_UTIL=100` and `MIN_FREE_MIB=12288`.
- Formal preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/logs/train_20260612_224738.log`, exit code `0`.
- Formal training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/logs/train_20260612_225300.log`, exit code `0`, completed step `300`.
- Best-val GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/metrics/test_eval_best_val_gpu.json`.
- Best-val GPU task eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/metrics/test_task_metrics_best_val_gpu.json`.
- Final GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/metrics/test_eval_final_gpu.json`.

## Result Gate

- Best validation: step `50`, `val_psnr=18.256583743058567`,
  `val_loss=0.12251933488994837`.
- Best-val fast-test: `psnr_mean=17.78064128762362`,
  `ssim_mean=0.6971213883161544`, `loss_mean=0.12059733912348747`,
  `total_loss_mean=0.12382887728512287`.
- Final fast-test: `psnr_mean=17.035365402315897`,
  `loss_mean=0.13115204714238643`,
  `total_loss_mean=0.13438382618129252`.
- Decision: gate failed for candidate promotion. `A_4kd23` validates the
  infrastructure but not the single-checkpoint conversion hypothesis.
