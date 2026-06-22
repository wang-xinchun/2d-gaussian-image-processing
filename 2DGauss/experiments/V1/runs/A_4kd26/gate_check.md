# Gate Check: V1/T01/A_4kd26

## Deployment Gate

- Created: `2026-06-13 CST`
- Target GPU policy: physical GPU0/GPU3 only.
- Default GPU: physical GPU3.
- Memory gate: `TWODGAUSS_MIN_FREE_MIB=12288`.
- Utilization gate: `TWODGAUSS_MAX_GPU_UTIL=100`.
- CUDA memory fraction: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- GPU1/GPU2 remain forbidden.

## Code Gate

- Dataset changed: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/data/paired_image_dataset.py`
- Training entrypoint changed: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- New default-off training keys:
  - `training.input_std_route_balanced_crop_sampling`
  - `training.input_std_route_balanced_crop_attempts`

## Planned Checks

- `python -m py_compile` for changed Python files.
- Crop-aware DataLoader smoke.
- TOML parse and checkpoint/dataset path checks.
- `bash -n` for A26 launchers.
- GPU memory-gate dry run on GPU3.
- GPU preflight before formal training.

## Completed Checks

- `python -m py_compile scripts/train_v1_t01.py src/uhd_gaussian/data/paired_image_dataset.py`: passed.
- Crop-aware DataLoader smoke: passed; route target tuple indices reached the
  dataset and produced mixed-route batches more often than file-level sampling.
- A26 TOML parse and launcher path checks: passed.
- `bash -n` for A26 launchers: passed.
- GPU3 memory-gate dry run: passed.
- GPU3 preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/logs/train_20260613_210619.log`, exit code `0`.
- Formal training started on GPU3 and was stopped early after step `150`
  validation remained far below A24.

## Result Gate

- Best early validation: step `100`, `val_psnr=18.236443073291603`,
  `val_loss=0.12283354680985212`.
- Step `150`: `val_psnr=18.2230662628357`,
  `val_loss=0.12306080736219883`.
- Decision: gate failed. Crop-aware 50/50 balancing technically worked but hurt
  the natural training distribution and underperformed A24.
