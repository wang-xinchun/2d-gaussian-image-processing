# Gate Check: V1/T01/A_4kd25

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
- New default-off training keys:
  - `training.input_std_route_balanced_sampling`
  - `training.input_std_route_balanced_threshold`
  - `training.input_std_route_balanced_high_fraction`

## Planned Checks

- `python -m py_compile` for changed Python files.
- TOML parse and checkpoint/dataset path existence checks.
- Sampler smoke: confirm high/low sample pools and 1/1 batch composition.
- `bash -n` for A25 launchers.
- GPU memory-gate dry run on GPU3.
- GPU preflight with `TWODGAUSS_PREFLIGHT_ONLY=1`.

## Completed Checks

- `python -m py_compile scripts/train_v1_t01.py`: passed.
- A25 TOML parse and checkpoint/dataset path checks: passed.
- Sampler smoke: passed; file-level pools were high=`429`, low=`83`, first
  sampled batch had one high-file and one low-file item.
- `bash -n` for A25 launchers: passed.
- GPU3 memory-gate dry run: passed.
- GPU3 preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd25/logs/train_20260613_204956.log`, exit code `0`.
- Formal training started on GPU3 and was stopped early after step `120+`
  diagnostics showed poor validation trajectory.

## Result Gate

- Best early validation: step `50`, `val_psnr=18.231016319544015`,
  `val_loss=0.12287615351378918`.
- Step `100`: `val_psnr=18.198464752448107`,
  `val_loss=0.12331412188708782`.
- Decision: gate failed. File-level balanced sampling did not reliably balance
  crop-level route labels and underperformed A24.
