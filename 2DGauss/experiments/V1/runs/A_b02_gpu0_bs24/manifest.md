# A_b02_gpu0_bs24 Manifest

- Full id: `V1/T01/A_b02_gpu0_bs24`
- Purpose: GPU0 large-batch local UHD_haze patch-cache run after GPU0 became available.
- Config: `experiments/V1/runs/A_b02_gpu0_bs24/configs/a_b02_gpu0_bs24_uhdhaze_patch_cache.toml`
- Template config: `experiments/V1/templates/T01/configs/a_b02_gpu0_bs24_uhdhaze_patch_cache.toml`
- Launcher: `experiments/V1/runs/A_b02_gpu0_bs24/launch_tmux.sh`
- Default physical GPU: `0`
- Batch size: `24`
- Data cache: `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`

## Launch Guard

- Requires GPU0 free memory at least `42000MiB` by default.
- Records GPU0 utilization at launch, but does not block on utilization when enough memory is free.
- Uses project venv `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python`.
- Refuses non-GPU0 launch for this package.

## Calibration Evidence

- GPU0 `batch_size=24` 3-step smoke passed before launch-package creation.
- Smoke peak reserved memory: about `37.619GB`.
- Smoke average step time: about `1.276s`.
- GPU0 became occupied by another user's `python -u trainval.py` jobs immediately afterward, so the package must pass a fresh memory gate before actual launch.
