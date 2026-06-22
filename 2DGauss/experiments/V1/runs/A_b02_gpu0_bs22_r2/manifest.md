# A_b02_gpu0_bs22_r2 Manifest

- Full id: `V1/T01/A_b02_gpu0_bs22_r2`
- Purpose: GPU0 large-batch rerun after `A_b02_gpu0_bs24` reached step 860 without OOM or traceback but exited before final checkpoint.
- Config: `experiments/V1/runs/A_b02_gpu0_bs22_r2/configs/a_b02_gpu0_bs22_r2_uhdhaze_patch_cache.toml`
- Template config: `experiments/V1/templates/T01/configs/a_b02_gpu0_bs22_r2_uhdhaze_patch_cache.toml`
- Launcher: `experiments/V1/runs/A_b02_gpu0_bs22_r2/launch_tmux.sh`
- Default physical GPU: `0`
- Batch size: `22`
- Data cache: `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`

## Launch Guard

- Requires GPU0 free memory at least `42000MiB` by default.
- Records GPU0 utilization at launch, but does not block on utilization when enough memory is free.
- Uses project venv `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python`.
- Refuses non-GPU0 launch for this package.
- Runs Python unbuffered and writes `<log>.exit_code` after process exit.

## Calibration Evidence

- GPU0 `batch_size=24` smoke passed with peak reserved memory about `37.619GB`.
- Full `batch_size=24` attempt reached step 860 and completed step 300/600 validation/checkpoints, but left GPU0 with only about `3GB` free at one point and exited before `final.pt`.
- `batch_size=22` keeps the run large while giving more memory headroom for other GPU0 users.
