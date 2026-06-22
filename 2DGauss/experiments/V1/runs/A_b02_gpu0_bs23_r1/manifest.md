# A_b02_gpu0_bs23_r1 Manifest

- Full id: `V1/T01/A_b02_gpu0_bs23_r1`
- Purpose: GPU0-only large-batch resource calibration after `batch_size=22` completed and the older `batch_size=24` run exited before `final.pt`.
- Config: `experiments/V1/runs/A_b02_gpu0_bs23_r1/configs/a_b02_gpu0_bs23_r1_uhdhaze_patch_cache.toml`
- Template config: `experiments/V1/templates/T01/configs/a_b02_gpu0_bs23_r1_uhdhaze_patch_cache.toml`
- Launcher: `experiments/V1/runs/A_b02_gpu0_bs23_r1/launch_tmux.sh`
- Default physical GPU: `0`
- Batch size: `23`
- Data cache: `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`

## Launch Guard

- Requires GPU0 free memory at least `43000MiB` by default.
- Refuses non-GPU0 launch for this package.
- Records GPU0 free memory and utilization at launch.
- Uses project venv `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python`.
- Runs Python unbuffered and writes `<log>.exit_code` after process exit.

## Boundary

This is a GPU0 memory and process-stability calibration package on the existing `UHD_haze` patch cache. It is not a valid ultra-high-definition video deshaking training run.
