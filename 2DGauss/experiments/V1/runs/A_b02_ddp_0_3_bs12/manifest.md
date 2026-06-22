# A_b02_ddp_0_3_bs12 Manifest

- Full id: `V1/T01/A_b02_ddp_0_3_bs12`
- Purpose: two-process DDP training on physical GPU0 and GPU3 without exhausting other users' remaining VRAM.
- Current-task classification: `UHD_haze` resource-calibration package only. Per `DEC-000145/146`, this is not a valid ultra-high-definition video deshaking/stabilization training package.
- Physical GPUs: `0,3`
- Per-rank batch size: `12`
- Effective global batch size: `24`
- Config: `experiments/V1/runs/A_b02_ddp_0_3_bs12/configs/a_b02_ddp_0_3_bs12_uhdhaze_patch_cache.toml`
- Template config: `experiments/V1/templates/T01/configs/a_b02_ddp_0_3_bs12_uhdhaze_patch_cache.toml`
- Launcher: `experiments/V1/runs/A_b02_ddp_0_3_bs12/launch_ddp_tmux.sh`
- Python env: `/home/ldj/wxc/.venvs/2dgauss-py313`

## Safety Gate

- The launcher requires physical GPUs exactly `0,3`.
- It requires at least `30000MiB` free memory on each of GPU0 and GPU3 before launch.
- It allows high utilization because this package is intended to share under-filled GPUs, but it does not ignore free-memory safety.
- It exports `TWODGAUSS_ALLOW_MULTI_GPU_VISIBLE=1` intentionally for DDP.

## Batch Evidence

- GPU0 single-process `batch_size=24` smoke reserved about `37.619GB` on an empty GPU0.
- GPU3 single-process `batch_size=8` smoke reserved about `12.525GB`.
- This DDP package uses per-rank `batch_size=12`, expected to be materially below the single-GPU `bs24` VRAM pressure on each card while keeping effective global batch at `24`.
- Later GPU0-only calibration found `batch_size=24` too tight for shared use because it exited without `final.pt`, while `batch_size=22` completed and left about `6.9-8.2GiB` free. Keep this DDP package as a guarded calibration artifact unless a deshaking-specific dataset/config replaces the `UHD_haze` inputs.
