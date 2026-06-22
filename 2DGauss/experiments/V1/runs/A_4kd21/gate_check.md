# A_4kd21 Gate Check

## Status

Prepared and queued for GPU3-accelerated A08/A15 router evaluation. Immediate
dry-run was refused because GPU3 became externally busy (`util > 20%`).
The user subsequently relaxed the GPU policy: GPU0/GPU3 may be used when
memory is available, so high utilization alone no longer blocks A21 launch.
After that change, the first smoke exposed a CUDA memory-fraction device-index
bug in the evaluator. The evaluator was patched and relaunched successfully:
smoke completed, and full `200/200` GPU eval completed on GPU3.

## Required Checks

- TOML parse.
- Launcher `bash -n`.
- `scripts/evaluate_checkpoint_router_v1_t01.py` `py_compile`.
- GPU dry-run gate.
- Small GPU smoke with `val_limit=2`, `test_limit=2`.
- Full GPU run with offset `200`, limit `200` if smoke passes.
- Wait launcher:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/wait_and_launch_gpu_router_safe.sh`
- Active tmux wait session: `2dgauss_wait_a_4kd21_gpu_router_safe`.
- Active wait log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/logs/wait_gpu_router_20260612_203950.log`.
- Initial queue state: GPU3 gate refused with `free_mib=25820`, `util=97`;
  queue sleeps and rechecks every `120s`.
- Updated memory-gated launch policy: `max_gpu_util=100`, `min_free_mib=12288`.
- First memory-gated smoke attempt failed before evaluation due to
  `torch.cuda.set_per_process_memory_fraction(..., device=cuda)`.
- Fixed in:
  `/home/ldj/wxc/2DGauss/scripts/evaluate_checkpoint_router_v1_t01.py`
  and `/home/ldj/wxc/2DGauss/scripts/evaluate_checkpoint_blend_v1_t01.py`.
- Relaunch wait log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/logs/wait_gpu_router_20260612_211825.log`.
- Smoke passed and wrote:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/metrics/a08_a15_gpu_router_offset200_smoke_val2_test2.json`.
- Full eval completed and wrote:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/metrics/a08_a15_gpu_router_offset200_val200_test200.json`.
- Result analysis:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/result_analysis.md`.

## Key Result

- A08 test: `PSNR=14.61942131606054`, `SSIM=0.6726915469765663`.
- A15 test: `PSNR=14.570231632564433`, `SSIM=0.6721390201151372`.
- Fixed A18 input-std rule test: `PSNR=14.64236451612018`,
  `SSIM=0.6756442764401436`, choose A15 `107/200`.
- Val-selected `pred_a_std` rule test: `PSNR=14.634471636481214`,
  `SSIM=0.6754481521248817`, choose A15 `117/200`.
- Binary oracle test: `PSNR=14.662330739170043`, choose A15 `73/200`.
- Conclusion: fixed input-std router remains the stronger held-out rule;
  one-block richer-feature threshold selection is not stable enough.

## GPU Boundary

- Approved physical GPUs: `0` and `3`.
- Default physical GPU: `3`.
- GPU1/GPU2 forbidden even if idle.
- Do not kill external processes; GPU0/GPU3 may be shared if free memory is sufficient.
