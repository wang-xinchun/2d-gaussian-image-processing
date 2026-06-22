# A_4kd32 Gate Check

Updated: 2026-06-14 19:29:38 CST

## Planned Checks

- `py_compile` for:
  - `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- TOML parse for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/configs/a_4kd32_fast_4kdehaze_high_a08_transplant_low_a24_locked.toml`.
- Checkpoint path checks:
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Launcher syntax:
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/launch_tmux.sh`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/wait_and_launch_a_4kd32_memory_gated_gpu_safe.sh`
- Transplant smoke:
  - high route head equals A08 `decoder.2`;
  - low route head equals A24 base `decoder.2`;
  - low route parameters are frozen;
  - trainable prefixes only include shared/high route modules.
- GPU3 preflight under GPU0/GPU3 memory gate.
- Formal training and four evals: standard fast-test, task metrics, A18 alignment, input_std route-regime diagnostics.

## Current Result

Completed. A32 is not promoted as a standalone checkpoint because best-val fast-test PSNR is `17.786780968730323`, slightly below A24 `17.790202882742463`. It is still a positive diagnostic: low-route gap recovered to `+0.05380978168030712 dB` and low selected delta stayed `0.0`, but high-route gap remained `-0.211220889864822 dB`.
