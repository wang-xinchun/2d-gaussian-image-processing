# A_4kd33 Gate Check

Updated: 2026-06-14 19:29:38 CST

## Planned Checks

- TOML parse for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/configs/a_4kd33_fast_4kdehaze_a31_high_a24_low_surgery.toml`.
- Checkpoint surgery smoke:
  - copied A31 shared/high_feature/high_head;
  - low_head equals A24 `decoder.2`;
  - low_feature final conv is zero;
  - forward aux keys exist;
  - low_delta_abs_mean is zero or near zero on random input.
- GPU3 eval:
  - standard fast-test;
  - task metrics;
  - A18 alignment;
  - input_std route-regime diagnostics.

## Current Result

Passed as the current best single-checkpoint fast-probe candidate. A33 standard test PSNR is `17.823062311698262`, which exceeds A24 `17.790202882742463` and clears the `17.82` target. It remains below A18 router `17.84693437641316`; high-route gap is `-0.12677254958688497 dB`, low-route gap is `+0.053754617905617835 dB`.
