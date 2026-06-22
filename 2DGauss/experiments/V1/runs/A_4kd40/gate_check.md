# A_4kd40 Gate Check

时间: 2026-06-15 CST

## Planned Checks

- Static `py_compile` for training/evaluation scripts and model/config code.
- TOML parse and required path checks.
- Launcher `bash -n` and GPU dry-run.
- CPU smoke:
  - strict load A37 checkpoint with zero missing/unexpected keys;
  - native-config A08/A15 teachers load with zero missing/unexpected keys;
  - trainable prefixes exclude low route and `gaussian_control_canvas`;
  - high route, Gaussian guidance/refinement, and `decoder.1` receive gradients;
  - balanced sampler can form high-route batches.
- GPU3 preflight before formal training.
- Formal training only after preflight passes.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- PASS: Static `py_compile` for training/evaluation scripts, model/config code, and paired dataset code.
- PASS: TOML parse and required path checks.
- PASS: launcher `bash -n` and GPU3 dry-run with free memory above `8192MiB`.
- PASS: CPU smoke. Balanced sampler enabled with `batch_size=1`, high per batch `1`, low per batch `0` on the smoke subset. A37 init checkpoint loaded with missing/unexpected `0/0`; A08/A15 native teachers loaded with missing/unexpected `0/0`; trainable params `76643/168406`; `gaussian_control_canvas` trainable count `0`; low route trainable count `0`; high route/shared/guidance/refinement/decoder.1 received gradients; low route and control canvas gradients stayed zero.
- PASS: GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/logs/train_20260615_225246.log`, exit code `0`. It used physical GPU3, online A08/A15 teachers, and high-route balanced sampling. Preflight metrics: `val_psnr=18.562165492684464`, `val_route_high_delta_abs_mean=0.01713104228489101`, `val_route_low_delta_abs_mean=0.0`, `preflight_route_high_hard_mean=1.0`, `trainable_parameter_count=76643`.
- PASS: GPU3 formal training `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/logs/train_20260615_225430.log`, exit code `0`. Best/final step `120`, `val_psnr=18.619183648392536`, `val_route_high_delta_abs_mean=0.015635613431222738`, `val_route_low_delta_abs_mean=0.0`.
- PASS: Standard test eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_eval_best_val_gpu.json`, exit code `0`, `psnr_mean=17.900562951956445`, samples `100`.
- PASS: Task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_task_metrics_best_val_gpu.json`, exit code `0`, `ssim_mean=0.7017474815249443`.
- PASS: A18 record alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/a18_record_alignment_best_val_gpu.json`, exit code `0`: overall gap `+0.0536285761420247 dB`, high-route gap `-0.1180115677202543 dB`, low-route gap `+0.18311149168725271 dB`.
- PASS: Route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/input_std_route_regime_best_val_gpu.json`, exit code `0`: overall selected delta `0.0056926918867975474`, high selected delta `0.013238818341389645`, low selected delta `0.0`.
