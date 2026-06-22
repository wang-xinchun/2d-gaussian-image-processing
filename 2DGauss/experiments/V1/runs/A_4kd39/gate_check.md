# A_4kd39 Gate Check

时间: 2026-06-15 CST

## Planned Checks

- Static: `py_compile` for training/evaluation scripts and model/config code.
- TOML parse and path checks for A39 config.
- Launcher `bash -n`.
- CPU smoke:
  - strict load A37 checkpoint with zero missing/unexpected keys;
  - native-config A08/A15 teachers load with zero missing/unexpected keys;
  - trainable parameter prefixes match intended modules;
  - low route remains non-trainable;
  - high route and control/guidance/refinement paths receive gradient.
- GPU preflight on physical GPU0 or GPU3 only.
- Formal training only after preflight passes.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- PASS: static `py_compile` for training/evaluation scripts and model/config code.
- PASS: A39 TOML parse and path checks.
- PASS: launcher `bash -n`.
- PASS: CPU smoke. A37 init checkpoint loaded with missing/unexpected `0/0`; A08/A15 native teachers loaded with missing/unexpected `0/0`; trainable params `101156/168406`; low route was non-trainable and had zero gradient; high route, control canvas, Gaussian guidance/refinement, and `decoder.1` all received teacher-loss gradients.
- PASS: GPU3 dry-run gate with free memory above `8192MiB`.
- PASS: GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/logs/train_20260615_222110.log`, exit code `0`. It used physical GPU3, `batch_size=1`, online A08/A15 teachers, `cuda_memory_fraction=0.35`, and `preflight_train_samples=2`.
- Preflight metrics: `val_psnr=18.569757065698496`, `val_route_high_delta_abs_mean=0.017143623558804392`, `val_route_low_delta_abs_mean=0.0`, `val_route_high_hard_mean=0.37`, `trainable_parameter_count=101156`.
- PASS: Formal GPU3 training `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/logs/train_20260615_222621.log`, exit code `0`. Best/final step `120`, `val_psnr=18.62166244241818`, `val_route_high_delta_abs_mean=0.015583764696493745`, `val_route_low_delta_abs_mean=0.0`.
- PASS: Standard GPU3 test eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/test_eval_best_val_gpu.json`, `psnr_mean=17.900369585886775`, samples `100`.
- PASS: Task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/test_task_metrics_best_val_gpu.json`, `ssim_mean=0.7017246654629707`, `dark_channel_l1_mean=0.13971996370702983`.
- PASS: A18 alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/a18_record_alignment_best_val_gpu.json`, overall gap `+0.053435210072354966`, low-route gap `+0.18320352678672946`, high-route gap `-0.11858325626995542`.
- PASS: Route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/input_std_route_regime_best_val_gpu.json`, high selected delta `0.01329365114919668`, low selected delta `0.0`.

## Gate Verdict

A39 clears deployment and aggregate PSNR gates, but fails the intended high-route gate. The high-route A18 gap worsened relative to A37 (`-0.11159004428502844 -> -0.11858325626995542`), so A39 is a useful diagnostic rather than a high-route promotion.
