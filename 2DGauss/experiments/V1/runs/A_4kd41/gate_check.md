# A_4kd41 Gate Check

时间: 2026-06-16 CST

## Planned Checks

- Static `py_compile` for training/evaluation scripts and model/config code.
- TOML parse and required path checks.
- Launcher `bash -n` and GPU dry-run.
- CPU smoke:
  - strict load A37 checkpoint with zero missing/unexpected keys;
  - native-config A08/A15 teachers load with zero missing/unexpected keys;
  - trainable prefixes exclude low route and `gaussian_control_canvas`;
  - high route, Gaussian guidance/refinement, and `decoder.1` receive gradients;
  - new oracle-weighted high branch loss is active only for A41 and logs finite weights;
  - low route and control canvas gradients stay zero.
- GPU3 preflight before formal training.
- Formal training only after preflight passes.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- PASS: Static `py_compile` for training/evaluation scripts, model/config code, and paired dataset code.
- PASS: TOML parse and required path checks. A41 config enables `input_std_route_high_branch_oracle_weighting=true` with weight range `0.75..2.25`.
- PASS: launcher `bash -n`.
- PASS: CPU smoke. Balanced sampler enabled with `batch_size=1`, high per batch `1`, low per batch `0` on the smoke subset. A37 init checkpoint loaded with missing/unexpected `0/0`; A08/A15 native teachers loaded with missing/unexpected `0/0`; trainable params `76643/168406`; `gaussian_control_canvas` trainable count `0`; low route trainable count `0`. Oracle weighting was active with finite weight mean `1.6626904010772705`; the smoke crop had high_hard `1.0`, input_std `0.07450006902217865`, A08-vs-A15 oracle advantage `0.002645067870616913`, and A08-better fraction `1.0`. High route, shared route, guidance/refinement, and `decoder.1` received gradients; low route and control canvas gradients stayed zero.
- PASS: GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/logs/train_20260615_233303.log`, exit code `0`. It used physical GPU3, online A08/A15 teachers, high-route balanced sampling, and oracle-weighted high branch loss. Preflight metrics: `val_psnr=18.565064143991755`, `val_route_high_delta_abs_mean=0.017135478127747773`, `val_route_low_delta_abs_mean=0.0`, `input_std_route_high_branch_oracle_weight_mean=1.6529526710510254`, `input_std_route_high_branch_oracle_a_better_fraction=1.0`, `trainable_parameter_count=76643`.
- PASS: GPU3 formal training `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/logs/train_20260615_233454.log`, exit code `0`. Best/final step `120`, `val_psnr=18.617112345875956`, `val_route_high_delta_abs_mean=0.015633893799968063`, `val_route_low_delta_abs_mean=0.0`.
- PASS: Standard test eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_eval_best_val_gpu.json`, exit code `0`, `psnr_mean=17.900201657598004`, samples `100`.
- PASS: Task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_task_metrics_best_val_gpu.json`, exit code `0`, `ssim_mean=0.7017551222443581`.
- PASS: A18 record alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/a18_record_alignment_best_val_gpu.json`, exit code `0`: overall gap `+0.05326728178358334 dB`, high-route gap `-0.11813195131188482 dB`, low-route gap `+0.182568457627533 dB`.
- PASS: Route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/input_std_route_regime_best_val_gpu.json`, exit code `0`: overall selected delta `0.005682019237428904`, high selected delta `0.013213998226578845`, low selected delta `0.0`, high branch L1 vs A08 teacher `0.01348984414754912`.
