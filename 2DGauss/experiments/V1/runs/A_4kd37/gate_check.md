# A_4kd37 Gate Check

时间: 2026-06-14 23:23:35 CST

## Planned Checks

- Static: `py_compile` for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- TOML parse and path check for A37 config.
- Launcher `bash -n`.
- Teacher override smoke: `router_distillation_config_a/b` must load native A08/A15 configs with zero missing/unexpected keys.
- GPU3 preflight before formal training.
- Formal training only after preflight passes.

## Completed Checks

- PASS: `py_compile` for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`
- PASS: A37 TOML parse and path check.
- PASS: launcher `bash -n`.
- PASS: CPU teacher override smoke. A33 init missing/unexpected `0/0`; A08 teacher native config missing/unexpected `0/0`; A15 teacher native config missing/unexpected `0/0`; low route trainable list empty.
- FAIL: first GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_233259.log` exit code `1`, CUDA OOM during student forward with A08/A15 teachers resident. Deployment adjustment: changed `batch_size=2, grad_accum=4` to `batch_size=1, grad_accum=8` to keep effective batch while reducing activation memory.
- PASS/CONCERN: second GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_233515.log` exit code `0`, but `val_psnr=18.277983364055927`, materially below the A33/A35 initial regime. Diagnosis: ordinary GT/proxy losses update shared late modules on low-route samples and risk damaging the protected low route before any high-route improvement is visible.
- ADJUSTED: disabled ordinary reconstruction/proxy/control losses and kept only high/A08 route branch distillation plus delta trust. This turns A37 into high-route-only distillation, so low samples do not drive shared late-module updates.
- FAIL: third GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_234026.log` exit code `1` because `charbonnier_weight=0.0` requires explicit `allow_proxy_only_loss=true`.
- ADJUSTED: set `loss.allow_proxy_only_loss=true` to mark A37 as a diagnostic high-route distillation run.
- PASS: fourth GPU3 preflight `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_234303.log` exit code `0`, `val_psnr=18.347379254981835`, `val_route_high_delta_abs_mean=0.02297587492503226`, `val_route_low_delta_abs_mean=0.0`.
- PASS: GPU3 formal training `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_234511.log` exit code `0`. Best validation checkpoint is step `120` with `val_psnr=18.56340678071067`, `val_route_high_delta_abs_mean=0.017279619527980686`, `val_route_low_delta_abs_mean=0.0`.
- PASS: standard test eval with training config `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_gpu.json`, exit code `0`, `psnr_mean=17.889658453438816`.
- PASS: task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_task_metrics_best_val_gpu.json`, exit code `0`, `ssim_mean=0.7026848953962326`.
- PASS: A18-record alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/a18_record_alignment_best_val_gpu.json`, exit code `0`, overall gap `+0.042724077624394126 dB`, low-route gap `+0.1591364853806252 dB`, high-route gap `-0.11159004428502844 dB`.
- PASS: route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/input_std_route_regime_best_val_gpu.json`, exit code `0`, high selected delta `0.014578644511138284`, low selected delta `0.0`.
- PASS: comparable standard-loss eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_standard_loss_gpu.json`, exit code `0`, `loss_mean=0.11911950781941413`, `total_loss_mean=0.12235063299536705`.

## Final Gate

- PROMOTED: A37 is the new best single-checkpoint fast-probe candidate on aggregate PSNR and standard-loss re-eval.
- Caveat: high-route gap improves materially but remains `-0.11159004428502844 dB`, narrowly short of the planned `>-0.10 dB` high-route target.
