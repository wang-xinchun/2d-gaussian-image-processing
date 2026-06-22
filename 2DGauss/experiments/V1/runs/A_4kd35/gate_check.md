# A_4kd35 Gate Check

时间: 2026-06-14 22:01:30 CST

## Static

- PASS: `py_compile /home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- PASS: `py_compile /home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- PASS: `py_compile /home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- PASS: TOML parse and checkpoint path check for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/configs/a_4kd35_fast_4kdehaze_a33_post_high_adapter.toml`
- PASS: `bash -n /home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/launch_tmux.sh`

## Smoke

- PASS: old A34 config keeps `use_input_std_route_post_adapter=false`
- PASS: A35 aux keys exist and have expected shapes
- PASS: zero-init post adapter does not alter A33 output, `post_delta_max=0.0`
- PASS: high branch backward reaches `input_std_route_post_adapter.high_out`, grad abs sum `1.7656699419021606`
- PASS: loading A33 surgery checkpoint with `strict=false` leaves restored output identical, `max_restored_diff=0.0`
- PASS: only post adapter is trainable, `12995/181401` trainable parameters

## GPU

- PASS: GPU0 preflight log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/logs/train_20260614_214051.log`, exit code `0`
- PASS: GPU0 formal log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/logs/train_20260614_214218.log`, exit code `0`

## Evaluation

- PASS: standard fast-test `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_eval_best_val_gpu.json`
- PASS: task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_task_metrics_best_val_gpu.json`
- PASS: A18 alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/a18_record_alignment_best_val_gpu.json`
- PASS: route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/input_std_route_regime_best_val_gpu.json`

## Promotion Gate

- Aggregate PSNR: PASS vs A24, `17.823132133493647 > 17.790202882742463`
- Hopeful target: PASS numerically, `17.823132133493647 >= 17.82`
- A18 replacement: FAIL, gap `-0.02380224232077156 dB`
- High-route target: FAIL, high-route gap `-0.12662823419234123 dB`, still below `>-0.10 dB`
- Scientific promotion: FAIL. Improvement over A33 is only `+0.00006982179538539413 dB`; A35 is not a meaningful bottleneck breakthrough.
