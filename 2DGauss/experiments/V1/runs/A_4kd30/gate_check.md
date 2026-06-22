# Gate Check: V1/T01/A_4kd30

## Pre-Launch Checks

- [x] `py_compile` for changed Python files.
- [x] Route trust-region loss smoke.
- [x] TOML parse and checkpoint path checks.
- [x] Launcher `bash -n`.
- [x] GPU0/GPU3 dry-run gate.
- [x] GPU preflight.

Notes:

- A30 reuses A29 hard decoder-head model structure.
- New training-loss controls are default-off and only enabled by the A30 config.
- A30 must use only physical GPU0/GPU3 under the memory-first gate.

## Training

- [x] Formal GPU training completed.
- [x] Best-val checkpoint selected by `val_psnr`.
- [x] Route trust-region diagnostics logged.

Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/logs/train_20260614_173319.log`, exit code `0`.

Best-val step: `50`; `val_psnr=18.231969745567717`, `val_loss=0.122691053673625`.

Validation curve:

- step50: `18.231969745567717`
- step100: `18.153006463219054`
- step150: `18.164903592335143`
- step200: `18.17848947550911`
- step250: `18.160505904371178`

## Evaluation

- [x] Standard fast-test eval.
- [x] Task-specific fast-test eval.
- [x] A18-record alignment eval.
- [x] Input-std route-regime eval.

Evaluation files:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/metrics/test_task_metrics_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/metrics/a18_record_alignment_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/metrics/input_std_route_regime_best_val_gpu.json`

## Promotion Gate

- Required: beat A24 single-checkpoint PSNR `17.790202882742463`.
- Strong target: PSNR >= `17.82`.
- Milestone target: approach or beat A18 router `17.84693437641316`.
- High-route target: improve A29 high-route gap `-0.236088142902285 dB` and A24 high-route gap `-0.224697 dB` to at least `>-0.10 dB`.
- Low-route target: keep low-route gap >= `+0.03 dB`.
- Proxy targets: SSIM >= `0.699`, loss <= `0.1205`, dark/edge/local/chroma not obviously worse than A18 router.

Result:

- PSNR `17.75180602939797`; failed A24/17.82/A18 targets.
- SSIM `0.7003354308009148`; passed SSIM target.
- Loss `0.12060016021132469`; failed loss target.
- Low-route gap `+0.012763517901581372 dB`; failed low-route floor.
- High-route gap `-0.23814779446593295 dB`; failed high-route target.

Promotion: no.
