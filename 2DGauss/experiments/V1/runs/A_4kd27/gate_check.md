# Gate Check: V1/T01/A_4kd27

## Pre-Launch Checks

- [x] `py_compile` for changed Python files.
- [x] Terminal-head forward smoke.
- [x] TOML parse and checkpoint path checks.
- [x] Launcher `bash -n`.
- [x] GPU0/GPU3 dry-run gate.
- [x] GPU preflight.

Notes:

- `py_compile` passed for `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_a18_record_alignment_v1_t01.py`, and `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`.
- First GPU3 preflight with memory fraction `0.30` failed due to process memory cap, then GPU3 preflight with `0.50` passed.

## Training

- [x] Formal GPU training completed.
- [x] Best-val checkpoint selected by `val_psnr`.
- [x] Route terminal-head diagnostics logged.

Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/logs/train_20260614_131135.log`; exit code `0`; best step `400`; `val_psnr=18.41533773497595`, `val_loss=0.11968508075922728`.

## Evaluation

- [x] Standard fast-test eval.
- [x] Task-specific fast-test eval.
- [x] A18-record alignment eval.
- [x] Input-std route-regime eval.

Evaluation outputs:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/test_task_metrics_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/a18_record_alignment_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/input_std_route_regime_best_val_gpu.json`

## Promotion Gate

- Required: beat A24 single-checkpoint PSNR `17.790202882742463`.
- Strong target: PSNR >= `17.82`.
- Milestone target: approach or beat A18 router `17.84693437641316`.
- High-route target: improve A24 high-route gap from `-0.224697 dB` to `>-0.10 dB`.
- Low-route target: keep low-route gap >= `+0.03 dB`.

Result:

- Fast-test PSNR `17.7513865188403`: fail vs A24, fail vs `17.82`, fail vs A18.
- High-route gap vs A18 router `-0.3054244593410088 dB`: fail and worse than A24.
- Low-route gap vs A18 router `+0.06278010621493545 dB`: pass.
- SSIM `0.700952832698822`: pass.
- Loss `0.1202681017667055`: pass.
- Conclusion: A27 is completed as an engineering/mechanism diagnostic, but it is not promoted.
