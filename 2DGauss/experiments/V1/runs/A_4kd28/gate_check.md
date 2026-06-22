# Gate Check: V1/T01/A_4kd28

## Pre-Launch Checks

- [x] `py_compile` for changed Python files.
- [x] Feature-expert forward smoke.
- [x] TOML parse and checkpoint path checks.
- [x] Launcher `bash -n`.
- [x] GPU0/GPU3 dry-run gate.
- [x] GPU preflight.

Notes:

- `py_compile` passed for `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, and `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`.
- TOML parse passed; init checkpoint, teacher checkpoints, and dataset paths exist.
- Feature-expert forward smoke passed: default configs keep route aux disabled; enabled route feature expert emits high/low restored outputs, gate, and route deltas.
- Branch-loss smoke passed with synthetic low/high `input_std` batch: high/A08 and low/A15 branch losses are nonzero and gradients reach both feature experts.
- GPU3 dry-run passed with free memory `24114MiB`, util `100%`, memory fraction `0.50`.
- GPU3 preflight log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/logs/train_20260614_154419.log` completed with exit code `0`; preflight validation `val_psnr=18.1672422635604`, `val_loss=0.12408415004611015`, route aux available.

## Training

- [x] Formal GPU training completed.
- [x] Best-val checkpoint selected by `val_psnr`.
- [x] Route feature-expert diagnostics logged.

Notes:

- Formal training log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/logs/train_20260614_154636.log` completed with exit code `0`.
- Best-val checkpoint `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/checkpoints/best_val.pt` selected at step `400` with `val_psnr=18.44844816281949` and `val_loss=0.11907587297260762`.
- Route diagnostics remained small at step400: train high/low route delta means stayed near `1e-3`.

## Evaluation

- [x] Standard fast-test eval.
- [x] Task-specific fast-test eval.
- [x] A18-record alignment eval.
- [x] Input-std route-regime eval.

Notes:

- Standard eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_eval_best_val_gpu.json`: `psnr_mean=17.73849582669033`, `loss_mean=0.12039212018251419`, `total_loss_mean=0.12362300843000412`.
- Task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_task_metrics_best_val_gpu.json`: `ssim_mean=0.6989976793527604`.
- A18 alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/a18_record_alignment_best_val_gpu.json`: low-route gap `+0.06555415143863476 dB`, high-route gap `-0.3390800359165398 dB`.
- Route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/input_std_route_regime_best_val_gpu.json`: high selected delta abs mean `0.0008386419774682889`, low selected delta abs mean `0.0013225519980647061`.

## Promotion Gate

- Required: beat A24 single-checkpoint PSNR `17.790202882742463`.
- Strong target: PSNR >= `17.82`.
- Milestone target: approach or beat A18 router `17.84693437641316`.
- High-route target: improve A27 high-route gap `-0.3054244593410088 dB` and A24 high-route gap `-0.224697 dB` to at least `>-0.10 dB`.
- Low-route target: keep low-route gap >= `+0.03 dB`.

Result:

- [x] Low route preserved: `+0.06555415143863476 dB`.
- [ ] Required PSNR failed: `17.73849582669033 < 17.790202882742463`.
- [ ] Strong PSNR failed: `17.73849582669033 < 17.82`.
- [ ] A18 router target failed: `17.73849582669033 < 17.84693437641316`.
- [ ] High route failed: `-0.3390800359165398 dB`, worse than A27 and A24.
- [ ] SSIM proxy narrowly failed: `0.6989976793527604 < 0.699`.

Conclusion: A28 is engineering-complete but not promoted. Move to A29 hard route decoder heads instead of sweeping A28 strength/weights.
