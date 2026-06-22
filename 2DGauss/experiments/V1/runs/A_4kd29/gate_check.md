# Gate Check: V1/T01/A_4kd29

## Pre-Launch Checks

- [x] `py_compile` for changed Python files.
- [x] Decoder-head forward smoke.
- [x] Teacher branch-loss gradient smoke.
- [x] TOML parse and checkpoint path checks.
- [x] Launcher `bash -n`.
- [x] GPU0/GPU3 dry-run gate.
- [x] GPU preflight.

Notes:

- `py_compile` passed for `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`, and `/home/ldj/wxc/2DGauss/scripts/evaluate_a18_record_alignment_v1_t01.py`.
- Decoder-head forward smoke passed: default configs keep route aux disabled; enabled decoder heads emit high/low restored outputs, gate, and route deltas.
- Baseline safety smoke passed: initial high/low delta abs means are `0.0`, high/low restored outputs match, and final restored differs from high restored only by float noise.
- Branch-loss smoke passed with synthetic low/high `input_std` batch: high/A08 and low/A15 weighted losses are nonzero, and gradients reach both high and low decoder heads.
- TOML parse passed; init checkpoint, teacher checkpoints, and dataset paths exist.
- Launcher `bash -n` passed for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/launch_tmux.sh` and `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/wait_and_launch_a_4kd29_memory_gated_gpu_safe.sh`.
- GPU3 dry-run failed because free memory was only `7548MiB`, below the `12288MiB` gate. GPU0 dry-run passed with free memory `14852MiB`, util `99%`, memory fraction `0.50`.
- GPU0 preflight log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/logs/train_20260614_163111.log` completed with exit code `0`; preflight validation `val_psnr=17.932861879591563`, `val_loss=0.12783942330628634`, route aux available.
- Preflight after one update showed route deltas are trainable: `val_route_high_delta_abs_mean=0.0015141946246149019`, `val_route_low_delta_abs_mean=0.0015184472425607963`.

## Training

- [x] Formal GPU training launched.
- [x] Best-val checkpoint selected by `val_psnr`.
- [x] Route decoder-head diagnostics logged.

Notes:

- Formal training used physical GPU0 because GPU3 was below the free-memory gate at launch time.
- Formal log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/logs/train_20260614_163328.log` was stopped early after step `270` because validation degraded monotonically after step50; no normal formal exit-code file was produced.
- Best-val checkpoint `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/checkpoints/best_val.pt` selected step `50` with `val_psnr=18.263113043745197` and `val_loss=0.12214793119579553`.
- Validation degraded while route deltas grew: step50 high/low delta `0.0003283282634220086`/`0.0002078584379341919`; step250 high/low delta `0.005968559517059475`/`0.0058696489268913864`.

## Evaluation

- [x] Standard fast-test eval.
- [x] Task-specific fast-test eval.
- [x] A18-record alignment eval.
- [x] Input-std route-regime eval.

Notes:

- Standard eval `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_eval_best_val_gpu.json`: `psnr_mean=17.76127769536133`, `loss_mean=0.12043320007622242`, `total_loss_mean=0.12366428487002849`.
- Task metrics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_task_metrics_best_val_gpu.json`: `ssim_mean=0.7003751814365387`, dark-channel `0.142566970475018`, edge `0.15857212394475936`, local contrast `0.024722029119729996`, chroma `0.030339077804237603`.
- A18 alignment `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/a18_record_alignment_best_val_gpu.json`: overall gap `-0.08565668045309131 dB`, low-route gap `+0.027826703499809195 dB`, high-route gap `-0.236088142902285 dB`.
- Route-regime diagnostics `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/input_std_route_regime_best_val_gpu.json`: selected delta abs mean `0.00025362832675455136`, high selected delta `0.0002827254742611373`, low selected delta `0.0002316778470566006`.

## Promotion Gate

- Required: beat A24 single-checkpoint PSNR `17.790202882742463`.
- Strong target: PSNR >= `17.82`.
- Milestone target: approach or beat A18 router `17.84693437641316`.
- High-route target: improve A28 high-route gap `-0.3390800359165398 dB` and A24 high-route gap `-0.224697 dB` to at least `>-0.10 dB`.
- Low-route target: keep low-route gap >= `+0.03 dB`.
- Proxy targets: SSIM >= `0.699`, loss <= `0.1205`, dark/edge/local/chroma not obviously worse than A18 router.

Result:

- [ ] Required PSNR failed: `17.76127769536133 < 17.790202882742463`.
- [ ] Strong PSNR failed: `17.76127769536133 < 17.82`.
- [ ] A18 router target failed: `17.76127769536133 < 17.84693437641316`.
- [ ] High route failed: `-0.236088142902285 dB`, still below A24's `-0.224697 dB` reference and far from `>-0.10 dB`.
- [ ] Low route narrowly failed: `+0.027826703499809195 dB < +0.03 dB`.
- [x] SSIM proxy passed: `0.7003751814365387 >= 0.699`.

Conclusion: A29 is engineering-complete but not promoted. The next run should keep route decoder-head capacity but add route-delta trust-region/staged training constraints.
