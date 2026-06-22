# V1/T01/A_4kd29 Result Analysis

## Summary

`A_4kd29` tested hard high/low `input_std` route decoder heads after A28 showed that small route feature deltas were still too weak. The new module lets each route own the final RGB projection while copying the loaded A24 base decoder head at initialization and starting route feature deltas at zero.

The engineering path works, but the scientific result is negative. A29 does not beat A24 and does not repair the high-`input_std` / A08-tail gap. Its useful finding is more precise: route capacity can move, but unconstrained movement quickly hurts validation, while the best checkpoint retreats to near-zero route deltas.

## Implementation

- Code paths:
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
  - `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/configs/a_4kd29_fast_4kdehaze_a24_route_decoder_heads.toml`
- Main switch: `use_input_std_route_decoder_heads=true`
- Disabled old route mechanisms: residual, terminal heads, feature expert.
- Init: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`, non-strict.
- Teachers: A08 high route and A15 low route.
- Branch weights: high/A08 `0.18`, low/A15 `0.04`.
- Sampling: natural fast-probe sampling; no A25/A26 balanced sampler.

## Checks

- `py_compile` passed for changed Python files.
- Forward smoke passed: default configs keep route aux disabled; enabled decoder heads emit high/low restored outputs, gate, and deltas.
- Initialization safety smoke passed: initial high/low deltas are `0.0`, high/low restored outputs match, and restored output only differs by float noise.
- Branch-loss gradient smoke passed: high/A08 and low/A15 weighted losses are nonzero and gradients reach both route decoder heads.
- TOML parse, checkpoint path checks, and launcher `bash -n` passed.
- GPU3 dry-run initially failed due to free memory below gate; GPU0 dry-run passed and was used.

## Training

- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/logs/train_20260614_163111.log`
- Preflight exit code: `0`
- Preflight validation: `val_psnr=17.932861879591563`, `val_loss=0.12783942330628634`
- Formal train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/logs/train_20260614_163328.log`
- Formal status: stopped early after step `270` because validation degraded monotonically; no normal formal exit-code file was produced.
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/checkpoints/best_val.pt`
- Best-val step: `50`
- Best-val PSNR: `18.263113043745197`
- Best-val loss: `0.12214793119579553`

Validation curve:

- step50: `val_psnr=18.263113043745197`, high delta `0.0003283282634220086`, low delta `0.0002078584379341919`
- step100: `val_psnr=18.159902146770662`, high delta `0.0013044568512123077`, low delta `0.001041918820119463`
- step150: `val_psnr=18.071118839008665`, high delta `0.0025912851525936274`, low delta `0.002276030392386019`
- step200: `val_psnr=18.010054095891146`, high delta `0.004265504041686654`, low delta `0.004059408595785499`
- step250: `val_psnr=17.973219683890125`, high delta `0.005968559517059475`, low delta `0.0058696489268913864`

This is the central diagnostic: the route heads are trainable, but larger route deltas correlate with worse validation.

## Fast-Test Metrics

- Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_task_metrics_best_val_gpu.json`
- PSNR: `17.76127769536133`
- SSIM: `0.7003751814365387`
- Loss: `0.12043320007622242`
- Total loss: `0.12366428487002849`
- Edge magnitude L1: `0.15857212394475936`
- High-frequency L1: `0.02160167632624507`
- Local-contrast L1: `0.024722029119729996`
- Chroma L1: `0.030339077804237603`
- Dark-channel L1: `0.142566970475018`
- Saturation L1: `0.05568681165575981`

A29 fails the required A24 fast-test PSNR threshold `17.790202882742463`, fails the hoped-for `17.82`, and stays below the A18 router target `17.84693437641316`. It passes the SSIM floor, but PSNR and route-regime evidence are not enough for promotion.

## A18 Alignment

- Alignment file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/a18_record_alignment_best_val_gpu.json`
- Overall gap vs A18 router: `-0.08565668045309131 dB`
- Low-route gap vs A18 router: `+0.027826703499809195 dB`
- High-route gap vs A18 router: `-0.236088142902285 dB`
- A15-oracle group gap vs router: `+0.07751041681591958 dB`
- A08-oracle group gap vs router: `-0.27720066420366934 dB`

A29 improves over A28's overall gap but still does not beat A24. The high-route gap is slightly worse than the A24 target line (`-0.224697 dB`) and far from the required `>-0.10 dB`. Low-route gap also falls below the required `+0.03 dB` floor by a small margin.

## Route-Regime Diagnostics

- Route-regime file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/input_std_route_regime_best_val_gpu.json`
- Overall selected delta abs mean: `0.00025362832675455136`
- High selected delta abs mean: `0.0002827254742611373`
- Low selected delta abs mean: `0.0002316778470566006`
- High branch L1 vs A08 teacher: `0.262734996371491`
- Low branch L1 vs A15 teacher: `0.34177368774748684`

The best checkpoint is effectively conservative: selected deltas are only `2e-4` to `3e-4`. Later checkpoints move the heads more, but validation falls. That means A29 did not simply fail because the heads cannot move; it failed because the current branch supervision lets the heads move in a way that is not trusted by the main reconstruction objective.

## Decision

A29 is not promoted. A24 remains the best single-checkpoint fast-probe candidate, and A18 remains the two-checkpoint router target.

The next mechanism should keep the hard route-capacity idea but add stabilizers:

- constrain route deltas with a trust-region or delta regularization;
- ramp branch losses after the base reconstruction stays stable;
- lower high/A08 branch pressure from the A29 setting;
- optionally train route heads in a two-stage schedule: copy A24 base head, freeze shared trunk briefly, then unfreeze with a small route delta budget;
- evaluate the first useful checkpoint before route deltas exceed the validation-safe range.

Proposed next run: `A_4kd30`, conservative hard route decoder heads with route-delta trust region and delayed/asymmetric branch pressure.
