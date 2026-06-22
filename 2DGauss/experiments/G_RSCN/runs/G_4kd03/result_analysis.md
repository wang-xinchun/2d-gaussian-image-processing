# G_4kd03 Result Analysis

- Created: `2026-06-21 19:45:00 CST`
- Status: `smoke01 deployed/evaluated; not promoted`

## Smoke01 Plan

`G_4kd03_smoke01` tests whether a protected validation selector can stop the failure mode seen in `G_4kd02_smoke01`: high/A08 improves, but low/A15 silently regresses and still becomes the saved best checkpoint.

Expected signal:

- validation baseline metrics should be recorded before training;
- candidate best checkpoints must pass low-route, A15-oracle, and low-A15-oracle no-regression gates versus the initialized baseline;
- high/A08 branch signal should remain active, but A15/low protection must prevent selected best from spending the low side;
- final A37-aligned evaluation must improve high/A08 without losing low/A15.

Blocking rule:

- Do not run formal long training until best/final A37-aligned GPU evaluation is recorded here.

## Failed Launch 01

- Log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/logs/smoke01_20260621_200430.log`
- Exit code: `1`
- Failure point: validation at step `10`
- Cause: `best_selection_gate_status()` returns `(passed, info)`, but the first integration treated the tuple as the info dict when writing `best_selection_gate_enabled`.
- Action: fixed tuple unpacking in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`; rerun required.

## Successful Launch 02

- Log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/logs/smoke01_20260621_200941.log`
- Exit code: `0`
- Physical GPU: `3`
- Startup free memory: `46580MiB`
- Startup util: `17%`
- Util gate override: `TWODGAUSS_MAX_GPU_UTIL=100`, per user instruction not to idle when memory is safe.
- CUDA memory fraction: `0.35`
- CPU fallback: no
- Final step: `40`
- Peak CUDA reserved: `3.130859375GiB`
- Trainable params: `37698`
- Trainable prefix: `gaussian_route_control`

The run initialized from `G_4kd02_smoke01/best_val.pt` at parent step `10`. The protected selector wrote `best_val.pt` at step `0` before training, then refused to overwrite it because every later candidate failed the low/A15 validation gates.

## Validation Selector

Baseline at initialized step `0`:

| metric | value |
|---|---:|
| val_psnr | 16.84168937420473 |
| val_router_a08_oracle_gap_mean | -0.10329246520996094 |
| val_router_low_route_gap_mean | 0.4279048283894857 |
| val_router_a15_oracle_gap_mean | 0.25194562565196643 |
| val_router_low_a15_oracle_gap_mean | 0.4279048283894857 |

Candidate history:

| step | val_psnr | selection | candidate better | gate passed | low route | A15 oracle | low A15 |
|---:|---:|---:|---|---|---:|---:|---:|
| 10 | 16.83863949294249 | -0.10380935668945312 | false | false | 0.4239347457885742 | 0.2486658529801802 | 0.4239347457885742 |
| 20 | 16.836555677122085 | -0.1038351058959961 | false | false | 0.4207997639973958 | 0.24639459089799362 | 0.4207997639973958 |
| 30 | 16.835379212401637 | -0.10393047332763672 | false | false | 0.41911360422770183 | 0.24512026526711203 | 0.41911360422770183 |
| 40 | 16.834650534225442 | -0.1040201187133789 | false | false | 0.41810264587402346 | 0.24433318051424893 | 0.41810264587402346 |

Conclusion: the selector worked. It prevented the training trajectory from replacing the protected baseline with a checkpoint that was already losing low/A15 validation surplus.

## A37-Aligned Evaluation

Best-val checkpoint is step `0`, the protected initialized baseline:

| metric | G_4kd03 best | A37 | delta |
|---|---:|---:|---:|
| test PSNR | 17.888740401218694 | 17.889658453438816 | -0.0009180522201219787 |
| test SSIM | 0.7027190274000168 | 0.7026848953962326 | +0.000034132003784215215 |
| overall gap | +0.04180602540427511 | +0.042724077624394126 | -0.0009180522201190158 |
| high-route gap | -0.111354498048944 | -0.11159004428502844 | +0.0002355462360844468 |
| low-route gap | +0.15734817467600182 | +0.1591364853806252 | -0.0017883107046233748 |
| A08-oracle gap | -0.19803029212725382 | -0.19984098726138502 | +0.0018106951341312072 |
| A15-oracle gap | +0.2461110366348368 | +0.2493535773419097 | -0.0032425407070728884 |

Final checkpoint is step `40`:

| metric | G_4kd03 final | A37 | delta |
|---|---:|---:|---:|
| test PSNR | 17.886903500846003 | 17.889658453438816 | -0.002754952592812998 |
| test SSIM | 0.7027181923389435 | 0.7026848953962326 | +0.000033296942710925315 |
| overall gap | +0.03996912503158077 | +0.042724077624394126 | -0.002754952592813359 |
| high-route gap | -0.11112928873792333 | -0.11159004428502844 | +0.00046075554710511835 |
| low-route gap | +0.15395564769980316 | +0.1591364853806252 | -0.005180837680822031 |
| A08-oracle gap | -0.1955453928141209 | -0.19984098726138502 | +0.004295594447264128 |
| A15-oracle gap | +0.24059260319643774 | +0.2493535773419097 | -0.008760974145471945 |

Worst-gap first record remains `94_9400006.jpg`:

| checkpoint | PSNR | router gap | oracle gap |
|---|---:|---:|---:|
| best-val step 0 | 18.152418478105123 | -0.5271850418562707 | -0.8299010832985623 |
| final step 40 | 18.162709183919397 | -0.5168943360419966 | -0.8196103774842882 |

## Interpretation

`G_4kd03` separates two facts that were previously entangled:

- The validation gate is useful. It turns low/A15 preservation into an enforceable checkpoint-selection rule and correctly refuses all trained candidates.
- The training recipe is still not the missing mechanism. It continues to move in the familiar high/A08-positive direction while spending low/A15 surplus.

Therefore `G_4kd03_smoke01` is deployment-complete, scientifically useful, and not promotable. Formal long training remains blocked. The next version should not merely increase low/A15 scalar weights or make the selector stricter; it should change how updates are applied, for example by adding route-conditioned gradient isolation, per-route trust/projection, or a mechanism that freezes/rolls back low/A15-harmful route-control deltas while allowing high/A08-positive movement.
