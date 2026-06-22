# G_4kd14 Result Analysis

- Created: `2026-06-22 21:45:00 CST`
- Status: `smoke01 complete; not promoted`

## Smoke01 Question

`G_4kd14_smoke01` tests whether G13 exposed a second coupling bottleneck:

```text
zero-init dual adapter + every-2-step rollback
=> no accumulated GRC delta
```

G14 keeps the dual high/low route-control structure but defers the transfer-aware gate:

```text
train 20 short steps without step rollback
=> validate/save only if transfer full gate passes
=> evaluate final even if best_val remains A37
```

## Expected Useful Outcomes

- GRC high/low delta metrics become non-zero by final step.
- Step-20 transfer validation passes the full high/A08 plus low/A15 gate.
- A37-aligned GPU evaluation of `final` improves high/A08 without spending low/A15.
- If `best_val` remains A37 but `final` has a directional tradeoff, the next experiment can add staged guard rather than rollback-from-zero.

## Negative Outcomes

- GRC deltas remain effectively zero even without step rollback.
- Final gains only low/A15 and regresses high/A08, matching G12/G13's protect-only direction.
- Final lifts high/A08 only by spending low/A15.
- Full transfer gate still fails and final test evaluation gives no usable direction.

## Promotion Rule

Do not promote from transfer-validation behavior alone. Promotion requires A37-aligned GPU evaluation to pass high-route, A08-oracle, low-route, and A15-oracle at or above A37 simultaneously.

## Smoke01 Training Result

`G_4kd14_smoke01` completed on physical GPU3 with train exit code `0`.

Static and preflight checks passed:

- Python compile, TOML parse, bash syntax checks, launcher dry-run, eval dry-run;
- GPU3-bound preflight with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`;
- A37 initialization with only expected `gaussian_route_control.*` missing keys and no unexpected keys;
- trainable prefix only `gaussian_route_control`, trainable params `50114`;
- A18 `val_records` alignment `100/100`.

Training summary:

```text
final step: 20
step-level rollback: disabled
best_val step: 0
step20 transfer full gate: failed
final gaussian_route_control_high_delta_abs_mean: 2.574286554590799e-05
final gaussian_route_control_low_delta_abs_mean: 1.9299128325656056e-05
```

Step-20 transfer validation missed all five full-gate metrics:

```text
baseline high-route transfer gap: 0.03710367872908309
step20   high-route transfer gap: 0.036676999684926624

baseline A08-oracle transfer gap: 0.07303136937758502
step20   A08-oracle transfer gap: 0.07297538308536305

baseline low-route transfer gap: 0.417457762218657
step20   low-route transfer gap: 0.41615719265407985

baseline A15-oracle transfer gap: 0.3613124905210553
step20   A15-oracle transfer gap: 0.35986068032004614

baseline low-A15 transfer gap: 0.4313206589012815
step20   low-A15 transfer gap: 0.42999906707228275
```

## A37-Aligned GPU Evaluation

A37 baseline:

```text
PSNR 17.889658453438816
high-route -0.11159004428502844
A08-oracle -0.19984098726138502
low-route 0.1591364853806252
A15-oracle 0.2493535773419097
```

`step_000010`:

```text
PSNR 17.88950629774918 (-0.000152155689637)
high-route -0.11149740290558754 (+0.000092641379441)
A08-oracle -0.19942862311188653 (+0.000412364149498)
low-route 0.15879965821783007 (-0.000336827162795)
A15-oracle 0.24872053438190425 (-0.000633042960005)
```

`final`:

```text
PSNR 17.889438099539603 (-0.000220353899213)
high-route -0.11146861464370558 (+0.000121429641323)
A08-oracle -0.19927614814070793 (+0.000564839120677)
low-route 0.15865829477539373 (-0.000478190605231)
A15-oracle 0.24846435531464198 (-0.000889222027268)
```

Route-regime diagnostics show the input-std prior boundary remained effectively unchanged. The overall selected route delta was about `0.0062685`, and the high branch carried the visible delta while low-route selected delta remained `0.0` in the route-regime evaluation.

## Conclusion

G14 proves the deferred window works mechanistically: the dual adapter can accumulate non-zero high/low deltas when it is not reset every two steps. It also proves this is not yet the breakthrough. The unconstrained learned direction is high/A08-positive but low/A15-negative, reproducing the old tradeoff in a cleaner structural setting.

The next useful experiment should decouple the branch objectives more deeply, not just defer the guard:

- keep the dual high/low adapter capacity;
- keep A18 transfer-aware best selection and A37-aligned promotion gates;
- add a low/A15 protected branch objective or branch-specific acceptance that prevents low-route/A15 spend while allowing high/A08 lift;
- consider a staged or projected update policy after a non-zero adapter warmup, rather than rollback-from-zero.
