# G_4kd15 Result Analysis

- Created: `2026-06-22 22:18:00 CST`
- Status: `smoke01 complete; not promoted`

## Smoke01 Question

`G_4kd15_smoke01` tests the current bottleneck more directly than G13/G14:

```text
Can dual Gaussian route-control capacity form a non-zero high/A08 direction
while a delayed transfer-aware no-harm guard prevents low/A15 spend?
```

This is a schedule/acceptance probe, not a promotion shortcut. The A37-aligned route/oracle test gate remains unchanged.

## Expected Useful Outcomes

- GRC high/low delta metrics become non-zero by step 10.
- At least one step guard check after step 10 is accepted by full or projected protect+target gate.
- Final or best-val A37-aligned GPU evaluation improves high/A08 without spending low/A15.
- If every post-warmup candidate rolls back, the next bottleneck is not only warmup timing; it is branch objective shape or transfer gate observability.

## Negative Outcomes

- GRC deltas stay near zero despite warmup.
- All post-warmup candidates roll back and no projection is accepted.
- Projected updates preserve low/A15 but fail to lift high/A08.
- Final/test behavior repeats G14: high/A08 up while low/A15 down.

## Promotion Rule

Do not promote from transfer-validation behavior alone. Promotion requires A37-aligned GPU evaluation to pass high-route, A08-oracle, low-route, and A15-oracle at or above A37 simultaneously.

## Smoke01 Training Result

`G_4kd15_smoke01` completed on physical GPU3 with train exit code `0`.

Static and preflight checks passed:

- Python compile, TOML parse, bash syntax checks, launcher dry-run, and eval dry-run;
- GPU3-bound preflight with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`;
- A37 initialization with only expected `gaussian_route_control.*` missing keys and no unexpected keys;
- trainable prefix only `gaussian_route_control`, trainable params `50114`;
- A18 `val_records` alignment `100/100`.

Training summary:

```text
final step: 20
guard start step: 10
guard interval: 2
guard checks: 6
guard accepted: 0
guard rollbacks: 6
projection accepted: 0
best_val step: 0
```

Delayed guard checks:

```text
step 10: rollback, full/protect/target failed
step 12: rollback, full/protect/target failed
step 14: rollback, full/protect/target failed
step 16: rollback, full/protect/target failed
step 18: rollback, full/protect/target failed
step 20: rollback, full/protect/target failed
```

Step-20 transfer guard candidate:

```text
baseline high-route transfer gap: 0.03710367872908309
step20   high-route transfer gap: 0.03732513737034153

baseline A08-oracle transfer gap: 0.07303136937758502
step20   A08-oracle transfer gap: 0.07288708406336167

baseline low-route transfer gap: 0.417457762218657
step20   low-route transfer gap: 0.4171813329060872

baseline A15-oracle transfer gap: 0.3613124905210553
step20   A15-oracle transfer gap: 0.36124710603193805

baseline low-A15 transfer gap: 0.4313206589012815
step20   low-A15 transfer gap: 0.43102905206512987
```

The high-route transfer metric moved in the desired direction, but A08 and all low/A15 protection metrics stayed below the A37 transfer baseline. Since `target_gate_passed=false`, the current projection implementation made no projection attempts.

Final route-regime diagnostics show the final checkpoint is not a pure zero-signal state:

```text
overall selected_delta_abs_mean: 0.0062689986173063515
overall high_delta_abs_mean:    0.017307523572817445
overall low_delta_abs_mean:     0.0
```

The useful interpretation is therefore not "no adapter signal"; it is "the warmup signal is not yet a promotable direction."

## A37-Aligned GPU Evaluation

A37 baseline:

```text
PSNR 17.889658453438816
high-route -0.11159004428502844
A08-oracle -0.19984098726138502
low-route 0.1591364853806252
A15-oracle 0.2493535773419097
```

G15 `final`:

```text
PSNR 17.889589332557936 (-0.000069120880880)
high-route -0.11164105487970442 (-0.000051010594676)
A08-oracle -0.19993386251938058 (-0.000092875257996)
low-route 0.15905370235401622 (-0.000082783026609)
A15-oracle 0.24930469167116928 (-0.000048885670740)
```

Promotion decision: `NOT PROMOTED`.

## Conclusion

G15 closes the warmup-schedule question. Delaying the guard allows a small non-zero adapter state and prevents the larger G14-style low/A15 spend from accumulating, but it does not create a breakthrough direction. The warmup state itself is already slightly below A37 on the transfer/test gates, and every post-warmup optimizer update remains unsafe.

Next work should not lengthen G15 or only retune warmup length. The next useful probe should change the branch objective or acceptance geometry so that the first accepted movement is already A37-safe on A08 and low/A15, for example by making projection/line search able to test smaller candidate segments before requiring target pass, or by adding a direct A37/teacher anti-regression constraint on low/A15 while preserving high/A08 pressure.
