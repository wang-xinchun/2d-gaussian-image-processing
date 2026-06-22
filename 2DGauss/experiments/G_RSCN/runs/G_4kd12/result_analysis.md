# G_4kd12 Result Analysis

- Created: `2026-06-22 16:55:00 CST`
- Status: `smoke01 complete; not promoted`

## Smoke01 Question

`G_4kd12_smoke01` tests whether G11 failed because ordinary validation metrics did not transfer to the A37-aligned route/oracle test surface.

Expected useful outcomes:

- transfer-aware guard accepts at least one candidate that later passes A37-aligned high-route and A08-oracle on test;
- transfer-aware guard rejects all candidates, proving the current G-RSCN slot has no safe transfer-positive movement under the existing capacity;
- projection accepts a smaller candidate segment that passes transfer target/protect gates.

Negative outcomes:

- transfer-aware validation accepts candidates but test high/A08 still regress;
- accepted candidates improve high/A08 but spend low/A15;
- all accepted movement is aggregate-only.

## Deployment Plan

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/configs/g_4kd12_smoke01_a37_transfer_guard_short.toml`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Trainable prefix: `gaussian_route_control`
- Best-selection metric: `transfer_val_a08_oracle_gap_mean`
- Transfer records: A18 `val_records`
- Projection ratios: `[0.75, 0.50, 0.25]`.

## Promotion Rule

Do not promote from transfer-validation behavior alone. Run A37-aligned GPU evaluation only if at least one accepted/projection/protect update creates a real new weight signal. Promotion requires high-route, A08-oracle, low-route, and A15-oracle at or above A37 on test-side evaluation.

## Smoke01 Training Result

`G_4kd12_smoke01` completed on physical GPU3 with train exit code `0`.

Static and preflight checks passed:

- `py_compile scripts/train_v1_t01.py`
- TOML parse for the G12 config
- `bash -n` for launch/eval scripts
- CUDA preflight with single visible physical GPU3
- A18 `val_records` alignment: `100/100` matched.

Training summary:

```text
final step: 16
trainable prefix: gaussian_route_control
trainable params: 37698
peak CUDA reserved: 3.15234375GiB
guard checks: 8
accepted updates: 6
accepted full-gate updates: 0
accepted protect-only updates: 6
rolled back updates: 2
projection-accepted updates: 0
best_val step: 0
```

Step behavior:

```text
step 2: rollback
step 4: rollback
step 6: protect_only
step 8: protect_only
step 10: protect_only
step 12: protect_only
step 14: protect_only
step 16: protect_only
```

The key candidate was step `8`: it was the best transfer-val A08 point but still missed the full-gate A08 threshold.

```text
baseline A08: 0.07303136937758502
required A08 full gate: 0.07313136937758502
step 8 A08: 0.07311837813433479
shortfall: -0.000012991243250
```

This is not a generic validation acceptance problem anymore. The transfer guard did accept movement, but only in protect-only mode. It never found a checkpoint that satisfied high/A08 target and low/A15 protection simultaneously.

## A37-Aligned GPU Evaluation

Because protect-only updates existed, both `best_val` and `final` were evaluated.

```text
A37 baseline:
PSNR 17.889658453438816
high-route -0.11159004428502844
A08-oracle -0.19984098726138502
low-route 0.1591364853806252
A15-oracle 0.2493535773419097

best_val:
PSNR 17.889658453438816 (+0.0)
high-route -0.11159004428502844 (+0.0)
A08-oracle -0.19984098726138502 (+0.0)
low-route 0.1591364853806252 (+0.0)
A15-oracle 0.2493535773419097 (+0.0)

final:
PSNR 17.889701410936713 (+0.000042957497897)
high-route -0.11164274339298112 (-0.000052699107953)
A08-oracle -0.2000703652075532 (-0.000229377946168)
low-route 0.15925160487943105 (+0.000115119498806)
A15-oracle 0.2496285242921635 (+0.000274946950254)
```

`best_val` is simply the A37 parent because no candidate passed the full transfer-aware gate. `final` confirms the protect-only trajectory is not a breakthrough: it buys small aggregate/low/A15 gains but still spends high/A08 on test.

## Conclusion

G12 rules out another tempting explanation. The bottleneck is not ordinary validation selection, not the lack of transfer-aware guard plumbing, and not test evaluation noise. The same-structure `gaussian_route_control` slot can make protective low/A15 improvements, and it can come very close to the A08 transfer target, but it cannot produce a full-gate checkpoint under the current shared route-control capacity.

The next useful probe should be structural, not another scalar guard tweak:

- split Gaussian-conditioned high-preserve and low-dehaze route-control capacity;
- give high/A08 and low/A15 separate route-conditioned control paths or masks before RGB residual generation;
- keep A18 transfer-aware guard as the selection surface;
- keep A37-aligned test promotion gates unchanged.

Formal long training remains blocked.
