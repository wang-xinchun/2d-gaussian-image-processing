# G_4kd11 Result Analysis

- Created: `2026-06-22 16:05:00 CST`
- Status: `smoke01 completed; validation-safe but not promoted`

## Smoke01 Question

`G_4kd11_smoke01` tests whether the G10 failure is an update-application problem rather than an absence of safe nearby movement.

Expected useful outcomes:

- `accepted_mode=protect_only`: low/A15 surplus can be built under A37-start G-RSCN without immediately requiring high/A08 lift.
- `accepted_mode=projection`: high/A08-positive candidate movement can be partially applied while preserving low/A15.
- `accepted_mode=full`: a candidate directly passes all five validation margins.

Negative outcomes:

- all candidates roll back;
- projection attempts never pass protect+target;
- protect-only movement accumulates but still cannot support high/A08 lift by final step.

## Deployment Plan

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/configs/g_4kd11_smoke01_a37_two_phase_projection_short.toml`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Trainable prefix: `gaussian_route_control`
- Step guard: full five-metric guard plus low/A15 protect gate and high/A08 target gate.
- Projection ratios: `[0.75, 0.50, 0.25]`.

## Promotion Rule

Do not promote from validation behavior alone. Run A37-aligned GPU evaluation only if at least one accepted/projection/protect update creates a real new weight signal. Promotion requires high-route, A08-oracle, low-route, and A15-oracle at or above A37 on test-side evaluation.

## Smoke01 Training Result

Training completed on physical GPU3 with exit code `0`.

- Max step: `20`.
- Step guard checks: `10`.
- Accepted updates: `10`.
- Rollbacks: `0`.
- Accepted modes: `full=10`.
- Projection attempts: `0`, because every checked candidate passed the full five validation gates.
- Best validation step: `6`.

Validation trajectory:

- Baseline A08-oracle gap: `-0.10040664672851562`.
- Best step `6` A08-oracle gap: `-0.09999656677246094`.
- Best step `6` high-route gap: `-0.012229283650716146`.
- Best step `6` low-route gap: `0.433758544921875`.
- Best step `6` A15-oracle gap: `0.2590391852638938`.
- Final step `20` A08-oracle gap: `-0.10014915466308594`.
- Final step `20` high-route gap: `-0.012641270955403646`.
- Final step `20` low-route gap: `0.4331509272257487`.
- Final step `20` A15-oracle gap: `0.25847023183649237`.

This proves G11 can accept repeated full-five-validation-margin updates from A37, unlike G10. It does not prove projection is the missing mechanism, because projection was not exercised.

## A37-Aligned Test Result

A37 baseline:

```text
test PSNR: 17.889658453438816
high-route gap: -0.11159004428502844
low-route gap: 0.1591364853806252
A08-oracle gap: -0.19984098726138502
A15-oracle gap: 0.2493535773419097
```

`best_val`:

```text
test PSNR: 17.889756173256146 (+0.000097719817330)
high-route gap: -0.111665766148642 (-0.000075721863613)
low-route gap: 0.159365047167789 (+0.000228561787164)
A08-oracle gap: -0.200214915987967 (-0.000373928726582)
A15-oracle gap: 0.249853071844794 (+0.000499494502884)
```

`final`:

```text
test PSNR: 17.889620811442292 (-0.000037641996524)
high-route gap: -0.111663922610018 (-0.000073878324990)
low-route gap: 0.159126179561716 (-0.000010305818909)
A08-oracle gap: -0.200062964555102 (-0.000221977293717)
A15-oracle gap: 0.249472961709662 (+0.000119384367752)
```

Promotion result: `NOT PROMOTED`.

## Interpretation

G11 moves the bottleneck forward. The previous G10 failure was "no accepted update under the chosen margin logic"; G11 fixes that on validation, accepting every checked candidate under the full five validation gates. The remaining failure is validation-to-test transfer: the selected checkpoint improves aggregate PSNR, low-route, and A15-oracle on test by tiny margins, but still regresses high-route and A08-oracle.

The route gate regime is still essentially the A37 input-std prior. `best_val` has high-route gate mean `0.966229241947795`, low-route gate mean `0.023600109885921854`, and overall gate mean `0.42893063667252734`, matching the prior means. The learned signal is therefore tiny route-control deltas around a fixed prior boundary, not a real boundary correction.

Next recommended direction:

- Do not promote or lengthen G11.
- Do not tune only scalar margins.
- `G_4kd12` should either add a test-transfer-aware validation guard using a held-out A18-aligned route subset, or split Gaussian-conditioned route-control capacity into high-preserve and low-dehaze components so validation-safe low/A15 movement does not leave high/A08 transfer behind.
