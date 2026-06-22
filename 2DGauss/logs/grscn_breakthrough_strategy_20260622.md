# G-RSCN Breakthrough Strategy Analysis

- Created: `2026-06-22 13:14:59 CST`
- Trigger: `MSG-000709` - 用户要求综合分析新架构、G-RSCN 已有训练结论、旧架构总结，并判断新架构如何突破瓶颈。
- Scope: old A-series stop-loss evidence, G-RSCN `G_4kd01` through `G_4kd10`, current `GaussianRouteControl` implementation, and next deployable mechanism direction.

## Verdict

The new architecture direction is correct, but the first G-RSCN implementation has reached the limit of scalar objectives plus rollback guards.

The bottleneck is now:

```text
update direction / candidate application under competing route constraints
```

not:

```text
Gaussian core design, GPU execution, teacher plumbing, hard-case exposure, or a single validation threshold.
```

`G_4kd11` should therefore not be a longer G10 or another scalar-weight sweep. It should introduce a new update-selection/application mechanism that can separate low/A15-preserving movement from high/A08-lifting movement.

## What The Old Architecture Taught Us

The A-series route-patch line proved the route signal is real:

- A18 fixed `input_std` router shows A08/A15 complementarity.
- A36 full route-specific capacity nearly reproduces the two-checkpoint router inside one checkpoint.
- A37 proves partial compact compression around the Gaussian foundation is useful.

But the old line also proved the failure mode:

- A39/A40 improve aggregate and low/A15 while worsening high/A08.
- A46 improves high/A08 while collapsing low/A15.
- A47 protects low/A15 but suppresses high/A08 again.

This is the route-patch trap:

```text
shared compact representation + late gates/heads/samplers
=> high/A08 and low/A15 trade off instead of moving together.
```

The old architecture should remain evidence and baseline, not the mainline. Repeating late adapters, gate-only correction, sampler-only correction, scalar weighting, or aggregate-PSNR promotion would repeat the known dead end.

## What G-RSCN Fixed

G-RSCN correctly preserved the project identity:

- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- Gaussian attention/control/guidance/refinement;
- cross-resolution restoration path;
- A08/A15 conditional evaluation.

The current `GaussianRouteControl` also fixed one old architectural defect: route-specific capacity now enters before the RGB residual, using decoder features plus Gaussian/control/context maps. Zero-init A37 equivalence passed, and only `gaussian_route_control` is trainable in the G-line probes.

This means G-RSCN is not just another late route head. It is the right architectural direction.

## What G-RSCN Training Proved

`G_4kd01` through `G_4kd02`:

- high/A08 signal is real and trainable;
- branch-level Gaussian route-control supervision is live;
- but high/A08 movement still spends low/A15 surplus.

`G_4kd03`:

- protected checkpoint selection works;
- but selection alone only rejects bad candidates, it does not create a safe update direction.

`G_4kd04`:

- step no-harm rollback works as an update stabilizer;
- but protecting a parent already below A37 cannot satisfy the formal gate.

`G_4kd05`:

- strongest near miss;
- A37 initialization plus full-validation no-harm improves aggregate/low/A15 over A37;
- misses high-route by only `-0.000021236730810156312` and A08 by `-0.00008147229078794838`.

`G_4kd06` through `G_4kd10`:

- G06 preserves the near-miss shape but still fails high/A08 on test.
- G07 proves continuing from G05 final cannot pass five-metric rollback.
- G08 proves validation non-regression can pass but not transfer to test high/A08.
- G09 proves positive validation margins protect transfer but can reject all updates.
- G10 proves A08 was not the only blocker: early high/A08-positive candidates spend low/A15; later balanced candidates miss all margins slightly.

## Current Diagnosis

The current trainable module has enough capacity to move metrics in either direction:

- it can lift high/A08;
- it can preserve or improve low/A15;
- it cannot yet apply a candidate update that does both simultaneously.

So the next problem is not simply "more capacity" or "more training." It is constrained update geometry.

The guard sees two candidate shapes:

1. High/A08-positive movement:
   - good for high route and A08 oracle;
   - spends low route, A15 oracle, and low-A15.

2. Balanced late movement:
   - no catastrophic collapse;
   - but slightly below every required positive margin.

This means plain optimizer steps are mixing incompatible route effects inside the same small `gaussian_route_control` parameter set.

## Required Breakthrough Shape

The next breakthrough should combine two ideas:

```text
route-protected update application
+ Gaussian-conditioned route-specific capacity expansion only if needed
```

Do not jump immediately to a larger architecture without first testing whether update projection/selection can unlock the existing G-RSCN module. But if projection still rejects every candidate, the next escalation must be structural, not scalar.

## Recommended G_4kd11

`G_4kd11` should be a new G-line probe, not a G10 rerun.

Keep:

- A37 best-val initialization;
- immutable Gaussian core;
- trainable prefix only `gaussian_route_control`;
- five positive validation margins;
- full 24-batch step guard;
- single approved GPU policy;
- no formal long training.

Change:

- candidate acceptance/application, not just thresholds.

Recommended mechanism:

```text
two-phase projected candidate acceptance
```

Phase 1: low/A15 protection.

- Evaluate candidate movement on low-route, A15-oracle, and low-A15 margins first.
- Reject or project away any update whose low/A15 margins are below guard.
- Low/A15 surplus is the budget that high/A08 movement is allowed to spend.

Phase 2: high/A08 lift.

- Among candidates that preserve low/A15 surplus, accept only those that improve high-route and A08-oracle margins.
- If a full optimizer step fails, try a smaller projected step or parameter interpolation between baseline and candidate.
- Do not accept high/A08 movement that only works by consuming low/A15.

Implementation-friendly first version:

- after each optimizer step, keep the candidate state;
- evaluate guard metrics;
- if high/A08 passes but low/A15 fails, test one or more interpolation ratios between the previous accepted state and candidate state;
- accept the largest ratio that keeps low/A15 margins above guard and improves high/A08;
- otherwise rollback.

This directly tests whether the safe point lies along the candidate update segment. It is cheaper and less invasive than adding more modules.

## Escalation If G11 Fails

If projected/interpolated acceptance also cannot find a safe update, the conclusion becomes stronger:

```text
the current one-adapter GaussianRouteControl slot is too coupled.
```

Then `G_4kd12` should change architecture, not training scalar knobs:

- split `gaussian_route_control` into separate high-preserve and low-dehaze parameter groups with explicit masks or adapters;
- keep low-dehaze branch near A37 unless low/A15 guard is surplus-positive;
- allow high-preserve branch to learn only where low branch output remains protected;
- consider `adapter_count > 1` only as Gaussian-conditioned multi-depth route control, still zero-init A37 equivalent.

This escalation is different from the old A-line patches because capacity remains Gaussian-conditioned and pre-RGB, not terminal/output-level.

## Stop Rules

Do not:

- lengthen G10;
- rerun G10 with only A08 threshold relaxation;
- raise scalar anchor weights as the main change;
- promote any run by aggregate PSNR alone;
- launch formal long training before all A37-aligned gates pass.

Promote only if A37-aligned test passes simultaneously:

- high-route at or above A37;
- A08-oracle at or above A37;
- low-route at or above A37;
- A15-oracle at or above A37;
- aggregate PSNR not worse in a way that invalidates the route gains.

## Paper-Level Meaning

If G11 succeeds, the paper story is strong:

```text
Gaussian-preserved route-specific control can work,
but only with constrained update application that respects competing restoration policies.
```

If G11 fails and G12 succeeds with split Gaussian-conditioned capacity, the story is also strong:

```text
the Pareto conflict is structural;
compact Gaussian route control needs route-decoupled pre-RGB capacity.
```

Both paths preserve the core idea. The only losing path is continuing scalar patch training after the evidence says the bottleneck is update geometry and route coupling.
