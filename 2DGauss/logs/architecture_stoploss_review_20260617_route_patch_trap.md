# Architecture Stop-Loss Review: Route Patch Trap

- Time: `2026-06-17 18:31:46 CST`
- Trigger: user asked whether the current framework is still worth patching, whether recent work is "拆东墙补西墙", and whether the project is drilling into a dead end.
- Scope: A36/A37/A39-A47, current `CrossResolutionGaussianRestorer`, input-std route modules, Gaussian-control modules.

## Verdict

We should stop treating more small patches on the current A37/A47 line as the main breakthrough path.

This does not mean the project should be abandoned. A36 and A37 are real discoveries:

- A36 proves A08/A15 route complementarity is real when full route-specific capacity is preserved.
- A37 proves part of that complementarity can be compressed into a shared single-checkpoint model and beat the A18/A36 aggregate router line.

But A39-A47 show a clear architectural limit for the current recipe:

```text
 shared compact backbone
+ mostly late route heads / tiny learned gates / sampler-loss patches
= aggregate/high/low groups trade against each other instead of improving together
```

The current line has become a route-patch trap. More A48-style gate/sampler tweaks may still be useful as a bounded diagnostic, but should not be the main research route unless they change internal route capacity, not only the gate target.

## Evidence

| run | test PSNR | overall gap | high gap | low gap | A08-oracle gap | A15-oracle gap | meaning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A37 best | `17.889658453438816` | `+0.042724077624394126` | `-0.11159004428502844` | `+0.1591364853806252` | `-0.19984098726138502` | `+0.2493535773419097` | compressed/shared milestone |
| A40 best | `17.900562951956445` | `+0.0536285761420247` | `-0.1180115677202543` | `+0.18311149168725271` | `-0.23833660113789967` | `+0.3023396530841825` | aggregate goes up by low/A15; high/A08 worsens |
| A46 final | `17.87790077389511` | `+0.030966398080690814` | `-0.10929431364232904` | `+0.1367771104331444` | `-0.1659231102631667` | `+0.1986870903736065` | high/A08 improves; aggregate/low/A15 collapses |
| A47 final | `17.889420757227708` | `+0.04248638141328485` | `-0.11217793301246365` | `+0.1591629694888495` | `-0.2010047959040689` | `+0.24990479172066027` | low protected; high/A08 suppressed |

This is the pattern the user warned about:

- A39/A40/A41 push aggregate PSNR up, but mainly by improving low/A15 and worsening high/A08.
- A46 pushes high/A08 up, but breaks low/A15 and aggregate.
- A47 restores low/A15 behavior, but gives back the high/A08 improvement.

That is not random training noise. It is a Pareto trade-off created by the current architecture.

## Architectural Cause

The current model is not a full route-specific model. It is a compact shared CNN/Gaussian-control restorer with late conditional heads:

```text
input
-> shared highres/lowres CNN backbone
-> shared Gaussian attention/control/guidance
-> shared decoder feature
-> route-conditioned decoder heads / terminal heads / gates
-> RGB residual
```

Most attempts after A37 changed one of these late or auxiliary mechanisms:

- high/low route terminal heads;
- route feature expert;
- selected-output anchors;
- learned route gate;
- hard-case sampler;
- global-context gate;
- scalar weighting and trust anchors.

These mechanisms can steer the final blend, but they do not give the model enough clean internal capacity to represent two different restoration policies at the same time:

- A15-like low route wants stronger dehazing/restoration demand.
- A08-like high route wants more conservative high-fidelity preservation.
- Some low-`input_std` samples are A08-oracle winners, so the fixed scalar route boundary is not expressive enough.

With a mostly shared representation, improving one side changes the representation or blend seen by the other side. This is why the experiments repeatedly exchange one gain for another.

## Why This Is Not "No Progress"

The work was not wasted. It narrowed the uncertainty:

- A36: route signal plus full capacity works.
- A37: partial compression works and gives a real aggregate milestone.
- A39-A41: more high sampling, shared control capacity, and scalar oracle weighting do not solve the high tail.
- A42-A43: feature capacity and stricter anchors still fail when the fixed boundary sends A08-oracle low samples to the low route.
- A44-A45: residual-logit was a parameterization failure; bounded residual gate is trainable.
- A46-A47: hard positives and gate mobility are solved; selectivity/capacity remains the bottleneck.

So the current knowledge is sharper: the bottleneck is not training plumbing anymore. It is the architecture's route-specific capacity and boundary representation.

## Stop-Loss Decision

Do not continue the current line as "one more gate/loss/sampler patch" unless it is explicitly capped as a falsification probe.

Stop the following as mainline:

- more hard-case fraction sweeps;
- more gate-head variants with only ~1k trainable parameters;
- more scalar loss reweighting on the same route heads;
- more late output adapters;
- promotion by aggregate PSNR alone.

A48 selective positive-up with hard negatives is still a reasonable one-shot diagnostic, but it should be time-boxed. If it does not improve high/A08 while preserving low/A15, this subline should close.

## Recommended Pivot

Start a new B-series architecture line. The main idea should be:

```text
route-specific internal capacity first,
gate/loss second.
```

Recommended candidates:

1. Slim dual-expert decoder:
   - share highres stem, lowres Gaussian prior, and early fusion;
   - split mid/late decoder blocks or route adapters before RGB residual;
   - initialize/teacher-distill from A08/A15;
   - keep parameter growth measured against A36 full dual backbone.

2. Conditional adapter / LoRA experts at multiple depths:
   - small route-specific adapters in highres stem, lowres fusion, Gaussian guidance, and decoder;
   - not only terminal RGB heads;
   - train with route-specific distillation and hard-case validation.

3. Learned boundary router with real features:
   - use input_std as a prior, not a hard truth;
   - add learned confidence from image/content/Gaussian-control features;
   - train router on A08-vs-A15 oracle records, especially boundary samples.

4. Route-specific Gaussian control:
   - current Gaussian control is mostly shared; make high-preserve and low-dehaze control channels explicitly separate;
   - route-specific control should feed the decoder before final residual, not only blend final outputs.

## Paper-Level Framing

The likely paper contribution should not be "we kept adding small gates until the number moved." A cleaner story is:

```text
Conditional restoration has a route-capacity conflict.
Full dual-backbone solves it but is too expensive.
Naive shared compression causes Pareto trade-offs.
A route-specific compact architecture recovers the Pareto front.
```

A36/A37/A40/A46/A47 are already strong evidence for that story:

- A36: upper bound with full route capacity.
- A37: compact aggregate breakthrough.
- A40: aggregate-only gain that hurts high/A08.
- A46: high/A08 gain that hurts low/A15.
- A47: conservative gate that returns to A37-like behavior.

The next worthwhile breakthrough is not another patch; it is a cleaner architecture that avoids this zero-sum behavior.
