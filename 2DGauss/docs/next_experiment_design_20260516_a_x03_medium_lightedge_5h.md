# Next Experiment Design: V1/T01/A_x03

## Record Time

- Time: `2026-05-16 02:00:18 CST`

## Position In The Line

`A_x03` is prepared after:

1. frozen official deblur baseline `M03/A_k03`;
2. completed decoder-guidance short family `A_m03/A_n03/A_o03`;
3. completed medium follow-up `A_p03`;
4. completed negative terminal-control probe `A_q03`;
5. completed post-`A_q03` short family `A_r03/A_s03/A_t03`;
6. completed rebalance probes `A_u03` and `A_v03`, where `A_u03` remains the
   official family leader.

## Why This Branch Exists

The rebalance family has now answered a more specific question than `A_t03`
alone could answer.

- `A_t03` showed only a tiny positive signal.
- `A_u03` converted that into the clearest official positive result inside the
  family by reducing supervision-side pressure.
- `A_v03` then tested the opposite hypothesis and failed to beat `A_u03` on
  official test.

So the correct next long-run action is:

> keep `A_u03` as the structural anchor, and spend the next `5`-hour window on
> long-budget optimization of the lighter-edge feature-refinement line rather
> than on stronger feature pressure.

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_x03
```

It keeps from `A_u03`:

- Gaussian-guided decoder modulation enabled
- decoder-guidance `strength = 0.15`
- Gaussian-guided feature refinement enabled
- feature-refinement `strength = 0.10`
- edge-weighted reconstruction enabled
- lighter edge-weighted supervision `strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective train split after the first `40` pairs

It changes relative to `A_u03`:

- runtime tier: `short -> medium`
- wall-clock cap: `58 -> 300` minutes
- max steps: `2500 -> 24000`
- launch geometry: re-calibrated branch-locally for the medium budget

## What This Branch Tests

`A_x03` asks the first real promotion question for the lighter-edge
feature-refinement line:

> if we keep the current best rebalance structure fixed and give it a `5`-hour
> run under a safe `85-93G` branch-local geometry, does the line stay merely a
> small short-run improvement, or does it compound into a more meaningful
> medium-budget gain?

## Promotion Logic

Before launch, this branch must:

1. complete fresh branch-local smoke calibration;
2. choose the highest-value launch geometry that stays inside the practical
   memory envelope;
3. pass formal preflight on the final run-local config;
4. launch in tmux on a clean GPU;
5. verify startup metrics before treating it as an active medium run.
