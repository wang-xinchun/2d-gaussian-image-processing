# Next Experiment Design: V1/T01/A_p03

## Record Time

- Time: `2026-05-15 03:17:59 CST`

## Position In The Line

`A_p03` is the first `5`-hour medium-budget promotion run launched **after**
the completed `A_m03/A_n03/A_o03` short-scout family.

It follows:

1. frozen official deblur baseline `M03/A_k03`;
2. design-only decoder-guidance branch `A_l03`;
3. short scouts `A_m03`, `A_n03`, and `A_o03`.

## Why This Branch Exists

The short-scout family has now answered the two key pre-long-run questions.

First, the mechanism itself is directionally positive:

- `A_m03` best short-run full-image val PSNR = `28.8486`
- `A_n03` best short-run full-image val PSNR = `28.9612`
- `A_o03` best short-run full-image val PSNR = `28.8394`

Second, the preferred structural strength and the preferred launch shape are
not the same question.

- `A_n03` gives the strongest short-run architectural signal.
- `A_o03` answers the near-`A_k03` high-fill launch question, but it does so
  with a practical warning: although `7x748` smoke reports
  `reserved_gb=92.5859`, the real startup window rises to about `95.5G`,
  above the user’s `93G` upper bound.

So the correct next long-run action is:

> keep `A_n03` as the structural anchor, but recalibrate a separate medium-run
> launch geometry that stays safely inside the real `85-93G` window.

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_p03
```

It keeps from `A_n03`:

- Gaussian-guided decoder modulation enabled
- default guidance strength `0.15`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective train split after the first `40` pairs
- official full-image validation/test protocol
- lightweight Gaussian-first model width/depth

It changes relative to `A_n03`:

- run identity: `A_n03 -> A_p03`
- runtime tier: `short -> medium`
- wall-clock cap: `58 -> 300` minutes
- max steps: `2500 -> 24000`
- branch-local launch geometry: `batch_size=6`, `train_context_size=800`

## Why `6x800`

This geometry is chosen from fresh branch-local calibration on a clean GPU:

- `6x792`: `reserved_gb=89.5762`
- `6x800`: `reserved_gb=91.3105`

Both points fit the configured `85-93G` smoke window, but `6x800` gives the
largest validated context increase without touching the more dangerous
`batch_size=7` envelope exposed by `A_o03`.

This means `A_p03` tries to spend the `5`-hour window in the highest-value
way currently available:

1. preserve the winning decoder-guidance strength from `A_n03`;
2. modestly enlarge context beyond the short-probe `784`;
3. keep real-launch risk materially lower than the `A_o03` high-fill path.

## What This Branch Tests

`A_p03` asks the first real promotion question for the decoder-guidance line:

> if we keep the strongest short-scout structure fixed and give it a full
> `5`-hour medium-budget run under a safe high-VRAM geometry, can it move the
> official `UHD_deblur` result beyond the frozen `A_k03/M03` baseline?

## Why This Variable Comes Next

Among the completed short scouts, `A_n03` is the best nomination anchor
because:

1. it has the highest short-run full-image validation signal;
2. it also has the strongest corroborative official test among the family;
3. unlike `A_o03`, it does not rely on a startup envelope that already showed
   real-window overflow.

## Deployment Expectation

Before launch, this branch should:

1. complete branch-local smoke calibration near the intended `6x800` shape;
2. run formal preflight on the final config;
3. launch in tmux on a clean GPU;
4. verify startup metrics and real memory occupancy before treating it as the
   active medium run.
