# Next Experiment Design: V1/T01/A_k03

## Record Time

- Time: `2026-05-14 10:31:38 CST`

## Position In The Line

`A_k03` is the first longer-budget branch designed **after** the `A_j03`
negative short probe.

It follows:

1. accepted and frozen baseline `A_i03/M02`;
2. rejected `A_j03` context-`864` short probe.

## Why This Branch Exists

`A_j03` already told us what **not** to do next:

- do not directly promote `864` context into the next long run.

At the same time, `A_i03` still leaves one unusually clean opening:

- it was configured with `wall_clock_minutes=300`;
- but it actually stopped at `step=12000` after only `156.1359` minutes.

So the strongest next question is not “what new mechanism should we add,” but:

> what happens if we let the accepted `A_i03` setting actually use something
> much closer to its intended runtime budget?

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_k03
```

It keeps from `A_i03/M02`:

- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `train_context_size=784`
- `train_supervision_crop_size=512`
- effective training split after the first `40` pairs
- batch size `7`
- target real-launch VRAM window `85-93G`
- wall-clock cap `300` minutes
- official full-image validation/test protocol

It changes only:

- run identity: `A_i03 -> A_k03`
- `max_steps`: `12000 -> 24000`

## What This Branch Tests

This branch asks the cleanest post-`A_j03` long-budget question:

> if we do not change context, data split, or mechanism, and only remove the
> earlier step cap, does the accepted consistency line keep improving on the
> official full-image test?

## Why This Variable Comes Next

Among the plausible post-`A_j03` moves, this one has the best signal quality:

1. it fully respects the negative result from `A_j03`;
2. it does not re-open the context-allocation branch prematurely;
3. it preserves the current accepted method story exactly;
4. it can directly answer whether `A_i03` still had unused optimization headroom.

## Deployment Expectation

Before launch, this branch should:

1. re-check `7x784` on a clean GPU with branch-local smoke calibration;
2. verify that the selected setting still sits inside `85-93G`;
3. run formal preflight;
4. launch in tmux and verify real startup memory and metric writing.
