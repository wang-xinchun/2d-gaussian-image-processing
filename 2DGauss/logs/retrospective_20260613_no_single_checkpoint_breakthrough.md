# Retrospective: Why There Is No Final Breakthrough Yet

- Date: `2026-06-13 CST`
- Scope: Review `A_4kd08` through `A_4kd23`, especially why the project has a mechanism milestone but not yet a single-checkpoint or paper-level breakthrough.

## Short Answer

The project has already found a real mechanism signal: `A_4kd08` and
`A_4kd15` are complementary, and the fixed `input_std` router validated by
`A_4kd18`, `A_4kd19`, and `A_4kd21` beats either checkpoint alone. The reason
there is still no final breakthrough is that this gain is conditional. It lives
in per-sample selection between two behaviors, not in one globally better set
of weights.

Current single-checkpoint attempts still use global pressure: EMA, haze guards,
teacher consistency, or weak output distillation. Those knobs move the whole
student in one direction, so they either preserve A08-like high-fidelity samples
and lose A15 hard-sample gains, or keep A15-like gains and lose A08's protected
tail. The project has identified the route, but has not yet built a model
structure that can hold both route behaviors.

## Evidence Chain

### A_4kd08: First strong fast-probe candidate, but unstable

`A_4kd08` fixed the overly conservative detail-gate problem and became the
strict fast-probe PSNR candidate:

- CPU fast-test PSNR: `17.789133503142946`
- It clearly beat earlier `A_4kd05`, but had late-training instability and
  dark-channel/saturation side effects.

This was a real improvement, but not a stable final model.

### A_4kd12 to A_4kd15: EMA and no-guard variants recovered stability, not a clean supersede

The EMA/no-haze-guard line improved stability, loss, and several proxies. The
best point, `A_4kd15`, reached:

- CPU fast-test PSNR: `17.781017099305146`
- SSIM: `0.6972798517346382`
- Loss: `0.12060526199638844`

It nearly matched A08 and had better loss/proxy balance, but it still missed
A08 by about `0.00812 dB`. More importantly, sample-level review showed that
A15 damaged some samples where A08 was already strong. That is the first clear
sign that the bottleneck is not a scalar hyperparameter.

### A_4kd17: Teacher consistency used the right idea too weakly and globally

`A_4kd17` tried to preserve A08 behavior through teacher consistency, but it
fell below both A08 and A15:

- A17 PSNR: `17.722200177109993`
- Below A08 by about `0.06693 dB`
- Below A15 by about `0.05882 dB`

It nudged some watched A08-tail samples in the right direction but harmed the
many samples where A15 had gained. This showed that simply adding an A08 teacher
loss to the same shared student path creates gradient conflict rather than a
clean conditional behavior.

### A_4kd18, A_4kd19, A_4kd21: Real mechanism milestone, but two-checkpoint

The router line is the strongest evidence so far.

- `A_4kd18` fast-test router PSNR: `17.84693437641316`
- `A_4kd19` non-overlapping offset100 fixed-rule router beat both single checkpoints.
- `A_4kd21` larger offset200 GPU eval also beat both single checkpoints.

This proves the project found a real mechanism: A08 and A15 are not merely two
nearby checkpoints; they specialize differently. `input_std <=
0.07026463374495506` is a robust proxy for choosing A15, while higher
`input_std` tends to need A08 preservation.

But this is still two-checkpoint inference. It is a mechanism milestone, not a
single-model breakthrough.

### A_4kd22: Richer learned routers overfit; the stable signal is constrained

`A_4kd22` tested saved-record learned logistic routers across A18/A19/A21 using
leave-one-source-out validation. The fixed A18 input-std rule remained the best
aggregate method:

- Fixed rule PSNR: `15.761350330784339`
- Best learned logistic, `input_mean_std`: `15.758408523282599`
- Richer `all_common`: `15.748658098940552`

This says the problem is not "find more features and train a free router." The
stable signal is low-dimensional and constrained. Richer features can win a
block and lose the aggregate.

### A_4kd23: Weak output distillation did not convert the router into one checkpoint

`A_4kd23` made the first direct single-checkpoint conversion attempt: start from
A15 and distill toward the A18 input-std router target.

Results:

- A23 best-val step50 PSNR: `17.78064128762362`
- A15 PSNR: `17.781017099305146`
- A08 PSNR: `17.789133503142946`
- A18 router PSNR: `17.84693437641316`
- A23 final step300 PSNR: `17.035365402315897`

The branch was active and the engineering path worked, but the student remained
A15-like early and then drifted. Weak whole-output distillation did not create a
separate A08-like high-input-std behavior inside the model.

## Root Causes

1. The current gain is conditional, not global.
   A08 and A15 are each useful on different sample regimes. A single global
   training pressure tends to average those regimes or favor one side.

2. The architecture has no explicit place to store the route.
   The current student has one shared prediction path. A router gain needs a
   route-conditioned correction path, dual head, gating head, or equivalent
   internal mechanism.

3. The losses used so far are too soft or too global.
   Teacher consistency and router distillation are whole-output constraints
   applied to the same shared model. They do not force a hard, protected
   high-input-std A08 branch.

4. Training distribution is not balanced by the discovered regimes.
   Without explicit high/low `input_std` balancing, the dominant A15-like mode
   can absorb training, while the rarer A08-protected tail is not held strongly
   enough.

5. Validation gain and test gain are small enough to be easy to erase.
   The real router gain is about `0.058-0.066 dB` on the fast-test block. That
   is meaningful for this protocol, but tiny relative to training noise and
   loss tradeoffs. A weak conversion method can lose it immediately.

6. The project was initially optimizing knobs before the bottleneck was named.
   Earlier runs tested detail gates, reliability gates, haze guards, EMA, input
   identity, and teacher consistency. Those were useful diagnostics, but the
   later evidence shows the bottleneck is conditional specialization.

## What This Means

The project is not blocked because there is no signal. The signal is now quite
clear. The project is blocked because the current model/training path cannot
represent the signal as one checkpoint.

The next breakthrough attempt should not be another weak scalar sweep. It should
be a structural conversion attempt:

- hard per-sample router-label supervision from the A18 rule;
- balanced batches or loss reweighting across low/high `input_std` regimes;
- an explicit gating head or dual-head output path;
- A08-preserving branch for high `input_std`;
- residual/correction distillation rather than whole-output soft averaging;
- cross-block validation against A18/A19/A21 records before full promotion.

## Practical Next Step

The next experiment should be named as a structural conversion probe, not just a
distillation-weight sweep. A reasonable `A_4kd24` would use the A18 rule as a
hard route label, train an explicit route-conditioned correction or dual-head
student, and report per-regime metrics:

- low `input_std`, expected A15 branch;
- high `input_std`, expected A08 branch;
- aggregate fast-test PSNR/SSIM/loss;
- comparison against A08, A15, A18 router, and A23.

Promotion bar: the single checkpoint must at least beat both A08 and A15 on the
same fast-test protocol, and preferably close most of the gap to the A18
two-checkpoint router.
