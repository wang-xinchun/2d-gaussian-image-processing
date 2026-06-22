# Performance Shortfall Analysis: Architecture, Logs, Data, and Theory

- Date: `2026-06-13 CST`
- Request: Analyze the model architecture together with logs and training data, identify performance shortfalls, and give the scientific explanation.
- Scope:
  - Current V1/T01 architecture after A24-A26.
  - Dehazing line `A_4kd08` through `A_4kd26`.
  - Training data views under `/home/ldj/wxc/2DGauss/datasets/current_training_ready`.
  - Router records from A18/A19/A21.

## Core Conclusion

The main performance shortfall is not that the model cannot dehaze, and not
that the `input_std` signal is weak. The shortfall is that the current
single-checkpoint architecture still represents two conflicting restoration
regimes through mostly shared features and a shared terminal prediction path.

The data and logs show two regimes:

- low `input_std`: A15-like behavior improves hard/low-variance samples;
- high `input_std`: A08-like behavior preserves already-good, higher-variance,
  high-fidelity samples.

A24 proves structural routing helps, but its route branch is too late and too
small. It improves low-route samples strongly, but still fails to preserve the
high-route A08 tail. Therefore the remaining gap to A18 is scientifically a
conditional-specialization and capacity-placement problem.

## Data Context

Current trainable paired restoration inventory:

- T01-ready paired datasets: `6`.
- T01-ready paired train pairs: `31719`.
- 4KDehaze local split:
  - raw paired files: `15606`;
  - local train: `14045`;
  - local val: `780`;
  - local test: `781`.

Current active dehazing fast-probe view:

- `4KDehaze_local_90_5_5_fast_probe_512`
- train: `512`;
- val: `100`;
- test: `100`;
- crop size: `512`.

The fast-probe validation/test view is deliberately small and scene-concentrated:

- fast val prefixes: `8` has `72` samples, `89` has `28`;
- fast test prefixes: `94` has `53` samples, `95` has `47`;
- full local val has `7` prefixes and `780` samples;
- full local test has `7` prefixes and `781` samples.

This means the fast-probe protocol is excellent for mechanism discovery and
rapid iteration, but not enough for final generalization claims. A18/A19/A21
partially address this by testing non-overlapping and larger offset blocks.

## Architecture Context

The current `CrossResolutionGaussianRestorer` is best understood as:

```text
input RGB
  -> high-resolution CNN stem
  -> low-resolution pixel-unshuffle branch
  -> Gaussian parameter head + Gaussian attention renderer
  -> attention-modulated high-resolution features
  -> shared decoder feature
  -> shared RGB residual
  -> optional terminal corrections
  -> restored = input + residual
```

The Gaussian part is a structural prior, not the final RGB renderer. It predicts
attention/control signals that modulate a CNN restoration backbone.

A24 adds:

```text
decoder_feature + input + input_std prior
  -> shared route encoder
  -> high_delta branch
  -> low_delta branch
  -> input_std gate blend
  -> add to shared base residual
```

This is a real structural route, but it is inserted after most shared feature
processing has already happened. It does not give high/low regimes independent
Gaussian priors, independent decoder features, or independent terminal RGB
heads.

## Evidence From A08-A26

### A08: strong high-fidelity candidate, unstable and side-effect prone

A08 reached fast-test:

- PSNR `17.789133503142946`;
- SSIM `0.6947606813907623`;
- loss `0.1228142374008894`.

It improved the early fast-probe frontier, but the training curve was volatile:
best validation step700 was `18.163864803283275`, while final step800 fell to
`14.466338503701529`.

Interpretation: A08 has useful high-fidelity behavior, especially on high
`input_std` tail samples, but it is not stable and has worse proxy metrics such
as dark-channel and saturation.

### A15: better loss/proxies, but damages the A08 high-fidelity tail

A15 reached:

- PSNR `17.781017099305146`;
- SSIM `0.6972798517346382`;
- loss `0.12060526199638844`.

It improves loss, SSIM, and several proxy diagnostics over A08, but is lower in
mean PSNR. It wins `54/100` fast-test samples against A08, with median delta
`+0.019215287 dB`, yet mean delta is `-0.008116404 dB` because it damages
high-PSNR A08-tail samples.

This is the first strong sign of a conditional tradeoff: A15 is not globally
better or worse; it is better on one sample regime and worse on another.

### A18/A19/A21: the conditional signal is real

The fixed A18 rule:

```text
input_std <= 0.07026463374495506 -> A15
input_std  > 0.07026463374495506 -> A08
```

beats single checkpoints across multiple blocks:

- A18 fast-test: router `17.84693437641316`;
- A19 offset100 test: fixed router `15.913737915082573`;
- A21 offset200 test: fixed router `14.64236451612018`.

The same pattern appears in router records. Across A18/A19/A21 val/test records
(`800` records total):

- A15 wins `403/800` against A08;
- A08 wins `397/800`;
- low `input_std` group (`413` records) has mean `A15 - A08 = +0.137083 dB`;
- high `input_std` group (`387` records) has mean `A15 - A08 = -0.117659 dB`;
- fixed threshold rule agrees with the binary oracle on `644/800` records;
- correlation between `input_std` and `A15 - A08` is `-0.5774`.

This is the strongest scientific evidence. The performance ceiling is not a
single smooth optimum; it is a mixture of two conditional optima.

### A22: richer free routers overfit

A22 showed that learned logistic routers on richer features do not beat the
fixed input-std rule in aggregate across A18/A19/A21:

- fixed input-std rule: `15.761350330784339`;
- best learned `input_mean_std`: `15.758408523282599`;
- richer `all_common`: `15.748658098940552`.

Interpretation: the stable signal is constrained and low-dimensional. The
solution should not be a free learned router that chases one validation block.
The model should use `input_std` as a strong prior, with only limited learned
adaptation.

### A23: weak output distillation cannot solve gradient conflict

A23 tried to distill the two-checkpoint router into one A15-initialized student
through final-output distillation. It reached only:

- PSNR `17.78064128762362`;
- essentially A15-like;
- `-0.06629308878954 dB` below A18 router.

Reason: a single shared output path receives conflicting supervision. The
student tends to average or bias toward the dominant/shared behavior rather than
storing two behaviors.

### A24: structural route is positive, but the high route remains weak

A24 reached:

- PSNR `17.790202882742463`;
- SSIM `0.7000000587105751`;
- loss `0.12000417917966842`;
- total loss `0.1232352689653635`.

It beats A08/A15/A23 as a single checkpoint, and it beats A18 router on SSIM,
loss, total loss, dark-channel, edge, high-frequency, local-contrast, saturation,
and chroma proxies. But it is still below A18 router PSNR by `0.056731493670697
dB`.

The per-sample alignment with A18 explains why:

- low-route samples (`57/100`): A24 is above A18 router by `+0.069979 dB` on
  average;
- high-route samples (`43/100`): A24 is below A18 router by `-0.224697 dB` on
  average;
- samples where A15 is the oracle winner (`54/100`): A24 is above oracle by
  `+0.132435 dB` on average;
- samples where A08 is the oracle winner (`46/100`): A24 is below oracle by
  `-0.358408 dB` on average.

Worst A24 gaps versus the A18 router are exactly the known A08-tail samples:

- `94_94000077.jpg`: A24 `21.249`, A08/router `22.051`, gap `-0.802`;
- `94_94000084.jpg`: A24 `21.684`, A08/router `22.485`, gap `-0.801`;
- `94_94000073.jpg`: A24 `21.535`, A08/router `22.332`, gap `-0.797`;
- `94_94000080.jpg`: A24 `21.116`, A08/router `21.907`, gap `-0.791`;
- `94_9400009.jpg`: A24 `20.338`, A08/router `21.099`, gap `-0.760`.

Therefore A24's low route is already strong; the shortfall is high-route
A08-like preservation.

### A25/A26: forced balancing is not the missing ingredient

A25 failed to guarantee crop-level route balance. A26 fixed that technically,
with observed `input_std_route_high_hard_mean=0.5`, but early validation stayed
around `18.236443073291603`, far below A24's `18.349088096641417`.

Interpretation: the bottleneck is not simply exposure to high-route samples.
Hard 50/50 sampling changes the natural distribution and hurts the learned
restoration manifold.

## Scientific Explanation Of The Shortfall

### 1. Conditional risk, not global risk

The data distribution is a mixture:

```text
P(data) = P(low_input_std regime) + P(high_input_std regime)
```

The best action differs by regime. A15 is better in the low-`input_std` regime;
A08 is better in the high-`input_std` regime. A single shared model trained with
one global objective minimizes average risk, so gradients from the two regimes
partially cancel.

This is why whole-output distillation, teacher consistency, and simple loss
weights fail: they apply pressure to the same parameters for two incompatible
local optima.

### 2. Negative transfer between restoration regimes

Low-variance heavy-haze samples need stronger correction. High-variance samples
often contain more texture, contrast, or already-recoverable high-fidelity
structure. The same correction strength can help the first group but overcorrect
the second.

In multi-task learning terms, the two regimes produce conflicting gradients.
A15 improves many hard samples but damages A08's already-good tail. A17's
teacher consistency confirms this: it pulled toward A08 but harmed A15's hard
sample gains.

### 3. PSNR is sensitive to high-fidelity tail damage

A24 improves many L1/proxy metrics, but PSNR remains lower than A18 because
PSNR punishes squared-error spikes. A few high-quality A08-tail samples losing
`0.7-0.8 dB` can erase broad small gains elsewhere.

This explains why A24 can beat A18 on SSIM/loss/proxies but still lose PSNR.
The model is visually/structurally improving the average, but it has not
protected the high-PSNR tail.

### 4. The current route is late and low-capacity

A18 routes the entire forward path by choosing a full checkpoint. A24 routes
only a terminal correction. This creates an expressivity gap:

```text
A18 high route: full A08 feature extractor + Gaussian prior + decoder + output
A24 high route: shared A15-like path + small high_delta
```

If the shared path has already discarded or changed A08-specific feature
behavior, a small terminal delta cannot fully recover it.

### 5. The Gaussian prior is shared before route selection

The model's Gaussian attention/control stack is a shared structural prior. If
A08 and A15 differ in how they use attention, detail gating, control canvas, or
decoder refinement, A24 cannot express that difference because routing happens
after those modules.

This is not a bug; it is the current architecture's capacity boundary.

### 6. The fast-probe data protocol amplifies both discovery and uncertainty

The fast-probe split is small and scene-concentrated, so small PSNR differences
can be real mechanisms but still fragile as final claims. A18/A19/A21 reduce
this concern by validating the input-std rule on other blocks, but A24 itself
still needs broader/full split verification once the architecture is stronger.

## Performance Shortbacks, Ranked

1. High-`input_std` A08-tail preservation is the primary shortfall.
   Evidence: A24 is `-0.224697 dB` vs A18 router on high-route samples and
   loses up to `-0.802 dB` on known A08-tail samples.

2. Route capacity is too small and too late.
   Evidence: A24 route deltas are small and only terminal; A18 gets a full A08
   path.

3. Shared Gaussian/control features are not route-conditioned.
   Evidence: all major feature construction happens before A24's route head.

4. Loss/objective mismatch remains.
   Evidence: A24 improves SSIM/loss/proxies over A18 but loses PSNR, pointing to
   squared-error tail sensitivity.

5. Fast-probe protocol is not broad enough for final claims.
   Evidence: fast val/test are only `100/100` and each has two scene prefixes.

## Practical Implication

The next architecture should not mainly increase the route loss or rebalance
the sampler. It should give high/low regimes independent capacity at the point
where PSNR-critical residuals are formed.

Most direct next design:

```text
decoder_feature
  -> high terminal decoder head -> high residual/restored
  -> low terminal decoder head  -> low residual/restored
  -> fixed input_std prior + small learned gate
  -> blended restored output
```

This preserves A24's constrained route prior while giving the high route enough
capacity to reconstruct A08-like tail behavior.

The success criterion should be per-regime:

- keep A24's low-route gains;
- reduce high-route gap to A08/A18;
- preserve A24's SSIM/loss/proxy improvements;
- validate on fast-test, then A19/A21-style larger offset blocks.

## Bottom Line

The project's shortfall is scientifically well explained: the data contains
two restoration regimes with opposite preferences, and the current single model
still shares too much of the computation before routing. A24 proves the route
mechanism is useful, but not yet strong enough. The next breakthrough requires
route-specific terminal or feature-level expert capacity, especially to protect
high-`input_std` A08-tail samples.
