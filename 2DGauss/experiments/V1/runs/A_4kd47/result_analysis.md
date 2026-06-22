# A_4kd47 Result Analysis

- Status: completed, not promoted.
- Formal training: `logs/train_20260617_123508.log`, exit code `0`.
- Mechanism: `bounded_residual_global` gate head with local decoder features plus global pooled route context, initialized from A37 best-val, reusing the A46 hard-case cache with lower hard-case fraction `0.45`.

## What Worked

- The new global gate mode loaded cleanly from A37. Missing keys were exactly the six new global gate-head parameters; unexpected keys were `[]`.
- Trainable parameter count rose from A46's `545` to `1073`, still limited to `input_std_route_decoder_heads.gate_head`.
- The initial gate matched the fixed prior within clamp tolerance, preserving default behavior at initialization.
- The final low-route gate stayed protected: `0.023422431846196752` vs prior `0.023600109885921854`, avoiding A46-final's low-route mean `0.1611760899163129`.

## Results

| checkpoint | test PSNR | overall gap | high-route gap | low-route gap | A08-oracle gap | A15-oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best-val step10 | `17.88923343442933` | `+0.04229905861490691` | `-0.11264421059134787` | `+0.1591860862617307` | `-0.20194105744695595` | `+0.25035545377871604` |
| final step140 | `17.889420757227708` | `+0.04248638141328485` | `-0.11217793301246365` | `+0.1591629694888495` | `-0.2010047959040689` | `+0.24990479172066027` |

Route gate diagnostics:

| checkpoint | overall gate/prior | high gate/prior | low gate/prior |
| --- | ---: | ---: | ---: |
| best-val | `0.42309092255555697 / 0.42893063667252734` | `0.9530857829160468 / 0.966229241947795` | `0.023270238423959342 / 0.023600109885921854` |
| final | `0.42565524119421755 / 0.42893063667252734` | `0.9588475698648498 / 0.966229241947795` | `0.023422431846196752 / 0.023600109885921854` |

## Interpretation

A47 is the complement of A46. It fixes the catastrophic low-route over-lift, but the learned global context makes the gate conservative and slightly depresses the high route: final high gate `0.9588475698648498` vs prior `0.966229241947795`. That protects aggregate PSNR near A37/A45 levels but misses the intended A08/high-tail improvement.

Together, A46 and A47 localize the next bottleneck. More hard-case exposure creates an upward gate bias that is too broad; adding global context suppresses that bias but also suppresses the target cases. The next run should make the target asymmetric and contrastive: positive-up loss on low-route A08-positive hard cases, plus hard-negative or prior-preservation constraints on low-route A15-positive cases and high-route priors.
