# A_4kd18 Result Analysis

## Summary

`A_4kd18` is the first fast-probe result in this sequence to exceed the strict
`A_4kd08` test PSNR reference after using `A_4kd08` as one branch of a dynamic
selector.

It is not a single-checkpoint model. It is a validation-tuned inference router
between `A_4kd08` and `A_4kd15`, using only input standard deviation as the
selection feature.

## Metrics

| method | test PSNR | SSIM | loss | total loss |
| --- | ---: | ---: | ---: | ---: |
| A_4kd08 | 17.789133503142946 | 0.6947606813907623 | 0.1228142374008894 | 0.12604619041085244 |
| A_4kd15 | 17.781017099305146 | 0.6972798517346382 | 0.12060526199638844 | 0.12383682064712048 |
| A_4kd18 router | 17.84693437641316 | 0.6992742219567298 | 0.12033024244010448 | 0.12356196045875549 |

Relative to `A_4kd08`, the router gains `+0.057800873270213556 dB` PSNR,
`+0.0045135405659675` SSIM, and lowers loss by `0.00248399496078492`.

Relative to `A_4kd15`, the router gains `+0.06591727710801139 dB` PSNR,
`+0.0019943702220916` SSIM, and lowers loss by `0.00027501955628396`.

## Diagnostics

The fixed global A08/A15 output blend was not enough to explain the gain:
the best fixed alpha on test was `0.20`, with `17.789821380590535` PSNR,
only `+0.000687877447589 dB` over `A_4kd08`.

The binary oracle between A08 and A15 on test was much higher:
`17.883555795086536` PSNR with `54/100` samples choosing A15. This shows
that the useful signal is sample-adaptive routing, not constant interpolation.

The validation-tuned rule selected:

```text
use A_4kd15 if input_std <= 0.07026463374495506
otherwise use A_4kd08
```

On test it chose A15 for `57/100` samples and reached `17.84693437581442`
PSNR in the router-search JSON. Aggregating the already-computed A08/A15
task metrics for the same choices gives `17.84693437641316` PSNR.

The selector is correct against the binary oracle on `71/100` test samples.
It preserves the watched high-A08-tail samples by selecting A08 for:
`94_94000077`, `94_94000080`, `94_94000073`, `94_94000084`, `94_9400009`,
`94_94000053`, and `94_94000082`.

## Decision

`A_4kd18` should supersede `A_4kd08` only within the fast-probe inference-router
scope. It should not be reported as a single-model or official/full benchmark
breakthrough.

The useful next line is to convert this evidence into a model-internal or
lightweight learned router that avoids two full checkpoint forwards, then
validate on a larger/current split. Continuing to increase teacher-consistency
weight is lower value because `A_4kd17` already showed that teacher loss
suppresses A15's hard-sample gains while failing to restore the high-A08 tail.
