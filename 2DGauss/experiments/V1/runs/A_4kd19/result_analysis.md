# A_4kd19 Result Analysis

## Summary

`A_4kd19` validates the `A_4kd18` A08/A15 input-std router outside the original
100-sample fast-probe validation/test block.

The evaluation uses the full current `4KDehaze_local_90_5_5` view with
`val_offset=100` and `test_offset=100`, then limits each split to `100` samples.
This makes the evaluated samples non-overlapping with the fast-probe block used
by `A_4kd18`.

## Metrics

Tail test, offset `100`, limit `100`, crop `512`:

| method | PSNR | SSIM | choose A15 |
| --- | ---: | ---: | ---: |
| A_4kd08 | 15.87689249278197 | 0.6992935019731522 | 0/100 |
| A_4kd15 | 15.885454203061169 | 0.7010528445243835 | 100/100 |
| Fixed A18 rule | 15.913737915082573 | 0.7024011361598969 | 40/100 |
| Tail-val retuned rule | 15.914886179979078 | 0.702488362789154 | 42/100 |
| Binary oracle | 15.928149885150932 | n/a | 51/100 |

The fixed `A_4kd18` rule improves over the stronger single checkpoint on this
tail test subset by `+0.028283712021403673 dB` PSNR and `+0.00134829163551331`
SSIM.

Retuning the threshold on the tail validation subset selects:

```text
use A_4kd15 if input_std <= 0.0743667222559452
otherwise use A_4kd08
```

This is close to the original A18 threshold `0.07026463374495506` and improves
tail test PSNR by only `+0.0011482648965041 dB` over the fixed rule.

## Diagnostics

- Tail validation baseline: A08 `18.02941065426511`, A15
  `18.07617320484514`, binary oracle `18.147767660295607`.
- Tail validation fixed A18 rule: `18.13404498432711`, `59/100` choose A15.
- Tail validation retuned rule: `18.136410774746352`, `64/100` choose A15.
- Tail test fixed A18 rule: `73/100` oracle-correct choices.
- Tail test retuned rule: `75/100` oracle-correct choices.

## Decision

`A_4kd19` strengthens the A18 dynamic-routing hypothesis: the input-std rule is
not only a fit to the first 100-sample fast-probe block. It still improves PSNR
and SSIM on a non-overlapping tail subset, and the retuned threshold remains
near the A18 threshold.

This remains a CPU two-checkpoint router diagnostic. It is not an official/full
benchmark and not a single-model result. The next useful step is to implement a
lightweight learned or model-internal router that can approximate this dynamic
selection without running two full checkpoints at inference time.
