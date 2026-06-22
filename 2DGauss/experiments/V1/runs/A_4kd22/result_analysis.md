# A_4kd22 Result Analysis

## Summary

`A_4kd22` tests whether the A18/A19/A21 router milestone can be converted into
a lightweight learned saved-record router. The answer is mostly no: richer
logistic routers have signal, but they do not beat the fixed A18 `input_std`
rule under leave-one-source-out validation.

This is a useful negative/steering result. It tells us the next single-model
step should preserve the fixed input-std prior or use strongly regularized
input-std-like gating, rather than freely selecting richer features from one
validation block.

## Protocol

Sources:

- A18 fast-probe val/test records
- A19 offset100 tail val/test records
- A21 offset200 val/test records

For each held-out source:

1. Train a logistic router on validation records from the other sources.
2. Tune the probability threshold on those training validation records.
3. Evaluate on the held-out source's test records.
4. Compare against the fixed A18 rule and A08/A15 baselines.

## Aggregate Leave-One-Source-Out Results

| Method | PSNR | Oracle Agreement | Choose A15 |
| --- | ---: | ---: | ---: |
| Fixed A18 input_std | `15.761350330784339` | `0.77` | `204/400` |
| Logistic input_mean_std | `15.758408523282599` | `0.7575` | `219/400` |
| Logistic input_std | `15.753757145011031` | `0.7525` | `223/400` |
| Logistic prediction_stats | `15.753746408415735` | `0.735` | `224/400` |
| Logistic control_stats | `15.752713770191024` | `0.725` | `228/400` |
| Logistic all_common | `15.748658098940552` | `0.7075` | `237/400` |
| Logistic input_residual | `15.745576353235847` | `0.695` | `232/400` |

The fixed rule wins the aggregate leave-one-source-out leaderboard.

## Per-Holdout Details

### A18 Held Out

- A08: `17.789133503142946`
- A15: `17.781017124695964`
- Fixed rule: `17.84693437581442`
- Oracle: `17.883555795086536`
- Best logistic: `control_stats`, `17.85025635312045`

A18 is the one held-out block where a richer logistic rule can slightly exceed
the fixed rule, but this does not generalize to the larger A21 block.

### A19 Held Out

- A08: `15.87689249278197`
- A15: `15.885454203061169`
- Fixed rule: `15.913737915082573`
- Oracle: `15.928149885150932`
- Best logistic: `input_std`, `15.915906320381305`

A19 still favors an input-std-like rule, not a broad richer-feature model.

### A21 Held Out

- A08: `14.61942131606054`
- A15: `14.570231632564433`
- Fixed rule: `14.64236451612018`
- Oracle: `14.662330739170043`
- Best logistic: fixed rule; best learned `input_std` logistic is
  `14.63672184440178`

A21 is decisive for stability: all learned feature sets fall below fixed
input_std on the larger offset200 block.

## Interpretation

A22 does not reduce the importance of the A18/A19/A21 milestone. It sharpens it.
The stable finding is not "learn any router from all available features"; it is
"input_std captures a robust part of the A08/A15 complementarity, and richer
features overfit without stronger constraints."

Next recommended direction:

1. Convert fixed input-std routing into a single-model mechanism, likely through
   router-output distillation or a constrained gating head.
2. Keep the A18 threshold as a prior or initialization.
3. Validate cross-block before using any richer feature.

Boundary: A22 is still offline and record-level. It is not a trainable
single-model checkpoint and not an official/full benchmark.
