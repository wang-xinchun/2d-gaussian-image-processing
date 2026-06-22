# A_4kd46 Result Analysis

- Status: completed, not promoted.
- Formal training: `logs/train_20260617_115700.log`, exit code `0`.
- Mechanism: offline low-route A08-positive hard-case cache plus `bounded_residual` learned gate, initialized from A37 best-val, training only `input_std_route_decoder_heads.gate_head`.

## What Worked

- The hard-case cache and sampler worked as intended:
  - cache samples: `128`
  - low-route samples: `99`
  - oracle-positive samples: `31`
  - hard-positive samples: `20`
  - sampler hard-case fraction: `0.65`
  - first 64 sampled batches hit hard cases `45` times.
- The preflight hit a hard-positive low-route sample immediately:
  - oracle target mean: `0.800000011920929`
  - prior mean: `1.0795604687974247e-10`
  - gate mean: `9.999999747378752e-06`
  - oracle gate loss: `9.210342407226562`
- The gate is no longer immobile. Final route-regime shows overall gate `0.5094007767736912` vs prior `0.42893063667252734`, and low-route gate `0.1611760899163129` vs prior `0.023600109885921854`.

## Results

| checkpoint | test PSNR | overall gap | high-route gap | low-route gap | A08-oracle gap | A15-oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best-val step10 | `17.88893291708372` | `+0.04199854126930214` | `-0.11143757779238696` | `+0.15774859600005006` | `-0.1975166637318077` | `+0.2460300121961735` |
| final step140 | `17.87790077389511` | `+0.030966398080690814` | `-0.10929431364232904` | `+0.1367771104331444` | `-0.1659231102631667` | `+0.1986870903736065` |

Route gate diagnostics:

| checkpoint | overall gate/prior | high gate/prior | low gate/prior |
| --- | ---: | ---: | ---: |
| best-val | `0.43450256807729604 / 0.42893063667252734` | `0.9665607602097267 / 0.966229241947795` | `0.03312533541598864 / 0.023600109885921854` |
| final | `0.5094007767736912 / 0.42893063667252734` | `0.9710009430730066 / 0.966229241947795` | `0.1611760899163129 / 0.023600109885921854` |

## Interpretation

A46 is a mechanism-positive but model-negative run. It solves A45's exposure problem: low-route A08-positive hard cases are now sampled early and often, and the bounded gate can move upward. However, the objective is not selective enough. By step140 the gate over-lifts the entire low-route regime, which improves final high/A08-side gaps but destroys the low/A15 surplus and aggregate PSNR.

The bottleneck is therefore not hard-case sampling alone and not gate mobility alone. The missing component is selective gate supervision: lift only nominal-low samples where A08 is genuinely better, while explicitly preserving ordinary low-route/A15-positive samples and high-route priors.
