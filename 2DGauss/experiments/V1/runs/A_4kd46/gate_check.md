# A_4kd46 Gate Check

## Static

- `scripts/train_v1_t01.py` compiles with the hard-case sampler extension.
- `scripts/build_input_std_route_oracle_hard_cases_v1_t01.py` compiles and has executable bit set.
- A46 config parses as TOML and resolves identity `V1/T01/A_4kd46`.

## Cache

- CPU smoke cache (`limit=2`, `crop_size=128`) completed successfully.
- GPU cache fallback (`limit=128`, `crop_size=512`) completed successfully:
  - hard-positive samples: `20`
  - oracle-positive samples: `31`
  - low-route samples: `99`
  - hard-positive fraction: `0.15625`
- Full train cache generation was started but stopped because GPU0 contention made it too slow; A46 first deployment intentionally uses the completed `limit128` cache.

## Sampler Smoke

- Full train sampler sees:
  - hard-case samples: `20`
  - high samples: `429`
  - low samples: `83`
  - hard-case fraction config: `0.65`
- First 64 sampled batches:
  - hard-case hits: `45`
  - hard-case hit fraction: `0.703125`
  - low labels: `59`
  - high labels: `5`

## GPU Preflight

- Log: `logs/train_20260617_115420.log`
- Exit code: `0`
- Device: physical GPU0 through `CUDA_VISIBLE_DEVICES=0`.
- Trainable parameters: `545`, limited to `input_std_route_decoder_heads.gate_head`.
- Preflight sampled a hard-positive low-route case:
  - `input_std_route_oracle_gate_positive_fraction=1.0`
  - `input_std_route_oracle_gate_target_mean=0.800000011920929`
  - `loss_input_std_route_oracle_gate=9.210342407226562`
  - `input_std_route_prior_mean=1.0795604687974247e-10`
  - `input_std_route_gate_mean=9.999999747378752e-06`

## Formal

- Formal training completed:
  - log: `logs/train_20260617_115700.log`
  - exit code: `0`
  - GPU: physical GPU0
  - memory fraction: `0.25`
  - best step: `10`
  - best validation PSNR: `18.55976332681181`
  - final step: `140`
- Standard test:
  - best-val PSNR: `17.88893291708372`
  - final PSNR: `17.87790077389511`
- A18 alignment:
  - best-val gaps: overall `+0.04199854126930214`, high-route `-0.11143757779238696`, low-route `+0.15774859600005006`, A08-oracle `-0.1975166637318077`, A15-oracle `+0.2460300121961735`
  - final gaps: overall `+0.030966398080690814`, high-route `-0.10929431364232904`, low-route `+0.1367771104331444`, A08-oracle `-0.1659231102631667`, A15-oracle `+0.1986870903736065`
- Route-regime:
  - best low-route gate/prior: `0.03312533541598864 / 0.023600109885921854`
  - final low-route gate/prior: `0.1611760899163129 / 0.023600109885921854`

Conclusion: A46 proves the hard-case sampler and bounded gate can move the low-prior A08-positive cases, but the final checkpoint over-lifts the low route globally. It improves the final A08/high gaps while collapsing low/A15 and aggregate PSNR, so it is not promotable.
