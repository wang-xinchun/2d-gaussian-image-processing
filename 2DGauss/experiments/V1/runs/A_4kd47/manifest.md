# V1/T01/A_4kd47 Manifest

- Run: `A_4kd47`
- Created: `2026-06-17 12:45 CST`
- Baseline: `A_4kd46` hard-case sampler and bounded residual gate, initialized from `A_4kd37` best-val.
- Goal: keep A46's ability to rescue nominal-low/A08-positive boundary samples, but avoid the final-checkpoint failure where low-route gate is lifted globally.

## Mechanism

- Adds default-off `input_std_route_learned_gate_mode="bounded_residual_global"`.
- The new gate head combines local decoder features with global pooled route context, then uses the same bounded residual mapping as A45/A46.
- The last gate projection is zero-initialized, so the initial gate remains the fixed `input_std` prior.
- A47 reuses the A46 `limit128` low-route A08-positive hard-case cache:
  - samples: `128`
  - low-route samples: `99`
  - oracle-positive samples: `31`
  - hard-positive samples: `20`
- Training sampler:
  - hard-case fraction: `0.45`
  - hard-case route label: `0`
  - crop attempts: `24`
  - trainable parameters: `input_std_route_decoder_heads.gate_head`

## Promotion Criteria

- Primary: improve A18 high-route and A08-oracle gaps without giving back the A37/A45 low-route surplus.
- Secondary: standard test PSNR should not drop below A45 by more than tiny noise.
- Diagnostic success: low-route gate should rise on hard-positive boundary cases but not approach A46-final's low-route mean around `0.161`.
