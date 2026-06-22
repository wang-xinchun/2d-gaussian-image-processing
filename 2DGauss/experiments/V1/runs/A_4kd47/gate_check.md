# A_4kd47 Gate Check

## Static

- `src/uhd_gaussian/config.py`, `src/uhd_gaussian/models/restorer.py`, `scripts/train_v1_t01.py`, and `scripts/build_input_std_route_oracle_hard_cases_v1_t01.py` py_compile passed.
- A47 TOML parses as `V1/T01/A_4kd47`.
- Launcher dry-run passed with A47 config/run_dir/session.

## Expected Mechanism

- New mode: `bounded_residual_global`.
- The global gate head should add a sample-level/context cue to the local gate head.
- Initial gate should equal the prior because the final gate projection is zero-initialized.
- Trainable parameter prefix remains `input_std_route_decoder_heads.gate_head`.

## Preflight

- CPU smoke passed:
  - A37 init missing keys are exactly the 6 new global gate-head parameters.
  - unexpected keys: `[]`
  - trainable gate-head parameters: `1073`
  - initial `max_abs(gate-prior)` on random CPU input: about `1.0e-5`
- GPU preflight passed:
  - log: `logs/train_20260617_123305.log`
  - exit code: `0`
  - device: physical GPU0 through `CUDA_VISIBLE_DEVICES=0`
  - preflight trainable/frozen policy accepted, with total params `169479`
  - sampled low-route oracle-negative case:
    - `input_std_route_oracle_gate_target_mean=0.0`
    - `input_std_route_gate_mean=0.00024448276963084936`
    - `input_std_route_prior_mean=0.00024448276963084936`
  - The 4-sample preflight subset overlaps one cached hard case; full formal sampler is expected to see all 20 A46 cached hard cases.

## Formal

- Formal training completed:
  - log: `logs/train_20260617_123508.log`
  - exit code: `0`
  - GPU: physical GPU0
  - memory fraction: `0.18`
  - best step: `10`
  - best validation PSNR: `18.563402798042727`
  - final step: `140`
  - final validation PSNR: `18.563393906314797`
- Standard test:
  - best-val PSNR: `17.88923343442933`
  - final PSNR: `17.889420757227708`
- A18 alignment:
  - best-val gaps: overall `+0.04229905861490691`, high-route `-0.11264421059134787`, low-route `+0.1591860862617307`, A08-oracle `-0.20194105744695595`, A15-oracle `+0.25035545377871604`
  - final gaps: overall `+0.04248638141328485`, high-route `-0.11217793301246365`, low-route `+0.1591629694888495`, A08-oracle `-0.2010047959040689`, A15-oracle `+0.24990479172066027`
- Route-regime:
  - best overall gate/prior: `0.42309092255555697 / 0.42893063667252734`
  - final overall gate/prior: `0.42565524119421755 / 0.42893063667252734`
  - final high gate/prior: `0.9588475698648498 / 0.966229241947795`
  - final low gate/prior: `0.023422431846196752 / 0.023600109885921854`

Conclusion: A47 prevents A46's low-route over-lift, but it becomes conservative and suppresses the high route. It protects aggregate/low behavior but worsens the target A08/high-side gaps, so it is not promotable.
