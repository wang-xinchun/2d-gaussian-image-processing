# V1/T01/A_4kd46 Manifest

- Run: `A_4kd46`
- Created: `2026-06-17 11:35 CST`
- Baseline: `A_4kd45` bounded residual gate, initialized from `A_4kd37` best-val.
- Goal: make low-route A08-positive oracle samples appear early and frequently enough for the bounded learned gate to move upward before best-checkpoint selection.

## Mechanism

- Added default-off hard-case injection to `InputStdRouteBalancedBatchSampler`.
- Added offline cache builder `scripts/build_input_std_route_oracle_hard_cases_v1_t01.py`.
- A46 first deployment uses `input_std_route_oracle_hard_cases_train_limit128.json`:
  - samples: `128`
  - low-route samples: `99`
  - oracle-positive samples: `31`
  - hard-positive samples: `20`
  - hard-positive fraction: `0.15625`
- Training sampler:
  - hard-case fraction: `0.65`
  - hard-case route label: `0`
  - crop attempts: `24`
  - residual gate mode: `bounded_residual`
  - trainable parameters: `input_std_route_decoder_heads.gate_head`

## Promotion Criteria

- Primary: improve A18 high-route and A08-oracle gaps over A45/A37.
- Secondary: avoid losing A37 aggregate behavior and low-route/A15 advantage.
- Diagnostic success: early training metrics should show nonzero low-route oracle-positive samples, `oracle_gate_target_mean` near `0.8`, and measurable gate movement from prior.
