# A_4kd45 Gate Check

时间: 2026-06-17 CST

## Required Checks

- Static compile: passed.
- TOML/path check: passed; sampler cache is a runtime artifact.
- Launcher `bash -n`: passed.
- Dry-run launch: passed on GPU3 with low memory gate for validation.
- CPU smoke: passed.
- GPU preflight: passed on GPU3, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/logs/train_20260617_013337.log`, exit code `0`.
- Formal launch: completed on GPU3, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/logs/train_20260617_013440.log`, exit code `0`; best step `70`, validation PSNR `18.563397419824184`.

## Mechanism Checks

- `bounded_residual` gate initializes to the clamped fixed prior.
- Low-prior synthetic sample with oracle target `0.8` produces non-zero gate-head gradient (`gate_grad_l1=8204.0751953125`).
- A37 init is non-strict only because four new gate-head keys are absent from the checkpoint.
- A08/A15 teachers load strict with missing/unexpected `0/0`.
- Trainable parameters are restricted to `input_std_route_decoder_heads.gate_head` (`545` params).
- GPU preflight trainable params remain `545`; A08/A15 teachers load strict. Validation route gate mean `0.4098978623985204` versus prior mean `0.41056968903159036`, confirming the bounded gate can move during the first update.
