# A_4kd34 Gate Check

更新时间: 2026-06-14 20:44:54 CST

## Status

All required gates passed. A34 was deployed on physical GPU3 under the GPU0/GPU3 memory-gated policy.

## Completed Gates

- `logs/messages.md` and `logs/messages.jsonl` include the user original message: `请你部署`.
- `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py` supports `training.init_checkpoint_initialize_input_std_route_decoder_heads_from_base=false`.
- `py_compile` passed for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, and `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`.
- TOML parse and checkpoint path checks passed.
- Launcher `bash -n` passed.
- Smoke confirmed A33 route decoder-head weights are preserved after init.
- Smoke confirmed low-route delta stays zero at init and low-route parameters are frozen.
- GPU gate only allowed physical GPU0/GPU3; GPU3 dry-run passed with free memory `13041 MiB`, util `100`, memory fraction `0.35`.
- GPU3 preflight passed, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/logs/train_20260614_202945.log`, exit code `0`.
- GPU3 formal training passed, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/logs/train_20260614_203136.log`, exit code `0`.
- Standard fast-test, task metrics, A18 alignment, and route-regime diagnostics all completed.

## Result

A34 is deployment-complete but not promoted. It preserves low route, clears `17.82`, but is slightly below A33.
