# A_4kd36 Gate Check

时间: 2026-06-14 22:08:00 CST

## Static

- PASS: `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python -m py_compile /home/ldj/wxc/2DGauss/scripts/create_dual_backbone_router_checkpoint_v1_t01.py /home/ldj/wxc/2DGauss/scripts/evaluate_dual_backbone_router_v1_t01.py`
- PASS: A08 config/checkpoint exists
- PASS: A15 config/checkpoint exists
- PASS: A18 records path exists
- PASS: A35 dataset config path exists for test split reuse

## Pack

- PASS: created `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/checkpoints/a08_a15_single_checkpoint_dual_backbone_input_std_router.pt`
- PASS: manifest `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/dual_backbone_checkpoint_manifest.json`
- PASS: high/low state key sets match, `state_key_count=236`

## Smoke

- PASS: GPU3 smoke, 2 samples
- Smoke metric path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu_smoke_limit2.json`
- Smoke log path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu_smoke_limit2.stdout.log`
- Smoke result: `psnr_mean=17.69232222528726`, router gap `+0.001006125572006411 dB`

## Full Test

- PASS: GPU3 full test, 100 samples, exit code `0`
- Metric path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu.json`
- Log path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.stdout.log`
- Exit-code path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.exit_code`

## Acceptance

- PASS: exceeds A24 single checkpoint: `17.8470899392002 > 17.790202882742463`
- PASS: exceeds hopeful target: `17.8470899392002 >= 17.82`
- PASS: reaches A18 router target: `17.8470899392002 > 17.84693437581442`
- PASS: high route is fixed for the upper-bound artifact: high-route gap `-0.00019952460737376242 dB`
- PASS: low route is not collapsed for A18-equivalent routing: low-route gap `+0.00042343678412253124 dB`
- PASS: `SSIM=0.6992770892381668 >= 0.699`
- PASS: `loss=0.11994637824594974 <= 0.1205`

## Boundary

A36 is accepted as a single checkpoint milestone and an upper-bound mechanism proof, not as a lightweight single-backbone replacement. It uses two complete internal backbones and therefore shifts the next bottleneck to compression/parameter sharing.
