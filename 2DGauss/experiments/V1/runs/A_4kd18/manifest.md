# V1/T01/A_4kd18

## Purpose

`A_4kd18` is an inference-router diagnostic after `A_4kd17` showed that
teacher-consistency fine-tuning does not recover the A08 high-PSNR tail enough
and suppresses A15 hard-sample gains.

The experiment tests whether the A08/A15 tradeoff can be recovered by dynamic
selection rather than another teacher-loss sweep. It uses `A_4kd08` best-val
step700 as checkpoint A and `A_4kd15` best/final step300 as checkpoint B.

## Router

- Rule selected on validation split only:
  `use A_4kd15 when input_std <= 0.07026463374495506`, otherwise use `A_4kd08`.
- Test split is not used to tune the threshold.
- This is a two-checkpoint inference wrapper, not a single-model training run.

## Result

- Test PSNR: `17.84693437641316`.
- Test SSIM: `0.6992742219567298`.
- Test loss: `0.12033024244010448`.
- Test total loss: `0.12356196045875549`.
- Test choices: `57/100` samples choose `A_4kd15`, `43/100` choose `A_4kd08`.
- Watched high-A08-tail samples all choose `A_4kd08`, repairing the A15 tail
  failure by selection rather than by changing A15.

## Key Files

- Config: `experiments/V1/runs/A_4kd18/configs/a_4kd18_a08_a15_input_std_router.toml`
- Blend grid: `experiments/V1/runs/A_4kd18/metrics/a08_a15_blend_grid_test_cpu.json`
- Val-to-test router search: `experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`
- Aggregated task metrics: `experiments/V1/runs/A_4kd18/metrics/a08_a15_input_std_router_test_task_metrics_cpu.json`
