# V1/T01/A_4kd19

## Purpose

`A_4kd19` tests whether the `A_4kd18` A08/A15 input-std router generalizes
outside the 100-sample fast-probe validation/test subset.

It uses the full current-server `4KDehaze_local_90_5_5` view, but starts both
validation and test evaluation at offset `100`, so the first fast-probe block
used by `A_4kd18` is excluded.

## Checkpoints

- A branch: `A_4kd08` best-val step `700`.
- B branch: `A_4kd15` best/final step `300`.

## Planned Evaluation

- CPU only, no GPU.
- Crop size `512`.
- Tail validation: offset `100`, limit `100`.
- Tail test: offset `100`, limit `100`.
- Report both:
  - fixed `A_4kd18` rule: `input_std <= 0.07026463374495506 -> A_4kd15`;
  - newly selected tail-val threshold and its tail-test result.

## Result

- CPU evaluation completed in `1110.576572418213` seconds.
- Fixed A18 rule tail-test PSNR: `15.913737915082573`.
- Fixed A18 rule tail-test SSIM: `0.7024011361598969`.
- Tail-val retuned rule: `input_std <= 0.0743667222559452 -> A_4kd15`.
- Retuned rule tail-test PSNR: `15.914886179979078`.
- Retuned rule tail-test SSIM: `0.702488362789154`.
- A08/A15 single-checkpoint baselines on the same tail test subset:
  `15.87689249278197` / `15.885454203061169` PSNR.
- Binary oracle on the same tail test subset: `15.928149885150932` PSNR.

## Key Files

- Config: `experiments/V1/runs/A_4kd19/configs/a_4kd19_a08_a15_input_std_router_tail_generalization.toml`
- CPU router metrics: `experiments/V1/runs/A_4kd19/metrics/a08_a15_router_tail_offset100_cpu.json`
- CPU router stdout: `experiments/V1/runs/A_4kd19/logs/a08_a15_router_tail_offset100_cpu.stdout.log`
- Analysis: `experiments/V1/runs/A_4kd19/result_analysis.md`

## Boundary

This is a router generalization diagnostic, not a training run, not a
single-model result, and not an official/full benchmark.
