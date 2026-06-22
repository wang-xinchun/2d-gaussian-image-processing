# Run Manifest: V1/T01/A_4kd09

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd09`
- Full identifier: `V1/T01/A_4kd09`

## Current Status

Completed. Accepted as a bottleneck diagnostic, not promoted as the new
fast-probe candidate. `A_4kd08` remains the current fast-probe candidate.

## Purpose

`A_4kd09` follows the current fast-probe candidate `A_4kd08`.

`A_4kd08` showed that a medium detail gate plus protection losses can improve
both validation and CPU test metrics, but it also exposed the current bottleneck:
validation is volatile, the final checkpoint falls back after the best step, and
dark-channel/saturation diagnostics worsen. `A_4kd09` tests whether a lightweight
control reliability gate can reduce over-trusting the shared control canvas when
it disagrees with the original Gaussian attention.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` through `A_4kd08`.

## Model Delta

Relative to `A_4kd08`, this run keeps the medium detail gate and protection
losses unchanged, and adds only the control-to-guidance reliability gate:

- `gaussian_control_canvas_share_to_guidance_reliability_gate = true`
- `gaussian_control_canvas_share_to_guidance_reliability_strength = 0.35`
- `gaussian_control_canvas_share_to_guidance_reliability_floor = 0.70`
- `gaussian_control_canvas_share_to_guidance_reliability_kernel_size = 7`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule = "linear"`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule_start = 0.40`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule_end = 0.75`

The reliability gate is computed from local mismatch between the predicted
control canvas and the original Gaussian attention. It is intended to keep the
`A_4kd08` early learning behavior while reducing late overcorrection after the
control canvas becomes more influential.

## Runtime Envelope

- Planned wall-clock cap: `58` minutes
- Planned max steps: `800`
- Planned batch size: `2`
- Gradient accumulation: `4`
- Training crop: `512`
- Validation crop: `512`
- Validation subset: `100` samples
- DataLoader workers: `1`
- Best checkpoint selection: `val_psnr`
- CUDA memory fraction: `0.30`
- Allowed physical GPUs: `0` or `3` only
- Default preferred physical GPU: `3`

## Acceptance Lens

This is a fast-probe mechanism run, not an official/full benchmark. It should be
judged against `A_4kd08` first:

- Useful outcome: preserve most of `A_4kd08` validation/test gain while reducing
  late validation collapse or improving dark-channel/saturation diagnostics.
- Failure mode: reduced PSNR/SSIM without stability or proxy repair, which would
  mean the reliability gate is too conservative or mismatched to this task.

## Final Result

- Formal preflight: passed on physical GPU3.
- Formal training: completed 800 steps on physical GPU3.
- Training exit code: `0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.265625GiB`.
- Best validation checkpoint: step `800`.
- Best/final validation:
  - `val_psnr=17.894769024624548`
  - `val_loss=0.11876579158008099`
- CPU standard test eval:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/metrics/test_eval_best_val_cpu.json`
  - `psnr_mean=16.482337414450953`
  - `loss_mean=0.14191464424133302`
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/metrics/test_task_metrics_best_val_cpu.json`
  - `ssim_mean=0.6357330930233002`
  - `dark_channel_l1_mean=0.12478554897010326`
  - `saturation_l1_mean=0.047565420120954514`
- Result analysis:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/result_analysis.md`

## Interpretation

`A_4kd09` fixed the targeted final-step collapse: unlike `A_4kd08`, its final
checkpoint is also the best validation checkpoint. It also improves
dark-channel and saturation diagnostics relative to `A_4kd08`.

The cost is too large for promotion. CPU test PSNR drops from `A_4kd08`'s
`17.789133503142946` to `16.482337414450953`, and SSIM drops from
`0.6947606813907623` to `0.6357330930233002`. This means the current reliability
gate setting is over-conservative: useful control guidance is being suppressed
along with harmful late overcorrection.
