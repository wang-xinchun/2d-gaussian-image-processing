# Run Manifest: V1/T01/A_4kd07

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd07`
- Full identifier: `V1/T01/A_4kd07`

## Current Status

Completed and closed as a neutral/negative fast-probe schedule diagnostic.

`A_4kd07` completed formal GPU3 preflight, 800-step low-footprint training,
CPU standard test evaluation, and CPU task-specific diagnostics. The run did
not beat `A_4kd05` on CPU test PSNR/SSIM and did not recover `A_4kd04`
validation strength, so it is not promoted as the new fast-probe candidate.

## Purpose

`A_4kd07` is the schedule follow-up after `A_4kd06`.

`A_4kd06` showed that a lighter detail-aware gate alone mostly reverts to
`A_4kd04` and does not reproduce `A_4kd05` test PSNR/SSIM. This run therefore
keeps the `A_4kd05` final detail-protection strength and protection losses, but
does not apply the strong gate from the first step.

The narrow hypothesis is that early strong detail suppression may interfere
while the restoration-demand canvas is still noisy. `A_4kd07` starts with
`A_4kd04`-style uniform blend, then linearly enables `A_4kd05`-style
detail-preserve suppression from training progress `0.25` to `0.75`.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` through `A_4kd06`.

## Model Delta

Relative to `A_4kd05`, this run adds only the blend-gate schedule:

- `gaussian_control_canvas_share_to_guidance_detail_preserve_schedule = "linear"`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_schedule_start = 0.25`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_schedule_end = 0.75`

With `max_steps=800`, the effective detail-preserve factor is:

- steps `1-200`: no detail suppression, uniform `0.25` blend
- steps `201-600`: linear ramp toward the `A_4kd05` detail gate
- steps `600-800`: full `A_4kd05` detail gate

All other model/loss settings match `A_4kd05`.

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

`A_4kd07` is not an official/full benchmark. It should be judged only against
the same fast-probe references:

- `A_4kd04`: validation peak reference
- `A_4kd05`: CPU test PSNR/SSIM and detail-protection reference
- `A_4kd06`: light-gate-alone negative/neutral diagnostic

Useful outcome: recover more of `A_4kd04` validation while keeping or exceeding
`A_4kd05` CPU test PSNR/SSIM and detail diagnostics. If it simply reproduces
`A_4kd05`, the schedule is not harmful but not sufficient. If it reverts to
`A_4kd04`, the late gate/protection interaction is too weak under this schedule.

## Result

- Formal preflight: passed on physical GPU3 with `train_exit_code=0`.
- Formal training: completed on physical GPU3 only with `train_exit_code=0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Best validation checkpoint: step `600`.
- Best validation: `val_psnr=17.571617667185173`,
  `val_loss=0.13438679296523331`.
- Final step `800`: `val_psnr=17.255021137009727`,
  `val_loss=0.14160969708114862`.
- CPU test eval of `best_val.pt`: `psnr_mean=17.059463021451332`,
  `loss_mean=0.13263863679021598`,
  `auxiliary_loss_mean=0.003231624343898147`,
  `total_loss_mean=0.1358702614158392`.
- CPU task diagnostics: `ssim_mean=0.6861055958271026`,
  `edge_magnitude_l1_mean=0.1648006558418274`,
  `high_frequency_l1_mean=0.022252602344378827`,
  `local_contrast_l1_mean=0.028178518442437052`,
  `dark_channel_l1_mean=0.13591586358845234`,
  `luminance_l1_mean=0.11381764061748982`,
  `chroma_l1_mean=0.03937912782654166`,
  `saturation_l1_mean=0.055852453988045454`.

## Interpretation

The linear schedule behaved mechanically as designed: validation diagnostics
were uniform during warmup (`0.25` high/low blend through step `200`) and reached
the same final gate profile as `A_4kd05` by step `600`
(`high_detail_mean=0.15471820175647735`,
`low_detail_mean=0.22401548877358438`,
`detail_corr=-0.8495775413513184`).

However, delaying the detail-preserve gate did not resolve the tradeoff. Best
validation is below both `A_4kd04` and `A_4kd05`, and CPU test PSNR is slightly
below `A_4kd05` (`17.059463021451332` vs `17.067359898802888`) despite a tiny
SSIM increase (`0.6861055958271026` vs `0.6860608831048012`). This is a useful
diagnostic, but not a candidate promotion.

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd07/configs/a_4kd07_fast_4kdehaze_scheduled_detail_blended_demand_control_probe.toml`
