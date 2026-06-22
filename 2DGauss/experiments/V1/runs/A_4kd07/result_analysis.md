# Result Analysis: V1/T01/A_4kd07

## Summary

`A_4kd07` tested a linear schedule for the `A_4kd05` detail-preserve
control-to-guidance gate. The run starts with uniform `A_4kd04`-style blend,
ramps the detail-aware suppression from training progress `0.25` to `0.75`,
and reaches the full `A_4kd05` gate by step `600`.

The schedule behaved as intended, but it did not improve the candidate set.
`A_4kd04` remains the validation-peak reference and `A_4kd05` remains the
fast-probe generalization/detail-protection candidate.

## Training Outcome

- Formal preflight: passed on physical GPU3, `train_exit_code=0`.
- Formal training: completed 800 steps on physical GPU3, `train_exit_code=0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Best selection metric: `val_psnr`.
- Best validation step: `600`.
- Best validation metrics:
  - `val_psnr=17.571617667185173`
  - `val_loss=0.13438679296523331`
- Final step `800` metrics:
  - `val_psnr=17.255021137009727`
  - `val_loss=0.14160969708114862`

## CPU Test Outcome

Evaluation used `best_val.pt` at checkpoint step `600`, CPU device, test split,
`100` samples, and `crop_size=512`.

- `psnr_mean=17.059463021451332`
- `loss_mean=0.13263863679021598`
- `auxiliary_loss_mean=0.003231624343898147`
- `total_loss_mean=0.1358702614158392`

Task-specific CPU diagnostics:

- `ssim_mean=0.6861055958271026`
- `edge_magnitude_l1_mean=0.1648006558418274`
- `high_frequency_l1_mean=0.022252602344378827`
- `local_contrast_l1_mean=0.028178518442437052`
- `dark_channel_l1_mean=0.13591586358845234`
- `luminance_l1_mean=0.11381764061748982`
- `chroma_l1_mean=0.03937912782654166`
- `saturation_l1_mean=0.055852453988045454`

## Comparisons

Against `A_4kd04`:

- Validation is lower: `17.571617667185173` vs `17.6847619706854`.
- CPU test PSNR is higher: `17.059463021451332` vs `16.901385528055062`.
- CPU SSIM is higher: `0.6861055958271026` vs `0.6761810025572776`.
- Detail proxies improve relative to `A_4kd04`, matching the broad behavior of
  the `A_4kd05` family.

Against `A_4kd05`:

- Validation is lower: `17.571617667185173` vs `17.595543271291344`.
- CPU test PSNR is slightly lower: `17.059463021451332` vs
  `17.067359898802888`.
- CPU SSIM is fractionally higher: `0.6861055958271026` vs
  `0.6860608831048012`.
- Edge/high-frequency/local-contrast proxies are fractionally better, but
  dark-channel, luminance, chroma, and saturation proxies are worse.

Against `A_4kd06`:

- `A_4kd07` improves CPU test PSNR and SSIM, confirming that the stronger
  `A_4kd05`-style gate/loss family matters more than a light gate alone.
- `A_4kd07` does not improve over the already accepted `A_4kd05` candidate.

## Gate Behavior

The schedule was mechanically correct:

- Step `100`: high-detail and low-detail blend both `0.25`.
- Step `200`: high-detail and low-detail blend both `0.25`.
- Step `300`: high-detail `0.22617954865098`, low-detail
  `0.24350387260317802`.
- Step `500`: high-detail `0.17853864774107933`, low-detail
  `0.23051161378622054`.
- Step `600`: full gate reached, high-detail `0.15471820175647735`,
  low-detail `0.22401548877358438`, correlation `-0.8495775413513184`.

This means the negative result is not due to the schedule failing to engage.
It indicates that simply delaying the strong detail-preserve gate is not enough
to recover validation while preserving `A_4kd05` test behavior.

## Decision

`A_4kd07` is a completed schedule diagnostic, not a promoted candidate.
Do not continue the same schedule family unchanged. The next useful experiment
should change the mechanism more directly, for example by separating protection
loss timing from gate timing, reducing color/dehaze-side penalties, or moving
from schedule-only tuning to an explicit confidence/reliability estimate for
the control canvas.
