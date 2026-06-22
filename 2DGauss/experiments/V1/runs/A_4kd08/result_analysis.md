# Result Analysis: V1/T01/A_4kd08

## Summary

`A_4kd08` tested a medium-strength detail-preserve gate while keeping the
`A_4kd05` protection losses. It is the first run in the `A_4kd04`-`A_4kd08`
family to improve both the fast-probe validation frontier and the CPU test
frontier.

It is accepted as the current fast-probe candidate. It is not an official/full
benchmark.

## Training Outcome

- Formal preflight: passed on physical GPU3, `train_exit_code=0`.
- Formal training: completed 800 steps on physical GPU3, `train_exit_code=0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Best selection metric: `val_psnr`.
- Best validation step: `700`.
- Best validation metrics:
  - `val_psnr=18.163864803283275`
  - `val_loss=0.12629025239497424`
- Final step `800` metrics:
  - `val_psnr=14.466338503701529`
  - `val_loss=0.20616793725639582`

Validation trajectory:

- Step `100`: `val_psnr=14.382204708090098`
- Step `200`: `val_psnr=14.290587181634887`
- Step `300`: `val_psnr=16.698328530404186`
- Step `400`: `val_psnr=16.632339128364002`
- Step `500`: `val_psnr=11.811753653207282`
- Step `600`: `val_psnr=14.630566782044534`
- Step `700`: `val_psnr=18.163864803283275`
- Step `800`: `val_psnr=14.466338503701529`

The volatility matters: the best checkpoint is strong, but training stability
is not solved.

## CPU Test Outcome

Evaluation used `best_val.pt` at checkpoint step `700`, CPU device, test split,
`100` samples, and `crop_size=512`.

- `psnr_mean=17.789133503142946`
- `loss_mean=0.1228142374008894`
- `auxiliary_loss_mean=0.003231953252106905`
- `total_loss_mean=0.12604619041085244`

Task-specific CPU diagnostics:

- `ssim_mean=0.6947606813907623`
- `edge_magnitude_l1_mean=0.16361528269946576`
- `high_frequency_l1_mean=0.02172296936623752`
- `local_contrast_l1_mean=0.026271848641335964`
- `dark_channel_l1_mean=0.1491835382580757`
- `luminance_l1_mean=0.1057051732763648`
- `chroma_l1_mean=0.030328415371477603`
- `saturation_l1_mean=0.05636739023029804`

## Comparisons

Against `A_4kd05`, the previous fast-probe generalization/detail-protection
candidate:

- Validation PSNR improves from `17.595543271291344` to
  `18.163864803283275`, about `+0.5683 dB`.
- CPU test PSNR improves from `17.067359898802888` to
  `17.789133503142946`, about `+0.7218 dB`.
- CPU test SSIM improves from `0.6860608831048012` to
  `0.6947606813907623`.
- CPU test loss improves from `0.13247113279998302` to
  `0.1228142374008894`.
- Edge, high-frequency, local-contrast, luminance, and chroma diagnostics
  improve.
- Dark-channel and saturation diagnostics worsen.

Against `A_4kd04`, the previous validation-peak reference:

- Validation PSNR improves from `17.6847619706854` to
  `18.163864803283275`, about `+0.4791 dB`.
- CPU test PSNR improves from `16.901385528055062` to
  `17.789133503142946`, about `+0.8877 dB`.
- CPU test SSIM improves from `0.6761810025572776` to
  `0.6947606813907623`.

## Mechanism Lesson

The bottleneck was not simply "detail gate needed" or "schedule needed".

- `A_4kd05`: strong gate plus protection losses improved CPU test but hurt
  validation and color/dehazing proxies.
- `A_4kd06`: light gate alone mostly reverted to `A_4kd04`.
- `A_4kd07`: schedule-only did not improve the candidate frontier.
- `A_4kd08`: medium gate plus protection losses improved both validation and
  CPU test metrics.

The current best interpretation is that gate-strength calibration was a real
short-train bottleneck. The remaining bottleneck is now stability and
generalization: the run is volatile, the dark-channel/saturation side effects
remain, and the result still needs a larger/current protocol before any
paper-level claim.

## Decision

Promote `A_4kd08` to the current fast-probe candidate. The next step should be
either:

- medium-length training or larger fast/current split validation for
  `A_4kd08`, or
- a confidence/reliability version that keeps the medium gate but stabilizes
  late training and reduces dark-channel/saturation cost.
