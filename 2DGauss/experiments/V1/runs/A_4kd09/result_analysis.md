# Result Analysis: V1/T01/A_4kd09

## Summary

`A_4kd09` tested a control-to-guidance reliability gate on top of the current
fast-probe candidate `A_4kd08`. The reliability gate reduced late
over-correction and fixed the `A_4kd08` final-step collapse, but it was too
conservative for test-set fidelity.

This run is accepted as a completed bottleneck diagnostic, not promoted as the
new fast-probe candidate. `A_4kd08` remains the current fast-probe candidate.

## Training Outcome

- Formal preflight: passed on physical GPU3, `train_exit_code=0`.
- Formal training: completed 800 steps on physical GPU3, `train_exit_code=0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.265625GiB`.
- Best selection metric: `val_psnr`.
- Best validation step: `800`.
- Best/final validation metrics:
  - `val_psnr=17.894769024624548`
  - `val_loss=0.11876579158008099`

Validation trajectory:

- Step `100`: `val_psnr=14.223458527250752`
- Step `200`: `val_psnr=12.99173355447079`
- Step `300`: `val_psnr=15.261487347413631`
- Step `400`: `val_psnr=16.879464151128612`
- Step `500`: `val_psnr=15.428744574879856`
- Step `600`: `val_psnr=17.420986692896435`
- Step `700`: `val_psnr=16.12619242145213`
- Step `800`: `val_psnr=17.894769024624548`

Compared with `A_4kd08`, the main stability issue improved: `A_4kd08` peaked at
step `700` but fell to `14.466338503701529` at step `800`, while `A_4kd09`
ended at its best validation point.

## Reliability Gate Behavior

The reliability schedule behaved as intended. Early validation keeps reliability
at `1.0`, then ramps the gate after progress `0.40`:

- Step `300`: `val_control_guidance_reliability_mean=1.0`,
  `val_control_guidance_blend_mean=0.20332577750086783`
- Step `400`: `val_control_guidance_reliability_mean=0.951480570435524`,
  `val_control_guidance_blend_mean=0.1934926301240921`
- Step `600`: `val_control_guidance_reliability_mean=0.8280281221866608`,
  `val_control_guidance_blend_mean=0.1684147034585476`
- Step `800`: `val_control_guidance_reliability_mean=0.8269809144735336`,
  `val_control_guidance_blend_mean=0.16819747000932694`

This confirms that the new mechanism did not silently no-op: it materially
reduced effective control-to-guidance blending late in training.

## CPU Test Outcome

Evaluation used `best_val.pt` at checkpoint step `800`, CPU device, test split,
`100` samples, and `crop_size=512`.

- `psnr_mean=16.482337414450953`
- `loss_mean=0.14191464424133302`
- `auxiliary_loss_mean=0.0032319549773819744`
- `total_loss_mean=0.14514659851789474`

Task-specific CPU diagnostics:

- `ssim_mean=0.6357330930233002`
- `edge_magnitude_l1_mean=0.1690767514705658`
- `high_frequency_l1_mean=0.022298699477687478`
- `local_contrast_l1_mean=0.029207105338573455`
- `dark_channel_l1_mean=0.12478554897010326`
- `luminance_l1_mean=0.12720242224633693`
- `chroma_l1_mean=0.03518328143283725`
- `saturation_l1_mean=0.047565420120954514`

## Comparison With A_4kd08

`A_4kd09` improves the exact stability and dehazing-proxy symptoms targeted by
the run:

- Final validation PSNR improves over `A_4kd08` final by about `+3.4284 dB`
  (`17.894769024624548` vs `14.466338503701529`).
- Best-to-final collapse is removed because the final checkpoint is also the
  best validation checkpoint.
- Dark-channel L1 improves from `0.1491835382580757` to
  `0.12478554897010326`.
- Saturation L1 improves from `0.05636739023029804` to
  `0.047565420120954514`.

But the cost is too high:

- Best validation PSNR is lower than `A_4kd08` by about `0.2691 dB`
  (`17.894769024624548` vs `18.163864803283275`).
- CPU test PSNR drops by about `1.3068 dB`
  (`16.482337414450953` vs `17.789133503142946`).
- CPU test SSIM drops from `0.6947606813907623` to `0.6357330930233002`.
- CPU test loss worsens from `0.1228142374008894` to
  `0.14191464424133302`.
- Edge, high-frequency, local-contrast, luminance, and chroma diagnostics all
  worsen.

## Mechanism Lesson

The reliability gate validates the bottleneck diagnosis but not the operating
point. Over-trusting the shared control canvas is a real issue, because reducing
trust stabilizes the late trajectory and repairs dark-channel/saturation side
effects. However, the current reliability rule suppresses useful guidance too
aggressively and damages test-set fidelity.

Future variants should not reuse this setting unchanged. More promising narrow
next probes are:

- weaker reliability strength or higher schedule start, so the gate only acts
  after the control canvas is more mature;
- reliability applied only to the most suspect spatial or channel components;
- separated timing for protection losses and reliability gating;
- confidence predicted from the control branch rather than purely from local
  mismatch with the original attention map.

## Decision

Close `A_4kd09` as a completed reliability diagnostic. Keep the default-off code
path and diagnostics, but keep `A_4kd08` as the current fast-probe candidate.
The next productive step is a less conservative confidence/reliability variant
or a larger/current-split confirmation of `A_4kd08`, not promotion of
`A_4kd09`.
