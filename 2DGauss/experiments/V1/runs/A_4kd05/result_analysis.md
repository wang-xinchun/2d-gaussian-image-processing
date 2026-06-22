# Result Analysis: V1/T01/A_4kd05

## Status

`A_4kd05` completed on 2026-06-10. It is accepted as a useful fast-probe
generalization/detail-protection candidate, not as an official/full split
benchmark.

The result is not a simple validation win over `A_4kd04`: validation peak is
slightly lower. The useful signal is that CPU test PSNR and SSIM improve while
some of the `A_4kd04` detail-regression proxies move back in the right
direction.

## Run Safety

- Allowed GPUs: physical GPU0 or GPU3 only.
- Actual training GPU: physical GPU3.
- GPU1/GPU2 use by this project: none.
- External process handling: no external process was killed or displaced.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory recorded in training metrics: about
  `11.24609375GiB`.
- Cleanup: no `A_4kd05`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or A_4kd05 tmux
  session remained after completion.

## Model Delta

Relative to `A_4kd04`, `A_4kd05` keeps blended shared demand-control and adds
detail-aware protection around the control-to-guidance blend:

- `gaussian_control_canvas_share_to_guidance_detail_preserve = true`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_strength = 0.65`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_floor = 0.35`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_kernel_size = 5`

It also adds small edge/high-frequency/local-contrast/chroma protection losses
and selects `best_val.pt` by `val_psnr`.

The protection gate behaved as intended. Validation diagnostics consistently
showed lower effective blend in high-detail regions than low-detail regions:

- `val_control_guidance_blend_high_detail_mean`: `0.15471820175647735`
- `val_control_guidance_blend_low_detail_mean`: `0.22401548877358438`
- `val_control_guidance_blend_detail_corr`: `-0.8495775413513184`

## Training Result

- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/logs/train_20260610_183150.log`
- Exit code: `train_exit_code=0`
- Completed step: `800`
- Training wall time to final validation: about `17.66` minutes
- Best selection metric: `val_psnr`
- Best validation step: `600`
- Best validation loss: `0.133898965716362`
- Best validation PSNR: `17.595543271291344`
- Final step `800` validation loss: `0.13759984984993934`
- Final step `800` validation PSNR: `17.33071631834636`

Validation trajectory:

| Step | Val Loss | Val PSNR | Control Corr | Control Std |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.26816445630043745 | 12.154267020652284 | 0.18174902511294933 | 0.0026126876508351417 |
| 200 | 0.22951363626867533 | 13.577124621371283 | 0.15515564426779746 | 0.006754143736325205 |
| 300 | 0.1676899965852499 | 16.072779444834147 | 0.15261890170164405 | 0.00951432227389887 |
| 400 | 0.25282471496611836 | 12.537675102298294 | 0.15170264491811394 | 0.008561895182356238 |
| 500 | 0.14730290342122315 | 16.962716911317116 | 0.15515727820806205 | 0.008751078108325601 |
| 600 | 0.133898965716362 | 17.595543271291344 | 0.1550898371823132 | 0.008817849517799913 |
| 700 | 0.16956035759299992 | 15.93838793094081 | 0.15383999765384943 | 0.007548154536634683 |
| 800 | 0.13759984984993934 | 17.33071631834636 | 0.16455508021172136 | 0.010373912127688527 |

## CPU Test Evaluation

Standard paired eval:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/metrics/test_eval_best_val_cpu.json`
- Checkpoint: `best_val.pt`
- Checkpoint step: `600`
- Device: `cpu`
- Split: `test`
- Samples: `100`
- Crop size: `512`
- `psnr_mean`: `17.067359898802888`
- `loss_mean`: `0.13247113279998302`
- `auxiliary_loss_mean`: `0.0032316413219086824`
- `total_loss_mean`: `0.13570277445018292`

Task-specific diagnostics:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/metrics/test_task_metrics_best_val_cpu.json`
- `ssim_mean`: `0.6860608831048012`
- `edge_magnitude_l1_mean`: `0.16485614955425262`
- `high_frequency_l1_mean`: `0.022253094930201767`
- `local_contrast_l1_mean`: `0.028198902057483793`
- `dark_channel_l1_mean`: `0.13558692712336778`
- `luminance_l1_mean`: `0.11371097903698683`
- `chroma_l1_mean`: `0.0393355306237936`
- `saturation_l1_mean`: `0.05573128338903189`

## Comparison

| Metric | A_4kd02 | A_4kd04 | A_4kd05 | 05-04 | 05-02 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val PSNR | 17.529067811582 | 17.684761970685 | 17.595543271291 | -0.089218699394 | 0.066475459709 |
| Test PSNR | 15.909636046480 | 16.901385528055 | 17.067359898803 | 0.165974370748 | 1.157723852323 |
| Test loss | 0.147730302103 | 0.128418906741 | 0.132471132800 | 0.004052226059 | -0.015259169303 |
| SSIM | 0.667810640037 | 0.676181002557 | 0.686060883105 | 0.009879880548 | 0.018250243068 |
| Edge L1 | 0.144763366207 | 0.168340641856 | 0.164856149554 | -0.003484492302 | 0.020092783347 |
| High-freq L1 | 0.020703289686 | 0.022525039371 | 0.022253094930 | -0.000271944441 | 0.001549805244 |
| Local-contrast L1 | 0.022215690576 | 0.028941890737 | 0.028198902057 | -0.000742988680 | 0.005983211482 |
| Dark-channel L1 | 0.138143544011 | 0.130480288416 | 0.135586927123 | 0.005106638707 | -0.002556616887 |
| Luminance L1 | 0.142751846835 | 0.115998680778 | 0.113710979037 | -0.002287701741 | -0.029040867798 |
| Chroma L1 | 0.033509960715 | 0.038946636226 | 0.039335530624 | 0.000388894398 | 0.005825569909 |
| Saturation L1 | 0.056755381487 | 0.053027442358 | 0.055731283389 | 0.002703841031 | -0.001024098098 |

## Interpretation

`A_4kd05` did not beat `A_4kd04` on best validation PSNR, but it did improve the
independent CPU test PSNR and SSIM. It also partially repaired the specific
detail regressions that motivated the run: edge, high-frequency, and
local-contrast L1 are all lower than `A_4kd04`. That suggests the detail
protection gate is useful for generalization and texture preservation, even
though the added constraints cost some validation peak and worsen dark-channel,
chroma, and saturation relative to `A_4kd04`.

The mechanism lesson is therefore not "add more losses"; it is that the blend
should be spatially selective. A fixed uniform blend can over-smooth or damage
high-detail regions. Detail-aware suppression of the control overlay is a
better bias, but the current hand-set gate/loss weights are still a tradeoff.

## Boundary

This result is still a `4KDehaze_local_90_5_5_fast_probe_512` result with `512`
train pairs, `100` validation pairs, and `100` test pairs. It must not be
described as an official/full benchmark or compared directly with historical
official anchors such as `M04/A_y03` or `A_a02`.

## Next Step

Keep `A_4kd05` as the current fast-probe generalization/detail-protection
candidate, while keeping `A_4kd04` as the validation-peak reference. The next
narrow experiment should reduce the manual tradeoff: preserve the detail-aware
blend map, but tune or schedule its strength/floor and reduce the added
chroma/saturation penalty. A cleaner next probe would be a warmup/schedule:
start near `A_4kd04` uniform blending, then progressively enable detail
suppression after the control canvas becomes less noisy.
