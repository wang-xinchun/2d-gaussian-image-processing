# Result Analysis: V1/T01/A_4kd06

## Status

`A_4kd06` completed on 2026-06-10. It is accepted as a useful isolation
diagnostic, but it is not promoted over `A_4kd04` or `A_4kd05`.

The result is clean scientifically: a lighter detail-aware blend gate, without
the extra `A_4kd05` protection losses, mostly reproduces `A_4kd04` rather than
`A_4kd05`. That makes it evidence against "light gate alone" as the next
solution.

## Run Safety

- Allowed GPUs: physical GPU0 or GPU3 only.
- Actual training GPU: physical GPU3.
- GPU1/GPU2 use by this project: none.
- External process handling: no external process was killed or displaced.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory recorded in training metrics: about
  `11.244140625GiB`.
- Cleanup: no `A_4kd06`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or A_4kd06 tmux
  session remained after completion.

## Model Delta

Relative to `A_4kd04`, `A_4kd06` keeps the same blended shared demand-control
and the same loss settings, then adds only a milder detail-aware control
blend-map:

- `gaussian_control_canvas_share_to_guidance_detail_preserve = true`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_strength = 0.45`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_floor = 0.55`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_kernel_size = 5`

It deliberately removes the extra edge/high-frequency/local-contrast/chroma
protection pressure used by `A_4kd05`.

The gate behaved directionally as intended, with lower effective blend in
high-detail regions:

- `val_control_guidance_blend_high_detail_mean`: `0.18403567776083946`
- `val_control_guidance_blend_low_detail_mean`: `0.23201072290539743`
- `val_control_guidance_blend_detail_corr`: `-0.8495775282382965`

## Training Result

- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/logs/train_20260610_225710.log`
- Exit code: `train_exit_code=0`
- Completed step: `800`
- Training wall time to final validation: about `19.28` minutes
- Best selection metric: `val_psnr`
- Best validation step: `600`
- Best validation loss: `0.1261259661614895`
- Best validation PSNR: `17.674754617905425`
- Final step `800` validation loss: `0.13434834714978933`
- Final step `800` validation PSNR: `17.121298116103546`

Validation trajectory:

| Step | Val Loss | Val PSNR | Control Corr | Control Std |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2598635434359312 | 12.143340964048821 | 0.18262479040306062 | 0.002625319864600897 |
| 200 | 0.22268389403820038 | 13.516026916214484 | 0.15858855729922652 | 0.006861099447123706 |
| 300 | 0.1607355586066842 | 16.119270364268708 | 0.15421996306627989 | 0.009444087410811335 |
| 400 | 0.2366262026876211 | 12.792125397308286 | 0.1514870880357921 | 0.008626000001095235 |
| 500 | 0.1393875202909112 | 17.08930547268828 | 0.15446350684389473 | 0.008644997000228614 |
| 600 | 0.1261259661614895 | 17.674754617905425 | 0.1552286056894809 | 0.00881028919480741 |
| 700 | 0.15345496542751788 | 15.942381044613601 | 0.15607098589651286 | 0.00895103286486119 |
| 800 | 0.13434834714978933 | 17.121298116103546 | 0.1710128222592175 | 0.007334174485877156 |

## CPU Test Evaluation

Standard paired eval:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/metrics/test_eval_best_val_cpu.json`
- Checkpoint: `best_val.pt`
- Checkpoint step: `600`
- Device: `cpu`
- Split: `test`
- Samples: `100`
- Crop size: `512`
- `psnr_mean`: `16.89408572072857`
- `loss_mean`: `0.12854937952011825`
- `auxiliary_loss_mean`: `0.003232535165734589`
- `total_loss_mean`: `0.13178191486746071`

Task-specific diagnostics:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/metrics/test_task_metrics_best_val_cpu.json`
- `ssim_mean`: `0.6761693635582924`
- `edge_magnitude_l1_mean`: `0.16829446755349636`
- `high_frequency_l1_mean`: `0.02252224282361567`
- `local_contrast_l1_mean`: `0.02893139421008527`
- `dark_channel_l1_mean`: `0.13064265988767146`
- `luminance_l1_mean`: `0.1160976618155837`
- `chroma_l1_mean`: `0.03894915344193578`
- `saturation_l1_mean`: `0.05304279964417219`

## Comparison

| Metric | A_4kd04 | A_4kd05 | A_4kd06 | 06-04 | 06-05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val PSNR | 17.684761970685 | 17.595543271291 | 17.674754617905 | -0.010007352780 | 0.079211346614 |
| Test PSNR | 16.901385528055 | 17.067359898803 | 16.894085720729 | -0.007299807326 | -0.173274178074 |
| Test loss | 0.128418906741 | 0.132471132800 | 0.128549379520 | 0.000130472779 | -0.003921753280 |
| SSIM | 0.676181002557 | 0.686060883105 | 0.676169363558 | -0.000011638999 | -0.009891519547 |
| Edge L1 | 0.168340641856 | 0.164856149554 | 0.168294467553 | -0.000046174303 | 0.003438317999 |
| High-freq L1 | 0.022525039371 | 0.022253094930 | 0.022522242824 | -0.000002796547 | 0.000269147893 |
| Local-contrast L1 | 0.028941890737 | 0.028198902057 | 0.028931394210 | -0.000010496527 | 0.000732492153 |
| Dark-channel L1 | 0.130480288416 | 0.135586927123 | 0.130642659888 | 0.000162371472 | -0.004944267236 |
| Luminance L1 | 0.115998680778 | 0.113710979037 | 0.116097661816 | 0.000098981038 | 0.002386682779 |
| Chroma L1 | 0.038946636226 | 0.039335530624 | 0.038949153442 | 0.000002517216 | -0.000386377182 |
| Saturation L1 | 0.053027442358 | 0.055731283389 | 0.053042799644 | 0.000015357286 | -0.002688483745 |

## Interpretation

`A_4kd06` almost perfectly preserves `A_4kd04`'s validation peak: only about
`0.0100 dB` lower. It also leaves most task diagnostics close to `A_4kd04`.
That is useful because the milder detail gate is not destabilizing.

But it does not carry the main benefit that made `A_4kd05` interesting. CPU
test PSNR is about `0.1733 dB` below `A_4kd05`, and SSIM is about `0.00989`
lower. The tiny improvements over `A_4kd04` on edge/high-frequency/local
contrast are too small to count as the A_4kd05 detail-repair effect.

The mechanism lesson is that "directionally correct but weak" detail gating is
not enough. The A_4kd05 gain likely came from stronger spatial selectivity,
the added protection losses, or their interaction. The next useful probe
should therefore test scheduling/strength rather than merely weakening the
gate from step 1.

## Boundary

This result is still a `4KDehaze_local_90_5_5_fast_probe_512` result with `512`
train pairs, `100` validation pairs, and `100` test pairs. It must not be
described as an official/full benchmark or compared directly with historical
official anchors such as `M04/A_y03` or `A_a02`.

## Next Step

Keep `A_4kd04` as the validation-peak reference and `A_4kd05` as the
generalization/detail-protection candidate. Treat `A_4kd06` as a negative
isolation result for light gate alone. The next narrow experiment should use a
schedule: start with `A_4kd04`-style uniform blending while the control canvas
is noisy, then progressively enable stronger `A_4kd05`-style detail
suppression after the control signal has stabilized.
