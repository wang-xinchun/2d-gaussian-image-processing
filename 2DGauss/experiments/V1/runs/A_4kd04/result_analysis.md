# Result Analysis: V1/T01/A_4kd04

## Status

`A_4kd04` completed on 2026-06-10. It is accepted as a useful fast-probe
mechanism candidate for blended demand-control, not as an official/full split
benchmark.

The main conclusion is narrow but important: hard replacement in `A_4kd03` was
too destructive, while conservative blending in `A_4kd04` preserved enough of
the original Gaussian guidance to beat the `A_4kd02` fast-probe reference.

## Run Safety

- Allowed GPUs: physical GPU0 or GPU3 only.
- Actual training GPU: physical GPU3.
- GPU1/GPU2 use by this project: none.
- External process handling: no external process was killed or displaced.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory recorded in training metrics: about `11.22265625GiB`.
- Cleanup: no `A_4kd04`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or A_4kd tmux
  session remained after completion.

## Training Result

- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/logs/train_20260610_122251.log`
- Exit code: `train_exit_code=0`
- Completed step: `800`
- Training wall time to final validation: about `17.61` minutes
- Best selection metric: `val_loss`
- Best validation step: `600`
- Best validation loss: `0.12592033293098212`
- Best validation PSNR: `17.6847619706854`
- Final step `800` validation loss: `0.13507467851042748`
- Final step `800` validation PSNR: `17.09676323141356`
- Best summary:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/metrics/best_val_summary.json`

Validation trajectory:

| Step | Val Loss | Val PSNR | Control Corr | Control Std |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.25986311316490174 | 12.14336371684264 | 0.18262168530374764 | 0.002628382737748325 |
| 200 | 0.22268393870443107 | 13.516037826383469 | 0.15862393343821168 | 0.006893569966778159 |
| 300 | 0.1607209062203765 | 16.120085324965977 | 0.15424220249056816 | 0.009457392380572855 |
| 400 | 0.23668597608804703 | 12.789936942709032 | 0.1515310400072485 | 0.00861094856634736 |
| 500 | 0.13942490115761758 | 17.089125400247006 | 0.15442832973785697 | 0.008655817951075733 |
| 600 | 0.12592033293098212 | 17.6847619706854 | 0.1552173734921962 | 0.00879866886883974 |
| 700 | 0.1531327561661601 | 15.959205565872283 | 0.1564554020902142 | 0.008938682721927762 |
| 800 | 0.13507467851042748 | 17.09676323141356 | 0.17027385641820728 | 0.007498935731127858 |

## CPU Test Evaluation

Standard paired eval:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/metrics/test_eval_best_val_cpu.json`
- Checkpoint: `best_val.pt`
- Checkpoint step: `600`
- Device: `cpu`
- Split: `test`
- Samples: `100`
- Crop size: `512`
- `psnr_mean`: `16.901385528055062`
- `loss_mean`: `0.12841890674084425`
- `auxiliary_loss_mean`: `0.0032325189001858236`
- `total_loss_mean`: `0.13165142569690944`

Task-specific diagnostics:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/metrics/test_task_metrics_best_val_cpu.json`
- `ssim_mean`: `0.6761810025572776`
- `edge_magnitude_l1_mean`: `0.16834064185619355`
- `high_frequency_l1_mean`: `0.022525039371103048`
- `local_contrast_l1_mean`: `0.02894189073704183`
- `dark_channel_l1_mean`: `0.13048028841614723`
- `luminance_l1_mean`: `0.11599868077784777`
- `chroma_l1_mean`: `0.0389466362260282`
- `saturation_l1_mean`: `0.0530274423584342`

## Comparison With A_4kd02

Same fast-probe validation protocol:

- `A_4kd02` best/final validation PSNR: `17.529067811582326`
- `A_4kd04` best validation PSNR: `17.6847619706854`
- Delta: `+0.15569415910307394 dB`

Same CPU test protocol:

- `A_4kd02` test PSNR: `15.909636046480303`
- `A_4kd04` test PSNR: `16.901385528055062`
- Delta: `+0.991749481574759 dB`
- `A_4kd02` test loss: `0.14773030210286378`
- `A_4kd04` test loss: `0.12841890674084425`

Task diagnostics are mixed. `A_4kd04` improves test `ssim_mean`,
`dark_channel_l1_mean`, `luminance_l1_mean`, and `saturation_l1_mean` versus
`A_4kd02`, but worsens edge/high-frequency/local-contrast/chroma proxies. That
is consistent with a dehazing-oriented gain that may still need detail and edge
protection before scaling up.

## Interpretation

`A_4kd03` answered the negative half of the bottleneck question: an immature
demand-control canvas should not replace the main Gaussian attention path.
`A_4kd04` answers the constructive half: a small blended demand-control overlay
can help when the original Gaussian attention remains the primary guidance.

The control canvas still has much lower variation than its target
(`val_control_target_std=0.10203479953110219`), so the mechanism is not fully
solved. But the blended path is the first fast-probe in this sequence that
improves both validation and CPU test performance over the old `A_4kd02`
reference.

## Boundary

This result is not an official dehazing or deblurring benchmark. It uses the
`4KDehaze_local_90_5_5_fast_probe_512` view with `512` train pairs, `100`
validation pairs, and `100` test pairs. It should guide the next model
iteration, but it must not be compared directly with historical full/official
anchors such as `M04/A_y03` or `A_a02`.

## Next Step

Use `A_4kd04` step `600` as the fast-probe candidate checkpoint. The next model
iteration should keep blended shared demand-control, then improve edge/detail
protection and reduce late validation oscillation. Do not return to hard
attention replacement unless there is a separate warmup or confidence schedule.
