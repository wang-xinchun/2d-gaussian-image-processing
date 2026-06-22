# Result Analysis: V1/T01/A_4kd02

## Status

`A_4kd02` completed its low-footprint shared-GPU fast-probe training and CPU
test evaluation on 2026-06-10. It used the `4KDehaze_local_90_5_5_fast_probe_512`
subset and did not overwrite `A_4kd01`.

## Run Safety

- Allowed GPUs: physical GPU0 or GPU3 only.
- Actual training GPU: physical GPU3.
- GPU1/GPU2 use by this project: none.
- External process handling: no external process was killed or displaced.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory recorded in metrics: about `10.21875GiB`.
- Cleanup: no `A_4kd02`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or project tmux
  session remained after completion.

## Training Result

- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/logs/train_20260609_232733.log`
- Exit code: `train_exit_code=0`
- Completed step: `800`
- Training wall time: about `33.43` minutes
- Best selection metric: `val_loss`
- Best/final validation step: `800`
- Best/final `val_loss`: `0.12707203164696693`
- Best/final `val_psnr`: `17.529067811582326`
- Best summary:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/best_val_summary.json`

Validation trajectory:

| Step | Val Loss | Val PSNR |
| ---: | ---: | ---: |
| 100 | 0.19756620861589907 | 14.261986201609318 |
| 200 | 0.21155653156340123 | 13.757093252523482 |
| 300 | 0.20246284995228053 | 14.285158329013957 |
| 400 | 0.21947744872421027 | 13.665306345150338 |
| 500 | 0.13587064921855926 | 17.352591334730707 |
| 600 | 0.1352057947218418 | 17.003091859743964 |
| 700 | 0.2770421342179179 | 12.11811651064695 |
| 800 | 0.12707203164696693 | 17.529067811582326 |

## CPU Test Evaluation

Standard paired eval:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/test_eval_best_val_cpu.json`
- Checkpoint: `best_val.pt`
- Checkpoint step: `800`
- Device: `cpu`
- Split: `test`
- Samples: `100`
- Crop size: `512`
- `psnr_mean`: `15.909636046480303`
- `loss_mean`: `0.14773030210286378`

Task-specific diagnostics:

- File:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd02/metrics/test_task_metrics_best_val_cpu.json`
- `ssim_mean`: `0.6678106400370598`
- `edge_magnitude_l1_mean`: `0.14476336620748043`
- `high_frequency_l1_mean`: `0.02070328968577087`
- `local_contrast_l1_mean`: `0.022215690575540067`
- `dark_channel_l1_mean`: `0.138143544010818`
- `luminance_l1_mean`: `0.1427518468350172`
- `chroma_l1_mean`: `0.033509960714727643`
- `saturation_l1_mean`: `0.056755381487309936`

## Interpretation

This run is useful as a safety and viability probe: it shows the old-mainline
dehaze-prior variant can train on the shared server by squeezing into GPU3 with
about 10GiB reserved memory, while respecting the GPU0/3-only and no-displacement
constraints.

It is not an official breakthrough claim. The dataset is a `512/100/100`
fast-probe subset and evaluation uses `512` crops, so these metrics cannot be
compared directly with historical official/full-image `UHD_deblur` results such
as `M04/A_y03`, or with accepted historical `UHD_haze` references such as
`A_a02`.
