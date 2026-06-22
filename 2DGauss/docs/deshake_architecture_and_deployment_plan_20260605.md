# UHD Video Deshaking Architecture And Deployment Plan

Updated: 2026-06-05 21:31 CST

## Current Architecture Readiness

The current V1/T01 model is a single-frame UHD restoration model. It has useful components for frame-level enhancement:

- a high-resolution decoder/detail path for local texture and residual recovery;
- a low-resolution Gaussian/control path for low-entropy structural guidance;
- optional explicit low/high-frequency split, Gaussian control canvas, and guided feedback modules;
- proven UHD_deblur evidence through the frozen `M04/A_y03` milestone.

It is not yet a complete video deshaking / stabilization model. The current training and evaluation path uses `PairedImageDataset`, which reads paired still images from `input` and `target` directories. There is no sequence loader, motion/warp estimation, temporal consistency loss, crop/border policy, or video-stabilization metric in the active training path.

## Correct Task Scope

The current new task is ultra-high-definition video deshaking / video stabilization. `UHD_haze` caches and `A_b02*` runs are resource-calibration history only and must not be counted as deshaking progress.

## Deployment Strategy While Data Transfers

Use a staged path that avoids pretending the single-frame model is already a stabilizer:

1. **Data gate**: inspect the incoming 4K/8K video dataset, identify whether it is raw video, frame sequences, paired shaky/stable frames, or metadata-supervised stabilization.
2. **Proxy metrics gate**: compute no-reference temporal stability proxies for frame sequences and, if stable targets exist, reference L1/PSNR-like proxies.
3. **Frame-enhancement baseline**: run the frozen `A_y03` single-frame model over extracted frames as a frame-level enhancement baseline. This is not a deshaking result.
4. **Deshaking model design**: after the data/supervision form is known, decide whether to add a motion-aware wrapper, a temporal-control branch, or a stabilization pre/post module around the existing Gaussian restorer.

## Prepared Scripts

- `scripts/inspect_deshake_dataset.py`: read-only dataset scanner for incoming deshaking data.
- `scripts/evaluate_deshake_sequence_metrics.py`: lightweight temporal stability proxy metrics for frame folders.
- `scripts/run_v1_frame_enhancer.py`: applies the current V1/T01 restorer to frame directories as a frame-enhancement baseline, with optional tiled inference for UHD/8K frames.

## Pre-Launch Boundary

Do not launch deshaking training until all of these are true:

- deshaking data path exists locally;
- supervision form is known;
- train/val/test split is defined;
- frame or video decoding path is verified;
- task-specific metrics are selected;
- run package points to deshaking data, not `UHD_haze`;
- GPU0/3 gate passes and does not displace external processes.
