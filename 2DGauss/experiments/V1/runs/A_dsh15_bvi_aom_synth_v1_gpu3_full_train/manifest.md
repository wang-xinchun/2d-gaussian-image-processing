# A_dsh15 Manifest

- Full id: `V1/T01/A_dsh15_bvi_aom_synth_v1_gpu3_full_train`
- Status: completed and accepted as a high-quality controlled paired-frame deshaking training run.
- Completed at: `2026-06-06 16:28:31 CST`
- Task: UHD video deshaking / stabilization, evaluated as synthetic shaky-frame to clean-frame supervised restoration.
- Data root: `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`
- Train view: `112` linked `.npy` input-target pairs.
- Val view: `32` linked `.npy` input-target pairs.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh15_bvi_aom_synth_v1_gpu3_full_train.toml`
- Launch wrapper: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_train_once.sh`
- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_161547_120913.log`
- Exit code file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_161547_120913.log.exit_code`

## Training Protocol

- Physical GPU: `3` only.
- Reason for GPU3-only: GPU0 was occupied by external processes at launch time; GPU1/2 are not allowed for this project. GPU3 was idle and inside the approved GPU set.
- Batch size: `24`, effective global batch `24`.
- Crop: `512x512` train and val.
- Workers: `0`.
- CUDA memory fraction: `0.85`.
- Max steps: `1200`.
- Validation: every `50` steps on all `32` val batches.
- Checkpoint interval: every `100` steps.
- Best checkpoint selection: lowest validation loss.

## Produced Artifacts

- Train metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/metrics/train_metrics.jsonl`
- Val metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/metrics/val_metrics.jsonl`
- Best summary: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/metrics/best_val_summary.json`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/checkpoints/best_val.pt`
- Final checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/checkpoints/final.pt`
- Latest checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/checkpoints/latest.pt`
- Independent best-val eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/metrics/deshake_eval_best_val.json`
- Independent final eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train/metrics/deshake_eval_final.json`

## Boundary

A_dsh15 is a meaningful controlled training and acceptance point for the synthesized BVI-AOM paired-frame deshaking view. It is not a complete video-stabilization system: the model still operates as a single-frame paired restorer and does not yet model temporal warping, sequence-level crop planning, boundary completion, or subjective long-video stability.
