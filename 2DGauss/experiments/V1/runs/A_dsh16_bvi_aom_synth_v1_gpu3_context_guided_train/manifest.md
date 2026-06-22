# A_dsh16 Manifest

- Full id: `V1/T01/A_dsh16_bvi_aom_synth_v1_gpu3_context_guided_train`
- Status: launched after preparation and failed quickly with a project-local CUDA OOM at `batch_size=4`; no checkpoint or metric artifact was produced.
- Purpose: next controlled GPU3-only deshaking branch after `A_dsh15`, using the same verified BVI-AOM synthetic paired-frame view and the same full 32-pair validation protocol.
- Dataset root: `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh16_bvi_aom_synth_v1_gpu3_context_guided_train.toml`
- Run-local config mirror: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh16_bvi_aom_synth_v1_gpu3_context_guided_train/configs/a_dsh16_bvi_aom_synth_v1_gpu3_context_guided_train.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_context_guided_train_once.sh`

## Delta From A_dsh15

- Keeps the corrected UHD video deshaking task, paired-frame data, `112` train pairs, `32` val pairs, `num_workers=0`, and full-val cadence.
- Increases model capacity from `width=32` to `48`, with `highres_blocks=3`, `lowres_blocks=6`, `decoder_blocks=3`, `affine_rank=6`, and `gaussian_atoms=6`.
- Enables existing default-off structural modules: edge guidance, directional-frequency guidance, train-inference consistency, and Gaussian decoder guidance.
- Uses `train_context_size=728` with `train_supervision_crop_size=512`, so training sees a larger crop but computes reconstruction loss on the centered 512px supervision area.
- Lowers single-GPU batch size to `4` to trade the A_dsh15 high batch for larger context and safer GPU3 memory headroom.

## Launch Boundary

The dedicated wrapper defaults to dry-run. A real launch requires:

```bash
TWODGAUSS_DRY_RUN=0 \
TWODGAUSS_DESHAKE_CONFIRM_GPU3_TRAIN=launch_gpu3_deshake_context_guided_train \
bash /home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_context_guided_train_once.sh
```

The wrapper keeps physical GPU3 only, requires GPU3 free memory at least `42000MiB`, blocks GPU utilization above `20%`, caps batch/global batch at `4`, caps `train_context_size` at `728`, requires `train_supervision_crop_size=512`, and leaves GPU1/GPU2 disallowed.

## Preparation Evidence

- `bash -n` passed for the base GPU3 launcher and the A_dsh16 context-guided wrapper.
- `py_compile` passed for `scripts/train_v1_t01.py`, `scripts/preflight_deshake_ddp_gpu0_3.py`, and `scripts/evaluate_deshake_checkpoint.py`.
- TOML/model construction passed: `width=48`, `train_context_size=728`, `train_supervision_crop_size=512`, edge guidance on, directional guidance on, Gaussian decoder guidance on, total parameters `267864`.
- CPU preflight passed with `4` train samples and the full `32` validation samples. The observed batch tensor was `(4, 3, 544, 960)` because the chosen training frames are smaller than the requested `728` context in height and therefore use the full common frame size for context.
- GPU3 launcher dry-run passed. Preflight saw physical GPU3 with `48518MiB` free and `0%` utilization; pair counts were `112` train and `32` val; config gate recorded `batch_size=4`, `max_steps=1600`, `wall_clock_minutes=70.0`, `train_context_size=728`, and `train_supervision_crop_size=512`.
- Dry-run artifacts: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_165731_451317.log.preflight.json` and `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/gpu3_train_config_gate_20260606_165731_451317.json`.

## Launch Outcome

- Real launch used physical GPU3 only after live preflight passed.
- Training exited quickly with `train_exit_code=1` due to CUDA OOM inside the project process while using the `batch_size=4` context-guided config.
- Failure log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_172806_738530.log`
- GPU0 was not used, GPU1/2 remained disallowed, no external process was killed or displaced, and GPU3 returned to idle after failure.
- No checkpoint or metric artifact exists for A_dsh16.
