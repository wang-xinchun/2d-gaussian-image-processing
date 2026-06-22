# deshake_ddp_0_3_pending Manifest

- Full id: `V1/T01/deshake_ddp_0_3_pending`
- Purpose: guarded launch wrappers for the corrected UHD video deshaking / stabilization task.
- Physical GPUs: `0,3` for DDP; explicit GPU0-only for the large-batch smoke path; explicit GPU3-only for no-displacement full training when GPU0 is busy.
- Status: A_dsh15 GPU3-only full training completed and remains accepted on the synthesized BVI-AOM paired-frame subset. A_dsh16 context-guided GPU3-only batch-size-4 launch failed with project-local CUDA OOM. A_dsh17 context-guided GPU3-only batch-size-2 launch completed safely and was independently evaluated, but did not beat A_dsh15.
- Launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_tmux.sh`
- Smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_smoke_once.sh`
- Queued DDP smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_smoke.sh`
- Explicit shared-GPU DDP smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_shared_smoke_once.sh`
- Explicit shared-GPU DDP train launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_shared_train_once.sh`
- Explicit GPU3-only train launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_train_once.sh`
- Explicit GPU3-only context-guided candidate launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_context_guided_train_once.sh`
- Queued shared-GPU DDP smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_shared_smoke.sh`
- GPU0 smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu0_smoke_once.sh`
- Prepare-and-queue smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_smoke.sh`
- Prepare-and-queue shared smoke launcher: `experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_shared_smoke.sh`
- Readiness checker: `experiments/V1/runs/deshake_ddp_0_3_pending/check_deshake_ddp_readiness.sh`
- Readiness monitor: `experiments/V1/runs/deshake_ddp_0_3_pending/monitor_deshake_ddp_readiness.sh`
- Paired-frame preparer: `scripts/prepare_deshake_paired_frames.py`
- Config generator: `scripts/create_deshake_ddp_config.py`
- Preflight gate: `scripts/preflight_deshake_ddp_gpu0_3.py`
- Gate doc: `docs/deshake_ddp_gpu0_3_deployment_gate.md`
- Positive preflight selftest: `experiments/V1/runs/deshake_ddp_0_3_pending/preflight_selftest`

## Launch Boundary

This package does not contain a baked-in dataset path or config. It must be
called with:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/deshake_ddp_config.toml \
bash experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_tmux.sh
```

The launcher runs the deshaking preflight gate before starting DDP. It exits
before launch if the data root is missing, too small, lacks video/frame evidence,
points at old `UHD_haze` data, or if the config lacks deshaking/stabilization
intent.

Default data gate values can be adjusted only when the dataset layout is
explicitly logged:

- `TWODGAUSS_MAX_GPU_UTILIZATION`, default `20`
- `TWODGAUSS_GPU_UTILIZATION_POLICY`, default `block`
- `TWODGAUSS_MAX_PER_RANK_BATCH_SIZE`, default `2`
- `TWODGAUSS_MAX_GLOBAL_BATCH_SIZE`, default `4`
- `TWODGAUSS_MAX_NUM_WORKERS`, default `2`
- `TWODGAUSS_DESHAKE_MIN_DATA_BYTES`, default `322122547200` (`300GiB`)
- `TWODGAUSS_DESHAKE_MIN_VIDEO_FILES`, default `1`
- `TWODGAUSS_DESHAKE_MIN_FRAME_FILES`, default `0`
- `TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS`, default `1`
- `TWODGAUSS_DESHAKE_MIN_VAL_PAIRS`, default `1`
- `TWODGAUSS_DESHAKE_MAX_SCAN_FILES`, default `500000`
- `TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS`, default `600`

## Current Safe State

As of `2026-06-06 17:45:15 CST`, context-guided follow-up deployment has
completed without displacing external training. `A_dsh16` launched on physical
GPU3 only after live preflight passed, but failed quickly with `train_exit_code=1`
from a project-local CUDA OOM at `batch_size=4`; it produced no checkpoints or
metrics. `A_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train` lowered the
same branch to `batch_size=2`. Its first dry-run was correctly blocked when a
new process appeared on GPU3; after GPU3 returned to idle, dry-run and real
launch passed.

A_dsh17 real training used physical GPU3 only, completed `1600` steps with
`train_exit_code=0`, produced `321` train metric records and `32` full-val
records, saved `best_val.pt` at step `1350`, and wrote final/latest/step_001600
checkpoints. The train log is
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_173118_784237.log`.
Independent eval improved over the shaky input baseline, but A_dsh17 best-val
(`18.343223705783423` PSNR / `0.4939711121842265` SSIM) remained below A_dsh15
final (`18.504268639458388` PSNR / `0.5011140927672386` SSIM). Therefore
A_dsh15 remains the accepted baseline; A_dsh17 is a safe completed comparison
run, not the new best model.

Post-run cleanup found no A_dsh17 train/tmux/launcher residue, GPU3 returned to
`0MiB` used, GPU0 was avoided, GPU1/2 were unused, and no external process was
killed.

As of `2026-06-06 16:58:06 CST`, next candidate
`A_dsh16_bvi_aom_synth_v1_gpu3_context_guided_train` has been prepared but not
launched. It keeps the same verified BVI-AOM synthetic paired-frame data as
A_dsh15, uses `112` train pairs and `32` val pairs, and preserves the full
32-pair validation protocol. It changes the branch from the A_dsh15 conservative
base model to the already implemented structural path: `width=48`,
`highres_blocks=3`, `lowres_blocks=6`, `decoder_blocks=3`, `affine_rank=6`,
`gaussian_atoms=6`, edge guidance enabled, directional-frequency guidance
enabled, `train_context_size=728` with `train_supervision_crop_size=512`, and
Gaussian decoder guidance enabled. Batch size is reduced to `4` on physical
GPU3 to trade A_dsh15's high batch for larger context and safer memory headroom.

Validation completed for preparation only: `bash -n` and `py_compile` passed,
TOML/model construction passed with `267864` parameters, CPU preflight completed
one forward/backward plus full 32-batch validation, and the dedicated GPU3
context-guided launcher dry-run passed live GPU/data/config/context gates. The
dry-run preflight saw physical GPU3 at `48518MiB` free and `0%` utilization and
wrote
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_165731_451317.log.preflight.json`
plus
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/gpu3_train_config_gate_20260606_165731_451317.json`.
No real training log, tmux session, checkpoint, or A_dsh16 metric artifact was
created.

As of `2026-06-06 16:28:31 CST`, controlled full training run
`A_dsh15_bvi_aom_synth_v1_gpu3_full_train` has completed and is the current
latest accepted run. It used the verified BVI-AOM synthetic deshaking paired
`.npy` view, but intentionally avoided GPU0 because GPU0 was occupied by
external processes at launch time. It used physical GPU3 only,
`batch_size=24` / global `24`, `dataset.num_workers=0`, 512px train/val crop,
`max_steps=1200`, full 32-batch validation every 50 steps, and
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`.

The launch used
`experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_train_once.sh`.
The train log is
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_161547_120913.log`,
and it records `train_exit_code=0`. Run artifacts are under
`experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train`: `241` train
metric records, `24` full-validation records, `best_val.pt` at step `1000`,
and final/latest/step_001200 checkpoints at step `1200`.

Independent validation on the 32-pair val view shows the final checkpoint
improving over the shaky input baseline from `16.507102228205603` PSNR and
`0.37895671324804425` SSIM to `18.504268639458388` PSNR and
`0.5011140927672386` SSIM. Post-run cleanup found no project tmux/training
residue, no GPU1/2 use, and GPU3 returned to `0MiB` used.

As of `2026-06-06 13:00:46 CST`, controlled shared training run
`A_dsh14_bvi_aom_synth_v1_shared_ddp_train` has completed on the verified
BVI-AOM synthetic deshaking paired-frame view. The config is
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh14_bvi_aom_synth_v1_shared_ddp_train.toml`.
It used physical GPU0+GPU3, per-rank `batch_size=12`, global `24`,
`dataset.num_workers=0`, 512px train/val crop, `max_steps=60`,
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`, a live free-memory gate of `18000MiB`,
and runtime own-session stop threshold `8192MiB`.

The launch used the explicit shared warning wrapper
`launch_deshake_ddp_shared_smoke_once.sh`, not the default formal
block-policy path. Startup preflight passed data/config/pair/crop/batch/worker
checks; high utilization was handled only under `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`.
The training log
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_ddp_20260606_125827.log`
records `train_exit_code=0`, and the wrapper completed with
`shared_smoke_status=completed`.

Run artifacts are under
`experiments/V1/runs/A_dsh14_bvi_aom_synth_v1_shared_ddp_train`: `60` train
metric records, `6` validation records, `latest.pt`, `final.pt`, and
`step_000060.pt`. Final validation is `val_loss=0.0661571198143065` and
`val_psnr=20.984133518913794`. Rank metrics map rank0 to physical GPU0 and
rank1 to physical GPU3, with peak reserved memory `18.91796875GiB` and
`18.875GiB` respectively.

Post-run cleanup found no project tmux/training/torchrun/deshake-launcher
residue, no GPU1/2 use, and no killed external process.

As of `2026-06-06 12:49:02 CST`, the first real synthesized deshaking smoke
dataset exists at `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`. It was built
from the uploaded BVI-AOM 2176p videos with the extracted UHD shake synthesis
algorithm and strict quality filtering. The manifest
`/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1/manifest.json` records
`clip_count=24`, `pass_count=18`, `fail_count=6`; the six failed clips were
excluded because `trajectory_discontinuity_detected` tripped the quality gate.
The training view contains `112` train pairs and `32` val pairs as
`uint16 RGB .npy` frames under `paired_frames/train` and `paired_frames/val`.

The current first-smoke config is
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh13_bvi_aom_synth_v1_shared_ddp_smoke.toml`.
It uses physical GPU0+3, per-rank `batch_size=2`, global `batch_size=4`,
`dataset.num_workers=0`, 512px train/val crop, `max_steps=5`, and a 10-minute
wall-clock cap. CPU preflight passed end-to-end on `.npy` frames, and the queued
shared DDP smoke dry-run reached the shared wrapper with `status=dry_run_invoked`
in `queued_shared_ddp_smoke_summary_20260606_124629_2789952.json`.

Formal non-dry-run training has not started. The final conservative live
preflight artifact
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/a_dsh13_preflight_gpu0_3_block_final.json`
shows data/config/batch/worker/crop/pair checks passing, but launch blocked by
GPU0 utilization `25% > 20%` and GPU3 utilization `21% > 20%`. A later live
check saw GPU0/GPU3 utilization `69%/27%`. Do not launch a formal smoke until
GPU0 and GPU3 are both under the live utilization gate, unless the user
explicitly approves the shared warning path and the card load is visibly light.

There is one explicit exception for a bounded shared-GPU smoke:
`launch_deshake_ddp_shared_smoke_once.sh` may run only with
`TWODGAUSS_GPU_UTILIZATION_POLICY=warn`. The current calibrated shared-smoke
default allows per-rank `batch_size<=12`, global `batch_size<=24`,
`dataset.num_workers<=0`, 512px train/val crop, short step/wall-clock caps,
high free-memory gates, and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`. This
exception is for memory-capped smoke validation only; it does not waive real
data/config readiness, and it is not the default formal launch path. For
A_dsh13, prefer the lower conservative shared setting used in the dry-run:
per-rank `batch_size=2`, global `4`, `num_workers=0`,
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.20`.

The A_dsh13 prepared smoke launcher remains the conservative first-smoke
artifact. A_dsh14 is the first completed controlled shared training run on the
synthesized paired-frame subset.

The launch default is intentionally conservative for the shared server:
`batch_size` is interpreted as per-rank batch size and is capped at `2`, so the
two-GPU effective global batch is capped at `4`. Any larger value must be
treated as a new resource-calibration decision, not a routine launch tweak.

The separate GPU0-only large-batch smoke path is explicit and does not change
the DDP default. It defaults to conservative `batch_size=8`, but `A_dsh08`
calibrated an explicit physical-GPU0 `batch_size=24` selftest with
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`. The launcher still refuses incomplete
data by default and requires explicit cap overrides for larger GPU0-only smoke
runs.

The launch also requires the dataset files to be stable by default: any file in
the scanned data root modified within the last `600` seconds is treated as
evidence of an active or recently completed transfer and blocks launch.

## Selftest Evidence

The `preflight_selftest` fixture is intentionally tiny and exists only to prove
that the GPU0+GPU3 preflight/launcher path can pass under controlled dry-run
conditions. The dry-run selftest passed with:

- physical GPUs `0,3`
- per-rank `batch_size=2`, global `batch_size=4`
- `train_pairs=2`, `val_pairs=1`
- one `.y4m` video evidence file
- `TWODGAUSS_DRY_RUN=1`, so no tmux training session was created

Do not use the selftest fixture as real training data.

The paired-frame preparation selftest also passed:

- `scripts/prepare_deshake_paired_frames.py` rejects same-file input/target
  video pairs;
- plan-only mode for the current two Netflix Y4M files writes no files;
- tiny explicit input/target Y4M fixtures are converted into paired PNG frames
  under `preflight_selftest/data/generated_from_y4m`;
- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh04_y4m_prepared_ddp_smoke_selftest_generated.toml`
  passes preflight and launcher/smoke dry-run without creating tmux.
- the prepare-and-queue wrapper selftest generated
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh05_prepared_queue_ddp_smoke_selftest_generated.toml`,
  reached the queued DDP smoke wrapper in dry-run, and did not create tmux.

The explicit shared-GPU DDP smoke selftest also passed on physical GPU0+GPU3:

- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh06_shared_ddp_bs1_smoke_selftest_generated.toml`;
- real one-step DDP run, not dry-run, with per-rank `batch_size=1`, global
  `batch_size=2`, `dataset.num_workers=0`, and
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.12`;
- train log
  `experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_ddp_20260606_004314.log`;
- exit code `0`, no project tmux session or training process left afterward,
  and no GPU1/GPU2 visibility.

The calibrated shared-GPU DDP smoke selftest then passed at the larger default
shared smoke setting:

- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh10_shared_ddp_bs12_rankmem_selftest_generated.toml`;
- real one-step DDP run, not dry-run, with per-rank `batch_size=12`, global
  `batch_size=24`, `dataset.num_workers=0`, 512px train/val crop, and
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`;
- rank metrics under
  `experiments/V1/runs/A_dsh10_shared_ddp_bs12_rankmem_selftest/metrics/rank_metrics/`
  map rank0 to physical GPU0 and rank1 to physical GPU3;
- both ranks recorded `cuda_max_memory_reserved_gb=18.6015625`;
- exit code `0`, no project tmux session or training process left afterward,
  and no GPU1/GPU2 visibility.

The prepare-and-queue shared-GPU selftest also passed in dry-run mode:

- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh07_prepared_shared_queue_ddp_smoke_selftest_generated.toml`;
- generated paired PNG frames under the selftest data root only;
- enforced shared caps: per-rank `batch_size=1`, global `batch_size=2`,
  and `dataset.num_workers=0`;
- reached the queued shared smoke wrapper in dry-run with
  `status=dry_run_invoked`.

The calibrated prepare-and-queue shared-GPU selftest also passed:

- generated one train pair and one val pair only under
  `preflight_selftest/data/generated_from_shared_bs12_queue`;
- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh11_prepared_shared_ddp_bs12_smoke_selftest_generated.toml`;
- config uses per-rank `batch_size=12`, global `24`, `dataset.num_workers=0`,
  and 512px crop;
- reached queued shared smoke dry-run with `status=dry_run_invoked` and did
  not create tmux.

The explicit GPU0-only large-batch selftest was then expanded after the user
approved direct GPU0 large-batch probing:

- generated config
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh08_gpu0_bs24_smoke_selftest_generated.toml`;
- physical GPU0 only, `batch_size=24`, global `24`, `dataset.num_workers=0`,
  `max_steps=1`;
- `TWODGAUSS_CUDA_MEMORY_FRACTION=0.75` intentionally hit the process cap
  before completing, with the project process around `35.57GiB`;
- `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85` completed one train step plus validation,
  exit code `0`;
- train metrics recorded `cuda_max_memory_allocated_gb=36.588324546813965`
  and `cuda_max_memory_reserved_gb=37.189453125`;
- after exit, GPU0 returned to the pre-existing external-process baseline and
  no project tmux/process residue remained.

This `A_dsh08` evidence is a GPU0 selftest calibration result, not a real
`/home/ldj/wxc/4K_8K` training launch.

## Readiness Check

Use the readiness checker before any short DDP smoke or formal launch:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/deshake_ddp_config.toml \
bash experiments/V1/runs/deshake_ddp_0_3_pending/check_deshake_ddp_readiness.sh
```

It writes dataset/preflight/summary JSON files under
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/` and returns nonzero until
the preflight gate is clean.

Latest readiness evidence:

- Positive selftest readiness summary:
  `experiments/V1/runs/deshake_ddp_0_3_pending/metrics/readiness_summary_20260605_224242_3327268.json`
  returned `ready_for_short_ddp_smoke=true`.
- Real `/home/ldj/wxc/4K_8K` readiness summary:
  `experiments/V1/runs/deshake_ddp_0_3_pending/metrics/readiness_summary_20260606_002408_3920611.json`
  returned `ready_for_short_ddp_smoke=false` and preflight status `2`.

## Readiness Monitor

The monitor is read-only and never launches training:

```bash
TWODGAUSS_DESHAKE_MONITOR_MAX_CHECKS=12 \
TWODGAUSS_DESHAKE_MONITOR_INTERVAL_SECONDS=300 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/monitor_deshake_ddp_readiness.sh
```

It appends one JSONL record per readiness attempt under
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/` and exits nonzero unless
the latest readiness attempt is clean. The real-data one-shot monitor returned
not-ready with artifact
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/readiness_monitor_summary_20260605_225529_3402794.json`.
The positive selftest monitor returned ready with artifact
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/readiness_monitor_summary_20260605_225606_3406534.json`.

## Paired-Frame Preparation

Use `scripts/prepare_deshake_paired_frames.py` only when real shaky/input and
stable/target video pairs are explicitly known. It is not an unsupervised
dataset generator and it refuses same-file input/target pairs.

The script defaults to plan-only mode. It writes PNG frames only with `--write`
and requires the output root to stay under the declared deshaking data root so
the downstream DDP preflight can verify path containment.

Example:

```bash
/home/ldj/wxc/.venvs/2dgauss-py313/bin/python \
  scripts/prepare_deshake_paired_frames.py \
  --data-root /path/to/deshake_dataset \
  --output-root /path/to/deshake_dataset/generated_pairs/smoke \
  --train-pair /path/to/deshake_dataset/train_shaky.y4m:/path/to/deshake_dataset/train_stable.y4m \
  --val-pair /path/to/deshake_dataset/val_shaky.y4m:/path/to/deshake_dataset/val_stable.y4m \
  --train-frames-per-pair 16 \
  --val-frames-per-pair 4 \
  --stride 60 \
  --write
```

## Prepare-And-Queue Wrapper

Use `prepare_and_queue_deshake_ddp_smoke.sh` only when explicit input/target
video pairs are known and a short DDP smoke should be prepared from them. The
wrapper first calls `scripts/prepare_deshake_paired_frames.py`, then
`scripts/create_deshake_ddp_config.py`, then the queued DDP smoke wrapper.

Default behavior is plan-only and does not write frames:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_TRAIN_PAIR=/path/to/train_shaky.y4m:/path/to/train_stable.y4m \
TWODGAUSS_DESHAKE_VAL_PAIR=/path/to/val_shaky.y4m:/path/to/val_stable.y4m \
TWODGAUSS_DRY_RUN=1 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_smoke.sh
```

Writing frames requires both:

- `TWODGAUSS_DESHAKE_WRITE_FRAMES=1`
- `TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=prepare_explicit_deshake_pairs`

Validated behavior:

- missing explicit train/val video pairs are refused;
- plan-only mode against the current real `/home/ldj/wxc/4K_8K` writes no
  `/home/ldj/wxc/4K_8K/generated_pairs`;
- the tiny Y4M selftest writes paired PNG frames, generates `A_dsh05`, reaches
  the queued DDP smoke dry-run, and creates no tmux training session.

## Prepare-And-Queue Shared Wrapper

Use `prepare_and_queue_deshake_ddp_shared_smoke.sh` only when explicit
input/target video pairs are known and the target smoke should use the
memory-capped shared GPU0+GPU3 path. The wrapper first calls
`scripts/prepare_deshake_paired_frames.py`, then generates a shared-safe config,
then invokes `wait_and_launch_deshake_ddp_shared_smoke.sh`.

Default behavior is plan-only and does not write frames:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_TRAIN_PAIR=/path/to/train_shaky.y4m:/path/to/train_stable.y4m \
TWODGAUSS_DESHAKE_VAL_PAIR=/path/to/val_shaky.y4m:/path/to/val_stable.y4m \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_GPU_UTILIZATION_POLICY=warn \
TWODGAUSS_CUDA_MEMORY_FRACTION=0.50 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_shared_smoke.sh
```

Writing frames requires both:

- `TWODGAUSS_DESHAKE_WRITE_FRAMES=1`
- `TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=prepare_explicit_deshake_pairs`

The shared prepare wrapper generates the calibrated shared smoke config by
default: per-rank `batch_size=12`, global `24`, `dataset.num_workers=0`, 512px
crop, and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`, while still allowing explicit
lower batch caps. The generated `A_dsh07` selftest validated the queue flow at
bs1, `A_dsh10` validates the current resource-calibrated default, and `A_dsh11`
validates the calibrated prepare+queue generation path. Plan-only mode against
the current real `/home/ldj/wxc/4K_8K` wrote no
`/home/ldj/wxc/4K_8K/generated_pairs`.

## Config Generation

Use `scripts/create_deshake_ddp_config.py` only after the real dataset has
extracted paired frame directories, for example train/val `input` and `target`
directories under the deshaking data root. The generator refuses raw Y4M-only
layouts and refuses paths outside the declared data root.

The generated selftest config
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh02_ddp_smoke_selftest_generated.toml`
exists only as a fixture proving the generator/preflight/launcher dry-run path.
It is not real training data.

For the explicit GPU0-only large-batch smoke path, pass
`--single-gpu0-smoke`. The generated selftest config
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh03_gpu0_bs8_smoke_selftest_generated.toml`
uses `batch_size=8` and is also only a fixture. The later generated
`A_dsh08` fixture uses `batch_size=24` for GPU0 resource calibration only.

## Smoke Launch Wrapper

Use `launch_deshake_ddp_smoke_once.sh` for the first real DDP smoke after
readiness turns clean. This wrapper:

- runs the readiness checker first and refuses to continue on any issue;
- enforces a short-smoke config gate (`max_steps <= 50`,
  `wall_clock_minutes <= 30`, per-rank `batch_size <= 2`);
- defaults to `TWODGAUSS_DRY_RUN=1`;
- requires `TWODGAUSS_DESHAKE_CONFIRM_SMOKE=launch_gpu0_3_deshake_smoke` for a
  non-dry-run smoke launch.

Validated behavior:

- real incomplete `/home/ldj/wxc/4K_8K` is refused at readiness;
- generated selftest smoke config reaches launcher dry-run without creating
  tmux;
- a selftest config above the smoke step cap is refused by the config gate.

## Queued DDP Smoke Wrapper

Use `wait_and_launch_deshake_ddp_smoke.sh` only when it is acceptable to poll
for a readiness window and then invoke the existing DDP smoke wrapper. It does
not bypass readiness or smoke caps.

Default behavior:

- requires `TWODGAUSS_DESHAKE_DATA_ROOT` and `TWODGAUSS_DESHAKE_CONFIG`;
- defaults to `TWODGAUSS_DRY_RUN=1`;
- checks readiness up to `TWODGAUSS_DESHAKE_QUEUE_MAX_CHECKS=12` times;
- sleeps `TWODGAUSS_DESHAKE_QUEUE_INTERVAL_SECONDS=300` seconds between checks;
- writes queue JSONL and summary artifacts under `logs/`;
- for non-dry-run, requires
  `TWODGAUSS_DESHAKE_CONFIRM_QUEUE=wait_and_launch_gpu0_3_deshake_smoke` in
  addition to the downstream smoke confirmation.

Validated behavior:

- current real `/home/ldj/wxc/4K_8K` returns `not_ready` after one check and
  does not call the smoke wrapper;
- generated selftest DDP config reaches the downstream smoke wrapper in
  dry-run only and writes `status=dry_run_invoked`;
- no tmux training session is created by the queued dry-run selftest.

## Explicit Shared-GPU DDP Smoke Wrapper

Use `launch_deshake_ddp_shared_smoke_once.sh` only when a calibrated GPU0+GPU3 DDP
smoke is intentionally allowed to share cards with existing external compute
processes. This wrapper is stricter than the default smoke wrapper on memory
and work size, while allowing the utilization signal to be recorded as a
warning instead of a launch blocker.

Default behavior:

- requires `TWODGAUSS_DESHAKE_DATA_ROOT` and `TWODGAUSS_DESHAKE_CONFIG`;
- defaults to `TWODGAUSS_DRY_RUN=1`;
- requires `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`;
- caps per-rank `batch_size` at `12`, global `batch_size` at `24`,
  `dataset.num_workers` at `0`, and train/val crop at `512`;
- defaults to `TWODGAUSS_MIN_FREE_MIB=32000`;
- exports `TWODGAUSS_CUDA_MEMORY_FRACTION`, default `0.50`, into the training
  process;
- for non-dry-run, requires
  `TWODGAUSS_DESHAKE_CONFIRM_SHARED_SMOKE=launch_shared_gpu0_3_deshake_ddp_smoke`.

Validated behavior:

- dry-run selftest passed with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.12` while
  recording high GPU utilization as warnings;
- real one-step selftest `A_dsh10_shared_ddp_bs12_rankmem_selftest` completed
  with exit code `0`, `cuda_memory_fraction=0.50`, per-rank `batch_size=12`,
  global `24`, rank0 on physical GPU0, rank1 on physical GPU3, and peak
  reserved memory about `18.60GiB` on both ranks;
- current real `/home/ldj/wxc/4K_8K` with the incomplete `A_dsh01` config is
  still refused before launch because the config lacks smoke-safe batch/step
  fields and paired train/val paths.

## Queued Shared-GPU DDP Smoke Wrapper

Use `wait_and_launch_deshake_ddp_shared_smoke.sh` when the operator wants the
memory-capped shared DDP smoke to start only after readiness becomes clean. It
uses the same explicit shared policy as `launch_deshake_ddp_shared_smoke_once.sh`
and does not bypass data/config gates.

Default behavior:

- requires `TWODGAUSS_DESHAKE_DATA_ROOT` and `TWODGAUSS_DESHAKE_CONFIG`;
- defaults to `TWODGAUSS_DRY_RUN=1`;
- requires `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`;
- polls readiness up to `TWODGAUSS_DESHAKE_SHARED_QUEUE_MAX_CHECKS=12` times;
- sleeps `TWODGAUSS_DESHAKE_SHARED_QUEUE_INTERVAL_SECONDS=300` seconds between
  checks;
- carries shared caps through readiness and launch: per-rank `batch_size<=12`,
  global `batch_size<=24`, `dataset.num_workers<=0`, crop `<=512`, and
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`;
- for non-dry-run, requires
  `TWODGAUSS_DESHAKE_CONFIRM_SHARED_QUEUE=wait_and_launch_shared_gpu0_3_deshake_ddp_smoke`
  plus the downstream shared-smoke confirmation.

Validated behavior:

- generated `A_dsh06` selftest reaches the shared smoke wrapper in dry-run and
  writes `status=dry_run_invoked`;
- current real `/home/ldj/wxc/4K_8K` plus incomplete `A_dsh01` returns
  `not_ready` at readiness and does not call the shared smoke wrapper.

## GPU0-Only Smoke Wrapper

Use `launch_deshake_gpu0_smoke_once.sh` only for an explicit GPU0-only
large-batch smoke after the real dataset/config pass preflight. This wrapper:

- checks physical GPU0 only;
- defaults to smoke `batch_size<=8`, `max_steps<=50`, and
  `wall_clock_minutes<=30`, with larger caps requiring explicit environment
  overrides and fresh calibration evidence;
- defaults to `TWODGAUSS_DRY_RUN=1`;
- forwards `TWODGAUSS_CUDA_MEMORY_FRACTION` into the training process when set;
- requires `TWODGAUSS_DESHAKE_CONFIRM_GPU0_SMOKE=launch_gpu0_deshake_bs8_smoke`
  for a non-dry-run launch.

Validated behavior:

- current real `/home/ldj/wxc/4K_8K` is refused because GPU0 utilization is
  `99%`, data is only about `53G`, and the real config lacks paired train/val
  frame paths;
- generated GPU0 selftest config reaches launcher dry-run without creating
  tmux;
- generated `A_dsh08` GPU0 selftest completed one real `batch_size=24` step
  with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`, peak reserved memory about
  `37.19GiB`, and no project residue afterward;
- the original GPU0+GPU3 DDP launcher still remains capped at per-rank
  `batch_size=2`, global `4`.
