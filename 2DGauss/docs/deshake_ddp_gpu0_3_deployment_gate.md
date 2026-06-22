# UHD Video Deshaking DDP GPU0/3 Deployment Gate

This gate exists to prevent the corrected UHD video deshaking task from being
accidentally launched with the old `UHD_haze` dehazing package.

## Current Boundary

- Current task scope: UHD video deshaking / stabilization.
- Current first synthesized smoke dataset:
  `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- Current first smoke config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh13_bvi_aom_synth_v1_shared_ddp_smoke.toml`.
- Current first controlled shared training config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh14_bvi_aom_synth_v1_shared_ddp_train.toml`.
- Current controlled full-train config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh15_bvi_aom_synth_v1_gpu3_full_train.toml`.
- Current context-guided completed config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train.toml`.
- Allowed physical GPUs: `0,3`.
- Disallowed GPUs: `1,2`.
- Default DDP memory gate: both GPU0 and GPU3 must have at least `30000MiB`
  free before launch.
- Default DDP utilization gate: both GPU0 and GPU3 must be at or below `20%`
  utilization before launch.
- Default utilization policy: `block`. High utilization is not ignored unless
  the explicit shared-smoke wrapper below is used.
- Default batch gate: config `[training].batch_size` is per rank and must be
  `<=2`; with GPU0+GPU3 the effective global batch must be `<=4`.
- Explicit shared-GPU DDP smoke gate: only
  `launch_deshake_ddp_shared_smoke_once.sh` may set
  `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`. The current calibrated shared-smoke
  cap is per-rank `batch_size<=12`, global `batch_size<=24`,
  `num_workers<=0`, `train_crop_size/val_crop_size<=512`, short step/time
  limits, and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50` by default.
- For A_dsh13, the conservative first real-data smoke config is smaller than
  the calibrated cap: per-rank `batch_size=2`, global `4`, `num_workers=0`,
  512px crop, `max_steps=5`, and the tested dry-run used
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.20`.
- For A_dsh14, the first controlled real shared training run used the explicit
  shared warning path after A_dsh10 calibration evidence: per-rank
  `batch_size=12`, global `24`, `num_workers=0`, 512px crop, `max_steps=60`,
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`, live free-memory gate `18000MiB`, and
  runtime own-session stop threshold `8192MiB`.
- For A_dsh15, the current full-train baseline used physical GPU3 only because
  GPU0 was occupied by external processes at launch time. This path is explicit:
  `launch_deshake_gpu3_train_once.sh`, with `CUDA_VISIBLE_DEVICES=3`,
  `batch_size=24` / global `24`, `num_workers=0`, 512px crop,
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`, and full 32-batch validation.
- For A_dsh17, the completed context-guided comparison run used physical GPU3
  only after A_dsh16 OOM at `batch_size=4`. A_dsh17 uses
  `launch_deshake_gpu3_context_guided_train_once.sh`, `CUDA_VISIBLE_DEVICES=3`,
  `batch_size=2` / global `2`, `train_context_size=728`,
  `train_supervision_crop_size=512`, `num_workers=0`,
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`, and full 32-batch validation. It is
  recorded as safe and completed, but not promoted over A_dsh15.
- Explicit GPU0-only smoke gate: the wrapper defaults to conservative
  `batch_size<=8`, but larger GPU0-only smoke caps may be enabled by explicit
  environment override after a logged calibration. Current calibration evidence:
  `A_dsh08_gpu0_bs24_smoke_selftest` completed one physical-GPU0 step at
  `batch_size=24` with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`.
- Default worker gate: config `[dataset].num_workers` must be `<=2` per rank.
- Default transfer-stability gate: scanned dataset files must not have been
  modified within the last `600` seconds.
- Existing `A_b02_ddp_0_3_bs12` packages are `UHD_haze` resource calibration
  artifacts only. They must not be counted as deshaking training.

## Required Evidence Before Launch

- A deshaking-specific dataset root exists on this server.
- The dataset root is not the old `UHD_haze` patch cache and not the small
  `/home/ldj/wxc/任务四` NASA image-text dataset.
- The dataset has enough size for the expected migrated video corpus, with
  default minimum `300GiB` unless an explicitly logged subset policy replaces it.
- The dataset contains real video files, or an explicitly logged extracted-frame
  layout with enough frame files and paired/stabilized supervision. Raw Y4M
  files are treated as video evidence.
- Raw Y4M files alone are not treated as supervised training pairs. If the
  source provides shaky/input and stable/target videos, first create an explicit
  paired-frame view with `scripts/prepare_deshake_paired_frames.py`.
- A deshaking-specific config exists and contains deshake/stabilization/video
  task intent.
- The config does not contain `UHD_haze`, dehaze, or haze task identifiers.
- The config's `train_input`, `train_target`, `val_input`, and `val_target`
  directories exist, are under the declared dataset root, and contain paired
  frame samples.
- The config's per-rank `batch_size` is small enough for shared GPU0/GPU3 use.
  The default launch cap is `2` per rank / `4` global. Larger values require an
  explicit, logged smoke result and must be enabled by changing the launch env.
- The explicit shared-GPU path has such a logged calibration: `A_dsh10` uses
  per-rank `batch_size=12`, global `24`, `num_workers=0`, 512px train/val crop,
  and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`. Do not apply that cap to larger
  crop configs without a new calibration.
- For the GPU0-only large-batch path, use the deshaking-specific calibration
  result rather than old `UHD_haze` resource-calibration values. The current
  validated deshaking selftest point is `batch_size=24` on physical GPU0 with
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`; real data/config readiness is still
  required before applying it to `/home/ldj/wxc/4K_8K`.
- GPU0 and GPU3 pass the live free-memory and utilization gates immediately
  before launch. A card with high compute utilization must be treated as busy
  even if it still has free memory.
- The dataset transfer has visibly settled. The default launcher refuses data
  roots with files modified in the last 10 minutes, which prevents reading
  partially copied 4K/8K video files.

## Current A_dsh13 Evidence

- `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1/manifest.json` records
  `clip_count=24`, `pass_count=18`, `fail_count=6`; failed clips were excluded
  by `trajectory_discontinuity_detected`.
- Train/val paired views are verified `.npy` frames: `112` train pairs and
  `32` val pairs.
- `scripts/inspect_deshake_dataset.py` and `PairedImageDataset` both recognize
  the synthesized `.npy` frame layout.
- CPU preflight passed on A_dsh13 with `(2,3,512,512)` batches and 8 val
  batches.
- Queued shared DDP smoke dry-run reached the shared wrapper:
  `experiments/V1/runs/deshake_ddp_0_3_pending/logs/queued_shared_ddp_smoke_summary_20260606_124629_2789952.json`
  has `status=dry_run_invoked`.
- Formal block-policy launch is still held only by live GPU utilization:
  `experiments/V1/runs/deshake_ddp_0_3_pending/metrics/a_dsh13_preflight_gpu0_3_block_final.json`
  has data/config/pair/batch/crop checks passing, but GPU0 `25% > 20%` and
  GPU3 `21% > 20%`. A later live check saw `69%/27%`, so no non-dry-run was
  launched.

## Current A_dsh14 Evidence

- A_dsh14 uses the same verified BVI-AOM synthetic paired `.npy` view as
  A_dsh13: `112` train pairs and `32` val pairs under
  `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- Config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh14_bvi_aom_synth_v1_shared_ddp_train.toml`.
- Launch path: explicit shared wrapper
  `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_shared_smoke_once.sh`
  with `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`,
  `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`, per-rank `batch_size=12`, global `24`,
  `num_workers=0`, 512px crop, and runtime headroom monitoring.
- Training completed with exit code `0` in
  `experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_ddp_20260606_125827.log`.
  The run directory is
  `experiments/V1/runs/A_dsh14_bvi_aom_synth_v1_shared_ddp_train`.
- Metrics: `60` train steps, `6` validation records, final validation
  `val_loss=0.0661571198143065`, `val_psnr=20.984133518913794`.
- Rank mapping and memory audit: rank0 mapped to physical GPU0 with peak
  reserved `18.91796875GiB`; rank1 mapped to physical GPU3 with peak reserved
  `18.875GiB`; no GPU1/2 use was recorded.
- Post-run cleanup found no project `train_v1_t01.py`, `torchrun`, DDP tmux,
  deshake launcher, or synthesis process residue, and no external process was
  killed.

## Current A_dsh15 Evidence

- A_dsh15 uses the same verified BVI-AOM synthetic paired `.npy` view as
  A_dsh14: `112` train pairs and `32` val pairs under
  `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- Config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh15_bvi_aom_synth_v1_gpu3_full_train.toml`.
- Launch path:
  `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_train_once.sh`
  with physical GPU3 only, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`,
  `batch_size=24`, global `24`, `num_workers=0`, 512px crop, and runtime
  headroom monitoring.
- GPU rationale: GPU0 was busy with external processes at launch; GPU1/2 are
  disallowed. GPU3 preflight passed with `48520MiB` free and `0%` utilization.
- Training completed with exit code `0` in
  `experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_161547_120913.log`.
  The run directory is
  `experiments/V1/runs/A_dsh15_bvi_aom_synth_v1_gpu3_full_train`.
- Metrics: `1200` train steps, `241` train metric records, `24` full-validation
  records, best validation loss at step `1000`
  (`val_loss=0.09517641738057137`, `val_psnr=18.49445972137987`), and final
  validation at step `1200`
  (`val_loss=0.09540756104979664`, `val_psnr=18.504268557322963`).
- Independent final evaluation improves over the shaky input baseline on the
  32-pair val view from `16.507102228205603` PSNR / `0.37895671324804425` SSIM
  to `18.504268639458388` PSNR / `0.5011140927672386` SSIM.
- Post-run cleanup found no project training residue, no GPU1/2 use, and GPU3
  returned to `0MiB` used.

## Current A_dsh17 Evidence

- A_dsh16 first attempted the context-guided branch at `batch_size=4` on
  physical GPU3 only. It passed live preflight but failed quickly with
  project-local CUDA OOM and produced no checkpoints or metrics.
- A_dsh17 lowered the context-guided branch to `batch_size=2` while keeping the
  same verified BVI-AOM synthetic paired `.npy` view: `112` train pairs and `32`
  val pairs under `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- Config:
  `experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train.toml`.
- Launch path:
  `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu3_context_guided_train_once.sh`
  with physical GPU3 only. A first dry-run correctly blocked when a new GPU3
  process appeared; the real launch happened only after GPU3 returned to idle.
- Training completed with exit code `0` in
  `experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_173118_784237.log`.
  The run directory is
  `experiments/V1/runs/A_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train`.
- Metrics: `1600` train steps, `321` train metric records, `32` full-validation
  records, best validation loss at step `1350`
  (`val_loss=0.09807008923962712`, `val_psnr=18.343223704028027`), and final
  validation at step `1600`
  (`val_loss=0.09814217837993056`, `val_psnr=18.34113527708859`).
- Independent best-val evaluation improves over the shaky input baseline, but
  reaches only `18.343223705783423` PSNR / `0.4939711121842265` SSIM, below
  A_dsh15 final's `18.504268639458388` PSNR / `0.5011140927672386` SSIM.
- Post-run cleanup found no project training residue, no GPU1/2 use, GPU0 was
  avoided, and GPU3 returned to `0MiB` used.

## Command

```bash
/home/ldj/wxc/.venvs/2dgauss-py313/bin/python \
  scripts/preflight_deshake_ddp_gpu0_3.py \
  --data-root /path/to/deshake_dataset \
  --config /path/to/deshake_ddp_config.toml \
  --physical-gpus 0,3 \
  --min-free-mib 30000 \
  --max-gpu-utilization 20 \
  --max-per-rank-batch-size 2 \
  --max-global-batch-size 4 \
  --max-num-workers 2 \
  --min-file-age-seconds 600 \
  --json
```

The command exits `0` only when the data, config, and GPU evidence all satisfy
the gate. A nonzero exit means do not launch distributed training.

## Explicit Shared-GPU DDP Smoke Command

This path is for a memory-capped short smoke while GPU0/GPU3 are shared. It
does not bypass dataset/config gates:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/shared_bs12_smoke_config.toml \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_GPU_UTILIZATION_POLICY=warn \
TWODGAUSS_CUDA_MEMORY_FRACTION=0.50 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_shared_smoke_once.sh
```

For a real non-dry-run shared smoke, the launcher additionally requires:

```bash
TWODGAUSS_DESHAKE_CONFIRM_SHARED_SMOKE=launch_shared_gpu0_3_deshake_ddp_smoke
```

Validated behavior: `A_dsh10_shared_ddp_bs12_rankmem_selftest` completed a real
one-step DDP run on physical GPU0+GPU3 with per-rank `batch_size=12`, global
`24`, `num_workers=0`, 512px train/val crop,
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`, and exit code `0`. Per-rank metrics map
rank0 to physical GPU0 and rank1 to physical GPU3, with both ranks recording
peak reserved memory about `18.60GiB`. The same wrapper still refuses incomplete
real `/home/ldj/wxc/4K_8K` configs before launch.

## GPU0-Only Smoke Command

This is a separate smoke path, not a DDP default:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/gpu0_bs8_smoke_config.toml \
TWODGAUSS_DRY_RUN=1 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu0_smoke_once.sh
```

For a real non-dry-run smoke, the launcher additionally requires:

```bash
TWODGAUSS_DESHAKE_CONFIRM_GPU0_SMOKE=launch_gpu0_deshake_bs8_smoke
```

The wrapper checks only physical GPU0, caps `batch_size` at `8`, caps
`max_steps` at `50`, caps `wall_clock_minutes` at `30`, and still rejects high
GPU0 utilization, recent/incomplete data, missing paired train/val frame paths,
and old-task config evidence.

For an explicitly approved direct-GPU0 large-batch smoke, raise the launcher
caps and set the memory fraction:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/gpu0_bs24_smoke_config.toml \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_GPU0_SMOKE_MAX_BATCH_SIZE=24 \
TWODGAUSS_GPU0_SMOKE_MAX_GLOBAL_BATCH_SIZE=24 \
TWODGAUSS_GPU0_SMOKE_MAX_NUM_WORKERS=0 \
TWODGAUSS_CUDA_MEMORY_FRACTION=0.85 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu0_smoke_once.sh
```

Validated behavior: `A_dsh08_gpu0_bs24_smoke_selftest` first hit the intended
process cap at `TWODGAUSS_CUDA_MEMORY_FRACTION=0.75`; rerunning at `0.85`
completed one train step plus validation with exit code `0`. Metrics recorded
`cuda_max_memory_allocated_gb=36.588324546813965` and
`cuda_max_memory_reserved_gb=37.189453125`. This is GPU0 selftest calibration,
not a real `/home/ldj/wxc/4K_8K` training launch.

## Readiness Helpers

- `experiments/V1/runs/deshake_ddp_0_3_pending/check_deshake_ddp_readiness.sh`
  performs one read-only readiness check and writes dataset/preflight/summary
  JSON artifacts.
- `experiments/V1/runs/deshake_ddp_0_3_pending/monitor_deshake_ddp_readiness.sh`
  repeats that readiness check and writes JSONL attempt records; it never
  launches training and exits nonzero while the latest attempt is not ready.
- `scripts/create_deshake_ddp_config.py` creates a conservative paired-frame
  smoke config only after train/val input-target frame directories already
  exist under the declared deshaking data root. It refuses raw Y4M-only
  incomplete layouts. Pass `--single-gpu0-smoke` only for the explicit GPU0
  large-batch smoke config.
- `scripts/prepare_deshake_paired_frames.py` prepares a paired-frame smoke view
  only from explicit shaky/input and stable/target video pairs. It defaults to
  plan-only mode, writes frames only with `--write`, keeps output under the
  declared data root, refuses same-file input/target pairs, and is not an
  unsupervised dataset generator.
- `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_smoke_once.sh`
  is the first-smoke wrapper. It runs readiness first, enforces short-smoke caps
  (`max_steps <= 50`, `wall_clock_minutes <= 30`, per-rank `batch_size <= 2`),
  defaults to dry-run, and requires
  `TWODGAUSS_DESHAKE_CONFIRM_SMOKE=launch_gpu0_3_deshake_smoke` for a non-dry-run
  smoke.
- `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_ddp_shared_smoke_once.sh`
  is the explicit shared-GPU smoke wrapper. It allows high utilization only as
  warnings via `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`, caps the calibrated run
  at per-rank `batch_size<=12`, global `batch_size<=24`, `num_workers<=0`,
  512px train/val crop, and uses `TWODGAUSS_CUDA_MEMORY_FRACTION` to limit each
  PyTorch process's allocator.
- `experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_shared_smoke.sh`
  polls readiness for the explicit shared-GPU smoke path, then invokes
  `launch_deshake_ddp_shared_smoke_once.sh` only after readiness passes. It
  defaults to dry-run and requires
  `TWODGAUSS_DESHAKE_CONFIRM_SHARED_QUEUE=wait_and_launch_shared_gpu0_3_deshake_ddp_smoke`
  for non-dry-run queue usage.
- `experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_smoke.sh`
  polls readiness for a GPU0+GPU3 DDP smoke window, then invokes the existing
  smoke wrapper only after readiness passes. It defaults to dry-run and, for
  non-dry-run, requires
  `TWODGAUSS_DESHAKE_CONFIRM_QUEUE=wait_and_launch_gpu0_3_deshake_smoke` in
  addition to the downstream smoke confirmation.
- `experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_smoke.sh`
  first prepares paired frames from explicit input/target video pairs, then
  generates a conservative DDP smoke config, then invokes the queued DDP smoke
  wrapper. It defaults to plan-only and requires
  `TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=prepare_explicit_deshake_pairs` before
  writing frames.
- `experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_shared_smoke.sh`
  first prepares paired frames from explicit input/target video pairs, then
  generates a shared-safe config, then invokes the queued shared-GPU DDP smoke
  wrapper. It defaults to plan-only, requires `TWODGAUSS_GPU_UTILIZATION_POLICY=warn`,
  and keeps the generated config inside the calibrated shared cap: per-rank
  `batch_size<=12`, global `<=24`, `dataset.num_workers=0`, and 512px crop.
- `experiments/V1/runs/deshake_ddp_0_3_pending/launch_deshake_gpu0_smoke_once.sh`
  is the explicit GPU0-only large-batch smoke wrapper. It defaults to dry-run
  and requires `TWODGAUSS_DESHAKE_CONFIRM_GPU0_SMOKE=launch_gpu0_deshake_bs8_smoke`
  for a non-dry-run launch.

## Queued DDP Smoke Command

This wrapper is useful when the real dataset/config are expected to become
ready and the operator wants a guarded readiness poll rather than a manual loop:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/deshake_ddp_config.toml \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_DESHAKE_QUEUE_MAX_CHECKS=12 \
TWODGAUSS_DESHAKE_QUEUE_INTERVAL_SECONDS=300 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_smoke.sh
```

It writes queued-attempt JSONL and summary artifacts under
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/`. A dry-run success is
reported as `status=dry_run_invoked`; it is not a real training launch.

## Queued Shared-GPU DDP Smoke Command

This wrapper is the queued form of the memory-capped shared smoke:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_CONFIG=/path/to/shared_bs12_smoke_config.toml \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_GPU_UTILIZATION_POLICY=warn \
TWODGAUSS_CUDA_MEMORY_FRACTION=0.50 \
TWODGAUSS_DESHAKE_SHARED_QUEUE_MAX_CHECKS=12 \
TWODGAUSS_DESHAKE_SHARED_QUEUE_INTERVAL_SECONDS=300 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/wait_and_launch_deshake_ddp_shared_smoke.sh
```

For a real queued shared smoke, set:

```bash
TWODGAUSS_DESHAKE_CONFIRM_SHARED_QUEUE=wait_and_launch_shared_gpu0_3_deshake_ddp_smoke \
TWODGAUSS_DESHAKE_CONFIRM_SHARED_SMOKE=launch_shared_gpu0_3_deshake_ddp_smoke
```

Validated behavior: the generated `A_dsh10` selftest reaches the shared smoke
wrapper in dry-run with `status=dry_run_invoked` and records the calibrated
caps (`12/24`, crop `512`, fraction `0.50`) in the queued summary; the current
real `/home/ldj/wxc/4K_8K` + incomplete `A_dsh01` config remains not launchable
without verified paired train/val data and a valid config.

## Paired-Frame Preparation Command

Run this only after the real dataset provides explicit input/target video
pairs:

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

The paired-frame selftest generated
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh04_y4m_prepared_ddp_smoke_selftest_generated.toml`
from tiny explicit Y4M fixtures and reached DDP launcher/smoke dry-run without
creating tmux. The current two Netflix Y4M files were not written as paired
training data because their input/target supervision relationship is unverified.

## Prepare-And-Queue Command

This wrapper combines explicit video-pair frame preparation, config generation,
and queued DDP smoke readiness polling. It is plan-only by default:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_TRAIN_PAIR=/path/to/train_shaky.y4m:/path/to/train_stable.y4m \
TWODGAUSS_DESHAKE_VAL_PAIR=/path/to/val_shaky.y4m:/path/to/val_stable.y4m \
TWODGAUSS_DRY_RUN=1 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_smoke.sh
```

For a real frame-writing preparation step, additionally set:

```bash
TWODGAUSS_DESHAKE_WRITE_FRAMES=1 \
TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=prepare_explicit_deshake_pairs
```

The wrapper was validated in three modes: missing video pairs are refused;
plan-only mode on the current real `/home/ldj/wxc/4K_8K` writes no
`generated_pairs`; tiny explicit Y4M fixtures generated paired PNG frames,
created `A_dsh05`, reached queued DDP smoke dry-run with
`status=dry_run_invoked`, and created no tmux training session.

## Prepare-And-Queue Shared Command

This wrapper combines explicit video-pair frame preparation, shared-safe config
generation, and queued shared-GPU DDP smoke readiness polling. It is plan-only
by default:

```bash
TWODGAUSS_DESHAKE_DATA_ROOT=/path/to/deshake_dataset \
TWODGAUSS_DESHAKE_TRAIN_PAIR=/path/to/train_shaky.y4m:/path/to/train_stable.y4m \
TWODGAUSS_DESHAKE_VAL_PAIR=/path/to/val_shaky.y4m:/path/to/val_stable.y4m \
TWODGAUSS_DRY_RUN=1 \
TWODGAUSS_GPU_UTILIZATION_POLICY=warn \
TWODGAUSS_CUDA_MEMORY_FRACTION=0.50 \
bash experiments/V1/runs/deshake_ddp_0_3_pending/prepare_and_queue_deshake_ddp_shared_smoke.sh
```

Writing frames requires `TWODGAUSS_DESHAKE_WRITE_FRAMES=1` and
`TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=prepare_explicit_deshake_pairs`. The
The current generated shared config default is per-rank `batch_size=12`, global
`24`, `num_workers=0`, and 512px crop, capped by the same shared wrapper. The
older `A_dsh07` selftest validated the plan/write/queue flow at bs1; the later
`A_dsh10` run validates the calibrated shared DDP resource point; `A_dsh11`
validates the calibrated prepare+queue generation path. Plan-only validation
against the current real `/home/ldj/wxc/4K_8K` wrote no
`generated_pairs`.

## Current Audit Result

As of the latest live file check at `2026-06-06 00:30:13 CST`, the candidate
incoming root `/home/ldj/wxc/4K_8K` was still actively growing: about `59G`,
two Y4M files, and no verified paired train/val input-target frame view. The
latest readiness check at `2026-06-06 00:24:08 CST` saw both files modified
inside the default 600-second transfer-stability window (`recent_file_count=2`,
latest file age about `0.01s`), and later file mtimes kept advancing through
`00:30:13 CST`.

Latest real readiness artifact:
`experiments/V1/runs/deshake_ddp_0_3_pending/metrics/readiness_summary_20260606_002408_3920611.json`.
It reports `ready_for_short_ddp_smoke=false`: GPU0 utilization was `99% > 20%`,
`bytes_scanned=59597209600` was below the default `300GiB` gate, the root had
recently modified files, the real config still lacked `training.batch_size` plus
`dataset.train_input/train_target` and `dataset.val_input/val_target`, train/val
paired counts were `0`, and no paired frame candidate was found. The safe state
is therefore: DDP, GPU0-only smoke, paired-frame preparation, and prepare+queue
infrastructure prepared; true deshaking launch still pending real
dataset/config/GPU verification.

Additional shared-smoke validation at `2026-06-06 00:43:14 CST`: the generated
fixture config
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh06_shared_ddp_bs1_smoke_selftest_generated.toml`
ran one real DDP training step on physical GPU0+GPU3 and exited `0`. This proves
the memory-capped shared DDP launch path works, but it is selftest-only evidence
and does not make the incomplete real `/home/ldj/wxc/4K_8K` dataset launchable.

Additional queued shared-smoke validation at `2026-06-06 00:56:49 CST`: the
queued shared wrapper wrote
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/queued_shared_ddp_smoke_summary_20260606_005649_4086491.json`
with `status=dry_run_invoked` for the `A_dsh06` selftest, and wrote
`experiments/V1/runs/deshake_ddp_0_3_pending/logs/queued_shared_ddp_smoke_summary_20260606_005649_4086500.json`
with `status=not_ready` for the current real `/home/ldj/wxc/4K_8K`. The latest
real queued readiness saw about `63.6GB`, two Y4M files, no paired train/val
frame paths, and the incomplete `A_dsh01` config missing shared-smoke batch and
paired dataset fields.

Additional GPU0 large-batch calibration at `2026-06-06 01:23:59 CST`: generated
fixture config
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh08_gpu0_bs24_smoke_selftest_generated.toml`
ran on physical GPU0 only. `batch_size=24` with
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.75` hit the intended allocator cap before
completion; the same config with fraction `0.85` completed one train step and
validation with exit code `0`, peak reserved memory about `37.19GiB`, and no
project process residue. This validates the GPU0 large-batch smoke machinery
only; the real `/home/ldj/wxc/4K_8K` dataset still lacks verified paired
train/val supervision and has not been launched.

Additional calibrated shared-GPU DDP validation at `2026-06-06 01:43:11 CST`:
generated fixture config
`experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh10_shared_ddp_bs12_rankmem_selftest_generated.toml`
ran one real DDP step on physical GPU0+GPU3 with per-rank `batch_size=12`,
global `24`, `num_workers=0`, 512px crop, and
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.50`. It exited `0`; rank metrics recorded
`physical_gpu=0` for rank0 and `physical_gpu=3` for rank1, with peak reserved
memory about `18.60GiB` on both ranks. The shared and queued wrappers now
dry-run successfully with these calibrated defaults, but this still does not
make incomplete real data launchable.
