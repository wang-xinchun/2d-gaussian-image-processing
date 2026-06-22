# Run Manifest: V1/T01/A_4kd01

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd01`
- Full identifier: `V1/T01/A_4kd01`

## Current Status

This branch is prepared but not launched. It is the first current-server
training package after returning from the historical deshake side branch to the
old UHD image restoration task.

## Purpose

`A_4kd01` is a short T01 probe for the newly available current-server
`4KDehaze` paired dataset. The goal is to re-open the dehazing line on the
local Baidu Wangpan data copy before deciding whether to scale up, add a patch
cache, or introduce task-aware dehazing losses.

## Dataset

- Dataset: `4KDehaze_local_90_5_5`
- Root data asset: `/home/ldj/wxc/baiduwangpan/2D高斯数据集/4KDehaze/4KDehazing`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5`
- Train pairs: `14045`
- Validation pairs: `780`
- Test pairs: `781`
- Split rule: deterministic filename order, local `90/5/5`.
- Audit document:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/TRAINING_DATASETS_READY.md`

## Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `2000`
- Planned batch size: `16`
- Training crop: `512`
- Validation crop: `512`
- Validation subset for training-time checks: first `100` validation pairs

## Launch State

The run-local config exists at:

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd01/configs/a_4kd01_short_4kdehaze_local_probe.toml`

Formal GPU training has not been launched yet. At preparation time the GPUs
were busy with external utilization, so this package stops at config/data
readiness and gate documentation.
