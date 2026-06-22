# Dataset Gap From 2D Dataset Paper

## Record Time

- Time: `2026-04-24 19:49:40 CST`
- Paper inventory source: `docs/dataset_inventory_from_2d_dataset_paper.md`
- Current local status source: `datasets/status.tsv`

## Paper Dataset Scope

The local paper inventory identifies the following dataset groups:

- UHD low-light: `UHD-LL`
- UHD dehazing: `UHD-Haze`
- UHD deblurring: `UHD-Blur`
- General low-light: `LOL`
- General dehazing: `SOTS-ITS` / `SOTS-Indoor` from the RESIDE data line
- General deblurring: `GoPro`

## Already Available Locally

- `UHD-LL`: available and accepted.
- `LOL`: locally available as `LOL-v1` and `LOL-v2`, both accepted.

## 2026-04-25 Update

The following highest-priority UHD datasets have now been downloaded, unpacked, and accepted locally:

- `UHD-Haze`
  - Local path: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`
  - Acceptance: train `2290` pairs; local test `231` pairs.
  - Note: the local test split has one more pair than the paper inventory count of `230`; future manifests must explicitly record the chosen test-set口径.
- `UHD-Blur`
  - Local path: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`
  - Acceptance: train `1964` pairs; test `300` pairs.

The remaining paper-coverage downloads after this update are `SOTS-ITS` / `SOTS-Indoor` and `GoPro`.

## Still Missing For Paper-Coverage Reproduction

### Highest Priority

- `SOTS-ITS` / `SOTS-Indoor`
  - Task: general dehazing.
  - Paper role: Setting 1 general dehazing train/test line.
  - Source trail: `Li et al., 2019a`, RESIDE benchmark.
  - Local status: not found.

- `GoPro`
  - Task: general dynamic deblurring.
  - Paper role: Setting 1 deblurring training set and general benchmark.
  - Source trail: `Nah et al., 2017`.
  - Local status: not found.

## Important Non-Substitution Note

`4KDehaze` is locally available only as a clean UHD groundtruth image source through `4KDehaze/usable/groundtrues`. It does not currently provide paired hazy/input images, so it should not be counted as the paper's `UHD-Haze` training dataset.

`4K-Rain13k` and `8K` are useful for this project, but they are not part of the local paper inventory's core UHD low-light/dehaze/deblur reproduction scope.

## Current Download Gap Summary

To cover the paper's dataset scope, the remaining downloads are:

1. `SOTS-ITS` / `SOTS-Indoor` from RESIDE
2. `GoPro`
