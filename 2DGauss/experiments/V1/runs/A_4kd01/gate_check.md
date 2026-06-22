# Gate Check: V1/T01/A_4kd01

Status: prepared; data/config gate complete; GPU launch deferred.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd01`.
- [x] Full run identifier: `V1/T01/A_4kd01`.
- [x] Run folder: `experiments/V1/runs/A_4kd01`.
- [x] Run-local config exists.
- [x] Template config exists.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5`.
- [x] Train split has `14045` paired files.
- [x] Validation split has `780` paired files.
- [x] Test split has `781` paired files.
- [x] Split view has `31212` symlinks and `0` broken symlinks.
- [x] Input/target files are same-name paired.
- [x] Sample files load as RGB images.

## Current Training-Ready Inventory

- [x] T01 paired-restoration trainable datasets: `6`.
- [x] T01 paired-restoration train pairs: `31719`.
- [x] SR aligned train pairs across all LR variants: `56196`.
- [x] SR datasets are excluded from this T01 run until an SR-specific template
      is added.
- [x] Historical `UHD_haze` and `UHD_deblur` datasets are not present in the
      current Baidu Wangpan copy on this server.

## Runtime Check

- [x] Planned batch size: `16`.
- [x] Planned train crop: `512`.
- [x] Planned validation crop: `512`.
- [x] Planned max steps: `2000`.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Training script supports this paired-restoration directory format through
      `PairedImageDataset`.
- [ ] Formal GPU preflight.
- [ ] Formal launch.

## Launch Hold

GPU launch is intentionally deferred until explicit launch time because the
preparation-time GPU snapshot showed high external utilization on all visible
GPUs. No checkpoint has been created for this run yet.
