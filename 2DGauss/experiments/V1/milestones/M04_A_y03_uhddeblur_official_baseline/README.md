# Milestone: M04_A_y03_uhddeblur_official_baseline

## Identity

- Milestone ID: `M04`
- Bound run: `V1/T01/A_y03`
- Task: `UHD_deblur`
- Status: frozen accepted baseline

## Why This Exists

This directory is a write-protected milestone freeze of the currently accepted
`UHD_deblur` official baseline so that future mainline development cannot
accidentally mutate:

- the accepted official weights;
- the exact code snapshot used by the accepted run;
- the accepted post-hoc metrics and analysis artifacts.

## Accepted Result

Official full-image test result:

```text
checkpoint: A_y03 final.pt
PSNR: 26.272695829299355
```

Baseline continuity:

```text
previous accepted baseline: A_k03 final.pt
previous PSNR: 26.199959289578075
delta: +0.07273653972128045
```

Confirmed replay continuity:

```text
best observed pre-refresh anchor: A_x03 final.pt
anchor PSNR: 26.207660130733675
delta vs A_x03: +0.06503569856568035
```

Lightweight target status:

```text
current UHD-Blur 90% threshold: 26.5743
remaining gap: 0.3016041707006434
```

## Contents

- `weights/`
  - accepted official weight `A_y03_final.pt`
  - last periodic checkpoint before final endpoint `A_y03_step_021000.pt`
- `template_snapshot/`
  - exact run-time code/config snapshot copied from the accepted run archive
- `artifacts/`
  - config, logs, metrics, manifest, gate check, result analysis
- `acceptance_scripts/`
  - evaluation entrypoint used to reproduce the acceptance metrics
- `SHA256SUMS.sha256`
  - integrity list for the frozen milestone payload

## Protection Rule

This directory is intentionally set read-only with `chmod a-w`.

If a future milestone supersedes this one:

1. create a new `M**` directory instead of editing this one;
2. repoint the symbolic link
   `experiments/V1/milestones/current_uhddeblur_official_baseline`;
3. leave this directory untouched for audit and reproducibility.
