# Milestone: M02_A_i03_uhddeblur_official_baseline

## Identity

- Milestone ID: `M02`
- Bound run: `V1/T01/A_i03`
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
checkpoint: A_i03 final.pt
PSNR: 25.85061787687282
```

Baseline continuity:

```text
previous accepted baseline: A_g03 final.pt
previous PSNR: 25.3800849048404
delta: +0.47053297203242295
```

## Contents

- `weights/`
  - accepted official weight `A_i03_final.pt`
  - matching final periodic checkpoint `A_i03_step_012000.pt`
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
