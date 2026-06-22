# 2D Gaussian Image Processing

This repository is a lightweight GitHub mirror for the UHD 2D Gaussian restoration project.

The authoritative research workspace remains on the server at:

```text
/home/ldj/wxc/2DGauss
```

GitHub is used for:

- versioned model/source code review;
- architecture analysis across worktrees;
- coordinated development before experiments;
- lightweight documentation and experiment manifests.

GitHub is not used for:

- datasets;
- checkpoints or model weights;
- raw experiment logs/metrics/images;
- private SSH credentials or server-local Codex state;
- paper PDF archives.

## Current Research Line

The active direction is G-RSCN, a Gaussian-preserved route-specific control architecture for ultra-high-resolution image restoration. The model preserves the project core:

- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- cross-resolution Gaussian attention/control;
- Gaussian-guided decoder, refinement, and control paths.

The old A-series route-patch line is kept as evidence and baseline. The current mainline lives under:

```text
2DGauss/experiments/G_RSCN
```

## Repository Layout

```text
2DGauss/src/                 model package
2DGauss/scripts/             training, evaluation, audit, and logging scripts
2DGauss/docs/                project design and training policy docs
2DGauss/logs/*.md            compact readable research history
2DGauss/experiments/G_RSCN/  lightweight G-RSCN configs, gates, and analyses
```

Large local-only assets are ignored by git. Before any push, inspect staged files:

```bash
git status --short
git diff --cached --stat
git ls-files --cached
```

Do not use `git add .` from `/home/ldj/wxc`.
