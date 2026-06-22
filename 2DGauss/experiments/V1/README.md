# V1 / A-Series Experiment Line

This directory preserves the old architecture line and its A-series experiment evidence.

V1 is not discarded. It is the evidence branch that established the route-specific restoration problem and directly motivated G-RSCN.

## Role In The Project

V1 is the first executable cross-resolution 2D Gaussian restoration framework:

- high-resolution CNN feature stem;
- low-resolution compact branch;
- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- Gaussian-guided attention/control/refinement paths;
- RGB residual restoration head;
- later A-series route-patch, route-head, and gate mechanisms.

The current G-RSCN line keeps the Gaussian core and learns from V1's failure modes.

## Key Evidence Chain

- `A_4kd18`: fixed `input_std` A08/A15 router proves route complementarity.
- `A_4kd24`: late input-std route residual gives the first positive single-checkpoint structural conversion, but remains too late/small.
- `A_4kd33`: checkpoint surgery proves high/low component recombination can help while protecting low route.
- `A_4kd36`: single-checkpoint dual-backbone upper bound nearly reproduces A18 router behavior.
- `A_4kd37`: compact/shared aggregate breakthrough and the main parent for G-RSCN probes.
- `A_4kd38` through `A_4kd47`: stop-loss evidence showing late heads, scalar losses, samplers, and gate-only patches trade high/A08 against low/A15 instead of moving both.

## GitHub Mirror Boundary

GitHub stores only lightweight, reviewable V1 artifacts:

- configs;
- launch/evaluation shell scripts;
- `manifest.md`;
- `gate_check.md`;
- `result_analysis.md`;
- mechanism/failure analysis docs;
- milestone README files.

The local server remains the source for full experiment state:

- checkpoints and weights;
- metrics JSON files;
- raw train/eval logs;
- diagnostic images;
- datasets and caches;
- template snapshots.

Do not promote V1 patches as the current mainline unless a future experiment card explicitly reopens a bounded falsification probe. The current mainline is `../G_RSCN`.
