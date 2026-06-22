# A_dsh01_predeploy Gate Check

Updated: 2026-06-05 21:31 CST

## Pending Gates

- [ ] Confirm local path of the incoming UHD video deshaking dataset.
- [ ] Identify layout: raw videos, frame sequences, paired shaky/stable frames, or metadata-supervised stabilization.
- [ ] Verify train/val/test split or create a split plan.
- [ ] Run `scripts/inspect_deshake_dataset.py` and store JSON under this run's `metrics/`.
- [ ] Define deshaking metrics: temporal jitter/stability, crop/border behavior, visual panel, and optional paired fidelity.
- [ ] Run `scripts/evaluate_deshake_sequence_metrics.py` on representative sequences.
- [ ] Decide whether `A_y03` frame enhancement is only a baseline or part of a motion-aware model.
- [ ] Create a true training config that points to deshaking data, not `UHD_haze`.
- [ ] Re-run GPU0/3 gate before any launch.

## Current Decision

Training is blocked. The current package is ready for dataset inspection and frame-level baseline preparation only.
