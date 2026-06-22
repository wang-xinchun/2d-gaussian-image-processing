# A_b02_gpu0_bs23_r1 Gate Check

Updated: 2026-06-05 21:48 CST

## Preconditions

- Physical GPU must be `0`.
- GPU0 free memory must be at least `43000MiB` at launch.
- `CUDA_VISIBLE_DEVICES` is set to the selected physical GPU inside the tmux session.
- `TWODGAUSS_ALLOWED_PHYSICAL_GPUS` is set to `0,3`.
- A tmux session with the same name must not already exist.

## Current Rationale

- `batch_size=22` completed 1200 steps and wrote `final.pt`.
- The older `batch_size=24` run reached validation checkpoints but exited without a final checkpoint or exit-code file.
- `batch_size=23` is the next larger GPU0-only trial with explicit exit-code capture.
