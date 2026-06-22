# A_dsh17 Gate Check

- [x] A_dsh16 failed with project-local CUDA OOM at `batch_size=4`.
- [x] A_dsh17 uses a new run id and does not overwrite A_dsh16 evidence.
- [x] Uses physical GPU3 only; no GPU0/GPU1/GPU2 launch.
- [x] Uses verified BVI-AOM synthetic paired-frame data, `112` train pairs and `32` val pairs.
- [x] Lowers `batch_size` to `2` while keeping `train_context_size=728` and center supervision `512`.
- [x] Initial dry-run correctly blocked when GPU3 became busy, without touching the external process.
- [x] Dry-run/live preflight passes after GPU3 returned to idle.
- [x] Real training completes with `train_exit_code=0`.
- [x] Independent evaluation completes for both final and best-val checkpoints.
- [x] Post-run cleanup finds no project training/tmux residue and GPU3 returns to `0MiB` used.
- [x] Logs/current_state and raw append-only logs are updated.

Current rule: A_dsh17 is a safe completed deployment, not the accepted best baseline. A_dsh15 remains the current accepted run because it is better under the same independent evaluation protocol.
