# A_dsh16 Gate Check

- [x] Uses the corrected UHD video deshaking / stabilization task, not `UHD_haze`.
- [x] Uses verified BVI-AOM synthetic paired-frame data at `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- [x] Uses the same train/val pair counts as A_dsh15: `112` train pairs and `32` val pairs.
- [x] Keeps `dataset.num_workers=0` and 512px validation crop.
- [x] Keeps the A_dsh15 full-val protocol with `max_val_batches=32`.
- [x] Enables only model features already implemented in the V1/T01 code path.
- [x] Keeps A_dsh15 artifacts untouched and creates an independent A_dsh16 run id.
- [x] TOML parse and model config construction pass.
- [x] GPU3-only launcher `bash -n` passes.
- [x] CPU preflight forward/backward and full 32-batch validation pass.
- [x] Deshaking data/config/GPU preflight dry-run passes.
- [x] Dry-run did not create a tmux training session or real training log.
- [x] Real GPU3-only launch was attempted after user approval.
- [x] Real launch failed with a project-local CUDA OOM at `batch_size=4`; no checkpoint or metric artifact was produced.
- [x] Post-failure cleanup found no project process residue and no external process was killed or displaced.
- [x] Logs/current_state and append-only raw logs are updated.

Current rule: A_dsh16 is a failed batch-size-4 context-guided attempt, not an accepted result. Use A_dsh17 or later lower-memory variants for this branch.
