# A_dsh15 Gate Check

- [x] Current task is UHD video deshaking / stabilization, not `UHD_haze`.
- [x] Config contains deshake/stabilization/video task intent and no banned haze/dehaze identifiers.
- [x] Data root exists: `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`.
- [x] Data view is paired `.npy` synthetic shaky -> clean frames.
- [x] Train pairs verified: `112`.
- [x] Val pairs verified: `32`.
- [x] Validation protocol uses full val view: `32` batches.
- [x] GPU1 and GPU2 are not used.
- [x] GPU0 was avoided because it was occupied by external compute.
- [x] Physical GPU3 preflight passed before launch with `48520MiB` free and `0%` utilization.
- [x] Launch used `CUDA_VISIBLE_DEVICES=3`.
- [x] Runtime memory cap used: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`.
- [x] Runtime GPU3 free-memory monitor stayed above the stop threshold.
- [x] Training completed with `train_exit_code=0`.
- [x] Train metrics exist and contain `241` records.
- [x] Val metrics exist and contain `24` full-validation records.
- [x] `best_val.pt` exists and has checkpoint step `1000`.
- [x] `final.pt`, `latest.pt`, and `step_001200.pt` exist and have checkpoint step `1200`.
- [x] Independent best-val evaluation completed on the 32-pair val view.
- [x] Independent final evaluation completed on the 32-pair val view.
- [x] Post-run cleanup found no project tmux session and no project training process residue.
- [x] GPU3 returned to `0MiB` used after training and independent evaluation.

Conclusion: A_dsh15 passes the controlled full-train deployment gate.
