# A_4kd31 Result Analysis

Updated: 2026-06-14 18:53:44 CST

## Purpose

A31 directly targeted the high-`input_std` / A08-tail bottleneck identified after A30. Instead of another small randomly initialized terminal route head, A31 loaded A24 as the shared base, transplanted A08 `decoder.2` into the high route decoder head, transplanted A15 `decoder.2` into the low route decoder head, and froze all non-route-head parameters.

## Changed Paths

- `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/configs/a_4kd31_fast_4kdehaze_a08_a15_transplanted_route_heads_route_only.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/launch_tmux.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/wait_and_launch_a_4kd31_memory_gated_gpu_safe.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/result_analysis.md`

## Verification

- `py_compile` passed for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, and `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`.
- TOML parse and checkpoint path checks passed.
- Launcher `bash -n` and GPU3 dry-run gate passed.
- Transplant smoke passed: high head vs A08 `decoder.2` max diff `0.0`; low head vs A15 `decoder.2` max diff `0.0`; trainable parameters `19718/168406`.
- GPU3 preflight completed: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183403.log`, exit code `0`.

## Training

Formal training completed on physical GPU3:

- Log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183538.log`
- Exit code: `0`
- Best validation checkpoint: step `50`
- Best validation PSNR: `18.25040839715078`
- Best validation loss: `0.12270757514983416`

Validation degraded after route deltas grew:

- step50: `val_psnr=18.25040839715078`, high delta `0.02367719597183168`, low delta `0.007631122474558652`
- step100: `val_psnr=16.8359389742469`, high delta `0.04923995513468981`, low delta `0.047412882931530474`
- step150: `val_psnr=16.069862055515074`, high delta `0.0648205789551139`, low delta `0.06804869346320629`
- step200: `val_psnr=15.353025043042464`, high delta `0.07024982444941998`, low delta `0.09419666267931462`
- step250: `val_psnr=15.389124336368274`, high delta `0.06899694245308638`, low delta `0.09509719841182232`
- step300: `val_psnr=15.844803117995628`, high delta `0.06569075476378203`, low delta `0.07857906743884087`

## Best-Val Test Metrics

- Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_task_metrics_best_val_gpu.json`

Results:

- `psnr_mean=17.770243709681385`
- `ssim_mean=0.7060290277004242`
- `loss_mean=0.12073371022939682`
- `total_loss_mean=0.1239648000895977`
- `dark_channel_l1_mean=0.14252242375165225`
- `edge_magnitude_l1_mean=0.15756582029163838`
- `local_contrast_l1_mean=0.024481982365250586`
- `chroma_l1_mean=0.03137393491342664`

This does not beat A24 single-checkpoint fast-test PSNR `17.790202882742463` and remains below A18 router `17.84693437641316`.

## A18 Alignment

Alignment file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/a18_record_alignment_best_val_gpu.json`

- Overall gap vs A18 router: `-0.07669066613303403 dB`
- Low-route gap vs A18 router: `-0.02801579592015501 dB`
- High-route gap vs A18 router: `-0.14121316850824578 dB`

Compared with A30, high route improved substantially (`-0.23814779446593295` to `-0.14121316850824578`). Compared with A24, high route also improved (`-0.224697` to `-0.14121316850824578`). However, the target was `>-0.10 dB`, and low route regressed from A24's `+0.069979 dB` to `-0.02801579592015501 dB`.

## Route-Regime Diagnostics

Route-regime file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/input_std_route_regime_best_val_gpu.json`

- Overall selected delta abs mean: `0.012584832096472383`
- High selected delta abs mean: `0.019272921736850294`
- Low selected delta abs mean: `0.007539431139696063`
- High selected branch L1 vs A08: `0.12602153940256253`
- Low selected branch L1 vs A15: `0.39231843226834345`

The important mechanism signal is positive: A31 broke A30's near-zero-delta failure and created meaningful high-route specialization. The failure mode is also clear: low route became worse because the A15 terminal head is not compatible enough with frozen A24 shared features, and the low branch remained far from the A15 teacher.

## Conclusion

A31 is not promoted. It does not exceed A24 or A18. But it is a useful mechanism breakthrough relative to A30: high-route preservation improved materially and route deltas became meaningful. The next run should keep A31's high-route A08 transplant idea but stop damaging low route: keep the low head at A24/base initialization, train only the high route or high-biased route modules, add a small route-delta budget, and stop early around the useful delta scale observed at step50.
