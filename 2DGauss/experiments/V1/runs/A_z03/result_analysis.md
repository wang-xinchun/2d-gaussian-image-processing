# Result Analysis: V1/T01/A_z03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_z03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `35.9080` minutes
- Final checkpoint: `experiments/V1/runs/A_z03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_z03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_z03/logs/train_20260516_151129.log`
- Train metrics: `experiments/V1/runs/A_z03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_z03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_z03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Their serialized checkpoint files differ because of non-model payload, but
  the effective trained endpoint is the same.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`2500` endpoint.

## Training Behavior

The branch showed a visibly non-monotonic live curve during training:

- `step=500`: `val_psnr=28.4670`
- `step=1000`: `val_psnr=28.6419`
- `step=1500`: `val_psnr=28.1106`
- `step=2000`: `val_psnr=28.5236`
- `step=2500`: `val_psnr=28.9776`

Interpretation:

- The decoder-detail-aware hint did not produce a smooth upward trajectory.
- Mid-run live evidence alone would have been easy to overread in either
  direction.
- The correct acceptance boundary for this branch was therefore the final
  endpoint, not any single intermediate validation point.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024387 | 28.4670 |
| 1000 | 0.024018 | 28.6419 |
| 1500 | 0.025631 | 28.1106 |
| 2000 | 0.025085 | 28.5236 |
| 2500 | 0.023185 | 28.9776 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.023185035306960346
val_psnr = 28.97758263585435
```

The lowest training-time validation loss is:

```text
step 2500
val_loss = 0.023185035306960346
val_psnr = 28.97758263585435
```

Interpretation:

- Unlike the earlier mixed mid-run signal, the final endpoint becomes the best
  internal validation point of the whole short run.
- This confirms that the branch recovered by the endpoint rather than merely
  ending on a random near-tie.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_z03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.036455 | 25.3857 |

Interpretation:

- `A_z03` converts a mixed live-training story into a clear completed positive
  short result on the official protocol.
- The decoder-detail-aware hint is therefore not just runtime-safe; it has
  now demonstrated a real same-protocol benchmark gain over the nearby short
  anchors.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_u03 final.pt official full-image test PSNR = 25.310064081965912
A_t03 final.pt official full-image test PSNR = 25.294446219726133
A_v03 final.pt official full-image test PSNR = 25.271115184810476
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_r03 final.pt official full-image test PSNR = 25.286687395113344
A_s03 final.pt official full-image test PSNR = 25.25673342402521
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_k03 final.pt official full-image test PSNR = 26.199959289578075
A_y03 final.pt official full-image test PSNR = 26.272695829299355
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_z03 Final |
|---|---|---:|---:|
| `A_z03` | `final.pt` | 25.3857 | 0.0000 |
| `A_u03` | `final.pt` | 25.3101 | -0.0756 |
| `A_t03` | `final.pt` | 25.2944 | -0.0913 |
| `A_n03` | `final.pt` | 25.2919 | -0.0938 |
| `A_r03` | `final.pt` | 25.2867 | -0.0990 |
| `A_v03` | `final.pt` | 25.2711 | -0.1146 |
| `A_s03` | `final.pt` | 25.2567 | -0.1290 |
| `A_p03` | `final.pt` | 26.0087 | +0.6230 |
| `A_k03` | `final.pt` | 26.2000 | +0.8142 |
| `A_y03` | `final.pt` | 26.2727 | +0.8870 |

Additional deltas:

```text
A_z03 final.pt vs A_u03 final.pt = +0.07564673839926073 dB
A_z03 final.pt vs A_t03 final.pt = +0.09126460063903963 dB
A_z03 final.pt vs A_v03 final.pt = +0.11459563555469643 dB
A_z03 final.pt vs A_n03 final.pt = +0.0937817227876998 dB
A_z03 final.pt vs A_r03 final.pt = +0.09902342525182917 dB
A_z03 final.pt vs A_s03 final.pt = +0.12897739633996252 dB
A_z03 final.pt vs A_p03 final.pt = -0.6229961513651219 dB
A_z03 final.pt vs A_k03 final.pt = -0.8142484692129024 dB
A_z03 final.pt vs A_y03 final.pt = -0.8869850089341824 dB
```

Interpretation:

- `A_z03` is now the strongest completed short result in the local post-`M04`
  continuation family.
- The gain over `A_u03` is large enough to matter within this short-run
  regime; it is no longer a rounding-level tie.
- But the branch remains far below the current medium-budget milestone line
  `A_y03/M04`, so the correct reading is “useful short positive continuation,”
  not “new official baseline candidate.”

## Main Conclusion

`V1/T01/A_z03` is accepted as a successfully completed short probe with a real
positive official result above the nearby short anchors.

It proves:

- the decoder-detail-aware refinement hint is runtime-safe under the same
  practical VRAM envelope;
- the branch can improve same-protocol official testing over
  `A_u03/A_t03/A_v03/A_n03/A_r03/A_s03`;
- early mixed validation behavior did not prevent the branch from closing as
  the strongest completed short result in this local family.

It does not prove:

- that this mechanism is already strong enough to replace the medium-budget
  milestone line `A_y03/M04`;
- that the absolute-performance gap to the accepted medium line is closed;
- that another nearby short tweak is necessarily better than a more deliberate
  confirmation or promotion design.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: yes
official baseline replacement: no
recommended next action: treat A_z03 as the new short-run leader above M04/A_y03 and decide between confirmation short replay or medium-budget promotion
```
