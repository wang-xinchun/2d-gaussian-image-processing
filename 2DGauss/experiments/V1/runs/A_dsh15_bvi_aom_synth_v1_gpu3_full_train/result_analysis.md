# A_dsh15 Result Analysis

## Summary

A_dsh15 is the first high-quality controlled training run for the current UHD video deshaking task on the synthesized BVI-AOM paired-frame view. It ran on physical GPU3 only because GPU0 was occupied by external processes at launch time; GPU1/2 were never used.

Training completed normally with `train_exit_code=0`, reached `1200` steps, and wrote full train/val metrics plus `best_val.pt`, `final.pt`, `latest.pt`, and periodic checkpoints.

## Training Metrics

- Train metric records: `241`.
- Val metric records: `24`.
- Validation interval: every `50` steps.
- Val batches per validation: `32`, covering the full val paired view.
- First validation at step `50`: `val_loss=0.10983952623791993`, `val_psnr=17.31706473147199`.
- Best validation loss at step `1000`: `val_loss=0.09517641738057137`, `val_psnr=18.49445972137987`.
- Final validation at step `1200`: `val_loss=0.09540756104979664`, `val_psnr=18.504268557322963`.
- Final train loss at step `1200`: `loss_total=0.08317220956087112`.
- Peak reserved GPU3 memory: `38.72265625GiB`.

Compared with the first validation point, final validation improves by about `+1.1873 dB` PSNR and reduces val loss by about `0.01443`.

## Independent Evaluation

Independent evaluation compares the model output against the shaky input baseline on the same 32 validation pairs.

Best-val checkpoint, selected by lowest val loss:

- Checkpoint step: `1000`.
- Input baseline PSNR: `16.507102228205603`.
- Prediction PSNR: `18.494459822676237`.
- Mean PSNR gain: `+1.9873575944706348 dB`.
- Input baseline L1: `0.11410748690832406`.
- Prediction L1: `0.09307080949656665`.
- Input baseline SSIM: `0.37895671324804425`.
- Prediction SSIM: `0.4998005349189043`.
- Mean SSIM gain: `+0.12084382167086005`.

Final checkpoint:

- Checkpoint step: `1200`.
- Input baseline PSNR: `16.507102228205603`.
- Prediction PSNR: `18.504268639458388`.
- Mean PSNR gain: `+1.997166411252786 dB`.
- Input baseline L1: `0.11410748690832406`.
- Prediction L1: `0.09331451682373881`.
- Input baseline SSIM: `0.37895671324804425`.
- Prediction SSIM: `0.5011140927672386`.
- Mean SSIM gain: `+0.12215737951919436`.

All four validation motion preset groups have positive PSNR and SSIM gains for both best-val and final checkpoints.

## Sequence Proxy

The independent evaluation also records adjacent-frame proxy metrics per validation clip. These metrics are lightweight indicators only; they are not final full-video stabilization metrics.

For the final checkpoint, prediction adjacent-frame L1 moves closer to the target sequence than the shaky input on all four validation clips. Prediction adjacent edge-L1 is also lower than target-adjacent edge-L1 in the current proxy, which indicates stronger smoothing of edge changes; this is useful to flag for future visual review because over-smoothing can look stable while losing detail.

## Acceptance

A_dsh15 passes as a meaningful full-train paired-frame baseline:

- It is trained on the corrected deshaking task data.
- It avoids busy GPU0 and never uses GPU1/2.
- It completes a 1200-step run with full-val protocol.
- It saves a best-val checkpoint.
- It independently improves over the shaky input baseline on PSNR, L1, and SSIM.
- It includes sequence proxy evidence and motion-preset grouping.

The result should not be described as complete UHD video stabilization. The model is still a single-frame restoration baseline trained on synthetic paired frames. Next work should add longer and more diverse sequence evaluation, visual panel export, and eventually temporal/warp-aware training or evaluation.
