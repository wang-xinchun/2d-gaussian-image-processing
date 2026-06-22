# V1/T01/A_4kd20

## Purpose

`A_4kd20` is a CPU-only offline learned-router diagnostic. It does not run
model inference again and does not train a restoration checkpoint.

It reads saved A08/A15 per-sample router records from `A_4kd18` and `A_4kd19`,
then fits a tiny input-only logistic router with `input_mean` and `input_std`.

## Data

- Train records: A18 validation `100` + A19 tail validation `100`.
- Eval records: A18 fast test `100` and A19 tail test `100`.
- Features: `input_mean`, `input_std`.
- Fixed-rule reference: A18 input-std threshold `0.07026463374495506`.

## Result

- Logistic weights: `[1.1996748955399474, 0.5272583947116886, -2.3482197247109773]`.
- Tuned probability threshold: `0.51`.
- Fast test:
  - fixed A18 rule PSNR `17.84693437581442`;
  - input-mean/std logistic PSNR `17.839935257341832`.
- Tail test:
  - fixed A18 rule PSNR `15.913737915082573`, SSIM `0.7024011361598969`;
  - input-mean/std logistic PSNR `15.917272808208413`, SSIM `0.7027021211385727`.

## Decision

The two-feature logistic router is not stable enough to replace the A18 rule:
it helps on A19 tail test but loses on A18 fast test. However, it supports the
idea that a learned router may be useful if it sees richer input/control
features than `input_mean` and `input_std` alone.

The next implementation should avoid treating this two-feature linear router as
the final design. Prefer a richer lightweight router using input statistics plus
existing model-side confidence/control summaries, or a model-internal gating
head trained with routing labels.
