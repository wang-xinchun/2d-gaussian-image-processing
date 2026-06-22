# GPU Usage Policy For G-RSCN

- Created: `2026-06-21 16:06:13 CST`
- Scope: `G-RSCN` experiments, starting with `G_4kd01`.

## Decision

G-RSCN inherits the project shared-GPU policy and tightens it for the first new-architecture probe.

Approved physical GPUs:

```text
0
3
```

Forbidden physical GPUs:

```text
1
2
```

Default physical GPU:

```text
3
```

## First-Run Policy For G_4kd01

The first `G_4kd01` smoke or preflight must use exactly one approved physical GPU.

Forbidden for the first run:

- `CUDA_VISIBLE_DEVICES=0,3`;
- any DDP launch;
- any physical GPU outside `0` or `3`;
- launching when `nvidia-smi` shows the selected GPU above the utilization gate;
- launching formal training before mechanism/equivalence/parameter gates pass.

Required launch environment:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<one approved physical GPU only>
TWODGAUSS_ALLOWED_PHYSICAL_GPUS=0,3
```

Default resource gates:

```text
TWODGAUSS_PHYSICAL_GPU=3
TWODGAUSS_MIN_FREE_MIB=8192
TWODGAUSS_MAX_GPU_UTIL=20
TWODGAUSS_CUDA_MEMORY_FRACTION=0.35
```

First-run data-loader gate:

```text
dataset.num_workers <= 1
batch_size = 1
grad_accum = 2..8 for smoke probes, chosen by written run-local gate
```

The controlling GPU safety limits are the single approved physical GPU, `TWODGAUSS_CUDA_MEMORY_FRACTION<=0.35`, `num_workers<=1`, and the live free-memory/utilization gate. A short smoke may reduce `grad_accum` when the previous smoke established a low peak CUDA reservation and the run-local gate records the change.

## Confirmation Gates

`G_4kd01` launch scripts must refuse non-dry-run execution unless all of these are explicitly set after `gate_check.md` is updated:

```text
TWODGAUSS_G_4KD01_CONFIRM=launch_g_4kd01_grscn_gpu_safe
TWODGAUSS_G_4KD01_MECHANISM_GATE=passed
TWODGAUSS_G_4KD01_EQUIVALENCE_GATE=passed
TWODGAUSS_G_4KD01_GPU_GATE=passed
```

Formal training remains blocked until a future update removes the current preflight-only guard after implementation and gate approval.

## Rationale

The server is shared. Previous decisions already restricted this project to physical GPUs `0` and `3`, with conservative single-GPU visibility to avoid accidentally mapping logical `cuda:0` to the wrong physical card. G-RSCN is a new architecture line, so the first run must also protect against wasting GPU time on an unimplemented or non-equivalent model. Mechanism gates and GPU gates therefore both have launch authority.
