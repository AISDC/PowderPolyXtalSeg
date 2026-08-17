# PowderPolyXtalSeg

Semantic segmentation of **HEDM diffraction spots** and **powder diffraction
rings** in X-ray area-detector images, with training code for three
architectures — **U-Net**, **UNet++** and **SegFormer (MiT-B2)** — under one
pipeline.

Every pixel of a detector frame is assigned one of three classes:

| Class | Name | Typical pixel fraction |
|:-----:|------|-----------------------|
| 0 | Background | ~95% |
| 1 | HEDM spot | ~0.4% |
| 2 | Powder ring | ~4.6% |

The extreme class imbalance is what drives most of the design decisions here:
frames with almost no signal are filtered out before training, the loss is a
Focal + Dice combination with the HEDM class weighted 5x, and models are
selected on IoU rather than pixel accuracy.

The three architectures are trained as a **controlled comparison**: all of them
are randomly initialized and share one optimizer protocol, epoch budget and
effective batch size, so the only variable is the architecture. A fourth run,
`segformer_pretrained`, repeats SegFormer with an ImageNet-pretrained MiT-B2
encoder. It is a control, not a competitor: transformers normally rely on
ImageNet pretraining, so without it a weak from-scratch SegFormer could not be
attributed to the architecture rather than to the missing initialization.

This repository contains the model-training code for the study described in
[the accompanying paper](#citation). The synthetic-data generator used to build
the training set is released separately as the *PowHEDM Simulator*.

---

## Installation

```bash
git clone <repository-url>
cd PowderPolyXtalSeg
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 1.13 with CUDA. `transformers` is only
needed for SegFormer. The published runs used 4–8x Tesla V100-SXM2-32GB.

---

## Quick start

The repository ships a small example dataset (~57 MB), so the pipeline runs
without downloading anything:

```bash
python train.py --model unet                 --data-root examples --no-ddp --epochs 1 --limit-batches 2
python train.py --model unetpp               --data-root examples --no-ddp --epochs 1 --limit-batches 2
python train.py --model segformer            --data-root examples --no-ddp --epochs 1 --limit-batches 2
python train.py --model segformer_pretrained --data-root examples --no-ddp --epochs 1 --limit-batches 2
```

This exercises the full path — 3-way mixing, weighted sampling, the three
validation domains, checkpointing — but the subset is far too small to produce
meaningful metrics. See [`examples/README.md`](examples/README.md).

`segformer_pretrained` downloads the MiT-B2 weights from HuggingFace on first
use. On an offline node, pre-populate the cache and then set
`HF_HUB_OFFLINE=1`; `segformer` needs the hub only for the architecture config.

---

## Data

### Layout

All scripts read from a single dataset directory, given by `--data-root`, by
`$POWDERSEG_DATA_ROOT`, or defaulting to `./data`:

```
<data-root>/
├── real/                       hand-labelled detector frames, one dir per sample
│   └── <sample_name>/
│       ├── <anything>.tif           normalized image, float32 in [0, 1]
│       └── labkit_segmented.tif     uint8 mask, values {0, 1, 2}
├── synth_train.h5              old synthetic frames
├── synth_val.h5
├── filtered_indices.json       {"train": [...], "val": [...]} — non-empty frames
├── ring_train.h5               ring synthetic patches
└── ring_val.h5
```

Every HDF5 file uses the same two datasets:

| Key | dtype | Shape | Range |
|-----|-------|-------|-------|
| `images` | float32 | (N, 512, 512) | normalized to [0, 1] |
| `masks` | uint8 | (N, 512, 512) | {0, 1, 2} |

### The three sources

Training draws from three sources at fixed proportions, chosen because no single
source covers the problem: the real data is the target domain but tiny, the old
synthetic set is rich in HEDM spots and material diversity, and the ring set
supplies the continuous powder rings that the other two under-represent.

| Source | Share of each epoch | What it is |
|--------|--------------------|------------|
| Real hand-labelled | 20% | 7 experimental frames (GE 2048², Varex 2880²), Labkit-annotated, tiled into 314 patches |
| Old synthetic | 20% | 6,412 frames from the PowHEDM simulator — MIDAS forward simulation (HEDM) + Materials Project XRD (powder) |
| Ring synthetic | 60% | 10,000 powder-rich patches from simulated ring frames |

### Preprocessing

Raw 16-bit frames have a dynamic range that a few saturated spots dominate, so
all images are **clip-log normalized**: clip to the [0.1, 99.9] percentiles,
`log1p`, then min-max scale to [0, 1]. Detector frames are tiled into 512×512
patches at stride 384, and patches with < 0.5% foreground pixels are dropped.

### Building the dataset

```bash
# 1. Normalize the hand-labelled frames
python scripts/prepare_real_data.py --src raw_hand_labelled --out data/real

# 2. Convert simulated ring frames (TIF pairs) into HDF5
python scripts/prepare_ring_data.py --src gen_dataset_for_ML --out data

# 3. Index the non-empty frames of the old synthetic set
python scripts/filter_empty_frames.py \
    --train data/synth_train.h5 --val data/synth_val.h5 \
    --out data/filtered_indices.json
```

> **Full dataset and trained checkpoints:** `TODO — add the Zenodo/DOI link here
> before publication.`

---

## Training

```bash
# Multi-GPU DDP — uses every visible GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --model unet --data-root data

# Single GPU
python train.py --model segformer --data-root data --no-ddp

# Resume an interrupted run (restores optimizer, scheduler and AMP scaler state)
python train.py --model unetpp --resume runs/unetpp_mixedv2/checkpoints/checkpoint_epoch_030.pth
```

Outputs land in `runs/<model>_mixedv2/`:

```
checkpoints/best_model.pth              best combined metric so far
checkpoints/checkpoint_epoch_NNN.pth    every 10 epochs
logs/history.json                       per-epoch losses and per-class IoU/Dice
```

### Validation protocol

Each epoch is validated separately on all three domains, and the scalar used for
best-model selection, LR scheduling and early stopping is their weighted mean:

```
combined = 0.2 * real_mIoU + 0.2 * old_synth_mIoU + 0.6 * ring_mIoU
```

IoU and Dice are computed from pixel counts accumulated over the whole
validation set (and, under DDP, all-reduced across ranks) — not by averaging
per-batch values, which the background class would otherwise dominate.

### Hyperparameters

Shared by every run: 512×512 patches, Focal(γ=2.0, α=[0.1, 5.0, 1.0]) + Dice
loss at 0.5/0.5, `ReduceLROnPlateau` (factor 0.5, patience 10) on the combined
metric, gradient clipping at 1.0, AMP, seed 42, and augmentation by flips, 90°
rotations and ±10% brightness/contrast jitter.

| | U-Net | UNet++ | SegFormer | SegFormer (IN-pre) |
|---|---|---|---|---|
| Parameters | 17,261,955 | 36,621,900 | 27,342,659 | 27,342,659 |
| Initialization | random | random | random | ImageNet MiT-B2 |
| Optimizer | Adam | Adam | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 | 1.5e-4 | 6e-5 |
| Weight decay | 1e-5 | 1e-5 | 0.01 | 0.01 |
| LR warmup | – | – | 3 epochs, linear | 3 epochs, linear |
| Deep supervision | – | 4 heads, weights [0.25]×4 | – | – |
| Batch size / GPU | 8 | 8 | 8 | 8 |
| Epochs | 200 | 200 | 200 | 200 |
| Early-stop patience | 30 | 30 | 30 | 30 |
| Minimum LR | 1e-6 | 1e-6 | 1e-6 | 1e-6 |

U-Net and UNet++ are configured identically — same optimizer, LR, schedule,
batch size and epoch budget — so the two differ only in architecture. SegFormer
follows the transformer convention of AdamW with a short linear warmup. The
pretrained variant differs from the from-scratch SegFormer in exactly two
respects: its encoder starts from ImageNet weights, and it uses the SegFormer
reference fine-tuning LR of 6e-5 instead of 1.5e-4, which would wash the
pretrained features out. Its RGB patch-embedding stem is collapsed to a single
channel by summing the three input-channel kernels, so the pretrained stem is
adapted rather than discarded.

---

## Results

Best-epoch validation metrics on the full dataset:

| Model | Domain | mIoU | BG IoU | HEDM IoU | Powder IoU |
|-------|--------|------|--------|----------|------------|
| **U-Net** (epoch 197) | real | 0.9689 | 0.9925 | 0.9563 | 0.9579 |
| | old synthetic | 0.9590 | 0.9988 | 0.9104 | 0.9679 |
| | ring synthetic | 0.9399 | 0.9969 | 0.8367 | 0.9860 |
| | **combined** | **0.9495** | | | |
| **UNet++** (epoch 200) | real | 0.9638 | 0.9912 | 0.9499 | 0.9502 |
| | old synthetic | 0.9577 | 0.9988 | 0.9072 | 0.9672 |
| | ring synthetic | 0.9362 | 0.9968 | 0.8262 | 0.9854 |
| | **combined** | **0.9460** | | | |
| **SegFormer** (epoch 195) | real | 0.9503 | 0.9901 | 0.9136 | 0.9473 |
| | old synthetic | 0.9457 | 0.9980 | 0.8877 | 0.9516 |
| | ring synthetic | 0.8590 | 0.9955 | 0.6022 | 0.9794 |
| | **combined** | **0.8946** | | | |
| **SegFormer (IN-pre)** (epoch 194) | real | 0.9624 | 0.9920 | 0.9382 | 0.9569 |
| | old synthetic | 0.9568 | 0.9985 | 0.9072 | 0.9648 |
| | ring synthetic | 0.8768 | 0.9961 | 0.6524 | 0.9818 |
| | **combined** | **0.9099** | | | |

**The two convolutional models are effectively tied.** U-Net finishes ahead of
UNet++ by 0.0035 in combined mIoU, which is too small to claim a real accuracy
advantage from a single run per architecture; treat them as equally accurate.
What is not close is the cost: U-Net reaches that accuracy at roughly four times
UNet++'s throughput (see below). Accuracy and speed therefore do not trade off
against each other here, and U-Net is the model deployed for the rest of the
study.

**SegFormer's deficit is architectural, not a pretraining artefact.** Every
model segments background almost perfectly (IoU ≈ 0.99 everywhere) and handles
powder rings well (IoU ≳ 0.94 across all domains). The gap is concentrated in a
single cell — HEDM IoU on the ring-synthetic domain, where small, sparse spots
sit on top of strong rings — and there SegFormer collapses to 0.602 against
U-Net's 0.837. ImageNet pretraining lifts that cell only to 0.652, recovering
less than a third of the 0.235 gap, even though it brings SegFormer to parity
with the convolutional models everywhere else (real HEDM 0.938, old-synthetic
HEDM 0.907, background and powder throughout). So the weakness is specific to
isolating sparse spots in dense ring backgrounds, which is exactly the regime
that matters for partitioning: the patch-based tokenization and 1/4-resolution
decoder are poorly matched to targets a few pixels across, whereas the
convolutional models retain full-resolution detail through their skip
connections.

### Inference speed

512×512 patches, FP16, batch 16, on one Tesla V100-SXM2-32GB. Full-frame rates
assume non-overlapping 512×512 tiling.

| Model | Parameters | Patches/s | ms/patch | GE 2048² (16 tiles) | Varex 2880² (36 tiles) |
|-------|-----------|-----------|----------|---------------------|------------------------|
| SegFormer | 27.3 M | 160.0 | 6.25 | 10.0 fps | 4.4 fps |
| U-Net | 17.3 M | 133.1 | 7.51 | 8.3 fps | 3.7 fps |
| UNet++ | 36.6 M | 32.0 | 31.28 | 2.0 fps | 0.9 fps |

UNet++'s dense nested skip connections cost roughly 4x the throughput for no
measurable accuracy gain, and on the larger Varex frame they drop it below the
1 fps real-time threshold. U-Net and SegFormer both stay comfortably above it.
`segformer_pretrained` is the same graph as `segformer`, so its timings are
identical.

---

## Evaluation and benchmarking

```bash
# One model
python evaluate.py --model unetpp --checkpoint runs/unetpp_mixedv2/checkpoints/best_model.pth

# All four side by side
python evaluate.py \
    --model unet       --checkpoint runs/unet_mixedv2/checkpoints/best_model.pth \
    --model unetpp     --checkpoint runs/unetpp_mixedv2/checkpoints/best_model.pth \
    --model segformer  --checkpoint runs/segformer_mixedv2/checkpoints/best_model.pth \
    --model segformer_pretrained \
        --checkpoint runs/segformer_pretrained_mixedv2/checkpoints/best_model.pth \
    --json results.json

# Inference speed (weights optional — timing does not depend on them)
CUDA_VISIBLE_DEVICES=0 python benchmark_speed.py
```

---

## Repository layout

```
train.py                    training entry point for all four runs
evaluate.py                 per-class IoU/Dice on the three validation domains
benchmark_speed.py          latency and throughput benchmark
configs/
  base.py                   shared recipe: paths, mixing, loss, augmentation
  unet.py  unetpp.py  segformer.py  segformer_pretrained.py
powderpolyxtalseg/
  models/                   U-Net, UNet++, SegFormer definitions
  data/                     real TIF dataset, HDF5 dataset, 3-way mixed loaders
  losses.py                 Focal, Dice and Combined losses
  metrics.py                global-pixel IoU/Dice accumulators
  engine.py                 shared training loop, DDP helpers, checkpoint I/O
scripts/                    dataset preparation
examples/                   small runnable subset
```

Most model modules run standalone as a self-test, e.g. `python -m
powderpolyxtalseg.models.unet`. The SegFormer self-test builds both
initializations and checks that the pretrained weights actually landed.

---

## Citation

```bibtex
TODO — add the BibTeX entry once the paper is published.
```

## License

MIT — see [LICENSE](LICENSE).
# PowderPolyXtalSeg
