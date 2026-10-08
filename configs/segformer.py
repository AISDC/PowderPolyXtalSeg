"""
SegFormer MiT-B2 trained from scratch on Mixed-v2.

The HuggingFace backbone is instantiated from its *config* only — the weights are
random, not ImageNet-pretrained, because the input is single-channel detector
data. Transformer conventions apply: AdamW, a short linear warmup, and gradient
clipping.
"""

from .base import MixedV2Config


class SegFormerConfig(MixedV2Config):
    MODEL = 'segformer'

    # ── Model ─────────────────────────────────────────────────────────────────
    BACKBONE = 'nvidia/mit-b2'   # architecture config only; weights are random

    # ── Optimizer — uniform-LR AdamW with linear warmup ───────────────────────
    OPTIMIZER      = 'adamw'
    LEARNING_RATE  = 1.5e-4      # scaled for an effective batch of 64 (8 GPUs)
    WEIGHT_DECAY   = 0.01
    GRAD_CLIP_NORM = 1.0
    WARMUP_EPOCHS  = 3

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE = 8       # fits 512^2 with AMP on a V100-32GB
    NUM_EPOCHS = 200

    # ── LR scheduler ──────────────────────────────────────────────────────────
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR   = 0.5
    MIN_LR             = 1e-6

    # ── Early stopping ────────────────────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 30

    # ── Data loading ──────────────────────────────────────────────────────────
    NUM_WORKERS = 6

    DDP_PORT = '12359'
