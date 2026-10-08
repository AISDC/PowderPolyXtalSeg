"""
U-Net trained from scratch on the Mixed-v2 dataset.

Randomly initialized, so a uniform learning rate is used. Every optimizer and
schedule setting here is identical to configs/unetpp.py, so that U-Net and
UNet++ differ only in architecture — that is what makes the reported comparison
a controlled one.
"""

from .base import MixedV2Config


class UNetConfig(MixedV2Config):
    MODEL = 'unet'

    # ── Model ─────────────────────────────────────────────────────────────────
    FEATURES = 64
    BILINEAR = True

    # ── Optimizer — uniform-LR Adam (from scratch) ────────────────────────────
    OPTIMIZER      = 'adam'
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP_NORM = 1.0
    WARMUP_EPOCHS  = 0

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

    DDP_PORT = '12371'
