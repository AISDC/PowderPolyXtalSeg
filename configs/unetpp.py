"""
UNet++ (nested U-Net with deep supervision) trained from scratch on Mixed-v2.

Randomly initialized, so a uniform learning rate is used. The training loss is a
weighted sum over the four deep-supervision heads; validation metrics use only
the final head (X^{0,4}).
"""

from .base import MixedV2Config


class UNetPPConfig(MixedV2Config):
    MODEL = 'unetpp'

    # ── Model ─────────────────────────────────────────────────────────────────
    FEATURES                 = 64
    USE_DEEP_SUPERVISION     = True
    DEEP_SUPERVISION_WEIGHTS = [0.25, 0.25, 0.25, 0.25]   # 4 output heads

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

    DDP_PORT = '12361'
