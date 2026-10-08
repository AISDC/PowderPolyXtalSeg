"""
SegFormer MiT-B2 with an ImageNet-pretrained encoder, fine-tuned on Mixed-v2.

This run exists as a control. Training a hierarchical transformer from scratch
on ~17k images is a severe handicap — transformers lack the locality prior of a
CNN and normally rely on ImageNet pretraining — so the from-scratch SegFormer of
configs/segformer.py cannot on its own separate "the architecture is unsuited to
sparse HEDM spots" from "the encoder never got its pretrained features". Loading
the real pretrained weights settles the question.

Everything except the initialization and the learning rate is inherited from
SegFormerConfig, so those two are the only variables under test.
"""

from .segformer import SegFormerConfig


class SegFormerPretrainedConfig(SegFormerConfig):
    MODEL = 'segformer_pretrained'

    # ── Model ─────────────────────────────────────────────────────────────────
    BACKBONE            = 'nvidia/mit-b2'
    PRETRAINED_BACKBONE = True   # load ImageNet weights, not just the config

    # How to adapt the pretrained RGB stem (3ch) to grayscale (1ch):
    #   'sum'  — sum the three input-channel kernels, which preserves the
    #            response magnitude a gray image would have produced as an RGB
    #            triplet (standard practice)
    #   'mean' — average them, which preserves the per-channel scale
    GRAYSCALE_INIT = 'sum'

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # 6e-5 is the SegFormer reference fine-tuning LR (ADE20K/Cityscapes). The
    # from-scratch run uses 1.5e-4, which would wash out the pretrained features.
    # MiT is fine-tuned end-to-end in the reference recipe rather than with a
    # frozen or lower-LR backbone, so a single uniform LR is kept here too.
    LEARNING_RATE = 6e-5

    DDP_PORT = '12372'
