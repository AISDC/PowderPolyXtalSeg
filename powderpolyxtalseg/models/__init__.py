"""Model registry — `build_model(cfg)` returns the network for `cfg.MODEL`."""

from .unet import UNet, count_parameters
from .unetpp import UNetPP


def build_model(cfg):
    """
    Instantiate the network described by a config class.

    All networks take a single-channel 512x512 input and produce
    `cfg.NUM_CLASSES` logits at full resolution. UNet++ with deep supervision
    returns a list of four such tensors (coarse to fine).

    'segformer' and 'segformer_pretrained' share one architecture and differ
    only in initialization, so they share one builder.
    """
    if cfg.MODEL == 'unet':
        return UNet(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
            features=cfg.FEATURES,
            bilinear=cfg.BILINEAR,
        )

    if cfg.MODEL == 'unetpp':
        return UNetPP(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
            features=cfg.FEATURES,
            deep_supervision=cfg.USE_DEEP_SUPERVISION,
        )

    if cfg.MODEL in ('segformer', 'segformer_pretrained'):
        # Imported lazily: only these models need `transformers` installed.
        from .segformer import create_segformer
        return create_segformer(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
            backbone=cfg.BACKBONE,
            pretrained=cfg.PRETRAINED_BACKBONE,
            grayscale_init=cfg.GRAYSCALE_INIT,
        )

    raise ValueError(f"Unknown model '{cfg.MODEL}'")


__all__ = ['build_model', 'count_parameters', 'UNet', 'UNetPP']
