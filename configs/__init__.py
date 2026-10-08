"""
Configuration registry.

`get_config('unet' | 'unetpp' | 'segformer' | 'segformer_pretrained')` returns
the config class for one of the four published runs — three architectures, plus
SegFormer with an ImageNet-pretrained encoder as a control on the effect of
initialization.
"""

from .base import MixedV2Config
from .unet import UNetConfig
from .unetpp import UNetPPConfig
from .segformer import SegFormerConfig
from .segformer_pretrained import SegFormerPretrainedConfig

CONFIGS = {
    'unet':                UNetConfig,
    'unetpp':              UNetPPConfig,
    'segformer':           SegFormerConfig,
    'segformer_pretrained': SegFormerPretrainedConfig,
}

MODEL_NAMES = tuple(CONFIGS)


def get_config(name):
    """Return the config class for a model name."""
    try:
        return CONFIGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}"
        ) from None


__all__ = [
    'MixedV2Config', 'UNetConfig', 'UNetPPConfig', 'SegFormerConfig',
    'SegFormerPretrainedConfig', 'CONFIGS', 'MODEL_NAMES', 'get_config',
]
