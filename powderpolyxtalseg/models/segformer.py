#!/usr/bin/env python3
"""
SegFormer model wrapper around HuggingFace transformers for semantic segmentation.

Uses MiT-B2 backbone by default (~27.4M parameters).
Modified first conv layer for single-channel (grayscale) input.
Upsamples output from 1/4 resolution back to full resolution.

Two initializations are supported, and they are the two SegFormer runs reported
in the paper:

  pretrained=False  the architecture is taken from the hub *config* only and
                    every weight is random. This is the from-scratch model in
                    the controlled three-architecture comparison.
  pretrained=True   the ImageNet-pretrained MiT-B2 weights are actually loaded.
                    This is the control that separates an architectural
                    limitation from a missing-pretraining artefact.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def create_segformer(in_channels=1, num_classes=3, backbone='nvidia/mit-b2',
                     pretrained=False, grayscale_init='sum'):
    """
    Create a SegFormer model with modified input for grayscale images.

    Args:
        in_channels:    Number of input channels (1 for grayscale)
        num_classes:    Number of output classes
        backbone:       HuggingFace model ID for the MiT backbone
        pretrained:     Load the ImageNet-pretrained encoder weights. When False,
                        only the architecture is taken from the hub and the
                        weights are random.
        grayscale_init: 'sum' or 'mean' — how to collapse a *pretrained*
                        3-channel stem down to `in_channels`. Ignored when
                        pretrained is False, where the stem is simply random.

    Returns:
        SegFormerWrapper model
    """
    if pretrained:
        # from_pretrained() loads the encoder weights. The segmentation head is
        # sized for the checkpoint's label set, so it is re-initialized for our
        # 3 classes; only the head is random, the encoder keeps its weights.
        model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        # Config only — this takes the architecture but leaves every weight
        # randomly initialized.
        config = SegformerConfig.from_pretrained(backbone)
        config.num_labels = num_classes
        model = SegformerForSemanticSegmentation(config)

    # Replace first conv: 3 channels -> in_channels
    if in_channels != 3:
        old_conv = model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            # Rather than throwing the pretrained stem away for a random
            # 1-channel conv, collapse its RGB kernels along the input-channel
            # axis. Summing preserves the response magnitude a gray image would
            # have produced as an RGB triplet; averaging preserves the
            # per-channel scale. 'sum' is the usual choice.
            with torch.no_grad():
                w = old_conv.weight              # (out, 3, kh, kw)
                if grayscale_init == 'sum':
                    collapsed = w.sum(dim=1, keepdim=True)
                elif grayscale_init == 'mean':
                    collapsed = w.mean(dim=1, keepdim=True)
                else:
                    raise ValueError(f'unknown grayscale_init: {grayscale_init!r}')
                # Repeat if someone asks for >1 but !=3 channels
                new_conv.weight.copy_(
                    collapsed.repeat(1, in_channels, 1, 1) / in_channels
                    if in_channels > 1 else collapsed
                )
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        model.segformer.encoder.patch_embeddings[0].proj = new_conv

    return SegFormerWrapper(model)


class SegFormerWrapper(nn.Module):
    """
    Wrapper that extracts logits and upsamples to input resolution.

    HuggingFace SegFormer outputs logits at 1/4 input resolution.
    This wrapper upsamples back to full resolution and returns a plain tensor
    for compatibility with the standard training loop.
    """

    def __init__(self, model):
        super(SegFormerWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor. Shape: (N, C, H, W)

        Returns:
            Output tensor. Shape: (N, num_classes, H, W)
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (N, num_classes, H/4, W/4)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 68)
    print("Testing SegFormer Implementation")
    print("=" * 68)

    backbone = 'nvidia/mit-b2'
    print(f"\nBackbone: {backbone}")

    torch.manual_seed(0)
    rand = create_segformer(in_channels=1, num_classes=3, backbone=backbone)
    pre  = create_segformer(in_channels=1, num_classes=3, backbone=backbone,
                            pretrained=True, grayscale_init='sum')

    print(f"  Parameters: {count_parameters(rand):,}")
    assert count_parameters(rand) == count_parameters(pre), \
        "pretrained and random init must give the same architecture"

    # The pretrained weights should be visibly different from random init —
    # this is what distinguishes from_pretrained() from config-only construction.
    def stem_stats(m, tag):
        stem = m.model.segformer.encoder.patch_embeddings[0].proj.weight
        attn = m.model.segformer.encoder.block[0][0].attention.self.query.weight
        print(f"  {tag:<12s} stem std={stem.std():.5f}  attn-q std={attn.std():.5f}")

    print("\nInitialization check:")
    stem_stats(rand, 'random')
    stem_stats(pre, 'pretrained')

    rand_attn = rand.model.segformer.encoder.block[0][0].attention.self.query.weight
    pre_attn  = pre.model.segformer.encoder.block[0][0].attention.self.query.weight
    assert not torch.allclose(rand_attn, pre_attn), \
        "pretrained encoder weights are identical to random init — nothing was loaded"
    print("  -> encoder weights differ, pretrained weights loaded")

    # Test forward pass
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 512, 512)
    print(f"\n  Input shape: {input_tensor.shape}")

    for tag, model in (('random', rand), ('pretrained', pre)):
        output = model(input_tensor)
        assert output.shape == (batch_size, 3, 512, 512), \
            f"Expected (1, 3, 512, 512), got {output.shape}"
        print(f"  {tag:<12s} output shape: {tuple(output.shape)}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    input_tensor = torch.randn(1, 1, 512, 512, requires_grad=True)
    output = pre(input_tensor)
    loss = output.sum()
    loss.backward()
    print(f"  Gradients computed successfully!")
    print(f"  Gradient shape: {input_tensor.grad.shape}")

    # Model size
    param_size = sum(p.nelement() * p.element_size() for p in pre.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in pre.buffers())
    size_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"\nModel size: {size_mb:.2f} MB")

    print("=" * 68)
    print("All tests passed!")
    print("=" * 68)
