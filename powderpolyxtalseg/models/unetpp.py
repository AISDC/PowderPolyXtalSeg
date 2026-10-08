#!/usr/bin/env python3
"""
UNet++ (Nested U-Net) model implementation for semantic segmentation.

Reference:
    Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018).
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
    Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical
    Decision Support (DLMIA 2018).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block: (Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPP(nn.Module):
    """
    UNet++ (Nested U-Net) architecture for semantic segmentation.

    Dense nested skip connections between encoder and decoder at multiple levels.
    Optional deep supervision with output heads at multiple decoder depths.

    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes
        features: Number of features in first layer (default: 64)
        deep_supervision: Whether to use deep supervision (default: True)
    """
    def __init__(self, in_channels=1, num_classes=3, features=64, deep_supervision=True):
        super(UNetPP, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        nb_filter = [features, features * 2, features * 4, features * 8, features * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder (backbone) nodes: X^{i,0}
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Dense decoder nodes
        # X^{0,1}: receives X^{0,0} + up(X^{1,0})
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        # X^{1,1}: receives X^{1,0} + up(X^{2,0})
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        # X^{2,1}: receives X^{2,0} + up(X^{3,0})
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        # X^{3,1}: receives X^{3,0} + up(X^{4,0})
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        # X^{0,2}: receives X^{0,0} + X^{0,1} + up(X^{1,1})
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        # X^{1,2}: receives X^{1,0} + X^{1,1} + up(X^{2,1})
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        # X^{2,2}: receives X^{2,0} + X^{2,1} + up(X^{3,1})
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        # X^{0,3}: receives X^{0,0} + X^{0,1} + X^{0,2} + up(X^{1,2})
        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        # X^{1,3}: receives X^{1,0} + X^{1,1} + X^{1,2} + up(X^{2,2})
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        # X^{0,4}: receives X^{0,0} + X^{0,1} + X^{0,2} + X^{0,3} + up(X^{1,3})
        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # Deep supervision output heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor. Shape: (N, C, H, W)

        Returns:
            If deep_supervision=True: list of 4 output tensors [X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}]
            If deep_supervision=False: single output tensor from X^{0,4}
        """
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Dense decoder - column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_cat(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_cat(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_cat(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_cat(x4_0, x3_0)], dim=1))

        # Dense decoder - column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_cat(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_cat(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_cat(x3_1, x2_0)], dim=1))

        # Dense decoder - column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_cat(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_cat(x2_2, x1_0)], dim=1))

        # Dense decoder - column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_cat(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            return self.final(x0_4)

    def _up_cat(self, x_lower, x_target):
        """Upsample x_lower and pad to match x_target spatial dims."""
        x_up = self.up(x_lower)
        # Handle size mismatch
        diffY = x_target.size()[2] - x_up.size()[2]
        diffX = x_target.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        return x_up


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing UNet++ Implementation")
    print("=" * 60)

    # Test with deep supervision
    model = UNetPP(in_channels=1, num_classes=3, features=64, deep_supervision=True)
    print(f"\nModel created (deep_supervision=True):")
    print(f"  Input channels: 1 (grayscale)")
    print(f"  Output classes: 3")
    print(f"  Features: 64")
    print(f"  Parameters: {count_parameters(model):,}")

    # Test forward pass
    print("\nTesting forward pass (deep supervision)...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 512, 512)
    print(f"  Input shape: {input_tensor.shape}")

    outputs = model(input_tensor)
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i} shape: {out.shape}")

    # Test without deep supervision
    model_no_ds = UNetPP(in_channels=1, num_classes=3, features=64, deep_supervision=False)
    print(f"\nModel created (deep_supervision=False):")
    print(f"  Parameters: {count_parameters(model_no_ds):,}")

    output = model_no_ds(input_tensor)
    print(f"  Output shape: {output.shape}")

    # Test different input sizes
    print("\nTesting different input sizes...")
    for size in [256, 512, 768]:
        input_tensor = torch.randn(1, 1, size, size)
        outputs = model(input_tensor)
        print(f"  Input: {input_tensor.shape} -> Output: {outputs[-1].shape}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    input_tensor = torch.randn(1, 1, 512, 512, requires_grad=True)
    outputs = model(input_tensor)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    print(f"  Gradients computed successfully!")
    print(f"  Gradient shape: {input_tensor.grad.shape}")

    # Model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"\nModel size: {size_mb:.2f} MB")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
