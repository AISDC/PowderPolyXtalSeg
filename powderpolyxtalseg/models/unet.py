#!/usr/bin/env python3
"""
U-Net model implementation for semantic segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv -> BN -> ReLU) * 2
    """
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


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled features from previous layer
            x2: Skip connection features from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch (if input size is not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 conv to map to num_classes
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        features: Number of features in first layer (default: 64)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, in_channels=1, num_classes=3, features=64, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # Decoder (upsampling path)
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # Output layer
        self.outc = OutConv(features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor. Shape: (N, C, H, W)

        Returns:
            Output logits. Shape: (N, num_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the U-Net implementation
    print("="*60)
    print("Testing U-Net Implementation")
    print("="*60)

    # Create model
    model = UNet(in_channels=1, num_classes=3, features=64, bilinear=True)
    print(f"\nModel created:")
    print(f"  Input channels: 1 (grayscale)")
    print(f"  Output classes: 3")
    print(f"  Features: 64")
    print(f"  Parameters: {count_parameters(model):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 512, 512)
    print(f"  Input shape: {input_tensor.shape}")

    output = model(input_tensor)
    print(f"  Output shape: {output.shape}")

    # Test with different input sizes
    print("\nTesting different input sizes...")
    for size in [256, 512, 768, 1024]:
        input_tensor = torch.randn(1, 1, size, size)
        output = model(input_tensor)
        print(f"  Input: {input_tensor.shape} -> Output: {output.shape}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    input_tensor = torch.randn(1, 1, 512, 512, requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    print(f"  ✓ Gradients computed successfully!")
    print(f"  Gradient shape: {input_tensor.grad.shape}")

    # Model summary
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {count_parameters(model):,}")

    # Calculate model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_mb:.2f} MB")

    print("="*60)
    print("All tests passed!")
    print("="*60)
