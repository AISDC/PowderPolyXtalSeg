#!/usr/bin/env python3
"""
Focal Loss implementation for handling class imbalance in segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection/segmentation.

    Reference:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV 2017.

    Args:
        alpha: Class weights (tensor or list). Shape: (num_classes,)
        gamma: Focusing parameter. Higher values give more focus to hard examples.
               Typical values: 0 (no focusing), 1, 2 (default), 5
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Class index to ignore in loss calculation (e.g., for unlabeled pixels)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Convert alpha to tensor if provided
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs: Model predictions (logits). Shape: (N, C, H, W) or (N, C)
            targets: Ground truth labels. Shape: (N, H, W) or (N,)

        Returns:
            Focal loss value
        """
        # Compute standard cross-entropy loss (with reduction='none' to get per-pixel loss)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index
        )

        # Get predicted probabilities for the true class
        # pt = exp(-ce_loss) is the probability of the true class
        pt = torch.exp(-ce_loss)

        # Compute focal loss: FL = -(1 - pt)^gamma * log(pt)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            # Move alpha to the same device as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Create alpha_t: weight for each pixel based on its target class
            # Handle ignore_index by creating a mask
            if self.ignore_index >= 0:
                # Create a valid mask
                valid_mask = (targets != self.ignore_index)

                # Get alpha values for valid targets
                alpha_t = torch.zeros_like(targets, dtype=torch.float32)
                for c in range(len(self.alpha)):
                    alpha_t[targets == c] = self.alpha[c]

                # Apply alpha weighting
                focal_loss = alpha_t * focal_loss

                # Apply valid mask
                focal_loss = focal_loss * valid_mask.float()
            else:
                # No ignore index, simpler case
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                # Only average over valid (non-ignored) pixels
                valid_mask = (targets != self.ignore_index)
                return focal_loss.sum() / (valid_mask.sum() + 1e-7)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss for better segmentation performance.

    Args:
        alpha: Class weights for Focal Loss
        gamma: Focusing parameter for Focal Loss
        focal_weight: Weight for focal loss component (default: 0.5)
        dice_weight: Weight for dice loss component (default: 0.5)
    """

    def __init__(self, alpha=None, gamma=2.0, focal_weight=0.5, dice_weight=0.5, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean', ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def dice_loss(self, inputs, targets, smooth=1.0):
        """
        Compute Dice loss.

        Args:
            inputs: Model predictions (logits). Shape: (N, C, H, W)
            targets: Ground truth labels. Shape: (N, H, W)
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)

        # Get number of classes
        num_classes = inputs.shape[1]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # Handle ignore_index
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # (N, 1, H, W)
        else:
            valid_mask = torch.ones_like(probs[:, 0:1])

        # Compute Dice coefficient per class
        dice_losses = []
        for c in range(num_classes):
            pred_c = probs[:, c:c+1] * valid_mask
            target_c = targets_one_hot[:, c:c+1] * valid_mask

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
            dice_losses.append(1.0 - dice_coeff)

        # Return mean dice loss across classes
        return torch.stack(dice_losses).mean()

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs: Model predictions (logits). Shape: (N, C, H, W)
            targets: Ground truth labels. Shape: (N, H, W)

        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        combined = self.focal_weight * focal + self.dice_weight * dice
        return combined


if __name__ == "__main__":
    # Test the Focal Loss implementation
    print("="*60)
    print("Testing Focal Loss Implementation")
    print("="*60)

    # Create dummy data
    batch_size = 4
    num_classes = 3
    height, width = 512, 512

    # Random predictions (logits)
    predictions = torch.randn(batch_size, num_classes, height, width)

    # Random targets
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test 1: Focal Loss without class weights
    print("\nTest 1: Focal Loss (no class weights)")
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(predictions, targets)
    print(f"  Loss: {loss.item():.4f}")

    # Test 2: Focal Loss with class weights
    print("\nTest 2: Focal Loss (with class weights)")
    alpha = [0.1, 5.0, 1.0]  # Background, HEDM, Powder
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    loss = criterion(predictions, targets)
    print(f"  Loss: {loss.item():.4f}")

    # Test 3: Combined Loss
    print("\nTest 3: Combined Loss (Focal + Dice)")
    criterion = CombinedLoss(alpha=alpha, gamma=2.0, focal_weight=0.5, dice_weight=0.5)
    loss = criterion(predictions, targets)
    print(f"  Loss: {loss.item():.4f}")

    # Test 4: Gradient flow
    print("\nTest 4: Testing gradient flow")
    predictions.requires_grad = True
    loss = criterion(predictions, targets)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient shape: {predictions.grad.shape}")
    print(f"  Gradient mean: {predictions.grad.mean().item():.6f}")
    print(f"  ✓ Gradients computed successfully!")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
