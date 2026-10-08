"""
Segmentation metrics.

IoU and Dice are computed from *global* pixel counts accumulated over the whole
validation set (and, under DDP, over all ranks), not by averaging per-batch
values. With 95% background and 0.4% HEDM pixels, per-batch averaging would be
dominated by batches that happen to contain no HEDM pixels at all.

Accumulator layout: counts[class] = [intersection, pred_sum, target_sum].
"""

import numpy as np
import torch


def new_counts(num_classes: int, device) -> torch.Tensor:
    """Zeroed accumulator, float64 so large pixel counts stay exact."""
    return torch.zeros(num_classes, 3, device=device, dtype=torch.float64)


def accumulate_counts(counts: torch.Tensor, preds: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """Add one batch of predictions/labels into the accumulator (in place)."""
    for cls in range(counts.shape[0]):
        pred_cls = (preds == cls)
        tgt_cls  = (targets == cls)
        counts[cls, 0] += (pred_cls & tgt_cls).sum()
        counts[cls, 1] += pred_cls.sum()
        counts[cls, 2] += tgt_cls.sum()
    return counts


def counts_to_metrics(counts: torch.Tensor) -> dict:
    """Per-class IoU/Dice plus their unweighted means."""
    num_classes = counts.shape[0]
    metrics = {}
    for cls in range(num_classes):
        inter    = counts[cls, 0].item()
        pred_sum = counts[cls, 1].item()
        tgt_sum  = counts[cls, 2].item()
        union    = pred_sum + tgt_sum - inter
        metrics[f'iou_class_{cls}']  = inter / (union + 1e-7)
        metrics[f'dice_class_{cls}'] = 2 * inter / (pred_sum + tgt_sum + 1e-7)
    metrics['mean_iou'] = float(np.mean(
        [metrics[f'iou_class_{c}'] for c in range(num_classes)]))
    metrics['mean_dice'] = float(np.mean(
        [metrics[f'dice_class_{c}'] for c in range(num_classes)]))
    return metrics


def combined_metric(cfg, real_miou: float, old_synth_miou: float,
                    ring_miou: float) -> float:
    """Weighted mIoU across the three validation domains (0.2 / 0.2 / 0.6)."""
    return (cfg.REAL_METRIC_WEIGHT      * real_miou +
            cfg.OLD_SYNTH_METRIC_WEIGHT * old_synth_miou +
            cfg.RING_METRIC_WEIGHT      * ring_miou)
