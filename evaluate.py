#!/usr/bin/env python3
"""
Evaluate one or more trained checkpoints on the three validation domains
(real hand-labelled, old synthetic, ring synthetic) and report per-class IoU,
per-class Dice and the weighted combined mIoU.

Usage
-----
    # A single model
    python evaluate.py --model unetpp --checkpoint runs/unetpp_mixedv2/checkpoints/best_model.pth

    # All four side by side (the per-class IoU comparison from the paper)
    python evaluate.py \
        --model unet       --checkpoint runs/unet_mixedv2/checkpoints/best_model.pth \
        --model unetpp     --checkpoint runs/unetpp_mixedv2/checkpoints/best_model.pth \
        --model segformer  --checkpoint runs/segformer_mixedv2/checkpoints/best_model.pth \
        --model segformer_pretrained \
            --checkpoint runs/segformer_pretrained_mixedv2/checkpoints/best_model.pth

    # JSON output for downstream plotting
    python evaluate.py --model unet --checkpoint best.pth --json results.json
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from configs import MODEL_NAMES, get_config
from powderpolyxtalseg import CLASS_NAMES
from powderpolyxtalseg.data import build_mixed_dataloaders
from powderpolyxtalseg.engine import final_head, load_model_weights
from powderpolyxtalseg.metrics import (
    accumulate_counts,
    combined_metric,
    counts_to_metrics,
    new_counts,
)
from powderpolyxtalseg.models import build_model

DOMAINS = ('real', 'old_synth', 'ring')


@torch.no_grad()
def evaluate_split(model, loader, device, cfg, split_name: str) -> dict:
    model.eval()
    counts = new_counts(cfg.NUM_CLASSES, device)
    for images, masks in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        if cfg.USE_AMP and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = final_head(model(images))
        else:
            logits = final_head(model(images))
        accumulate_counts(counts, torch.argmax(logits, dim=1), masks)
    return counts_to_metrics(counts)


def print_table(results: dict):
    """results: {model_name: {domain: metrics, 'combined': float}}"""
    models = list(results)
    col = max(len(name) for name in CLASS_NAMES) + 6
    # Wide enough for the longest run name ('segformer_pretrained')
    mcol = max([len(name) for name in models] + [len('Model')]) + 2
    header = (f"{'Domain':<12}{'Model':<{mcol}}{'mIoU':>9}"
              + "".join(f"{name + ' IoU':>{col}}" for name in CLASS_NAMES)
              + f"{'mDice':>10}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for domain in DOMAINS:
        for name in models:
            m = results[name][domain]
            print(f"{domain if name == models[0] else '':<12}{name:<{mcol}}"
                  f"{m['mean_iou']:>9.4f}"
                  + "".join(f"{m[f'iou_class_{c}']:>{col}.4f}"
                            for c in range(len(CLASS_NAMES)))
                  + f"{m['mean_dice']:>10.4f}")
        print("-" * len(header))
    for name in models:
        print(f"{'combined' if name == models[0] else '':<12}{name:<{mcol}}"
              f"{results[name]['combined']:>9.4f}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained checkpoints on the three validation domains')
    parser.add_argument('--model', action='append', required=True, choices=MODEL_NAMES,
                        help='Model architecture; repeat to compare several models')
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='Checkpoint path, paired with the --model at the same position')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset directory (default: $POWDERSEG_DATA_ROOT or ./data)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override the number of dataloader workers')
    parser.add_argument('--json', type=str, default=None,
                        help='Also write the results to this JSON file')
    args = parser.parse_args()

    if len(args.model) != len(args.checkpoint):
        parser.error(f"Got {len(args.model)} --model and {len(args.checkpoint)} "
                     f"--checkpoint arguments; they must be paired.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data settings are shared by every model, so the loaders are built once
    # from the first config; only the batch size could differ, and it does not
    # affect the metrics (they are accumulated over global pixel counts).
    loader_cfg = get_config(args.model[0])
    if args.batch_size is not None:
        loader_cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        loader_cfg.NUM_WORKERS = args.num_workers
    loader_cfg.resolve(data_root=args.data_root, output_dir=Path('runs') / 'eval')

    _, real_loader, old_synth_loader, ring_loader = build_mixed_dataloaders(
        loader_cfg, world_size=1, rank=0)
    loaders = {'real': real_loader, 'old_synth': old_synth_loader, 'ring': ring_loader}

    results = {}
    for model_name, ckpt_path in zip(args.model, args.checkpoint):
        cfg = get_config(model_name)
        cfg.resolve(data_root=args.data_root, output_dir=Path('runs') / 'eval')

        print(f"\n=== {model_name} : {ckpt_path} ===")
        model = build_model(cfg).to(device)
        load_model_weights(model, ckpt_path, device, strict=True)

        entry = {domain: evaluate_split(model, loaders[domain], device, cfg, domain)
                 for domain in DOMAINS}
        entry['combined'] = combined_metric(
            cfg,
            entry['real']['mean_iou'],
            entry['old_synth']['mean_iou'],
            entry['ring']['mean_iou'],
        )
        entry['checkpoint'] = str(ckpt_path)
        # A repeated architecture would otherwise overwrite its own row
        key = model_name if model_name not in results else f'{model_name}({ckpt_path})'
        results[key] = entry

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print_table(results)
    print(f"\nCombined metric weights: real={loader_cfg.REAL_METRIC_WEIGHT} "
          f"old_synth={loader_cfg.OLD_SYNTH_METRIC_WEIGHT} "
          f"ring={loader_cfg.RING_METRIC_WEIGHT}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == '__main__':
    main()
