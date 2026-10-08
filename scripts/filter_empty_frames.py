#!/usr/bin/env python3
"""
Build filtered_indices.json for the old-synthetic HDF5 files.

A large fraction of simulated frames contain essentially no diffraction signal.
Training on them wastes throughput and pushes the loss further toward the
background class, so frames whose background fraction exceeds a threshold
(default 99.9%) are excluded up front. The surviving indices per split are what
the training dataloader iterates over.

Also reports the empirical per-class pixel frequencies. Note that the published
models did *not* use inverse-frequency weights: they use the hand-set focal
alpha [0.1, 5.0, 1.0] (see configs/base.py), which was more stable than the
~250x weight that pure inverse frequency assigns to HEDM.

Usage:
    python scripts/filter_empty_frames.py \
        --train data/synth_train.h5 --val data/synth_val.h5 \
        --out data/filtered_indices.json
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CLASS_NAMES = {0: 'Background', 1: 'HEDM', 2: 'Powder'}


def filter_empty_frames(filepath: Path, threshold: float = 0.999):
    """Return (kept_indices, stats) for one HDF5 file."""
    print(f"\n{'=' * 60}\nFiltering: {filepath}\n{'=' * 60}")

    kept = []
    stats = {'empty': 0, 'with_hedm': 0, 'with_powder': 0}

    with h5py.File(filepath, 'r') as f:
        masks = f['masks']
        total_samples = masks.shape[0]
        total_pixels  = masks.shape[1] * masks.shape[2]
        print(f"Total frames: {total_samples}   threshold: "
              f"{threshold * 100:.1f}% background")

        for i in range(total_samples):
            unique, counts = np.unique(masks[i], return_counts=True)
            bg_count = counts[unique == 0][0] if 0 in unique else 0

            if bg_count / total_pixels < threshold:
                kept.append(i)
                if 1 in unique:
                    stats['with_hedm'] += 1
                if 2 in unique:
                    stats['with_powder'] += 1
            else:
                stats['empty'] += 1

            if (i + 1) % 1000 == 0:
                print(f"  processed {i + 1}/{total_samples} ...")

    print(f"\nEmpty frames (dropped): {stats['empty']} "
          f"({stats['empty'] / total_samples * 100:.2f}%)")
    print(f"Frames kept           : {len(kept)} "
          f"({len(kept) / total_samples * 100:.2f}%)")
    print(f"  with HEDM pixels    : {stats['with_hedm']}")
    print(f"  with powder pixels  : {stats['with_powder']}")
    return kept, stats


def class_frequencies(filepaths: dict, indices: dict, max_frames: int = 1000,
                      seed: int = 42) -> dict:
    """Per-class pixel frequency over a random subset of the kept frames."""
    print(f"\n{'=' * 60}\nClass pixel frequencies\n{'=' * 60}")
    rng = np.random.default_rng(seed)
    pixel_counts = {0: 0, 1: 0, 2: 0}

    for split, filepath in filepaths.items():
        split_indices = indices[split]
        if not split_indices:
            continue
        sample = (split_indices if len(split_indices) <= max_frames
                  else rng.choice(split_indices, max_frames, replace=False))
        print(f"  {split}: sampling {len(sample)} of {len(split_indices)} frames")
        with h5py.File(filepath, 'r') as f:
            masks = f['masks']
            for idx in sorted(int(i) for i in sample):
                unique, counts = np.unique(masks[idx], return_counts=True)
                for value, count in zip(unique, counts):
                    pixel_counts[int(value)] += int(count)

    total = sum(pixel_counts.values()) or 1
    freqs = {cls: count / total for cls, count in pixel_counts.items()}
    for cls, freq in freqs.items():
        print(f"  {CLASS_NAMES[cls]:<12} (class {cls}): {freq * 100:.4f}%")
    return freqs


def main():
    parser = argparse.ArgumentParser(
        description='Filter near-empty synthetic frames and report class balance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', required=True, type=Path, help='Training HDF5 file')
    parser.add_argument('--val', required=True, type=Path, help='Validation HDF5 file')
    parser.add_argument('--test', type=Path, default=None, help='Optional test HDF5 file')
    parser.add_argument('--out', required=True, type=Path,
                        help='Where to write filtered_indices.json')
    parser.add_argument('--threshold', type=float, default=0.999,
                        help='Background fraction above which a frame is "empty"')
    args = parser.parse_args()

    files = {'train': args.train, 'val': args.val}
    if args.test is not None:
        files['test'] = args.test

    indices, stats = {}, {}
    for split, filepath in files.items():
        indices[split], stats[split] = filter_empty_frames(filepath, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(indices, f, indent=2)
    print(f"\nSaved filtered indices to {args.out}")

    freqs = class_frequencies(files, indices)
    weights_path = args.out.with_name('class_weights.json')
    with open(weights_path, 'w') as f:
        json.dump({
            'focal_alpha_used_by_published_models': [0.1, 5.0, 1.0],
            'measured_pixel_frequency': {CLASS_NAMES[c]: freqs[c] for c in freqs},
            'note': 'Inverse-frequency weights were unstable for HEDM; the '
                    'published runs use the hand-set alpha above.',
        }, f, indent=2)
    print(f"Saved class statistics to {weights_path}")

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for split in files:
        kept     = len(indices[split])
        original = kept + stats[split]['empty']
        print(f"{split:<6}: {original:6d} -> {kept:6d} "
              f"({stats[split]['empty']} removed, {kept / original * 100:.1f}% kept)")


if __name__ == '__main__':
    main()
