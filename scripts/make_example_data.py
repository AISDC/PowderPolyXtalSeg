#!/usr/bin/env python3
"""
Build the small `examples/` dataset bundled with this repository.

The full training set is several gigabytes, so the repository ships a tiny
subset with the same directory layout and the same HDF5 schema: enough to run
every command in the README end to end, far too little to reproduce the
published metrics.

This script is kept in the repository so the subset is reproducible; users who
already have the full dataset do not need to run it.

Usage:
    python scripts/make_example_data.py \
        --real-src   /path/to/hand_labelled_data_cliplog \
        --synth-train /path/to/train.h5 --synth-val /path/to/val.h5 \
        --synth-indices /path/to/filtered_indices.json \
        --ring-train /path/to/ring_train.h5 --ring-val /path/to/ring_val.h5 \
        --out examples
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile


def crop_real_sample(sample_dir: Path, out_dir: Path, crop: int, seed: int):
    """
    Copy one hand-labelled sample, cropped to the `crop`x`crop` window with the
    most foreground pixels (so the example patches are not all background).
    """
    raw_tifs = [f for f in sorted(sample_dir.glob('*.tif'))
                if 'labkit_segmented' not in f.name]
    if len(raw_tifs) != 1:
        raise ValueError(f"Expected 1 raw TIF in {sample_dir}, found {raw_tifs}")

    img  = tifffile.imread(raw_tifs[0]).astype(np.float32)
    mask = tifffile.imread(sample_dir / 'labkit_segmented.tif')

    H, W = mask.shape
    if H < crop or W < crop:
        raise ValueError(f"{sample_dir.name} is smaller than the crop window")

    # Coarse search over a grid of candidate windows
    best, best_fg = (0, 0), -1
    step = max(crop // 4, 1)
    for r in range(0, H - crop + 1, step):
        for c in range(0, W - crop + 1, step):
            fg = int((mask[r:r + crop, c:c + crop] > 0).sum())
            if fg > best_fg:
                best, best_fg = (r, c), fg
    r, c = best

    img_crop  = img[r:r + crop, c:c + crop]
    mask_crop = mask[r:r + crop, c:c + crop].astype(np.uint8)

    dst = out_dir / sample_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst / raw_tifs[0].name, img_crop,
                     compression='zlib', compressionargs={'level': 6})
    tifffile.imwrite(dst / 'labkit_segmented.tif', mask_crop,
                     compression='zlib', compressionargs={'level': 6})

    fg_frac = best_fg / (crop * crop)
    print(f"  {sample_dir.name}: crop at ({r}, {c}) {crop}x{crop}, "
          f"foreground {fg_frac:.2%}, classes={np.unique(mask_crop).tolist()}")


def subset_h5(src: Path, dst: Path, indices, chunk: int = 16):
    """Copy the selected frames of an images/masks HDF5 file."""
    indices = sorted(int(i) for i in indices)
    with h5py.File(src, 'r') as fin:
        images, masks = fin['images'], fin['masks']
        n, h, w = len(indices), images.shape[1], images.shape[2]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, 'w') as fout:
            out_img = fout.create_dataset('images', shape=(n, h, w), dtype=np.float32,
                                          chunks=(min(chunk, n), h, w),
                                          compression='gzip', compression_opts=4)
            out_msk = fout.create_dataset('masks', shape=(n, h, w), dtype=np.uint8,
                                          chunks=(min(chunk, n), h, w),
                                          compression='gzip', compression_opts=4)
            for out_i, src_i in enumerate(indices):
                out_img[out_i] = images[src_i]
                out_msk[out_i] = masks[src_i]
            fout.attrs['source'] = src.name
            fout.attrs['subset_of'] = images.shape[0]
    size_mb = dst.stat().st_size / 1e6
    print(f"  {dst.name}: {n} frames ({size_mb:.1f} MB)")
    return n


def pick_frames(h5_path: Path, candidates, count: int, seed: int, min_fg=0.002):
    """Choose `count` frame indices that actually contain foreground."""
    rng = np.random.default_rng(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    chosen = []
    with h5py.File(h5_path, 'r') as f:
        masks = f['masks']
        total_pixels = masks.shape[1] * masks.shape[2]
        for idx in pool:
            if (masks[idx] > 0).sum() / total_pixels >= min_fg:
                chosen.append(int(idx))
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise RuntimeError(f"Only found {len(chosen)}/{count} frames with "
                           f">={min_fg:.1%} foreground in {h5_path}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description='Build the bundled examples/ subset')
    parser.add_argument('--real-src', required=True, type=Path,
                        help='Normalized hand-labelled sample directories')
    parser.add_argument('--real-samples', nargs='+', default=None,
                        help='Sample directory names to include (default: first two)')
    parser.add_argument('--real-crop', type=int, default=1024,
                        help='Crop window kept from each real sample')
    parser.add_argument('--synth-train', required=True, type=Path)
    parser.add_argument('--synth-val', required=True, type=Path)
    parser.add_argument('--synth-indices', required=True, type=Path)
    parser.add_argument('--ring-train', required=True, type=Path)
    parser.add_argument('--ring-val', required=True, type=Path)
    parser.add_argument('--n-train', type=int, default=48,
                        help='Frames per training HDF5 file')
    parser.add_argument('--n-val', type=int, default=16,
                        help='Frames per validation HDF5 file')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = args.out
    (out / 'real').mkdir(parents=True, exist_ok=True)

    # ── Real samples ──────────────────────────────────────────────────────────
    available = sorted([d for d in args.real_src.iterdir() if d.is_dir()])
    if args.real_samples:
        names = set(args.real_samples)
        selected = [d for d in available if d.name in names]
        missing = names - {d.name for d in selected}
        if missing:
            raise SystemExit(f"Sample(s) not found in {args.real_src}: {sorted(missing)}")
    else:
        selected = available[:2]

    print(f"Real samples -> {out / 'real'}")
    for sample_dir in selected:
        crop_real_sample(sample_dir, out / 'real', args.real_crop, args.seed)

    # ── Old synthetic ─────────────────────────────────────────────────────────
    with open(args.synth_indices) as f:
        full_indices = json.load(f)

    print(f"\nOld synthetic -> {out}")
    train_ids = pick_frames(args.synth_train, full_indices['train'],
                            args.n_train, args.seed)
    val_ids   = pick_frames(args.synth_val, full_indices['val'],
                            args.n_val, args.seed + 1)
    n_train = subset_h5(args.synth_train, out / 'synth_train.h5', train_ids)
    n_val   = subset_h5(args.synth_val,   out / 'synth_val.h5',   val_ids)

    # The subset is re-indexed from 0, so the index file must be rebuilt
    with open(out / 'filtered_indices.json', 'w') as f:
        json.dump({'train': list(range(n_train)), 'val': list(range(n_val))},
                  f, indent=2)
    print(f"  filtered_indices.json: {n_train} train / {n_val} val")

    # ── Ring synthetic ────────────────────────────────────────────────────────
    print(f"\nRing synthetic -> {out}")
    with h5py.File(args.ring_train, 'r') as f:
        ring_train_pool = range(f['images'].shape[0])
    with h5py.File(args.ring_val, 'r') as f:
        ring_val_pool = range(f['images'].shape[0])
    subset_h5(args.ring_train, out / 'ring_train.h5',
              pick_frames(args.ring_train, ring_train_pool, args.n_train, args.seed + 2))
    subset_h5(args.ring_val, out / 'ring_val.h5',
              pick_frames(args.ring_val, ring_val_pool, args.n_val, args.seed + 3))

    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6
    print(f"\nDone. examples/ total size: {total:.1f} MB")


if __name__ == '__main__':
    main()
