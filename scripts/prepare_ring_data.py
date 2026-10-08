#!/usr/bin/env python3
"""
Convert simulated powder-ring TIF pairs into the ring_train.h5 / ring_val.h5
files used as the powder-rich 60% of the mixed training set.

Input (one pair per simulated frame):
    <src>/sample_XXX_input.tif   uint16 detector image
    <src>/sample_XXX_mask.tif    uint8 mask, values {0 = BG, 1 = HEDM, 2 = powder}

Output:
    <out>/ring_train.h5
    <out>/ring_val.h5
        images  float32  (N, 512, 512)  normalized to [0, 1]
        masks   uint8    (N, 512, 512)  values {0, 1, 2}

Processing:
    1. Shuffle the pairs with a fixed seed and split train/val
    2. Clip-log normalize: clip to [0.01, 99.9] percentiles, log(x + 1), min-max
    3. Tile into 512x512 patches with stride 384 (boundary-clamped)
    4. Drop patches with less than 0.5% foreground pixels
    5. Write to HDF5 with gzip compression

Usage:
    python scripts/prepare_ring_data.py --src gen_dataset_for_ML --out data
"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import tifffile
from tqdm import tqdm

PATCH_SIZE      = 512
STRIDE          = 384
MIN_FG_FRACTION = 0.005   # 0.5% non-background pixels
SEED            = 42
CHUNK           = 256     # HDF5 chunk size along N


def normalize_cliplog(img: np.ndarray) -> np.ndarray:
    img   = img.astype(np.float32)
    p_lo  = float(np.percentile(img, 0.01))
    p_hi  = float(np.percentile(img, 99.9))
    img   = np.log(np.clip(img, p_lo, p_hi) + 1.0)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return img.astype(np.float32)


def make_starts(dim: int, patch: int, stride: int):
    starts = list(range(0, dim - patch, stride))
    starts.append(dim - patch)
    return list(dict.fromkeys(starts))


def extract_patches(img: np.ndarray, mask: np.ndarray, patch: int, stride: int,
                    min_fg: float):
    """Yield (img_patch, mask_patch) for every patch that clears the fg filter."""
    H, W = img.shape
    for r in make_starts(H, patch, stride):
        for c in make_starts(W, patch, stride):
            mp = mask[r:r + patch, c:c + patch]
            if (mp > 0).sum() / mp.size >= min_fg:
                yield img[r:r + patch, c:c + patch].copy(), mp.copy()


def write_h5(pairs, h5_path: Path, split_name: str, args):
    all_imgs, all_masks = [], []
    for input_path, mask_path in tqdm(pairs, desc=f'  {split_name} - extracting'):
        img = tifffile.imread(str(input_path))
        msk = tifffile.imread(str(mask_path))
        if img.ndim != 2 or msk.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {img.shape} / {msk.shape}")
        if img.shape != msk.shape:
            raise ValueError(f"Image/mask shape mismatch for {input_path}")

        img_norm = normalize_cliplog(img)
        for ip, mp in extract_patches(img_norm, msk, args.patch_size,
                                      args.stride, args.min_fg):
            all_imgs.append(ip)
            all_masks.append(mp.astype(np.uint8))

    n = len(all_imgs)
    print(f"  {split_name}: {len(pairs)} images -> {n} patches")
    if n == 0:
        print(f"  WARNING: no patches kept for {split_name}, skipping.")
        return

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    p = args.patch_size
    with h5py.File(h5_path, 'w') as f:
        ds_img = f.create_dataset('images', shape=(n, p, p), dtype=np.float32,
                                  chunks=(min(CHUNK, n), p, p),
                                  compression='gzip', compression_opts=4)
        ds_msk = f.create_dataset('masks', shape=(n, p, p), dtype=np.uint8,
                                  chunks=(min(CHUNK, n), p, p),
                                  compression='gzip', compression_opts=4)
        ds_img[:] = np.stack(all_imgs, axis=0)
        ds_msk[:] = np.stack(all_masks, axis=0)
        f.attrs['patch_size']      = args.patch_size
        f.attrs['stride']          = args.stride
        f.attrs['min_fg_fraction'] = args.min_fg
        f.attrs['normalization']   = 'cliplog_0.01_99.9'
        f.attrs['n_source_images'] = len(pairs)

    with h5py.File(h5_path, 'r') as f:
        sample_img  = f['images'][0][:]
        sample_mask = f['masks'][0][:]
    print(f"  Saved {n} patches to {h5_path}")
    print(f"  Sanity: img range=[{sample_img.min():.4f}, {sample_img.max():.4f}], "
          f"mask classes={np.unique(sample_mask).tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description='Build ring_train.h5 / ring_val.h5 from simulated TIF pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src', required=True, type=Path,
                        help='Directory holding sample_*_input.tif / sample_*_mask.tif')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory (write into your --data-root)')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Fraction of source images held out for validation')
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--min-fg', type=float, default=MIN_FG_FRACTION,
                        help='Drop patches below this foreground pixel fraction')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    input_paths = sorted(args.src.glob('sample_*_input.tif'))
    mask_paths  = sorted(args.src.glob('sample_*_mask.tif'))
    if not input_paths:
        raise SystemExit(f"No sample_*_input.tif files found in {args.src}")
    if len(input_paths) != len(mask_paths):
        raise SystemExit(f"Mismatch: {len(input_paths)} inputs vs {len(mask_paths)} masks")

    pairs = list(zip(input_paths, mask_paths))
    print(f"Found {len(pairs)} image pairs in {args.src}")

    random.Random(args.seed).shuffle(pairs)
    n_train = int(round(len(pairs) * (1.0 - args.val_fraction)))
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]
    print(f"Split: {len(train_pairs)} train / {len(val_pairs)} val")

    print(f"\nBuilding ring_train.h5 ...")
    write_h5(train_pairs, args.out / 'ring_train.h5', 'train', args)

    print(f"\nBuilding ring_val.h5 ...")
    write_h5(val_pairs, args.out / 'ring_val.h5', 'val', args)

    print("\nDone.")


if __name__ == '__main__':
    main()
