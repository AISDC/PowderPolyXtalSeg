#!/usr/bin/env python3
"""
Normalize the hand-labelled detector images into the `real/` directory that
training reads.

Raw detector frames are 16-bit with a huge dynamic range: a few saturated HEDM
spots sit on top of a background that carries the powder rings. A percentile
clip followed by a log transform brings both onto a comparable scale, which is
what the published Mixed-v2 models were trained on (`--method cliplog`).

Input layout (one directory per sample):
    <src>/<sample_name>/<anything>.tif        raw detector image
    <src>/<sample_name>/labkit_segmented.tif  uint8 mask, 0 = BG / 1 = HEDM / 2 = powder

Output layout (same structure, images rewritten as float32 in [0, 1]):
    <out>/<sample_name>/<anything>.tif
    <out>/<sample_name>/labkit_segmented.tif

Usage:
    python scripts/prepare_real_data.py --src raw_hand_labelled --out data/real
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile

NORM_PERCENTILE_LOW  = 0.1
NORM_PERCENTILE_HIGH = 99.9


def normalize_logminmax(img: np.ndarray) -> np.ndarray:
    """Shift to non-negative -> log1p -> min-max to [0, 1]."""
    img = img.astype(np.float32)
    img_log = np.log1p(img - img.min())
    return (img_log / (img_log.max() + 1e-7)).astype(np.float32)


def normalize_cliplog(img: np.ndarray,
                      p_low_pct: float = NORM_PERCENTILE_LOW,
                      p_high_pct: float = NORM_PERCENTILE_HIGH):
    """Percentile clip -> shift to non-negative -> log1p -> min-max to [0, 1]."""
    img    = img.astype(np.float32)
    p_low  = float(np.percentile(img, p_low_pct))
    p_high = float(np.percentile(img, p_high_pct))
    img_log = np.log1p(np.clip(img, p_low, p_high) - p_low)
    return (img_log / (img_log.max() + 1e-7)).astype(np.float32), p_low, p_high


def find_raw_tif(sample_dir: Path) -> Path:
    raw_tifs = [f for f in sorted(sample_dir.glob('*.tif'))
                if 'labkit_segmented' not in f.name]
    if len(raw_tifs) != 1:
        raise ValueError(f"Expected 1 raw TIF in {sample_dir}, found: {raw_tifs}")
    return raw_tifs[0]


def process_sample(sample_dir: Path, out_dir: Path, method: str,
                   p_low_pct: float, p_high_pct: float) -> None:
    raw_tif  = find_raw_tif(sample_dir)
    mask_src = sample_dir / 'labkit_segmented.tif'
    if not mask_src.exists():
        raise FileNotFoundError(f"Missing labkit_segmented.tif in {sample_dir}")

    img_raw = tifffile.imread(raw_tif)
    img     = img_raw.astype(np.float32)

    print(f"\n  {sample_dir.name}")
    print(f"    input  dtype={img_raw.dtype}  shape={img.shape}"
          f"  range=[{img.min():.4g}, {img.max():.4g}]")

    if method == 'cliplog':
        img_norm, p_low, p_high = normalize_cliplog(img, p_low_pct, p_high_pct)
        print(f"    cliplog    p_low={p_low:.4g}  p_high={p_high:.4g}"
              f"  output range=[{img_norm.min():.6f}, {img_norm.max():.6f}]")
    else:
        img_norm = normalize_logminmax(img)
        print(f"    logminmax  output range=[{img_norm.min():.6f}, {img_norm.max():.6f}]")

    dst = out_dir / sample_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst / raw_tif.name, img_norm,
                     compression='zlib', compressionargs={'level': 1})
    shutil.copy2(mask_src, dst / 'labkit_segmented.tif')


def main():
    parser = argparse.ArgumentParser(
        description='Normalize hand-labelled detector images for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src', required=True, type=Path,
                        help='Directory of raw sample sub-directories')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory (point --data-root/real at this)')
    parser.add_argument('--method', choices=('cliplog', 'logminmax'), default='cliplog',
                        help='Normalization used by the published models: cliplog')
    parser.add_argument('--p-low', type=float, default=NORM_PERCENTILE_LOW,
                        help='Lower clip percentile (cliplog only)')
    parser.add_argument('--p-high', type=float, default=NORM_PERCENTILE_HIGH,
                        help='Upper clip percentile (cliplog only)')
    args = parser.parse_args()

    sample_dirs = sorted([d for d in args.src.iterdir() if d.is_dir()])
    if not sample_dirs:
        raise SystemExit(f"No sample sub-directories found in {args.src}")

    print(f"Source: {args.src}")
    print(f"Output: {args.out}   (method={args.method})")
    print(f"Found {len(sample_dirs)} sample(s): {[d.name for d in sample_dirs]}")

    for d in sample_dirs:
        process_sample(d, args.out, args.method, args.p_low, args.p_high)

    print(f"\nDone. Wrote {len(sample_dirs)} samples to {args.out}")


if __name__ == '__main__':
    main()
