#!/usr/bin/env python3
"""
Inference speed benchmark for the three architectures.

`segformer_pretrained` is deliberately excluded from the default set: it is the
same architecture as `segformer` and differs only in initialization, which does
not affect timing. Pass `--model segformer_pretrained` to measure it anyway.

Measures GPU latency and throughput on 512x512 patches (batch 1/4/8/16, FP32 and
AMP FP16) and derives full-detector-frame rates for GE (2048^2 = 16 patches) and
Varex (2880^2 ~ 36 patches) at stride-512 tiling. The timed region includes the
argmax, because that is part of the deployed pipeline.

Weights do not affect timing, so checkpoints are optional: without them the
networks run with random initialization.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python benchmark_speed.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_speed.py \
        --checkpoint unet=runs/unet_mixedv2/checkpoints/best_model.pth \
        --checkpoint unetpp=runs/unetpp_mixedv2/checkpoints/best_model.pth
"""

import argparse
import json
import time

import torch

from configs import MODEL_NAMES, get_config
from powderpolyxtalseg.engine import load_model_weights
from powderpolyxtalseg.models import build_model, count_parameters

# One entry per distinct architecture. segformer_pretrained shares SegFormer's
# graph, so benchmarking it would just repeat the same measurement.
DEFAULT_MODELS = ('unet', 'unetpp', 'segformer')

PATCH         = 512
BATCH_SIZES   = [1, 4, 8, 16]
WARMUP_ITERS  = 10
TIMED_ITERS   = 50
# Full-frame patch counts at stride 512 (non-overlapping tiling)
FRAME_PATCHES = {'GE 2048x2048': 16, 'Varex 2880x2880': 36}


@torch.no_grad()
def bench(model, batch_size: int, device, use_amp: bool) -> float:
    """Mean seconds per forward pass, post-warmup and CUDA-synchronized."""
    x = torch.randn(batch_size, 1, PATCH, PATCH, device=device)

    def forward():
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)
        out = out[-1] if isinstance(out, (list, tuple)) else out
        out.argmax(dim=1)

    for _ in range(WARMUP_ITERS):
        forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        forward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / TIMED_ITERS


def parse_checkpoints(entries):
    """--checkpoint model=path ... -> {model: path}"""
    mapping = {}
    for entry in entries or []:
        if '=' not in entry:
            raise SystemExit(f"--checkpoint expects 'model=path', got '{entry}'")
        name, path = entry.split('=', 1)
        if name not in MODEL_NAMES:
            raise SystemExit(f"Unknown model '{name}' in --checkpoint")
        mapping[name] = path
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference speed')
    parser.add_argument('--model', action='append', choices=MODEL_NAMES,
                        help='Models to benchmark (default: the three distinct '
                             'architectures)')
    parser.add_argument('--checkpoint', action='append', metavar='MODEL=PATH',
                        help='Optional weights to load, e.g. unet=best_model.pth')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=BATCH_SIZES)
    parser.add_argument('--json', type=str, default='inference_speed_results.json',
                        help='Where to write the raw numbers')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark.')

    models      = args.model or list(DEFAULT_MODELS)
    checkpoints = parse_checkpoints(args.checkpoint)
    device      = torch.device('cuda:0')
    gpu         = torch.cuda.get_device_name(device)

    print(f'GPU: {gpu}\nPatch: {PATCH}x{PATCH}  warmup={WARMUP_ITERS}  '
          f'timed={TIMED_ITERS} iters\n')

    results = {}
    for name in models:
        cfg   = get_config(name)
        model = build_model(cfg).to(device).eval()
        if name in checkpoints:
            load_model_weights(model, checkpoints[name], device)
        else:
            print(f'{name}: random initialization (timing is weight-independent)')

        n_params = count_parameters(model)
        print(f'{name}: {n_params/1e6:.1f}M params')
        results[name] = {'params_M': n_params / 1e6}

        for use_amp in (False, True):
            prec = 'fp16' if use_amp else 'fp32'
            for bs in args.batch_sizes:
                sec = bench(model, bs, device, use_amp)
                results[name][f'{prec}_bs{bs}'] = {
                    'latency_ms_per_batch': sec * 1e3,
                    'ms_per_patch':         sec / bs * 1e3,
                    'patches_per_s':        bs / sec,
                }
                print(f'  {prec} bs={bs:2d}: {sec*1e3:7.2f} ms/batch  '
                      f'{sec/bs*1e3:6.2f} ms/patch  {bs/sec:7.1f} patches/s')

        del model
        torch.cuda.empty_cache()
        print()

    # Full-frame projections from the best throughput (fp16, largest batch)
    largest = args.batch_sizes[-1]
    print(f'Full-detector-frame rates (stride-512 tiling, fp16, bs={largest}):')
    print(f'  {"Model":<10}' + ''.join(f'{k:>22}' for k in FRAME_PATCHES))
    for name in models:
        pps = results[name][f'fp16_bs{largest}']['patches_per_s']
        row = f'  {name:<10}'
        for n_patches in FRAME_PATCHES.values():
            row += f'{pps / n_patches:>17.1f} fps'
        print(row)

    with open(args.json, 'w') as f:
        json.dump({'gpu': gpu, 'patch': PATCH, 'results': results}, f, indent=2)
    print(f'\nSaved: {args.json}')


if __name__ == '__main__':
    main()
