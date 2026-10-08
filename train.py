#!/usr/bin/env python3
"""
Train U-Net, UNet++ or SegFormer on the 3-way mixed dataset
(20% real hand-labelled + 20% old synthetic + 60% ring synthetic).

Four runs are available. `unet`, `unetpp` and `segformer` are randomly
initialized under one shared protocol and form the controlled architecture
comparison; `segformer_pretrained` is the same SegFormer with an ImageNet-
pretrained MiT-B2 encoder, included as a control on the effect of pretraining.

Best-model selection and the LR schedule both use the combined validation metric
    combined = 0.2 * real_mIoU + 0.2 * old_synth_mIoU + 0.6 * ring_mIoU

Usage
-----
    # Multi-GPU DDP (uses every visible GPU)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --model unetpp

    # Single GPU
    python train.py --model segformer --no-ddp

    # SegFormer with an ImageNet-pretrained MiT-B2 encoder
    python train.py --model segformer_pretrained

    # Resume an interrupted run
    python train.py --model unetpp --resume runs/unetpp_mixedv2/checkpoints/checkpoint_epoch_030.pth

    # Smoke test on the bundled example subset
    python train.py --model unet --data-root examples --no-ddp --epochs 1 --limit-batches 2
"""

import argparse
import json

import torch
import torch.distributed as dist

from configs import MODEL_NAMES, get_config
from powderpolyxtalseg.data import build_mixed_dataloaders
from powderpolyxtalseg.engine import (
    apply_warmup,
    build_optimizer,
    build_scheduler,
    cleanup_ddp,
    load_model_weights,
    save_checkpoint,
    set_seed,
    setup_ddp,
    train_one_epoch,
    validate,
)
from powderpolyxtalseg.losses import CombinedLoss
from powderpolyxtalseg.metrics import combined_metric
from powderpolyxtalseg.models import build_model, count_parameters
from torch.nn.parallel import DistributedDataParallel as DDP


def build_config(args):
    """
    Resolve the config class for --model and apply the CLI overrides.

    Called once per process: `torch.multiprocessing.spawn` pickles a class by
    reference, so a config resolved in the parent would arrive in the workers
    with its class attributes reset.
    """
    cfg = get_config(args.model)
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.pretrained is not None:
        cfg.PRETRAINED = args.pretrained
    cfg.resolve(data_root=args.data_root, output_dir=args.output_dir)
    return cfg


def epoch_record(epoch, train_loss, lr, real, old_synth, ring, combined):
    """One row of logs/history.json."""
    (real_loss, real_m)           = real
    (old_synth_loss, old_synth_m) = old_synth
    (ring_loss, ring_m)           = ring
    record = {'epoch': epoch, 'train_loss': train_loss, 'lr': lr}
    for prefix, loss, m in (('real', real_loss, real_m),
                            ('old_synth', old_synth_loss, old_synth_m),
                            ('ring', ring_loss, ring_m)):
        record[f'{prefix}_val_loss']    = loss
        record[f'{prefix}_mean_iou']    = m['mean_iou']
        record[f'{prefix}_mean_dice']   = m['mean_dice']
        for cls in range(3):
            record[f'{prefix}_iou_class_{cls}'] = m[f'iou_class_{cls}']
    record['combined_metric'] = combined
    return record


def train(rank: int, world_size: int, args):
    cfg = build_config(args)
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size, port=cfg.DDP_PORT)

    set_seed(cfg.SEED + rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cfg.describe()

    # ── Data: every rank builds the loaders; val is sharded ───────────────────
    (train_loader,
     real_val_loader,
     old_synth_val_loader,
     ring_val_loader) = build_mixed_dataloaders(cfg, world_size=world_size, rank=rank)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if rank == 0:
        print(f"\n{cfg.MODEL}: {count_parameters(model):,} trainable parameters")

    criterion = CombinedLoss(
        alpha=cfg.FOCAL_ALPHA,
        gamma=cfg.FOCAL_GAMMA,
        focal_weight=cfg.FOCAL_WEIGHT,
        dice_weight=cfg.DICE_WEIGHT,
    )

    start_epoch      = 1
    best_metric      = 0.0
    patience_counter = 0
    best_record      = {}
    history          = []

    if args.resume:
        ckpt = load_model_weights(model, args.resume, device, verbose=(rank == 0))
        optimizer = build_optimizer(model, cfg)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler = build_scheduler(optimizer, cfg)
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler = torch.cuda.amp.GradScaler() if cfg.USE_AMP else None
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch      = ckpt['epoch'] + 1
        best_metric      = ckpt.get('best_metric', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        best_record      = ckpt.get('metrics', {})
        history_path = cfg.LOG_DIR / 'history.json'
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        if rank == 0:
            print(f"Resuming at epoch {start_epoch}, best combined={best_metric:.4f}")
    else:
        if cfg.PRETRAINED:
            load_model_weights(model, cfg.PRETRAINED, device, verbose=(rank == 0))
            if rank == 0:
                print(f"Warm start from {cfg.PRETRAINED}")
        elif rank == 0:
            print("\nImageNet-pretrained encoder." if cfg.PRETRAINED_BACKBONE
                  else "\nTraining from scratch (random initialization).")
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        scaler = torch.cuda.amp.GradScaler() if cfg.USE_AMP else None

    # ── Wrap in DDP only after the weights are in place ───────────────────────
    if use_ddp:
        dist.barrier()
        model = DDP(model, device_ids=[rank], output_device=rank)

    if rank == 0:
        lr_desc = f"lr={cfg.LEARNING_RATE}"
        print(f"\n{cfg.MODEL} mixed training: epochs {start_epoch}..{cfg.NUM_EPOCHS}")
        print(f"Optimizer      : {cfg.OPTIMIZER} {lr_desc} wd={cfg.WEIGHT_DECAY} "
              f"(grad clip {cfg.GRAD_CLIP_NORM}, warmup {cfg.WARMUP_EPOCHS} epochs)")
        print(f"Mixing ratios  : real={cfg.REAL_RATIO:.0%}  "
              f"old-synth={cfg.OLD_SYNTH_RATIO:.0%}  ring={cfg.RING_RATIO:.0%}")
        print(f"Metric weights : real={cfg.REAL_METRIC_WEIGHT:.0%}  "
              f"old-synth={cfg.OLD_SYNTH_METRIC_WEIGHT:.0%}  "
              f"ring={cfg.RING_METRIC_WEIGHT:.0%}")
        print(f"Early stopping : patience={cfg.EARLY_STOPPING_PATIENCE}")
        print(f"Output         : {cfg.OUTPUT_DIR}\n")

    stop_flag = cfg.LOG_DIR / '.stop'
    if rank == 0 and stop_flag.exists():
        stop_flag.unlink()

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        scheduler_active = apply_warmup(optimizer, epoch, cfg)
        if rank == 0 and not scheduler_active:
            print(f"Epoch {epoch}: warmup lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, rank, cfg, limit_batches=args.limit_batches,
        )

        real      = validate(model, real_val_loader,      criterion, device, epoch,
                             'real_val',      cfg, rank, world_size, args.limit_batches)
        old_synth = validate(model, old_synth_val_loader, criterion, device, epoch,
                             'old_synth_val', cfg, rank, world_size, args.limit_batches)
        ring      = validate(model, ring_val_loader,      criterion, device, epoch,
                             'ring_val',      cfg, rank, world_size, args.limit_batches)

        combined = combined_metric(cfg, real[1]['mean_iou'],
                                   old_synth[1]['mean_iou'], ring[1]['mean_iou'])

        # Identical on every rank, so the LR stays in sync without communication
        if scheduler_active:
            scheduler.step(combined)

        if rank == 0:
            def fmt(name, pair):
                m = pair[1]
                return (f"{name} mIoU={m['mean_iou']:.4f} "
                        f"(BG={m['iou_class_0']:.3f} HEDM={m['iou_class_1']:.3f} "
                        f"Pow={m['iou_class_2']:.3f})")

            print(f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | train={train_loss:.4f} | "
                  f"{fmt('real', real)} | {fmt('old-synth', old_synth)} | "
                  f"{fmt('ring', ring)} | combined={combined:.4f}")

            record = epoch_record(epoch, train_loss,
                                  optimizer.param_groups[0]['lr'],
                                  real, old_synth, ring, combined)
            history.append(record)

            if combined > best_metric:
                best_metric      = combined
                best_record      = record
                patience_counter = 0
                save_checkpoint(cfg.CHECKPOINT_DIR / 'best_model.pth',
                                epoch, model, optimizer, scheduler, scaler,
                                best_metric, best_record, patience_counter, cfg)
                print(f"  -> New best  combined={best_metric:.4f}")
            else:
                patience_counter += 1

            if epoch % cfg.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(cfg.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch:03d}.pth',
                                epoch, model, optimizer, scheduler, scaler,
                                best_metric, record, patience_counter, cfg)

            with open(cfg.LOG_DIR / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_counter} epochs)")
                if use_ddp:
                    stop_flag.touch()   # tell the other ranks to leave the loop
                else:
                    break

        if use_ddp:
            dist.barrier()
            if stop_flag.exists():
                break

    if rank == 0:
        if stop_flag.exists():
            stop_flag.unlink()
        print(f"\nTraining complete. Best combined: {best_metric:.4f}")
        for domain in ('real', 'old_synth', 'ring'):
            if f'{domain}_mean_iou' in best_record:
                print(f"  {domain:<10} mIoU={best_record[f'{domain}_mean_iou']:.4f}  "
                      f"BG={best_record[f'{domain}_iou_class_0']:.4f}  "
                      f"HEDM={best_record[f'{domain}_iou_class_1']:.4f}  "
                      f"Powder={best_record[f'{domain}_iou_class_2']:.4f}")
        print(f"Best model: {cfg.CHECKPOINT_DIR / 'best_model.pth'}")

    if use_ddp:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(
        description='Train a segmentation model on the 3-way mixed dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True, choices=MODEL_NAMES,
                        help='Which architecture to train')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset directory (default: $POWDERSEG_DATA_ROOT or ./data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Run directory (default: runs/<model>_mixedv2)')
    parser.add_argument('--no-ddp', action='store_true',
                        help='Disable DDP and run on a single GPU')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume a run from (restores optimizer state)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Checkpoint to initialize weights from. Not used by '
                             'any of the reported runs, which all start from '
                             'random weights (or, for segformer_pretrained, from '
                             'the ImageNet MiT-B2 encoder).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the per-GPU batch size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override the number of dataloader workers per rank')
    parser.add_argument('--limit-batches', type=int, default=None,
                        help='Smoke test: cap train/val loops at N batches')
    args = parser.parse_args()

    # Fail fast on a bad --model / missing dataset before spawning any worker
    build_config(args)

    if args.no_ddp:
        print("Single-GPU mode.")
        train(0, 1, args)
        return

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Only {world_size} GPU(s) found - falling back to single-GPU.")
        train(0, 1, args)
        return

    print(f"Launching DDP training on {world_size} GPUs ...")
    torch.multiprocessing.spawn(train, args=(world_size, args),
                                nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
