"""
Training engine shared by every model.

The per-model differences are driven entirely by the config class:
  - OPTIMIZER        Adam (U-Net, UNet++) vs AdamW (SegFormer)
  - GRAD_CLIP_NORM   gradients are clipped to this norm; None disables clipping
  - WARMUP_EPOCHS    linear LR warmup before the plateau scheduler takes over
  - USE_DEEP_SUPERVISION  UNet++ returns 4 heads: the training loss is their
                          weighted sum, validation uses the final head only
"""

import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .metrics import accumulate_counts, counts_to_metrics, new_counts


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp(rank: int, world_size: int, port: str = '12356'):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', port)
    # Generous timeout so no rank trips the 30-minute NCCL watchdog while it
    # waits at a barrier during a long validation pass.
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=3))
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# ── Optimizer / scheduler ─────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg):
    """
    Adam or AdamW at a single uniform learning rate.

    Every reported run is trained end-to-end at one LR: the three from-scratch
    models have no pretrained encoder to protect, and the SegFormer reference
    fine-tuning recipe also trains MiT end-to-end rather than with a frozen or
    lower-LR backbone.
    """
    optim_cls = {'adam': torch.optim.Adam,
                 'adamw': torch.optim.AdamW}[cfg.OPTIMIZER.lower()]

    return optim_cls(model.parameters(), lr=cfg.LEARNING_RATE,
                     weight_decay=cfg.WEIGHT_DECAY)


def build_scheduler(optimizer, cfg):
    """ReduceLROnPlateau on the combined validation mIoU (higher is better)."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.MIN_LR,
    )


def apply_warmup(optimizer, epoch: int, cfg) -> bool:
    """
    Linear LR warmup for the from-scratch transformer.

    Returns True when the plateau scheduler should step this epoch (i.e. warmup
    is over), False while warmup is still overriding the learning rate.
    """
    if cfg.WARMUP_EPOCHS and epoch <= cfg.WARMUP_EPOCHS:
        warmup_lr = cfg.LEARNING_RATE * epoch / cfg.WARMUP_EPOCHS
        for group in optimizer.param_groups:
            group['lr'] = warmup_lr
        return False
    return True


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_metric, metrics, patience_counter, cfg=None):
    ckpt = {
        'epoch':                epoch,
        'model_state_dict':     unwrap(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric':          best_metric,
        'metrics':              metrics,
        'patience_counter':     patience_counter,
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if cfg is not None:
        ckpt['model_name'] = cfg.MODEL
    torch.save(ckpt, path)


def load_model_weights(model: nn.Module, ckpt_path, device,
                       strict: bool = True, verbose: bool = True) -> dict:
    """Load weights into `model`, tolerating checkpoints saved from DDP."""
    ckpt  = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    unwrap(model).load_state_dict(state, strict=strict)
    if verbose:
        best = ckpt.get('best_metric', float('nan'))
        print(f"Loaded weights from {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')}, stored best_metric={best:.4f})")
    return ckpt


# ── Loss over (possibly deeply supervised) outputs ────────────────────────────

def training_loss(outputs, masks, criterion, cfg):
    """Weighted sum over deep-supervision heads, or the plain loss."""
    if isinstance(outputs, (list, tuple)):
        return sum(w * criterion(out, masks)
                   for w, out in zip(cfg.DEEP_SUPERVISION_WEIGHTS, outputs))
    return criterion(outputs, masks)


def final_head(outputs):
    """The full-resolution head used for validation metrics (X^{0,4} for UNet++)."""
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


# ── Train / validate ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, rank, cfg, limit_batches=None) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    iterator = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False) \
               if rank == 0 else loader

    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        if cfg.USE_AMP and scaler is not None:
            with torch.cuda.amp.autocast():
                loss = training_loss(model(images), masks, criterion, cfg)
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = training_loss(model(images), masks, criterion, cfg)
            loss.backward()
            if cfg.GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        if limit_batches is not None and n_batches >= limit_batches:
            break

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, split_name: str,
             cfg, rank: int = 0, world_size: int = 1,
             limit_batches=None) -> Tuple[float, dict]:
    """
    Evaluate one validation domain.

    Every rank evaluates its own shard; the pixel counts are all-reduced so all
    ranks end up with identical global metrics (and therefore an identical LR
    schedule and early-stopping decision).
    """
    model.eval()

    counts     = new_counts(cfg.NUM_CLASSES, device)
    loss_sum   = torch.zeros(1, device=device, dtype=torch.float64)
    sample_sum = torch.zeros(1, device=device, dtype=torch.float64)

    iterator = tqdm(loader, desc=f'Epoch {epoch} [{split_name}]', leave=False) \
               if rank == 0 else loader

    n_batches = 0
    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        n = images.shape[0]

        if cfg.USE_AMP:
            with torch.cuda.amp.autocast():
                logits = final_head(model(images))
                loss   = criterion(logits, masks)
        else:
            logits = final_head(model(images))
            loss   = criterion(logits, masks)

        # Weight by the real batch size so a short final batch is not over-counted
        loss_sum   += loss.item() * n
        sample_sum += n

        accumulate_counts(counts, torch.argmax(logits, dim=1), masks)

        n_batches += 1
        if limit_batches is not None and n_batches >= limit_batches:
            break

    if world_size > 1:
        dist.all_reduce(counts,     op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum / sample_sum.clamp(min=1)).item()
    return avg_loss, counts_to_metrics(counts)
