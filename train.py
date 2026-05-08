"""
train.py — Training Loop for the AI Lip-Reading System
======================================================
Usage:
    python train.py [--epochs 50] [--batch_size 8] [--lr 3e-4]

Features
--------
- Adam / AdamW optimiser with ReduceLROnPlateau scheduler
- Early stopping
- Best-model checkpointing
- CSV loss log + per-epoch console summary
"""

import os
import csv
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR, VOCABULARY,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PAT,
    PROCESSED_DIR, LOG_INTERVAL, DEVICE
)
from models.lipreader import LipReadingModel, count_parameters
from utils.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Argument parser ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the AI Lip-Reading System")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",type=float, default=WEIGHT_DECAY)
    p.add_argument("--resnet_depth",type=int,   default=18, choices=[18, 34])
    p.add_argument("--pool_mode",   type=str,   default="mean",
                                    choices=["mean", "last"])
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint .pth to resume training")
    return p.parse_args()


# ─── Training / validation step ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser, phase: str, device: str):
    """Run one epoch; return (avg_loss, accuracy)."""
    training = (phase == "train")
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch_idx, (clips, labels) in enumerate(loader):
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)                # (B, num_classes)
            loss   = criterion(logits, labels)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()

            preds   = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

            total_loss    += loss.item() * clips.size(0)
            total_correct += correct
            total_samples += clips.size(0)

            if training and (batch_idx + 1) % LOG_INTERVAL == 0:
                logger.info(
                    "  [%s] batch %d/%d  loss=%.4f  acc=%.2f%%",
                    phase, batch_idx + 1, len(loader),
                    loss.item(), 100.0 * correct / clips.size(0)
                )

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


# ─── Main training loop ──────────────────────────────────────────────────────

def train(args):
    logger.info("Device: %s", DEVICE)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Data  — train: %d  val: %d  test: %d  batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = LipReadingModel(
        resnet_depth=args.resnet_depth,
        pool_mode=args.pool_mode,
    ).to(DEVICE)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    no_improve     = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        logger.info("Resumed from %s  (epoch %d)", args.resume, start_epoch)

    # ── CSV log ──────────────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUTS_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_acc","val_loss","val_acc","lr"]
        )

    # ── Loop ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        current_lr = optimiser.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimiser, "train", DEVICE
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, "val", DEVICE
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d/%03d | %.0fs | LR %.2e | "
            "Train loss %.4f acc %.2f%% | Val loss %.4f acc %.2f%%",
            epoch + 1, args.epochs, elapsed, current_lr,
            train_loss, train_acc, val_loss, val_acc,
        )

        # ── Log to CSV ───────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}", f"{val_acc:.2f}", f"{current_lr:.2e}"
            ])

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optim_state":    optimiser.state_dict(),
            "best_val_loss":  best_val_loss,
            "val_acc":        val_acc,
            "vocab":          VOCABULARY,
        }
        last_path = os.path.join(CHECKPOINTS_DIR, "last.pth")
        torch.save(ckpt, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            best_path = os.path.join(CHECKPOINTS_DIR, "best.pth")
            torch.save(ckpt, best_path)
            logger.info("  ✓  New best val_loss=%.4f  saved to %s",
                        best_val_loss, best_path)
        else:
            no_improve += 1
            logger.info("  No improvement for %d epoch(s).", no_improve)

        # ── Early stopping ───────────────────────────────────────────────────
        if no_improve >= EARLY_STOP_PAT:
            logger.info("Early stopping triggered after %d epochs.", epoch + 1)
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation …")
    best_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "best.pth"),
                           map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, None, "test", DEVICE
    )
    logger.info("══ Test  loss=%.4f  acc=%.2f%% ══", test_loss, test_acc)
    logger.info("Training log saved to: %s", log_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
