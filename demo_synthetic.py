"""
scripts/demo_synthetic.py — Quick Demo with Synthetic Data
=========================================================
Run this BEFORE you have real data to verify the full pipeline works:
  - Generates synthetic mouth-crop sequences (random noise, then sine patterns)
  - Trains for a few epochs
  - Evaluates and prints results

Usage:
    python scripts/demo_synthetic.py

This script temporarily overrides PROCESSED_DIR so it does not touch real data.
"""

import os
import sys
import numpy as np
import torch

# ── Make imports work from project root ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg

# ── Point to a temp directory ─────────────────────────────────────────────────
DEMO_DATA_DIR = os.path.join(cfg.BASE_DIR, "data", "demo_synthetic")
cfg.PROCESSED_DIR   = DEMO_DATA_DIR
cfg.NUM_EPOCHS      = 10
cfg.BATCH_SIZE      = 4

from models.lipreader import LipReadingModel, count_parameters
from utils.dataset import LipReadingDataset, get_dataloaders, PIXEL_MEAN, PIXEL_STD


# ─── Generate synthetic dataset ──────────────────────────────────────────────

def generate_synthetic_data(
    root: str,
    vocab: list,
    samples_per_class: int = 30,
    n_frames: int = cfg.FRAMES_PER_CLIP,
    img_size: int = cfg.IMG_SIZE,
):
    """
    Create `samples_per_class` .npy clips per class.
    Each class gets a slightly different noise pattern so the model has
    _something_ to learn, even without real lip movements.
    """
    rng = np.random.default_rng(42)
    os.makedirs(root, exist_ok=True)

    for idx, word in enumerate(vocab):
        word_dir = os.path.join(root, word)
        os.makedirs(word_dir, exist_ok=True)

        for s in range(samples_per_class):
            # Class-specific signal: sine wave with class-dependent frequency
            freq    = (idx + 1) * 0.2
            t       = np.linspace(0, 2 * np.pi, n_frames)
            signal  = np.sin(freq * t) * 40 + 128    # (T,)

            # Broadcast signal across spatial dims + add noise
            clip    = signal[:, None, None] * np.ones((n_frames, img_size, img_size))
            noise   = rng.normal(0, 20, clip.shape)
            clip    = np.clip(clip + noise, 0, 255).astype(np.uint8)

            path = os.path.join(word_dir, f"sample_{s:04d}.npy")
            np.save(path, clip)

    print(f"Synthetic dataset: {len(vocab)} classes × {samples_per_class} samples")
    print(f"Saved to: {root}")


# ─── Minimal training loop (for demo only) ───────────────────────────────────

def demo_train():
    vocab = cfg.VOCABULARY
    generate_synthetic_data(DEMO_DATA_DIR, vocab)

    device = cfg.DEVICE
    print(f"\nDevice: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_root=DEMO_DATA_DIR,
        vocab=vocab,
        batch_size=cfg.BATCH_SIZE,
        num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}")

    model = LipReadingModel(num_classes=len(vocab)).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(cfg.NUM_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_n = 0., 0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(clips)
            loss   = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            tr_loss    += loss.item() * len(labels)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_n       += len(labels)

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_n = 0., 0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                loss   = criterion(logits, labels)
                va_loss    += loss.item() * len(labels)
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_n       += len(labels)

        print(
            f"Epoch {epoch+1:02d}/{cfg.NUM_EPOCHS} | "
            f"Train loss={tr_loss/tr_n:.4f} acc={100*tr_correct/tr_n:.1f}% | "
            f"Val loss={va_loss/va_n:.4f} acc={100*va_correct/va_n:.1f}%"
        )

    # ── Test ─────────────────────────────────────────────────────────────────
    model.eval()
    te_correct, te_n = 0, 0
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            preds       = model(clips).argmax(1)
            te_correct += (preds == labels).sum().item()
            te_n       += len(labels)
    print(f"\n{'═'*50}")
    print(f"  Test Accuracy: {100*te_correct/max(te_n,1):.2f}%  ({te_correct}/{te_n})")
    print(f"{'═'*50}")
    print("\nDemo complete. Pipeline is verified end-to-end.")
    print("Next step: provide real MIRACL-VC1 data and run train.py")


if __name__ == "__main__":
    demo_train()
