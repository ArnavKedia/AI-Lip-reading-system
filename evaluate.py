"""
evaluate.py — Evaluation & Analysis for the AI Lip-Reading System
=================================================================
Usage:
    python evaluate.py --checkpoint checkpoints/best.pth

Produces
--------
- Accuracy, WER, CER on the test set
- Confusion matrix  (PNG + CSV)
- Per-class report  (precision / recall / F1)
- Misclassified sample list  (CSV)
"""

import os
import argparse
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    top_k_accuracy_score,
)

from config import CHECKPOINTS_DIR, OUTPUTS_DIR, VOCABULARY, DEVICE
from models.lipreader import LipReadingModel
from utils.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Text-level metrics ───────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def word_error_rate(refs: list, hyps: list) -> float:
    """WER = sum(edit_dist at word level) / sum(reference lengths)."""
    total_errors = 0
    total_words  = 0
    for ref, hyp in zip(refs, hyps):
        r_words = ref.split()
        h_words = hyp.split()
        total_errors += _edit_distance(r_words, h_words)
        total_words  += len(r_words)
    return total_errors / max(total_words, 1)


def char_error_rate(refs: list, hyps: list) -> float:
    """CER = sum(edit_dist at char level) / sum(reference char lengths)."""
    total_errors = 0
    total_chars  = 0
    for ref, hyp in zip(refs, hyps):
        total_errors += _edit_distance(list(ref), list(hyp))
        total_chars  += len(ref)
    return total_errors / max(total_chars, 1)


# ─── Confusion matrix plot ────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                          labels: list,
                          save_path: str,
                          title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


# ─── Learning curve plot ──────────────────────────────────────────────────────

def plot_learning_curves(log_csv: str, save_path: str):
    import csv
    epochs, tr_loss, va_loss, tr_acc, va_acc = [], [], [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            tr_acc.append(float(row["train_acc"]))
            va_acc.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, tr_loss, label="Train", linewidth=2)
    ax1.plot(epochs, va_loss, label="Val",   linewidth=2, linestyle="--")
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, tr_acc, label="Train", linewidth=2)
    ax2.plot(epochs, va_acc, label="Val",   linewidth=2, linestyle="--")
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves — AI Lip-Reading System",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Learning curves saved → %s", save_path)


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vocab = ckpt.get("vocab", VOCABULARY)
    logger.info("Loaded checkpoint: %s  (epoch %d, val_acc=%.2f%%)",
                ckpt_path, ckpt.get("epoch", -1), ckpt.get("val_acc", -1))

    model = LipReadingModel(num_classes=len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── DataLoader ───────────────────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(vocab=vocab, batch_size=args.batch_size)

    # ── Collect predictions ───────────────────────────────────────────────────
    all_labels  = []
    all_preds   = []
    all_probs   = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_total    = 0

    with torch.no_grad():
        for clips, labels in test_loader:
            clips  = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(clips)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=-1)

            total_loss += loss.item() * clips.size(0)
            n_total    += clips.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_probs.append(probs.cpu().numpy())

    avg_loss  = total_loss / max(n_total, 1)
    all_probs = np.vstack(all_probs)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    top1_acc = np.mean(np.array(all_labels) == np.array(all_preds)) * 100
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=min(5, len(vocab))) * 100

    # ── WER / CER ─────────────────────────────────────────────────────────────
    ref_words = [vocab[i].lower() for i in all_labels]
    hyp_words = [vocab[i].lower() for i in all_preds]
    wer = word_error_rate(ref_words, hyp_words)
    cer = char_error_rate(ref_words, hyp_words)

    logger.info("══════════════════════════════════════════")
    logger.info("  Test Loss      : %.4f", avg_loss)
    logger.info("  Top-1 Accuracy : %.2f%%", top1_acc)
    logger.info("  Top-5 Accuracy : %.2f%%", top5_acc)
    logger.info("  WER            : %.4f  (%.2f%%)", wer, wer*100)
    logger.info("  CER            : %.4f  (%.2f%%)", cer, cer*100)
    logger.info("══════════════════════════════════════════")

    # ── Per-class report ──────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=vocab, digits=4
    )
    logger.info("\nPer-class report:\n%s", report)
    report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_png = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, vocab, cm_png)
    np.savetxt(os.path.join(OUTPUTS_DIR, "confusion_matrix.csv"),
               cm, delimiter=",", fmt="%d")

    # ── Learning curves ───────────────────────────────────────────────────────
    log_csv = os.path.join(OUTPUTS_DIR, "training_log.csv")
    if os.path.isfile(log_csv):
        plot_learning_curves(log_csv,
                             os.path.join(OUTPUTS_DIR, "learning_curves.png"))

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "checkpoint":  ckpt_path,
        "test_loss":   round(avg_loss, 4),
        "top1_acc":    round(top1_acc, 2),
        "top5_acc":    round(top5_acc, 2),
        "WER":         round(wer, 4),
        "CER":         round(cer, 4),
        "n_test_samples": n_total,
    }
    with open(os.path.join(OUTPUTS_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All outputs saved to: %s", OUTPUTS_DIR)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the AI Lip-Reading System")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(CHECKPOINTS_DIR, "best.pth"))
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    evaluate(args)
