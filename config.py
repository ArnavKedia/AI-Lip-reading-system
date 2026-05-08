"""
config.py — Central configuration for the AI Lip-Reading System
AIM 3230 — Deep Learning Lab, Manipal University Jaipur
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
CHECKPOINTS_DIR= os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET        = "MIRACL-VC1"          # "MIRACL-VC1" | "LRW"
VOCABULARY     = ["Hello", "Start","Stop"]
NUM_CLASSES    = len(VOCABULARY)

# ─── Video / Preprocessing ────────────────────────────────────────────────────
IMG_SIZE       = 96          # Mouth crop: 96×96 px  (Stafylakis 2017)
FRAMES_PER_CLIP= 29          # Number of frames per sample (LRW standard)
GRAYSCALE      = True        # Use grayscale (reduces depth by 3×)
IN_CHANNELS    = 1 if GRAYSCALE else 3
FACE_DETECTOR  = "hog"       # "hog" | "cnn"  (cnn is more accurate but slower)

PIXEL_MEAN = 96.8715
PIXEL_STD  = 27.3758

# ─── Model Architecture ───────────────────────────────────────────────────────
RESNET_DEPTH   = 18          # 18 | 34
LSTM_HIDDEN    = 256         # Units per direction → total 512 (bidirectional)
LSTM_LAYERS    = 2
LSTM_DROPOUT   = 0.5
EMBED_DIM      = 512         # ResNet output feature dimension

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 8
NUM_EPOCHS     = 50
LEARNING_RATE  = 3e-4        # Adam / AdamW initial LR
WEIGHT_DECAY   = 1e-4
LR_PATIENCE    = 3           # ReduceLROnPlateau patience (epochs)
LR_FACTOR      = 0.5
EARLY_STOP_PAT = 10          # Early stopping patience
TRAIN_SPLIT    = 0.8
VAL_SPLIT      = 0.1
# TEST_SPLIT   = 0.1         # remainder

# ─── Augmentation ─────────────────────────────────────────────────────────────
HFLIP_PROB     = 0.5         # Random horizontal flip probability
ROT_DEGREES    = 5           # ±5° rotation
CROP_SHIFT     = 5           # ±5 px random crop shift

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL   = 10          # Print loss every N batches
SAVE_BEST_ONLY = True        # Only save checkpoint when val_loss improves
