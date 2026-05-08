# AI Lip-Reading System
### A Deep Learning Approach to Visual Speech Recognition
**Course:** DEEP LEARNING LAB (AIM 3230) · Semester VI (Jan–May 2026)  
**Programme:** B.Tech (AI & ML) · Manipal University Jaipur  
**Course Coordinator:** Ms. Akanksha Mrinali

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Data Preparation (MIRACL-VC1)](#5-data-preparation--miracl-vc1)
6. [Step-by-Step Running Guide](#6-step-by-step-running-guide)
7. [Running on Google Colab / Kaggle](#7-running-on-google-colab--kaggle)
8. [Configuration Reference](#8-configuration-reference)
9. [Expected Results](#9-expected-results)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

---

## 1. Project Overview

This project implements an end-to-end deep neural network that maps **silent video sequences of a speaking face** to text. It addresses Course Outcomes (COs) of AIM 3230:

| CO | What We Implement |
|----|-------------------|
| AIM3230.2 | Dropout, Batch Normalization, Adam optimizer |
| AIM3230.3 | 3D CNN + ResNet-18 visual front-end |
| AIM3230.4 | Bidirectional LSTM temporal back-end |
| AIM3230.5 | WER, CER, confusion matrix, per-class F1 |

**Pipeline summary:**

```
Raw Video → Face Detection → Mouth ROI Crop (96×96) →
3D Conv (5×7×7) → ResNet-18 → Bi-LSTM (2 layers) → Softmax → Predicted Word
```

---

## 2. Architecture

```
Input: (B, 1, 29, 96, 96)   ← Batch × Channels × Frames × Height × Width
         │
    ┌────▼────────────────────────────────────┐
    │  Stage 1 · 3D Convolutional Front-End   │
    │  Conv3d(5×7×7) → BN → ReLU → MaxPool3d │
    │  Output: (B, 64, 29, 24, 24)            │
    └────────────────────────┬────────────────┘
                             │ reshape: (B×T, 64, 24, 24)
    ┌────────────────────────▼────────────────┐
    │  Stage 2 · ResNet-18 Backbone           │
    │  4 residual blocks + Global Avg Pool    │
    │  Output: (B, T, 512)                    │
    └────────────────────────┬────────────────┘
    ┌────────────────────────▼────────────────┐
    │  Stage 3 · Bidirectional LSTM (2 layers)│
    │  Hidden: 256 per direction → 512 total  │
    │  Dropout: 0.5 between layers            │
    │  Output: (B, T, 512)                    │
    └────────────────────────┬────────────────┘
                             │ Mean pooling over T
    ┌────────────────────────▼────────────────┐
    │  Stage 4 · Classification Head          │
    │  Linear(512→256) → ReLU → Linear(256→K)│
    │  Output: (B, K)  where K = vocabulary  │
    └─────────────────────────────────────────┘
```

---

## 3. Project Structure

```
lip_reading_system/
│
├── config.py                   ← All hyperparameters & paths
│
├── train.py                    ← Main training loop
├── evaluate.py                 ← Evaluation, confusion matrix, WER/CER
├── inference.py                ← Predict from video file or webcam
│
├── models/
│   ├── __init__.py
│   └── lipreader.py            ← Full model (3D CNN + ResNet + Bi-LSTM)
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py           ← Face detection, landmark, mouth crop
│   └── dataset.py              ← PyTorch Dataset + DataLoaders
│
├── scripts/
│   └── demo_synthetic.py       ← End-to-end test WITHOUT real data
│
├── data/
│   ├── raw/                    ← Place MIRACL-VC1 videos here
│   └── processed/              ← Auto-generated .npy crops
│
├── models/                     ← dlib model files (.dat)
├── checkpoints/                ← Saved model weights
├── outputs/                    ← Confusion matrix, learning curves, logs
│
└── requirements.txt
```

---

## 4. Setup & Installation

### Option A — Local Machine (Linux / macOS / Windows)

#### Step 1: Clone / copy the project
```bash
# If using git:
git clone <your-repo-url>
cd lip_reading_system

# Or just cd into the folder:
cd lip_reading_system
```

#### Step 2: Create and activate a virtual environment
```bash
# Create environment
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Step 3: Install PyTorch (GPU-enabled)

Visit https://pytorch.org/get-started/locally/ and choose your CUDA version.

```bash
# Example for CUDA 12.1 (most common in 2024-2025):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only (slower but works everywhere):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4: Install remaining dependencies

**Linux (Ubuntu/Debian) — install dlib build dependencies first:**
```bash
sudo apt-get update
sudo apt-get install -y cmake libopenblas-dev liblapack-dev libx11-dev
pip install dlib
```

**macOS:**
```bash
brew install cmake
pip install dlib
```

**Windows:**
```bash
# Use the pre-compiled binary wheel (no CMake needed):
pip install dlib-binary
```

**All platforms — install the rest:**
```bash
pip install -r requirements.txt
```

#### Step 5: Download dlib model files

Download both files and place them in the `models/` directory:

| File | Download URL | Size |
|------|-------------|------|
| `shape_predictor_68_face_landmarks.dat` | http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | ~100 MB |
| `mmod_human_face_detector.dat` *(optional, CNN mode)* | http://dlib.net/files/mmod_human_face_detector.dat.bz2 | ~695 KB |

```bash
# Linux/macOS — download and extract:
cd models/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Alternative using Python:
python -c "
import urllib.request, bz2, os
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
urllib.request.urlretrieve(url, 'models/shape_predictor_68_face_landmarks.dat.bz2')
with bz2.open('models/shape_predictor_68_face_landmarks.dat.bz2') as f_in, \
     open('models/shape_predictor_68_face_landmarks.dat', 'wb') as f_out:
    f_out.write(f_in.read())
print('Downloaded!')
"
```

---

## 5. Data Preparation — MIRACL-VC1

### Download the Dataset

1. Go to: https://abenhamadou.github.io/miraclvc1/index.html
2. Fill in the request form (free for academic use)
3. You will receive download links for the dataset

### Expected Directory Structure After Download

```
data/raw/
├── Begin/
│   ├── F01/
│   │   ├── 01.mp4
│   │   ├── 02.mp4
│   │   └── ...   (10 videos)
│   ├── F02/
│   └── ...
├── Choose/
├── Connection/
├── Navigation/
├── Next/
├── Previous/
├── Start/
├── Stop/
├── Hello/
└── Web/
```

> **Note:** MIRACL-VC1 may provide images instead of video files.  
> If you get per-frame PNG images, place them as:  
> `data/raw/<Word>/<SpeakerID>/<SampleID>/frame_XXXX.png`  
> Then convert to video with:
> ```bash
> ffmpeg -framerate 25 -i "data/raw/Begin/F01/01/frame_%04d.png" -c:v libx264 data/raw/Begin/F01/01.mp4
> ```

### Update config.py if your vocabulary differs

Open `config.py` and edit:
```python
VOCABULARY = ["Begin", "Choose", "Connection", "Navigation", "Next",
              "Previous", "Start", "Stop", "Hello", "Web"]
```
Match this exactly to the folder names in `data/raw/`.

---

## 6. Step-by-Step Running Guide

### Step 0: Verify the pipeline with synthetic data (no real data needed)

```bash
python scripts/demo_synthetic.py
```

Expected output:
```
Synthetic dataset: 10 classes × 30 samples
Device: cuda  (or cpu)
Model parameters: 13,xxx,xxx
Epoch 01/10 | Train loss=2.3xxx acc=10.x% | Val loss=2.2xxx acc=12.x%
...
Epoch 10/10 | Train loss=1.8xxx acc=35.x% | Val loss=1.9xxx acc=30.x%
══════════════════════════════════════════════════════
  Test Accuracy: 28.xx%  (x/xx)
══════════════════════════════════════════════════════
Demo complete. Pipeline is verified end-to-end.
```

> Accuracy on random synthetic data will be low — that is expected.  
> If this runs without errors, your installation is correct.

---

### Step 1: Preprocess the MIRACL-VC1 dataset

```bash
python utils/preprocess.py
```

This will:
- Detect faces in every video frame using dlib
- Extract 68 facial landmarks
- Affine-align based on eye centres
- Crop the 96×96 mouth region
- Save as `.npy` files in `data/processed/`

**Expected output:**
```
INFO: Saved data/raw/Begin/F01/01.mp4 -> data/processed/Begin/F01_01.npy
INFO: Saved data/raw/Begin/F01/02.mp4 -> data/processed/Begin/F01_02.npy
...
Clip counts per class:
  Begin               : 150
  Choose              : 150
  ...
Dataset mean=72.4123  std=48.3201
```

**IMPORTANT:** Copy the `mean` and `std` values and update `utils/dataset.py`:
```python
PIXEL_MEAN = 72.4123   # ← replace with your values
PIXEL_STD  = 48.3201
```

---

### Step 2: Train the model

```bash
python train.py
```

**With custom hyperparameters:**
```bash
python train.py --epochs 60 --batch_size 8 --lr 3e-4 --resnet_depth 18
```

**Available arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 8 | Samples per batch |
| `--lr` | 3e-4 | Initial learning rate |
| `--weight_decay` | 1e-4 | L2 regularisation |
| `--resnet_depth` | 18 | ResNet depth (18 or 34) |
| `--pool_mode` | mean | LSTM pooling ("mean" or "last") |
| `--resume` | None | Path to checkpoint to resume from |

**Expected console output:**
```
13:45:01  INFO      Device: cuda
13:45:01  INFO      Data  — train: 150  val: 18  test: 18  batches
13:45:01  INFO      Model parameters: 13,248,522
13:45:12  INFO      Epoch 001/050 | 11s | LR 3.00e-04 | Train loss 2.1823 acc 18.20% | Val loss 2.0912 acc 22.10%
13:45:12  INFO        ✓  New best val_loss=2.0912  saved to checkpoints/best.pth
...
13:52:31  INFO      Epoch 020/050 | 10s | LR 1.50e-04 | Train loss 0.4231 acc 88.50% | Val loss 0.3812 acc 91.20%
```

Training artifacts saved automatically:
- `checkpoints/best.pth` — best validation checkpoint
- `checkpoints/last.pth` — most recent epoch
- `outputs/training_log.csv` — per-epoch metrics

**To resume after interruption:**
```bash
python train.py --resume checkpoints/last.pth
```

---

### Step 3: Evaluate the model

```bash
python evaluate.py
```

**With a specific checkpoint:**
```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

**Output files generated in `outputs/`:**
| File | Description |
|------|-------------|
| `confusion_matrix.png` | Heatmap of per-class predictions |
| `confusion_matrix.csv` | Raw confusion matrix values |
| `learning_curves.png` | Train/val loss & accuracy over epochs |
| `classification_report.txt` | Precision, recall, F1 per class |
| `eval_summary.json` | Summary metrics (WER, CER, accuracy) |

**Expected terminal output (MIRACL-VC1):**
```
══════════════════════════════════════════
  Test Loss      : 0.2341
  Top-1 Accuracy : 93.50%
  Top-5 Accuracy : 99.80%
  WER            : 0.0650  (6.50%)
  CER            : 0.0420  (4.20%)
══════════════════════════════════════════
```

---

### Step 4: Run inference on a video file

```bash
python inference.py --source path/to/your_video.mp4
```

**Expected output:**
```
═════════════════════════════════════════════
  Video     : your_video.mp4
  Frames    : 29
  Prediction: HELLO                (94.3%)
  Top-5:
    1. Hello                94.30%
    2. Navigation           3.10%
    3. Stop                 1.20%
    4. Begin                0.80%
    5. Choose               0.60%
═════════════════════════════════════════════
```

### Step 4b: Webcam live inference

```bash
python inference.py --source webcam
```

- The system captures ~1.5 seconds of video, runs inference, and overlays the predicted word
- Press `r` to reset the buffer
- Press `q` to quit

---

## 7. Running on Google Colab / Kaggle

### Google Colab (recommended for GPU access)

```python
# Cell 1 — Mount Drive and set up
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/

# Cell 2 — Install dependencies
!apt-get install -y cmake libopenblas-dev liblapack-dev -q
!pip install dlib torch torchvision opencv-python seaborn scikit-learn -q

# Cell 3 — Download dlib predictor
!wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
!mv shape_predictor_68_face_landmarks.dat lip_reading_system/models/

# Cell 4 — Upload MIRACL-VC1 (zip) from your machine
from google.colab import files
uploaded = files.upload()   # select your dataset zip

# Cell 5 — Unzip into data/raw/
!unzip miracl_vc1.zip -d lip_reading_system/data/raw/

# Cell 6 — Run the demo first
%cd lip_reading_system
!python scripts/demo_synthetic.py

# Cell 7 — Preprocess
!python utils/preprocess.py

# Cell 8 — Train (GPU auto-detected)
!python train.py --epochs 50 --batch_size 16

# Cell 9 — Evaluate
!python evaluate.py

# Cell 10 — Download outputs
from google.colab import files
import shutil
shutil.make_archive('outputs', 'zip', 'outputs')
files.download('outputs.zip')
```

### Kaggle Notebooks

1. Upload the project as a dataset or use the notebook editor
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Install dlib: `!pip install dlib cmake`
4. Run cells same as Colab above

---

## 8. Configuration Reference

All settings live in `config.py`. Key parameters:

```python
# ── Change vocabulary ──────────────────────────────────────────────────────
VOCABULARY = ["Begin", "Choose", ...]   # Must match data/raw/ folder names
NUM_CLASSES = len(VOCABULARY)

# ── Change model size ──────────────────────────────────────────────────────
RESNET_DEPTH  = 18     # 18 (faster) or 34 (more accurate)
LSTM_HIDDEN   = 256    # increase to 512 for more capacity
LSTM_LAYERS   = 2      # increase to 3 if overfitting improves

# ── Change training behaviour ──────────────────────────────────────────────
NUM_EPOCHS    = 50
BATCH_SIZE    = 8      # reduce to 4 if out-of-memory
LEARNING_RATE = 3e-4   # try 1e-3 for faster convergence initially

# ── Preprocessing ──────────────────────────────────────────────────────────
IMG_SIZE      = 96     # mouth crop size
FRAMES_PER_CLIP = 29   # frames sampled per video
FACE_DETECTOR = "hog"  # "hog" (fast) or "cnn" (accurate, slower)
```

---

## 9. Expected Results

| Dataset | Metric | Expected Value | Notes |
|---------|--------|----------------|-------|
| MIRACL-VC1 | Top-1 Accuracy | > 90% | Small, controlled set |
| MIRACL-VC1 | WER | < 10% | Word-level classification |
| MIRACL-VC1 | CER | < 7% | Character-level |
| LRW (20-word subset) | Top-1 Accuracy | > 60% | In-the-wild |
| LRW (20-word subset) | Top-5 Accuracy | > 80% | |
| Inference latency | ms / 1-sec clip | < 100ms | On GPU |

**Confusion matrix interpretation:**
- High diagonal values → correct predictions
- Off-diagonal clusters between "Stop"/"Start" or "Begin"/"Navigation" → homophene confusion (expected and reported in Section 7.2 of the report)

---

## 10. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: shape_predictor_68_face_landmarks.dat` | Missing dlib model | Download as described in Step 5 of Setup |
| `RuntimeError: No .npy files found` | Preprocessing not run | Run `python utils/preprocess.py` first |
| `CUDA out of memory` | Batch too large | Reduce `BATCH_SIZE` to 4 in `config.py` |
| `0 clips` after preprocessing | Wrong directory structure | Check `data/raw/<Word>/<Speaker>/<Video>.mp4` format |
| Face detection fails on many frames | Poor lighting / angle | Switch to `FACE_DETECTOR = "cnn"` in `config.py` |
| Very low accuracy after 50 epochs | Too little data / LR issues | Try `--lr 1e-3` or add more augmentation |
| `ImportError: dlib` | dlib not installed | See Step 4 for platform-specific install |

**Check GPU availability:**
```python
import torch
print(torch.cuda.is_available())       # Should print True
print(torch.cuda.get_device_name(0))   # Should print your GPU name
```

---

## 11. References

1. Stafylakis, T., & Tzimiropoulos, G. (2017). *Combining Residual Networks with LSTMs for Lipreading.* Interspeech. https://arxiv.org/abs/1703.04105
2. Chung, J. S., et al. (2017). *Lip Reading Sentences in the Wild.* CVPR. https://openaccess.thecvf.com/content_cvpr_2017/papers/Chung_Lip_Reading_Sentences_CVPR_2017_paper.pdf
3. Afouras, T., et al. (2018). *Deep Lip Reading: A Comparison of Models.* Interspeech.
4. Rekik, A., et al. (2014). *MIRACL-VC1: an RGB-D dataset for lipreading.* https://abenhamadou.github.io/miraclvc1/
5. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
6. Manipal University Jaipur. (2026). *Course Hand-out: DEEP LEARNING LAB (AIM 3230).*

---

*Built for AIM 3230 Deep Learning Lab · Manipal University Jaipur · Jan–May 2026*
