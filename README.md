# AI Lip Reading System using Deep Learning

An AI-powered visual speech recognition system that predicts spoken words from lip movements using Deep Learning.

Built using:
- Python
- OpenCV
- PyTorch
- CNN + BiLSTM Architecture

The system processes silent video input, extracts mouth-region features, and predicts words such as:
- Hello
- Start
- Stop

---

# Features

- Real-time webcam inference
- Mouth region extraction using facial landmarks
- 3D CNN + ResNet-18 feature extraction
- Bidirectional LSTM temporal modeling
- Word-level lip reading prediction
- Evaluation using accuracy, WER, and CER

---

# Architecture

```text
Raw Video
   ↓
Face Detection
   ↓
Mouth ROI Extraction (96×96)
   ↓
3D CNN + ResNet-18
   ↓
BiLSTM
   ↓
Softmax Classification
   ↓
Predicted Word
```

---

# Project Structure

```bash
lip_reading_system/
│
├── train.py
├── evaluate.py
├── inference.py
├── config.py
│
├── models/
├── utils/
├── data/
├── checkpoints/
├── outputs/
│
└── requirements.txt
```

---

# Installation

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Download and place the dataset inside:

```bash
data/raw/
```

Recommended Dataset:
- MIRACL-VC1 Lip Reading Dataset

Expected structure:

```bash
data/raw/
├── Hello/
├── Start/
├── Stop/
```

---

# Running the Project

## 1. Preprocess Dataset

```bash
python utils/preprocess.py
```

This step:
- Detects face
- Extracts lip region
- Converts frames into processed sequences

---

## 2. Train the Model

```bash
python train.py
```

Optional custom training:

```bash
python train.py --epochs 50 --batch_size 8
```

---

## 3. Evaluate the Model

```bash
python evaluate.py
```

This generates:
- Accuracy metrics
- Confusion matrix
- WER/CER scores

---

## 4. Run Live Inference

```bash
python inference.py --source webcam
```

The webcam opens and predicts words in real time based on lip movement.

Press:
- `q` → Quit
- `r` → Reset buffer

---

# Model Pipeline

```text
Input Video Frames
        ↓
Preprocessing
        ↓
Lip Region Extraction
        ↓
CNN Feature Extraction
        ↓
BiLSTM Temporal Learning
        ↓
Dense Classification Layer
        ↓
Predicted Output Word
```

---

# Expected Results

| Metric | Expected Performance |
|--------|----------------------|
| Accuracy | >90% |
| WER | <10% |
| CER | <7% |

The model performs well on controlled vocabulary lip-reading tasks.

---

# Demo

Add screenshots or GIFs here showing:
- Webcam inference
- Predicted output
- Training graphs
- Confusion matrix

Example:

```md
![Demo](demo/demo.gif)
```

---

# Technologies Used

- Python
- OpenCV
- PyTorch
- NumPy
- dlib
- CNN
- BiLSTM

---

# Future Improvements

- Larger vocabulary support
- Sentence-level lip reading
- Transformer-based architectures
- Improved real-world robustness
- Multilingual visual speech recognition

---

# References

- MIRACL-VC1 Dataset
- PyTorch Documentation
- OpenCV Documentation
- Research papers on Lip Reading using Deep Learning

---

# Author

Arnav Kedia  
B.Tech CSE (AI & ML)
