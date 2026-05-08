"""
inference.py — Real-Time / File Inference for the AI Lip-Reading System
=======================================================================
Usage (video file):
    python inference.py --source path/to/video.mp4

Usage (webcam):
    python inference.py --source webcam

Outputs: predicted word + confidence, overlaid on video frames (if webcam).
"""

import os
import sys
import time
import argparse
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    CHECKPOINTS_DIR, VOCABULARY, DEVICE,
    IMG_SIZE, FRAMES_PER_CLIP, GRAYSCALE,
)
from models.lipreader import LipReadingModel
from utils.dataset import PIXEL_MEAN, PIXEL_STD

# Import preprocessing utilities — graceful fallback if dlib not installed
try:
    from utils.preprocess import FrameProcessor
    DLIB_AVAILABLE = True
except Exception as e:
    DLIB_AVAILABLE = False
    logging.warning("dlib not available (%s). Using raw centre-crop fallback.", e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Fallback centre-crop (no dlib) ─────────────────────────────────────────

def centre_crop(frame_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Crude centre crop for testing when dlib is unavailable."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size))
    if GRAYSCALE:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


# ─── Inference engine ────────────────────────────────────────────────────────

class LipReader:
    """Load model once; call predict() with a list of BGR frames."""

    def __init__(self, checkpoint: str, vocab: list = VOCABULARY):
        self.vocab  = vocab
        self.device = DEVICE

        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model = LipReadingModel(num_classes=len(vocab)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info("Model loaded from %s  (epoch %d)", checkpoint,
                    ckpt.get("epoch", -1))

        if DLIB_AVAILABLE:
            self.processor = FrameProcessor()
        else:
            self.processor = None

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.processor is not None:
            crop = self.processor.process(frame_bgr)
            if crop is None:
                crop = centre_crop(frame_bgr)
        else:
            crop = centre_crop(frame_bgr)
        return crop

    def _frames_to_tensor(self, crops: list) -> torch.Tensor:
        """Convert list of mouth crops → (1, C, T, H, W) tensor."""
        # Pad / truncate to FRAMES_PER_CLIP
        while len(crops) < FRAMES_PER_CLIP:
            crops.append(crops[-1])
        crops = crops[:FRAMES_PER_CLIP]

        arr = np.stack(crops, axis=0).astype(np.float32)  # (T, H, W)
        arr = (arr - PIXEL_MEAN) / (PIXEL_STD + 1e-6)

        if arr.ndim == 3:                # grayscale
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            # → (1, 1, T, H, W)
        else:                            # RGB
            tensor = torch.from_numpy(arr.transpose(3, 0, 1, 2)).unsqueeze(0)
            # → (1, 3, T, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frames_bgr: list):
        """
        Parameters
        ----------
        frames_bgr : list of np.ndarray (BGR uint8)
            At least 1 frame; will be resampled to FRAMES_PER_CLIP.

        Returns
        -------
        word       : str
        confidence : float  (0–1)
        top5       : list of (word, confidence)
        """
        crops  = [self._process_frame(f) for f in frames_bgr]
        tensor = self._frames_to_tensor(crops)

        logits = self.model(tensor)          # (1, num_classes)
        probs  = F.softmax(logits, dim=-1)[0]

        top5_vals, top5_idx = probs.topk(min(5, len(self.vocab)))
        top5 = [(self.vocab[i], float(v)) for i, v in zip(top5_idx, top5_vals)]

        pred_idx = int(probs.argmax())
        return self.vocab[pred_idx], float(probs[pred_idx]), top5


# ─── Video-file inference ────────────────────────────────────────────────────

def infer_video(lip_reader: LipReader, video_path: str):
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        logger.error("No frames read from %s", video_path)
        return

    word, conf, top5 = lip_reader.predict(frames)
    print(f"\n{'═'*45}")
    print(f"  Video     : {os.path.basename(video_path)}")
    print(f"  Frames    : {len(frames)}")
    print(f"  Prediction: {word.upper():<20s}  ({conf*100:.1f}%)")
    print(f"  Top-5:")
    for rank, (w, c) in enumerate(top5, 1):
        print(f"    {rank}. {w:<20s} {c*100:.2f}%")
    print(f"{'═'*45}\n")


# ─── Webcam / live inference ──────────────────────────────────────────────────

def infer_webcam(lip_reader: LipReader, buffer_seconds: float = 1.5):
    """
    Capture `buffer_seconds` of webcam frames, run inference, repeat.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    buffer_size = max(FRAMES_PER_CLIP, int(fps * buffer_seconds))

    buffer  = []
    last_pred = ""
    last_conf = 0.0

    logger.info("Webcam live inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame.copy())

        # Overlay last prediction
        display = frame.copy()
        cv2.putText(
            display,
            f"{last_pred}  ({last_conf*100:.1f}%)",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display, "Press 'q' to quit | 'r' to reset buffer",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
        cv2.imshow("AI Lip-Reading System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            buffer = []

        if len(buffer) >= buffer_size:
            t0 = time.time()
            word, conf, _ = lip_reader.predict(buffer[-buffer_size:])
            latency = (time.time() - t0) * 1000
            last_pred = word
            last_conf = conf
            logger.info("Prediction: %s  (%.1f%%)  latency=%.1fms",
                        word, conf * 100, latency)
            buffer = []   # reset buffer

    cap.release()
    cv2.destroyAllWindows()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Lip-Reading Inference")
    p.add_argument("--source",     type=str,   default="webcam",
                   help="Path to video file, or 'webcam'")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(CHECKPOINTS_DIR, "best.pth"))
    p.add_argument("--buffer",     type=float, default=1.5,
                   help="Webcam buffer duration in seconds")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}\n"
                 "Run train.py first.")

    reader = LipReader(args.checkpoint)

    if args.source.lower() == "webcam":
        infer_webcam(reader, buffer_seconds=args.buffer)
    else:
        infer_video(reader, args.source)
