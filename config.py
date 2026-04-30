"""
config.py  –  All hyperparameters and paths for AVT-CA.

Edit the PATH section below for your environment before running.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ─────────────────────────────────────────────────────────────
    # PATHS  ← EDIT THESE
    # ─────────────────────────────────────────────────────────────

    # ── Lightning AI  (dataset downloaded via kagglehub) ──────────
    DATA_ROOT: str = (
        "/teamspace/studios/this_studio/.cache/kagglehub"
        "/datasets/orvile/ravdess-dataset/versions/1"
    )
    CHECKPOINT_DIR: str = "/teamspace/studios/this_studio/avtca/checkpoints"
    LOG_DIR: str        = "/teamspace/studios/this_studio/avtca/logs"
    CACHE_DIR: str      = "/teamspace/studios/this_studio/avtca/cache"

    # ── Kaggle notebooks ───────────────────────────────────────────
    # DATA_ROOT      = "/kaggle/input/ravdess-dataset"
    # CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    # LOG_DIR        = "/kaggle/working/logs"
    # CACHE_DIR      = "/kaggle/working/cache"

    # ─────────────────────────────────────────────────────────────
    # AUDIO PREPROCESSING
    # ─────────────────────────────────────────────────────────────
    SAMPLE_RATE: int = 22050
    N_MELS: int = 64
    N_FFT: int = 1024
    HOP_LENGTH: int = 512
    MAX_AUDIO_LEN: int = 128

    # ─────────────────────────────────────────────────────────────
    # VIDEO PREPROCESSING
    # ─────────────────────────────────────────────────────────────
    NUM_FRAMES: int = 16
    FRAME_H: int = 112
    FRAME_W: int = 112

    # RAVDESS modality codes:
    #   "01" = full audio-video  ← recommended
    #   "02" = video-only
    #   ""   = accept all files
    MODALITY_FILTER: str = "01"

    # ─────────────────────────────────────────────────────────────
    # MODEL ARCHITECTURE
    # ─────────────────────────────────────────────────────────────
    D_MODEL: int = 256
    NUM_HEADS: int = 4
    NUM_TRANSFORMER_LAYERS: int = 2
    FFN_DIM: int = 1024
    DROPOUT: float = 0.1
    CNN_CH: int = 64

    # ─────────────────────────────────────────────────────────────
    # DATASET
    # ─────────────────────────────────────────────────────────────
    NUM_CLASSES: int = 8
    EMOTION_LABELS: List[str] = field(default_factory=lambda: [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised",
    ])
    TRAIN_RATIO: float = 0.8

    # ─────────────────────────────────────────────────────────────
    # TRAINING  ← KEY FIXES HERE
    # ─────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 32          # was 8 → bigger batch = stable gradients
                                  # L4 has 24GB, model uses ~2GB, lots of room
    LEARNING_RATE: float = 3e-4   # was 1e-2 (WAY too high for Adam)
                                  # 3e-4 is the standard Adam sweet spot
    WEIGHT_DECAY: float = 1e-4    # was 1e-3, slightly relaxed
    NUM_EPOCHS: int = 128
    SEED: int = 42
    NUM_WORKERS: int = 6          # L4 has 8 CPUs
    PIN_MEMORY: bool = True

    # ─────────────────────────────────────────────────────────────
    # CHECKPOINTING
    # ─────────────────────────────────────────────────────────────
    SAVE_EVERY_N_EPOCHS: int = 10
    EARLY_STOP_PATIENCE: int = 40  # was 30, give it more breathing room