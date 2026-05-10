"""
Inference-oriented config for AVT-CA (matches training hyperparameters).
Training paths are omitted; use HF_REPO_ID + checkpoint file for deployed weights.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    SAMPLE_RATE: int = 22050
    N_MELS: int = 64
    N_FFT: int = 1024
    HOP_LENGTH: int = 512
    MAX_AUDIO_LEN: int = 128

    NUM_FRAMES: int = 16
    FRAME_H: int = 112
    FRAME_W: int = 112

    D_MODEL: int = 256
    NUM_HEADS: int = 4
    NUM_TRANSFORMER_LAYERS: int = 2
    FFN_DIM: int = 1024
    DROPOUT: float = 0.1
    CNN_CH: int = 64

    NUM_CLASSES: int = 8
    EMOTION_LABELS: List[str] = field(default_factory=lambda: [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised",
    ])
