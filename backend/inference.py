"""
Load AVT-CA weights and run multimodal inference on a video file (audio + frames).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from huggingface_hub import hf_hub_download

from config import Config
from src.models.avtca import AVTCA

log = logging.getLogger(__name__)


def _load_waveform(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Match RAVDESS loader: torchaudio first, then ffmpeg, else short silence."""
    try:
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform, target_sr
    except Exception:
        pass

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        result = subprocess.run(
            [
                "ffmpeg", "-i", path,
                "-ar", str(target_sr), "-ac", "1",
                tmp, "-y", "-loglevel", "quiet",
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            waveform, sr = torchaudio.load(tmp)
            os.unlink(tmp)
            return waveform, sr
    except Exception as e:
        log.debug("ffmpeg extraction failed for %s: %s", path, e)

    log.warning("Returning silence for %s", path)
    return torch.zeros(1, target_sr * 3), target_sr


class AVTCAPredictor:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.cfg.SAMPLE_RATE,
            n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH,
            n_mels=self.cfg.N_MELS,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        self.model = AVTCA(
            num_classes=self.cfg.NUM_CLASSES,
            n_mels=self.cfg.N_MELS,
            d_model=self.cfg.D_MODEL,
            num_heads=self.cfg.NUM_HEADS,
            num_transformer_layers=self.cfg.NUM_TRANSFORMER_LAYERS,
            ffn_dim=self.cfg.FFN_DIM,
            dropout=self.cfg.DROPOUT,
            cnn_ch=self.cfg.CNN_CH,
        ).to(self.device)
        self._load_weights()
        self.model.eval()

    def _resolve_checkpoint_path(self) -> Path:
        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        filename = os.environ.get("HF_CHECKPOINT_FILENAME", "best_model.pt").strip()
        local_override = os.environ.get("CHECKPOINT_PATH", "").strip()

        if local_override:
            p = Path(local_override)
            if p.is_file():
                return p
            raise FileNotFoundError(f"CHECKPOINT_PATH set but not found: {p}")

        if repo_id:
            token = os.environ.get("HF_TOKEN") or None
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
            )
            return Path(path)

        for name in ("best_model.pt", "best.pt"):
            p = Path(__file__).resolve().parent / name
            if p.is_file():
                return p

        raise FileNotFoundError(
            "No weights found. Set HF_REPO_ID (+ optional HF_CHECKPOINT_FILENAME) "
            "or place best_model.pt next to inference.py, or set CHECKPOINT_PATH."
        )

    def _load_weights(self) -> None:
        ckpt_path = self._resolve_checkpoint_path()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Checkpoint must be a dict with model_state_dict or a raw state_dict")
        self.model.load_state_dict(state)
        log.info("Loaded weights from %s", ckpt_path)

    def process_audio(self, path: str) -> torch.Tensor:
        waveform, _ = _load_waveform(path, self.cfg.SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        mel = mel.squeeze(0)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        t_cur = mel.shape[1]
        if t_cur >= self.cfg.MAX_AUDIO_LEN:
            mel = mel[:, : self.cfg.MAX_AUDIO_LEN]
        else:
            mel = F.pad(mel, (0, self.cfg.MAX_AUDIO_LEN - t_cur))

        return mel.unsqueeze(0)

    def process_video(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1

        indices = np.linspace(0, total - 1, self.cfg.NUM_FRAMES, dtype=int)
        frames: List[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (self.cfg.FRAME_W, self.cfg.FRAME_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((self.cfg.FRAME_H, self.cfg.FRAME_W, 3), dtype=np.uint8)
            frames.append(frame)

        cap.release()

        arr = np.stack(frames).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / (std + 1e-8)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def predict_file(self, path: str) -> Dict[str, Any]:
        audio = self.process_audio(path).to(self.device)
        video = self.process_video(path).to(self.device)
        logits = self.model(audio, video)
        probs = F.softmax(logits, dim=1).squeeze(0)
        idx = int(probs.argmax().item())
        labels = self.cfg.EMOTION_LABELS
        out = {
            "emotion": labels[idx] if idx < len(labels) else "unknown",
            "confidence": round(float(probs[idx].item()), 4),
            "probabilities": {
                labels[i]: round(float(probs[i].item()), 4)
                for i in range(min(len(labels), probs.numel()))
            },
        }
        return out


_predictor: Optional[AVTCAPredictor] = None


def get_predictor() -> AVTCAPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AVTCAPredictor()
    return _predictor
