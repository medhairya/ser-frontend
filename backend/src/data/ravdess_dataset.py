"""
src/data/ravdess_dataset.py
───────────────────────────
RAVDESS dataset loader.

Filename format (RAVDESS standard):
    {modality}-{vocal_ch}-{emotion}-{intensity}-{statement}-{rep}-{actor}.mp4
    e.g.  01-01-05-01-01-01-12.mp4

    modality  01 = full-AV, 02 = video-only, 03 = audio-only
    emotion   01=neutral 02=calm 03=happy 04=sad 05=angry
              06=fearful 07=disgust 08=surprised
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

# ── Label maps ────────────────────────────────────────────────
EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}
EMOTION_NAMES = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]


# ─────────────────────────────────────────────────────────────
# Main Dataset Class
# ─────────────────────────────────────────────────────────────

class RAVDESSDataset(Dataset):
    """
    Loads audio-visual clips from RAVDESS.

    Args:
        root_dir        : path to RAVDESS root (contains Actor_XX folders)
        split           : 'train' | 'val'
        train_ratio     : fraction of samples used for training
        num_frames      : number of video frames to sample per clip
        frame_size      : (H, W) to resize each frame
        n_mels          : mel spectrogram bins
        sr              : target audio sample rate
        max_audio_len   : fixed time-axis length of mel spectrogram
        modality_filter : keep only files whose modality code matches
                          ('01' = full AV).  Pass '' to keep everything.
        seed            : RNG seed for reproducible train/val split
        cache_dir       : directory to cache processed tensors ('' = no cache)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        num_frames: int = 16,
        frame_size: tuple = (112, 112),
        n_mels: int = 64,
        sr: int = 22050,
        max_audio_len: int = 128,
        modality_filter: str = "01",
        seed: int = 42,
        cache_dir: str = "",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.n_mels = n_mels
        self.sr = sr
        self.max_audio_len = max_audio_len
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Audio transforms ──────────────────────────────────
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # ── Scan files and build split ─────────────────────────
        all_samples = self._scan_files(modality_filter)
        if not all_samples:
            raise RuntimeError(
                f"No valid files found in {root_dir}. "
                "Check DATA_ROOT and MODALITY_FILTER in config.py."
            )

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_samples))
        rng.shuffle(indices)
        cut = int(len(all_samples) * train_ratio)

        if split == "train":
            chosen = indices[:cut]
        else:
            chosen = indices[cut:]

        self.samples = [all_samples[i] for i in chosen]
        logger.info(f"[RAVDESSDataset/{split}] {len(self.samples)} samples")

    # ── Internal helpers ─────────────────────────────────────

    def _scan_files(self, modality_filter: str):
        """Recursively find video files and parse labels."""
        samples = []
        for path in sorted(self.root_dir.rglob("*.mp4")):
            label = self._parse_filename(path.stem, modality_filter)
            if label is not None:
                samples.append({"path": str(path), "label": label})

        # Fallback: accept .avi files
        if not samples:
            for path in sorted(self.root_dir.rglob("*.avi")):
                label = self._parse_filename(path.stem, modality_filter)
                if label is not None:
                    samples.append({"path": str(path), "label": label})

        # Fallback: .wav files (audio-only, video will be zeros)
        if not samples:
            logger.warning(
                "No .mp4/.avi found – falling back to .wav (video will be zeros)."
            )
            for path in sorted(self.root_dir.rglob("*.wav")):
                label = self._parse_filename(path.stem, modality_filter=None)
                if label is not None:
                    samples.append({"path": str(path), "label": label})

        return samples

    @staticmethod
    def _parse_filename(stem: str, modality_filter):
        """Return emotion label (0-indexed) or None if invalid / filtered."""
        parts = stem.split("-")
        if len(parts) != 7:
            return None
        modality, _vocal, emotion, _intensity, _stmt, _rep, _actor = parts
        if modality_filter and modality != modality_filter:
            return None
        return EMOTION_MAP.get(emotion)

    def _cache_path(self, path: str) -> Path:
        stem = Path(path).stem
        return self.cache_dir / f"{stem}.pt"

    # ── Public API ───────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        path, label = item["path"], item["label"]

        # ── Try loading from cache ─────────────────────────
        if self.cache_dir:
            cp = self._cache_path(path)
            if cp.exists():
                try:
                    data = torch.load(cp, map_location="cpu", weights_only=True)
                    return data["audio"], data["video"], label
                except Exception:
                    pass  # corrupted cache → re-process

        # ── Load from disk ─────────────────────────────────
        audio = self._load_audio(path)
        video = self._load_video(path)

        # ── Write to cache ─────────────────────────────────
        if self.cache_dir:
            try:
                torch.save({"audio": audio, "video": video}, self._cache_path(path))
            except Exception as e:
                logger.debug(f"Cache write failed for {path}: {e}")

        return audio, video, label

    # ── Audio loading ────────────────────────────────────────

    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Returns mel spectrogram of shape (n_mels, max_audio_len).
        """
        waveform, orig_sr = _load_waveform(path, self.sr)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Mel spectrogram → (1, n_mels, T)
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        mel = mel.squeeze(0)  # (n_mels, T)

        # Normalise per clip
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Pad or crop to fixed length
        T_cur = mel.shape[1]
        if T_cur >= self.max_audio_len:
            mel = mel[:, :self.max_audio_len]
        else:
            mel = F.pad(mel, (0, self.max_audio_len - T_cur))

        return mel  # (n_mels, max_audio_len)

    # ── Video loading ────────────────────────────────────────

    def _load_video(self, path: str) -> torch.Tensor:
        """
        Returns (num_frames, 3, H, W) float32 tensor,
        normalised with ImageNet mean/std.
        """
        if path.endswith(".wav"):
            return torch.zeros(self.num_frames, 3, *self.frame_size)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {path}")
            return torch.zeros(self.num_frames, 3, *self.frame_size)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, self.frame_size[::-1])   # (W, H)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            frames.append(frame)

        cap.release()

        frames = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, 3)

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames = (frames - mean) / (std + 1e-8)

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, 3, H, W)
        return frames


# ─────────────────────────────────────────────────────────────
# Helper: robust waveform loader
# ─────────────────────────────────────────────────────────────

def _load_waveform(path: str, target_sr: int):
    """
    Load waveform from a .wav or .mp4/.avi file.
    Tries three strategies in order:
      1. torchaudio.load  (works for .wav and, with ffmpeg, .mp4)
      2. ffmpeg subprocess → temp .wav → torchaudio.load
      3. Return silence (zeros)
    """
    # Strategy 1: direct torchaudio load
    try:
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform, target_sr
    except Exception:
        pass

    # Strategy 2: ffmpeg subprocess
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
            timeout=30,
        )
        if result.returncode == 0:
            waveform, sr = torchaudio.load(tmp)
            os.unlink(tmp)
            return waveform, sr
    except Exception as e:
        logger.debug(f"ffmpeg extraction failed for {path}: {e}")

    # Strategy 3: silence
    logger.warning(f"Returning silence for {path}")
    return torch.zeros(1, target_sr * 3), target_sr
