from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import cv2
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import tempfile
import os

from src.models.avtca import AVTCA
from config import Config

app = FastAPI(title="Emotion Recognition API")

# Allow CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to your vercel domain later for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 1. Initialize Configuration and Model
cfg = Config()
device = torch.device("cpu") # Use CPU for deployment to save costs

# Initialize the architecture
model = AVTCA(
    num_classes=cfg.NUM_CLASSES,
    n_mels=cfg.N_MELS,
    d_model=cfg.D_MODEL,
    num_heads=cfg.NUM_HEADS,
    num_transformer_layers=cfg.NUM_TRANSFORMER_LAYERS,
    ffn_dim=cfg.FFN_DIM,
    dropout=cfg.DROPOUT,
    cnn_ch=cfg.CNN_CH,
).to(device)

# Load the weights
try:
    checkpoint = torch.load("best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

EMOTION_NAMES = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]

# Audio transform setup
mel_transform = T.MelSpectrogram(
    sample_rate=cfg.SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=cfg.N_MELS,
    power=2.0,
)
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

def process_audio(path: str) -> torch.Tensor:
    # Use torchaudio directly
    try:
        waveform, sr = torchaudio.load(path)
        if sr != cfg.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.SAMPLE_RATE)
    except Exception:
        # Fallback to silence if torchaudio fails without ffmpeg
        waveform = torch.zeros(1, cfg.SAMPLE_RATE * 3)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    mel = mel_transform(waveform)
    mel = amplitude_to_db(mel)
    mel = mel.squeeze(0)  # (n_mels, T)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    T_cur = mel.shape[1]
    if T_cur >= cfg.MAX_AUDIO_LEN:
        mel = mel[:, :cfg.MAX_AUDIO_LEN]
    else:
        mel = F.pad(mel, (0, cfg.MAX_AUDIO_LEN - T_cur))

    return mel.unsqueeze(0)  # Add batch dim (1, n_mels, max_audio_len)

def process_video(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1

    indices = np.linspace(0, total - 1, cfg.NUM_FRAMES, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, (cfg.FRAME_W, cfg.FRAME_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((cfg.FRAME_H, cfg.FRAME_W, 3), dtype=np.uint8)
        frames.append(frame)

    cap.release()

    frames = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / (std + 1e-8)

    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, 3, H, W)
    return frames.unsqueeze(0)  # Add batch dim (1, T, 3, H, W)

@app.get("/")
def read_root():
    return {"message": "AVTCA Emotion Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 1. Process Video
            video_tensor = process_video(tmp_path).to(device)
            # 2. Process Audio
            audio_tensor = process_audio(tmp_path).to(device)

            # 3. Inference
            with torch.no_grad():
                logits = model(audio_tensor, video_tensor)
                probs = F.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class_idx].item()
                
            predicted_emotion = EMOTION_NAMES[predicted_class_idx] if predicted_class_idx < len(EMOTION_NAMES) else "unknown"

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "status": "success",
            "prediction": predicted_emotion,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
