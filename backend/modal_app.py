"""
Modal deployment: GPU-accelerated FastAPI for AVT-CA (audio + video → emotion).

Deploy:
  cd backend
  pip install modal
  modal setup
  modal deploy modal_app.py

Set function secrets / env in the Modal dashboard (recommended):
  HF_REPO_ID          e.g. your-username/avtca-ravdess-weights
  HF_CHECKPOINT_FILENAME   default best_model.pt
  HF_TOKEN            only if the HF model repo is private
  ALLOWED_ORIGINS     comma-separated list, e.g. https://your-app.vercel.app

Local smoke test (CPU, weights next to this file or HF env set):
  pip install -r requirements.txt
  uvicorn local_app:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "ser-avtca-api"
LOCAL_DIR = Path(__file__).parent.resolve()

# CUDA 12.4 wheels; T4 is cost-effective for this model size.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libgl1-mesa-glx",
    )
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "python-multipart>=0.0.9",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.24.0",
        "pydantic==2.12.6",
    )
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(LOCAL_DIR, remote_path="/root")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    from inference import get_predictor

    web = FastAPI(title="AVT-CA Speech & Video Emotion Recognition", version="1.0.0")

    origins_raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

    web.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/health")
    def health():
        return {"status": "ok", "service": "ser-avtca"}

    @web.post("/predict")
    async def predict(file: UploadFile = File(...)):
        import tempfile

        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        suffix = Path(file.filename).suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
            raise HTTPException(
                status_code=400,
                detail="Upload a video file (.mp4, .webm, .avi, .mov, .mkv)",
            )

        try:
            predictor = get_predictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}") from e

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            result = predictor.predict_file(tmp_path)
            return {"status": "success", **result}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return web
