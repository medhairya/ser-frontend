"""
Local FastAPI server (CPU by default). Useful before Modal deploy.

  set HF_REPO_ID=your/name
  set HF_CHECKPOINT_FILENAME=best_model.pt
  uvicorn local_app:app --host 0.0.0.0 --port 8000

Or place best_model.pt in this directory and omit HF_REPO_ID.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import get_predictor

app = FastAPI(title="AVT-CA SER (local)", version="1.0.0")

_origins = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_allow = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow if _allow else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ser-avtca-local"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
