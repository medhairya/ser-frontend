from fastapi import FastAPI, UploadFile, File
import torch
import uvicorn
from src.models.avtca import AVTCA
from config import Config

app = FastAPI(title="Emotion Recognition API")

# 1. Initialize Configuration and Model
cfg = Config()
device = torch.device("cpu") # Use CPU for deployment to save costs

# Initialize the architecture (matches the one in train.py)
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
checkpoint = torch.load("best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() # Set model to evaluation mode

@app.get("/")
def read_root():
    return {"message": "AVTCA Emotion Recognition API is running"}

@app.post("/predict")
async def predict(audio: UploadFile = File(...), video: UploadFile = File(...)):
    # 2. Add your Preprocessing logic here
    # Read the audio/video bytes, convert them to the expected tensors
    # audio_tensor = ...
    # video_tensor = ...
    
    # 3. Perform Inference
    # with torch.no_grad():
    #     logits = model(audio_tensor, video_tensor)
    #     predicted_class = torch.argmax(logits, dim=1).item()
    
    return {"status": "success", "prediction": "happy"} # Replace with predicted_class

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
