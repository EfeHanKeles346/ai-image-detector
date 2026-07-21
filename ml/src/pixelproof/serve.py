"""Phase 6: inference service for the web demo.

Loads both trained models and routes each upload by resolution — the practical
answer to E5/E6's finding that each model is blind outside its resolution domain:
- small inputs (max side < 128px) -> SmallCNN trained on 32x32 CIFAKE
- everything else -> ResNet-18 fine-tuned on native-resolution GenImage
Predictions inside the E3 uncertainty band (|p - 0.5| < 0.1) are reported as
"uncertain" instead of a hard verdict.

Run: PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799
"""

import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pixelproof.evaluate import eval_transform
from pixelproof.models import create_model

ROUTING_THRESHOLD_PX = 128
UNCERTAINTY_BAND = 0.1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
artifacts = Path(__file__).resolve().parents[2] / "artifacts"


def load(checkpoint_name: str):
    checkpoint = torch.load(artifacts / checkpoint_name, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    transform = eval_transform(config["data"]["image_size"], config["data"].get("normalization", "default"))
    return model, transform


MODELS = {
    "small_cnn_cifake": load("best.pt"),
    "resnet18_genimage": load("best_genimage.pt"),
}

app = FastAPI(title="PixelProof inference")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    raw = await image.read()
    picture = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = picture.size

    model_name = "small_cnn_cifake" if max(width, height) < ROUTING_THRESHOLD_PX else "resnet18_genimage"
    model, transform = MODELS[model_name]
    with torch.no_grad():
        probability = torch.sigmoid(model(transform(picture).unsqueeze(0).to(device))).item()

    if abs(probability - 0.5) < UNCERTAINTY_BAND:
        verdict = "uncertain"
    else:
        verdict = "ai" if probability >= 0.5 else "real"
    return {
        "p_ai": round(probability, 4),
        "verdict": verdict,
        "model": model_name,
        "resolution": f"{width}x{height}",
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "models": list(MODELS)}
