"""Inference service for the web demo.

Module 1 — "is this image AI-generated?" — offers three methods. The caller picks
one; the service does not blend them. E9 tested eight blending rules and the best
beat the CNN alone by +0.002, because a fixed blend relocates accuracy rather
than adding it (Defactify +0.036, archive1 -0.036).

| method    | what it does                                    | strongest where          |
|-----------|-------------------------------------------------|--------------------------|
| `auto`    | picks by image size using the measured crossover | default                  |
| `cnn`     | SmallCNN (<128px) or ResNet-18, resized to 224   | small / compressed input |
| `stats`   | 68 statistics over every pixel, no resizing      | mid-size, low-texture    |
| `tiles`   | 6x6 grid of native 128px crops, top-3 mean       | >=700px (best overall)   |

The crossover is measured, not guessed (E11): above ~700px the tile method beats
the CNN by +0.17 to +0.23 AUC on every modern generator tested; below it the CNN
wins. `auto` encodes exactly that rule.

Tile scores double as a localisation map — see /predict `tile_map`.

Run: PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799
"""

import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pixelproof.evaluate import eval_transform
from pixelproof.feature_model import load as load_feature_models
from pixelproof.feature_model import score_image, score_tiles
from pixelproof.models import create_model
from pixelproof.verdict import VerdictService

# --- measured constants (see EXPERIMENTS.md) --------------------------------
CNN_ROUTING_PX = 128      # E6: below this only SmallCNN has seen anything similar
TILE_RELIABLE_PX = 700    # E11: above this the tile method wins decisively
UNCERTAINTY_BAND = 0.1    # E3: errors concentrate inside |p-0.5| < 0.1
EVIDENCE_FLOOR_PX = 48    # below this no method has enough pixels to measure texture

STATS_VARIANTS = {"stats": "full"}

METHODS = {
    "auto":   "Otomatik — görsel boyutuna göre en güçlü yöntem",
    "cnn":    "CNN — küçültülmüş görsel, sinir ağı",
    "stats":  "İstatistik — 68 ölçüm, 9.9k GenImage ile eğitildi",
    "tiles":  "Kare kare — 6×6 ızgara, orijinal çözünürlük",
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
artifacts = Path(__file__).resolve().parents[2] / "artifacts"


def load(checkpoint_name: str):
    checkpoint = torch.load(artifacts / checkpoint_name, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    transform = eval_transform(config["data"]["image_size"],
                               config["data"].get("normalization", "default"))
    return model, transform


CNNS = {"small_cnn_cifake": load("best.pt"), "resnet18_genimage": load("best_genimage.pt")}
FEATURES = load_feature_models(artifacts)
# The decision layer (E22-E24): frozen external arms behind the asymmetric
# band. Research signals above stay untouched; this is the served verdict.
VERDICT = VerdictService(device, artifacts.parent)

app = FastAPI(title="PixelProof inference")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def verdict_for(probability: float) -> str:
    if abs(probability - 0.5) < UNCERTAINTY_BAND:
        return "uncertain"
    return "ai" if probability >= 0.5 else "real"


def run_cnn(picture: Image.Image) -> tuple[float, str]:
    name = ("small_cnn_cifake" if max(picture.size) < CNN_ROUTING_PX else "resnet18_genimage")
    model, transform = CNNS[name]
    with torch.no_grad():
        probability = torch.sigmoid(model(transform(picture).unsqueeze(0).to(device))).item()
    return probability, name


@app.post("/predict")
async def predict(image: UploadFile = File(...), method: str = Form("auto")):
    raw = await image.read()
    picture = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = picture.size
    longest = max(width, height)

    if method not in METHODS:
        method = "auto"
    chosen = method
    if method == "auto":
        chosen = "tiles" if longest >= TILE_RELIABLE_PX else "cnn"

    tile_map = None
    if chosen == "cnn":
        probability, engine = run_cnn(picture)
    elif chosen in STATS_VARIANTS:
        variant = STATS_VARIANTS[chosen]
        if variant not in FEATURES:                    # model not trained yet
            variant = "full"
            chosen = "stats"
        probability = score_image({variant: FEATURES[variant]}, picture)[variant]
        engine = f"feature_{variant}"
    else:
        tile_map = score_tiles(FEATURES, picture)
        probability = tile_map["p_ai"]
        engine = "feature_tiles"

    decision = VERDICT.run(picture, len(raw)) if VERDICT.available else None

    return {
        "p_ai": round(probability, 4),
        "verdict": verdict_for(probability),
        "method": chosen,
        "method_label": METHODS[chosen],
        "auto_selected": method == "auto",
        "engine": engine,
        "resolution": f"{width}x{height}",
        "enough_evidence": longest >= EVIDENCE_FLOOR_PX,
        "tile_map": tile_map,
        "decision": decision,
    }


@app.get("/methods")
def methods():
    return {"methods": [{"id": k, "label": v} for k, v in METHODS.items()],
            "tile_reliable_px": TILE_RELIABLE_PX}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device),
            "cnns": list(CNNS), "features": list(FEATURES),
            "verdict_arms": [a.name for a in VERDICT.arms],
            "verdict_primary": VERDICT.primary.name if VERDICT.available else None}
