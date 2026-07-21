# PixelProof — AI Image Detector

Experimental image-classification project that decides whether a photo is **AI-generated** or a **real photograph**.

Current status: a small CNN baseline trained on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (100k train / 20k test, 32×32) reaches **96.75% accuracy / 0.995 ROC-AUC** on the held-out test set, and 77.1% on an external high-resolution out-of-distribution set — see [ROADMAP.md](ROADMAP.md) for full metrics, methodology, and the phased plan (classical-ML hybrids, transfer learning, patch-based high-res inference, anomaly detection).

## Layout

```text
app/                 Web UI
ml/configs/          Reproducible experiment configs
ml/src/pixelproof/   Data, model, training, evaluation, prediction code
ml/tests/            ML unit tests
ml/artifacts/        Trained weights (not committed)
```

The model and the UI are deliberately decoupled. New architectures (ResNet, EfficientNet, ViT) plug into `MODEL_REGISTRY` without changing the metrics pipeline or the UI contract. Label convention everywhere: `1 = AI-generated`, `0 = real`.

## ML quickstart

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install torch torchvision scikit-learn pillow pyyaml pytest

# train (data path lives in the config; datasets are not committed)
PYTHONPATH=src .venv/bin/python -m pixelproof.train --config configs/baseline.yaml

# evaluate on the held-out test set (+ optional external OOD set)
PYTHONPATH=src .venv/bin/python -m pixelproof.evaluate \
  --external-ai  /path/to/ai_images --external-real /path/to/real_images

# classify your own photos
PYTHONPATH=src .venv/bin/python -m pixelproof.predict photo.jpg

.venv/bin/python -m pytest   # sanity tests
```

## Web UI

```bash
npm install
npm run dev
```
