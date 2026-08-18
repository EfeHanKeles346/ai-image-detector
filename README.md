# PixelProof — AI Image Detector

Research project that decides whether a photo is **AI-generated** or a **real photograph**, and —
work in progress — **where** an image was manipulated.

## Where the project stands (2026-08-18)

**The strongest model is a ResNet-18 fine-tuned on 128px native tiles** (E20): Defactify AUC
0.770 and 61.4% AI recall on the untouched evaluation half — the project's best numbers. It is
deliberately **not** exposed as an API verdict, because a threshold fitted for a 10% false-positive
budget reaches **96% false positives on the worst unseen camera source** (E20-v2). Ranking is
solved; deciding is not. The plan for that lives in [`PLAN.md`](PLAN.md).

The demo still serves the four earlier detectors, none of them blended, because blending was
measured and did not help:

| Method | What it does | Best at | Defactify AUC |
|---|---|---|---|
| SmallCNN | 32×32 CIFAKE baseline | tiny inputs | — |
| ResNet-18 | fine-tuned on native-resolution GenImage | <700px, compressed | 0.760 |
| Statistics | 68 hand-crafted features over every pixel, no resizing | mid-size, low-texture | 0.717 |
| **Tiles** | 6×6 grid of native 128px crops, mean of top 3 | **≥700px** | **0.948 on SDXL** |

> ⚠️ **Read `HISTORY.md` §12b before trusting any of these numbers.** Measuring the tile model's
> *operating point* rather than its AUC showed it calls **79% of real photographs "AI"** — and the
> cause is a narrow real class, not calibration. Every detector here was partly learning "what my
> training set's real photographs look like" instead of "what a generated image looks like".
> Widening the real class raises AUC 0.55 → 0.884 at no cost to AI recall.

Measured on **Defactify** — 16,875 images from five generators (SD 2.1, SDXL, SD 3, DALL-E 3,
Midjourney v6), none of which any model was trained on.

**The finding that drove all of it:** the pipeline was resizing every image to 224×224 before the
model saw it, and generation artefacts live in exactly the high frequencies a downscale removes.
Detection quality lined up almost perfectly with source resolution — 270px scored 0.896, 1024px
scored 0.670. The tile method dissolves the problem: the model always sees 128×128 native pixels,
so resolution changes only *how many tiles* come out, never what a tile looks like.

Full reasoning: [`HISTORY.md`](HISTORY.md) §4 and §9. Experiment log: [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md).

## Documentation map

| File | Read it when |
|---|---|
| [`PLAN.md`](PLAN.md) | You want to know what happens **next** — the only living planning document |
| [`HISTORY.md`](HISTORY.md) | You want the full decision history (frozen) — the internship report's reference |
| [`DATASETS.md`](DATASETS.md) | You need to know which data you may train on, and in which mode |
| [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) | You want one experiment's hypothesis, config, numbers and conclusion |
| [`IMAGE_FORENSICS_REFERENCE.md`](IMAGE_FORENSICS_REFERENCE.md) | You need the field background — how images are generated, edited, and detected |
| [`IMAGE_STRUCTURE_NOTES.md`](IMAGE_STRUCTURE_NOTES.md) | You need the technical basis for a feature-design decision |

## Layout

```text
app/                       Web UI (Next.js)
ml/configs/                Reproducible experiment configs
ml/src/pixelproof/         The live spine: data, models, training, features, protocol, serving
ml/src/pixelproof/archive/ Retired modules, frozen (E2–E4 analysis, ELA, DINOv2 extraction)
ml/experiments/            Runnable protocol scripts (e20, e21)
ml/experiments/archive/    Finished evidence scripts (E7–E18), frozen
ml/tools/                  Dataset acquisition + auditing
ml/artifacts/              Trained weights (not committed)
ml/artifacts/archive/      Superseded artifacts — moved aside, never deleted
```

Label convention everywhere: `1 = AI-generated`, `0 = real`.

## Quickstart

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install torch torchvision scikit-learn pillow pyyaml pytest scipy joblib

# train a CNN (data paths live in the config; datasets are not committed)
PYTHONPATH=src .venv/bin/python -m pixelproof.train --config configs/genimage.yaml

# fit and save the two feature models
PYTHONPATH=src .venv/bin/python -m pixelproof.feature_model

# evaluate a CNN on the held-out test set + an external set
PYTHONPATH=src .venv/bin/python -m pixelproof.evaluate \
  --external-ai /path/to/ai --external-real /path/to/real

# reproduce any experiment
PYTHONPATH=src .venv/bin/python experiments/e11_tile_grid_search.py

# audit a dataset BEFORE using it (see HISTORY §1b — this rule was learned the hard way)
.venv/bin/python tools/audit_datasets.py /path/to/dataset

.venv/bin/python -m pytest
```

## Demo

Two processes:

```bash
cd ml && PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799
npm install && npm run dev     # from the repo root, UI on :3000
```

Upload an image, pick a method (or leave it on **Otomatik**, which applies the measured 700px
crossover). With the tile method the per-tile scores are drawn over the image as a heat-map —
uniform red means the whole image is synthetic, a single red region means that region is suspect.
That heat-map is also the groundwork for Module 2.

## Two modules

- **Module 1 — is this AI-generated?** The four detectors above.
- **Module 2 — where was it manipulated?** Not built yet, but no longer hypothetical: the tile
  scorer already produces a per-region map, and a 78 GB compilation of 13 forensic datasets with
  **pixel-level ground-truth masks** is on disk, so localisation can be scored rather than argued
  about. See `HISTORY.md` §12.
