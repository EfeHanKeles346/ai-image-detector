# AI Image Detector — Project Roadmap

Goal: build a model that decides whether a photo is **AI-generated** or a **real photograph**, starting with a small CNN and iterating toward stronger models, then serving it through the web app in this repo.

---

## 1. The Data

| Dataset | Location | Contents | Role |
|---|---|---|---|
| CIFAKE (`archive`) | `~/Desktop/archive` | 100k train + 20k test images, 32×32×3, balanced `REAL`/`FAKE` folders. Real half comes from CIFAR-10, fake half from Stable Diffusion. | Main training + held-out test set |
| External set (`archive1`) | `~/Desktop/archive1` | High-resolution images in `Ai_generated_dataset/` and `real_dataset/`, organized by category (animals, city, food, nature, people). | **Out-of-distribution (OOD) evaluation only** — never used for training |

Why keep `archive1` out of training? A model can score very well on data that looks like its training set while failing on anything else. Evaluating on a dataset from a completely different source answers the real question: *does the model generalize, or did it just memorize CIFAKE's statistics?*

**Label convention (important):** everywhere in the code, `1 = AI-generated`, `0 = real`. `torchvision.datasets.ImageFolder` sorts folders alphabetically (`FAKE`=0, `REAL`=1), so we invert its labels (`invert_label` in `data.py`) to keep the convention consistent.

## 2. Project Layout (`ml/`)

```
ml/
├── configs/baseline.yaml        # all hyperparameters in one place — no magic numbers in code
├── src/pixelproof/
│   ├── data.py                  # datasets, transforms, train/validation split
│   ├── models.py                # model definitions + a registry to add new ones
│   ├── train.py                 # training loop, checkpoints best model to artifacts/best.pt
│   ├── evaluate.py              # test-set metrics + external (OOD) evaluation
│   └── predict.py               # classify arbitrary image files from the CLI
├── tests/                       # fast sanity tests (model output shape, etc.)
└── .venv/                       # Python virtualenv (PyTorch with Apple MPS GPU support)
```

Design principles worth remembering for any ML project:
- **Config-driven experiments.** Changing the learning rate or epochs means editing YAML, not code. Each checkpoint stores the config it was trained with, so results are reproducible.
- **Fixed random seed** for the train/validation split and weight init — reruns are comparable.
- **Model registry** (`MODEL_REGISTRY` dict): adding a new architecture later is one entry, and the config's `model.name` selects it.

## 3. Phase 1 — Baseline CNN ✅ (current)

### Architecture (`SmallCNN`)
Three convolutional blocks (Conv → BatchNorm → ReLU, ×2 per block) with max-pooling in between, then **global average pooling** and a single-logit linear head. ~300k parameters — small, fast, and hard to overfit on 100k images.

- **Why one output logit instead of two classes?** Binary classification only needs one number; `sigmoid(logit)` is the probability of "AI". Trained with `BCEWithLogitsLoss` (numerically stabler than sigmoid + BCE separately).
- **Why BatchNorm?** Stabilizes training and lets us use a higher learning rate.
- **Why global average pooling?** No giant fully-connected layer → far fewer parameters, and the network technically accepts any input size.

### Training setup
- Split: 90k train / 10k validation (10%), stratified by the seed-fixed shuffle.
- Augmentation: **horizontal flip only.** We deliberately avoid color jitter / blur / JPEG-style augmentations, because subtle color statistics and generation artifacts are exactly the signal that separates AI images from real ones — destroying them would hurt the model.
- Validation uses a deterministic transform (no augmentation) — you evaluate the model, not the noise.
- Optimizer: AdamW, lr 1e-3, weight decay 1e-4, batch 128, 20 epochs, on Apple MPS (GPU).
- The epoch with the best validation accuracy is saved to `artifacts/best.pt` (early-stopping-lite).

### Evaluation
Accuracy alone is not enough. `evaluate.py` reports:
- **Precision / Recall / F1** — is the model biased toward one class? Recall on the AI class = "what fraction of AI images do we catch?"
- **ROC-AUC** — quality of the probability ranking, independent of the 0.5 threshold.
- **Confusion matrix** — where exactly the mistakes are.
- The same metrics on `archive1` → the generalization check.

### Commands
```bash
cd ml
PYTHONPATH=src .venv/bin/python -m pixelproof.train --config configs/baseline.yaml
PYTHONPATH=src .venv/bin/python -m pixelproof.evaluate \
  --external-ai  ~/Desktop/archive1/Ai_generated_dataset \
  --external-real ~/Desktop/archive1/real_dataset
PYTHONPATH=src .venv/bin/python -m pixelproof.predict some_photo.jpg
.venv/bin/python -m pytest        # sanity tests
```

## 4. Known Limitation: Input Resolution

The model was trained on 32×32 images. `predict.py` resizes any input (even 1980×1980) down to 32×32 first, so nothing crashes — but downscaling destroys the fine textures and generation artifacts that betray modern AI images. Expect degraded accuracy on high-resolution, in-the-wild photos. Phases 2–3 address this directly.

## 5. Experimental Methodology — why the phases are ordered this way

The phase ordering below follows the consensus workflow from three authoritative sources: [Karpathy's "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/), [Google Research's Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), and the ablation-study literature (e.g. [ABLATOR, PMLR 2023](https://proceedings.mlr.press/v224/fostiropoulos23a/fostiropoulos23a.pdf)).

1. **Data understanding first, before any model code.** (done — Section 1)
2. **Full pipeline + simple baseline before anything fancy.** Get train/eval/metrics working end-to-end with a deliberately simple model. Studies show simple baselines are competitive with complex architectures in ~40% of published benchmarks — you must know what "simple" achieves before paying for complexity. (done — Phase 1)
3. **Insight before complexity.** The Tuning Playbook: spend most time on *exploration* (understanding why the model behaves as it does), not *exploitation* (chasing score). That is exactly Phase 2 — error analysis, embedding analysis, and data ablations are cheap and tell us *where* the next gains are, so the expensive phases aim at the right target.
4. **Scale complexity only with evidence.** Bigger architectures (transfer learning, high-res) come after the cheap experiments justify them — "incrementally add complexity while building up strong evidence." (Phases 3–4)
5. **Rigor rules adopted for all experiments from now on:**
   - One controlled config per experiment; change one thing at a time.
   - A single seed produces "a number, not evidence" — key comparisons get ≥3 seeds, we report mean ± std.
   - Keep a written experiment log (`ml/EXPERIMENTS.md`): date, config, seed, metrics, conclusion.

## 6. Phase 2 — Hybrid ML Experiments (next)

A core learning goal of this internship project: combine deep learning with classical ML instead of treating them as rivals. All three experiments reuse the trained CNN, so none of them require expensive retraining.

**A note on terminology first.** This project is **supervised learning** (we have REAL/FAKE labels and train a classifier on them), *not* unsupervised learning. Clustering algorithms like k-means only enter the picture below as *analysis tools* on top of the supervised model — except for Phase 5, which is a genuinely unsupervised formulation of the problem.

### 2a. CNN as a feature extractor + classical classifiers
- Take the 128-dim embedding from the CNN's penultimate layer for every image.
- Train classical models on those embeddings: Logistic Regression, SVM, Random Forest, Gradient Boosting.
- Compare all of them against the CNN's own classification head on the same test sets.
- What this teaches: deep nets as representation learners; strengths/weaknesses of each classical algorithm; a clean comparison table for the report.

### 2b. Embedding analysis with clustering & projection
- Run k-means (k-means++ init) on the embeddings; project to 2D with t-SNE/UMAP, color by true label and by cluster.
- Questions to answer: do real/AI images separate cleanly? Where do the misclassified images live? Do clusters align with semantic categories (animals, city, food…)?
- What this teaches: what the network actually learned, communicated visually — and the correct role of clustering: *exploration and error analysis*, not classification.

### 2c. Learning-curve experiment (data-size ablation)
- Retrain the same CNN on subsets (e.g. 10k / 20k / 50k / 90k) and plot accuracy vs. training-set size.
- Answers empirically: "how much does more data matter?" Expected: logarithmic gains and a widening train/val gap at small sizes.

## 7. Phase 3 — Transfer Learning + Ensemble

- Fine-tune a pretrained backbone (ResNet-18 → EfficientNet-B0) at 224×224 input.
- Compare against the baseline on the *same* test sets — this is why the metrics pipeline came first.
- Ensemble idea: average/vote the CNN (pixel domain) with a gradient-boosting model trained on frequency-domain features (FFT/DCT) — diffusion models leave periodic fingerprints in the frequency spectrum that pixel-space models can miss.
- Concepts to learn here: freezing vs. full fine-tuning, discriminative learning rates, pretrained normalization statistics, why diverse ensembles beat their members.

## 8. Phase 4 — High-Resolution / Patch-Based Inference

- Instead of downscaling a large image, crop several patches at native resolution, classify each, and aggregate (mean or max probability).
- Needs a higher-resolution training dataset (e.g. GenImage or a scraped SD/Midjourney set) — CIFAKE alone can't teach high-res artifacts.

## 9. Phase 5 — Unsupervised Track: Anomaly Detection (parked — do not skip)

Train a model **only on real photographs** and flag anything that deviates as suspicious (one-class SVM on embeddings, or an autoencoder with reconstruction error).

**Why this matters.** Every supervised detector has a built-in blind spot: it learns the artifacts of the generators it was trained against. When a new generator ships (Midjourney v7, Flux, whatever comes next), those artifacts change and supervised accuracy silently collapses — our own CIFAKE→`archive1` drop (96.75% → 77%) is a small-scale preview of exactly this failure mode. An anomaly detector inverts the question: instead of "what does AI look like?" it learns "what do real photos look like?" — and real photos don't change when a new generator is released. This is the closest thing the field has to future-proofing, which is why it deserves a dedicated phase even though it comes last.

## 10. Phase 6 — Serving in the Web App

- Export the trained model (TorchScript or ONNX).
- Add an inference endpoint (Python microservice, or ONNX Runtime in the Node worker).
- Web UI: upload a photo → probability gauge "AI-generated vs real".
- Report calibrated confidence, not just a hard yes/no.

## 11. Progress Checklist

- [x] Datasets inspected (CIFAKE 100k/20k + external OOD set)
- [x] Python env with PyTorch + MPS
- [x] Baseline CNN, config-driven training pipeline
- [x] Evaluation script (test metrics + OOD) and CLI predictor
- [x] Baseline training run finished (20 epochs, MPS): best val accuracy **96.8%**
  - Held-out test (20k): **accuracy 96.75%, F1 0.967, ROC-AUC 0.995**
  - External OOD set (995 high-res images): **accuracy 77.1%, F1 0.570, ROC-AUC 0.800** — the expected resolution-driven generalization gap; motivates Phases 2–3
- [x] Phase 2a: embeddings + classical classifiers — all four match the CNN head (±0.2 pts) → representation is the bottleneck, not the classifier (see `ml/EXPERIMENTS.md` E2)
- [x] Phase 2b: k-means + t-SNE analysis — test embeddings nearly linearly separable (purity 0.965); structure collapses on OOD (purity 0.749, ARI 0.013); errors concentrate in the borderline band |p−0.5|<0.1 (see E3)
- [ ] Phase 2c: learning-curve experiment (10k → 90k)
- [ ] Phase 3: transfer-learning comparison (ResNet-18 / EfficientNet) + frequency-domain ensemble
- [ ] Phase 4: high-res strategy (patches / new dataset)
- [ ] Phase 5: unsupervised anomaly-detection track (train on real only)
- [ ] Phase 6: web integration + demo
