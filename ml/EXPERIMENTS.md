# Experiment Log

One entry per experiment: date, config, seed, metrics, conclusion. Rule from
the methodology section of the roadmap: a single seed is a number, not
evidence — key comparisons get ≥3 seeds.

---

## 2026-07-20 — E1: Baseline CNN (Phase 1)

- **Config:** `configs/baseline.yaml` (SmallCNN, 20 epochs, AdamW lr=1e-3, wd=1e-4, batch 128, hflip augmentation, seed 42), Apple MPS.
- **Data:** CIFAKE 90k train / 10k validation; best-val-accuracy checkpoint kept.
- **Result:** best val acc **96.8%** (epoch 19); held-out test: **acc 96.75, F1 0.967, ROC-AUC 0.995**; external OOD (995 high-res): **acc 77.1, F1 0.570, ROC-AUC 0.800**.
- **Conclusion:** no classic overfitting (train/val gap 1.2 pts, val loss still falling). Large distribution-shift gap on OOD data → resolution/generator mismatch is the main bottleneck, not model capacity.

## 2026-07-21 — E2: Classical classifiers on frozen embeddings (Phase 2a)

- **Config:** 128-dim penultimate embeddings from E1 checkpoint; LogReg / LinearSVC / RF(200) / HistGB trained on the same seeded 90k split (seed 42); single seed — differences below are within noise, treat as ties.
- **Result (test, 20k):**

| model | accuracy | f1 | roc_auc |
|---|---|---|---|
| cnn_head (reference) | 0.9675 | 0.9673 | 0.9953 |
| logistic_regression | 0.9685 | 0.9685 | 0.9956 |
| linear_svm | 0.9690 | 0.9690 | 0.9957 |
| random_forest | 0.9684 | 0.9684 | 0.9939 |
| hist_gradient_boosting | 0.9685 | 0.9685 | 0.9954 |

- **Result (external OOD, 995):** all models ~0.77 accuracy; classical AUCs 0.81–0.82 slightly above CNN head's 0.80.
- **Conclusion:** every classifier lands within ~0.2 pts of the CNN head → **the representation, not the classifier, is the bottleneck**. Even plain logistic regression on good embeddings matches the deep head. Improving OOD performance requires better representations (Phase 3+), not a fancier classifier.

## 2026-07-21 — E3: Embedding clustering & error analysis (Phase 2b)

- **Config:** k-means (k=2, k-means++, n_init=10, seed 42) + t-SNE (5k sample) on E1 embeddings.
- **Result (test):** cluster purity **0.965**, ARI **0.866** — unlabeled k-means almost rediscovers the real/AI split, i.e. the embedding space is nearly linearly separable. Errors: 3.26% overall, but **45%** among borderline predictions (|p−0.5|<0.1, 240 images) vs 2.75% elsewhere; t-SNE shows errors concentrated on the boundary between the two clusters.
- **Result (external):** purity collapses to **0.749**, ARI **0.013** — in embedding space the OOD real/AI images are no longer separated; cluster 1 mixes 383 real with 210 AI (31.7% error rate).
- **Figures:** `artifacts/figures/tsne_test.png`, `artifacts/figures/tsne_external.png`.
- **Conclusion:** confirms E2. On in-distribution data the learned space is cleanly structured; on OOD data the structure disappears entirely → the model has learned CIFAKE-specific features. The borderline-probability band is a natural candidate for a "not sure" rejection threshold in the product.

## 2026-07-21 — E4: Learning curve / data-size ablation (Phase 2c)

- **Hypothesis (pre-registered):** accuracy grows roughly logarithmically with data; train/val gap widens at small sizes.
- **Config:** identical to E1 (same seed, arch, hyperparameters, 20 epochs); only training-set size varies. Validation (10k) and test (20k) fixed across runs. Each run trained **from scratch**.
- **Result:**

| train size | test acc | best val acc | final train−val gap |
|---|---|---|---|
| 10k | 93.83% | 93.99% | 5.1 pts |
| 20k | 94.81% | 94.64% | 3.8 pts |
| 50k | 95.98% | 95.89% | 2.9 pts |
| 90k (E1) | 96.75% | 96.83% | 1.2 pts |

- **Figure:** `artifacts/figures/learning_curve.png`.
- **Conclusion:** hypothesis confirmed on both counts. Accuracy is almost perfectly linear in log(data) — each ~doubling of data buys ≈1 point — and the overfitting gap shrinks monotonically with data (5.1 → 1.2 pts). Notably the curve has **not saturated at 90k**: more CIFAKE-like data would still help in-distribution. But per E2/E3 the binding constraint for real-world performance is representation quality under distribution shift, so Phase 3 (transfer learning) remains the priority over collecting more 32×32 data.

## 2026-07-21 — E5: ResNet-18 transfer learning (Phase 3)

- **Hypothesis (pre-registered):** a stronger pretrained backbone improves representation quality → small in-distribution gain, significant OOD improvement over the 77.1% baseline.
- **Config:** `configs/resnet18.yaml` — ImageNet-pretrained ResNet-18, new 512→1 head, full fine-tune, 3 epochs, AdamW lr=1e-4, batch 64, 224×224 inputs (32×32 CIFAKE upscaled), ImageNet normalization, seed 42.
- **Result:**

| metric | SmallCNN (E1) | ResNet-18 (E5) |
|---|---|---|
| val acc | 96.8% | **97.6%** (epoch 1 already 97.0%) |
| test acc | 96.75% | **97.66%** (ROC-AUC 0.9975) |
| external OOD acc | 77.1% | **25.2%** (ROC-AUC 0.523 ≈ random; predicts "AI" for 984/995 images) |

- **Hypothesis FALSIFIED on OOD** — the interesting half. Control experiment: re-evaluating the external set with a 32px bottleneck (Resize 32 → Resize 224, mimicking the training distribution) recovers accuracy 25.2% → **72.0%** and AUC 0.523 → **0.802**.
- **Conclusion:** the collapse is a **preprocessing domain shift**, not lost capability. Training images are blurry 32→224 upscales; native high-resolution photos are sharp and land far outside the training distribution, biasing the model toward "AI". Transfer learning improved in-distribution accuracy (+0.9 pts) but model capacity cannot compensate for a train/inference preprocessing mismatch. Strongest evidence yet for Phase 4: high-resolution detection requires natively high-resolution training data (e.g. GenImage) — upscaled CIFAKE actively hurts.
- **Practical note:** until Phase 4, high-res inputs to the ResNet model must go through the 32px bottleneck at inference; the SmallCNN baseline remains the safer OOD choice.

## 2026-07-21 — E6: ResNet-18 on native-resolution GenImage (Phase 4)

- **Hypothesis (pre-registered):** training on natively high-resolution data removes the E5 preprocessing domain shift → archive1 OOD performance beats the 77.1% SmallCNN baseline.
- **Data:** unbiased-tiny-GenImage (Kaggle `cartografia/unbiased-tiny-genimage`) — REAL = 5,828 ImageNet nature photos (~500px), FAKE = balanced 833/generator across 7 generators (Midjourney, SDv1.5, ADM, BigGAN, GLIDE, VQDM, Wukong; 256–1024px). Stratified seeded split: 9,917 train / 1,742 test; "unbiased" variant avoids the JPEG-vs-PNG format shortcut.
- **Config:** `configs/genimage.yaml` — same ResNet-18 recipe as E5 but native-res data + RandomResizedCrop; 5 epochs; best val 93.2% (epoch 3).
- **Result:**

| eval | accuracy | ROC-AUC | note |
|---|---|---|---|
| GenImage test | 92.14% | 0.982 | harder task than CIFAKE (7 generators) |
| **archive1 OOD** | 69.4% @0.5 | **0.888** | **best AUC of any model** (SmallCNN 0.800, E5 0.523); AI recall 89.6% but precision 0.445 — biased toward "AI" |
| CIFAKE test | 50.0% | 0.634 | near-total loss of the low-res domain (catastrophic forgetting / reverse domain shift) |

- **Threshold analysis:** archive1 accuracy at default 0.5 threshold is 69.4%; oracle threshold reaches 85.2% (diagnostic only — uses external labels). Calibrating the threshold on GenImage validation (0.15) transfers badly (57.1%) → **calibration itself is domain-sensitive**; the model's ranking information (AUC) transfers much better than its probability scale.
- **Conclusion:** hypothesis **confirmed at the information level**: native high-res training gives the strongest OOD representation so far (AUC 0.80 → 0.89), and the E5 collapse is fixed. Remaining problem is **calibration under distribution shift**, not discrimination. Also a clean symmetry: each model is blind outside its resolution domain (SmallCNN fails on sharp inputs' fine detail; GenImage model fails on 32×32 CIFAKE). Practical direction: resolution-routed ensemble (small input → SmallCNN, large input → GenImage model) + uncertainty band, and/or calibration fixes (temperature scaling) in the web demo phase.
