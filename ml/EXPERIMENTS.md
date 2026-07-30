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

## 2026-07-27 — E7: Modern-generator stress test + the downscaling penalty (Phase 4 follow-up)

- **Hypothesis (pre-registered):** the GenImage ResNet degrades on generators newer than its training set (2021–22 era), per the cross-generator failure mode documented in `IMAGE_FORENSICS_REFERENCE.md` §4.4.
- **New data:** Defactify / MS-COCOAI test split (HuggingFace `Rajarshi-Roy-research/Defactify_Image_Dataset`, 3 of 8 shards) → 16,875 images extracted at native quality (raw JPEG bytes, no re-encoding): 2,851 real MS-COCO photos + ~2,800 each from SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney v6. **Every generator is newer than anything in training.** Both classes are JPEG → no format shortcut.
- **Result (ResNet-18 / GenImage, `best_genimage.pt`):**

| generator | source px | AUC | AI recall |
|---|---|---|---|
| DALL-E 3 | 270 | 0.896 | 93.7% |
| Midjourney v6 | 436 | 0.821 | 86.5% |
| SDXL | 1024 | 0.717 | 75.1% |
| SD 2.1 | 768 | 0.696 | 71.1% |
| SD 3 | 1024 | 0.670 | 68.7% |
| **all** | | **0.760** | 79.0% |

- **Hypothesis confirmed** (0.888 on archive1 → 0.760 here), **plus an unanticipated finding**: AUC orders almost perfectly by *source resolution*, and in the direction opposite to a resolution shortcut — the smallest images are detected best. Mechanism: `eval_transform` resizes everything to 224×224, so a 1024² image is downscaled 4.6× while a 270² one is barely touched. Downscaling is a low-pass filter; generation artefacts are high-frequency. **We are removing the evidence before the model sees it.**
- **Control — native-resolution patches** (5 crops of 224px, mean-aggregated, 1,000 imgs/class): discrimination improved exactly where predicted (SD 3 0.672→0.776, SDXL 0.725→0.800, SD 2.1 0.705→0.791) but the false-positive rate on real photographs exploded from 43.8% to 95.6%, leaving overall AUC flat (0.764→0.776).
- **Conclusion:** the downscaling penalty is real and quantified. But patch inference cannot be bolted onto a model trained on downscaled crops — this is the **third** instance of the same law (E5, E6, here): *whatever the model will be shown at test time must be what it was shown during training.* `configs/genimage.yaml` sets `crop_augmentation: true` → `RandomResizedCrop(224, scale=(0.7,1.0))`, which downscales every 1024² training image by 3.8–4.6×. The model has never seen a native-resolution pixel.

## 2026-07-27 — E8: Resolution-independent hand-crafted features + classical ML

- **Hypothesis (pre-registered):** statistics computed over every pixel at native resolution are resolution-independent by construction, so the E7 resolution ordering should disappear.
- **Method:** `features.py` — 68 features per image, all ratios/per-pixel averages (never totals): per-channel moments, cross-channel correlations, **Bayer sub-lattice variance (CFA/demosaicing trace)**, noise-residual statistics, 16-band radial FFT spectrum, local-variance percentiles, 8×8 JPEG-grid blockiness, HSV statistics. Rationale per feature group in `IMAGE_STRUCTURE_NOTES.md`.
- **Design:** trained on the *identical* GenImage split used by the ResNet (9,917 images) → controlled method-vs-method comparison. Two input modes: `full` (whole image) and `crop128` (128×128 native centre crop, giving both classes identical dimensions). Three learning setups on the same features: supervised, one-class on real only, one-class on AI only.
- **Result (supervised HistGradientBoosting, AUC):**

| eval set | ResNet-18 | features `full` | features `crop128` |
|---|---|---|---|
| GenImage test (in-distribution) | **0.982** | — | 0.919 |
| archive1 | **0.888** | — | 0.505 |
| Defactify (5 unseen modern) | **0.760** | 0.717 | 0.717 |

- **Per generator on Defactify:**

| generator | src px | ResNet | feat `full` | feat `crop128` |
|---|---|---|---|---|
| DALL-E 3 | 270 | **0.896** | 0.808 | 0.377 |
| Midjourney v6 | 436 | **0.821** | 0.796 | 0.793 |
| SD 2.1 | 768 | 0.696 | 0.676 | **0.784** |
| SD 3 | 1024 | 0.670 | 0.620 | **0.760** |
| SDXL | 1024 | 0.717 | 0.685 | **0.867** |

- **Hypothesis confirmed, with a cost.** In `crop128` the E7 resolution ordering does not merely vanish — it inverts: the 1024px generators become the *easiest*. The method beats the CNN by +0.09 to +0.15 AUC on exactly the three generators the CNN handles worst, and collapses on DALL-E 3 (0.377, below chance — those images are 270×270 at ~16 KB, i.e. heavily compressed; compression noise appears to mask the generation trace).
- **One-class comparison (the "learn only one side" question):** on archive1, the most out-of-distribution set — one-class on **real** 0.688, supervised 0.505, one-class on **AI** 0.358. The asymmetry predicted from first principles (a new generator makes "what AI looks like" obsolete, while "what a photograph looks like" is fixed by sensor physics) is measurable. On in-distribution and Defactify data both one-class setups are far weaker than supervised (0.54–0.57 vs 0.72), so this is evidence about *robustness*, not about overall accuracy.
- **Caveat — shortcut probe:** a logistic model predicts the original image width from the 68 features with 92.6% accuracy *even in `crop128` mode*, where every input is 128×128. Resolution leaks through texture, not through dimensions. Some part of the numbers above is attributable to that leak; quantifying it is open.
- **Conclusion:** hand-crafted low-level statistics are a **specialist, not a replacement**. They are worse overall (0.717 vs 0.760) but better exactly where the CNN fails, which is the complementarity the literature's "RGB branch + low-level branch" architectures exploit.

## 2026-07-27 — E9: Ensemble of ResNet + feature model — negative result

- **Hypothesis (pre-registered):** since E8 showed the two methods fail in disjoint places, combining them should beat either alone.
- **Method:** eight combination rules over identical images (raw mean, weighted 75/25, max, min, and rank-normalised variants — ranks remove the probability-scale mismatch between a neural net and gradient boosting).
- **Result (AUC):**

| strategy | GenImage test | archive1 | Defactify | mean |
|---|---|---|---|---|
| ResNet alone | 0.982 | **0.888** | 0.760 | 0.876 |
| features alone | 0.919 | 0.505 | 0.717 | 0.714 |
| rank weighted 75/25 | **0.985** | 0.852 | 0.796 | **0.878** |
| max | 0.976 | 0.843 | **0.801** | 0.874 |
| mean 50/50 | 0.979 | 0.779 | 0.782 | 0.847 |

- **Hypothesis NOT supported.** The best rule beats ResNet alone by +0.002 on average — noise. Per test set it *relocates* accuracy rather than adding it: Defactify +0.036, archive1 −0.036. Per generator on Defactify the ensemble helps on 4 of 5 (SDXL 0.717→0.848, SD 3 0.670→0.762, SD 2.1 0.696→0.791, Midjourney 0.821→0.856) and destroys the fifth (DALL-E 3 0.896→0.685).
- **Conclusion:** a *fixed-weight* blend cannot exploit a specialist. The feature model is near-random on archive1 (0.505), and averaging a random signal into a good one costs as much as the gains elsewhere. A conditional combination would need a reliable "when is the feature model trustworthy?" signal, which we do not have. **Decision:** the demo reports both scores side by side and flags disagreement, rather than averaging them.

## 2026-07-27 — E10: archive1 audit and confound controls

- **Motivation:** archive1 has been the OOD benchmark since E1 but was never inspected. E8's anomalous scores there (features 0.505, logistic regression 0.217 — systematically *inverted*) prompted an audit.
- **Audit result — archive1 is maximally confounded:**

| | real (745) | AI (250) |
|---|---|---|
| format | 100% JPEG | 100% PNG |
| distinct sizes | 138 | 2 |
| square | 2% | 100% (512×512) |
| EXIF | none | none |

  Every real image is 1080px wide with EXIF stripped — i.e. social-media processed ("laundered"), not camera-original. **A logistic model on width/height/aspect alone separates the classes at AUC 1.000.** This is the exact format shortcut warned about in `IMAGE_FORENSICS_REFERENCE.md` §5.
- **Controls (one variable changed at a time, 995 images):**

| condition | ResNet AUC | SmallCNN AUC |
|---|---|---|
| A — as downloaded (real JPEG rect / AI PNG square) | 0.888 | 0.800 |
| B — AI re-encoded to JPEG q90 | 0.918 | 0.797 |
| C — both classes re-encoded to JPEG q90 | 0.912 | 0.798 |
| D — C plus both centre-cropped to square at native resolution | **0.896** | **0.808** |

- **Conclusion — the confound is present but unusable by these models.** Removing both shortcuts moved AUC by **+0.008** for both networks (upward, not downward). The reason is mechanical: `PIL` decoding discards the container format and `Resize((N,N))` discards dimensions and aspect ratio, so neither network can see the leak. **E1's 77.1% and E6's 0.888 stand as genuine detection performance.**
- **Important scope limit:** this immunity is a property of the *resize* pipeline, not of the data. The E8 feature model reads native pixels and its shortcut probe scores 92.6%, so archive1's bias **is** exploitable there. Any future native-resolution method must control for it explicitly.
- **General lesson worth carrying:** a biased dataset is only dangerous if the model can perceive the bias. The aggressive downscaling criticised in E7 for destroying signal also, accidentally, destroyed the shortcut.

## 2026-07-29 — E11: Tile-based inference — grid size and aggregation rule

- **Origin.** E8 gave the feature model two input modes: whole-image and a single 128×128 native centre crop. Testing a ChatGPT-generated image (1122×1402) in the demo exposed the flaw in the second: the CNN said 48% (wrong), whole-image said 94% (correct), single-crop said 47% (wrong). The reason was measurable — a 128×128 centre crop of that image is **1.04% of its pixels**, and the centre happened to be the subject's plain navy t-shirt: grey-level std 0.027 against 0.283 for the full image, **10.6× flatter**. The model was handed a featureless patch of fabric and correctly answered "no idea". The idea was sound; the sampling was blind.
- **Hypothesis (pre-registered):** covering the image with a grid of native-resolution tiles instead of sampling one arbitrary window will improve discrimination, because coverage rises from ~1% to ~100% while the fixed-size property that blocks the resolution shortcut is preserved.
- **No retraining.** `feature_crop128` was fitted on 128×128 native crops and every tile *is* one — the same input distribution, evaluated several times per image. This is the one place in the project where the preprocessing law (E5/E6/E7) does **not** demand a retrain. Mild caveat: training used centre crops while tiles include edges, so content statistics differ slightly.
- **Config:** `features.extract_tiles()` — grid centred on the image, thinned evenly when the tile count exceeds the cap. Evaluated on Defactify's high-resolution generators (SD 2.1, SD 3, SDXL, Midjourney), 800 images per class.
- **Result — grid size** (best rule per grid, mean AUC over the four generators):

| grid | tiles/image | mean | top3 | top half | median | max |
|---|---|---|---|---|---|---|
| 2×2 | 4.0 | 0.742 | 0.754 | **0.760** | 0.750 | 0.747 |
| 3×3 | 8.5 | 0.775 | **0.799** | 0.797 | 0.790 | 0.774 |
| 4×4 | 11.2 | 0.777 | **0.801** | 0.792 | 0.791 | 0.779 |
| 5×5 | 13.8 | 0.781 | **0.807** | 0.802 | 0.794 | 0.788 |
| **6×6** | 13.8 | 0.781 | **0.821** | 0.807 | 0.795 | 0.802 |

- **Result — per generator at 6×6 + top-3:**

| generator | source px | tiled | CNN (E7) | Δ |
|---|---|---|---|---|
| SDXL | 1024 | **0.948** | 0.717 | **+0.231** |
| SD 3 | 1024 | **0.894** | 0.670 | **+0.224** |
| SD 2.1 | 768 | **0.863** | 0.696 | **+0.167** |
| Midjourney v6 | 436 | 0.580 | 0.821 | −0.241 |

- **Hypothesis confirmed. 0.948 is the highest AUC this project has produced**, above E6's 0.888 headline, and it is achieved on a generator the model never trained on.
- **Aggregation matters as much as the grid.** Top-3 mean beats a plain mean at every grid size (0.821 vs 0.781 at 6×6). Mechanism: flat tiles score ≈0.5 and an ordinary average lets them drown the tiles carrying evidence — the t-shirt problem at scale. Measured: **21.4% of all tiles** fall below the texture floor (grey std < 0.04). Explicitly dropping low-texture tiles was also tested and gave no advantage over top-3, so the simpler rule was kept.
- **Crossover measured:** above ~700px the tile method beats the CNN decisively; below it the CNN wins (Midjourney at 436px). This replaces the invented `128px` routing constant in `serve.py` with `TILE_RELIABLE_PX = 700`.
- **Conclusion:** the scale problem that dominated E5, E6 and E7 is dissolved rather than mitigated — the model always sees 128×128 native pixels, and resolution changes only *how many tiles* come out, never what a tile looks like. Two consequences beyond Module 1: image dimensions can no longer act as a shortcut (so the datasets flagged in ROADMAP §1c become usable), and the per-tile scores are directly a localisation map, which is Module 2's core machinery obtained as a side effect.
- **Known limit:** low-resolution and heavily-compressed sources get worse, not better (DALL-E 3 at 270px, ~16 KB, drops below chance). Those inputs remain the CNNs' domain. The suspected mechanism — the feature model reading compression level as a proxy — is untested and listed as open.
