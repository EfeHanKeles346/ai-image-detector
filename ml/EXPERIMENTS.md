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

## 2026-07-30 — E12: statistics model, 10x more training data

- **Hypothesis (pre-registered):** training the 68-feature model on ~101k images instead of 9,917 improves detection, because E2 established that the representation — not the classifier — is the bottleneck, and more data should sharpen the representation's fit.
- **Data:** a merged pool built from five sources (CommunityForensics 301 generators, AI-vs-Real-balanced, GenImage, AIGC-Benchmark, ai-vs-real-200k). Indexed by `build_pool.py`, features cached by `pool_features.py`. Two checks ran before any training:
  - **contamination:** 38 pool images were perceptual-hash matches for Defactify test images (all from AIGC-Benchmark, which shares MS-COCO with Defactify's real half). Excluded from the index. Without this check we would have trained on our own test set.
  - **merged-pool audit:** individually clean sources combined into a pool with a **3.4x resolution gap** between classes (real median 1024px, AI 300px). Rebalanced across six resolution bands to 1.08x, costing 40% of the data. Both models were then trained on this same balanced pool, so the v1/v2 comparison stays controlled.
- **Result (mean over 3 seeds):**

| eval set | v1 (9.9k) | v2 (101k) | delta |
|---|---|---|---|
| GenImage test | 0.974 | 0.913 | −0.061 |
| archive1 | 0.706 | **0.839** | **+0.133** |
| Defactify (all) | 0.717 | 0.692 | −0.025 |
| pool held-out | — | 0.895 | — |

| generator | src px | v1 | v2 | delta |
|---|---|---|---|---|
| DALL-E 3 | 270 | **0.808** | 0.620 | **−0.189** |
| Midjourney v6 | 436 | 0.796 | 0.818 | +0.022 |
| SD 2.1 | 768 | 0.676 | 0.661 | −0.015 |
| SD 3 | 1024 | 0.620 | 0.662 | +0.043 |
| SDXL | 1024 | 0.685 | 0.699 | +0.014 |

- **Hypothesis NOT supported.** Ten times the data produced a large gain on archive1, a large loss on DALL-E 3, and a wash elsewhere. Note the GenImage comparison is unfair to v2: v1 trained exclusively on GenImage, so 0.974 is an in-distribution number while v2's 0.913 is nearly out-of-distribution (GenImage is 7.8% of v2's pool).
- **Two hypotheses tested for the DALL-E 3 collapse:**
  1. *Low-resolution contamination.* 26% of the pool was 32x32 (`ai_vs_real_balanced`), where the 68 statistics have almost nothing to measure. **Refuted** — v3, trained with a 256px floor on 74,139 images, scored 0.607 on DALL-E 3 against v2's 0.611. No change.
  2. *Compression domain gap.* **Supported.** The pool sits at ~0.9 bytes/pixel (largely PNG and lightly-compressed JPEG); Defactify sits at ~0.12 — a **7x gap**. The extra data came from a different compression regime than the test set, which is the preprocessing law (E5/E6/E7) appearing in a fourth dimension. Compression is not a class cue *within* the pool (real 0.861 vs AI 0.922), so this is a train/test gap rather than a shortcut.
- **Conclusion:** data volume alone does not help when the added data occupies a different domain. Compression augmentation — proposed before training and skipped — is the indicated fix and remains untested.

### Correction, 2026-08-04 — the balanced pool had no producer script

The script that turned `pool_index.csv` into `pool_balanced.csv` was never committed: five files read that CSV and none wrote it, so **E12–E16 were not reproducible from the repo**. `make_balanced_pool.py` now reconstructs it. Three things were found while doing so, and two of them correct the entry above.

- **Four bands, not six.** Candidate band grids were fitted against the surviving CSV. The rule is `min(n_real, n_ai)` per band over longest-side cut points **`[0,128) [128,256) [256,1024) [1024,∞)`** — this reproduces the artifact exactly, band for band, at 51,246 rows per class and 102,492 in total (39.6% of the index dropped, matching the "40%" above). No six-band grid reproduces it.
- **The residual gap was 1.68×, not 1.08×.** The real class matches the original exactly (median 431px) but the AI class does not (256px here against the 400px reported). The original therefore sampled *non-uniformly inside* a band, and that rule is not recoverable from counts alone. The 1.08× figure should not be quoted.
- **Band granularity saturates, and the knee is measurable.** Balancing is only as tight as the bands are narrow, because composition inside a band is unconstrained:

| cut points | rows | class gap | cost |
|---|---|---|---|
| `128,256,1024` (the original) | 102,492 | 1.68× | 39.6% |
| `128,256,512,768,1024` | 100,104 | 1.56× | 41.0% |
| **`128,256,384,512,768,1024,1536`** | 74,162 | **1.00×** | 56.3% |
| ten cut points | 74,104 | 1.00× | 56.3% |

  Adding 384 and 1536 closes the gap completely, and refining further buys nothing — the pool's resolutions are piled on a few discrete values (32, 256, 500, 512, 1024), so once those are separated there is nothing left to split. **Any future pool should be built with the seven-cut grid**; the four-band default is kept only so the E12–E16 counts stay reproducible.
- **Caveat on the metric:** the gap is a ratio of medians, so 1.00× means the medians coincide, not that the distributions are identical. Per-band counts are equal by construction, so the distributions match *at band granularity* — which is exactly why a finer grid is the stronger claim.

## 2026-07-30 — E13: the tile model's false-positive rate on real photographs

- **Motivation.** E11 reported the tile method's ranking quality (SDXL 0.948) and stopped there. AUC is threshold-free: a model can rank almost perfectly and still place its decision boundary in the wrong place. Manual testing surfaced a real photograph scored at 99% AI, so the operating point was measured directly.
- **Result — real photographs, three sources with different processing histories:**

| real-photograph set | n | called AI | median p | > 0.9 | > 0.99 |
|---|---|---|---|---|---|
| GenImage (ImageNet) — **trained on** | 300 | 45.3% | 0.461 | 4.0% | 0.0% |
| Defactify (MS-COCO) | 300 | 93.3% | 0.935 | 63.3% | 11.7% |
| archive1 (Instagram) | 300 | 99.3% | 0.939 | 73.7% | 2.0% |
| **all** | 900 | **79.3%** | 0.887 | 47.0% | 4.6% |

- **Operating point** — real photographs and generators on the same scale:

| set | median p | >= 0.5 | >= 0.9 | >= 0.99 |
|---|---|---|---|---|
| real (Defactify) | 0.935 | 93.3% | 63.3% | 11.7% |
| DALL-E 3 | 0.534 | 56.7% | 1.0% | 0.0% |
| Midjourney v6 | 0.955 | 99.7% | 86.7% | 0.3% |
| SD 2.1 | 0.990 | 100.0% | 98.7% | 51.3% |
| SD 3 | 0.992 | 100.0% | 99.3% | 61.0% |
| SDXL | 0.993 | 100.0% | 100.0% | 82.7% |

  Real photographs sit at 0.935 and SDXL at 0.993 — 0.06 apart. Pushing the threshold to 0.992 for a 5% false-positive rate drops overall AI recall to **27.3%** (DALL-E 3 and Midjourney to 0%).
- **Conclusion: the tile model has no usable operating point.** Its 0.948 AUC on SDXL is genuine ranking information and simultaneously useless in deployment. The decisive number is in the first table: the model scores its own training real source at 0.461 and every unseen real source at 0.93–0.94.

## 2026-07-30 — E14: the cause — a narrow real class

- **Hypothesis (pre-registered):** the failure in E13 is not calibration but a narrow negative class. A model whose real half comes from one source learns "does this look like that source" rather than "does this carry camera traces", and rejects real photographs from any other pipeline. Widening the real class — with the AI half held fixed — should cut the false-positive rate on unseen real sources at little cost to AI recall.
- **Design.** Five arms, each training on real photographs from one source, plus one arm using all five. The AI half (50,940 images) is **identical in every arm**, and the real budget is equalised at 3,697 so the comparison isolates *diversity*, not volume. Each arm is evaluated on real photographs from sources it never trained on.
- **Result — false-positive rate (% of real photographs called AI):**

| training real source | own source | held-out sources |
|---|---|---|
| CommunityForensics | **0.3%** | **99.9%** |
| GenImage | 23.7% | 91.7% |
| ai-vs-real-200k | 43.7% | 98.2% |
| AI-vs-Real-balanced | 45.4% | 99.3% |
| AIGC-Benchmark | 64.0% | 88.6% |
| **all five sources** | 14–94% | — |

| arm | AI recall | AUC (whole pool) |
|---|---|---|
| single source (any) | 99.5–100% | 0.548–0.661 |
| **all five sources** | 99.8% | **0.884** |

- **Hypothesis confirmed, and the effect is the largest this project has measured.** A model trained on one real source calls 88–99.9% of other sources' real photographs "AI". Training on five sources raises AUC from ~0.6 to **0.884** — and costs nothing: AI recall stays at 99.8% in every arm.
- **Caveat:** the real budget was fixed at the smallest source (3,697) while the AI half stayed at 50,940, so every arm carries a 1:14 imbalance that inflates the absolute false-positive rates. The comparison is unaffected — the imbalance is identical across arms — and AUC is threshold-free, so the 0.55 → 0.884 jump stands. Absolute operating points need a class-balanced repeat.
- **Conclusion — this reframes every earlier result.** The models were not learning "what generated images look like"; they were learning "what my training set's real photographs look like" and labelling everything else AI. It explains the asymmetry noticed in manual testing (the CNN defaults to "real" because downscaling makes unfamiliar inputs look smooth; the statistics models default to "AI" because native texture unlike ImageNet's triggers them), why E12's tenfold data increase did not help (volume rose, real-class *diversity* did not), and why calibration collapses on archive1 (Instagram-processed reals are an unseen pipeline). `IMAGE_FORENSICS_REFERENCE.md` §4.1 states the correct target directly: a detector should read **camera traces** — PRNU, CFA correlation, compression history, which are physics and therefore source-independent — not "unlike my training set", which is source identity.
- **Priority change:** real-class diversity now precedes any backbone upgrade. A stronger network trained on the same narrow real class would answer the same wrong question more sharply.

## 2026-08-04 — E15: Step 0 — class-balanced, multi-source real half

- **Motivation:** E14 showed real-class diversity is the dominant lever but left every arm at a 1:14 class imbalance, which inflated the absolute false-positive rates. AUC was unaffected (it is threshold-free) but the operating point was unreadable.
- **Config:** 36,970 images, exactly balanced (18,485 real / 18,485 AI), the real half drawn evenly — 3,697 from each of five sources. 3 seeds. Compared against v1 (9.9k GenImage) and v2 (101k, source-skewed real half).
- **Result:**

| eval set | v1 | v2 | v3 balanced |
|---|---|---|---|
| GenImage test | 0.974 | 0.914 | 0.919 |
| **archive1** | 0.706 | 0.832 | **0.904** |
| Defactify | 0.717 | 0.694 | 0.692 |

| false positives on real photos | v1 | v2 | v3 |
|---|---|---|---|
| GenImage (trained on) | 8.2% | 12.8% | 13.8% |
| archive1 (unseen) | 30.1% | 31.6% | **19.8%** |
| Defactify (unseen) | 12.7% | 9.6% | **9.4%** |

| AI recall at a 10% false-positive budget | v1 35.6% | v2 32.6% | v3 33.7% |
|---|---|---|---|

- **Conclusion — the fix works exactly where it was predicted to, and nowhere else.** Diversifying the real half cut the false-positive rate on unseen real sources (archive1 30.1% → 19.8%) and raised archive1 AUC by +0.198 over v1. Defactify was untouched: 0.717 → 0.692, and all three models sit at ~33–36% AI recall at a usable threshold. **E14's 0.55 → 0.884 was measured within the pool** — generalisation across the pool's own sources — and does not transfer to Defactify. Two separate problems: a narrow real class (fixed) and weak discrimination of modern generators (not fixed).

## 2026-08-04 — E16: a frozen DINOv2 probe — a large negative result

- **Hypothesis (pre-registered):** E15 exhausted the data explanation, so the ceiling is what the 68 features can express. `IMAGE_FORENSICS_REFERENCE.md` §4.4 names CLIP-style features as "currently among the best out-of-distribution generalizers"; a frozen backbone with a linear probe should beat hand-crafted statistics.
- **Config:** DINOv2 ViT-S/14 at **518px** (a 1024px image is downscaled 2.0× instead of the 4.6× a 224px model forces), frozen, 384-dim embeddings, logistic-regression probe, 3 seeds. Trained on the *same* balanced multi-source pool as E15's v3, so representation is the only variable.
- **Result:**

| eval set | statistics v3 | DINOv2 probe | delta |
|---|---|---|---|
| GenImage test | 0.919 | **0.940** | +0.021 |
| archive1 | 0.904 | 0.873 | −0.031 |
| **Defactify** | 0.692 | **0.480** | **−0.212** |

  Per generator on Defactify: 0.42–0.54 — chance, or inverted. False positives on real photographs: 9.2% on GenImage (trained on), 49.0% on archive1, 63.6% on Defactify. AI recall at a 10% false-positive budget: **9.5%** against the statistics model's 33.7%.
- **Hypothesis falsified, and the failure is diagnostic.** DINOv2 is a *semantic* encoder: its features describe what is in an image, not how the image was produced. The two test sets differ in exactly the way that exposes this:

| set | content control | DINOv2 |
|---|---|---|
| GenImage | reals are ImageNet nature photos, fakes are other content | **0.940** |
| Defactify | fakes generated **from the same MS-COCO captions** as the reals | **0.480** |

- **This carries a warning backwards.** Defactify is content-controlled by construction, so a semantic model has nothing to grab and scores at chance. GenImage is not — which means **a model can score highly there by recognising content rather than generation**, and every GenImage number in this log since E1 inherits that doubt. Defactify is the harder benchmark because it is the fairer one.
- **Conclusion:** hand-crafted low-level statistics are the right *family* — they are weak (0.692) but they read production traces rather than subject matter. A stronger semantic backbone is not the upgrade path. The recommendation in ROADMAP §13b was wrong and is corrected there.

## 2026-08-04 — E17: Module 2's first measurement, against ground-truth masks

- **Motivation:** ROADMAP §9c called tile-based localisation "a well-founded hypothesis that is still unvalidated" for lack of ground truth. The manipulation compilation supplies it: pixel-level masks, plus a `.json` pointer to the authentic original.
- **No training.** The question is narrower than "can we localise": does a model trained to answer *"does this tile look generated"* already answer *"was this tile edited"*? Those coincide for a diffusion-inpainted region and diverge for a Photoshop splice, so results are reported **per sub-dataset**.
- **Prediction (pre-registered):** CocoGlide (diffusion inpainting) should work — the pasted region genuinely is generated texture. CASIA 2.0 and Columbia (classic splices) should not — the pasted region is camera output, just from a different camera.
- **Result:**

| sub-dataset | manipulation type | tampered tiles | clean tiles | tile AUC | IoU | image AUC |
|---|---|---|---|---|---|---|
| **CocoGlide** | diffusion inpainting | 0.600 | 0.455 | **0.648** | **0.419** | **0.721** |
| CASIA 2.0 | classic splice | 0.625 | 0.517 | 0.606 | 0.284 | **0.481** |

- **Prediction confirmed.** On CocoGlide the tile map carries signal at both levels — the model distinguishes tampered from clean tiles *within the same image*, and tampered images from authentic ones. On CASIA the image-level number is 0.481, i.e. chance: manipulated images score 0.760 and authentic ones 0.755. The model is answering its own question correctly; the question is simply the wrong one for a splice, where both host and donor pixels carry camera traces.
### E17 extended, 2026-08-04 — nine sub-datasets, and the IoU column was measuring the mask

Re-running E17 after scripting the dataset preparation (`prepare_manipulation.py`) widened it from 2 usable sub-datasets to 9. The original CASIA/CocoGlide numbers reproduced exactly. Three things came out of the wider run, and one of them invalidates how the original IoU column was read.

**1. IoU was a restatement of mask size, not a measure of skill.** The experiment flags exactly `mask_frac` of the tiles by construction, so flagging *at random* already scores `f/(2-f)` — 0.82 when a mask covers 90% of the image. Against that baseline the ranking nearly inverts:

| sub-dataset | manipulation | mask % | tile AUC | IoU | at random | **margin** | image AUC |
|---|---|---|---|---|---|---|---|
| **CocoGlide** | **diffusion inpainting** | 42 | **0.648** | 0.419 | 0.264 | **+0.155** | **0.721** |
| CASIA 2.0 | splice | 30 | 0.606 | 0.284 | 0.175 | +0.109 | 0.481 |
| VIPP_Realistic | splice | 10 | 0.578 | 0.145 | 0.052 | +0.094 | 0.548 |
| IMD2020 | mixed | 13 | 0.491 | 0.159 | 0.072 | +0.087 | 0.449 |
| DSO-1 | splice | 85 | 0.669 | 0.808 | 0.738 | +0.070 | 0.537 |
| Coverage | copy-move | 14 | 0.498 | 0.130 | 0.078 | +0.052 | 0.479 |
| NIST2016 | splice | 90 | 0.632 | **0.864** | 0.823 | **+0.041** | 0.326 |
| RealisticTampering | splice | 8 | 0.490 | 0.042 | 0.039 | +0.003 | 0.513 |
| CMFD | copy-move | 6 | 0.458 | 0.031 | 0.029 | +0.003 | 0.471 |
| Columbia | splice | — | — | — | — | — | no usable pairs |

  NIST2016's 0.864 is the best-looking IoU in the project and is **+0.041 over chance**; CocoGlide's 0.419 is the real result. The baseline is now printed alongside every IoU.

**2. The original prediction holds far more strongly with nine sets than with two.** CocoGlide is the only sub-dataset built from *diffusion inpainting*, and it is the only one with a real margin at both levels — pixel +0.155 and image 0.721. Every classic-manipulation set sits between 0.326 and 0.548 at image level, i.e. chance. The absolute/relative split argued in E17 is now supported by nine measurements instead of two.

**3. The narrow real class (§12b) shows up again, on completely fresh data.** Read the raw scores rather than the AUCs — on the classic photographic sets the model calls *everything* AI:

| sub-dataset | manipulated | authentic |
|---|---|---|
| NIST2016 | 0.992 | 0.992 |
| CMFD | 0.989 | 0.990 |
| RealisticTampering | 0.976 | 0.976 |
| DSO-1 | 0.975 | 0.976 |

  These are camera photographs from forensics datasets the model has never seen, and it scores them at 0.98. This is E13's 79% false-positive rate reappearing on nine independent sets, and it is why the image-level AUCs sit at chance: both classes are pinned to the ceiling, so there is nothing left to separate. NIST2016 is actually *inverted* (0.326).

**4. Sample sizes are small, and the skips were silent.** A manipulated image is only usable when at least one tile falls clearly inside the mask and one clearly outside. Of a 120-image cap: CASIA 39, CocoGlide 35, VIPP_Realistic 38, CMFD 45, Coverage 62, NIST2016 68, IMD2020 77, RealisticTampering 76, DSO-1 95, **Columbia 0**. These are direction-of-effect results, not precise ones.
- **Conclusion — Module 2 needs two capabilities, not one.** An *absolute* detector ("does this region look generated") covers AI inpainting. A *relative* one ("is this region inconsistent with the rest of this image") is required for classic splices, because the donor region differs from the host in sensor noise, demosaicing signature and JPEG history — none of which is a question about AI.

## 2026-08-04 — E18: ELA for the splice case, and its positive control

- **Hypothesis (pre-registered):** ELA covers the case E17 showed the tile model cannot. `IMAGE_FORENSICS_REFERENCE.md` §4.3 scopes it precisely — it works on JPEG splices where donor and host have different compression histories, and fails **by design** on generated images and uniformly re-encoded ones. So the prediction is the mirror of E17: ELA beats the tile model on CASIA and loses on CocoGlide.
- **Result — on the compilation:**

| sub-dataset | ELA pixel AUC | tile pixel AUC | ELA image AUC | tile image AUC |
|---|---|---|---|---|
| CASIA 2.0 | 0.468 | 0.606 | 0.567 | 0.481 |
| CocoGlide | 0.339 | 0.648 | 0.470 | 0.721 |

  ELA is at or below chance everywhere — including the case it was chosen for.
- **The reference doc requires a positive control before reading a negative ELA result, and it changes the conclusion.** A hand-made splice — host re-encoded at JPEG q95, donor region at q55, composite saved at q90 — gives **tile AUC 0.719**. The implementation is sound and the method works when its precondition is met.
- **Splitting CASIA by the original file extension makes the cause explicit:**

| CASIA manip images | ELA tile AUC |
|---|---|
| originally `.tif` | 0.578 |
| originally `.jpg` | 0.338 |

  The compilation converted every image to PNG. That uniform re-encode is exactly the documented failure mode: the differential compression history ELA reads has been flattened. Testing ELA here was testing it outside its scope.
- **Conclusion.** The two-detector design is sound — an absolute detector for generated regions, ELA for classic splices — and the pairing is supported by a controlled test (0.648 and 0.719 in their respective domains). **It cannot be validated on this dataset**, whose PNG pipeline removes ELA's input. Validating it needs manipulation data that preserves JPEG history, or splices we construct ourselves. Recorded so the negative number above is not read as "ELA does not work".

## 2026-08-05 — E19: pool hygiene, and a shortcut created by fixing another one

- **Motivation (ROADMAP §13d Phase 1):** three defects were known before this ran — a 32px floor, a resolution shortcut in CommunityForensics, and an auditor that had missed it. The point of the phase was to clean the pool everything else will be built on. The result includes one thing nobody predicted.
- **Method — a standing metadata probe.** Train a gradient-boosting model to predict the CLASS from image metadata alone (width, height, aspect, bytes/pixel, squareness). This is archive1's test, where the same probe scored **AUC 1.000**. Anything above chance is a shortcut a model could take instead of looking at content.

### The three fixes

| # | fix | evidence |
|---|---|---|
| 1.1 | 128px floor | `ai_vs_real_balanced` has a **median longest side of 32px**; `features.py` reflection-pads anything below one tile, so the model is shown a synthetic pattern. 27,153 rows dropped |
| 1.2 | `communityforensics` → `whole_image_safe=False` | class 0 is **entirely** 1024², class 1 **entirely** 512² — p10 = median = p90 in both, i.e. two disjoint constants |
| 1.3 | auditor: 2.5× → **2.0× inclusive**, plus a **non-overlap check** on p10–p90 | the split above is a ratio of *exactly* 2.0 and the old rule tested for `> 2.5`. A ratio cannot distinguish "overlapping distributions" from "two disjoint constants", and only the second is a perfect shortcut |

Policy is now read from `SOURCES` at balancing time rather than from the `whole_image_safe` column in `pool_index.csv` — the index is a snapshot, and a stale snapshot silently reintroduces a shortcut. 39,990 rows were carrying the old value.

### The unpredicted result: cleaning one axis broke another

E12 measured compression and explicitly cleared it: *"compression is not a class cue within the pool (real 0.861 vs AI 0.922)"*. After the 32px floor, it was:

| pool | metadata probe (all) | **compression alone** | size alone | bytes/pixel gap |
|---|---|---|---|---|
| raw index (169,668) | 0.916 | 0.684 | 0.885 | 1.01× |
| old balanced (102,492) | 0.853 | 0.673 | 0.804 | 1.07× |
| **resolution-only (45,712)** | 0.818 | **0.633** | 0.720 | **1.65×** |
| **resolution × compression (43,010)** | 0.750 | **0.554** | 0.676 | **1.02×** |

The 32px images were holding the compression axis in balance. Removing them — a fix — produced a **1.65× bytes-per-pixel split between the classes** that had not existed before. Balancing resolution and compression **jointly** removes it at a cost of 2,702 rows.

### Which axes actually matter, and why the remaining 0.750 is not alarming

| axis | AUC alone | survives into a 128px tile? |
|---|---|---|
| size (longest side) | 0.720 | **no** — a tile carries no record of its parent's dimensions |
| aspect ratio | 0.591 | no |
| squareness | 0.506 | no |
| **compression** | **0.633 → 0.554** | **yes** — JPEG artefacts are in every tile |

Only compression survives tiling, and it is now at chance. The residual 0.750 is carried by size and aspect, which a tile-trained model cannot see. **This is §1b's rule in action: a flaw is a usage condition. This pool is clean for tile training and still unfit for whole-image native-resolution training.**

- **Caveat, and it is a real one.** "Size does not survive tiling" is true of *metadata*, not of *texture*. E8's shortcut probe predicted image width from the 68 features at **92.6% accuracy in crop128 mode**, where every input was already 128×128. Resolution leaks through texture. That probe can only be re-run once features are extracted for this pool (Phase 2), and it should be.
- **Conclusion.** Auditing a merged pool is not a one-off gate but an invariant to re-check after every change, because the axes interact: the fix for one created a shortcut on another, and only re-running the probe caught it. The metadata probe is cheap and now belongs in the pool build itself.

### E19b — the label direction was never checked, and two sources declare the opposite of ours

Found on 2026-08-05 while auditing the pool for E19. It is the most consequential bug this project has produced, and the five existing audit checks could not have caught it: they inspect pixels and file properties, and this is a question about what a **number means**.

**The project uses `0 = real, 1 = AI` everywhere. Two of the five training sources declare the reverse in their own HuggingFace metadata**, and `build_pool.py` was reading `int(row[label_col])` raw:

| source | ClassLabel `names` | direction |
|---|---|---|
| `theminji/AI-vs-Real-balanced` | `["AiArtData", "RealArt"]` | **0 = AI** — inverted |
| `theminji/ai-vs-real-200k` | `["ai", "real"]` | **0 = AI** — inverted |
| `TheKernel01/AIGC-Detection-Benchmark` | `["real", "fake"]` | 0 = real — correct |
| `OwensLab/CommunityForensics-Small` | *(no metadata)* | resolved below — correct |
| `genimage` | *(folders)* | correct by construction |

**Verification, three independent ways** — the metadata alone was not treated as sufficient:

1. **Visual.** Sampling only images ≥200px (the 32px half is unreadable to the eye and misled a first attempt), every `label 0` sample from `AI-vs-Real-balanced` is unmistakably diffusion output — a product-render smartwatch in a rain-lit alley, a hyperreal golden-hour wheat field, both 1024². Every `label 1` sample is an ordinary candid photograph.
2. **CommunityForensics has no ClassLabel metadata**, so its `model_name` column settled it: label 0 is **100% `FFHQ`** (Flickr-Faces-HQ, a real photograph set), label 1 carries diffusion model ids (`WarriorMama777/AbyssOrangeMix`, `lewdryuna/A-Rainier`, …). Correct order. *(Side finding: its entire real half is FFHQ — one dataset, one content type, one resolution. That is E14's narrow-real-class problem in its purest form, and it explains why the CommunityForensics-only arm scored 0.3% false positives on itself and 99.9% on everything else.)*
3. **A transfer probe was run and proved inconclusive**, which is itself informative: a model trained on genimage scores `ai_vs_real_200k`'s two classes at 0.445 and 0.444, and CommunityForensics' at 0.741 and 0.739 — no separation at all. The reference model does not generalise to those sources (E14 again), so this test *cannot* resolve label direction, and reporting it as evidence either way would have been wrong.

**Blast radius:**

| artifact | rows | inverted |
|---|---|---|
| `pool_index.csv` | 169,668 | **79,838 (47.1%)** |
| `pool_features.npz` — E12/E14/E15's training data | 101,027 | **49,724 (49.2%)** |
| `pool_balanced.csv` | 102,492 | 50,816 (49.6%) |
| `pool_balanced_v3.csv` — E15/E16 | 36,970 | 17,349 (46.9%) |
| `pool_tile_v1.csv` — built earlier the same day | 43,010 | 21,343 (49.6%) |

**Affected: E12, E14, E15, E16.** Every pool-trained model (`feature_full_v2`, `v3`, `v4`, the DINOv2 probe, E14's five arms) learned from a target that was wrong about half the time.

**Not affected:** everything trained from image folders, where the mapping is ours — `best.pt`, `best_genimage.pt`, `feature_full.joblib`, and critically **`feature_crop128.joblib`**, the tile model behind the demo and all of Module 2. **E1–E11, E17 and E18 stand.**

**What it might explain** — listed as suspects for re-measurement, not as conclusions: E12's "ten times the data did not help", E15's "the gain did not transfer to Defactify", E16's DINOv2 at 0.480, and the absolute magnitude of E14's false-positive rates.

**Fixes:**
- `SOURCES` now carries `label_map` (raw → project) and `label_names`; `to_project_label()` **raises** on an undeclared source rather than assuming, and `verify_labels()` compares the file's own ClassLabel order against what we expect and **raises on a mismatch** — a dataset re-exported with swapped classes must crash, not silently invert.
- The auditor gained a sixth check, `label_direction()`, which reads the ClassLabel names and flags any source whose index 0 means AI.
- The index was rebuilt rather than patched in place: a CSV that might hold raw or mapped labels is a double-inversion waiting to happen. The mislabelled one is kept as `pool_index_BOZUK_etiket.csv.bak`.

**The lesson, and it is a new one for §1b:** auditing has been about whether a model can separate the classes *without looking at the image*. This bug is the opposite failure — the images were fine and the **question** was wrong. "Is this dataset biased?" and "does this dataset mean what I think it means?" are different checks, and only the first was being run.

## 2026-08-05 — E19c: re-running E12/E14/E15/E16 with corrected labels

Same images, same seeds, same code — **only the label column changed** (E19b), so the label bug is the single variable. `feature_full_v2` and `v4` were retrained and overwritten; the poisoned artifacts are kept as `*.BOZUK_etiket.bak`.

| experiment | verdict |
|---|---|
| **E14** — narrow real class | **stands, essentially unchanged** |
| **E12** — ten times the data | **partly revised** — it helped more than reported, on the axis it was reported not to help |
| **E15** — balanced multi-source | **partly revised** — one conclusion inverted, the headline survives |
| **E16** — frozen DINOv2 probe | **overturned. The falsification was an artifact of the bug** |

### E14 stands

| | old | corrected |
|---|---|---|
| single-source AUC (pool) | 0.548–0.661 | **0.558–0.684** |
| all five sources | **0.884** | **0.894** |
| held-out false positives | 88.6–99.9% | 85.6–99.9% |
| AI recall, every arm | 99.5–100% | 99.6–100% |

The dominant-lever finding is unaffected: a real class from one source rejects 86–99.9% of other sources' photographs, five sources lift pool AUC from ~0.6 to ~0.89, and diversity still costs nothing in AI recall. **§12b's reframing of the project survives the bug that was under it.**

### E12 — "data volume alone does not help" was too pessimistic

| eval set | v1 | v2 old | **v2 corrected** |
|---|---|---|---|
| GenImage test | 0.974 | 0.913 | 0.918 |
| **archive1** | 0.706 | 0.839 | **0.922** (+0.217 over v1) |
| Defactify | 0.717 | 0.692 | **0.715** (flat, not a loss) |
| pool held-out | — | 0.895 | 0.904 |

And false positives, which the original entry never reported: archive1 **30.1% → 12.3%**, Defactify **12.7% → 7.6%**. So ten times the data *did* help — substantially on an unseen real source, and on both false-positive rates. What it did not move is Defactify, the content-controlled benchmark. **The DALL-E 3 collapse persists and is larger (0.808 → 0.559)**, so E12's compression-gap explanation for it still stands.

### E15 — the balanced arm's advantage inverted

| eval set | v1 | v2 | v3 old | **v3 corrected** |
|---|---|---|---|---|
| **Defactify** | 0.717 | 0.705 | 0.692 | **0.728** — now the best of the three |
| archive1 | 0.706 | 0.911 | 0.904 | 0.902 |

| false positives | v2 old | v2 corrected | v3 old | v3 corrected |
|---|---|---|---|---|
| archive1 (unseen) | 31.6% | **13.6%** | **19.8%** | 28.6% |
| Defactify (unseen) | 9.6% | **8.2%** | **9.4%** | 19.3% |

v3 was reported as the deployment winner; corrected, **v2 is**. The premise for building v3 — that v2's real half was source-skewed — was itself partly a label artifact. But the conclusion that matters is untouched: **AI recall at a 10% false-positive budget is 35.6% / 29.1% / 33.8%.** Three pools, three recipes, ~33% either way. No configuration of the training data produces a usable operating point.

### E16 — overturned, and it was the biggest claim in the log

| eval set | statistics v3 | DINOv2 old | **DINOv2 corrected** |
|---|---|---|---|
| GenImage test | 0.919 | 0.940 | 0.917 |
| archive1 | 0.904 | 0.873 | **0.929** |
| **Defactify** | 0.692 | **0.480** | **0.764** |
| AI recall @10% FP | 33.7% | **9.5%** | **40.4%** |

Per generator on Defactify: dalle3 0.808, sd21 0.797, sdxl 0.770, sd3 0.706, midjourney 0.739 — nothing near chance.

**0.764 is the highest whole-image Defactify AUC this project has produced** (ResNet-18: 0.760), and **40.4% is the best operating point measured**. Everything E16 concluded was wrong:

- "DINOv2 scores at chance because it is a semantic encoder and Defactify is content-controlled" — it scores 0.764. The content-control argument was an explanation invented for a number produced by a broken label column.
- "This carries a warning backwards: every GenImage number inherits the doubt that a model can score highly there by recognising content" — that warning rested on the 0.480, and does not survive it. GenImage remains not content-controlled, which is still worth stating, but there is no measurement behind the alarm.
- "Hand-crafted low-level statistics are the right *family*; a semantic backbone is not the upgrade path" — reversed. The backbone beats the statistics on the fairest benchmark and at the operating point.

`ROADMAP.md` §13b struck through its own CLIP/DINOv2 recommendation on the strength of this experiment. The strike-through is removed.

- **Caveat.** DINOv2's false positives at threshold 0.5 are high (archive1 71.8%, Defactify 53.4%), so its *calibration* is poor while its *ranking* is the best available — the E11→E13 distinction again, now in the other direction. And this is still a **whole-image** probe at 518px; the native-tile version §13b actually recommended remains untested and is now the most promising open experiment in the project.
- **Conclusion.** One mislabelled column produced a confident, well-argued, three-part falsification of the correct research direction. The mechanism was invisible to five audit checks that all inspect pixels, and the write-up's own plausibility is what made it stick. Re-running everything downstream of a data fix is not optional.
