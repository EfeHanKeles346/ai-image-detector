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
- **Conclusion:** the scale problem that dominated E5, E6 and E7 is dissolved rather than mitigated — the model always sees 128×128 native pixels, and resolution changes only *how many tiles* come out, never what a tile looks like. Two consequences beyond Module 1: image dimensions can no longer act as a shortcut (so the datasets flagged in HISTORY §1c become usable), and the per-tile scores are directly a localisation map, which is Module 2's core machinery obtained as a side effect.
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
- **Conclusion:** hand-crafted low-level statistics are the right *family* — they are weak (0.692) but they read production traces rather than subject matter. A stronger semantic backbone is not the upgrade path. The recommendation in HISTORY §13b was wrong and is corrected there.

## 2026-08-04 — E17: Module 2's first measurement, against ground-truth masks

- **Motivation:** HISTORY §9c called tile-based localisation "a well-founded hypothesis that is still unvalidated" for lack of ground truth. The manipulation compilation supplies it: pixel-level masks, plus a `.json` pointer to the authentic original.
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

- **Motivation (HISTORY §13d Phase 1):** three defects were known before this ran — a 32px floor, a resolution shortcut in CommunityForensics, and an auditor that had missed it. The point of the phase was to clean the pool everything else will be built on. The result includes one thing nobody predicted.
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

`HISTORY.md` §13b struck through its own CLIP/DINOv2 recommendation on the strength of this experiment. The strike-through is removed.

- **Caveat.** DINOv2's false positives at threshold 0.5 are high (archive1 71.8%, Defactify 53.4%), so its *calibration* is poor while its *ranking* is the best available — the E11→E13 distinction again, now in the other direction. And this is still a **whole-image** probe at 518px; the native-tile version §13b actually recommended remains untested and is now the most promising open experiment in the project.
- **Conclusion.** One mislabelled column produced a confident, well-argued, three-part falsification of the correct research direction. The mechanism was invisible to five audit checks that all inspect pixels, and the write-up's own plausibility is what made it stick. Re-running everything downstream of a data fix is not optional.

## 2026-08-06 — E20: three model families on identical native tiles (Phase 2b)

- **Hypothesis (pre-registered):** after Phase 1 cleaned the pool and Phase 2a fixed the tiling, the statistics model still stalls at ~32% AI recall at a 10% false-positive budget — the same figure v1, v2 and v3 reached in E15. If the training data is no longer the limit, the limit is what the 68 features can express, and a learned representation on the *same* tiles should beat them.
- **Controlled:** all three arms read one cached tensor, `tiles_v1.npz` — 48,037 native 128px tiles, 24,011 real / 24,026 AI, one tile per pool image at a **seeded random position** (not the centre) with the **same texture floor inference uses**. Same crops, same seed, same evaluation. The only variable is the model.
- **Evaluation is end-to-end, not per-tile:** each test image is tiled exactly as `serve.py` tiles it (full coverage, edge-anchored, texture floor), every tile scored, top-3 mean taken as the image's score. 1 seed, 200 images per set, 8-epoch ceiling with the epoch chosen on a source-stratified validation slice.

### Result — AI recall at a 10% false-positive budget (the operating point)

| | statistics | **ResNet-18 @128** | SmallCNN @128 |
|---|---|---|---|
| SDXL (1024px) | 72.5% | **83.5%** | 54.5% |
| SD 3 (1024px) | 51.0% | **78.0%** | 41.5% |
| SD 2.1 (768px) | 40.5% | **67.5%** | 47.5% |
| Midjourney (436px) | 5.0% | **51.5%** | 24.5% |
| DALL-E 3 (270px) | 4.5% | 9.5% | 4.0% |
| **Defactify, all five** | **39.0%** | **55.5%** | 30.5% |

### AUC (ranking only)

| | statistics | **ResNet-18** | SmallCNN |
|---|---|---|---|
| Defactify | 0.603 | **0.770** | 0.655 |
| GenImage test | 0.641 | **0.783** | 0.499 |

- **Hypothesis confirmed. 55.5% is the best operating point this project has produced** — up from 39.0% on identical inputs, and above E19c's whole-image DINOv2 probe (40.4%). §13b's second half, "apply a strong representation to native tiles rather than whole images", was the untested half of that recommendation and it is now measured.
- **The largest single gain is Midjourney: 5.0% → 51.5%.** The statistics model was effectively blind to that generator; the same tiles through a pretrained backbone are not. This is E2's conclusion — *the representation, not the classifier, is the bottleneck* — finally tested with a strong representation on the right input.
- **SmallCNN loses to both (30.5%), and informatively.** 0.3M parameters from scratch on 48k tiles is not enough; the gap to ResNet-18 is 25 points on the same data. So the win is not "a CNN instead of statistics" — it is **ImageNet pretraining**. A from-scratch network of this size does worse than hand-crafted physics.

### Two things this does NOT fix

- **DALL-E 3 stays broken in all three arms** (4.0–9.5% recall, AUC 0.246–0.360, i.e. at or below chance). 270px at ~16 KB: the tile method has no measurable texture to read, exactly as E8 and E11 predicted. Small compressed inputs need a different route, not a better tile model.
- **Calibration is now THE blocker, and it is worse than the ranking suggests.** At threshold 0.5 every arm calls the overwhelming majority of real photographs AI:

| | statistics | ResNet-18 | SmallCNN |
|---|---|---|---|
| Defactify reals | 96.0% | 91.5% | 98.5% |
| **2,314 authentic camera photographs, 10 forensics datasets** | 93.0% | **86.5%** | 94.6% |

  So all three rank well and none can decide. This is the E11 → E13 pattern again: ranking quality and deployability are separate claims. **The bottleneck has moved — it is no longer the data (Phase 1) and no longer the representation (this experiment). It is the operating point.**

- **Evaluation note:** `archive1` is deliberately absent. E10 showed the CNNs were immune to its metadata confound because `Resize()` destroys dimensions — but a 128px tile carries its parent's **compression**, and archive1's real half sits at 0.190 bytes/pixel against 1.331 for its AI half. That 7× split survives tiling, so a tile model could score there without reading a generation trace. It is replaced by 2,314 authentic photographs from ten forensics datasets, a real-only probe that cannot be gamed because there is no second class to shortcut toward. **E13's and E15's archive1 numbers should be read with that caveat.**
- **Caveats.** One seed (the ≥3-seed rule is not met; the 16-point gap is far outside plausible seed noise but the figure should be repeated). 200 images per test set. The tile dataset holds one tile per image, chosen for the texture floor, so it under-represents flat regions by construction — deliberate, since inference drops them too, but it means the model has never seen the population it will refuse to score.

### E20 protocol v2 — implemented, full rerun pending

The numbers above are the original E20 measurement and are not relabelled as v2 results. The
evaluation script has now been hardened before spending another multi-hour training run:

- Every image's complete per-tile score vector is written to JSONL. Aggregation experiments no
  longer re-run the model, and the evidence behind an image score is inspectable.
- Defactify real images and each generator are split independently into stable, disjoint
  calibration/evaluation halves. The aggregation rule and 10% FP threshold see calibration only;
  AUC, recall and false positives come from untouched evaluation images.
- Five aggregation candidates are compared: top-3, top-10%, p90, mean, and a fixed-16-tile top-3
  control for the variable-tile-count/order-statistic shortcut.
- The Defactify-calibrated threshold is transferred unchanged to each of the ten forensic real
  sources. Macro and worst-source FP are now headline columns; pooled FP can no longer hide one
  camera pipeline failing catastrophically.
- Reportable runs default to all three registered seeds. CNN checkpoints now store the selected
  aggregation, threshold, normalization, tile contract, best epoch/AUC and training-data
  provenance instead of only `model/arm/seed`.

### E20-v2 checkpoint diagnostic — 2026-08-06

Before paying for a three-arm × three-seed retrain, the existing E20 ResNet-18 seed-42 checkpoint
was passed through the hardened protocol. This is a **checkpoint-only diagnostic**, not a new
training run and not a replacement for the registered three-seed comparison.

| aggregation | calibration AI recall | evaluation AI recall | evaluation FP | evaluation AUC | forensic macro FP | worst-source FP |
|---|---:|---:|---:|---:|---:|---:|
| **top-3 (selected on calibration)** | **61.4%** | **61.4%** | 19.0% | **0.770** | 45.0% | 96.0% |
| top-10% | 46.8% | 52.0% | 15.0% | 0.762 | 30.6% | 70.0% |
| p90 | 50.4% | 53.4% | 16.0% | 0.758 | 31.8% | 75.5% |
| mean | 47.2% | 52.4% | **10.0%** | 0.766 | 28.0% | **58.0%** |
| fixed-16 top-3 | 49.0% | 49.8% | 14.0% | 0.730 | **27.4%** | 61.5% |

- Top-3 is selected legitimately: only the calibration halves choose the rule, and it has the
  highest macro generator recall there. Its 61.4% untouched recall is stronger than E20-v1's
  55.5%, but deployability moves in the opposite direction.
- A threshold fitted for 10% FP on Defactify calibration transfers to **19% FP on Defactify's
  untouched half**. On ten authentic forensic sources it reaches **45.0% macro FP** and **96.0%
  worst-source FP** (`RealisticTampering`). This is not a small calibration error.
- Replacing top-3 helps but does not solve it. Mean aggregation reaches 28.0% macro / 58.0% worst
  FP; fixed-16 reaches 27.4% / 61.5%. DSO-1 has more tiles than the worst source yet only 2% FP
  under mean/top-10%, so variable tile count is a contributor, **not the primary cause**. The
  dominant failure is source/pipeline shift.
- Per-generator evaluation recall under selected top-3 is DALL-E 3 10%, Midjourney 57%, SD 2.1
  66%, SD 3 82%, SDXL 92%. The small/compressed DALL-E route remains unsolved.

**Decision:** do not expose this ResNet checkpoint as an API verdict and do not spend the next
compute block merely repeating it. First put frozen B-Free and CLIP baselines through this exact
evaluator. If neither improves cross-source specificity, the next experiment is source-balanced
real calibration/training; if one does, only then pay for its three-seed confirmation. The full
E20-v2 command remains the reproducibility target, but is no longer the highest-information next
run:

```bash
cd ml
PYTHONPATH=src .venv/bin/python experiments/e20_tile_model_shootout.py \
  --seeds 3 --arms stats resnet18 small_cnn \
  --raw-dir artifacts/e20/raw_scores --results artifacts/e20/results.json
```

## 2026-08-19 — E21: the strongest frozen external detector, on our protocol

- **Hypothesis (pre-registered in PLAN.md):** if any frozen external detector passes the
  cross-source specificity gate our own models fail (E20-v2: 45% macro / 96% worst-source FP),
  buying a representation is cheaper than fixing ours. First candidate: the **Community-Forensics
  ViT-S** (Park & Owens, CVPR 2025, MIT) — first on 8 of 12 datasets in the 23-detector
  out-of-the-box benchmark (arXiv 2602.07814).
- **Config:** `e21_external_detector_benchmark.py --detector community-forensics`, checkpoint
  `275ba982236ddd6a…` exactly as published (authors' processor: shortest edge 440, centre-crop
  384, CLIP normalisation — whole image, no tiling). E20-v2 protocol unchanged: threshold fitted
  for 10% FP on the Defactify real **calibration half only**, everything measured on untouched
  halves; the same threshold transferred to ten forensic real sources. 3,056 images, 111 s on MPS.
- **Result — against our best model (E20-v2 tile ResNet-18 checkpoint, selected top-3):**

| metric | tile ResNet-18 (ours) | **CF ViT-S (frozen)** |
|---|---|---|
| Defactify evaluation AUC | 0.770 | **0.876** |
| AI recall (untouched half) | 61.4% | **70.8%** |
| Defactify FP at the fitted threshold | 19.0% | **8.0% — the budget holds** |
| forensic macro FP | 45.0% | **29.9%** |
| worst-source FP | 96.0% (RealisticTampering) | **81.6% (NIST2016)** |
| DALL-E 3 recall | 10% | 23% (AUC 0.627) |
| GenImage AUC | 0.783 | 0.997 |

  Per forensic source: NIST2016 81.6, Columbia 48.6, CASIA2.0 32.0, VIPP 29.4, CocoGlide 24.0,
  Coverage 24.0, IMD2020 19.5, CMFD 18.8, RealisticTampering 18.5, DSO-1 3.0 (% FP).
- **Read with one caveat.** Community-Forensics trains its real class on FFHQ, VISION, COCO and
  Landscapes HQ — and Defactify's real half **is** MS-COCO. Its clean 8% Defactify FP is therefore
  partly an in-distribution number. The forensic sources are unseen camera pipelines for both
  models, so the 29.9% / 81.6% columns are the honest cross-source comparison — and they are also
  the gate.
- **Conclusion — better everywhere, and still not deployable.** A frozen detector trained on
  4,803 generators beats our tile ResNet on every headline column, holds its FP budget on its
  evaluation domain, and *still* calls 82% of one unseen camera source's real photographs "AI".
  Two consequences: (1) representation-shopping alone does not pass the gate — the cross-source
  decision problem is a property of the task, not of our model, which is exactly what E14
  predicted; (2) CF-ViT is now the strongest baseline in the project and the candidate
  representation for the source-robust calibration work (PLAN items 2–3). B-Free remains queued
  as the second arm; the interesting question it answers is whether content-aligned training
  (its bias-free recipe) closes the NIST2016-style source gap that generator diversity did not.

### E21b — B-Free, same protocol, 2026-08-19

- **Config:** `--detector bfree`, official checkout `c6a9f898`, weights `BFREE_dino2reg4`
  (MD5-verified against upstream), authors' native contract: five-crop mean at 504px, no
  resize. Resumed 1,050 scores from the 08-06 partial run via the JSONL cache; ~19 min total
  on MPS. Licence: informational/nonprofit — acknowledged on the CLI.
- **Result — all three detector families side by side:**

| metric | tile ResNet-18 (ours) | CF ViT-S | **B-Free** |
|---|---|---|---|
| Defactify evaluation AUC | 0.770 | 0.876 | **0.926** |
| AI recall (untouched half) | 61.4% | 70.8% | **81.2%** |
| Defactify FP at fitted threshold | 19.0% | **8.0%** | 11.0% |
| forensic macro FP | 45.0% | 29.9% | **23.6%** |
| **worst-source FP** | 96.0% (RealisticTampering) | 81.6% (NIST2016) | **96.8% (NIST2016)** |
| DALL-E 3 recall / AUC | 10% / ≤0.36 | 23% / 0.627 | **68% / 0.867** |
| Midjourney recall | 57% | **61%** | 39% |

  B-Free per source: NIST2016 **96.8**, Columbia 31.1, VIPP 22.1, CMFD 18.8, CASIA 17.5,
  IMD2020 16.5, Coverage 13.0, RealisticTampering 11.0, CocoGlide 8.5, DSO-1 1.0 (% FP).
- **Three observations, one conclusion.**
  1. **The gate stands against three independent training philosophies.** Our tiles, a ViT
     trained on 4,803 generators, and content-aligned bias-free training all fail the same
     test: at least one unseen camera pipeline above 81% FP. B-Free is the best model on
     nearly every column and simultaneously the *worst* on the gate column. Cross-source
     decision-making is a property of the task; no amount of representation shopping has
     touched it. PLAN items 2–3 (source-balanced calibration, conformal abstention) are now
     the only untested lever.
  2. **B-Free largely solves the DALL-E 3 route** (68% recall, AUC 0.867, against our 10%
     at chance) — E20 concluded small/compressed inputs "need a different route, not a
     better tile model", and this is that route: content-aligned training plus native
     five-crop inference reads compressed 270px images our whole pipeline could not.
  3. **NIST2016 is the universal poison source** (96.8 / 81.6 / inverted in E17). Whatever
     that pipeline does to its authentic images, every detector family reads it as
     synthetic — worth one diagnostic look before any calibration work, since a single
     source dominates every macro number.
- **Decision:** B-Free replaces CF-ViT as the strongest external baseline; the CLIP probe
  is dropped (a third frozen model cannot answer a question two have already answered).
  Next experiment: source-robust decision rules on top of the two external score sets we
  now hold — both JSONLs are cached, so calibration experiments cost seconds, not GPU time.

## 2026-08-19 — E22: source-robust calibration — the decision layer, measured

- **Hypotheses (pre-registered in the script header):** H1 — NIST2016's universal failure
  has a measurable pipeline explanation. H2 — a threshold calibrated on many real pipelines
  transfers to an unseen pipeline far better than the Defactify-only threshold, at a
  measurable recall cost; the worst-source (max-over-pipelines) rule is the only one with a
  chance of holding the budget. H3 — a two-threshold abstention band turns an undeployable
  single threshold into a deployable partial decision.
- **Config:** `e22_source_robust_calibration.py`, entirely on the cached per-image scores of
  three arms (tile ResNet-18 top-3, CF ViT-S, B-Free) over the identical 3,056 images;
  same deterministic path-hash splits as E20-v2/E21 (seed 2026, fraction 0.5); FP budget 10%,
  miss budget 10%. A full run costs ~2 s — no model is loaded.
- **H1 confirmed — NIST2016 is the 12-megapixel source.** Median 12.19 Mpx against 0.07–3.15
  for every other pipeline, at the lowest bytes/pixel of the forensic set (0.67). Both
  external models score it far above their other real sources (B-Free median −0.74 vs
  −3.5…−5.0). A second gap surfaced in passing: Defactify reals sit at 0.16 B/px while every
  forensic source sits at 1.1–1.9 — the calibration domain is heavily compressed, the
  transfer domain is not. E12's compression gap, reappearing at the decision layer.
- **H2 — leave-one-source-out, held-out pipelines only:**

| arm · rule | worst FP | macro FP | macro recall |
|---|---|---|---|
| tile ResNet-18 · defactify-only | 99.0% | 45.9% | 61.4% |
| tile ResNet-18 · worst-source | 12.0% | 1.2% | **1.2%** |
| CF ViT-S · defactify-only | 74.6% | 28.7% | 70.8% |
| **CF ViT-S · worst-source** | **6.6%** | **1.3%** | **28.4%** |
| B-Free · defactify-only | 96.8% | 25.0% | 81.2% |
| B-Free · worst-source | 44.4% (NIST2016; all others ≤5.0%) | 6.0% | 65.5% |

  **CF ViT-S under the worst-source rule is the first operating point in this project's
  history that passes the gate on genuinely unseen pipelines** — worst held-out FP 6.6%
  (Columbia), NIST2016 held out included (3.2%) — at 28.4% macro recall (sd21 67.5%,
  sdxl 42.8%, sd3 22.3%, midjourney 7.2%, dalle3 2.0%). B-Free passes on ten of eleven
  pipelines with far better recall (65.5%) but its NIST2016 shift is so large that no
  other source's calibration anticipates it. Our tile ResNet is **not salvageable by
  calibration**: a source-robust threshold leaves 1.2% recall — its scores are not
  source-invariant, full stop.
- **H3 — the abstention band (t_ai worst-source over all 11 pipelines' calibration halves;
  t_real at 10% miss on generator calibration halves; evaluation halves only):**

| arm | worst real FP | macro real FP | AI recall | AI abstain | AI wrongly-real |
|---|---|---|---|---|---|
| tile ResNet-18 | 8.0% | 0.8% | 1.2% | 90.2% | 8.6% |
| CF ViT-S | 6.6% | 1.3% | 28.0% | 59.0% | 13.0% |
| **B-Free** | **7.9%** | **2.7%** | **65.2%** | **21.2%** | 13.6% |

  The band does exactly what it exists for: NIST2016 lands at **92.1% abstain / 7.9% FP /
  0% "real"** — the model's confusion is routed to "insufficient evidence" instead of a
  false accusation. On the AI side the SD family is essentially solved (93–100% recall, 0%
  wrongly-real); the honest weakness is midjourney (9% recall, 40% actively called real).
- **Conclusion — the decision layer is solvable, and the lever is the pipeline library.**
  Two deployable stories now exist: (a) for *unseen* pipelines, CF ViT-S + worst-source
  calibration holds a ≤10% budget at 28% recall; (b) when every deployment pipeline family
  can contribute ~100 calibration images (no retraining, threshold-only), B-Free's band
  reaches 65% recall at ≤8% FP everywhere with a 21% abstention rate. What E13 called "no
  usable operating point" was true of the model *and* of a one-source decision rule; with
  the rule fixed, the frozen representations clear the bar the representations alone could
  not. Caveats: one deterministic split (34–100 evaluation images per source), thresholds
  are per-arm score-scale specific, and the Defactify-vs-forensics compression gap means
  part of every defactify-only failure is compression, not source identity.

## 2026-08-19 — E23a: the Midjourney wrongly-real diagnostic, and the price of the "real" verdict

- **Hypotheses (pre-registered in the script header):** H1 — the 40% of Midjourney that the
  B-Free band actively calls "real" is a measurable subgroup (resolution or compression).
  H2 — tightening the miss budget converts wrongly-real into abstention at a measurable
  cost in authentic "real" coverage; the frontier decides whether an asymmetric band is
  worth it. Cached scores only; ~2 s.
- **H1 refuted, informatively.** The wrongly-real Midjourney images are *not* a subgroup:
  same long side (436px) and same bytes/pixel (0.12) as the caught ones. The whole
  generator's score distribution simply sits near the reals in B-Free's space (MJ median
  −2.7 vs real −5.0, while the SD family sits at +5.2…+8.0). B-Free trains on SD-family
  reconstructions; Midjourney's artefacts are the furthest from that family, and the 39%
  recall of E21b is the same fact from the other side. CF-ViT separates MJ better (−5.9 vs
  real −11.3; only 10% wrongly-real) — the arms' blind spots differ, as in E8/E9.
- **H2 — the frontier (B-Free arm; AI recall stays 65.2% throughout, t_ai untouched):**

| miss budget | MJ wrongly-real | macro AI wrongly-real | macro real coverage | min real coverage |
|---|---|---|---|---|
| 10% | 40.0% | 13.6% | 66.0% | **0.0%** |
| 5% | 25.0% | 7.8% | 51.1% | **0.0%** |
| 2% | 10.0% | 2.8% | 27.4% | **0.0%** |
| none | 0.0% | 0.0% | 0.0% | 0.0% |

- **The decisive column is the last one: at every budget, at least one authentic pipeline
  (NIST2016) gets 0% "real" coverage.** The "real" verdict was never a consistent promise —
  it is generous on friendly pipelines and silent on hostile ones, and it is the only
  verdict through which AI content can be actively laundered ("a detector said this is
  real").
- **Decision: the band becomes asymmetric.** Two verdicts — "AI" (above the worst-source
  threshold) and "insufficient evidence" (below it). No "real" verdict: the honest phrasing
  is "no AI evidence found", which is not a certificate of authenticity. This costs nothing
  measurable (AI recall unchanged, FP unchanged) and removes the band's only
  laundering-capable output. If a product context ever demands a "leaning real" signal, the
  2% miss budget is the recorded compromise (2.8% macro wrongly-real, 27% coverage).

## 2026-08-19 — E25: the 2026-generator probe — five never-used SSD sources meet the band

- **Hypotheses (pre-registered in the script header):** H1 — recall varies by generator
  *family*: diffusion-family generators should be caught, native multimodal ones (GPT
  Image, Nano Banana) are the risk. H2 — julienlucas' real half is an honest 12th
  unseen-pipeline test for the frozen worst-source thresholds.
- **Config:** 200 images per source, stride-sampled and written as raw bytes (no
  re-encode) from five never-used SSD sets; julienlucas' inverted label direction
  (`0=fake`) declared and verified against parquet metadata — the E19b guard, firing on
  the first new dataset since it was written. Both external arms, frozen t_ai from E22/E23a
  (asymmetric band). ~13 min total.
- **Result (AI verdict % at frozen t_ai · AUC vs julienlucas reals):**

| source (2025–26 era) | CF ViT-S | B-Free |
|---|---|---|
| julienlucas real half — **FP** | **0.5%** | **6.0%** |
| julienlucas AI (MJ/DALL-E/SD/NBP mix) | 22.5% · 0.780 | 60.5% · 0.794 |
| FLUX.1-dev | 43.5% · 0.927 | 48.0% · 0.836 |
| Nano Banana (Gemini 2.5 Flash Image) | 46.0% · **0.940** | 39.0% · 0.802 |
| Nano Banana Pro | 23.0% · 0.869 | 51.5% · 0.848 |
| **GPT Image 4K** | 6.0% · 0.695 | 8.0% · **0.478 — chance** |

- **H2 confirmed, and it is the headline: the worst-source thresholds held on a genuinely
  fresh 2026-era real pipeline they had never seen.** 0.5% / 6.0% FP with no recalibration
  is the strongest evidence yet that E22's decision rule, not luck, passes the gate. (E24's
  personal-photo test remains queued as the second fresh pipeline.)
- **H1 confirmed with one surprise.** GPT Image 4K is the blind spot — 6–8% recall, and
  for B-Free literally chance-level ranking (0.478): an autoregressive/native-multimodal
  generator leaves none of the traces either detector reads. The literature's "18–30% on
  commercial 2026 APIs" is our measurement too. The surprise is Nano Banana: also a native
  multimodal stack, yet well-ranked (0.940 CF) — "native multimodal" is not one family
  forensically.
- **Caveats.** The AUC column pairs each AI set against julienlucas' reals — a
  cross-collection comparison that can carry format/pipeline bias; the frozen-threshold
  verdict rates are the cleaner claim. julienlucas' AI half is a pooled mix with no
  per-generator column. Single split, 200 images per cell.
- **Consequence.** The band ships with a known coverage statement: strong on
  diffusion-family output, honest abstention on GPT-Image-class output — and the
  "insufficient evidence" verdict is exactly what it returns there (92–94% of GPT Image
  gets no verdict rather than a false "real"). Detecting the autoregressive family needs a
  representation trained on it; that is a data acquisition item, not a calibration one.

## 2026-08-19 — E23b: the megapixel cap — the last failing pipeline passes

- **Hypotheses (pre-registered):** H1 — capping NIST2016's long side to 2048px moves its
  authentic scores toward the other forensic sources for both arms. H2 — under frozen
  thresholds, capped NIST2016 falls to a passing FP rate, making "cap before scoring" an
  input policy like the 48px floor.
- **Config:** lossless capped copies (PNG, LANCZOS) of exactly the 125 NIST2016 images the
  E21 runs scored — per-image before/after pairs, not population comparison. Both arms.
- **Result:**

| arm | median before → after | FP @ deployed t_ai | FP @ LOSO t_ai (NIST unseen) |
|---|---|---|---|
| CF ViT-S | −3.84 → −3.80 (no change) | 6.4% → 7.2% | — |
| **B-Free** | −0.74 → **−1.36** | 8.8% → **1.6%** | **35.2% → 8.8% — passes** |

- **H1 refuted for CF, and the refutation is mechanical:** CF's own preprocessing already
  shrinks every input (shortest edge 440), so a pre-cap is a no-op — its NIST elevation is
  about the pipeline's content, not resolution handling. **H1/H2 confirmed for B-Free**,
  and the mechanism is equally mechanical: five 504px crops of a 12 Mpx frame see ~2% of
  it; capped to 2048, the same five crops see ~30% and the score distribution drops toward
  the other authentic sources.
- **The decisive number is the LOSO one.** With NIST2016 held out of calibration entirely
  (the truly-unseen scenario that was E22's only B-Free failure at 44.4% on the evaluation
  half / 35.2% on all 125), capped scoring brings it to **8.8% — under the 10% budget.**
  Every other pipeline was already ≤5.0%. **The B-Free band now passes the cross-source
  gate on all eleven pipelines at ~65% recall** — up from CF's 28.4%, the best deployable
  configuration the project has produced.
- **Policy adopted, with its cost stated:** long side >2048px → downscale before scoring
  (B-Free arm; harmless no-op for CF). The trade: E7 taught that downscaling erases
  generation evidence, so a hypothetical >2048px *synthetic* image becomes harder to catch
  — the policy exchanges megapixel-AI detectability (rare; GPT-4K class output is already
  at chance) for megapixel-real protection (measured, was the worst failure mode). Recorded
  so the exchange is a decision, not an accident.

## 2026-08-19 — E22b: bootstrap intervals for the band's headline numbers

- **Motivation:** every band number so far comes from one deterministic split with 34–100
  evaluation images per source. Before any of them reaches the report they need honest
  uncertainty. Full-pipeline bootstrap (2,000 replicates): every population's calibration
  *and* evaluation half resampled, the worst-source threshold refit per replicate.
- **Result (point · 95% interval):**

| config | worst-source FP | macro FP | macro recall |
|---|---|---|---|
| CF ViT-S | 6.6% [2.0 … 13.2] | 1.3% [0.3 … 2.4] | 28.0% [17.8 … 34.2] |
| B-Free (no cap) | 7.9% [3.0 … **31.7**] | 2.7% [0.8 … 6.0] | 65.2% [60.6 … 68.6] |
| **B-Free + cap (deployed)** | 7.9% [4.0 … 15.9] | 4.0% [0.8 … 5.4] | **69.0%** [61.4 … 71.2] |

- **Two honest readings.** (1) The point estimates pass the 10% budget; the 95% upper
  bounds do not (13–16%). With ~50-image calibration halves per source this is expected
  arithmetic, not a method failure — the fix is more calibration data per pipeline, which
  is exactly what E24 grows. (2) The cap does more than rescue NIST: by lowering NIST's
  calibration cut it lowers the deployed threshold itself, and recall rises 65.2% → 69.0%
  for free. It also cuts the worst-FP upper tail in half (31.7 → 15.9) — the megapixel
  pipeline was the band's variance problem too.
- **Report rule adopted:** headline band numbers are quoted with their intervals from
  here on; a guarantee claimed at 95% confidence needs the interval, not the point, under
  the budget.

## 2026-08-19 — E23c: the compression column — the E12 debt, paid at the decision layer

- **Hypotheses (pre-registered):** H1 — degradation (resize 75% + 2048 cap + JPEG q50, the
  literature's social-media standard) hurts AI recall more than authentic FP: the band
  should fail SAFE. H2 — thresholds are compression-domain-specific; refitting on degraded
  calibration halves recovers the budget.
- **Config:** degraded copies of all 3,056 scored images; halves inherited from original
  paths so clean and degraded columns compare the same images. Both arms; ~35 min.
- **Result (worst FP · macro FP · macro recall on evaluation halves):**

| arm · scenario | worst FP | macro FP | recall | AUC (Defactify) |
|---|---|---|---|---|
| CF · clean reference | 6.6% | 1.3% | 28.0% | 0.882 |
| CF · degraded, frozen threshold | **0.0%** | 0.0% | 12.4% | 0.869 |
| CF · degraded, refit | 5.0% | 1.2% | 25.0% | — |
| B-Free · clean reference | 7.9% | 2.7% | 65.2% | 0.930 |
| B-Free · degraded, frozen threshold | **41.3% (NIST2016)** | 9.9% | 53.8% | 0.827 |
| B-Free · degraded, refit | 8.0% | 2.8% | 42.8% | — |

- **H1 split by arm, and the split matters.** CF fails safe exactly as predicted —
  degradation lowers all its scores, the frozen threshold turns ultra-conservative, FP
  goes to zero. **B-Free fails dangerous on the megapixel source:** q50 blocking pushes
  authentic NIST scores *up* toward "generated", and the frozen threshold accuses 41% of
  them. A compressed real photograph from the wrong pipeline is the risk case, not the
  compressed fake.
- **H2 confirmed for both.** Refitting on degraded calibration halves restores the budget
  (worst 5.0% / 8.0%) at a recall price: CF barely pays (28.0 → 25.0), B-Free pays a
  third of its recall (65.2 → 42.8). Ranking degrades likewise (0.930 → 0.827) — E12's
  compression gap, now measured end-to-end at the decision layer.
- **Design caveat:** the clean reference threshold here is fitted without the E23b cap, so
  the frozen-threshold row conflates two changes for NIST; the refit row is the clean
  claim. Single split; the E22b interval rule applies.
- **Serving consequence.** Compression regime must be part of the serving contract: bytes
  per pixel is already recorded per request, so route to a compression-matched threshold
  (clean-domain band: 69% recall; degraded-domain band: 43%) rather than pretending one
  threshold spans both. CF's robustness earns it the fallback role for heavily compressed
  input — the two arms now have complementary, *measured* domains.

## 2026-08-20 — E20 three-seed addendum: the numbers hold within seed noise

- **Config:** `--seeds 3 --arms resnet18` under protocol v2, results in
  `artifacts/e20/results_3seed.json` (the single-seed `results.json` untouched). Training
  is remarkably stable: validation AUC 0.909 ± 0.000, best epoch 6–7 in all seeds.
- **Result (top-3 aggregation, mean ± std over seeds 42/1337/2024):** Defactify evaluation
  AUC **0.751 ± 0.033**, evaluation recall **49.9% ± 6.1**, evaluation FP 8.7% ± 2.2,
  forensics macro FP 42.7% ± 1.0, **worst-source FP 86.2% ± 3.1**.
- **Conclusion:** every E20/E22 claim about our own model survives seed variance. The
  ranking sits where the single seed said (0.770 is inside the band), and the cross-source
  failure is not a seed artifact — the worst unseen pipeline is above 83% FP in *every*
  seed. Recall carries the largest variance (±6 points), which is why the report should
  quote the three-seed mean, not the best seed.

## 2026-08-20 — E24: the library promise, tested on a real phone

- **Motivation:** E22's product claim — a new real pipeline needs ~100 calibration images
  and a threshold refit, no retraining. E25 tested it on a downloaded set; this tests it
  on the most deployment-realistic pipeline available: **207 camera-original photographs
  from the project owner's iPhone** (203 × iPhone 15 Pro + 4 × iPhone 16e, EXIF-verified;
  median long side 4032px — a genuine 12-megapixel pipeline, the exact class that poisoned
  NIST2016). Screenshots and non-EXIF files excluded by audit; photos never enter the
  repo, only scores are kept. Scored on CPU overnight so the GPU stayed with the
  three-seed run.
- **Hypotheses (pre-registered):** H1 — frozen thresholds hold for CF; B-Free uncapped is
  at risk (12 Mpx) and the E23b cap contains it. H2 — adding the pipeline's calibration
  half and refitting meets the budget at little recall cost.
- **Result:**

| arm · variant | FP @ frozen threshold | FP @ refit (eval half) | macro recall after refit |
|---|---|---|---|
| CF ViT-S | **1.0%** | 1.9% (threshold unchanged) | 28.0% |
| B-Free, uncapped | **38.2%** | 12.6% | 58.6% |
| **B-Free + 2048 cap** | 12.6% | **9.7% — budget met** | **62.2%** |

- **Both hypotheses confirmed, and E23b is validated on real user data.** Uncapped B-Free
  would have accused 38% of the owner's own photographs; the cap alone cuts that to 12.6%,
  and one threshold-only refit with ~104 calibration photos brings the untouched half to
  9.7% at a three-point recall cost (65.2 → 62.2). CF passes untouched at 1.0% — its
  robustness column grows again. The deployment recipe is now measured twice, on a
  downloaded 2026 set (E25) and on a real phone (E24): **audit → cap → ~100 calibration
  images → refit → within budget.** That sentence is the product.

## 2026-08-20 — E26: the OR rule — a blind primary arm cannot veto a seeing one

- **Motivation:** live use surfaced a design fault the benchmarks had hidden. A
  ChatGPT-generated upload scored CF 4.95 (seven times its threshold, band "ai") and
  B-Free −4.41 (its documented GPT-family blindness, E25) — and the verdict was
  "insufficient", because the single-primary design let the blind arm decide alone. The
  fix candidate: any arm above its own worst-source threshold decides ("OR rule"). The
  risk: two 10% budgets need not union to 10%.
- **Measured on every cached score set (deployed thresholds, evaluation halves):**

| | single primary (old) | **OR rule** |
|---|---|---|
| worst-source FP (12 pipelines) | 9.7% | **9.7% — unchanged** |
| macro FP | — | 2.9% |
| Midjourney recall | 7% | **14%** |
| FLUX recall | 38% | **64.5%** |
| Nano Banana / Pro recall | 29% / 42.5% | **56.5% / 55.5%** |
| GPT Image 4K recall | 6.5% | 12.0% — the blind spot shrinks, does not close |
| the user's ChatGPT upload | missed | **caught (by CF)** |

  The union does not break the budget because the arms' false positives live on
  *different* sources (CF's worst: Columbia 6.6%; B-Free's: iPhone 9.7%) — training-family
  complementarity, the E8 lesson at the decision layer. **Adopted**; `combine()` in
  `verdict.py`, responses now carry `triggered_by`.
- **Gallery validation (the owner's prediction, tested):** all 207 iPhone camera
  originals through both systems. The old tile signal called **207/207 "AI"** — every
  single photo, median p 0.994 — E13's disease in its purest form, predicted by the owner
  before the run. The decision layer flags 21/207 (10.1%, the designed budget). That
  contrast — 100% → 10% — is the project's contribution in one line, measured on its
  owner's own photographs.
- **UI consequence (shipped with this entry):** the screen had been showing two
  contradictory verdicts — the research signal, with its 79–100% FP, dressed as the
  headline. Now there is exactly one verdict (the band's, with the triggering arm named),
  and the research signal sits below it in a labelled box: "araştırma sinyali — karara
  dahil değil", with its measured false-positive rate printed next to it. Verified
  end-to-end in the browser, including the originally-missed ChatGPT image (now "ai").

## 2026-08-20 — E27: our own GPT-family arm — trained behind a gate that fought back

- **Motivation:** the system's one measured capability gap — GPT Image recall 12% under the
  two-arm OR (E25/E26), the family behind the live demo miss. Goal: train OUR OWN
  specialist arm and admit it to the served ensemble only through a pre-registered gate.
- **Process, in order — because the process is the result:**
  1. **Adversarial design review before any training** (three-lens panel: shortcut /
     contamination / training design; 14 findings, 3 fatal). It rewrote the design:
     dimension-PAIR matching instead of a size cap, a pre-JPEG pass on half the AI class
     (single-vs-double compression history), FFHQ as a third real source + 200-image
     portrait-FP holdout, SHA256+dHash contamination scans, a real-vs-real CONTROL POOL
     to separate shortcut ceiling from signal, frozen-trunk ladder before any fine-tune,
     three seeds mandatory, the claim narrowed to **in-collection** recall.
  2. **Gate v1 fired exactly as designed:** metadata probe 0.992 — the aspect channel
     (all GPT images 2:3 portrait) survived my first encoder. Training was refused.
  3. **Encoder fixed** (each AI image adopts the (w,h) pair of a sampled real), pool
     rebuilt: 860 AI / 1,800 real (3 sources), zero dHash duplicates, zero cross-matches
     against probe/Defactify/forensics (19,389 files scanned).
  4. **Gate v2 passed:** metadata class-AUC **0.419** (channel closed; control 0.527);
     texture class-AUC 0.872 vs control ceiling 0.687 — the +0.185 excess is generation
     signal, not pipeline artifact (E18's positive-control method applied to a gate).
- **Arm: a logistic head on frozen CF-ViT CLS embeddings** (the ladder's first step
  sufficed; 22M-parameter fine-tuning never became necessary). Three seeds:

| seed | val AUC | probe recall | probe q75 | arm worst FP (12 srcs) | FFHQ FP | rho(size) |
|---|---|---|---|---|---|---|
| 42 | 0.995 | 37.5% | 29.0% | 4.0% | 0.0% | −0.01 |
| 1337 | 0.992 | 52.5% | 41.5% | 4.0% | 0.0% | −0.02 |
| 2024 (deployed) | 0.994 | 40.5% | 32.5% | 4.0% | 0.0% | +0.00 |

- **Union gate (exact, per-image, 12 pipelines incl. iPhone, cached e21/e23b/e24 scores):**
  two-arm baseline worst FP on these halves is 10.7% (sampling variance of the same
  deployed system; inside E22b's interval) and the three-arm union is **10.7% — the new
  arm adds zero worst-case FP**. Its contribution is pure recall: **GPT probe 12% → 40.5%**
  (in-collection, at the deployed threshold), and as a free side effect **DALL-E 3
  21% → 35%** on evaluation halves. No source is pushed beyond max(budget, baseline).
- **Integrated.** `artifacts/gpt_arm_v1.npz` + `GptFamilyArm` in `verdict.py`; the arm
  shares CF-ViT's forward pass, so the third arm costs **zero additional compute**. The
  live ChatGPT image: the arm alone scores it 7.82 (below its conservative 15.38 cut) —
  the ensemble still catches it through CF; recorded honestly.
- **Honest limits:** the recall claim is in-collection (train and probe share one Kaggle
  collection; an out-of-collection ChatGPT holdout of 30-50 hand-generated images is the
  owner's recorded TODO — gate G2b pending). q75 recompression costs ~8 points (32.5%),
  consistent with E23c's regime finding, no collapse. theminji reals' provenance remains
  unaudited upstream; FFHQ and genimage carry the diversity.
- **The one-line conclusion for the report:** the served system now contains a model we
  trained ourselves, admitted by the same gate that had rejected our earlier models —
  and the gate's v1 refusal (0.992) is the best evidence the gate is real.

## 2026-08-24 — E27 protocol correction: evaluation leaked into union threshold selection

- **Audit finding:** E27's union stage initially computed two-arm baseline and three-arm
  false positives on the evaluation halves, then increased the GPT-arm threshold in a loop
  until those evaluation results met `max(10%, baseline)`. The saved run happened not to
  enter the loop, but the algorithm made future threshold/model admission conditional on
  evaluation data and therefore violated the project's frozen-evaluation rule.
- **Correction:** `union_threshold_at_fpr()` now assigns the new arm only the false-positive
  capacity left by the frozen baseline on each **calibration half**. The strictest source
  cut is frozen; evaluation halves are measured once. A synthetic test replaces every
  evaluation arm score and proves the fitted threshold cannot change.
- **Recomputed result on the same cached images:** candidate threshold **15.38 → 21.71**;
  evaluation worst-source FP remains **10.7%** (iPhone 11/103; this is baseline sampling
  variance), macro FP **2.95%**, and the GPT arm adds zero evaluation false positives.
  But in-collection GPT-probe recall falls **40.5% → 14.5%** (q75: 32.5% → 9.0%);
  DALL-E 3 recall rises only 21% → 25%, while Midjourney stays 14%.
- **Decision:** **E27 fails its pre-registered G1 >=40% admission gate and is removed from
  serving.** The valid served scientific contract returns to E26's two-arm OR (CF-ViT by
  default; B-Free only under its explicit non-commercial opt-in). The earlier E27 entry is
  retained above as history and is superseded by this correction, not silently rewritten.

## 2026-08-24 — M4 operational folder-evaluation smoke

- **Purpose:** validate the new user-facing `pixelproof-evaluate-project` path with the real
  canonical checkpoint, not estimate generalisation from four images. The inputs were the four
  labelled demo files in the pinned local B-Free checkout (`metainfo.csv`: two real, two AI),
  copied into separate `real/bfree_demo/` and `ai/bfree_demo/` roots. They are upstream demo
  examples and are too few for a scientific performance claim.
- **Runtime:** MPS, E20-v2 seed 2024 checkpoint SHA-256
  `b9f39eda10ba3de54b706d6448b67d93ce8e4c7bae97a685f3c1b57ebfd65adf`, native 128 px tiles,
  texture floor 0.04, `top3`, stored threshold 0.9894907, maximum 256 tiles. The command wrote
  `artifacts/m4_folder_smoke/results.json` and `predictions.csv` with environment, command and
  git provenance; both files remain local because `ml/artifacts/` is intentionally ignored.
- **Result:** 4/4 decoded and scored, zero read/decode/inference failures. ROC-AUC **0.500**,
  stored-threshold recall **1.000**, false-positive rate **1.000**, accuracy **0.500**;
  TP=2, FN=0, FP=2, TN=0. Exact image scores were: real `img0000` 0.9999969,
  real `img0001` 0.9987636, AI `img0002` 0.9999561, AI `img0003` 0.9999164.
- **Interpretation:** the evaluator works end to end and preserves an uncomfortable result. Both
  authentic examples cross the 0.9894907 threshold, matching E20's already recorded cross-source
  false-positive failure. This smoke run is operational evidence for M4 and another warning
  against using the project model as an authenticity certificate; it is not a new benchmark.

## 2026-08-24 — M5 one-command local-demo verification

- **Purpose:** operational verification only. `./tools/pixelproof-demo start` must prove the
  canonical artifact, API, real inference contract and model-first web UI work together without
  manual environment variables or process management.
- **Preflight result:** Python 3.13.5, serving imports and `pip check` passed; the registry verified
  `e20-tile-resnet18-seed2024`; `pixelproof-predict` and `pixelproof-evaluate-project` were
  installed; Node v25.2.1 and `npm ls --depth=0` passed; loopback ports 8799/3000 were free.
- **Live result:** the API's project-only profile reached `status=ready`. The tracked
  `artifacts/figures/generators.png` smoke request returned score **0.2409**, stored threshold
  **0.9895**, **51 tiles** and checkpoint prefix `b9f39eda10ba...`. The command then received HTTP
  200 from the server-rendered E20 web shell at `127.0.0.1:3000`. One `Ctrl+C` shut down both
  process groups with exit code 0 and no leftover model-worker warning.
- **Interpretation:** this does not add a performance claim. It is direct evidence that the model
  can now be demonstrated from a fresh shell after documented setup, and that the smoke path is
  the same verified project-model contract used by CLI, folder evaluation and the browser.

## 2026-08-24 — M6 presentation disagreement evidence

- **Purpose:** freeze one input and both visible result layers for the internship presentation.
  This is an explanatory case study, not a metric. Input: B-Free upstream demo `img0000.png`,
  labelled `0 = real` by its `metainfo.csv`, SHA-256 `c7351aee67f37fe5acf1aa7781612b2760b90e0d56010038ec2e48ff9a79360e`,
  checkout revision `c6a9f898782fb466b29af01f21960b67415afb0e`.
- **Runtime:** PixelProof commit `95fe2b2`, full profile on MPS, real `POST /predict`, canonical E20
  hash `b9f39eda...65adf`; B-Free was not enabled as a detector, so the external comparison arm was
  the pinned MIT Community-Forensics ViT-S.
- **Result:** E20 returned **1.0000 >= 0.9895**, triggered, research-only, 69 tiles. CF-ViT returned
  **-2.4631 < 0.6617**, so E26 returned `insufficient`. The exact presentation payload and input
  provenance are committed in `evidence/demo_disagreement.json`.
- **Interpretation:** E20 is wrong on this authentic source while the external comparison does not
  trigger. The models have different representations and source populations; disagreement is
  expected under source shift. This case is presented specifically to justify separate UI cards,
  the research-only label and the prohibition on authenticity certification.

## 2026-08-24 — N0 pre-registration: source-robust project model v2

- **Observed problem:** the runnable E20 seed-2024 model has Defactify AUC 0.7197, recall 48.1%,
  Defactify authentic FP 11.3%, forensic macro FP 43.3% and worst-source FP 83.2%. The next
  experiment targets the source-specific authentic false positives; a better demo surface cannot
  repair this scientific failure.
- **Hypothesis:** constraining a newly initialized linear head to use only non-negative weights over
  E20's frozen non-negative ResNet18 features will reduce reliance on source-specific authentic
  features while retaining enough fake-associated signal to avoid an always-real solution. This is
  an independent implementation of the algorithm described in *Stay-Positive* (ICML 2025):
  <https://arxiv.org/abs/2502.07778>. The official repository is used only as provenance:
  <https://github.com/AniSundar18/AlignedForensics>. Its reviewed page did not expose an explicit
  licence, so no upstream code, weights or assets may enter this repository.
- **Frozen protocol:** train only on the existing 48,037 E20 tiles; make the validation split
  deterministic and source-stratified; freeze the backbone; reset the linear head to zero; clamp
  feature weights to `>= 0` after every optimizer update; leave bias unconstrained. Hyperparameters,
  checkpoint selection and threshold selection may not see E20 evaluation images.
- **Single-seed advancement gate (seed 2024; all required):** AUC >= 0.710, recall >= 42%,
  Defactify FP <= 15%, forensic macro FP <= 35%, worst-source FP <= 70%. Failure ends this candidate
  without evaluation-driven tuning.
- **Three-seed integration gate (only after single-seed pass):** population mean AUC >= 0.740,
  recall >= 45%, Defactify FP <= 15%, forensic macro FP <= 35%, worst-source FP <= 65%; every seed
  must remain below 75% worst-source FP. Passing this gate permits artifact/runtime integration;
  it does not permit a production or authenticity-certification claim.

## 2026-08-24 — N1 mechanical validation: independent Stay-Positive head

- **Implementation:** `pixelproof.stay_positive` loads the existing E20 ResNet18 state, freezes all
  backbone parameters, applies the stored ImageNet normalization, flattens explicit non-negative
  ReLU embeddings, zero-initializes a 512-to-1 head, trains it with BCE and AdamW, and clamps only
  feature weights to `>= 0` after every step. The bias remains unconstrained. The implementation
  uses no upstream code, weights or assets.
- **Isolation:** the installed `pixelproof-train-stay-positive` command writes a new candidate under
  `artifacts/e28/`; it does not alter E20, the artifact registry, API, web UI or served threshold.
- **Mechanical result:** five focused tests passed for zero initialization/projection, frozen
  backbone and non-negative embeddings, deterministic source+label holdout, balanced smoke
  sampling, invalid-input rejection and compatible head installation. The full Python suite passed
  **48/48**; compileall and `pip check` passed.
- **Real-checkpoint smoke:** CPU, seed 2024, balanced 120-tile subset, two head epochs, batch 32.
  The candidate reloaded into ResNet18, selected epoch 1 at validation AUC **0.9000**, and contained
  minimum feature weight **0.000000** with **zero negative weights**. This is execution evidence,
  not performance evidence; it cannot satisfy or revise N2's pre-registered full-data gate.

## 2026-08-24 — E28 / N2: Stay-Positive candidate fails the source-robustness gate

- **Training:** canonical E20 seed-2024 backbone SHA-256 `b9f39eda...65adf`, all **48,037** existing
  tiles, frozen 512-dimensional feature extractor, seed 2024, 15-epoch ceiling, batch 1024,
  AdamW lr 1e-3, source+label-stratified 90/10 training/validation. Validation alone selected epoch
  **1** at AUC **0.894699**. Minimum head feature weight was 0.0; negative count was zero. Candidate
  SHA-256: `73b8bed6...08a5`.
- **Evaluation:** unchanged E20 protocol v2, split seed 2026, 50% calibration, 10% real FP budget,
  150 Defactify real + 750 AI evaluation images and 1,776 authentic images from ten unseen forensic
  sources. Aggregation was selected by calibration macro generator recall before evaluation.

| model / rule | AUC | recall | Defactify FP | forensic macro FP | worst-source FP |
|---|---:|---:|---:|---:|---:|
| E20 seed 2024 · top3 baseline | 0.7197 | 48.1% | 11.3% | 43.3% | 83.2% |
| **E28 Stay-Positive · top3 selected** | **0.7290** | **48.9%** | **12.7%** | **44.6%** | **85.0%** |
| E28 · top10pct diagnostic | 0.6935 | 37.5% | 11.3% | 30.8% | 60.0% |
| E28 · p90 diagnostic | 0.7106 | 35.6% | 5.3% | 29.4% | 59.0% |
| E28 · mean diagnostic | 0.7135 | 37.6% | 8.7% | 28.7% | 72.0% |
| E28 · fixed16_top3 diagnostic | 0.6816 | 37.2% | 10.0% | 28.7% | 63.0% |

- **Gate:** AUC >=0.710 passed; recall >=42% passed; Defactify FP <=15% passed; macro FP <=35%
  **failed**; worst-source FP <=70% **failed** (`RealisticTampering`, 85.0%). The lower-FP
  aggregation diagnostics were not selected by the frozen calibration rule and all fell below the
  42% recall floor; choosing one after seeing evaluation would be leakage.
- **Decision:** **rejected after one seed.** Do not run seeds 42/1337, do not register the artifact,
  do not alter serving, and do not tune against these evaluation results. Exact compact evidence is
  committed as `evidence/e28_seed2024_rejection.json`; raw tile scores and candidate remain in the
  ignored local `artifacts/e28/` directory. The constraint preserved AUC/recall but did not repair
  source shift, so the next candidate must change the representation or training data rather than
  merely constraining E20's final head.

## 2026-08-24 — O0 pre-registration: representation feasibility line

- **Reason for the pivot:** E28 retained AUC/recall but worsened selected-rule macro and worst-source
  FP, so another head-only constraint on the same final E20 embedding is not justified. Evaluation
  diagnostics will not be used to retune E28.
- **Candidate class:** intermediate CLIP encoder-block representations with trainable block
  importance, motivated by RINE (ECCV 2024): <https://arxiv.org/abs/2402.19091>. The official
  repository is Apache-2.0: <https://github.com/mever-team/rine>. This entry records a feasibility
  direction, not permission to redistribute its checkpoints or their transitive base weights.
- **Ordered protocol:** first pin and audit code/checkpoint/base-weight/data licences; then build an
  isolated optional adapter and smoke; then evaluate once with E20 protocol v2. No RINE dependency,
  code, weight or serving path has been added at O0.
- **O2 feasibility gate (all required):** AUC >=0.850, recall >=35%, Defactify FP <=15%, forensic
  macro FP <=15% and worst-source FP <=30%. Only a pass may justify pre-registering and training a
  project-owned intermediate-block head; integration would still require three seeds.

## 2026-08-24 — O1 RINE feasibility/provenance audit

- **Pinned sources:** RINE `9b7fd585...620` (Apache-2.0), OpenAI CLIP `d05afc4...35f6` (MIT code),
  official RINE 4-class trainable checkpoint (25,298,182 bytes; Git blob `bf5cd405...c457`) and
  official CLIP ViT-L/14 URL-embedded SHA-256 `b8cca3fd...03836` (932,768,134 bytes).
- **Boundary:** the RINE save path explicitly excludes `clip` keys, so the small checkpoint contains
  only its trainable components. The CLIP base-weight page did not expose a separate weight licence;
  it may be fetched for local research but will not be committed or redistributed. No upstream
  training/evaluation dataset is required or authorized for O2.
- **Technical finding:** score direction is sigmoid(fake logit), evaluation is RGB center-crop 224
  plus CLIP normalization, and the backbone is frozen ViT-L/14 with hooks on intermediate `ln_2`
  blocks. The upstream recipe is not admitted to PixelProof: it uses an unpinned git dependency,
  Python 3.9/Torch 2.1/CUDA assumptions and dynamic `exec` state assignment.
- **Decision:** **conditional GO for isolated O2 only.** A strict project adapter may fetch pinned
  files into ignored storage, hash before deserialization, support CPU/MPS and leave serving locks,
  manifest, API, web and E20 unchanged. Full matrix and resource contract:
  `ml/RINE_FEASIBILITY.md`. No model/dependency/checkpoint was downloaded during O1.

## 2026-08-24 — P0 pre-registration: owner iPhone gallery input correction

- **Discovery:** the local gallery has 210 supported still-image extensions plus one MOV. Only 23
  stills passed the shared decoder; 187 were rejected because Pillow identifies iPhone two-frame
  JPEG files as `MPO`, outside the declared `JPEG`/`PNG`/`WEBP` set. A five-file probe confirmed
  failed files are RGB, two-frame MPO at ordinary iPhone dimensions while accepted files are
  one-frame JPEG or PNG.
- **Correction boundary:** admit `MPO` as a JPEG-family input and decode frame zero only. Preserve
  upload-byte, pixel, dimension, aspect, decompression-bomb, EXIF-orientation and RGB/alpha rules.
  MOV stays unsupported. This is an input compatibility fix, not a model or threshold change.
- **Frozen measurement:** after automated and real-file smoke checks, run every still once through
  E20, legacy CNN, full statistics, legacy tile statistics and the available E26 decision arm.
  The gallery is authentic-only and may measure false positives; it may not train, calibrate,
  select aggregation or change a threshold. Count exact duplicate bytes separately. Repository
  evidence will contain aggregates only—no personal image, GPS, filename or per-image hash.
- **Invalid partial result:** among the first 23 decodable files E20 triggered on 20 and legacy
  tiles on 23, while CF-ViT triggered on zero. These are a format-selected subset and are recorded
  only to explain why a complete rerun is required, not as the gallery result.

## 2026-08-24 — P1 iPhone MPO decoder correction

- **Implementation:** the shared bounded decoder admits Pillow's `MPO` identifier as a JPEG-family
  container and explicitly seeks frame zero before geometry validation, load, EXIF orientation and
  RGB conversion. It never iterates auxiliary frames. MOV remains unsupported and every existing
  upload/geometry/decompression rule is unchanged.
- **Verification:** an automated MPO-like two-frame contract test proves only frame zero is sought;
  existing malformed input, JPEG orientation, PNG alpha and resource tests remain green. Focused
  API tests passed 12/12; the full Python suite passed **50/50**, compileall and `pip check` passed.
- **Real-gallery smoke:** default decoder acceptance increased **23/210 -> 137/210**. The remaining
  73 files are no longer misreported as unsupported format; they are 5712x4284 (24.47 MP) and
  correctly receive the unchanged 16 MP product-limit response. P2 will preserve that product
  count while using an explicit 26 MP local-only ceiling to measure model behavior on all stills.

## 2026-08-24 — P2 pre-run amendment: rejected E28 gallery diagnostic

- The owner requested the new development beside every current model. Before reading any E28
  gallery score, the rejected `stay_positive_seed2024.pt` candidate is added as a diagnostic only.
  It uses N2's frozen `top3` aggregation and threshold 0.8282409708 under the same 256-tile ceiling.
- This run cannot revise E28's failed macro/worst-source gate, select a new threshold, enter the
  artifact manifest or change serving. Its sole question is whether the failed method behaves
  differently on this authentic iPhone pipeline.

## 2026-08-24 — P2: complete owner-gallery comparison

- **Accounting and privacy:** the local folder contained 210 supported still-image instances and
  one unsupported MOV. SHA-256 byte deduplication found 206 unique stills and four duplicate
  excess instances. The repository stores only these aggregates: no image, filename, EXIF/GPS or
  per-image digest entered evidence, training, calibration or threshold selection.
- **Two distinct decoder results:** the unchanged 16,000,000-pixel product policy accepted
  137/210 instances and rejected 73/210, all because 5712x4284 is 24.47 MP. The comparison then
  used the pre-declared local-only 26,000,000-pixel ceiling. All 206 unique stills decoded and all
  requested arms completed with **zero failures**. This measurement does not change API limits.
- **Authentic-only outcomes:** an AI trigger is a false positive here. Legacy three-band arms use
  AI / uncertain / real only as their historical output labels; CF-ViT is asymmetric and returns
  AI or `insufficient`, never proof of realness.

| arm | frozen decision outcome on 206 unique authentic photos | score summary |
|---|---|---|
| project E20, `top3`, threshold 0.9894907 | **178 AI / 28 below = 86.4% FP** | mean 0.9815; median 0.9998; range 0.2035–1.0000 |
| rejected E28 Stay-Positive, `top3`, threshold 0.8282410 | **170 AI / 36 below = 82.5% FP** | mean 0.8933; median 0.9127; range 0.5912–0.9840 |
| legacy ResNet-18 CNN | **100 AI / 18 uncertain / 88 real** | mean 0.5191; median 0.5615 |
| legacy full-image statistics | **134 AI / 40 uncertain / 32 real** | mean 0.7301; median 0.8668 |
| legacy tile statistics | **206 AI / 0 uncertain / 0 real = 100% FP** | mean 0.9884; median 0.9918 |
| legacy `auto` | identical to tile statistics: **206 AI** | every image exceeded its 700 px tile-selection boundary |
| external CF-ViT / E26 threshold 0.6617392 | **1 AI / 205 insufficient = 0.49% FP** | logits: mean -8.5315; median -8.8754; max 8.3387 |

- **Agreement:** E20 and CF-ViT never triggered together: 178 were E20-only, one was CF-only and
  27 triggered neither. All three legacy arms (CNN, full statistics and tile statistics) agreed on
  AI for 73/206. These disagreements are model-family/source-pipeline sensitivity, not confidence
  estimates that can be averaged into a valid probability.
- **E28 diagnostic:** candidate SHA-256
  `73b8bed630cfd125b745e986d4b24160184043a14ae3ce649f03896958ee08a5` ran on MPS with the frozen
  N2 threshold and at most 256 texture-qualified 128 px tiles (observed 11–255, median 152). Its
  82.5% FP is only eight images better than E20 and remains unusable. The earlier rejection stands;
  the candidate stays outside the manifest and serving.
- **Product conclusion:** MPO compatibility fixed a real input blocker, but none of the
  project-owned/legacy outputs is a trustworthy authenticity decision on this camera pipeline.
  CF-ViT is the strongest available comparison here, yet 205 abstentions mean this real-only run
  does not establish useful AI recall or a complete classifier. The next experiment must change
  representation and be evaluated on both authentic camera pipelines and held-out modern AI;
  threshold tuning on this gallery is forbidden.

## 2026-08-25 — E29/Q0 pre-registration: compact 2025-generator CF-ViT probe

- **Question:** at CF-ViT's frozen E24/E26 AI threshold `0.6617392`, what recall does the current
  strongest gallery arm achieve on a small, current, generator-balanced AI-only diagnostic?
- **Pinned source:** `saneval-ann/saneval-sample`, Hugging Face revision
  `e9e188f6018b3d491708f29e7a387f5043dc8841`, MIT dataset card. The source has 600 API-generated
  outputs across six commercial generators, five structured prompt types and simple/hard splits.
- **Frozen subset before scores:** exclude pre-2025 Imagen 3; retain GPT Image 1, Imagen 4, Imagen
  4 Ultra, Nano Banana and Seedream 3. For each model x prompt-type x split group, take the two
  lowest source row ids: 5 models x 5 types x 2 splits x 2 rows = exactly 100 images. No visual
  quality or detector output participates in selection.
- **Storage gate:** download only the dataset-server cached JPEG cells into ignored
  `ml/data/e29_saneval_2025/`. Preflight and enforce a strict 100,000,000-byte image total; require
  100 unique SHA-256 values, pinned revision response, declared model/group balance and successful
  shared decoding. Abort on any mismatch.
- **Representation caveat:** the dataset card describes raw PNG outputs, while the row service
  supplies JPEG cache assets. Results therefore measure current model content after one known web
  recompression and must not be presented as native-output performance or a full SANEval result.
- **Frozen report:** CF-ViT only, existing weights and preprocessing, threshold unchanged. Report
  recall and score distribution overall/per generator plus type/split diagnostics and failures.
  Because every item is AI, this probe cannot estimate FP, specificity, accuracy or AUC; no
  training, calibration, threshold choice or serving change is authorized.

## 2026-08-25 — E29/Q1 implementation checkpoint before download

- Added `experiments/e29_saneval_2025_probe.py`. It resolves all 600 source rows with bounded
  retry, rejects a changed `x-revision`, applies the frozen group selection, HEAD-preflights every
  selected JPEG and aborts above 100,000,000 bytes before downloading.
- Each downloaded cell must match HEAD length, decode as JPEG at the declared geometry and have a
  unique SHA-256. Writes use a temporary sibling followed by atomic replace; the ignored local
  manifest records row/model/type/split, file hash, total bytes and a deterministic content-set
  hash without persisting expiring cache URLs.
- The same command resolves the already-cached, hash-verified CF-ViT through the existing E21
  adapter and reads `CF_T_AI=0.6617392` from the served decision contract. It reports recall by
  model/type/split and preserves per-row local scores for audit.
- Automated selection/budget tests passed 2/2. Full Python tests passed **52/52**; compileall and
  `pip check` passed. The implementation checkpoint precedes all image download and scoring.

### Q1 network-interruption correction

The first real invocation stalled before preflight while waiting for a dataset-server row chunk;
it was interrupted after three silent 30-second polls. No image or partial dataset had been
created. The fetcher now prints each chunk boundary, uses a shorter bounded request timeout and
atomically caches each complete 100-row response. A resumed chunk is accepted only when its
revision matches and every signed asset URL has more than one hour before expiry. Completed image
files are likewise revalidated and reused after a later interruption. Focused tests, including the
new revision/completeness/expiry cache contract, passed 3/3 before retrying the network run.

## 2026-08-25 — E29/Q2 result: CF-ViT misses 81% of the compact 2025 slice

- **Dataset realization:** all 100 frozen rows downloaded as unique, decodable 1024x1024 cached
  JPEGs with zero failures. Image bytes are **11,546,660**; the whole ignored local E29 directory,
  including row caches, manifest and results, is **12,092,513 bytes**, safely below 100 MB.
  Content-set SHA-256 is
  `0e5a2452c2eac44846fb3bc0118fc6bb262db814f693f2183d489b0835c1b9be`.
- **Detector contract:** Community-Forensics ViT-S weights SHA-256
  `275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692`, authors' shortest-edge
  440 / center-crop 384 / CLIP normalization, MPS, frozen `t_ai=0.6617392`. All 100 scored with
  zero inference failures; no threshold or row changed after results.

| generator | n | AI triggers | recall | median logit |
|---|---:|---:|---:|---:|
| GPT Image 1 | 20 | 2 | **10%** | -3.7979 |
| Imagen 4 | 20 | 4 | **20%** | -3.7647 |
| Imagen 4 Ultra | 20 | 4 | **20%** | -3.4842 |
| Nano Banana | 20 | 4 | **20%** | -3.1395 |
| Seedream 3 | 20 | 5 | **25%** | -1.9572 |
| **overall** | **100** | **19** | **19%** | **-3.1943** |

- **Diagnostics, not selection:** recall by prompt type was color 10%, numeracy 15%, shape 25%,
  spatial 30% and texture 15%. Hard prompts reached 7/50 (14%) versus simple 12/50 (24%). Overall
  logits ranged -10.2364 to 8.4434 with mean -2.8989. No post-hoc threshold is substituted.
- **Interpretation:** E29 independently repeats E25's GPT-family blind spot (E25 GPT Image 4K was
  6% recall) and shows it extends across this compressed structured-prompt slice: even the best
  family reaches only 25%. Nano Banana's 20% here versus E25's 46% on a different 200-image source
  also warns that content/encoding distribution materially changes detector recall.
- **Boundary:** this is an AI-only, 20-per-model diagnostic from cached JPEGs, and the SANEval card
  itself says its 600-row sample is not for statistical inference. E29 cannot report FP,
  specificity, accuracy or AUC and does not represent native PNG performance. The defensible
  conclusion is narrow but important: CF-ViT's 0.49% false-alarm result on the owner's real gallery
  coexists with only 19% recall here, so it is not a complete detector for current generators.

## 2026-08-25 — E30/A0: five-role current-data and OOD protocol pre-registration

- **Question:** can the next candidate be judged without repeating E10's collection shortcut or
  E27's evaluation leakage, while keeping the unstable-network acquisition below a strict local
  budget? E30 separates TRAIN, CALIBRATION, DEVELOPMENT TEST, LOCKED FINAL TEST and chronological
  FUTURE TEST before source selection or scores.
- **Frozen source roles:** existing audited project data remains TRAIN; E22/E24 source scores remain
  CALIBRATION; pinned MLLMGenSet GPT Image 2/Nano Banana 2/matched-real JPEGs and capped
  LAION-Mobile reconstructions are DEVELOPMENT; pinned Qwen Image Bench 2026-generator source paths
  are the first LOCKED FINAL candidate; FUTURE stays empty until a post-cutoff release exists.
  The exposed owner gallery remains development regression only.
- **Frozen gates:** working-v1 requires real macro FP <=5%, worst-source point FP <=10%, current-AI
  macro recall >=50%, every generator/protocol >=30%, and q75/resize recall loss <=15 points.
  Per-source exact 95% intervals and abstention coverage are mandatory; 40 examples is a minimum
  gate cell and 5–10 is scout-only. A new native multi-phone vault is required before any universal
  real-photo claim.
- **Acquisition boundary:** low-bandwidth development target <=30 MB (hard 40 MB); Qwen scout hard
  70 MB. Rows are deterministic and score-blind, revisions/licences are pinned, third-party bytes
  are ignored, and every derivative inherits its parent role.

## 2026-08-25 — E30/A1: role/data contract implemented before image download

- **Pinned registry:** MLLMGenSet `1498eead...b9de`, Qwen Image Bench `d2493deb...7038` and
  LAION-Mobile `0c60f598...3465`, including upstream total sizes, licence boundaries, intended
  role and representation caveats.
- **Mechanical contract:** explicit label direction and five-role validation; safe relative paths;
  one revision per source; AI generator / real pipeline requirement; derived-parent role, label and
  content inheritance; exact SHA and underlying-content cross-role leak rejection; role-gated
  loading; deterministic stratification; per-file/total byte gates; SHA-256/dHash; metadata-only
  geometry/format/compression probe; immutable locked-final receipt.
- **Network contract:** exact source paths freeze before bytes, Range-capable partial files resume,
  a non-Range server restarts safely, streams cannot exceed declared/remaining bytes, verified
  files and JSON manifests replace atomically, and LAION URL replacements must match upstream
  hashes.
- **Verification:** 12 focused tests passed, including role violation, parent/content leakage,
  deterministic cells, hard byte failure, perfect metadata shortcut detection, immutable final
  receipt and interrupted-download resume. Full Python suite **65/65**, compileall and `pip check`
  passed. No E30 image was downloaded or scored before this checkpoint.

### A1 network-client correction before acquisition

The first `download-mllm` call failed before an HTTP response because Hugging Face Hub 1.25 exposes
an httpx client whose `request()` does not accept requests-style `stream=True`. No image or partial
file existed. The adapter now opens httpx streams through `build_request`/`send`, accepts both
`iter_bytes` and requests-compatible `iter_content`, closes the response and keeps the same Range
resume/byte ceiling. The frozen 180 paths and selection SHA `f71c8d02...035e` did not change.

## 2026-08-25 — E30/A2 result: development battery realized; LAION arm honestly incomplete

- **MLLM realization:** the frozen nine cells downloaded exactly 180/180 unique, decodable JPEGs:
  60 GPT Image 2, 60 Nano Banana 2 and 60 matched real, each split 20/20/20 over hybrid,
  structure and texture. Parent bytes are **4,419,610**; selection SHA-256 is
  `f71c8d02709b311308017f7bb1b30acc859e543579cf36df6a26ee0b0a88035e`; content-set SHA-256 is
  `1f3a733340582045225684ae49679d547035b5e1a995e05faf1e44dfe6b4df2e`.
- **Transport battery:** every parent produced deterministic q90, q75, q50 and resize256-q90
  children in the same DEVELOPMENT role and with the same content id. The 720 children occupy
  **14,029,255 B**; all 900 parent/child files have unique byte hashes. Combined content-set
  SHA-256 is `7634755c75f855064e0d6c3c2731c6fe63d6582f2b6065654698adbf32e924b8`.
- **Shortcut gate:** metadata-only AUC is 0.6238 on parents, then 0.6096 / 0.6191 / 0.6362 /
  0.6127 for q90 / q75 / q50 / resize256-q90. All are below the pre-registered 0.65 ceiling;
  formats are JPEG on both labels and no audit issue fired. This permits model evaluation but
  does not prove semantic independence or native-output performance.
- **LAION interruption and retry:** the first preflight stopped at 1/10 when connectivity failed.
  The corrected runner now persists incomplete selections and per-pipeline/failure diagnostics
  instead of throwing away the evidence. On retry, 55/80 rows passed unchanged requirements:
  10 each for iPhone 11, 11 Pro, XS and XR; 9 SM-G930F; 5 SM-G950F; 1 SM-G935F; 0 Redmi Note 4.
- **Why no partial download:** among 361 rejected URLs, 287 exceeded 375 KB, 33 lacked a declared
  length, 40 failed HTTP/network preflight and one returned non-image content. Even the ten
  smallest reachable candidates per frozen group total about **45.96 MB**, versus the arm's
  30 MB ceiling; combined with MLLM they exceed the 40 MB development stop. The source is recorded
  `source_incomplete`, no LAION image was downloaded and no replacement cell was introduced.
- **Acceptance boundary:** realized development image bytes are **18,448,865**, all from MLLM.
  `evidence/e30_development_realization.json` is the compact committed record. No detector has
  read E30 rows yet; A3 must seal the independent Qwen paths before any A4 development score.

## 2026-08-25 — E30/A3 seal: Qwen rows fixed before bytes or scores

- The first tree-listing attempt used the general Hugging Face paginator and was stopped after
  90 seconds without a response. It created neither image bytes nor a selection. The acquisition
  path now requests only the first 20 lexically ordered entries from the pinned directory endpoint,
  with a 15-second attempt timeout and bounded retries; selection still takes numeric rows 1–5.
- Repository metadata falsified the planned all-PNG assumption before download: the frozen paths
  contain **21 PNG and 19 JPEG** files. The tool now accepts declared image formats, retains each
  original suffix and marks transport `native_source`; it does not transcode during acquisition.
- Exactly **40 rows / 37,907,745 declared bytes** are frozen: five each for GPT Image 2,
  Nano Banana 2.0, Seedream 5.0, Qwen Image 2.0 Pro, FLUX.2 Max, FLUX.2 Pro, GLM-Image and
  HunyuanImage 3.0. Selection SHA-256 is
  `50e3fec166c900365145854bfe5183764bbb8d655149d81c524dcbff18901eeb`.
- `evidence/e30_qwen_sealed_selection.json` commits every source path, generator, prompt id and
  declared byte count while `detector_scored=false`. No image was downloaded before this seal.
  This remains a 5-per-generator scout; it cannot satisfy the >=40-per-cell reporting gate.

### A3 realization after the committed seal

- The post-seal acquisition downloaded and decoded **40/40** source files with zero failure and
  exact declared total **37,907,745 B**, safely below 70 MB. There are 21 PNG and 19 JPEG parents,
  five per generator, with 40 unique SHA-256 values. Native content-set SHA-256 is
  `0f25bfe73cf6fb7e06015fdc36d16044d554352f917fa1d7012f24815f3638a1`.
- A deterministic RGB JPEG q90 child was generated for every parent without network use. The 40
  children occupy **9,449,715 B**, inherit the same LOCKED FINAL role/content id and point to their
  parent. All 80 hashes are unique; combined content-set SHA-256 is
  `93dcbc01e517eaa61e693c4753a72e8d69136b0105c9c36cb8353c6ad98b749c`.
- This AI-only arm cannot support a metadata-only real-vs-AI shortcut AUC. Count, format, decode,
  role inheritance, exact-byte and uniqueness gates passed. `detector_scored=false` remains true;
  no locked image was inspected or inferred during acquisition.

## 2026-08-25 — E30/A4 implementation checkpoint before DEVELOPMENT scores

- Added `experiments/e30_development_benchmark.py` before running either model. It refuses any
  manifest other than the frozen 900-row DEVELOPMENT content set and has no code path to the Qwen
  LOCKED FINAL directory.
- E20 loads only through the verified project artifact registry, including checkpoint-owned 128 px
  tiling, texture floor, top-k aggregation and threshold. CF-ViT requires the pinned upstream
  revision, cached weights SHA-256 `275ba982...1692`, authors' processor and unchanged
  `t_ai=0.6617392`.
- Per-row JSONL scores are ignored, contract-hash bound, append-and-fsync resumable and keyed by
  record id. Each image byte hash is rechecked before inference. The aggregate includes full
  accounting, ROC-AUC, FP, recall, Clopper-Pearson 95% intervals, macro/worst transport-group
  rates, robustness deltas and abstention semantics.
- Synthetic exact-interval and transport-accounting tests plus the existing E30 acquisition tests
  passed 12/12. This checkpoint contains no E30 detector result.

## 2026-08-25 — E30/A4 DEVELOPMENT result: neither existing arm earns final access

- **Accounting:** E20 and CF-ViT each scored all 900 transport views of the 180 MLLM parents with
  zero decode/inference failure. Raw rows remain ignored and resumable. Qwen LOCKED FINAL was not
  opened or scored.
- **Repeated-measure boundary:** the 900 rows are five views of 180 underlying contents. Pooled
  rates and AUC are descriptive; they do not receive falsely narrow binomial intervals. Exact 95%
  Clopper-Pearson intervals are reported per independent transport (60 real / 120 AI) and per
  20-item transport x source cell.

| frozen arm / view | real FP | current-AI recall | ROC-AUC |
|---|---:|---:|---:|
| E20, standardized parent | 8/60 = 13.33% (95% CI 5.94–24.59) | 15/120 = 12.50% (7.17–19.78) | 0.468 |
| E20, all five views (descriptive) | 28/300 = 9.33% | 46/600 = 7.67% | 0.463 |
| CF-ViT, standardized parent | 0/60 = 0% (upper 95% 5.96) | 2/120 = 1.67% (0.20–5.89) | 0.616 |
| CF-ViT, all five views (descriptive) | 0/300 = 0% | 6/600 = 1.00% | 0.575 |

- **Generator/regime diagnosis:** across repeated views E20 recalls GPT Image 2 at 9.67% and Nano
  Banana 2 at 5.67%; CF-ViT reaches 2.00% and 0%. E20's AI signal is concentrated in texture
  (20%) while hybrid is 0% and, critically, matched-real texture false alarms reach 24%. CF-ViT
  reaches only 0%/1%/2% on hybrid/structure/texture and 0% FP in all three real regimes.
- **Transport robustness:** E20 standardized recall 12.5% rises to 15% at q90, then falls to 5%
  at q75, 0.83% at q50 and 5% after resize256-q90. CF-ViT is 1.67%, 2.5%, 0.83%, 0%, 0%.
  Compression-loss limits are moot because the undegraded operating points already fail recall.
- **Decision:** both candidates return to TRAIN/DEVELOPMENT. The 20-item cells are below the formal
  >=40 gate size, so no universal pass/fail claim is made; nevertheless their aggregate point
  recall is far below the 50% target and worst AI cells are 0%. Scoring the Qwen set would consume
  the locked final without a viable candidate, so it remains untouched. Threshold retuning on
  MLLM or Qwen is forbidden.
- **Evidence:** `evidence/e30_development_benchmark.json` contains contracts, hashes, counts,
  exact per-transport intervals, generator/regime breakdowns and rejection rationale.

## 2026-08-25 — E31/B0-B1: attached-disk audit before TRAIN v2

- **Question:** should the attached ~255 GiB collection justify retraining E20 or building a
  multi-model verdict now? The pre-registered answer is conditional: identical E20 retraining is a
  no-op because E20 already used the corrected E19 labels and three seeds were stable; fusion is
  admissible only after independently useful, complementary representations exist.
- **Pre-existing ensemble evidence:** E9's eight fixed ResNet/feature blends improved best AUC by
  only 0.002. On E30 DEVELOPMENT, E20 and CF-ViT have zero overlapping positive decisions, but
  their OR still yields only 52/600 descriptive AI detections and 28/300 real false alarms. This
  rejects “connect every checkpoint” as a candidate, not heterogeneous fusion as a future method.
- **Tool/contract:** `experiments/e31_ssd_audit.py` requires an explicit source root, rejects an
  output inside that root, ignores exFAT `._*` and cache files, verifies declared Hugging Face
  label order, counts all Parquet rows/generator fields, and samples first/middle/last shards
  deterministically. It commits aggregate JSON only. Missing disks and label drift fail clearly.
- **Inventory:** 10 registered sources occupy **173,576,436,217 B** and seven additional
  inventory-only sources **97,337,151,271 B**, for **270,913,587,488 B** total. Registered Parquet
  metadata covers **603,991 rows**. CommunityForensics-Small contains 44,884 rows—11,972 AI,
  32,912 real—and **300 distinct AI model names**, so the old 228-generator/local-coverage gap is
  closed at inventory level.
- **Bounded image evidence:** 300 rows per registered source / **3,000 total** decoded with zero
  failure. Exact sampled bytes had zero overlap with **980** protected E30 parent/derived hashes.
  This is a diagnostic sample, not full decontamination; B2 must hash every selected TRAIN-v2 row.

| paired source | native metadata AUC / issue | fixed 128 RGB-JPEG probe | E31 use |
|---|---:|---:|---|
| CommunityForensics-Small | **1.000** | 0.636 | native reject; conditional TRAIN v2 |
| AI-vs-Real-balanced | 0.549; format sets differ | 0.586 | fixed representation preferred |
| AIGC benchmark | **0.967** | 0.540 | native reject; conditional TRAIN v2 |
| ai-vs-real-200k | **0.841** | 0.552 | native reject; conditional TRAIN v2 |
| Julien Lucas modern test | **0.974** | 0.560 | remains test-only; native pooled claim unsafe |

- **Interpretation:** the user's data concern is supported, but the defect is not simply “too few
  images.” Large sources encode class in geometry/format, and a model can exploit it before
  learning generation traces. Identical fixed input removes the measured metadata separation but
  does not prove pixel-level compression/collection cues are gone. B2 therefore freezes a
  source-capped, group-disjoint TRAIN v2; B3 screens heterogeneous frozen representations before
  any expensive fine-tune; B4 fits fusion on out-of-fold CALIBRATION only.
- **Interrupted-run record:** the first audit spread 600 samples over every shard and was stopped
  after exposing Parquet row-group amplification. A nominal one-row read could decompress a
  4.09 GB CommunityForensics shard. A 12-shard revision remained needlessly expensive and was also
  stopped. The committed first/middle/last three-shard rule retains range coverage with bounded
  I/O; six focused tests pin this behavior and the complete run then finished locally.
- **Evidence:** `evidence/e31_ssd_audit.json`, SHA-256
  `2f7399bed965a8a428b4180aab059405fbcc4d4aa4d3754a5295ee4e97021f29`. No source byte was
  written or committed; E30 Qwen LOCKED FINAL remains unscored and no E31 training has started.

## 2026-08-25 — E31/B2 selection freeze: 11,300 rows fixed before bytes

- **Selection-only checkpoint:** `e31_train_v2.py freeze` reads label/generator metadata and
  Parquet row counts but never the image column. It pins each source by a digest over relative
  shard path, byte count, row count and schema. Missing/changed shards invalidate realization.
- **Composition:** 5,650 AI / 5,650 real. CommunityForensics supplies 2,400 AI (exactly 8 from
  each of 300 generator ids) and 2,400 real; AI-vs-Real-balanced supplies 2,000 AI / 3,250 real;
  current AI-only sources supply 500 Flux, 500 Nano Banana and 250 Nano Banana Pro. AIGC and 200k
  are deferred rather than adding redundant volume before this candidate is measured.
- **Role split:** 383 whole groups are stably ranked within source and round-robin assigned to five
  folds; fold 0 is CALIBRATION. TRAIN has 4,456 AI / 4,105 real and CALIBRATION 1,194 AI / 1,545
  real. Every source appears in both roles and no generator/shard group crosses them. The first
  dry rule used raw hash modulo and happened to place all seven Flux shards in TRAIN; it was
  corrected before byte access or commit because that left no Flux calibration evidence.
- **Frozen identity:** selection SHA
  `5907c14ba3e173c125c024a30658fb8e7e56788a469614808ad4ef5519a5fbfb`; complete deterministic
  row contract in `evidence/e31_train_v2_selection.json`. The report itself has SHA-256
  `59f95563da578c8274518ae0394b00064bd1b0109ad652077a68ad3967ff5620`.
- **Next gate:** realization must reproduce source/selection hashes, decode all 11,300 rows, reject
  any exact or dHash match against E30, Defactify, real-pipeline calibration, owner gallery and
  named test-only data, then create exactly one seeded native 128 px texture-qualified tile per
  parent. No model or embedding may read a selected image before this checkpoint is committed.

### B2 realization protection amendment before selected-byte access

The realization implementation originally hashed loose protected folders and E30 manifests but
could not inspect test-only images embedded in Parquet. Before running it, protection was extended
to stream every image in the Julien Lucas modern set and the separate CommunityForensics real/fake
test Parquets. The six focused tests include an embedded-image protection fixture. This change does
not alter selection SHA `5907c14b...bfb`; it only makes the post-freeze rejection gate stricter.

### B2 first realization rejected at the mechanical input gate

The committed 11,300-row selection decoded with zero corrupt-image failure but **3,534 rows** did
not yield a native 128 px texture-qualified tile. The run refused to silently drop them and wrote
no tile archive: post-selection loss would alter class/source proportions after the contract was
frozen. This repeats E19's known small-image floor at a larger, source-aware selection.

The next action is pre-registered as a data eligibility correction, not model-driven resampling:
stream all 143,070 AI-vs-Real-balanced rows once, record only keys that decode, meet both 128 px
dimensions and pass the unchanged 0.04 texture-floor tile rule, then freeze a new 11,300-row
selection from that eligible set. The old selection and failure count remain in the log. Seven
focused tests pin eligibility-set hashing before the scan runs.

### B2 balanced eligibility result and selection v2 freeze

- **Complete source result:** of 71,535 rows per class, 47,233 AI and 50,000 real are below 128 px;
  24,301 AI and 21,532 real pass decode/size/texture. One AI and three real rows are large enough
  but below the unchanged texture threshold. Eligible-set SHA is
  `91089e227821fb6e4dcdd06487c7943958afccb7a6e281d1f94718881bff1eb2`.
- **Selection v2:** exact composition, source caps, 383 groups and fold roles are unchanged. Of
  11,300 rows, 7,767 remain and 3,533 balanced rows are replaced by the next deterministic eligible
  candidates. Selection SHA is
  `5355e4307eb72053a01fcfc3c13e2a431feed7a313a316317fed4303bd2679b2`; evidence-file SHA is
  `594ca2cbec7450372c2c2876b5913c542e6dd251c9e4166e12da05b2b11891a1`.
- **Boundary:** eligibility contains no model score and cannot tune a test result. Selection v1
  remains committed as the rejected input. Selection v2 is committed before its image bytes are
  realized. The one-row difference between 3,534 total v1 rejects and 3,533 balanced replacements
  is left for the second exact realization to identify rather than guessed away.

Before selection-v2 byte access, rejection handling was made evidentiary: a failed realization now
writes a compact ignored receipt with record ids/reasons and protected-scope counts while still
refusing the tile archive. This prevents another 15-minute scan from ending with only a traceback;
it does not relax any gate or alter selection v2.

### B2 selection-v2 realization rejected by protected-content evidence

Selection v2 tiled 11,299/11,300 rows; one TRAIN Nano Banana Pro row remained too flat. More
importantly, the full protected library found **74 exact** and **83 dHash** overlaps (74 shared,
nine dHash-only). Exact hits comprise 8 balanced AI, 10 Flux, 10 Nano Banana and 46 Nano Banana Pro
rows; additional dHash-only hits affect one balanced real, six Flux and two Nano Banana rows. The
tile archive was refused. Receipt SHA is `55364ed2...b9c8` under ignored `ml/data/e31/`.

Rather than replacing only these 84 selected ids and risking another test collision, the next
pre-registered command builds one protected mechanical eligibility set over every row of balanced,
Flux, Nano Banana and Nano Banana Pro. It reuses the identical 176,961-exact / 172,087-dHash
protected scope and rejects only decode/input failures or protected content. CommunityForensics is
not reselected because v2 measured zero failure and zero overlap there. Nine focused tests pass
before this screen reads candidate bytes.

### B2 protected candidate screen and selection-v3 freeze

The full screen completed on 2026-08-26 without internet access. Across 163,777 candidates it
accepted 65,650 and rejected 97,982 exact protected matches, 137 additional dHash-only matches and
six texture-floor failures. Eligible counts remain comfortably above every frozen cap: balanced
24,133 AI / 21,528 real, Flux 9,709 AI, Nano Banana 9,232 AI and Nano Banana Pro 1,048 AI. The
detailed ignored receipt has SHA `16ff5f14...bad10`; aggregate committed evidence has SHA
`e1a3f811...122ff`.

Selection v3 preserves 11,300 parents, 5,650 per label, 383 indivisible groups, 303 AI identities
and the existing role/source counts. It keeps 11,216 v2 ids and replaces exactly 84: nine balanced,
16 Flux, 12 Nano Banana and 47 Nano Banana Pro rows. The 4,800 CommunityForensics ids are unchanged.
Selection SHA is `1a3a5c98c4b0614a0af4bd1bc65ca4fbb8ea33404dbb6a2db53b2da17b79df2e`;
evidence-file SHA is `c6748b12270f12298a2723bb2c338a9698a0d0bbdc05cb56e258575b8c20d98c`.
No image score influenced screening or selection. The next independent realization must still
reproduce zero rejection before any representation may train.

### B2 selection-v3 realization accepted

The independent byte pass reproduced selection SHA `1a3a5c98...df2e` and realized all
**11,300/11,300** rows into native 128 px tiles. Counts remain 5,650 AI / 5,650 real and 8,561 TRAIN
/ 2,739 CALIBRATION. Decode failures, size/texture rejects, exact protected overlaps and dHash
protected overlaps are all **zero**; all 11,300 tile hashes are unique. The 395,082,960-byte ignored
NPZ has SHA `508330c2...9f2b`, while the detailed receipt has SHA `340d593c...09dc`. Compact evidence
is committed at `evidence/e31_train_v2_realization_v3.json` with SHA `5bff123c...619d`. B2 is
accepted; B3 representations may now train, but no model result or E30 score exists yet.

## 2026-08-26 — E31/B3: source-aware representation ladder

- **Protocol:** accepted 11,300-tile SHA `508330c2...9f2b`; fold 1–4 TRAIN out-of-fold threshold
  evidence; untouched 2,739-row CALIBRATION; <=5% source-macro / <=10% worst-source real FP; >=50%
  current-AI macro / >=30% weakest current-source recall. E30 remained unopened.
- **Arms:** unchanged E20 control; cached timm DINOv2 ViT-S/14 frozen at 224 px with a balanced
  logistic head; existing 68 forensic/frequency features with the same head.

| arm | AUC | macro real FP | worst real FP | current AI macro recall | weakest current source |
|---|---:|---:|---:|---:|---:|
| E20 control | 0.960 | 4.49% | 6.70% | 84.49% | 74.0% |
| **frozen DINOv2** | **0.966** | 4.67% | 6.70% | **90.72%** | **84.0%** |
| 68 forensic features | 0.849 | 4.24% | 5.51% | 56.24% | 44.0% |

DINOv2 recalls Flux 91.67%, Nano Banana 96.49% and Nano Banana Pro 84.0%, materially improving
over E20's 84.72% / 94.74% / 74.0%. Seeds 42, 2024 and 2026 yield identical metrics because the
standardized logistic head converges to the same convex solution; this is a zero-width three-run
interval, not three independent data samples. DINOv2 advances. The feature arm clears the absolute
floor but is far below the control; B4 may retain it only if cached row scores demonstrate useful
incremental true positives inside the FP budget. Evidence: `evidence/e31_b3_representation_screen.json`.

## 2026-08-26 — E31/B4: cross-fitted ensemble gate

Row-level complementarity exists but is too expensive in real false positives. At B3 thresholds,
E20 recovers 12 of DINO's 24 current-AI misses while adding 50 real false positives; the 68-feature
arm recovers eight and adds 42. Five source-stratified group meta-folds then cross-fit coefficients
and thresholds for the frozen small rule set.

| rule | current AI macro recall | gain vs DINO | macro / worst real FP | paired group-bootstrap gain 95% |
|---|---:|---:|---:|---:|
| DINO single | 90.72% | — | 5.04% / 6.99% | — |
| DINO+E20 max | **93.77%** | **+3.05** | 5.34% / 8.04% | +1.87 to +4.20 |
| DINO+E20 stack | 92.90% | +2.18 | 5.13% / 7.74% | +1.59 to +3.05 |
| DINO+R2 max | 92.58% | +1.86 | 5.08% / 7.29% | +0.87 to +2.84 |
| DINO+R2 stack | 91.62% | +0.90 | 5.02% / 7.29% | +0.67 to +1.13 |

No fusion reaches the pre-registered +5-point gain and every fusion slightly exceeds the 5% macro
FP budget under fold transfer. The ensemble is rejected; this is a measured trade-off, not an
assumption. The packaged winner is single DINOv2 at final full-CALIBRATION threshold
`0.7090073824`; artifact SHA `99901219...4d860` includes the linear head and pinned encoder
contract. Compact evidence: `evidence/e31_b4_ensemble.json`. E30 remains unopened.

## 2026-08-26 — E31/B5: frozen MLLM DEVELOPMENT gate

The frozen single-DINO candidate scored the existing 900-row DEVELOPMENT contract once. It
accounted for 897 rows; three `resize256_q90` views were too flat for the unchanged tile floor.

| metric | frozen gate | result | decision |
|---|---:|---:|---|
| macro real FP | <=5% | **83.63%** | fail |
| worst real-group FP | <=10% | **100%** | fail |
| current AI macro recall | >=50% | 80.67% | pass |
| worst AI-group recall | >=30% | 60.0% | pass |
| q75 recall loss | <=15 points | 2.50 points | pass |
| resize recall loss | <=15 points | gain 4.76 points | pass |
| ROC-AUC | diagnostic | **0.385** | inverted ranking |

The failure is present before heavy degradation: standardized-JPEG real FP is 81.67% and AUC
0.387. Real scores have median 0.994 versus AI median 0.977. A diagnostic DEVELOPMENT-label oracle
must raise the threshold to 0.999986 to meet both real budgets, at which point macro AI recall is
0.33% and the worst AI group is 0%. This threshold was not adopted and no retry is allowed: the
representation is inverted on independent real content, not merely miscalibrated.

The candidate is technically executable and detects GPT Image 2 / Nano Banana 2 at about 80.6%,
but is unsafe as a real/AI detector. B5 fails; Qwen LOCKED FINAL remains unscored by construction.
Compact evidence: `evidence/e31_b5_development.json`.

### B6 operational hand-off

Serving was intentionally left unchanged. The research-only folder CLI loaded the hash-pinned E31
candidate and cached DINO weights on CPU, scored scikit-learn's bundled real `china.jpg` once and
returned 0.999826 / threshold 0.709007 (`ai_signal_detected`). This false positive agrees with the
DEVELOPMENT failure and validates that the CLI warning—not a deployment claim—is necessary. Its
JSON state is `rejected_for_serving_after_E30_DEVELOPMENT`; under-threshold and error cases use
`insufficient_evidence`, never “real.”

## 2026-08-26 — E32/C1a: authentic acquisition freeze

- **Hypothesis:** replacing E31's narrow real distribution requires a device/scene-groupable
  authentic pool; downloading volume before roles, licences and parent identity are frozen would
  make the next score uninterpretable.
- **Frozen sources:** 3,500 VISION native parents / 35 devices; three FODB archives totalling
  22,940,347,533 declared bytes / 3,851 expected originals; one 17,588,803,163-byte CSAFE Galaxy
  S21 archive, not the full 132.7 GB collection.
- **Controls:** owner gallery absent; `0=REAL, 1=AI`; VISION derivatives excluded; FODB derivatives
  inherit scene/device parents; CSAFE rows remain unselected; explicit TLS, `.partial` resume,
  retry and 100 GiB free-space floor.
- **Result:** selection metadata verified and frozen; zero image bytes and zero model scores. The
  1,166,007-byte detailed receipt SHA is `200a7aeb...ca4d`; eight focused tests pass.
- **Decision:** proceed to the frozen transfer, then decode and decontaminate. This is not a data or
  model acceptance result.

## 2026-08-26 — E32/C2a: modern-AI physical/provenance inventory

- **Hypothesis:** the attached disk may already satisfy the 10–20K / five-family goal, but only
  decoded image counts with explicit generator provenance and dataset licences may count.
- **Method:** complete file/byte, Parquet-row/schema, loose image/sidecar and ZIP central-directory
  inventory; pinned Hugging Face revisions and cards; no image decode, score or role change.
- **Corrections:** GPT holding is 1,060 PNG, not 2,122 images; Nano-Banana-150k ZIP has 127,835
  image members despite its >150k claim.
- **Result:** three admissible modern families—Nano Banana 9,457 rows (MIT), Nano Banana Pro 200
  PNG (MIT), GPT Image 1 partial 1,060 PNG (CC BY 4.0). FLUX 10,000, second NBP 1,250 and Nano
  editing 127,835 remain conditional because dataset licence/provenance/count contracts fail.
- **Decision:** C2 pool freeze stops at a two-family gap. Research two licensed modern families;
  protected test sets cannot be reassigned and ambiguous sources cannot be counted.

## 2026-08-26 — E32/C2b: licensed modern-family gap freeze

- **Hypothesis:** Qwen Image 2512 and FLUX.2 Klein 9B can close C2a's two-family gap with explicit
  provenance and licensing, but selection must cap each family and keep prompt variants grouped.
- **Sources:** pinned `46849cd…` Qwen / CC BY-SA 4.0 and `c07dd3c…` FLUX.2 / CC BY 4.0. Upstream
  counts reproduce 3,936/984 and 4,072/1,018 generated images/prompt groups; 160 FLUX reference
  inputs are excluded.
- **Selection:** category round-robin, 750 complete prompt groups × four = 3,000 images per source;
  7,108,445,821 Qwen + 4,400,537,141 FLUX native JXL bytes. Detailed SHA `b871004f...8ecc`.
- **Result:** gap selection is frozen with zero image bytes and five focused tests passing.
- **Decision:** run one JXL decoder smoke per source. Bulk download is forbidden until both pass;
  prompt groups remain indivisible and Qwen LOCKED FINAL remains untouched.

### C2b decoder-smoke result

- **Authorized bytes:** one selected image+prompt per source only. Qwen 2,579,073 B, SHA
  `288eaa...3965`; FLUX 1,215,314 B, SHA `f9d7dc...8890`.
- **Decode:** direct Pillow success; Qwen RGB 1328x1328, FLUX RGB 1024x1024.
- **Unexpected result:** both `.jxl` paths contain PNG payloads. Extension metadata is false; no
  JXL dependency or conversion is required.
- **Receipt version:** unchanged asset selection gained expected dimensions, so detailed SHA
  `b871004f...8ecc` is superseded by `e9c3d3da...af7a`.
- **Decision:** decoder gate passes and bulk may proceed, mechanically tied to the new SHA. Actual
  format is decoded from bytes; identical REAL/AI input normalization remains mandatory.

## 2026-08-26 — E32/C1-C2: source-realization gate implemented before bulk completion

- **Hypothesis:** successful transfer and a plausible folder name are insufficient evidence that
  a source is eligible; roles must remain impossible until complete byte-level validation passes.
- **Method:** selection-SHA binding; exact expected-file/byte/partial checks; full Pillow decode;
  payload format, dimensions, mode, EXIF, bytes/pixel, SHA-256 and dHash inventory; comparison with
  protected E30 manifests and earlier passed E32 source reports.
- **AI-specific contract:** four image plus four prompt members per frozen group, matching non-empty
  UTF-8 prompt text, expected dimensions and byte-derived format. A Qwen live sidecar sample had
  four identical prompt hashes as expected.
- **Verification:** 18 focused C1/C2 tests pass. A PNG payload named `.jxl` is correctly recorded as
  PNG; a missing group member produces `source_realization_rejected_no_role_assignment`.
- **Decision:** tooling is accepted, source data are not. Detailed production receipts will be
  written to the external E32 audit directory only after each transfer completes; a pass means
  `candidate_only` and cannot assign TRAIN/CALIBRATION.

## 2026-08-26 — E32/C2c: nominal 15K AI allocation preregistration

- **Metadata result:** Nano Banana 9,457 unique ids / uniform declared PNG RGB 1024²;
  CommunityForensics 11,972 AI rows across 300 AI model identities plus 32,912 excluded REAL rows;
  licensed NBP 200 PNG; GPT local 1,060 matched image/prompt pairs of 4,000 upstream.
- **Frozen allocation:** Qwen 3,000; FLUX.2 3,000; Nano Banana 3,000; GPT Image 1 3,000; NBP 200;
  CommunityForensics AI 2,800 = exactly 15,000 parents.
- **Caps/families:** no source exceeds 20%; Qwen, FLUX.2, Nano Banana, GPT Image 1 and NBP satisfy
  the five verified modern-family requirement. CommunityForensics remains a non-counting diversity
  anchor and will be sampled across model identity rather than by row order.
- **Stop rule:** failure to obtain the pinned GPT source's missing 1,940 selected pairs requires a
  documented source revision/replacement; it cannot be repaired by protected-test leakage or a
  larger share from another family.

### C2c exact-selection implementation checkpoint

- **Local dry run:** Qwen 3,000; FLUX.2 3,000; Nano 3,000/9,457 with fingerprint
  `65dfa0a3...ee60f`; NBP 200/200 with `fdbe22b1...6c58`; Community 2,800/11,972 AI across all 300
  model identities, max ten/model, fingerprint `375b8b66...e8055`.
- **Selection controls:** stable hash rather than row/download order for Nano and GPT; model-level
  round-robin for Community; inherited four-output groups for Qwen/FLUX; no image decode or score.
- **GPT stop:** repository metadata was unreachable once due to a connection reset on the active
  mobile link. No receipt was written. Exact revision/licence/4,000-pair verification remains a
  hard prerequisite, so the local partial checkout cannot bias the chosen 3,000.
- **Verification:** 26 E32 tests pass, including deterministic-order and local-availability tests.

### C2c exact-freeze result

- **GPT gate:** pinned SHA `bba366cb...4825`, CC-BY-4.0 tag, 4,000 PNG and 4,000 TXT files all
  reproduced from upstream metadata.
- **Receipt:** 15,000 exact parents; detailed 4,752,567 B / SHA `3230f026...80b7`; normalized
  record-selection SHA `2a31e792...0ef7`; inherited gap-selection SHA `e9c3d3da...af7a`.
- **Availability correction:** deterministic GPT selection overlaps 795 local pairs and requires
  2,205 downloads. The earlier 1,940 figure was only the raw 3,000-minus-1,060 volume gap and is
  superseded for transfer planning, not silently rewritten as a selection result.
- **Decision:** exact metadata freeze passes with zero image bytes downloaded. Proceed to the
  selected GPT transfer and per-source realization; all records remain role-free candidates.

### C2c GPT acquisition gate — implemented before selected bytes

- Recompute the 15K selection hash and reuse exact-size local/E32 assets; never select from
  availability or overwrite the original checkout.
- Download only missing selected pairs below the E32 root with TLS, retry, free-space floor and
  atomic `.partial` resume.
- Require one deterministic missing image/prompt pair to decode and contain non-empty UTF-8 before
  bulk; bind the gate to the current selection SHA.
- Implementation tests pass; this checkpoint downloaded zero GPT image bytes.

### C2c GPT decoder/prompt smoke result

- Selected missing pair: `GPTIMG_852.png`; 3,486,339 B; SHA `8f30398f...6e96`; direct Pillow
  decode as RGB PNG 1024x1536.
- Prompt: 1,341 B, valid non-empty UTF-8, normalized SHA `e4f291e3...09c28`.
- Gate binding: 15K record-selection SHA `2a31e792...0ef7`.
- **Decision:** smoke passes; bulk transfer is permitted, but eligibility still requires all 3,000
  selected pairs and the later byte/decontamination audit.

### C2c local-pool realization implementation checkpoint

- Nano/Community: revalidate selected source fingerprint and read actual embedded bytes at exact
  frozen Parquet shard/row locators.
- NBP: require every selected loose image and declared byte count.
- GPT: resolve each frozen image/prompt pair from original local or isolated E32 storage; reject
  partial, wrong-size, invalid/empty prompt or decode failure.
- Shared gate: payload format/geometry, SHA-256, dHash, within-source duplicate, protected E30 and
  previously passed E32 peer overlap; no role assignment.
- Verification: 34 E32 tests pass, including a real temporary Parquet byte-decode fixture. No
  production local-pool row was opened by this implementation checkpoint.

### C2c Nano Banana Pro full realization

- **Input:** all 200 frozen licensed NBP parents, selection `2a31e792...0ef7`.
- **Decode:** 200/200 PNG; 136 RGB, 64 RGBA; no EXIF.
- **Hygiene:** 200 unique SHA-256 and dHash; zero within-source duplicates; zero exact/dHash overlap
  with four protected E30 manifests (980 exact / 382 perceptual hashes).
- **Receipt:** external 91,762 B / SHA `bfc217f0...d17b`; compact evidence committed.
- **Decision:** source passes as `candidate_only`. Alpha/mode will be normalized identically across
  classes before any representation sees it; no role is assigned here.

### C2c Nano first realization attempt — infrastructure stop, no result

- All 3,000 selected Parquet images reached decode, but final peer comparison encountered a binary
  exFAT `._*.json` AppleDouble receipt and raised `UnicodeDecodeError` before evidence write.
- This is not a Nano data verdict. No compact/detailed Nano result was accepted.
- Fix: exclude AppleDouble receipts and tolerate undecodable/non-JSON filesystem debris; regression
  test added. Rerun the unchanged frozen selection after committing the fix.

### C2c Nano rerun — dHash collision reveals an audit false positive

- **Decode/hashes:** 3,000/3,000 decoded, 3,000 unique SHA-256, zero protected or passed-peer hits.
- **Old-gate failure:** five unrelated images share dHash `0f0f0f0f0f0f0f0f` solely through a
  dark-edge/bright-centre composition; bounded visual audit confirms different subjects.
- **Independent check:** their 64-bit DCT-pHash pair distances are 24–32 bits.
- **Schema-v2 rule frozen before rerun:** exact SHA is definitive; dHash creates a candidate; DCT
  pHash distance <=5 confirms a perceptual duplicate. Legacy protected E30 dHash remains a hard
  exclusion because no legacy pHash exists.
- **Decision:** old receipt remains rejected; selected rows and labels do not change. Commit the
  method correction, then rerun independently. E32 tests pass 36/36.

### C2c Nano schema-v2 realization — pass

- 3,000/3,000 selected rows decode as RGB PNG; 3,000 unique SHA-256 and 3,000 unique pHash.
- Zero exact duplicate, zero confirmed perceptual duplicate, zero protected or passed-peer overlap.
- The five-row equal-dHash bucket remains reported as one candidate collision; pHash distances
  24–32 prevent false rejection under the precommitted <=5 rule.
- External receipt 1,767,170 B / SHA `8cb04e52...fe2f`; compact evidence committed.
- **Decision:** Nano passes as `candidate_only`; no model role is assigned.

### C2c Qwen full realization — reject intact source, mechanical repair required

- 3,000/3,000 RGB PNG and 750/750 prompt groups decode; zero protected or passed-peer overlap.
- Eight exact duplicate groups: all four variants of `composition_00038` duplicate
  `architecture_00058`, and `composition_00039` duplicates `architecture_00059`.
- One additional confirmed pair exists within `style_00053` variants 1/2.
- Result: 2,992 unique SHA, 2,990 unique pHash; external receipt 2,020,166 B / SHA
  `fbdc34d4...ad57`.
- **Decision:** preserve rejection. Future eligibility overlay drops the two composition groups and
  entire style group (12 rows) and may only trim—not add—rows to restore source caps.

### C2c FLUX.2 full realization — reject intact source

- 3,000/3,000 RGB PNG and 750 groups decode; zero protected or passed-peer overlap.
- 2,964 unique SHA / 2,932 unique pHash; 28 exact and 41 confirmed perceptual duplicate groups.
- Combined duplicate graph touches 98 images / 32 prompt groups, concentrated in
  `diffusiondb_orig` repeated prompts and editing variants.
- External receipt 2,045,961 B / SHA `53c0793b...1451`; compact rejection committed.
- **Decision:** preserve rejection. Later overlay keeps deterministic canonical groups across
  cross-group conflicts, drops internally duplicated groups, and cannot add unseen replacements.

### C2c GPT first full realization — reject, diagnose prompt encoding

- Transfer completed all 3,000 selected image/prompt pairs: 6,000 assets, 1,703 reused and 4,297
  downloaded under the frozen selection.
- UTF-8-only prompt validation rejects 107 sidecars; 2,893 RGB PNG images therefore reach hashes.
- Realized subset: 2,893 unique SHA, 2,887 unique pHash, five confirmed perceptual duplicate pairs,
  zero protected or passed-peer overlap.
- Byte diagnosis: all 107 failures decode as Windows-1252; observed non-ASCII characters are em
  dashes, curly quotes, `é` and `ç` only.
- External rejected receipt 1,792,420 B / SHA `9ce487a2...5184`.
- **Decision:** preserve this rejection, precommit a UTF-8-first/Windows-1252-fallback decoder with
  tests, then rerun the unchanged selection. Duplicate rows remain visible for the later overlay.

### C1 VISION full realization — reject intact source, keep balanced core

- 3,500/3,500 RGB JPEG parents decode with EXIF; every one of 35 camera pipelines contributes
  exactly 100 images.
- 3,500 unique SHA, 3,497 unique pHash; four dHash buckets contain three confirmed perceptual pairs.
- Zero protected or passed-peer overlap.
- External rejected receipt 1,939,155 B / SHA `3312c774...e6b1`.
- **Decision:** preserve rejection. Later receipt-bound eligibility overlay keeps one stable
  canonical row from each pair and excludes losers without replacement or role assignment.

### C2c GPT prompt-decoder method checkpoint

- Decoder order is fixed: UTF-8, then Windows-1252; there is no open-ended codec guessing or
  replacement-character recovery.
- Every accepted GPT record preserves original prompt-byte SHA, normalized UTF-8 text SHA and the
  selected encoding; aggregate evidence publishes encoding counts.
- Regression coverage proves a valid UTF-8 prompt remains UTF-8, typographic Windows-1252 text
  decodes under the fallback, and undefined byte `0x81` still fails.
- Verification: 20 focused E32 tests pass.
- **Decision:** commit the method independently, then rerun all 3,000 frozen GPT pairs.

### C2c CommunityForensics diversity-anchor realization — pass

- 2,800/2,800 embedded images decode as RGB PNG.
- All 2,800 SHA, dHash and pHash values are unique; zero protected/passed-peer overlap.
- All 300 frozen model identities remain represented at nine or ten rows each.
- External receipt 1,980,274 B / SHA `cb4bffe2...76b2`.
- **Decision:** pass as `candidate_only`; assign no TRAIN/CALIBRATION role.

### C2c Nano Banana Pro schema-v2 refresh — pass

- 200/200 PNG (136 RGB, 64 RGBA), with 200 unique SHA/dHash/pHash and zero overlap.
- External schema-v2 receipt 98,924 B / SHA `55ec23ec...eb8e` supersedes the earlier schema-v1
  artifact at the same evidence path; HISTORY retains both results.
- **Decision:** remains `candidate_only`; selection and role are unchanged.

### C2c GPT full realization after encoding fix — reject intact source

- 3,000/3,000 RGB PNG and prompts realize; all 3,000 image SHA values are unique.
- Prompt encoding is measured rather than guessed: 2,893 UTF-8 and 107 Windows-1252, with original
  byte and normalized text hashes retained.
- Zero protected/passed-peer overlap; six dHash buckets are six confirmed perceptual duplicate
  pairs, leaving 2,993 unique pHash values.
- External receipt 2,239,691 B / SHA `48945f7f...73d5`.
- **Decision:** preserve intact-source rejection. Later overlay keeps a deterministic canonical row
  per pair and excludes six losers without replacement.

### C2 eligibility-overlay method checkpoint

- Validate exact selection/audit row-set equality and bind every audit receipt SHA.
- Recompute exact SHA and dHash+pHash components globally across AI and VISION, including sources
  rejected by their own intact-source gate.
- Preserve parent groups; keep one stable same-label unit, remove all REAL/AI ambiguous units, and
  apply an exact <=20% AI source cap with stable-hash trimming.
- Verification: 32 focused E32 tests pass; production receipts were not opened by this checkpoint.
- **Decision:** commit method first, then freeze the production role-free overlay independently.

### C2 production eligibility overlay — freeze succeeds

- Inputs: 15,000 AI + 3,500 VISION realized rows, immutable selection SHAs and seven audit receipts.
- Global result: 59 duplicate row components; 13 internal-parent exclusions, 20 noncanonical
  same-label units, zero REAL/AI ambiguous component.
- Eligible AI 14,786: Qwen 2,956; FLUX.2 2,916; Nano 2,957; GPT 2,957; NBP 200; Community 2,800.
  Maximum share 19.998647%; Qwen/FLUX four-row groups remain intact.
- Eligible VISION: 3,497/3,500.
- External overlay 913,980 B / SHA `b6c2101f...32e4`.
- **Decision:** freeze as `eligibility_frozen_role_free`; do not train until remaining REAL sources
  are acquired/audited and source/device-disjoint TRAIN/CALIBRATION roles are separately frozen.

### C1 archive safety/inventory method checkpoint

- Reject ZIP traversal/absolute/backslash paths, symlinks, encryption, duplicate names, CRC errors,
  declared-size drift, oversized members and >4:1 expansion.
- FODB contract requires 3,851 parents / 27 roots and one `orig` plus five parent-linked transports;
  extraction is atomic and restricted to `orig`.
- CSAFE repeats published MD5 and freezes hierarchy only; no internal row is selected.
- Verification: 15 focused tests pass; no production member was extracted.
- **Decision:** commit method before running it on completed archive transfers.

### C1 FODB realization method checkpoint

- Bind external extraction receipt to its compact SHA; reject missing, partial, size-changed or
  SHA-changed original parents.
- Decode and record format/geometry/EXIF, camera pipeline, device, scene and native state; apply
  shared SHA/dHash/pHash and protected-role checks.
- Verification: 16 focused archive/realization tests pass; production originals remain unopened.
- **Decision:** commit method before extraction, then run it only after archive inventory passes.

### C1 FODB first production inventory — safe stop

- Device roots conform, but part03 contains 4,004 extra JPEG / 2,834,597,196 B under
  `inspection/`: 3,861 device-check plus 143 scene-comparison helpers.
- These are derived inspection artifacts, not independent authentic parents; counting them would
  inflate and contaminate REAL.
- No receipt was emitted and no member extracted.
- **Decision:** precommit an exact-root exclusion with published excluded counts/bytes, retain hard
  failure for every other unknown path, then rerun from untouched archives.

### C1 FODB inspection-root repair checkpoint

- Exclude exactly top-level `inspection` from parent parsing and device-root counts; expose its
  member/root/byte totals in evidence.
- All other unknown roots and nonconforming device members still fail closed.
- Verification: 17 focused archive/realization tests pass; production rerun remains separate.

### C1 FODB production archive inventory — pass

- Three archives / 22,940,347,533 B pass CRC, SHA, path, symlink, encryption, duplicate and
  expansion gates; SHAs `c719cac3...517c`, `271e07da...e5f1`, `a3c2d69f...2a6d`.
- Exact parent contract: 3,851 parents, 27 camera pipelines, 143 scene groups, six linked transports.
- Explicit nonparents: 4,004 `inspection` JPEG / 2,834,597,196 B.
- External inventory 5,356,810 B / SHA `d378573f...9631`.
- **Decision:** pass inventory; commit receipt before extracting only `orig` members.

### C1 FODB original-only extraction — complete, role-free

- 3,851 JPEG parents / 15,416,129,383 B extracted atomically; per-file SHA/device/scene retained.
- Social transports and `inspection` helpers were not extracted.
- External receipt 1,311,414 B / SHA `a1626b0b...8b05`, bound to inventory `d378573f...9631`.
- **Decision:** commit extraction receipt, then run the independent full realization audit.

### C1 FODB full realization — pass

- 3,851/3,851 RGB JPEG with EXIF; 27 pipelines / 143 scene groups.
- 3,851 unique SHA; seven same-scene cross-camera dHash buckets, zero pHash-confirmed duplicate.
- Zero protected or passed-peer overlap.
- External audit 2,588,737 B / SHA `dcbf8b55...fd11`.
- **Decision:** pass as `candidate_only`; freeze roles only with scene/device-disjoint grouping.

### C2 FODB global-overlay extension checkpoint

- Bind FODB extraction receipt SHA/state and require exact equality with its schema-v2 audit.
- Preserve one parent unit per camera original while carrying scene metadata in the source audit.
- Verification: 13 focused overlay/realization tests pass; production overlay remains unchanged.
- **Decision:** commit method, then independently recompute across 22,351 selected rows.

### C2 global overlay with FODB — pass

- 22,351 selected rows compared globally; adding FODB creates zero new duplicate component and zero
  REAL/AI ambiguity.
- AI eligible subset remains bit-for-bit 14,786; REAL becomes 7,348 (VISION 3,497 + FODB 3,851).
- External overlay 1,179,329 B / SHA `510e94eb...fc3b`.
- **Decision:** freeze role-free result; continue CSAFE acquisition to reach the REAL floor.

### C1 CSAFE four-range recovery checkpoint

- Preserve the contiguous prefix; split only missing bytes into four exact resumable ranges.
- Require HTTP 206 and exact `Content-Range`/length; assemble beside the prefix and promote only
  after full published MD5.
- Verification: 19 focused acquisition/archive tests pass, including MD5-failure prefix survival.
- **Decision:** commit recovery method, then run it independently on the stalled production prefix.

### C1 CSAFE production range recovery — pass

- Preserved prefix: 4,723,834,880 B; fetched four exact remaining ranges.
- Final archive: 17,588,803,163 B; MD5 `5c5f79e3e508a5cbf7a19e75846091d8` matches publication.
- Prefix/range temporaries were removed only after atomic promotion.
- **Decision:** transfer passes; commit result, then inventory ZIP before selecting internal rows.

### C1 CSAFE production archive inventory — pass

- 17,588,803,163 B / ZIP SHA `54a7193c...25df`; safety/CRC/MD5 gates pass.
- 7,996 JPEG, ten S21 physical devices (798–800 each), four lens pipelines.
- Content contract: 4,000 `blank` flat fields + 3,996 `natural`; neither selected yet.
- External inventory 1,306,218 B / SHA `77a88649...fd8d`.
- **Decision:** commit inventory, explicitly exclude blank fields, then freeze natural-only rows.

### C1 CSAFE natural-only method checkpoint

- Fail closed outside ten device IDs, `blank|natural`, front/telephoto/ultra/wide and JPEG.
- Freeze exact natural metadata before bytes; extraction checks size/CRC and writes atomic SHA
  receipts; realization binds receipt and records device/lens/content.
- Verification: 23 focused archive/realization tests pass; no production member selected/opened.
- **Decision:** commit method, then freeze the 3,996-row production selection independently.

### C1 CSAFE natural metadata selection — freeze succeeds

- Selected all 3,996 natural JPEG; excluded all 4,000 blank fields without opening member bytes.
- Ten devices contribute 398–400 rows; lens counts front 998, telephoto 998, ultra 1,000, wide 1,000.
- External selection 1,193,310 B / SHA `3a24bd50...ad1c`.
- **Decision:** commit exact selection, then extract only these rows independently.

### C1 CSAFE natural-only extraction — complete, role-free

- 3,996 selected JPEG / 13,219,178,988 B extracted atomically with SHA/device/lens metadata.
- Zero blank members extracted.
- External receipt 1,775,854 B / SHA `32acdfb3...d7e4`, bound to selection `3a24bd50...ad1c`.
- **Decision:** commit extraction receipt, then run independent full realization.

### C1 CSAFE full realization — pass

- 3,996/3,996 RGB JPEG with EXIF; ten devices / four lens pipelines.
- 3,996 unique SHA and pHash; one dHash bucket, zero confirmed perceptual duplicate.
- Zero protected or passed-peer overlap.
- External audit 2,521,737 B / SHA `3ea951ec...b701`.
- **Decision:** pass as `candidate_only`; add to global overlay before role freezing.

### C2 CSAFE global-overlay extension checkpoint

- Bind exact natural-extraction receipt state/SHA and require exact schema-v2 audit row equality.
- Preserve each natural image as one camera parent and add no row outside the frozen selection.
- Verification: 14 focused overlay/realization tests pass; production overlay remains unchanged.
- **Decision:** commit method, then independently recompute across 26,347 selected rows.

### C2 final global eligibility overlay — C1/C2 volume gates pass

- 26,347 selected parents compared globally: 15,000 AI + 11,347 REAL.
- Eligible AI remains exactly 14,786; eligible REAL reaches 11,344: VISION 3,497, FODB 3,851,
  CSAFE 3,996.
- CSAFE adds zero duplicate component and zero cross-label ambiguity. Global components remain 59;
  exclusion reasons remain 20 same-label noncanonical units and 13 within-parent rows.
- Detailed overlay 1,431,190 B / SHA `45830283...78b6`; state
  `eligibility_frozen_role_free`.
- **Decision:** C1/C2 candidate-volume and source-diversity gates pass. Freeze this pool; create a
  group-aware TRAIN/CALIBRATION manifest before any feature extraction or model fitting.

## 2026-08-26 — E32/C3: balanced role-manifest preregistration

- **Hypothesis:** a balanced 22,688-parent pool with group-aware CALIBRATION will give a more honest
  first DINOv2-S screen than training over all 26,130 eligible rows with source/class imbalance.
- **Selection:** all 11,344 REAL; AI Qwen 2,232, FLUX.2 2,232, Nano 2,227, GPT 2,227, NBP 200,
  Community 2,226. Stable metadata hashes only; Qwen/FLUX four-output parents stay indivisible.
- **Roles:** target 20% CALIBRATION per source. Disjoint units are VISION/CSAFE device, FODB scene,
  Qwen/FLUX prompt, Community model identity and individual parent for Nano/GPT/NBP.
- **Known constraint:** FODB scenes cross all devices, so simultaneous scene/device disjointness
  collapses to one connected collection. Choose scene-disjointness and forbid an unseen-camera
  claim from this arm.
- **Stop rule:** fail on changed overlay/audit bindings, count mismatch, duplicate record IDs,
  group leakage, empty source-role cells or any DEVELOPMENT/LOCKED reference. Do not open pixels.
- **Decision:** commit this contract before implementing or running the role freezer.

### C3 role-freezer method checkpoint

- Implementation binds overlay size/SHA and all audit SHA receipts before selecting metadata.
- Exact AI downselection preserves Qwen/FLUX prompt units; deterministic subset assignment targets
  20% CALIBRATION while keeping each source's declared device/scene/prompt/generator group intact.
- Hard failures cover impossible counts, duplicate identities, empty source-role cells and group
  leakage. Detailed output is external; compact evidence contains hashes and aggregate checks.
- Verification: eight focused role-manifest/overlay tests pass.
- **Decision:** commit method before running it on the 26,130 eligible production records.

### C3 production balanced role manifest — pass

- Total 22,688, exactly 11,344 AI / 11,344 REAL.
- TRAIN 18,154 (AI 9,081 / REAL 9,073); CALIBRATION 4,534 (AI 2,263 / REAL 2,271).
- All nine source-role cells are nonempty; declared role-group overlap is zero; no DEVELOPMENT or
  LOCKED row exists.
- Record-list SHA `568e8e26...d887`; state `train_calibration_manifest_frozen`.
- **Decision:** freeze roles. Next run identical decoded-RGB preprocessing and the R0 frozen
  DINOv2-S control; threshold selection may read CALIBRATION, not protected final arms.

## 2026-08-26 — E32/C4-R0: runnable DINOv2-S preregistration

- **Input:** C3 parents only; EXIF transpose -> RGB -> short-side 256 -> centered 224 crop -> JPEG
  q90, 4:4:4 for both labels. Persist/hash one derived view per parent externally.
- **Representation:** cached `vit_small_patch14_dinov2.lvd142m`, frozen final embedding.
- **Head:** StandardScaler + class-weighted LogisticRegression fitted on TRAIN. C grid
  {0.01, 0.1, 1, 10}; CALIBRATION AUC chooses C, smaller value wins a tie.
- **Threshold:** lowest CALIBRATION threshold with authentic macro FP <=10% and worst-source FP
  <=20%; report AUC/AP/recalls/balanced accuracy/F1 and source-level FP/FN.
- **Gate:** AUC >=0.85, current-family macro recall >=60%, weakest sized current family >=40%,
  authentic macro FP <=10%, worst-source FP <=20%. This screen is source-stratified/group-held-out,
  not unseen-source final evidence.
- **Stop:** input/feature hash mismatch, unreadable parent, role count mismatch or accidental
  DEVELOPMENT/LOCKED access aborts; no silent row loss.
- **Decision:** commit before implementing input realization or feature fitting.

### C4-R0 implementation checkpoint

- The resumable input realizer resolves loose and Parquet-backed rows, rechecks original SHA,
  applies the identical 224-pixel JPEG contract and atomically records every derived SHA.
- The trainer binds the complete input receipt, verifies each file again, extracts record-aligned
  frozen DINOv2-S embeddings, evaluates the fixed C grid and emits a hash-bound joblib artifact.
- Verification: 13 focused role/input/trainer tests pass, including fixed RGB/geometry, deterministic
  transform, group selection and per-authentic-source threshold budgets.
- **Decision:** commit method; production input realization is the next separate action.

### C4-R0 standardized-input realization — pass

- 22,688/22,688 parent inputs; 11,344 AI / 11,344 REAL.
- TRAIN 18,154 / CALIBRATION 4,534; all nine source counts reproduce C3 exactly.
- Fixed 224x224 RGB JPEG q90/4:4:4; 487,845,683 logical bytes.
- Record SHA `f9424d3d...f14b`; detailed receipt 9,021,504 B / SHA `2255b123...5199`.
- **Decision:** freeze input receipt; DINOv2-S may now extract features from this root only.

### C4-R0 frozen DINOv2-S screen — pass

- Feature matrix 22,688x384; archive 33,439,283 B / SHA `716df956...be3b`.
- C-grid CAL AUC: 0.01=0.996305, 0.1=0.996404, 1=0.995980, 10=0.995866; choose C=0.1.
- Frozen threshold 0.141444 at authentic macro FP 9.97% / worst-source FP 13.84%.
- CAL: AUC 0.996404; AP 0.996769; AI recall 99.07%; REAL recall 90.14%; balanced accuracy
  94.60%; F1 94.82%.
- AI recalls: Qwen 99.77%, FLUX.2 99.55%, GPT 100%, Nano 98.88%, NBP 97.5%, Community 97.30%.
- REAL FP: CSAFE 4.63%, FODB 13.84%, VISION 11.44%.
- All five screen checks pass. Artifact 12,720 B / SHA `7f170340...a85e`.
- **Decision:** accept as a runnable candidate/control. Do not call it final until unseen-source and
  independent real-photo arms pass.

### C4-R0 serving/gallery preregistration

- Implement one-image/batch CLI with hard artifact/weight hashes and exact R0 preprocessing.
- Unit-test preprocessing and decisions before real image use.
- Score 210 supported owner-gallery stills; exclude one MOV. Gallery is already-consumed
  DEVELOPMENT, so report FP distribution only and forbid refit/threshold change.
- **Decision:** commit this boundary before implementing or opening gallery pixels.

### C4-R0 inference implementation checkpoint

- Added `pixelproof-predict-e32`: artifact/weight SHA checks, exact standardized input, batch MPS/CPU
  scoring and JSON results.
- Added aggregate-only owner-gallery DEVELOPMENT runner; no model/head/threshold mutation exists.
- Verification: ten focused candidate/input/trainer tests pass.
- **Decision:** commit method, then score the 210 supported gallery stills once.

### C4-R0 owner-real DEVELOPMENT smoke — fail, no refit

- 210 supported authentic stills scored; one MOV excluded.
- Fixed threshold 0.141444: 159 false positives, REAL recall 24.29%.
- Score median 0.6806; p90 0.9941; maximum 0.9999975.
- This sharply disagrees with CALIBRATION REAL recall 90.14% and proves authentic source/pipeline
  shift. It is not a threshold-selection dataset and no parameter changed.
- **Decision:** preserve runnable artifact but reject advancement to serving/LOCKED FINAL. Use the
  failure to redesign source-held-out validation and representation/data coverage.

### C4-R0 final engineering verification

- Hash-verified CLI scored `IMG_8540.jpeg` at 0.699661 against threshold 0.141444 (false AI).
- Full suite: 174 Python tests pass; six web tests pass; production web build and TypeScript
  typecheck pass. One existing Starlette/httpx deprecation warning remains.
- **Conclusion:** implementation is runnable and reproducible; model generalization is the failed
  layer. Keep E32 out of the service while planning source-held-out correction.

### C4-R0 LOCO postmortem preregistration

- Reuse only frozen R0 features and roles; hold one complete source out of fit and thresholding.
- Fit C=0.1 on remaining TRAIN; threshold on remaining CALIBRATION at macro REAL FP <=10% and
  worst-source FP <=20%; score every row from the absent source.
- Report FP for held-out REAL collections and recall for held-out AI sources across all nine arms.
- No owner-gallery row is opened and no accepted artifact field may change.
- **Decision:** commit before implementing or reading LOCO results.

### C4-R0 LOCO implementation checkpoint

- Nine-arm runner binds input receipt `2255b123...5199` and feature archive
  `716df956...be3b`, excludes one source from fit/threshold and uses the original FP budgets.
- Five focused LOCO/threshold tests pass.
- **Decision:** commit method before production diagnostic.

### C4-R0 nine-source LOCO postmortem — complete

- Held-out AI: macro recall 98.34%, worst 95.78% (Community); current families 97.67–99.50%.
- Held-out REAL FP: CSAFE 15.74%, VISION 19.82%, FODB 34.85%; macro 23.47%, worst 34.85%.
- Accepted artifact and gallery evidence unchanged; owner gallery was not opened.
- **Conclusion:** modern-generator representation transfers, authentic pipelines do not. Prioritize
  diverse/matched REAL coverage plus REAL-source-held-out gates before richer encoders or ensemble.

### C4 REAL-complement triage and R1a preregistration

- Reject CF-Small REAL: 32,912 rows are all FFHQ, not a diverse camera complement.
- Reject local `34data` REAL and `theminji` REAL for E32 enrollment: no adequate upstream
  provenance/licence contract; local bytes do not override that boundary.
- R1a: exact R0 inputs/roles, pinned Community-Forensics ViT-S frozen CLS, class-weighted logistic
  C grid {0.01, 0.1, 1, 10}, same CAL authentic FP budgets and screen metrics.
- No owner-gallery access during feature/head selection. A passing artifact must be frozen before a
  separately committed gallery stress.
- **Decision:** commit before implementation or CF feature extraction.

### C4-R1a implementation checkpoint

- Pinned revision `ac6ee457...db00`, weight SHA `275ba982...1692`; record/input SHA checks precede
  frozen CLS extraction.
- Separate feature cache/artifact; unchanged C grid, roles, threshold budgets and screen gate.
- Four focused CF-head/threshold tests pass.
- **Decision:** commit method before production extraction.

### C4-R1a frozen CF-ViT CLS screen — pass

- Feature matrix 22,688x384; 33,436,875 B / SHA `c170a1f6...bc6b`.
- C-grid AUC: 0.01=0.998222, 0.1=0.997986, 1=0.997487, 10=0.997285; choose C=0.01.
- CAL: AUC 0.998222, AP 0.998351, AI recall 99.91%, REAL recall 90.05%, balanced accuracy 94.98%,
  F1 95.20%, macro/worst REAL FP 9.97%/12.77%.
- Current-family macro/worst recall 99.95%/99.77%; all screen gates pass.
- Artifact 12,703 B / SHA `6288acba...d670`.
- **Decision:** freeze artifact. Owner gallery may be opened only by a separately committed,
  refit-free inference stress.

### C4-R1a owner-gallery stress preregistration

- Hard-bind artifact `6288acba...d670`, CF weights `275ba982...1692`, standardized JPEG round-trip
  and threshold 0.118110.
- Score 210 supported owner stills once; exclude MOV; no refit/recalibration.
- Compare REAL recall with R0 24.29% and historical frozen-CF 99.51% (206 unique images).
- **Decision:** commit before implementing scorer or reopening gallery pixels.

### C4-R1a inference implementation checkpoint

- Added `pixelproof-predict-e32-cf` with hard artifact/revision/weight identities and the frozen
  0.118110 threshold; input is the exact R0 JPEG round-trip followed by the official CF processor.
- Added a separate aggregate-only owner-gallery DEVELOPMENT runner. It records gallery identity,
  score distribution and highest-scoring basenames but has no fit or threshold-selection path.
- Verification: four focused R1a/R0 candidate and trainer tests pass; gallery pixels remain closed.
- **Decision:** commit method before the single frozen owner-gallery run.

### C4-R1a owner-real DEVELOPMENT smoke — fail, no refit

- 210 supported authentic stills scored; one MOV excluded; gallery identity
  `390e3c21...ac09` matches R0 exactly.
- Fixed threshold 0.118110: 154 false positives and 26.67% REAL recall, only +2.38 points over
  R0's 24.29%. Median score 0.4892; p90 0.9600; maximum 0.99839.
- Evidence: `evidence/e32_r1a_owner_gallery_smoke.json`, SHA `2e242ef5...b3a`; artifact unchanged.
- **Decision:** reject R1a from serving/LOCKED advancement. Two distinct encoders now reproduce the
  same authentic-source collapse, so add licensed camera-source diversity and require a complete
  REAL-source holdout before spending on R2/R3 or ensembles.
- Engineering verification: hash-checked single-image CLI reproduced; 178 Python tests, six web
  tests, production build and TypeScript typecheck pass.

### C4-R1b corrective data preregistration

- TRAIN/CAL candidate: official CSAFE MCSIDB `iPhone14.zip`, 20,428,338,922 B, published MD5
  `dfc01c89...946c`, CC BY 4.0. Inventory first; only frozen natural parents may advance.
- Absent-source DEVELOPMENT: IPN-NFID v3's twelve linked device articles, exactly 960 natural JPEGs
  / 3,889,897,594 B, CC BY 4.0. It cannot fit any data/model/threshold/policy choice.
- Acquisition must bind official API ids, versions, licence, filenames, sizes and MD5s; use
  resumable partials and retain >=100 GiB free. Archive traversal/CRC/decode/decontamination gates
  precede role assignment.
- R1b may freeze only after iPhone rows are grouped and balanced; owner/IPN pixels stay closed until
  artifact freeze. External pass requires <=20% IPN worst-device FP, <=20% owner FP and >=90%
  current-AI recall; no test-derived recalibration.
- **Decision:** commit before implementing acquisition or downloading selected bytes.

### C4-R1b acquisition implementation checkpoint

- Metadata freeze and selected-byte transfer are separate commands. Official Figshare id/version/
  licence plus every selected size/MD5 are hard-bound.
- IPN uses four bounded resumable workers; CSAFE uses one resumable stream. Both preserve partials,
  verify before atomic promotion and enforce >=100 GiB free.
- Four focused tests cover natural-only selection, device binding and checksum/size drift stops.
- **Decision:** commit method before production metadata freeze or selected-byte transfer.

### C4-R1b production metadata freeze — pass

- IPN: 960/960 natural JPEGs, twelve devices, 3,889,897,594 declared bytes.
- CSAFE: `iPhone14.zip`, 20,428,338,922 bytes, MD5 `dfc01c89...946c`.
- Detailed receipt: 385,191 B / SHA `c807d140...1c7f`; selected bytes downloaded by freeze: zero.
- **Decision:** freeze receipt; transfers may now resume only against these rows.

### C4-R1b iPhone 14 range-recovery preregistration

- Preserve the 92,159,662-byte serial prefix; do not restart or overwrite it.
- Divide `[92,159,662, 20,428,338,922)` into four disjoint exhaustive ranges. Require status 206,
  exact Content-Range and exact part length; resume each part independently.
- Assemble into a distinct temporary, verify 20,428,338,922 bytes and MD5 `dfc01c89...946c`, then
  atomically promote. Delete prefix/ranges only after verified promotion; preserve all on failure.
- **Decision:** commit this recovery boundary before implementation while IPN continues separately.

Append-only correction: post-stop `stat` is 92,274,688 bytes. The two occurrences of 92,159,662
above are transcription errors; production range planning binds the filesystem value.

### C4-R1b iPhone 14 range implementation checkpoint

- Exhaustive/disjoint four-range planner plus strict 206/Content-Range/length checks implemented.
- Each range resumes independently; assembly is separate and promotes only after exact total+MD5.
- Eighteen combined R1b and original acquisition tests pass.
- **Decision:** commit implementation before launching production ranges.

### C4-R1b IPN transfer and audit preregistration

- Transfer passes: 960/960 files, twelve devices, 3,889,897,594 bytes; every published MD5 passed.
- Audit must bind frozen selection+download receipt, decode every file, retain shared scene groups,
  reject exact/protected/passed-peer overlap and report dHash+pHash candidates.
- No detector is loaded and no score/threshold is produced; IPN remains DEVELOPMENT only.
- **Decision:** commit before implementing the decoder/audit or opening IPN pixels.

### C4-R1b IPN audit implementation checkpoint

- Selection/download receipt binding, exact decode/container/EXIF/SHA checks and protected/peer
  decontamination implemented; shared landscape/portrait scene ids are retained across devices.
- Same-scene perceptual candidates remain visible; only cross-scene candidates are gate failures.
- Twenty focused audit/acquisition/realization tests pass; no detector dependency exists.
- **Decision:** commit method before the production 960-image decode.

### C4-R1b IPN DEVELOPMENT realization — pass, unscored

- 960/960 RGB JPEGs decode; all retain EXIF and all SHA-256 values are unique.
- Twelve devices x 80 images; 80 shared scenes (50 landscape + 30 portrait) x twelve cameras.
- Zero protected/passed-peer overlap and zero cross-scene perceptual collision.
- Detailed report 642,208 B / SHA `f5827dce...243b`.
- **Decision:** freeze as a clean absent-source DEVELOPMENT gate; do not score before R1b freezes.

### C4-R1b iPhone 14 transfer and archive-method preregistration

- Four ranges plus preserved prefix assembled to 20,428,338,922 B; published MD5
  `dfc01c89...946c` passed before atomic promotion and temporary cleanup.
- Central directory reports 7,996 JPEGs / ten devices: 4,000 blank and 3,996 natural across four
  lenses. This is structural metadata, not yet a CRC/decode pass.
- Implement receipt-bound traversal/symlink/encryption/CRC/expansion inventory, then freeze only
  natural rows from inventory metadata and extract atomically with size/CRC/SHA checks.
- **Decision:** commit before reading any ZIP member payload.

### C4-R1b iPhone 14 archive implementation checkpoint

- Exact hierarchy parser plus receipt-bound safe inventory, metadata-only natural freezer and
  size/CRC/SHA-checked atomic extractor implemented.
- Blank rows cannot enter extraction; no function assigns TRAIN/CALIBRATION.
- Twenty-four combined iPhone/archive/acquisition tests pass.
- **Decision:** commit method before production CRC inventory.

### C4-R1b iPhone 14 production inventory — pass

- 20,428,338,922-byte ZIP; MD5 `dfc01c89...946c`; SHA-256 `22f04a95...8cbb9`.
- 7,996/7,996 JPEG members pass CRC/safety: 4,000 blank, 3,996 natural, ten devices, four lenses.
- Detailed inventory 1,295,576 B / SHA `8931a535...912e`; no member selected or extracted.
- **Decision:** freeze inventory before running the natural-only metadata selector.

### C4-R1b iPhone 14 natural selection — frozen before member bytes

- 3,996 natural selected; 4,000 blank excluded. Devices 398-400 each; lenses 998-1,000 each.
- Detailed selection 1,425,474 B / SHA `88dc326e...7b74`.
- No member payload read and no role assigned.
- **Decision:** commit selection before atomic extraction.

### C4-R1b iPhone 14 natural extraction — complete, role-free

- 3,996/3,996 selected members / 12,914,703,500 B; size+CRC verified, atomic write, per-file SHA.
- No blank member extracted. Detailed receipt 1,884,013 B / SHA `46b36e56...09de`.
- Realization will bind the receipt, decode/hash and compare protected E30/peers, stored IPN hashes
  and owner exact-byte hashes only after reproducing gallery identity `390e3c21...ac09`.
- No protected detector score or role assignment is allowed.
- **Decision:** commit extraction receipt and realization boundary before decoding iPhone pixels.

### C4-R1b iPhone 14 realization implementation checkpoint

- Extraction-bound decode/SHA+dHash+pHash plus existing protected/peer gates implemented.
- Stored IPN exact/perceptual hashes are protected. Owner access is raw exact SHA only and must
  reproduce the 210-file `390e3c21...ac09` identity; no detector/perceptual owner read exists.
- Eighteen focused iPhone/realization/identity tests pass; no role is assigned.
- **Decision:** commit method before production iPhone pixel decode.

### C4-R1b iPhone 14 realization — stopped on one duplicate component

- 3,996/3,996 RGB+EXIF decode; 3,996 unique SHA; zero protected/peer/IPN/owner overlap.
- Payload formats: 3,945 MPO, 51 JPEG despite `.JPG`; standardized JPEG input is mandatory.
- One confirmed duplicate burst: iPhone14_5 telephoto `IMG_1290.JPG` + `IMG_1291.JPG`, equal
  dHash+pHash. Detailed rejected audit 2,638,999 B / SHA `8325aaf4...05fd`.
- Precommitted correction: exclude both rows, freeze 3,994 role-free; never choose one member.
- **Decision:** preserve the rejection, commit before implementing the eligibility overlay.

### C4-R1b iPhone 14 eligibility implementation checkpoint

- Receipt transformer accepts only the exact one-failure/two-row component and excludes every side.
- Source bytes remain intact; output is role-free. Two focused component/identity tests pass.
- **Decision:** commit method before production eligibility freeze.

### C4-R1b iPhone 14 production eligibility — pass

- 3,996 selected -> exclude both members of one burst -> 3,994 eligible, role-free.
- Eligible payloads: 3,943 MPO + 51 JPEG; detailed receipt 2,364,384 B / SHA `a71c4a06...57bf`.
- R1b controlled rule: retain all old C3 roles, append iPhone only, complete devices 8/2 TRAIN/CAL,
  no new AI/rebalancing, class-weighted heads, identical derived JPEG input.
- **Decision:** freeze eligibility and role rule before implementing the R1b manifest.

### C4-R1b role-extension implementation checkpoint

- C3 manifest and every old row/role/order are hash-bound and preserved; only eligible iPhone REAL
  rows append. Existing stable subset rule assigns eight complete devices TRAIN, two CALIBRATION.
- No rebalancing/new AI rows; class-weighted heads isolate authentic Apple coverage.
- Six focused role tests pass; no DEVELOPMENT/LOCKED row is accessible.
- **Decision:** commit method before production role freeze.

### C4-R1b controlled production roles — frozen

- C3 prefix 22,688/22,688 exact; append iPhone 3,994 -> total 26,682.
- AI 11,344 / REAL 15,338; TRAIN 21,349 / CALIBRATION 5,333.
- iPhone: TRAIN 3,195 on eight devices; CAL 799 on iPhone14_4+iPhone14_8; zero device leakage.
- Detailed SHA `16deb276...750f`; record-list SHA `263af46b...5611`; no protected row.
- **Decision:** freeze roles before realizing appended standardized inputs.

### C4-R1b input-extension implementation checkpoint

- R1b manifest + old R0 receipt hard-bound; all 22,688 old derived rows reused byte-for-byte.
- Only 3,994 iPhone parents append through identical EXIF/RGB/256/224/JPEG-q90-4:4:4 transform.
- Seven focused input tests pass; no DEVELOPMENT/owner path exists.
- **Decision:** commit method before production standardization.

### C4-R1b production standardized inputs — pass

- 26,682/26,682 rows; 22,688 old bytes reused + 3,994 iPhone appended.
- 568,959,891 logical bytes; detailed SHA `400a990d...6af8`; records SHA `3e51f87a...1395`.
- MPO/JPEG source-container difference is removed before encoders; no protected input accessed.

### C4-R1b controlled trainer implementation checkpoint

- Frozen old DINO/CF caches hash-bound; extract only 3,994 new embeddings and merge by record id.
- Same class-weighted C grid and macro<=10%/worst<=20% real-FP threshold budgets.
- Six focused merge/head tests pass.
- Pre-result arm rule: passing arm with higher CAL AUC; exact tie -> smaller selected C -> DINO.
- **Decision:** commit before either production embedding run; IPN/owner model scores stay closed.

### C4-R1b controlled internal screens and arm freeze

- DINO: AUC 0.996860; current-AI macro/worst 99.18%/97.5%; REAL macro/worst FP 9.97%/15.91%;
  C=0.1, threshold 0.095101, artifact `aca41dd8...8e86`.
- CF: AUC 0.998079; current-AI macro/worst 99.82%/99.55%; REAL macro/worst FP 9.97%/12.64%;
  C=0.01, threshold 0.125935, artifact `68a54aa2...701c`.
- Both pass. Frozen rule selects CF on higher AUC; selection receipt binds both evidence SHAs.
- **Decision:** freeze CF as the only external candidate before implementing/scoring IPN or owner.

### C4-R1b external DEVELOPMENT gate preregistration

- Hard-bind CF artifact `68a54aa2...701c`, pinned CF weights and threshold 0.125935.
- Bind IPN realization `f5827dce...243b` (960/12 devices) and owner identity
  `390e3c21...ac09` (210 stills); exact standardized JPEG round-trip.
- Score selected CF once. Pass iff IPN worst-device FP <=20%, owner FP <=20% and already-frozen
  internal current-AI macro >=90%. Report per-device IPN and aggregate score distributions.
- No refit/rethreshold/DINO fallback/test policy. **Decision:** commit before scorer implementation.

### C4-R1b external scorer implementation checkpoint

- Hash-verified selected-CF CLI reproduces standardized JPEG round-trip and frozen threshold.
- Aggregate runner binds IPN/owner identities, reports every IPN device and exact frozen gate.
- Three focused candidate/gate tests pass; no fitting or DINO fallback path exists.
- **Decision:** commit method before the single external DEVELOPMENT run.

### C4-R1b external DEVELOPMENT run — fail, no repair

- Frozen candidate: CF-ViT head SHA `68a54aa2...701c`, C=0.01, threshold 0.125935.
- IPN-NFID: 249/960 FP; REAL recall 74.0625%; macro-device FP 25.9375%; worst-device FP 40.0%.
- Owner gallery: 144/210 FP; REAL recall 31.4286% (R0 24.2857%, R1a 26.6667%).
- Frozen internal current-AI macro recall: 99.8199%; that gate alone passes.
- Gate result: fail (`IPN worst-device <=20%` false, `owner FP <=20%` false). No threshold change,
  refit, DINO fallback or test-derived policy; no LOCKED AI population opened.
- Evidence: `evidence/e32_r1b_external_development.json`.
- **Decision:** reject R1b from serving. Preserve it as the controlled clean-real-data ablation;
  redesign the source-invariance objective/decision layer and reserve a new authentic source for
  the next final gate. IPN and owner are consumed DEVELOPMENT from now on.

### C4-R1b final engineering verification

- Evidence SHA-256: `8752699f643609588d8725f34c469c44791f6712cffcca7c03677de30b1c8d48`.
- 203 Python tests; production web build + 6 web tests; typecheck; ESLint; `pip check`; artifact
  registry: all pass (one upstream Starlette deprecation warning only).
- Hash-verified CLI smoke loaded the pinned CF artifact and scored one owner still; the incorrect
  AI verdict reproduces the measured generalization failure through the public research boundary.
- Storage audit: no real `.partial` or active transfer; E32 137 GB; LaCie 514 GiB free.

### C4-R1b local demo integration — research visibility only

- New `demo` runtime profile: E20 + E26; R1b only with explicit `--r1b-data-root`.
- R1b shares the verified CF-ViT backbone/processor and separately verifies selection, head and
  weight hashes. It never enters the E26 OR rule, canonical registry or readiness.
- Typed API invariants: `research_only=true`, `affects_decision=false`, band is only `ai_signal` or
  `insufficient_evidence`; optional load/inference failure degrades only the R1b card.
- Real owner-still E2E: E26 `insufficient` / CF -8.6586; R1b 0.3132 > 0.1259; E20 0.9988 > 0.9895.
  The disagreement is exposed, not blended or converted into confidence.
- Verification: focused 20/20 then full 207/207 Python; production build + 6 web tests, typecheck,
  ESLint, dependency graph and canonical artifact registry all pass. One upstream Starlette
  deprecation warning remains unrelated.
- **Decision:** keep the card as presentation-grade failure analysis, never a product verdict.

### C4-R1b demo hierarchy correction — presentation only

- The result surface now leads with R1b's direct `ai_signal` / `insufficient_evidence` answer and
  one 0–100 raw-signal bar. It states the score's distance above or below the frozen 0.125935
  threshold, eliminating the misleading impression that a low-looking percentage cannot trigger.
- The percentage remains explicitly non-probabilistic. E26, E20, artifact identifiers, thresholds
  and measured IPN/owner false-positive limits move under collapsed technical details.
- A real local R1b request returned 0.3132 > 0.1259 and rendered the direct AI-signal answer.
  R1b remains `research_only=true`, `affects_decision=false`; no score, threshold, API vote or
  artifact changed.
- Verification: 207 Python tests, production build + six web tests, TypeScript, ESLint, `pip check`
  and the six-entry canonical artifact registry all pass.

### C4-R1c pre-plan threshold-feasibility diagnostic — no candidate

- **Question:** did R1b fail because CF-ViT no longer separates the classes, or because its
  internally selected 0.125935 threshold does not transfer to independent authentic pipelines?
- **Boundary:** read-only rescore of the already-consumed 960-image IPN and 210-image owner
  DEVELOPMENT sets plus the frozen internal CAL AI feature cache. No output artifact, score cache,
  threshold, model parameter, serving path or decision was written. The sweep is post hoc and can
  justify only a future clean replication; it cannot select R1c.
- **Frozen R1b point:** owner FP 68.57%; IPN macro/worst-device FP 25.94%/40.0%; internal current-AI
  macro/worst-family recall 99.82%/99.55% (six-source macro 99.78%) at 0.125935.
- **First joint diagnostic point:** 0.863312 yields owner FP 20.0%, IPN macro/worst-device FP
  5.42%/15.0%, internal current-AI macro/worst recall 90.01%/80.0% (six-source macro 91.00%).
  Per-family internal recall: CommunityForensics 95.96%, FLUX.2 Klein 88.96%, GPT Image 1 95.73%,
  Nano Banana 92.81%, Nano Banana Pro 80.0%, Qwen Image 2512 92.57%.
- **Stricter diagnostic point:** 0.95 yields owner FP 9.52%, IPN macro/worst-device FP 3.02%/7.5%,
  internal current-AI macro/worst recall 83.28%/65.0% (six-source macro 85.13%).
- **Conclusion:** unlike E31's inverted DEVELOPMENT ranking, R1b contains a potentially useful
  conservative operating region. The next experiment is threshold-only transfer using genuinely
  new calibration sources. IPN/owner-derived 0.863312 and 0.95 are permanently ineligible values;
  failure to reproduce cleanly triggers paired semantic+frequency alignment, not another arbitrary
  data-volume or encoder sweep.

### E33 — licensed RRDataset threshold transfer: rejected on CAL

- **Boundary:** official ICCV 2025 RRDataset validation only, frozen as 250 REAL + 250 AI before
  scoring. R1b backbone/head/input/score direction stayed byte-identical. RR test, IPN and owner
  gallery remained unopened; no model weight was fitted.
- **Integrity:** 500/500 decode and inference successes; score SHA
  `1a3dd4c7...98f6`. Filenames expose seven AI scenarios but only one pooled REAL identity, so the
  authentic gate is aggregate and cannot claim camera-source transfer.
- **Ranking:** ROC-AUC 0.80728, EER 0.276, TPR@FPR=10% 0.52. This is below the preregistered working
  AUC 0.85 and far below internship-success AUC 0.90.
- **Frozen R1b threshold 0.125935:** AI recall 96.4%, REAL FP 82.8%, balanced accuracy 56.8%.
- **First REAL-safe frontier 0.998400:** REAL FP 10.0%, pooled AI recall 52.0%, balanced accuracy
  71.0%, AI scenario macro/worst recall 60.52%/26.88%. Everyday-life AI is the weakest at 26.88%.
- **Decision:** reject R1c-T before DEVELOPMENT and do not download/open the 20.12 GB locked RR
  test. Threshold transfer alone cannot satisfy both authentic safety and AI coverage. Open one
  paired semantic+frequency-aligned R1c-P experiment; do not sweep another threshold or ensemble.

### E35 — official DDA transfer screen: benchmark-strong, authentic gate failed

- **Frozen candidate:** official `Junwei-Xi/Dual-Data-Alignment` DINOv2-L/14 + rank-8 LoRA,
  revision `4390d902...16c`, full checkpoint SHA-256 `b27a31d3...e3e`, RGB center crop 336,
  published CLIP normalization, sigmoid score and untouched threshold 0.5. The local timm adapter
  enables dynamic positional interpolation to reproduce torch.hub DINOv2's published 336px path;
  a focused forward test and one-image smoke passed before production scoring.
- **Boundary/integrity:** one run over 500 RR, 960 IPN and the exact old 210-owner set; 1,670/1,670
  successes. The new 211th owner still was hash-identified, preserved and excluded unscored. Score
  stream is 348,372 B / SHA-256 `ae352ffe...83a`. DDA-COCO remained unopened.
- **RR result:** ROC-AUC **0.978192**, EER **0.080**, TPR@FPR=10% **0.920**, balanced accuracy
  **92.4%**, REAL FP **6.4%**, pooled AI recall **91.2%**, scenario macro/worst recall
  **94.38%/81.72%**. All RR gates pass.
- **Authentic transfer:** IPN macro-device FP **15.0%**, worst-device FP **36.25%**
  (Samsung Galaxy Note9); owner-gallery FP **34.76%** / REAL recall **65.24%**. Both frozen <=20%
  transfer gates fail, so the overall state is `dda_development_failed`.
- **Post-hoc diagnosis only:** the first score boundary satisfying all three consumed real gates is
  0.901156, with RR REAL FP 0.4%, IPN worst-device FP 20.0%, owner FP 13.33% and RR AI recall 82.4%.
  At 0.95, RR AI recall falls to 76.8%. This proves a conservative region may exist; it does not
  authorize 0.901156 because all populations used to find it are consumed DEVELOPMENT.
- **Decision:** keep DDA as the E36 representation candidate but reject its published operating
  point. Collect one compact, genuinely new multi-device + modern-generator CAL, estimate exactly
  one threshold under preregistered gates, then score one new LOCKED FINAL set. Do not serve DDA,
  open DDA-COCO, ensemble old models or download the 112.97 GB training release before that CAL.
  Evidence: `evidence/e35_dda_development.json` and
  `evidence/e35_dda_threshold_diagnostic.json`.
- **Engineering verification:** 236 Python tests, bytecode compilation, dependency integrity,
  six-entry artifact registry, production web build + six web tests, TypeScript and ESLint pass.
  One upstream Starlette/httpx deprecation warning remains unrelated to DDA inference.

### E36 — clean modern CAL rejects threshold-only DDA repair

- **Boundary:** the frozen 1,071-parent CAL contains 471 native originals from five unseen devices
  and 600 clean outputs from six 2026 generator families. All rows scored exactly once with the
  unchanged official DDA checkpoint; FINAL remained at zero downloaded/scored bytes.
- **Published threshold 0.5:** REAL device-macro/worst FP **16.61%/35.0%** and AI family-macro/
  worst recall **38.0%/6.0%**. It is neither camera-safe nor sufficiently AI-sensitive.
- **First preregistered REAL-safe frontier 0.756332:** REAL device-macro FP **9.36%** (bootstrap
  95% CI 7.00–11.76%) and worst-device FP **20.0%** pass, but AI family-macro recall is only
  **27.67%** (95% CI 24.50–30.83%) and worst-family recall is **1.0%** on FLUX.2 Max.
- **Ranking at the selected frontier:** ROC-AUC **0.58753**, TPR@FPR=10% **0.285**, EER
  **0.4267**, balanced accuracy **0.5895**, with 1,071/1,071 coverage. All representation/AI gates
  fail despite the authentic FP gates passing.
- **Decision:** state `calibration_failed`; create no candidate, download/score no FINAL row and do
  not complete/open DDA-COCO. E36 disproves the threshold-only hypothesis on current generators.
  Its rows become a consumed adaptation pool for the separately preregistered E37 source-held-out
  head; only out-of-fold source predictions may choose E37's operating point. Compact evidence:
  `evidence/e36_calibration.json`; local detailed scores remain under the external E36 directory.

### E37 — source-held-out DINO adaptation restores ranking, not the operating point

- **Boundary:** frozen DINOv2-S features from 21,349 E32 TRAIN rows were reused; only the 1,071 E36
  parents received new embeddings. Five fixed folds held every E36 REAL device and AI family out
  exactly once. Each head used fixed `C=0.1`, balanced classes and no FINAL/DDA score feature.
- **Result:** 1,071/1,071 OOF coverage, ROC-AUC **0.94811**, TPR@FPR10 **0.820** and EER
  **0.12976**. These pass the ranking gates and improve dramatically over E36 DDA's AUC 0.58753.
- **Joint frontier:** threshold `0.999121` gives REAL device-macro/worst FP **4.14%/19.72%**, but AI
  family-macro/worst recall only **57.5%/42.0%** and balanced accuracy **0.7716**. Device 009 alone
  contributes 14/15 false accusations; AI family recall ranges from 42% FLUX.2 Max to 70% GLM.
- **Decision:** `oof_gate_failed`; no candidate artifact and no FINAL access. A post-hoc uniform
  adaptation-weight diagnostic on this now-consumed DEVELOPMENT population found a feasible region
  without source/example selection; E38 must freeze one such setting and can be validated only by
  the untouched FINAL. Evidence: `evidence/e37_source_heldout.json`.

### E38 — uniform modern-domain emphasis passes DEVELOPMENT and freezes one candidate

- **Fixed change:** identical DINOv2-S backbone/input/folds and complete row set; `C=0.0003`, old
  E32 TRAIN sample weight 1 and every E36 adaptation row weight 100. No DDA feature, ensemble,
  per-source weighting, example removal or further grid. Contract SHA `c61ec080...eedd`.
- **OOF result:** 1,071/1,071 coverage; ROC-AUC **0.98062**, TPR@FPR10 **0.975**, EER **0.06162**
  and balanced accuracy **0.8955** at threshold `0.896190`.
- **Balanced frontier:** REAL device-macro/worst FP **4.34%/19.72%**; AI family-macro/worst recall
  **82.5%/77.0%**. Every preregistered DEVELOPMENT gate passes simultaneously. Bootstrap 95% CIs
  are 2.54–6.27% REAL macro FP and 79.5–85.5% AI macro recall.
- **Candidate:** external 13,078-byte artifact SHA-256 `fddbe475...4067`, fitted on 21,349 old TRAIN
  + all 1,071 adaptation rows with the OOF threshold unchanged.
- **Caveat/decision:** the uniform weight and C were chosen after consumed E37 diagnostics; this is
  a development-selected candidate, not independent proof. It earns exactly one evaluation on the
  source/family-disjoint FINAL frozen before those diagnostics. Evidence:
  `evidence/e38_development.json`.

### E38 FINAL — excellent separation, conservative threshold misses AI gate

- **One-shot boundary:** the frozen 13,078-byte candidate and `0.896190` threshold scored all 640
  untouched parents once: four unseen REAL devices x100 and six family-disjoint AI cells x40.
  There were no failures, retries, model changes or threshold changes.
- **Ranking:** ROC-AUC **0.98185**, TPR@FPR10 **0.950**, EER **0.075**. The representation transfers
  strongly to the held families/devices.
- **Frozen operating point:** 400/400 REAL correct, REAL device-macro/worst FP **0%/0%**; 162/240
  AI correct, AI family-macro recall **67.5%** (95% CI 61.67–73.33%) and worst-family **50.0%**
  (Seedream 4.5). Balanced accuracy **0.8375**. AI macro/worst and balanced-accuracy gates fail.
- **Decision:** state `final_failed`; E38 is not promoted and this FINAL cannot evaluate another
  retry. A post-hoc diagnostic finds a feasible threshold at `0.270069` (REAL macro/worst FP
  10%/17%; AI macro/worst 95%/90%), identifying refit-to-OOF score-scale transfer as the remaining
  defect. That threshold is contaminated and ineligible; it can only define an E39 candidate for a
  genuinely new FINAL. Evidence: `evidence/e38_final_result.json`.

### E39-A — consumed-FINAL threshold correction

- **Role boundary:** all 400 REAL +240 AI E38 FINAL rows are now consumed `E39_CALIBRATION`; E38
  remains `final_failed`. Role evidence binds the original candidate, manifest, result and detailed
  score hashes before E39 candidate packaging.
- **Controlled change:** threshold only. DINOv2-S weights, 224px preprocessing, fitted logistic
  head, positive label and 13,078-byte artifact remain byte-identical at SHA-256
  `fddbe475...4067`. No row/source selection or retraining occurred.
- **Selected threshold:** `0.27006945014`, the lowest candidate satisfying all frozen source-level
  budgets over the complete 640-row consumed calibration set.
- **Calibration frontier:** REAL device-macro/worst FP **10.0%/17.0%**; AI family-macro/worst
  recall **95.0%/90.0%**; balanced accuracy **0.925**, AUC **0.98185**, TPR@FPR10 **0.950**, EER
  **0.075**, coverage **640/640**. Every calibration gate passes.
- **Decision:** freeze one research candidate and obtain a genuinely new FINAL. These measurements
  are contaminated by threshold selection and cannot promote the detector. Candidate JSON SHA-256
  `7d497929...2cef`; compact evidence `evidence/e39_calibration.json`.

### E39-B — independent native/recent FINAL preregistration

- **REAL:** FloreView camera-native natural JPEGs, four source-new devices/brands x40: iPhone 13
  mini, DOOGEE S96 Pro, Pixel 5 and OnePlus 8T. CC BY-SA 4.0; official URL catalog SHA-256
  `90d8408c...186b`; fixed location-diverse ordering.
- **AI:** AIGenImages2026 revision `d634f663...c0c5`, seven unused 2025 generator cells x40: Reve
  1.0, HiDream I1 Dev, Ideogram 3, Midjourney v7, Firefly Image 5, Z Image Turbo and Gemini 3 Pro
  Image. CC BY 4.0; 11,138,511,098-byte archive SHA-256 `67c60427...c498`.
- **Population:** 160 REAL +280 AI =440 clean/native parents. Every source is equally capped;
  selection is deterministic and cannot access scores, embeddings or image appearance.
- **Boundary:** source contract and acquisition code recorded with zero E39 FINAL image bytes.
  Candidate/model access remains forbidden until decode, provenance and overlap audits freeze the
  unscored manifest.

Acquisition reproduced the 11,138,511,098-byte AI archive SHA-256 and all 160 REAL URLs. Tar safety
inventory passed 10,905 members/10,879 regular files and froze 40 rows per each of the seven AI
families from 150–307 eligible members. Detailed selection SHA-256 is `bf6a6ecf...40dfb`; no image
was scored or selected using model output.

- **Frozen realization:** 160/160 REAL and 280/280 AI decode; REAL >=2 MP with EXIF; AI prompt
  provenance 280/280. Prior-role exact/dHash overlap 0/0; within-FINAL exact/dHash duplicates 0/0.
  Detailed unscored manifest is 412,914 B / SHA-256 `1076df20...7306`. This is the one-use E39
  FINAL boundary; no score existed when it was written.

### E39 FINAL — new AI generalizes; new phone REAL does not

- **One-shot boundary:** 440/440 coverage, no retry; detailed score stream 146,705 B / SHA-256
  `2a47e8a8...bb86` at frozen threshold `0.270069`.
- **AI:** 261/280 correct; family macro/worst recall **93.21%/90.0%**. Per-family recall is Firefly
  97.5%, Gemini 3 90.0%, HiDream 90.0%, Ideogram 95.0%, Midjourney 97.5%, Reve 90.0% and Z Image
  92.5%. Both AI gates pass strongly.
- **REAL:** 75/160 correct; device FP **55.0%/55.0%/60.0%/42.5%**, macro/worst
  **53.13%/60.0%**. Both safety gates fail decisively.
- **Global:** AUC **0.90033** passes by 0.00033, but TPR@FPR10 **0.7714**, EER **0.1933** and
  balanced accuracy **0.7004** fail. State `final_failed`; no product promotion.
- **Post-hoc diagnosis only:** first REAL-safe threshold `0.622942` still gives AI macro/worst
  77.14%/67.5% and balanced accuracy 0.8357. No joint threshold exists, so E40 must adapt the
  representation/head using consumed E39 source/content-balanced development. Evidence:
  `evidence/e39_final_result.json`, `evidence/e39_threshold_diagnostic.json`.

### E40-A — preregistered consumed-data boundary

- **Role:** all 160 REAL +280 AI E39 parents become `E40_ADAPTATION_DEVELOPMENT` before any E40
  feature extraction or fitting. `evidence/e40_role_amendment.json` binds the E39 manifest,
  score-stream, result and decision-contract hashes.
- **Leakage rule:** each source receives one out-of-fold prediction from a head that never trained
  on that source. Frozen DINO content clusters affect weights only; they cannot select examples or
  define folds. This correction is model-free and was made before E40 scores existed.
- **FINAL rule:** E39 and unused same-collection rows are permanently ineligible. No new FINAL data
  may be acquired until an E40 development candidate passes its frozen gate and robustness checks.

### E40 fixed head-ladder contract

- **Inputs:** deterministic round(5%)-per-label/source E32 TRAIN replay (expected 1,067), all 1,071
  E36 consumed rows, and source-held-out E39 development rows. Backbone/preprocessing are unchanged.
- **Heads:** StandardScaler + LogisticRegression, C=0.01, weighted at both scaler and classifier;
  exactly uniform, class/source-balanced and class/source/occupied-content-cell-balanced variants.
- **Folds/content:** 7 complete E39 source folds. The content variant fits KMeans(k=16, n_init=10)
  only on modern training rows inside each fold, then uses clusters for weights only.
- **Selection:** evaluate all variants at seed 42, take the first full-gate pass in fixed complexity
  order, freeze its OOF threshold, and require that unchanged threshold to pass seeds 41/42/43.
- **Boundary:** `evidence/e40_fixed_contract.json` is hash-bound in code before any features/scores.
  Six new focused tests and all 264 Python tests pass. No measurements exist yet.

### E40-A — frozen E39 DINO feature cache

- **Coverage:** 440/440 consumed E39 parents, 160 REAL +280 AI; 440 unique IDs and no non-finite
  embedding value. Feature shape is 440x384 float32.
- **Binding:** local archive is 642,070 B / SHA-256 `ec050171...94e68`; DINO weight SHA-256 remains
  `04d27f34...0081`, manifest `1076df20...7306`, role amendment `c475a529...2f0f`.
- **Folds:** fixed assignments contain 80 rows in each paired REAL+AI fold and 40 in each of the
  remaining three AI-only folds. This checkpoint performs no clustering, fitting or scoring.

### E40-B — three-head source-held-out result

| Fixed head | Gate | AUC | TPR@FPR10 | EER | Bal. acc. | REAL macro/worst FP | AI macro/worst recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | pass | 0.99464 | 0.98929 | 0.03661 | 0.94464 | 10.0% / 17.5% | 98.93% / 97.5% |
| source-balanced | pass | 0.99699 | 0.99286 | 0.02009 | 0.94643 | 10.0% / 17.5% | 99.29% / 97.5% |
| source+content | pass | 0.99071 | 0.98571 | 0.05000 | 0.94286 | 10.0% / 15.0% | 98.57% / 95.0% |

- **Selection:** fixed simplest-first priority selects `uniform`, threshold `0.1708009988`; no
  post-result preference for the marginally stronger source-balanced score is permitted.
- **Stability:** seeds 41/42/43 with that unchanged threshold are identical and pass all nine gates.
- **Draft:** 12,690 B / SHA-256 `72b8d8cd...819c0`; detailed report/evidence is 61,190 B /
  SHA-256 `93f98f07...ebcc`. Status is development draft, not FINAL-validated product.

### E40-C — fixed robustness gate before scoring

- **Views:** each of the 440 consumed E39 parents appears under native, JPEG-q50 and
  75%-resize+q50 transports. All views retain one parent ID; derivative N is never added.
- **Fixed gate:** every derivative needs AUC >=0.85, TPR@FPR10 >=0.80, balanced accuracy >=0.80,
  REAL macro/worst FP <=20%/30%, AI macro/worst recall >=80%/60%, and >=80% native-decision
  agreement per class. Every transport must keep mean AI score above mean REAL score.
- **Owner smoke:** the previously consumed identity-bound 210-photo gallery must stay <=20% FP at
  unchanged threshold 0.170801. It cannot select or tune anything.
- **Boundary:** method and code are committed before scores. A pass packages only a research
  candidate awaiting new FINAL; a fail stops without retry.

### E40-C — robustness result and stop

| Population/view | AUC | Bal. acc. | REAL macro/worst FP | AI macro/worst recall | Decision agreement |
|---|---:|---:|---:|---:|---:|
| E39 native (diagnostic) | 0.99955 | 0.95759 | 8.13% / 15.0% | 99.64% / 97.5% | reference |
| E39 JPEG q50 | 0.99951 | 0.96696 | 6.25% / 15.0% | 99.64% / 97.5% | REAL 98.13%, AI 100% |
| E39 resize75+q50 | 0.99949 | 0.96071 | 7.50% / 15.0% | 99.64% / 97.5% | REAL 98.13%, AI 100% |
| owner gallery native | n/a | n/a | **69.52% aggregate** | n/a | n/a |

- **Gate:** fail only on owner-gallery FP <=20%; 146/210 false AI, REAL recall 30.48%. No candidate
  artifact exists and there is no retry/model switch.
- **Bindings:** 487,011-byte score stream SHA-256 `a126e814...0e3b`; 23,591-byte report/evidence
  SHA-256 `74e23b66...a463`.
- **Post-hoc diagnostic only:** on 370 consumed REAL +280 consumed AI native scores, threshold
  0.619554 would pass all gates: REAL 4%/20%, AI 92.14%/90%, AUC 0.97218, TPR@FPR10 0.90714,
  EER 0.10, balanced 0.90396. It cannot change E40 and may only seed a new-final E41 contract.

### E41 — fixed broad-real calibration-transfer protocol

- **Consumed CAL:** all 440 E39 native E40-draft rows +210 owner-gallery rows; derivatives excluded.
- **Only change:** reuse the byte-identical E40 uniform head and fixed preprocessing, replacing
  threshold 0.170801 with diagnostic threshold 0.619554. No fit or second threshold exists.
- **Integrity:** packager hashes every learned scaler/logistic numeric array before/after writing.
- **Boundary:** role amendment, fixed contract, code and tests are committed before packaging. The
  resulting artifact, if produced, remains research-only until a new one-shot FINAL passes.

### E41 candidate — frozen, independent FINAL absent

- **Artifact:** 13,064 B / SHA-256 `9bcc021e...ab65`; threshold 0.619554.
- **Identity:** learned-head numeric SHA-256 `4211d8d8...f49f` before and after packaging; identical
  to E40. Only name/status/calibration binding and threshold metadata changed.
- **Calibration:** 650 consumed rows, REAL macro/worst 4%/20%, AI macro/worst 92.14%/90%, AUC
  0.97218, TPR@FPR10 0.90714, EER 0.10, balanced accuracy 0.90396. These values selected the
  threshold and are not validation evidence.
- **Decision:** stop before data. No API/web promotion and no E41 FINAL byte or score exists.

### E41/B-Free viral external stress — failed, no retry

- **Population:** 811/1,111 MD5-verified surviving web versions, 278 REAL +533 AI, grouped under all
  17 REAL +17 AI original source events. Exact/dHash screening found no prior-role overlap.
- **Frozen decision:** unchanged E41 artifact `9bcc021e...ab65`, threshold 0.619554 and preprocessing.
  Each URL version is scored, but each original source event has equal decision weight.
- **Result:** AI parent-weighted recall **100%**, REAL parent-weighted recall **18.41%** (81.59% FP),
  parent-weighted balanced accuracy **59.20%**. Event-mean AUC is 0.76125, TPR@FPR10 is 0.35294,
  EER is 0.35294. Balanced-accuracy 95% bootstrap CI is [0.5226, 0.6775].
- **Gate:** fail: balanced accuracy <0.80 and REAL recall <0.75; AI recall alone passes. Version-
  weighted diagnostics agree (AUC 0.77456, balanced accuracy 0.58273, REAL FP 83.45%).
- **Decision:** E41 remains rejected/research-only; no threshold change, row exclusion, retry or
  product promotion. The unopened 20.12 GB RR test is preserved for the future E42 winner because
  a second E41 score cannot reverse this mandatory-gate failure.
- **Evidence:** `evidence/e42_bfree_result.json`; score stream SHA-256
  `83783551...c33fc`; exact manifest SHA-256 `338a2f2...f37ca2`.

### E42-S — texture/intermediate source-held-out DEVELOPMENT pass

- **Inputs:** 4,638 TRAIN +2,246 consumed DEVELOPMENT parents across 63 total sources; zero cross-
  role exact SHA/dHash overlap. Each view uses global +two deterministic texture crops and four
  DINOv2-S intermediate blocks. Five OOF folds keep all 34 DEVELOPMENT sources intact.
- **Frozen head/cut:** source-balanced StandardScaler + LogisticRegression C=0.01; first clean OOF
  REAL-safe threshold `0.6600460410`. The 11,230-row OOF stream has SHA-256
  `0fbd15d5...dd32ff` and full coverage.
- **Clean result:** AUC **0.99287**, TPR@FPR10 **0.98462**, EER **0.04047**, balanced accuracy
  **0.95477**; REAL macro/worst FP **1.23% /20.0%**; AI macro/worst recall **92.69% /75.0%**.
- **Robust result:** four conditions combined AUC **0.99338**, balanced accuracy **0.93923**; REAL
  macro/worst FP **0.84% /13.5%**; AI macro/worst recall **88.99% /68.13%**. Every individual
  JPEG/WebP/resize+JPEG/blur condition remains above AUC 0.992 and balanced accuracy 0.928.
- **Decision:** all 12 preregistered checks pass. The fixed smallest-pass rule selects S and skips
  DINOv2-L because L cannot replace a passing S. Refit candidate is 87,977 bytes /SHA-256
  `6768466a...9062e7`, research-only until one-shot RR external testing.

### E42 RR FINAL — extraction/manifest/scoring method fixed before data access

- **Archive boundary:** resumable transfer only; exact size and published MD5 must complete before
  inventory. Safe extraction is restricted to `original|transfer|redigital/{real,ai}` and verifies
  both member count and expanded bytes. Extraction does not decode images or load the model.
- **Unscored manifest:** every row is decoded, SHA-256/dHash audited and grouped by canonical
  filename parent. Prior E42 and B-Free exact/dHash overlap is fatal; cross-parent exact duplicates,
  duplicate conditions and label-crossing parents are fatal. dHash collisions within the external
  set remain an explicit diagnostic rather than an automatic identity claim.
- **Second lock:** the final manifest SHA, unchanged candidate `6768466a...9062e7`, threshold
  `0.6600460410` and declared row count must be frozen and committed before model loading.
- **Inference/gate:** reuse the exact E42 clean feature path; score one row once. Original must pass
  the full nine-check success gate; transfer and redigital each need at least 20 successful rows per
  class, AUC >=0.85, balanced accuracy >=0.80 and coverage 1.0. Completed output forbids retry.
- **Verification:** ten focused RR acquisition/evaluator tests pass; production score rows remain 0.
- **Observed package correction:** the first inventory rejected root `RRDataset_final` because the
  pre-transfer assumption was `RRDataset_test`. Read-only member names show the archive actually
  uses `RRDataset_final/{original,transfer,redigital}/{real,ai}`; code/tests are corrected and
  recommitted before reinventory. No decode or model access occurred.
- **Acquisition result:** exact size/MD5 pass; inventory/extraction contain 50,999 images and
  20,354,797,721 expanded image bytes. Counts are 8,500 per condition/class except redigital REAL
  8,499, below the paper-described 10,000+10,000 parent population. This discrepancy is explicit.
- **First decode audit stop:** 35 same-label exact duplicate components, 13 protected exact REAL
  overlaps and one protected-dHash AI parent were found before any score. The revised fixed rule
  removes protected parents in every condition, propagates through exact components and retains one
  lexical parent per other exact component. Exclusions and official coverage are immutable manifest
  fields; cross-label exact or structural parent failures still abort. Eleven focused tests pass.
- **Frozen unscored result:** 47 whole parents /141 rows excluded, leaving 50,858 rows /16,953
  parents /20,341,312,914 bytes and 99.7235% official archive coverage. Condition counts are
  original 8,454 REAL +8,499 AI, transfer 8,454+8,499 and redigital 8,453+8,499. Manifest SHA-256
  `b2d815af...30c98`; score rows 0.
- **One-shot binding:** detailed score contract SHA-256 `a5387eb9...de658` binds the unchanged E42-S
  candidate, threshold, manifest and 50,858 rows before the first model load.

### E42 RR FINAL — failed once, no retry

- **Coverage/integrity:** 50,858/50,858 scored rows, zero inference failures. Score stream is
  14,572,649 B /SHA-256 `c065957e...68434`; report/evidence is 17,498 B /SHA-256
  `516c6d92...6252e`. Candidate `6768466a...9062e7` and threshold `0.6600460410` are unchanged.
- **Original:** AUC **0.94448**, TPR@FPR10 **0.85139**, EER **0.12434**, balanced accuracy
  **0.84634**; AI recall **93.54%**, REAL FP **24.27%** (2,052/8,454). AUC, TPR, EER and AI-source
  checks pass; balanced accuracy and both pooled-REAL FP checks fail.
- **Transfer:** AUC **0.92582**, balanced accuracy **0.83993**, AI recall 89.89%, REAL FP 21.91%.
  Its preregistered AUC/balanced/coverage checks pass.
- **Redigital:** AUC **0.85629**, balanced accuracy **0.78756**, AI recall 77.03%, REAL FP 19.52%.
  AUC and coverage pass, balanced accuracy fails.
- **Gate/decision:** external final failed. No score deletion, threshold change, second run or
  serving promotion is permitted; E42 remains a reproducible rejected research candidate.
- **Post-hoc diagnosis only:** original's best balanced point is threshold 0.92704 /balanced
  0.87686 /REAL FP 9.25% /AI recall 84.62%. Redigital's condition-specific maximum is only
  balanced 0.78943 at threshold 0.74033, and no single threshold satisfies all declared gates.
  Therefore E43 needs a representation plus redigitalization-coverage change, not calibration
  theatre. RR is consumed DEVELOPMENT for any future candidate; a new untouched FINAL is required.

### E43 RR role freeze — score-blind transport adaptation population

- **Method:** require complete original/transfer/redigital triplets; select with SHA-256 namespace
  `E43_SELECT`; assign roles independently with `E43_ROLE`. Cap REAL at 1,960 and every one of seven
  AI scenario sources at 280. No E42 score file is an input.
- **Result:** 3,920 parents /11,760 rows; TRAIN 1,960 parents /5,880 rows, CAL 980 /2,940,
  DEVELOPMENT 980 /2,940. Every role is class-balanced; original/transfer/redigital each contain
  3,920 rows.
- **Integrity:** detailed manifest is 7,645,807 B /SHA-256 `29dd9b56...4b16`; source manifest is
  unchanged `b2d815af...30c98`. Zero scores read, zero model scores created and zero image copies.
- **Meaning:** RR is consumed local DEVELOPMENT and may improve/measure E43, never validate it
  externally. ITW-SM remains the untouched final pending manual author approval.

### E43-S RR features — complete before fit

- **Input:** exact role manifest `29dd9b56...4b16`; 11,760 condition rows /3,920 parents.
- **Representation:** unchanged E42-S DINOv2-S weights `04d27f34...0081`, CLS tokens from blocks
  2/5/8/11 over one global plus two deterministic texture crops; crop mean+std aggregation.
- **Result:** shape 11,760x3,072, 134,777,581 B /SHA-256 `fdc5d4c8...a4aa4`; complete coverage.
- **Boundary:** features only. Zero head fit, threshold selection, DEVELOPMENT scores or ITW-SM
  access occurred.

### E43-S pre-DEVELOPMENT fit — candidate and threshold frozen

- **Method:** fixed StandardScaler + LogisticRegression (`C=0.01`, seed 42) on 13,768 consumed E42
  fit-eligible views plus all 5,880 RR TRAIN triplet views. Sample weights equalize labels, sources,
  parents and each parent's views. No hyperparameter search was performed.
- **CAL rule:** only the 980 RR CAL originals may choose the REAL-safe operating point. RR
  DEVELOPMENT, the old RR score stream and ITW-SM are not inputs.
- **CAL result:** threshold **0.8712875247**; AUC **0.973686**, TPR@FPR10 **0.951020**, EER
  **0.073469**, balanced accuracy **0.925510**, REAL FP **10.0%**, AI recall **95.102%** and 100%
  coverage. Worst AI scenario recall is **70.0%** (`everyday_life`). These are calibration
  measurements, not independent success evidence.
- **Binding:** 87,916-byte candidate SHA-256 `a3aec445...47390`; 5,322-byte tracked report SHA-256
  `d24109c7...a14c8`. It explicitly records zero RR DEVELOPMENT and zero ITW-SM scores.
