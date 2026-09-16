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

### E43-S consumed DEVELOPMENT — passed once

- **Integrity:** candidate `a3aec445...47390` and threshold `0.8712875247` were hash-bound before
  score. Exactly 2,940 RR DEVELOPMENT rows plus 11,230 historical regression views were scored
  once with full coverage. The 4,192,797-byte score stream SHA-256 is `8398f763...1ccc4`; compact
  tracked report SHA-256 is `eda98604...5319`. ITW-SM scores remain zero.
- **RR original:** AUC **0.981941**, TPR@FPR10 **0.957143**, EER **0.069388**, balanced accuracy
  **0.932653**, REAL FP **7.959%**, AI macro/worst recall **94.490%/71.429%**.
- **RR transfer:** AUC **0.978263**, balanced accuracy **0.927551**, REAL FP **7.143%**, AI recall
  **92.653%**. **RR redigital:** AUC **0.951862**, balanced accuracy **0.886735**, REAL FP
  **8.571%**, AI recall **85.918%**. Redigital's `everyday_life` subgroup is still weak at
  **48.571%** recall; the preregistered robustness gate is pooled AUC/balanced accuracy, so this is
  a disclosed residual risk rather than a hidden extra failure rule.
- **Historical regression:** clean AUC/balanced **1.000000/0.991346**, robust
  **0.999900/0.958028**; all fixed tolerance checks pass. This population is consumed and partly
  replayed in fit, so it is only a forgetting diagnostic, not independent evidence.
- **Decision:** every frozen E43-S local gate passes. DINOv2-L stays locked. Package E43-S and wait
  for the untouched ITW-SM one-shot final; do not promote it solely from this consumed result.

### E43 Plan C — DDA-COCO acquisition and paired structure (unscored)

- **Synthetic archive:** official pinned DDA-COCO, 4,301,452,066 bytes /SHA-256
  `8cd60077...99c24`; safe ZIP and full CRC pass. It contains 29,969 images across six synthetic
  variants and no bundled REAL folder.
- **REAL companion:** official COCO val2017 transfer completed once using the prebound URL,
  815,585,330-byte size and S3 ETag. SHA-256 is `4f7e2ccb...82f05`; all 5,000 JPEG members pass the
  exact path schema and full CRC.
- **Pair structure:** exactly 4,969 parent IDs occur in REAL and every synthetic variant, yielding
  34,783 prospective rows. Parent-ID-set SHA-256 is `2b68c4aa...a3a7b`.
- **Boundary:** zero pixels decoded for manifest selection, zero model loads and zero scores. Next
  freeze a seven-view parent-group manifest after decode/hash and protected-overlap audit.
- **Frozen manifest:** all 34,783 candidate rows decoded; 19 protected dHash hits touched four
  parents, so their 28 views were removed together. The immutable remainder is 4,965 parents /
  34,755 rows, balanced across REAL and six synthetic conditions. It contains no within-pool exact,
  cross-label exact or cross-parent dHash group. Detailed manifest SHA-256 is
  `e663d679...a3db`; model scores remain zero.
- **One-shot binding:** tracked contract SHA-256 `a414e500...b69da` binds the immutable manifest,
  E43-S candidate/threshold, 34,755 declared rows and all eight success checks before model load.
  It forbids threshold repair, post-score row removal, test-informed fit and completed-run retry.

### E43 Plan C — DDA-COCO failed once, no retry

- **Integrity:** 34,755/34,755 rows scored, zero failures. Score stream is 7,171,203 bytes /SHA-256
  `1eefbdb7...42dd`; report SHA-256 is `b91f4a52...c844b`. Candidate, threshold, manifest and score
  contract are unchanged.
- **Pooled result:** AUC **0.54178**, TPR@FPR10 **0.11712**, EER **0.47051**, balanced accuracy
  **0.51114**, REAL FP **14.44%**, AI macro/worst-variant recall **16.67%/12.77%**. Only the 100%
  coverage check passes; the seven performance/safety checks fail.
- **AI recall:** FLUX.1 **12.77%**, SD VAE EMA **16.74%**, SD VAE MSE **18.27%**, SDXL VAE
  **18.61%**, SD 2.1 **18.21%**, SD 3.5 Large **15.43%**.
- **Post-hoc diagnosis only:** the best possible pooled balanced point is just **0.53159** at
  threshold 0.32859, with REAL FP **43.26%** and AI TPR **49.58%**. Per-variant best balanced
  accuracy ranges 0.50262–0.54804. This rules out threshold repair and exposes a representation /
  causal-generalization failure on content/frequency-aligned reconstructions.
- **Decision:** preserve the failed first result. DDA-COCO is now consumed DEVELOPMENT for E44; do
  not retry, tune on it and call it final, or promote E43-S. Train on distinct paired parents with
  generator-held-out validation while retaining real-camera/RR regression safety.

### E44-A — official DDA specialist screen frozen before scores

- **Reason:** the E43-S scalar is near chance on content/frequency-aligned pairs, but the disk
  already contains the official 1.26 GB DINOv2-L + rank-8 LoRA detector trained on 118K DDA pairs.
  Recreating a weaker VAE-only dataset before isolating this existing capability is unjustified.
- **Method:** select 700 complete DDA parents solely by SHA-256 namespace, retain all seven views,
  and score the published checkpoint/cut once. E43 scores, labels beyond required pairing and image
  appearance cannot affect selection. The screen is consumed comparative DEVELOPMENT.
- **Gate:** coverage 1.0, pooled AUC >=0.85, balanced accuracy >=0.80, REAL FP <=20%, core four-
  reconstruction macro recall >=80%, all-six macro >=70% and worst variant >=40%. Passing opens
  representation adaptation; failing opens separate full DDA pair construction.
- **Bound contract:** 700 parents /4,900 rows were selected without scores. The selected-parent-list
  SHA-256 is `b1ac6bb2...1990`; detailed contract SHA-256 is `df256498...5ce9`. It binds the frozen
  manifest `e663d679...a3db`, official checkpoint `b27a31d3...e3e`, threshold 0.5 and all gates.
  `model_scores_created` is zero at this checkpoint.
- **Fixed result:** all 4,900 rows scored successfully at threshold 0.5. AUC **0.99006**, balanced
  accuracy **0.93917**, EER **0.04845**, REAL FP **0.86%** and pooled AI recall **88.69%**. Core-
  variant macro recall is **98.61%**, all-six macro recall **88.69%** and worst recall **64.57%**.
  All seven preregistered gates pass. Score-stream SHA-256 is `3d24d1c1...31d75`; detailed report
  SHA-256 is `a57e001d...090e`.
- **Variant recall:** FLUX.1 64.57%, SD VAE EMA 99.71%, SD VAE MSE 99.71%, SDXL VAE 95.29%,
  SD 2.1 99.71% and SD 3.5 Large 73.14%. The official representation supplies the aligned-
  reconstruction capability missing from E43-S, but weaker modern-flow recall and prior smartphone
  false positives prohibit serving it alone.
- **Decision:** retain the official DDA representation as an E44 specialist and move to a frozen,
  conservative fusion/adaptation using existing training roles plus real-camera regression gates.
  Do not download replacement VAE data. This consumed screen is diagnostic, not final evidence.

### E44-B — two-score fusion method frozen before missing joint scores

- **Inputs:** unchanged E43-S generalist probability and official DDA specialist probability only.
  The fit may not inspect filename, source, device or label at inference. Existing consumed E44 and
  E35 populations supply aligned AI/REAL, RR, IPN and owner-gallery coverage; no new images transfer.
- **Roles:** assign complete DDA/RR parents and whole IPN devices by namespaced SHA-256. The owner
  gallery is development-only. This prevents sibling-view and device leakage and prevents personal
  images from teaching the detector their own appearance.
- **Model:** clamp each probability before logit conversion, then fit one source/label-weighted
  `StandardScaler + LogisticRegression`. Select one CAL threshold under real-safety and AI-recall
  constraints; freeze it before DEVELOPMENT aggregation.
- **Gate:** coverage 1.0, AUC >=0.90, balanced accuracy >=0.85, DDA macro/worst recall >=75%/50%,
  RR macro/worst AI recall >=80%/60%, RR REAL FP <=10%, IPN worst-device FP <=20%, owner FP <=20%.
  A failure keeps the experts separate; it cannot be repaired by a post-hoc threshold.
- **Bound identities:** 4,900 DDA rows and 1,670 E35 RR/IPN/owner rows were bound before the missing
  E43-S scores. Detailed contract SHA-256 is `25681b62...3fb4`; E35 byte-identity SHA-256 is
  `919a0586...6b10` and DDA role-map SHA-256 is `330000bf...8547`. The owner images are all
  development-only and `model_scores_created` remains zero.
- **Joint score completion:** E43-S scored all 1,670 frozen E35 rows locally with 100% coverage;
  the 455,631-byte stream SHA-256 is `35d9d2c2...ad5af`. No fusion head was fit and no DEVELOPMENT
  aggregate was read at this checkpoint.
- **Frozen candidate:** the two-logit head fit 3,657 FIT rows and selected threshold
  `0.3423850493` on 1,307 CAL rows. CAL AUC is 0.96936, balanced accuracy 0.91637, REAL FP 7.22%,
  AI group-macro recall 95.38% and worst recall 68.00%. Candidate SHA-256 is
  `19fd7bbc...b100`; `development_scores_created` is zero.
- **One-shot DEVELOPMENT result:** 1,606/1,606 rows scored. AUC **0.97165**, balanced accuracy
  **0.91099**, pooled REAL FP **9.54%**, pooled AI recall **91.74%**, DDA macro/worst recall
  **91.33%/74.67%**, RR AI macro/worst **99.29%/95.00%** and IPN worst-device FP **1.25%**.
  Score-stream SHA-256 is `ce10c43f...67969`.
- **Gate:** failed 2/10 checks. RR REAL FP is **12.00%** (6/50; limit 10%) and owner-gallery FP is
  **20.48%** (43/210; limit 20%). Each miss is exactly one image, but that does not authorize a
  threshold repair. The eight other coverage/ranking/AI/IPN checks pass.
- **Decision:** E44-B is not served as a universal scalar. Preserve it as a near-pass and design
  E44-C on a separately bound local population that has not received official-DDA scores; an
  untouched ITW-SM/NIST evaluation is still required for a final claim.

### E44-C — successor hypothesis before new DDA scores

- **Consumed diagnostic:** the minimum cut that would reduce E44-B RR REAL FP from 6/50 to 5/50
  and owner FP from 43/210 to 42/210 is `0.3477933653`. At that cut the old rows pass 10/10 while
  DDA macro/worst recall remains 91.22%/74.67% and RR AI macro/worst remains 99.29%/95.00%.
  This is explicitly not a repaired E44-B result.
- **New population:** freeze the diagnostic cut before official-DDA access to 2,940 E43 RR
  DEVELOPMENT views and 2,160 clean/hash-assigned-robust views from 1,080 E42 E36/E39 parents.
  E42 IPN/owner parents are excluded because E35 already scored them. Reject every E35 exact-byte
  overlap; use the already-frozen E43-S scores as the other arm. No image download is needed.
- **Bound contract:** 5,100 rows /4,020 unique paths passed exact-byte verification with zero E35
  overlap. Detailed contract SHA-256 is `b3c399e9...e1152`; population SHA-256 is
  `ac79ea36...89aa3`. Threshold `0.3477933653`, candidate `19fd7bbc...b100` and all gates are bound;
  `dda_scores_created` is zero.
- **DDA arm complete:** all 5,100 bound views scored successfully. The 1,793,353-byte stream
  SHA-256 is `3618b158...d3108`; coverage is 1.0. Fusion metrics had not been aggregated at this
  checkpoint.
- **One-shot result:** E44-C passed **20/22** frozen checks. Pooled AUC **0.98181**, balanced
  accuracy **0.92731**, AI recall **96.85%**, REAL FP **11.39%** and EER **0.07373**. The fused
  score stream SHA-256 is `0507cc4d...3d6bd`.
- **Strong cells:** RR original AI macro/worst recall 99.80%/98.57%; E42 clean AUC/BA
  0.99307/0.95501 with AI macro/worst 99.04%/95%; E42 robust AUC/BA 0.99198/0.96538 with AI
  macro/worst 98.08%/90%. All robustness, ranking, coverage and AI-recall gates pass.
- **Failed safety cells:** RR-original REAL FP is **16.33%** against 10%; E42 clean REAL macro FP
  passes at 6.94%, but `e36:device_004` reaches **31%**, making worst-device FP fail the 20% cap.
  The failure is camera-specific DDA overactivation, not weak modern-AI recognition.
- **Decision:** preserve the failed result. Stop scalar-threshold iteration and design selective
  classification with an explicit `UNCERTAIN` outcome; future evidence must report risk and
  automatic coverage together on a genuinely new population.

### E44-D — selective-policy post-hoc design, not validation

- **Population:** union of consumed E44-B and E44-C DEVELOPMENT, 6,706 rows (3,240 REAL /3,466 AI).
  Existing score streams only; no model rerun, refit or new byte.
- **Fixed design budgets:** choose the smallest AI cut with REAL group macro/worst FP <=5%/10%,
  and the largest REAL cut with AI group macro/worst false-REAL <=10%/20%. Scores between cuts
  abstain. This yields REAL `<0.2545712170`, AI `>=0.6938513176`, otherwise `UNCERTAIN`.
- **Consumed diagnostic:** automatic coverage **87.40%**, abstention **12.60%**, covered accuracy
  **96.47%**. It auto-decides 2,852 REAL and 3,009 AI, leaving 845 uncertain. REAL group macro/worst
  false-AI is **2.11%/10%**; AI group macro/worst false-REAL is **2.36%/20%**. Automatic AI recall
  is 83.81% and automatic REAL recall is 84.85% over all labeled rows.
- **Boundary:** this is a post-hoc hypothesis on consumed data, not a passed model or deployable
  threshold. Freeze it before ITW-SM/NIST or a genuinely new source-separated validation.

### E45 — official MediaEval/ITW-SM final contract frozen before transfer

- **Question:** can the fixed E44-D three-outcome policy retain high automatic coverage while
  avoiding the camera-specific false-AI errors seen in E44-C on genuinely in-the-wild images?
- **Source:** the official MediaEval 2026 SID repository links a labeled 10,000-image validation
  archive (5,000 REAL/5,000 synthetic). HTTP identity is frozen at 3,553,693,205 bytes, ETag
  `"68555a02-d3d10e15"` and Last-Modified `Fri, 20 Jun 2025 12:54:26 GMT`.
- **Candidate:** unchanged fusion artifact `19fd7bbc...b100`; REAL `<0.2545712170`, AI
  `>=0.6938513176`, middle `UNCERTAIN`. No E45 image, label or score may alter these values.
- **Gates:** score coverage 1.0; binary AUC >=0.90 /balanced accuracy >=0.85 /REAL false-AI <=10%
  /AI recall >=80%; source worst REAL false-AI <=20% /AI recall >=60%; selective automatic
  coverage >=80% /covered accuracy >=95% /uncertain <=20%.
- **Boundary:** source and gate contract only. E45 downloaded bytes, decoded images and model scores
  are all zero. Acquisition, safe inventory, decontaminated manifest and a second score lock must
  complete in that order. The official archive and gated HF snapshot may be the same distribution
  and can never be counted as two independent finals without identity evidence.
- **Acquisition result:** exact HTTP identity and 3,553,693,205-byte transfer passed; archive
  SHA-256 is `18f1806e...b6e3`. The central directory exposes all 10,000 declared class paths.
- **CRC result:** 9,999 members decompress successfully. Only `ITW-SM/1_fake/x_618.jpg` fails with
  `invalid stored block lengths`; a fresh HTTP byte-range is identical to the local compressed
  region, establishing publisher-artifact corruption rather than transfer corruption. Preserve the
  failure, exclude that member before manifest/model access and report official coverage 99.99%.
  Zero image decode and zero model score still hold.
- **Manifest result:** all 9,999 CRC-usable members decoded. Exclude 19 second copies from
  same-label exact duplicate pairs and two protected-dHash AI overlaps; zero cross-label exact
  duplicate exists. The frozen 9,978 rows contain 4,981 REAL /4,997 AI across all four publisher-
  named platforms. Detailed manifest SHA-256 is `3e7c1d7e...d7e03`; official coverage is 99.78%.
- **Boundary:** 141 within-E45 exact-dHash groups remain a disclosed similarity diagnostic, not an
  automatic identity judgment. No protected exact/dHash overlap survives, and zero model score
  exists. Commit this manifest before binding or loading E44-D.
- **Score lock:** detailed contract SHA-256 `4a5d4999...9ac83` binds the 9,978-row manifest,
  unchanged E43-S /official-DDA /fusion artifact hashes, binary threshold `0.3477933653`, E44-D
  selective cuts and all ten gates. Confidence intervals use 10,000 platform-and-label-stratified
  row bootstraps at seed 45. Model score rows are still zero.
- **Generalist arm:** E43-S completed 9,978/9,978 rows with 100% coverage. The 1,338,053-byte
  score stream SHA-256 is `43ecaa3f...fc171`. It was written in resumable, manifest-order batches;
  every ZIP payload hash matched. No specialist or fused result existed at this checkpoint.
- **Specialist arm:** official DDA completed 9,978/9,978 rows with 100% coverage. The 1,338,964-
  byte score stream SHA-256 is `88946986...69bb7`. The first 200 rows at batch 2 and the remainder
  at batch 8 used the same frozen model/transform; the resumable prefix prevented recomputation.
  Every payload hash matched, and the fused result remained unopened at this checkpoint.
- **Reporting interruption:** immutable fusion produced 9,978 rows /2,142,780 bytes /SHA-256
  `b84f8c40...3c67e`, then the metric helper rejected row 0 because it requires field `source`
  while the E45 schema names the same publisher group `platform`. No metric/report was created.
  Preserve that fused stream; resume only the reporter with the deterministic `source=platform`
  alias after committing the fix. Do not rerun either model or rewrite fused scores.
- **One-shot final result:** reporting resumed from the byte-identical fused stream after the
  committed schema alias. E45 failed 4/10 gates. AUC **0.95020**, TPR@FPR10 **0.87352** and AI
  recall **95.40%** pass, but balanced accuracy is **0.80634** and REAL false-AI is **34.13%**.
  Confusion is TN 3,281 /FP 1,700 /FN 230 /TP 4,767.
- **Platform transfer:** REAL false-AI is Facebook **39.30%**, Instagram **32.50%**, LinkedIn
  **32.25%**, X **32.11%**; all exceed the frozen worst-platform 20% budget. AI recall remains
  strong on every platform: 98.16%, 97.98%, 90.01% and 91.36%, respectively.
- **Selective result:** automatic coverage **80.54%** and uncertainty **19.46%** pass, but covered
  accuracy is only **90.07%** versus 95%. Its 95% bootstrap interval is 89.43–90.71%; the binary
  AUC interval is 94.62–95.41% and REAL false-AI interval 32.82–35.45%. This is not sampling noise.
- **Decision:** preserve `e45_independent_final_failed`; E45 is consumed forever and cannot tune a
  threshold or train E46. The causal target for the next, separately sourced development cycle is
  social-platform REAL safety while retaining modern-AI recall—not another E44 threshold sweep.

### E46 — consumed E45 arm diagnosis

- **Generalist:** AUC 0.80108, balanced accuracy 0.72970, REAL false-AI 21.22%, AI recall 67.16%.
- **Official DDA:** AUC 0.94010, balanced accuracy 0.87255, REAL false-AI 10.74%, AI recall 85.25%.
  It is substantially safer than the fusion and only narrowly misses the pooled 10% REAL budget.
- **Frozen fusion:** AUC 0.95020 and AI recall 95.40% improve, but its old cut raises REAL false-AI
  to 34.13%. Of 1,700 fusion false alarms, 763 are generalist-only positives, 345 specialist-only,
  140 both-positive and 452 fall below both arms' individual cuts; the learned fusion intercept and
  continuous margins do not transfer to social-media scores.
- **Forbidden post-hoc operating point:** the consumed-final REAL 90th percentile is
  `0.7541002115`. At that cut all six binary gates would pass: balanced accuracy 0.88667, REAL
  false-AI 10.02%, AI recall 87.35%, worst platform REAL false-AI 12.77% and worst AI recall
  72.40%. This is diagnostic proof of calibratability, not a valid repaired model.
- **Decision:** do not retrain a backbone first and do not deploy the post-hoc cut. E46 needs a
  distinct, licensed in-the-wild CAL population to estimate transfer-safe cuts, plus another
  untouched FINAL. Detailed diagnostic SHA-256 is `2ba9234a...41164`.

### E46-A — cross-platform calibration recovery pre-registration

- **Causal hypothesis:** E45 failed mainly because a development-domain fusion cut transferred
  poorly to social-media scores, not because both encoders lost ranking information. A cut or
  compact quality-conditioned calibration learned on an independent social population should
  recover REAL safety while preserving enough modern-AI recall.
- **Development role:** official SynthWildX `list.csv`, 2,000 X-hosted rows (500 REAL; 500 each
  DALL-E 3, Midjourney v5, Firefly), deterministically split by label/generator into 60% CAL and
  40% DEVELOPMENT before score creation. Unavailable URLs are failures, never substitutions.
- **Candidate comparison:** unchanged official DDA, frozen E44 fusion, a CAL-only global REAL-safe
  cut, and a compact QuAD-inspired quality-conditioned calibration. No backbone retraining.
  Prefer the simplest method unless the quality model materially improves REAL safety and AI
  retention on DEVELOPMENT. Minimum development gates: pooled REAL FP <=20%, worst reported REAL
  group FP <=25%, pooled AI recall >=80%, worst generator recall >=60%.
- **Untouched proof:** official TrueFake Facebook archive, advertised 3.9 GB and containing 60,000
  Facebook-processed rows. After byte/structure/overlap audit, freeze a score-blind balanced sample
  of 1,000 REAL and 1,000 AI across available origins/generators. Bind candidate and E45-style gates
  before the one permitted final score. E45 remains consumed and cannot repair E46.
- **Stop rules:** source-list drift, ambiguous labels, unacceptable licence, unrecoverable archive,
  excessive protected overlap, CAL/DEVELOPMENT leakage or a failed development gate stops the
  claim. Passing the easier development source is not success; only transferred TrueFake results
  can support the new checkpoint, and they do not erase the archived E45 failure.
- **Score-blind acquisition:** the immutable 206,017-byte publisher list matched SHA-256
  `a40a374e...a28188`. Valid payload recovery is 1,723/2,000 rows (86.15%) /553,125,164 bytes; 277
  rows are disclosed persistent 403/404 failures. Surviving CAL/DEVELOPMENT counts are 1,034/689,
  including 250/168 REAL. All three AI generators retain at least 159 development rows. Two exact
  hash groups are quarantined for the next identity audit. Manifest SHA-256 after external-store
  relocation is `fd8008a...a89f3f`; image bytes changed by relocation: zero; scores created: zero.
- **TrueFake acquisition:** the Facebook archive completed at 4,207,525,545 bytes /SHA-256
  `413cb7f9...cda0d63`. Full gzip integrity and TAR enumeration pass. Exactly 60,000 JPG payloads
  are present: 20,000 REAL evenly split FFHQ/FORLAB and 40,000 AI evenly split across eight
  publisher generators. This confirms all declared strata before sampling. No member was extracted
  and no detector loaded at this checkpoint; final selection remains hash-only.
- **Audit interruption (no result):** the first SynthWildX identity-audit invocation stopped before
  writing output because the acquisition rows intentionally contained SHA-256 but not dHash.
  The audited-manifest stage now decodes each validated local image and computes dHash itself;
  focused coverage was added. No row, role, score or source byte changed.
- **Identity result:** 15/1,723 recovered SynthWildX rows were removed before scoring—two repeated
  exact payload copies and thirteen protected exact/dHash overlaps. The remaining 1,708 rows are
  CAL 1,024 /DEVELOPMENT 684 with 415 REAL and 396–472 rows per AI generator. Audited manifest
  SHA-256 is `953490a9...da8d4`; model scores remain zero.
- **Final selection lock:** the verified 60,000-member TrueFake inventory facts hash to
  `b59e78de...8ba28b`. Contract SHA-256 `1e77dfbd...cead3` fixes a 3,500-row score-blind reserve,
  the lowest-rank-clean rule, 500-image quotas for each REAL origin and 125-image quotas for each
  of eight AI generators. No final payload is decoded or scored at this checkpoint.
- **Final manifest:** all 3,500 hash-ranked reserve members decoded; zero failures, protected
  overlaps, exact duplicate groups or dHash duplicate groups were observed. The frozen final is
  exactly 2,000 rows: 500 FFHQ +500 FORLAB REAL and 125 from each of eight AI generators. Detailed
  manifest SHA-256 is `4572339e...b225b`; model-score rows remain zero. This is the last permitted
  checkpoint before E46 candidate selection begins on SynthWildX only.
- **Development score lock:** contract SHA-256 `b3fe31a3...5c98c` binds the 1,708-row clean
  SynthWildX manifest, E43-S, official DDA and E44 fusion hashes, CAL/DEVELOPMENT roles, three
  pre-registered candidate methods, three inexpensive quality proxies and the four development
  gates. Counts are CAL 1,024 /DEVELOPMENT 684; model scores remain zero. TrueFake is not read.
- **Generalist arm:** E43-S scored all 1,708 rows with 100% coverage while also recording the three
  frozen quality proxies. The 394,197-byte stream SHA-256 is `8be0aefd...ce88d`. The specialist,
  fusion and candidate comparison remain unopened; no TrueFake row was read.
- **Specialist arm:** official DDA scored the same 1,708 identities with 100% coverage. The
  268,844-byte stream SHA-256 is `a7fbd7e2...257eda`. Both immutable arms now exist; no fusion,
  calibration, DEVELOPMENT metric or TrueFake score has yet been produced.
- **Calibration-method lock:** contract SHA-256 `6799231f...c9228c` assigns the 1,024 CAL rows
  source-stratified to QUALITY_FIT 612 and OPERATING_CAL 412, including 148/100 REAL. It freezes
  ridge-1.0 quality-Gaussian fitting, pooled REAL-10% thresholding, eligibility, the conservative
  quality-replacement test and CAL-only selective-band search. Score rows read: zero;
  DEVELOPMENT/TrueFake reads: zero.
- **CAL-only candidate:** all three methods were fit/thresholded exactly as frozen. DDA gives
  AUC/BA/REAL-FP/AI-recall/worst-generator 0.95670/0.90032/10.0%/90.06%/81.42%; global fusion gives
  **0.97362/0.91795/10.0%/93.59%/84.07%**. Quality Gaussian raises recall to 96.79% and worst
  generator to 92.04%, but AUC falls to 0.97120; it fails the registered all-metric
  non-inferiority condition. Select `fusion_global`, threshold `0.6688565013`, candidate SHA-256
  `9fec91b8...b84a1`. CAL selective cuts are REAL `<0.5185430496`, AI `>=0.6688565013`, yielding
  97.09% coverage at exactly 95% covered accuracy. DEVELOPMENT rows read: zero; final scores: zero.
- **One-shot DEVELOPMENT:** the unchanged `fusion_global` candidate passes all four registered
  gates on 684 rows. AUC **0.97203**, balanced accuracy **0.91217**, REAL false-AI **11.38%**, AI
  recall **93.81%**; DALL-E 3 /Firefly /Midjourney v5 recalls are 98.74% /84.82% /99.40%.
  The score stream is 110,362 bytes /SHA-256 `e9443455...24c9b8`. Selective coverage is 96.49%,
  but covered accuracy is 94.39%; preserve this near-miss and do not tune from DEVELOPMENT.
  TrueFake scores created: zero.
- **Final score lock:** contract SHA-256 `1cf28d2d...7c4262` binds the exact 2,000-row TrueFake
  manifest, E46 candidate and both inference-arm hashes, global-fusion threshold `0.6688565013`,
  selective cuts, ten E45-style gates and 10,000 source/label-stratified bootstraps at seed 46.
  Forbidden actions include row removal, repair, refit and retry. Model scores created: zero.
- **Final generalist arm:** E43-S completed 2,000/2,000 TrueFake rows with full coverage. The
  298,721-byte stream SHA-256 is `43eb1562...b5f25c`. DDA and fused metrics remain unopened; this
  checkpoint records inference output only, not a partial final claim.
- **Final specialist arm:** official DDA completed the same 2,000 rows with full coverage. Its
  298,587-byte stream SHA-256 is `13947caf...878d0b`. Both final arms are immutable and complete;
  fusion, metrics and gates remain unopened until this checkpoint is committed.
- **E46 one-shot independent final:** after both inference streams were committed, the fixed global
  fusion opened the 2,000 TrueFake Facebook labels once. Coverage is 100%. AUC is **0.815477**
  (95% bootstrap CI 0.802114–0.828992), balanced accuracy **0.73450** (0.72350–0.74550), REAL
  false-AI **5.60%** (4.20–7.00%), and AI recall **52.50%** (50.80–54.20%). Source behavior is
  highly non-uniform: FFHQ/FORLAB false-AI is 1.0%/10.2%; AI recall is FLUX.1 66.4%, SD1.5 100%,
  SD2 96.8%, SD3 51.2%, SDXL 99.2%, StyleGAN 1.6%, StyleGAN2 1.6%, StyleGAN3 3.2%.
  The selective band covers 95.85% (95.0–96.7%) but covered accuracy is only 74.86%
  (73.74–75.94%). The candidate passes 5/10 registered gates and therefore **fails** E46.
  The isolated scientific diagnosis is a strong GAN-family domain blind spot rather than a REAL
  safety collapse. Result evidence SHA-256 is `e7e14fdf...d7a7ed`; fused score stream SHA-256 is
  `6a51a9b1...68c97`. No retry or post-final threshold tuning is permitted.

### E47-R1 — pre-registered GAN recovery diagnostic

- **Question:** does the project's untouched, hash-pinned GenImage ResNet-18 contain a complementary
  GAN signal that E43-S + DDA missed after Facebook transport?
- **Data boundary:** the consumed 2,000-row E46 TrueFake final may be scored only as an explicitly
  post-final diagnostic. It cannot fit or validate an E47 threshold and can never become final
  evidence again.
- **Frozen arm:** `best_genimage.pt`, original deterministic resize-to-224 and ImageNet
  normalization. Measure pooled and per-generator AUC/recall at a diagnostic pooled-REAL 10% cut,
  plus OR-recovery against the frozen E46 fusion.
- **Unlock rule:** proceed with this family only when mean StyleGAN/2/3 recall is >=50%, each is
  >=30%, and OR-fusion pooled REAL false-AI is <=15%. Otherwise acquire one official frozen GAN
  detector before constructing new CAL/DEVELOPMENT data.
- **Result:** reject the legacy arm. At its post-hoc diagnostic pooled-REAL 10% cut, AUC is 0.6624
  and AI recall 23.6%. StyleGAN/2/3 recall is only 4%/8%/8% (per-source AUC 0.528/0.561/0.658).
  It recovers 60/475 E46 misses, but OR-fusion reaches only 58.5% AI recall and raises REAL
  false-AI from 5.6% to 15.1%. All three pre-registered unlock checks fail. Stream SHA-256 is
  `e4bbcde8...c7cb27`; report SHA-256 is `bcf7239e...b465de4`.

### E47-R2 — official UnivFD specialist acquisition and score lock

- **Rationale:** UnivFD freezes CLIP ViT-L/14 and learns only a linear head from ProGAN, explicitly
  targeting cross-generator asymmetry. It is therefore a materially different representation from
  E43-S/DDA and the failed GenImage CNN.
- **Artifacts:** official repository commit `030495a...c619`; MIT licence; linear-head SHA-256
  `47710074...c7847`; OpenAI CLIP backbone SHA-256 `b8cca3fd...03836`. The 72 GB training set was
  not acquired. A two-row smoke test produced finite hidden scores but opened no performance
  metric.
- **Frozen diagnostic:** original official center-crop/CLIP normalization and sigmoid head output;
  exact consumed E46 identities; the unchanged R1 10%-REAL diagnostic cut and unlock rule. Scores
  can select only whether this representation deserves new R3 data, never repair E46.
- **Result:** the representation is genuinely complementary but narrowly fails the full unlock.
  At the diagnostic pooled-REAL 10% cut, StyleGAN/2/3 recall is **94.4%/74.4%/80.0%** and their
  one-vs-REAL AUC is 0.977/0.936/0.952. Diffusion recall is intentionally weak (3.2–16.8%), making
  this a specialist. It recovers 310/475 E46 misses; OR-fusion AI recall reaches **83.5%**, but
  pooled REAL false-AI is **15.5%**, 0.5 points above the frozen 15% diagnostic ceiling. Therefore
  2/3 checks pass and UnivFD is not directly admitted. Stream SHA-256 is `faff2592...9104e`;
  report SHA-256 is `ca0c1d4c...782ce`. Next: evaluate the pre-authorized UNINA specialist.

### E47-R2b — UNINA StyleGAN2 ResNet50-NoDown score lock

- **Artifact:** official repository commit `543943c...df88`; 282,549,121-byte StyleGAN2-trained
  checkpoint SHA-256 `65467594...d5a08`; original native-resolution/ImageNet-normalized inference.
  The informational/nonprofit-only licence forbids treating this as an unrestricted product arm.
- **Pre-score state:** official real/fake examples produce finite logits in the documented
  direction (fake minus real +49.18). No E46 performance metric was opened. Use the unchanged R1
  diagnostic identities and unlock rule; reject it if REAL safety is not better than UnivFD.
- **Runtime amendment before metrics:** native-resolution processing was stopped at 655/2,000
  because 960 px throughput fell below 0.5 image/s, incompatible with a web detector. No aggregate,
  label-rate or threshold metric was opened. Preserve partial SHA-256 `87417d5f...733a4`, exclude
  it, and restart all rows with an aspect-preserving 512 px long-side Lanczos cap. This fixed cap is
  the only allowed UNINA variant and is bound before any result.
- **Result:** capped UNINA has pooled AUC 0.8408 and 61.5% AI recall at the diagnostic 10% REAL cut.
  Its intended domain is excellent: StyleGAN/2/3 recall **100%/94.4%/73.6%**, with one-vs-REAL AUC
  0.999/0.972/0.920. It recovers 375/475 E46 misses and diagnostic OR AI recall reaches **90.0%**.
  Yet 99 additional REAL errors make OR REAL false-AI **15.5%**, again failing the frozen 15% cap.
  Therefore neither specialist is directly admitted. Stream SHA-256 `17251695...6d99e6`; report
  SHA-256 `280b973a...ff7fa`. Scientific conclusion: use new CAL/DEVELOPMENT to learn when to trust
  the specialist, not another backbone or a post-final threshold.

### E47-R3 — frozen CAL/DEVELOPMENT design before extraction

- **Population:** 2,400 identities from TrueFake Facebook members outside the entire prior 3,500-
  member reserve. Namespace `E47_TRUEFAKE_CALDEV_V1`; 20% reserve headroom; exact/dHash audit
  before any detector loads.
- **Source-held roles:** CAL = 600 FFHQ REAL +200 StyleGAN2 +200 SD1.5 +200 SDXL. DEVELOPMENT =
  600 FORLAB REAL +200 StyleGAN +200 StyleGAN3 +100 FLUX.1 +100 SD3. This deliberately transfers
  both pristine-real origin and GAN generation.
- **Candidates:** frozen E46, regularized E46+UnivFD, E46+UNINA and all-three logistic gates. Fit
  and threshold on CAL only; backbones never update. DEVELOPMENT gates remain pooled REAL FP <=10%,
  worst REAL <=20%, pooled AI recall >=80%, worst AI source >=60%, AUC >=0.90 and BA >=0.85.
  Prefer the MIT candidate when it passes and trails nonprofit UNINA by <=2 recall points.
- **Selection lock:** 2,880 score-blind reserve members were hash-ranked outside every E46 reserve
  identity, 1,440 per role. Contract SHA-256 `c031ef92...d0753`; target 2,400; decoded images and
  model scores both zero. Next step is decode + protected exact/dHash audit only.
- **Identity result:** all 2,880 candidates decoded; one SD1.5 reserve row was removed for a
  protected dHash overlap. The selected manifest reaches 2,400 rows with exact 600/600 class
  balance inside both CAL and DEVELOPMENT. Manifest SHA-256 `378b83fe...85739`; scores zero.
- **Four-arm score lock:** contract SHA-256 `ee2a2958...95798` binds all 2,400 rows and exact E43-S,
  DDA, E44 fusion, UnivFD and capped-UNINA artifacts before any model load. Roles and backbones are
  immutable; score count zero.
- **E43-S arm:** 2,400/2,400 rows, 100% coverage; 399,398-byte stream SHA-256
  `073110f4...f30c03`. Other arms and all CAL/DEVELOPMENT metrics remain unopened.
- **DDA arm:** 2,400/2,400 rows, 100% coverage; 399,187-byte stream SHA-256
  `8001c60b...d75f5`. UnivFD/UNINA and all fitted results remain unopened.
- **UnivFD arm:** 2,400/2,400 rows, 100% coverage; 403,436-byte stream SHA-256
  `67b7b94c...e2829`. UNINA and all CAL/DEVELOPMENT results remain unopened.
- **Capped UNINA arm:** 2,400/2,400 rows, 100% coverage; 398,998-byte stream SHA-256
  `7efb36c0...5e16d`. All four frozen inference streams are complete. CAL/DEVELOPMENT
  metrics and fitted results remain unopened pending a separately committed decision rule.
- **Decision rule queued before metrics:** compare frozen E46, E46+UnivFD, E46+UNINA and
  E46+both using C=0.1 standardized logistic gates with equal CAL label/source mass. Each
  candidate receives the lowest CAL threshold satisfying pooled/worst REAL FP <=10%/20%
  and must also pass AUC/BA >=0.90/0.85 plus pooled/worst AI recall >=80%/60%. Rank passes
  by worst AI recall, pooled recall, AUC and BA; prefer MIT-only E46+UnivFD when it is within
  two points of an UNINA-bearing winner on both AI-recall measures. DEVELOPMENT is one-shot.
- **Decision contract lock:** the 2,385-byte contract SHA-256 is `a4515caf...875a`; it
  binds all four score-stream hashes, exact features, C=0.1 head, seed 47, CAL threshold
  rule, candidate ranking/licence rule and DEVELOPMENT gates before score interpretation.
  Six focused tests pass; CAL/DEVELOPMENT metrics opened: zero/zero.
- **CAL result and candidate freeze:** E46 alone fails at AUC 0.8735 /BA 0.8175 /73.5%
  AI recall and 20.5% StyleGAN2 recall. E46+UnivFD improves to 85.67% pooled AI but
  misses the frozen worst-source floor by one point (59% vs 60%). E46+UNINA passes;
  E46+UnivFD+UNINA ranks first at threshold `0.3353660721`, AUC 0.98969, BA 0.93667,
  10% REAL FP, 97.33% pooled AI recall and 95% worst AI-source recall. The 2,330-byte
  candidate SHA-256 is `f659ee4f...0b0d`; DEVELOPMENT metrics remain unopened.
- **One-shot DEVELOPMENT:** the frozen candidate passes six of seven gates on 1,200 rows:
  AUC 0.95491, BA 0.88417, pooled REAL false-AI 7.33% and pooled AI recall 84.17%.
  StyleGAN/StyleGAN3/SD3 recall is 100%/87.5%/84%, but held-source FLUX.1 recall is
  46%, below the frozen 60% worst-source gate. Therefore E47-R3 fails despite solving the
  GAN blind spot. Score stream SHA-256 `97fbe4b7...72cd`; no refit, threshold repair or
  retry on this consumed DEVELOPMENT is permitted.
- **Post-failure diagnostic queued:** inspect frozen E46, UnivFD and UNINA source-level
  score behavior on the now-consumed DEVELOPMENT only to determine whether the selected
  logistic gate vetoed an existing FLUX signal. This may inform E48's architecture but cannot
  select a threshold/candidate, repair E47, update serving or support a success claim.
- **Diagnostic result:** at the E46 CAL-only threshold, E46 detects 95% of held FLUX but
  falsely flags 30.67% of held FORLAB. The selected gate cuts FORLAB FP to 7.33% and fixes
  GAN recall, yet vetoes 50/95 E46 FLUX hits, leaving 46%. It also rescues 149 StyleGAN and
  141 StyleGAN3 rows over E46. Therefore E48 needs diverse REAL/diffusion CAL and a non-veto
  router or conditional union; these consumed rows cannot select that successor.

### E48 — monotone non-veto successor (pre-registered before selection)

- **Hypothesis:** frozen E46 supplies diffusion evidence and UnivFD/UNINA supply GAN evidence;
  mapping each to a FIT-REAL empirical percentile and taking their maximum preserves either signal
  without a negative specialist veto.
- **Fresh roles:** 600 FIT, 600 CAL and 1,200 DEVELOPMENT rows, balanced per role. FIT/CAL use
  unused VISION+CSAFE REAL and fresh TrueFake FLUX/StyleGAN2/SD1.5/SDXL. DEVELOPMENT holds REAL
  source to unused FODB and AI to fresh FLUX/SD3/StyleGAN/StyleGAN3 identities.
- **Frozen selection/gates:** compare four nested monotone expert sets; threshold and select on
  CAL only. Coverage=1, AUC>=.90, BA>=.85, pooled/worst-device REAL FP<=.10/.20 and pooled/worst-
  source AI recall>=.80/.60. Freeze before one-shot DEVELOPMENT; no repair or serving update.
- **Selection lock:** 2,880 score-blind candidates for a 2,400-row target, balanced by class
  inside 720 FIT, 720 CAL and 1,440 DEVELOPMENT reserve rows. All 6,380 prior E46/E47
  TrueFake candidates plus current E32 training identities are excluded. Contract SHA-256
  `dbb6f4aa...0e6e`; decoded new AI payloads and model scores remain zero.
- **Pre-score audit amendment:** the first extraction stopped because the legacy R1b ledger
  lists the entire 22,688-row candidate plan and masks every camera candidate, not just images
  consumed by the current model. Remove only that planning ledger from protected-role hashing;
  keep explicit E42 current-training IDs and all actual CAL/DEVELOPMENT/final manifests excluded.
  Candidate identities/quotas are unchanged; model scores remain zero.
- **Pinned-dHash compatibility:** camera SHA-256 values reproduce exactly, but a newer helper's
  EXIF orientation path does not reproduce the historical realization dHash. Keep the previously
  decoded/pinned audit dHash for overlap comparisons and verify current bytes by SHA-256. No row,
  role, quota or score changes.
- **Identity result:** 2,880/2,880 candidates verify/decode with zero failures. One VISION and
  one FODB reserve row are excluded for protected dHash overlap; headroom fills all quotas.
  The frozen 2,400-row manifest is exactly class-balanced inside 600 FIT, 600 CAL and 1,200
  DEVELOPMENT rows. Manifest SHA-256 `1404a3ff...5b68`; model scores remain zero.
- **FIT+CAL score lock:** contract SHA-256 `ea7de06c...9516` binds the exact manifest and
  frozen E43-S/DDA/E44/UnivFD/capped-UNINA identities. Scoring is restricted to 600 FIT and
  600 CAL; all 1,200 DEVELOPMENT rows remain explicitly forbidden. Six focused tests pass;
  model and DEVELOPMENT score counts are zero.
- **E43-S FIT+CAL arm:** 1,200/1,200 rows with full coverage; 258,603-byte stream SHA-256
  `f2a1be3b...137cf7`. Raw inference only; DEVELOPMENT rows and aggregate metrics remain zero.
- **DDA FIT+CAL arm:** 1,200/1,200 rows with full coverage; 257,443-byte stream SHA-256
  `7ebc7831...b4a23b`. Raw inference only; DEVELOPMENT rows and aggregate metrics remain zero.
- **UnivFD FIT+CAL arm:** 1,200/1,200 rows with full coverage; 259,689-byte stream SHA-256
  `e768d591...81d635`. Raw inference only; DEVELOPMENT rows and aggregate metrics remain zero.
- **Capped-UNINA FIT+CAL arm:** 1,200/1,200 rows with full coverage; 257,330-byte stream
  SHA-256 `e3d47527...4b3c01`. All four FIT+CAL arms are complete; DEVELOPMENT rows and
  aggregate metrics remain zero.
- **Decision contract:** SHA-256 `22154ab9...590e` was written before reading aggregate
  performance. It permits only 300 FIT REAL rows to build each empirical map and 600 CAL rows to
  choose among the four predeclared maximum-evidence candidates. It freezes the authentic-safety
  threshold, seven gates, ranking and MIT-licence preference; FIT AI and DEVELOPMENT are forbidden.
  Seven focused score/decision tests pass. If no candidate qualifies, measurements are archived
  with no deployable artifact rather than being lost or relaxed post hoc.
- **CAL result — clean failure:** all four monotone candidates failed. E46 was the strongest by
  AUC (0.9542) and REAL safety (2.33% pooled FP; 20% worst camera), but BA was 0.8067, pooled AI
  recall 0.6367 and StyleGAN2 recall only 0.11. The strongest specialist rescue reached just 0.19
  StyleGAN2 recall. Report SHA-256 `032944b8...75e`; candidate artifact absent; DEVELOPMENT rows
  scored = 0.
- **Bounded failure diagnosis / E50 rationale:** exact E43-S raw scores—already one frozen E48
  arm—pass all CAL gates alone at threshold `0.07940196245908739`: AUC 0.9884, BA 0.9383,
  pooled/worst-camera REAL FP 0.0333/0.20, pooled/worst-source AI recall 0.91/0.76. Per-source
  recall is FLUX.1 0.98, SD1.5 0.98, SDXL 1.00 and StyleGAN2 0.76. E46 reduces StyleGAN2 to 0.13,
  proving the DDA fusion vetoes signal before the monotone successor sees it. E50 will therefore
  pre-register the single frozen generalist and its CAL threshold for one untouched DEVELOPMENT;
  this is a new experiment, not a retroactive E48 candidate.
- **E50 DEVELOPMENT lock:** contract SHA-256 `18ae708f...20f6` binds exact E43-S artifact
  `a3aec445...7390`, threshold `0.07940196245908739`, source manifest `1404a3ff...5b68` and
  DEVELOPMENT identity digest `fc0428dc...3289`. It allows exactly 600 FODB REAL plus 600 AI
  (150 each FLUX.1/SD3/StyleGAN/StyleGAN3), one inference/evaluation, and the seven E48 gates.
  Specialist/fusion use, training, threshold repair, replacement and retry are forbidden. Scores
  and opened DEVELOPMENT metrics are zero; eight focused E48/E50 tests pass.
- **E50 inference checkpoint:** exact E43-S scored 1,200/1,200 frozen DEVELOPMENT identities with
  full coverage and no failure or replacement. The 271,063-byte stream hashes to
  `07461b09...d5fd`; aggregate DEVELOPMENT metrics remain unopened pending a separate commit.
- **E50 one-shot DEVELOPMENT — PASS:** all seven frozen gates pass without threshold change or
  retry: coverage 1.0, AUC 0.9784, BA 0.9017, pooled/worst-camera REAL FP 0.0383/0.1818 and
  pooled/worst-source AI recall 0.8417/0.6867. Per-source recall is FLUX.1 0.9867, SD3 0.9267,
  StyleGAN 0.6867 and StyleGAN3 0.7667. The score stream remains `07461b09...d5fd`. This qualifies
  E43-S for E49; it does not yet create Module-1 v1 or authorize a demo update.

### Module 1 completion and Module 2 re-entry contract

- Module 1 requires two remaining proofs: E48 one-shot DEVELOPMENT and an E49 >=2,000-row,
  publisher-separated, multi-REAL/multi-generator final with native/social-recompressed columns,
  source-stratified confidence intervals and all ten E46 gates. E48 alone cannot update serving.
- Module 2 model work starts only after an E49 pass freezes Module-1 v1. Its v1 scope is AI local
  inpainting/editing. E17's CocoGlide signal (0.648 tile AUC, 0.721 image AUC, +0.155 IoU margin)
  is the baseline; classic splices remain negative/specificity controls.
- Correct the old experiment before learning: disclose mask skips, use half-tile stride, replace
  raw-IoU headlines with pixel AUC/AP, mask-stratified F1/IoU and random-baseline margin, and test
  the measured noise-energy drop plus dense DINO tokens. Never localise fully re-rendered edits.
- Cross-module observations can pre-register a fresh Module 1 successor experiment, but cannot
  change the frozen Module-1 v1 threshold/artifact or reuse Module 2 validation rows to repair it.

### E49-A — final-of-final pre-registration, before image bytes

- **Question:** does exact E43-S preserve E50's balanced REAL safety and AI sensitivity on a
  new-publisher, 2,000-parent population and after a fixed social-media transform?
- **Parents:** 1,000 REAL Wikimedia Commons camera-category original uploads, balanced across ten
  declared phone/camera groups; 800 current AI outputs from five Datapoint 2026 provider/model
  cells; 200 StyleGAN2 outputs from the local AIGC Detection Benchmark. Every prior TRAIN/FIT/CAL/
  DEVELOPMENT/final identity is protected by exact and perceptual decontamination.
- **Candidate:** E43-S SHA `a3aec445...7390`; AI cut `0.07940196245908739`; selective REAL cut
  `0.011505939625203613`; no refit, calibration, fusion or alternative candidate. The selective
  cuts achieved 93.33% coverage /96.96% covered accuracy on consumed E50 CAL and 89.75% /95.36%
  on consumed E50 DEVELOPMENT; those rows select the policy but never count as E49 evidence.
- **Conditions:** received publisher bytes and deterministic RGB/JPEG q75 social child. Both share
  parent/source labels and are bootstrapped as paired parents. The test passes only if the same ten
  gates pass independently in both conditions: complete coverage, AUC .90, BA .85, pooled/worst
  REAL FP .10/.20, pooled/worst AI recall .80/.60, selective coverage .80, covered accuracy .95
  and uncertainty .20. Thus Module-1 v1 requires 20/20 checks, not a favourable pooled average.
- **Order:** freeze metadata identities -> commit; transfer/decode/hash without model -> freeze
  manifest -> commit; freeze score/evaluation contract -> commit; write raw scores -> commit; open
  metrics once -> archive pass/failure. Datapoint payload access is currently blocked only by its
  user contact-sharing gate; metadata research does not authorize the project to submit that form.
- **Acquisition implementation checkpoint:** the repository now has pure validators for pinned Hub
  identity/licence, licensed Commons JPEG metadata, deterministic per-uploader-capped selection and
  StyleGAN2 Parquet coordinates that never touch the image column. Four tests pass. A live metadata
  probe reproduced both exact Hub revisions and returned `GatedRepoError` for Datapoint payload
  access, with zero E49 image bytes. This is a correct stop, not an incomplete benchmark result.
- **Evaluation implementation checkpoint:** thresholds remain binary `0.07940196245908739` and
  REAL `0.011505939625203613`. The evaluator rejects any condition not containing the exact same
  2,000 parent ids, labels, sources and frozen 16-source quotas; executes the ten gates separately
  for original and social-Q75; counts score failures pessimistically; and bootstraps complete parents
  within label/source strata. No final row or aggregate metric exists yet. The acquisition,
  evaluation, shared benchmark and prior-final regression set passes 17/17 tests.
- **Local StyleGAN2 reserve probe:** the on-disk AIGC release matches pinned revision
  `c91d9024...5663`, contains the exact 60 shards/125,026 rows and exposes 1,997 AI rows with
  generator code 14. The model-blind hash rank yields a 240-row reserve with identity SHA-256
  `15e5c131...cc731`. The reusable reader requested only `label` and `generator`; image reads,
  decoding, model loading and network transfer all remained zero. Ten focused E49 tests pass.
- **Commons pre-bind feasibility amendment:** nine camera categories fill an uploader-capped
  reserve, but Fujifilm X-T5 reaches only 66/120 because too few contributors dominate its 793
  files. Before contract/image/model access, replace it with Nikon Z 8: 110/110 selected from 25
  uploaders. Reduce Commons reserve to 10% so the same 1,000-parent target stays under the 4 GiB
  network stop. Licence metadata queries now omit unused large fields. Eleven focused E49
  acquisition/evaluation tests pass; REAL image and score counts remain zero.
- **Open-component V1 — rejected before transfer:** all 1,100 Commons and 240 StyleGAN2 reserve
  identities froze, but Commons alone totals 4,140,590,955 bytes. That makes the full E49 impossible
  within the global 4 GiB stop once Datapoint is added. Preserve contract `c6f2cfb0...f794` as a
  zero-image/zero-score failed feasibility bind. Cached metadata proves a <=4 MiB original-file cap
  still fills every 110-row device reserve and reduces Commons to about 2.52 GiB; a separately
  committed V2 may use that rule without reopening model output.
- **Open-component V2 — locked:** the <=4 MiB filter is applied before the same deterministic rank
  and uploader cap. It fills 1,100/1,100 Commons rows at 2,706,581,778 bytes, leaving
  1,588,385,518 bytes for Datapoint, and freezes the same 240 local StyleGAN2 reserve coordinates.
  Contract `1d4e184c...82aa`, identity `31c0e420...e171`; zero images/scores. Do not spend the
  Commons transfer until Datapoint's exact bytes prove the full 4 GiB contract feasible.
- **Local StyleGAN2 realization:** all 240 prebound Parquet coordinates reproduce label 1/generator
  14 and decode with zero failure. Fifteen protected-role manifests plus internal exact/dHash checks
  exclude zero rows; the first 200 ranks freeze at manifest `150ed354...ec99` /20,111,615 selected
  bytes. All are 256x256 PNG, an explicit format/geometry shortcut warning. No detector score exists;
  only per-source recall inside the complete six-family E49 may use this component.
- **Datapoint access checkpoint:** the user submitted the gated contact-sharing request. Both the
  dataset page and authenticated CLI user `efehankeles` were checked; the page reports that author
  review is pending and the metadata-only probe still returns `GatedRepoError`. No retry was
  misreported as access and no final image byte was downloaded.

### E49-D1 pre-registration — open current-generator diagnostic, no image transferred

- **Question:** while Datapoint is under manual review, does frozen E43-S retain AI evidence on five
  2026 providers under original WebP and deterministic social-Q75 transport?
- **Source/selection:** ungated CC-BY-4.0 Dotting Test revision `0bcc6877...7311`; model-blind hash
  rank on successful rows; 160 target +32 reserve each for GPT Image 2, Nano Banana 2, FLUX.2 Pro,
  Ideogram 4 and Seedream 5.0 Lite. Network stop 512 MiB.
- **Candidate/metrics:** exact E43-S and E50 cuts only; pooled/per-model AI recall, automatic
  AI-decision rate, paired transport loss and 100% coverage. Gate pooled recall >=80% and worst
  model >=60% independently per condition. Commit score streams before first aggregate metric.
- **Boundary:** Turkish glyph/sign content is a narrow AI-only stress distribution. It cannot report
  AUC, balanced accuracy or authentic-photo safety, cannot promote Module 1 and cannot replace E49.
  It remains non-training diagnostic evidence regardless of outcome.
- **Bind result:** exact revision/licence/gate and all 8,400 metadata rows validated. The first bind
  stopped before writing when some successful rows exposed null declared geometry; the parser was
  fixed and regression-tested to preserve that value as unknown pending decode. The completed
  contract binds 960 WebP files /23,936,830 bytes, 192 per model, at SHA-256 `170f70db...ed36`;
  reserve identity SHA-256 is `9637626d...f5a`. Image bytes and scores are still zero.
- **Transfer result:** all 960 contract paths downloaded, totaling exactly 23,936,830 bytes and 192
  rows per model. Per-file LFS SHA-256 and size validation passes. The first post-transfer validator
  correctly withheld its receipt on exFAT `._*` AppleDouble sidecars; after a tested auxiliary-file
  exception, it rejects every other extra path and wrote the completed unscored receipt.
- **Unscored manifest freeze:** 960/960 candidates decode; zero failure. Exact/dHash comparison to
  15 protected-role manifests plus internal duplicate checks excludes six candidate identities.
  Model-blind reserve order still fills 160 clean parents for each of the five generators. The
  1,600 observations bind each parent to publisher WebP and deterministic social-Q75 JPEG; manifest
  SHA-256 `048572a4...ccc9`. Model scores remain zero and no row may enter training.
- **Pre-score lock:** contract SHA-256 `d567965d...1cf9` binds manifest `048572a4...ccc9`, exact
  E43-S artifact `a3aec445...7390`, DINOv2-S weights `04d27f34...0081`, thresholds and all six
  AI-only checks. It forbids training, threshold/row/source changes, second attempts, balanced-final
  claims and Module-1 promotion. Eighteen focused E49 tests pass; score/metric counts remain zero.
- **Raw-score checkpoint:** E43-S produced 1,600/1,600 successful scores. A schema mismatch in the
  final validator stopped after inference but before writing the receipt: completed score rows do
  not carry manifest-only payload paths. The two schemas were separated and regression-tested;
  the complete stream resumed without rescoring, is 326,693 bytes /SHA-256 `c97b02a4...fa90`, and
  aggregate metrics remain unopened.
- **One-shot diagnostic — PASS 6/6:** publisher-original pooled/worst-model recall is
  97.38%/91.25%; social-Q75 is 95.88%/86.88%. GPT Image 2 is the weakest in both columns. Q75 loses
  only 1.50 recall points; FLUX.2 Pro, Ideogram 4, Nano Banana 2 and Seedream 5.0 Lite each retain at
  least 96.25%. There was no threshold change, row/source removal or retry. Report SHA-256
  `bb62ad92...d77b`. This is strong modern-generator evidence, not a balanced-final promotion.

### E49-B pre-registration — ungated OpenFake fallback qualification

- **Why now:** Datapoint author review is an external wait, not a scientific reason to leave the
  balanced final dependent on one source. OpenFake is ungated and source-separated. It is selected
  from licence, recency, family coverage and byte feasibility without reading any OpenFake detector
  score; the successful Dotting diagnostic is not used to select or remove a model.
- **Pinned source:** `ComplexDataLab/OpenFake`, CC-BY-NC-4.0, revision
  `3fd1109dc3258874243fa31c5bda9ee24260163b`, `core/test`, 91,398 rows. Full Parquet shards are
  forbidden; only official Dataset Viewer `/rows` pages and revision-bound individual assets may be
  used.
- **Pre-result transport amendment:** `/rows` preserved 51,900 ordered rows but repeatedly failed
  with public-service 429/502/503 responses. Exact pinned Parquet HTTP ranges may now project only
  four scalar metadata columns, with transferred bytes instrumented. Image bytes and prompts remain
  forbidden. This changes no row order, cell, quota, rank, stop rule or detector boundary.
- **Blind qualification:** scan 100-row pages in increasing offset and stop at the first complete
  page where each exact cell has 192 eligible fake, non-video rows: GPT Image 2, Nano Banana Pro,
  Seedream v5.0, FLUX.2 Klein 9B and Midjourney 7. Cache only compact metadata, then rank identities
  by a fixed namespace hash and freeze 160 target +32 reserve per cell.
- **Feasibility gate:** after identities are immutable, resolve fresh Viewer assets and bind size,
  dimensions and response type. OpenFake plus the exact 2,706,581,778-byte Commons reserve must fit
  the unchanged 4 GiB network stop. Failure ends qualification before bulk transfer or scoring.
- **Claim boundary:** E49-B is a distinct successor candidate, not a rewrite or retry of E49-A.
  Viewer JPEGs are a declared received transport. Dotting remains AI-only diagnostic evidence;
  OpenFake remains non-training final evidence if the complete six-family balanced contract freezes.
- **Qualification result — FAIL, source only:** all 91,398 rows were counted with 2,597,624 measured
  metadata-range bytes /650 requests, and 52,600 Viewer-prefix rows matched the pinned Parquets.
  Eligible populations are GPT Image 2 470, Nano Banana Pro 60, Seedream v5.0 372, FLUX.2 Klein 9B
  8,093 and Midjourney 7 3,586. Nano misses the 192 reserve requirement. The experiment freezes zero
  identities and stops with zero asset requests, image bytes, detector scores or metrics; no model
  may be silently substituted inside E49-B.

### E49-C pre-registration — capacity-repaired OpenFake successor

- **Only change:** replace E49-B's underfilled Nano Banana Pro cell (60/192) with exact
  `z-image-turbo`. The choice is based only on source capacity: the validated first 52,600 rows
  contain 6,876 eligible Z-Image rows. No OpenFake detector score has been created or inspected.
- **Frozen cells:** GPT Image 2, Z-Image Turbo, Seedream v5.0, FLUX.2 Klein 9B and Midjourney 7;
  160 target +32 reserve each, exact fake/non-video filter, first-complete-100-row-page stop and new
  `E49_C_OPENFAKE_V1` hash namespace. Existing validated metadata may be reused without network.
- **Order:** commit 960 identities; bind only their fresh Viewer asset URL/type/geometry/bytes and
  global 4 GiB feasibility; transfer with exact receipt; decode/decontaminate; freeze 160 per family;
  only then assemble the balanced six-AI-family +ten-REAL-device final.
- **Immutable evaluation:** publisher/Viewer-received JPEG and deterministic social-Q75 remain paired
  conditions. Exact E43-S/E50 thresholds and all 20 E49 gates remain fixed. No source replacement,
  threshold repair, row removal or second metric opening is permitted after scoring begins.
- **Identity-freeze result:** first complete page is row 46,600, where GPT/Z-Image/Seedream/FLUX/
  Midjourney eligible counts are 236/6,111/192/4,068/1,791. Exactly 192 per cell freeze under
  reserve identity `f9f7bf74...69ec`; contract `0abae56a...d702`. Cached metadata was reused, and
  asset requests, new image bytes and model scores remain zero.
- **Asset-feasibility result — PASS:** 960/960 revision-bound HEADs total 241,736,938 OpenFake
  bytes. Combined with frozen Commons, 2,948,318,716 bytes are expected, 1,346,648,580 below the
  4 GiB stop. Broken aggregate Viewer pages were resolved only through their selected single rows.
  Generic S3 binary MIME is recorded for later JPEG decode validation. Contract `7b71449e...1415`;
  signed URLs stored zero, image-body and detector-score counts zero.
- **Transfer-format amendment:** the first strict JPEG-only run stopped at selected Seedream row
  8,770: `.jpg` Viewer path and generic MIME wrap valid 2,048-square PNG bytes. Keep the row and
  exact received bytes; admit only decoded JPEG/PNG/WebP and expose format counts per source. This
  occurred before decontamination/model access; 201 already-validated files remain resumable.
- **Resume-performance amendment:** a later restart reached 521 verified bodies but showed that
  sequential Viewer page resolution, not payload transfer, dominated elapsed time. It was stopped
  between atomic files. Resume resolves only the remaining preselected pages in batches of 24 with
  the same measured two-request ceiling, then uses the unchanged eight body workers. Identities,
  expected bytes, validation and score boundary do not change; existing payloads are revalidated.
- **Frozen-geometry correction:** selected Seedream row 43,863 is 6,144 x 11,008, the sole reserve
  member above the inherited 50 MP decoder guard. That geometry was already known and frozen by the
  no-body contract, so the guard is raised only to its exact 67,633,152 pixels. The row remains in
  place; its rejected HTTP 200 attempt created no admitted file, score or metric. Future transfer
  failures now disclose their validation cause instead of only the HTTP status.
- **Transfer result — COMPLETE:** all 960 bound files reproduce 241,736,938 expected bytes and the
  exact 192-per-family inventory. Decode coverage is 100%: 958 JPEG and two PNG. Receipt SHA-256 is
  `4dfb942c...26c2`; zero signed URLs are stored and no detector/model score has been created. The
  files remain a reserve pending the separately committed decontamination/pairing gate.
- **Realization result — PASS:** all 960 reserves decode and zero intersects a protected historical
  role. Twenty-six Seedream rows are duplicate payloads (both SHA-256 and dHash) and are excluded in
  prebound rank order; 166 clean Seedream rows remain, so every family still fills exactly 160.
  The frozen component contains 800 parents/1,600 paired observations, selected formats 798 JPEG +
  two PNG, and manifest SHA-256 `38048803...7442`. Model-score count remains zero.
- **Frozen realization method:** after the exact transfer receipt exists, re-open all 960 files and
  reproduce byte hash, decoded format, dimensions and full coverage. Compare original SHA-256 and
  dHash against every protected role, the scored Dotting diagnostic and the frozen StyleGAN2 final
  component; also exclude within-reserve duplicates. Walk the already-bound rank order per family,
  generate fixed 1080-long-side JPEG-Q75 children, reject protected/duplicate children, and require
  exactly 160 parents plus 160 children per family. No detector import, score or metric is allowed.
  Twenty-one focused E49 tests pass before this production step.
- **Commons transfer method:** use only the 1,100 identities in open-components V2. A resumable
  eight-worker transfer must reproduce all 2,706,581,778 expected bytes and every Wikimedia SHA1,
  decode only exact-geometry JPEG, preserve original bytes, record SHA-256 and available EXIF
  make/model, and reject unexpected files. Receipt and compact evidence remain explicitly unscored;
  eighteen focused E49 tests pass before network execution.
- **Rate-limit amendment:** the initial eight-worker execution stopped on Wikimedia's explicit 429
  after only two complete originals; no partial body was admitted. Resume uses one request stream,
  0.75-second pacing, a repository-linked research User-Agent and up to 60-second Retry-After/
  exponential backoff. Completed files are revalidated, and original URLs/bytes remain unchanged.
- **Native-container amendment:** Commons row 139,916,479 is a standards-valid Apple iPhone JPEG
  with an MPF multi-picture segment. Wikimedia MIME and `file` identify JPEG, while Pillow correctly
  reports `MPO`. Preserve its exact bytes and admit decoded JPEG/MPO as distinct reported formats;
  frozen dimensions, Wikimedia SHA1 and every other safety check remain mandatory.
- **EXIF-geometry amendment:** row 148,952,501 stores 4,032 x 3,024 pixels with orientation 6, while
  the bound Wikimedia geometry is its correct 3,024 x 4,032 display orientation. Preserve original
  bytes, record encoded and display dimensions separately, and validate the frozen contract against
  the latter. Only orientations 5–8 swap axes; a regression test binds this behavior.
- **Commons transfer result — COMPLETE:** all 1,100 originals reproduce 2,706,581,778 bytes and the
  exact 110-per-device quotas. Decoded container counts are 861 JPEG and 239 Apple MPO; 1,074 files
  preserve EXIF make and model. Receipt SHA-256 `2511f0ad...7e04`; payload/model-score coverage is
  100%/0%. These remain a reserve pending device-evidence and protected-overlap realization.
- **Commons realization method:** bind the completed transfer receipt; compare original SHA-256 and
  dHash against historical protected roles, the scored Dotting diagnostic and both frozen final-AI
  components. Within each prebound device rank, retain the first 100 parents whose original and
  deterministic Q75 child avoid protected/internal collision. The method requires exactly 1,000
  REAL parents and 2,000 paired observations, preserving uploader/licence and EXIF availability.
  Twelve focused E49 tests pass before production execution and no model import is present.
- **Device-evidence amendment before realization:** normalize known manufacturer/model spellings and
  Apple hardware aliases. When EXIF exists it must agree with the frozen category; missing EXIF is
  retained only as explicit `category_only_missing_exif`. One Nikon Z 8 category file reports Nikon
  D70 and must be excluded in rank order. This filter is fixed before selection/model access;
  thirteen focused transfer/realization/evaluation tests pass.
- **Realization wiring correction:** the download receipt correctly records cryptographic hashes but
  not perceptual dHash. The first execution read a nonexistent field and stopped before manifest or
  child output. Realization now reproduces each SHA-256 and EXIF-display geometry, derives dHash from
  the oriented RGB image, then enters the unchanged overlap audit. Fourteen focused tests pass.
- **Commons realization result — PASS:** all 1,100 reserves realize. Exactly two candidates are
  excluded: one protected dHash overlap and the disclosed Nikon-D70 EXIF mismatch. Fixed headroom
  still fills ten device cells at 100 each. The manifest binds 1,000 parents/2,000 paired rows at
  SHA-256 `657be9bb...8e7b`; selected device evidence is 979 EXIF matches plus 21 explicit category-
  only rows. No score or metric was created.
- **Final evaluator source amendment:** the original E49-A validator still named five Datapoint
  families whose images were never consumed. Before E49-C realization or scoring, its exact AI
  quota labels are corrected to GPT Image 2, Z-Image Turbo, Seedream v5.0, FLUX.2 Klein 9B,
  Midjourney 7 and StyleGAN2. Counts, REAL devices, thresholds and all twenty gates do not change.
- **Final assembly method:** require the exact unscored Commons, OpenFake and StyleGAN2 component
  manifests. Create the 200 StyleGAN2 Q75 children without replacement, rejecting any protected or
  internal child collision. Reproduce all 4,000 file SHA-256 values and dimensions, validate exact
  pairing and sixteen source quotas, and archive source geometry plus condition-format counts. The
  resulting manifest alone may feed the later score lock; twelve focused tests pass before use.
- **Final assembly result — PASS:** component hashes `657be9bb...8e7b`, `38048803...7442` and
  `150ed354...ec99` combine into exactly 2,000 balanced parents and 4,000 paired observations with
  all sixteen source quotas. All payload SHA-256 values and display geometries reproduce. Original
  format counts are 1,585 JPEG/213 MPO/202 PNG; all 2,000 social children are JPEG. Frozen manifest
  SHA-256 `9744a9d2...5909`; metrics and model scores remain unopened.
- **One-shot execution method:** three commands create irreversible checkpoints. `bind-score` seals
  final-manifest/component, E43-S artifact, DINOv2-S weight, thresholds and gate hashes before model
  load. `score` writes a resumable ordered raw stream and zero metrics. Only `open-metrics` may verify
  that stream and execute the frozen 10,000-bootstrap/20-gate evaluator. Fourteen focused tests pass;
  retries, source removal, threshold changes and metrics-before-score-lock are forbidden.
- **Pre-score lock complete:** contract SHA-256 `fecd724c...61dd` binds final manifest
  `9744a9d2...5909`, observation identity `3cf565a1...5242`, exact E43-S/DINO hashes, both fixed
  cuts and all twenty gates. The contract is committed with zero score rows and zero opened metrics.
- **Raw-score lock complete:** E43-S produced 4,000/4,000 ordered scores with 100% coverage. The
  1,005,967-byte stream hashes to `249f005c...10a8` and remains outside Git; compact evidence binds
  it to score contract `fecd724c...61dd`. No aggregate metric has been opened at this checkpoint.
- **One-shot E49-C result — FAIL 11/20:** publisher-original/social-Q75 pass 6/10 and 5/10 checks.
  Coverage is 100%. AI recall is 94.30%/95.50% (worst family 91.88%/91.25%), confirming strong
  modern-generator transfer. REAL false-AI is 39.10%/49.00% and worst-device false-AI 71%/84%; BA
  is 77.60%/73.25%, covered accuracy 76.27%/71.01%, AUC 90.24%/86.89%. Report `10fc0649...5573`;
  retry zero. The failure is consumed and cannot select a new cut or train a repaired candidate.

### E51 pre-registration — repair the representation, not the consumed final

- **Failure-derived boundary:** E49's TPR at 10% FPR is only 72.30% original and 58.80% Q75, so no
  threshold alone can meet simultaneous 10% REAL-FP and 80% AI-recall gates. Pairwise AND remains
  32.80% REAL FP at 93.50% AI recall. JPEG/MPO FP is 39.90%/36.15%; log-resolution correlation is
  just 0.056. These may motivate E51 but cannot choose a threshold, row or fitted parameter.
- **Data first:** audit new licensed camera sources and freeze publisher/device/scene-disjoint TRAIN,
  CAL and DEVELOPMENT before image transfer. E49 parents and unused reserves remain protected.
- **Two bounded candidates:** (A) source-balanced frozen-DINO head with authentic original/Q75/JPEG/
  resize hard negatives; (B) the same representation plus fixed residual/DCT statistics and source-
  bias suppression. A real-only arm may abstain but cannot output certified REAL.
- **Decision ladder:** select once on new grouped CAL; open fresh DEVELOPMENT once under both
  transports and the existing ten gates; only a pass permits a new E52 final. Approved Datapoint
  image bodies remain unopened for possible E52 AI evidence, not E51 tuning.
- **E49 diagnosis reproduction — COMPLETE:** a dedicated model-blind runner verifies the exact
  consumed manifest `9744a9d2...5909`, raw stream `249f005c...10a8` and final report
  `10fc0649...5573`, then joins all 4,000 observations/2,000 pairs without inference. Report
  `3e5caa86...bd70f` reproduces 10%-FPR TPR 72.30%/58.80%, REAL paired-AND FP 32.80%, AI paired-AND
  recall 93.50%, JPEG/MPO FP 39.90%/36.15% and log-megapixel correlation 0.0563. It creates no new
  scores or candidate threshold; E51 source audit is the next permitted action.
- **E51 REAL-source audit — PARTIAL, zero image payload:** all 9,940 SCMI30 v2 public entries were
  enumerated and normalized to 9,937 native JPEGs across 30 devices (35,592,810,773 image bytes).
  It is the primary device-disjoint TRAIN/CAL candidate because individual files can be frozen;
  source terms remain research/education CC-BY-NC-ND. SCIMD-17 is open and tiny but uniformly
  224x224, so it is auxiliary TRAIN only. RAISE is ~350 GB/three cameras; official Dresden is down;
  IMAGINE lacks verified TLS/explicit terms; SOCRatES is agreement-gated. Evidence
  `9ddab57a...c4883`; the audit deliberately authorizes no download until a different licensed
  DEVELOPMENT publisher is found.
- **E51 role route — FROZEN, zero image payload:** contract `975e8164...15e4` assigns historical
  legitimate parents plus SCIMD-17 resize hard negatives to TRAIN, 1,200 device-balanced SCMI30
  originals to CAL, all 2,640 IEEE SP Cup test TIFFs to REAL DEVELOPMENT and Datapoint to AI
  DEVELOPMENT. The Datapoint reserve is paired by content: every one of five current generators has
  the identical 23 score-blind prompts in each of eight categories (920 reserves, target 800 after
  decode/decontamination). Seven exact source shards cost 3,220,281,593 transfer bytes rather than
  the 34.9 GB repository. IEEE's test payload is 837,665,909 bytes but remains 403-gated until the
  user accepts Kaggle competition rules. Counts, identities and bytes are bound; model scores and
  image downloads remain zero. Datapoint can no longer serve E52 final.
- **IEEE REAL DEVELOPMENT acquisition — COMPLETE, unscored:** after rule acceptance, the original
  per-file route admitted 520 rows but hit Kaggle API throttling. A committed HTTP-Range reader then
  verified the official 5,391-member ZIP central directory and fetched only the remaining bound
  test members, never the 2,750 train rows. Final receipt `09188d49...3794` contains 2,640/2,640
  safely decoded 512x512 PNG bodies /837,665,909 bytes, equally split unaltered/postprocessed;
  identity SHA `fc3657dd...fb05` matches the pre-score contract. Model-score count is zero.
- **Datapoint AI DEVELOPMENT transport — COMPLETE, unopened:** the exact seven pinned Parquets are
  local at 3,220,281,593/3,220,281,593 bytes with a full-file SHA-256 receipt
  `18b8326a...bad1`. The transfer retains all 920 paired reserves and the prebound 800-parent target;
  it deliberately opens no image column, performs no decode/decontamination and creates no score.
- **SCMI30 REAL CAL acquisition — COMPLETE, unscored:** four independent TLS-verified ZIP Range
  readers admitted 1,200/1,200 bound bodies /4,247,339,334 bytes. Exact closure checks passed:
  30 devices x40, Random/Similar 600/600 and make/model EXIF 1,200/1,200. The run reused 59 verified
  rows, fetched 1,141 and left no partial. Receipt `01cc5921...f33e` and ordered identity digest
  `c47d411f...1d12` bind the result. This is acquisition evidence: model-score count remains zero.
- **SCIMD-17 TRAIN archive — COMPLETE, unopened/unscored:** the 174,438,734-byte Zenodo payload
  reproduces MD5 and SHA-256 `ef1fe3e7...0201`. Its safe central directory contains 17,620 images /
  172,781,180 expanded bytes. Receipt `8b38fa82...b230` records zero decoded bodies and scores.
  Realization is restricted to score-blind 224x224 REAL hard negatives in TRAIN.
- **SCMI30 identity amendment — COMPLETE, unscored:** a full exact+dHash+pHash audit rejected only
  the no-content black D04 frame. Five same-device/branch reserves were bound before 7,710,716 bytes
  transferred; all passed, and first-ranked `D04_nat_45.jpg` is the replacement. Receipt
  `66e3d063...9722`; balance remains 30 x40 and 600/600, with zero model scores.
- **E51 TRAIN/CAL manifests — COMPLETE, unscored:** TRAIN has 5,978 parents (4,035 REAL/1,943 AI),
  including 1,700 device-balanced SCIMD resize hard negatives; zero decode failures and two
  conservative dHash exclusions. CAL has 1,200 independent SCMI30 REAL plus 360 held-out historical
  AI parents, each original/Q75 paired for 3,120 observations. Hashes `41444640...77ef` and
  `60688291...2356` bind the manifests. DEVELOPMENT remains unopened and model scores remain zero.

### Gated-source approval audit — route decision before any image access

- Authenticated Datapoint access now succeeds at pinned revision `e1d8719a...c928`; authenticated
  ITW-SM succeeds at `3060094f...f86`. ITW-SM is already consumed through its official E45 copy and
  is not independent a second time.
- Datapoint metadata confirms 30 models/14,952 images/500 prompts and all five originally proposed
  frontier families. Only 1,737,709 bytes of models, test-response and prompt reference Parquets
  were fetched; image Parquets/image bodies and detector scores remain zero.
- **Routing decision:** do not pivot the single final after E49-C identity and byte freeze. OpenFake
  `core/test` is explicitly an OOD detection split, whereas Datapoint is a controlled shared-prompt
  preference benchmark. Preserve Datapoint as unconsumed, non-training post-final evidence.

### Module 2 archive audit before re-entry — lessons carried forward, no model run

- E17/E18 hard-code non-overlapping 128 px tiles capped at 36, then discard any manipulated image
  lacking both a >=50%-mask tile and a <50%-mask tile. Only 35/120 CocoGlide examples survived in
  the historical result, so the old number describes a selected mask geometry, not the full set.
- Tile AUC pools correlated tiles as independent rows. IoU flags a quantity derived from whole-image
  mask fraction but thresholds truth at tile coverage .5; its analytic random baseline uses pixel
  fraction, so numerator and baseline are not the same sampling unit. Raw IoU is therefore not a
  valid primary metric even though the positive margin remains a useful hypothesis.
- The old image score is top-three tile mean from `feature_crop128.joblib`; it is not the later E43-S
  DINOv2 representation. E43-S uses global plus two texture-crop intermediate embeddings, so Module
  2 cannot honestly call the current whole-image artifact a localiser. The transferable idea is the
  frozen DINOv2-S representation with dense patch tokens, not the existing logistic head itself.
- The sound observations remain: CocoGlide's generated region carries absolute AI signal; classic
  camera-to-camera splices ask a different relative-forensics question; ELA's JPEG positive control
  works while uniform PNG conversion destroys it; manipulated-region noise energy was lower. The
  next evaluator will preserve these hypotheses while correcting selection, unit-of-analysis and
  threshold leakage before any new training.

### E51 pre-fit remote audit — 2026-09-07 (model-blind)

- Frozen input manifests: TRAIN `41444640...77ef`, CAL `60688291...2356`; no edits or score-based
  replacements. New commands: `experiments.e51_prefit_audit` and `experiments.e51_protected_inventory`.
  Run with `PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets` and `PYTHONPATH=ml:ml/src` using
  `ml/.venv/bin/python -m ...`; pre-fit audit supports `--workers 1..8` and resumable fingerprints.
- Checks: byte SHA-256/size, parent role separation, paired original/Q75 label/source consistency,
  decoded EXIF-oriented RGB digest, canonical dHash radius 4 with pHash63 radius 4 confirmation.
  Both Q75 and original are compared; matched conditions are not counted as independent parents.
  A perceptual match is a review candidate, not automatic proof of mislabeled data.
- Method correction: opposite-comparison dHashes need not be complements on flat/tied pixels or
  when resize filters differ. Preserve prior manifests but do not use inverse bits as an admission
  proof. Correct stale Dotting path and fail when any required protected manifest is unavailable.
- Protected metadata result: 22 pinned inputs, 15,499 E49 identity/available hash keys, 2,900 new
  SCIMD/SCMI parents, zero overlaps under deliberate historical-TRAIN reuse. Includes the v2
  1,100 Commons/240 StyleGAN2/960 OpenFake reserves. Canonical protected-image screening and
  superseded reserve inventory remain pending; `training_authorized=false` is intentional.
- SCIMD filename review: official metadata MD5 reproduced; all eight `chatgpt-*` names carry camera
  metadata. The two selected files appear to be photos of laptop screens; retain publisher REAL
  labels, with uncertainty about provenance explicitly documented. No detector was consulted.
- Engineering verification: 500 Python tests pass, compilation passes; no serving artifact,
  threshold, DEVELOPMENT result or final-test gate changes. First-run historical `condition` schema
  mismatch was repaired with a regression test before audit resumption; no failed run wrote success.

### E51 offline locator checkpoint — 2026-09-09

- Recovered completed pre-fit evidence: 9,098 verified observations, 5,978 TRAIN parents and 1,560
  CAL parents, zero matched cross-role parent pairs. Full report SHA-256 `02180078...77a2`.
- New local-only command: with `PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets` and
  `PYTHONPATH=ml:ml/src`, run `ml/.venv/bin/python -m experiments.e51_offline_bodies`. It validates
  the frozen protected inventory and manifests, resolves relative file paths against each manifest,
  checks already-local ZIP members without extracting them and freezes locators without scores.
- Result: 117,898 distinct body locations; 73,165 file /44,733 ZIP; 50,577,346,337 bytes already local;
  zero unresolved body rows; zero bodies lacking a SHA-256 after identical-location merges.
  The 9,800 metadata-only references remain explicitly unjoined. Locator hash `a51cb457...eb81`.
- No image download, model training, score or threshold change. `training_authorized=false` until
  protected canonical-pixel and remaining reserve admission close. Passing path resolution does
  not certify pixel-level separation. All 504 Python tests passed; no new dependencies installed.

### E51 executable protected-pixel audit + B feature hypothesis — 2026-09-09

- Added `experiments.e51_protected_pixels --workers 4`, using the existing LaCie data root and
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. It verifies all current bytes, uses full RGB/dHash/pHash63,
  batches work, caches fingerprints in `ml/work/e51/canonical_fingerprints.sqlite3`, and writes
  recoverable progress on the external disk. No score is computed. Completion is not yet claimed.
- First attempt: 64,000 locations verified, 63,996 unique cache entries, then failed closed on the
  100 MP cap. Six frozen RR protected images are larger (maximum 178,562,880 pixels). Before restart,
  bind their six SHA-256/pixel-count pairs and serialize only their full-resolution decoding.
  All other images retain the default cap; no global Pillow bomb setting is changed. Resume started.
- E51-B feature specification is now implemented and unit-tested: 8 normalized radial DCT energy
  bins +8 residual/gradient statistics per 224x224 crop; mean/std over the exact three E42 crops
  produce 32 extra values. A=3,072 versus B=3,104 dimensions; no architecture/hyperparameter sweep.
  The branch is an unmeasured hypothesis, not a claimed detector gain or proven shortcut immunity.
- Verification: 512 Python tests, compilation and whitespace checks pass. No E51 real-data feature
  extraction, fitting, threshold search, DEVELOPMENT rerun or demo promotion took place. The old
  comprehensive final remains consumed; any eventual rerun must be labelled diagnostic/regression.

### E51 reserve closure and pre-fit method freeze — 2026-09-09

- Reserve report `2c2cdb07...9ad36`: 9,800/9,800 references joined, 434 additional local bodies,
  zero supplementary parent matches and identity overlaps. 419 never-downloaded superseded
  Commons identities stay forbidden, with no claim of pixel coverage. No transfer was performed.
- Before scores: exclude flagged whole parents at most 5% per TRAIN/CAL role, retain >=30 original
  REALs/device across all 30 CAL devices and >=15 AI/source across all 18 groups. Preserve originals.
- Fit fixed C=0.01 A/B using clean + fixed legacy transport + Q75 TRAIN views, balanced by class,
  source and parent. Standardization sees TRAIN only; frozen backbone reuse is hash-bound.
  CAL original/Q75 files are not recompressed. Source groups may be topics, not known generators.
- Select the smallest common AI threshold satisfying pooled 10% /device 20% REAL FPR on both CAL
  transports; choose the REAL band by >=95% covered accuracy and maximal minimum coverage.
  All original AUC/BA/recall/worst-group/coverage gates remain mandatory; prefer A if both pass.
- Repetitions: seeds 42/43/44 with deterministic lbfgs, reporting score differences, not claiming
  independent stochastic replication. Checkpoint candidate artifacts/scores before any E49 access.
- Verification: 523 tests pass. Main protected-query checks still pending; no E51 fit yet.

Admission completion addendum: protected report `2f070e7d...c1e54` verifies 117,898 locations
and 9,098 queries with zero matches. Admission `c406edb6...7a39b` excludes zero parents, retaining
5,978 TRAIN and 1,560 CAL /3,120 paired observations. This authorizes the already-frozen feature
and fit protocol, not a model-quality claim.

Pre-fit regression contract: both A/B at frozen CAL thresholds, all 4,000 consumed E49 original/Q75
observations with identical identity/label checks. Baseline hash `249f005c...610a8` must reproduce
within max absolute score difference 5e-5. Report AUC/BA/FPR/recall/coverage, counts and source rates,
plus 2,000 paired parent-bootstrap old/new deltas within fixed source strata per transport.
No bootstrap threshold refits, no pooling paired images as independent parents, no candidate
selection from E49. Code tests pass; feature extraction is underway and no E51 fit exists yet.

Feature completion: 21,054x3,072 DINO and 21,054x32 residual matrices, archive
`55f92db2...82d0a`, binding `f6c95616...9f816`. Reused 8,916 frozen E42 backbone views only;
12,138 new DINO views computed on MPS. Model scores still zero at this checkpoint. The fixed
six head fits (A/B x42/43/44) are now running with local dependencies and offline environment.

### E51 TRAIN/CAL result — 2026-09-09

| Candidate | Condition | AUC | BA | REAL FPR | AI recall | Worst REAL FPR | Worst AI source recall |
|---|---|---:|---:|---:|---:|---:|---:|
| A | original | 0.997699 | 0.970000 | 0.026667 | 0.966667 | 0.200 | 0.900 |
| A | Q75 | 0.998319 | 0.975556 | 0.018333 | 0.969444 | 0.150 | 0.900 |
| B | original | 0.997623 | 0.969167 | 0.028333 | 0.966667 | 0.200 | 0.900 |
| B | Q75 | 0.998236 | 0.974444 | 0.023333 | 0.972222 | 0.175 | 0.900 |

Both pass all CAL gates with coverage 1.0; A is selected by the predeclared simplicity rule.
A cut/REAL cut 0.3316505551338196; B 0.3200901746749878. Artifacts `60d56c0b...b6b39` and
`b1ac18f3...838e2`; all three seeds give identical scores. CAL labels selected cuts, so this is
not independent validation. No DEV/E49 model score existed before this result freeze; no serving
promotion. Next: fixed diagnostic E49 comparison and one-shot admitted fresh DEVELOPMENT of A.

DEV realization preregistration: all 2,640 IEEE REAL plus 800 selected Datapoint AI (five models
x160 prompts, shared 20/category across eight categories), original/Q75, 6,880 observations.
Decode the 920-row score-blind reserve from the seven previously downloaded hash-bound Parquets.
Protected/E51/internal cross-parent radius-4 dHash+pHash screens apply to both transports. An IEEE
match aborts; an AI match excludes its whole shared prompt group, with fixed ranked reserves only.
No source quota relaxation, label correction, score-dependent substitution or new transfer.
Worst-real transport group is only a proxy: IEEE hides camera ids and supplies 512px publisher
images, so neither worst-device generalization nor full-resolution performance can be established.

### E51 consumed-E49 diagnostic regression — 2026-09-09

| Candidate | Condition | AUC | BA | REAL FPR | AI recall | False AI /1,000 REAL | Misses /1,000 AI |
|---|---|---:|---:|---:|---:|---:|---:|
| Old E43 | original | 0.902425 | 0.7760 | 0.391 | 0.943 | 391 | 57 |
| E51 A | original | 0.884358 | 0.8080 | 0.159 | 0.775 | 159 | 225 |
| E51 B | original | 0.884528 | 0.8075 | 0.172 | 0.787 | 172 | 213 |
| Old E43 | Q75 | 0.868850 | 0.7325 | 0.490 | 0.955 | 490 | 45 |
| E51 A | Q75 | 0.843796 | 0.7570 | 0.307 | 0.821 | 307 | 179 |
| E51 B | Q75 | 0.844161 | 0.7585 | 0.313 | 0.830 | 313 | 170 |

Old score reproduction max difference=0.0. A fixed source-stratified paired bootstrap BA delta
95% CI: [+0.0150,+0.0485] original, [+0.0085,+0.0410] Q75. Original AI recall delta is
[-0.191,-0.146]; the loss is material, not hidden behind the BA improvement. A's worst REAL
group is 30% original and 55% Q75; worst AI group recall 61.25%/63.125%. Both fail the gate.
No selection/refit from these rows; A stays the CAL-selected research artifact, serving unchanged.
Fresh DEV scoring is preregistered independently of these diagnostic metrics, using A only and
the frozen CAL threshold; 2,000 bootstrap replicates keep the five-model shared AI prompt grouped.

DEV score-blind admission stop: audit `975569d3...258e8`, 2 protected-match observations (one AI
parent), 56 directed/transport internal matches (seven unordered parent pairs: five REAL, two AI).
No exact byte/pixel matches, no protected REAL match, model_scores_created=0. Visual review of
all ten IEEE images supports grouping four same-scene variant pairs; the grey pair is conservatively
grouped without claiming confirmed scene identity. The new grouping amendment keeps every image
and label, reduces detected REAL independence units 2,640→2,635, and makes interval resampling
cluster-aware. It cannot waive protected overlap or alter original/Q75 populations or any cutoff.
Before admission, repeat the canonical screen and require the preserved match multiset exactly.

DEV admitted unscored: manifest `1b1882f3...ea224`, grouping amendment `10792cb7...a495a`,
all previous audit match multisets reproduced. Exact 3,440 image parents /6,880 paired rows;
REAL independence accounting 2,635 detected scene groups, AI 160 shared five-model prompts.
Candidate A remains `60d56c0b...b6b39`, CAL report `247fac1f...16b29`, same cutoff; 532 tests pass.

### E51 one-shot fresh DEVELOPMENT — 2026-09-09 (completed)

| Condition | AUC | Balanced accuracy | REAL false-AI | AI recall | Total accuracy | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.993367 | 0.928542 | 44/2640 = 0.016667 | 699/800 = 0.873750 | 0.957849 | 1.0 |
| Q75 | 0.989087 | 0.920777 | 52/2640 = 0.019697 | 689/800 = 0.861250 | 0.952616 | 1.0 |

Frozen A cut/REAL cut 0.3316505551338196; no refit, rejection-band change, candidate substitution
or row removal after scores. Worst AI generator recall: Ideogram 0.78125/0.79375. Worst observable
REAL transport-cell false-AI: 0.017424/0.022727. The hidden worst-camera metric is **unobservable**,
not assumed equal to the transport-cell result. Point-estimate observable gates pass.

2,000 scene/prompt-cluster bootstrap intervals: original BA [0.912801,0.943839], REAL FPR
[0.011747,0.021986], AI recall [0.842500,0.902531]; Q75 BA [0.904528,0.937357], REAL FPR
[0.014756,0.025019], AI recall [0.828719,0.893750]. These cover this publisher population and
detected clusters, not future generator/device distribution shift. REAL and AI counts differ,
so headline balanced accuracy is preferable to the larger aggregate accuracy.

Raw scores `088718f8...724be`, manifest `1b1882f3...ea224`, A `60d56c0b...b6b39`;
`evidence/e51_development_result.json` records all source rates, gates, hashes and limitations.
DEV is now consumed and forbidden from later TRAIN/CAL/E52. The E49 diagnostic failure remains
valid; no serving promotion or universal-detector claim. Next research is pre-fit TRAIN-only
source-held-out transport robustness, with AI recall guarded; no old/fresh test used for fitting.

### E53 — TRAIN-only data/transport research protocol (2026-09-09, PLANNED, not run)

Question: can authentic false accusations decrease without reducing AI detection, rather than
merely raising balanced accuracy through an asymmetric tradeoff? E49 original E43→E51-A has
REAL FPR 39.1%→15.9% but AI recall 94.3%→77.5%; AUC also decreases. E51's distinct DEV
population is encouraging but does not isolate the cause or supply a paired E43 comparison.
E43/E51 fit populations differ (8,844 versus 5,978 parents); the next experiment must separate
data changes from transformations. There are no new trained candidates or test scores in E53.

Primary references and detailed scope are in PLAN's E53 section: B-Free for shortcut control,
Community Forensics for generator diversity, NTIRE 2026 for degradation robustness, Effort for
preserving pretrained information during adaptation. None guarantees local improvement; plain
JPEG augmentation is not B-Free, and LoRA is not Effort's orthogonal decomposition.

Proposed bounded design: existing-eligible versus expanded-eligible TRAIN pools, each with the
old three views versus clean + resize/JPEG order variants. Fixed DINOv2-S representation and
C=0.01 weighted logistic head; at most 16,000 admitted parents. Before fitting, freeze complete
role/duplicate/source admission, group-held-out outer folds, inner CAL, quotas, variant details,
feature hashes and support requirements. Refit all heads/scalers inside folds. Do not reuse a
historical classifier that already saw validation rows as an honest out-of-fold comparator.

Advance only with lower REAL false-AI and nonnegative AI recall point deltas on both original/Q75
and adequately supported declared AI sources, plus unchanged absolute gates. Report paired,
scene/prompt-cluster uncertainty; the pooled statistical no-loss/improvement claim requires
simultaneous bounds across both transports' AI and REAL deltas as specified in PLAN. Source
point guards do not prove source-level equivalence. Development selection cannot certify final
success. Retain current serving on failure/inconclusive results. Adapter research is conditional
and must receive its own frozen hyperparameters before execution, not post-test model shopping.

All old tests, E51 CAL and consumed IEEE/Datapoint DEV remain protected from fitting/selection.
E52 requires a genuinely independent local population; no available population means no final
claim and no download under the current constraint. Rechecking 18,154 historical TRAIN locators
is recorded in `evidence/e53_local_inventory_plan.json`; it is not a fresh training admission.

### E53 office slice — six-arm result and native-expansion precommit (2026-09-09)

Contract `95e9ea22...1d1c`; 5,978 admitted TRAIN parents /11 source components /3 outer folds.
RR topics share one publisher; E36 AI prompt batches and E32 FLUX/Qwen shared prompts remain
grouped. All scalers/heads are refit on FIT only; each fold has independent inner CAL. External
tests and serving remain unchanged. The following are **TRAIN-derived outer-fold** metrics with
fold-specific cuts, not the performance of a full-data model on E49/IEEE/Datapoint:

| Arm | Clean REAL FPR | Clean AI recall | Clean BA | Q75 REAL FPR | Q75 AI recall | Q75 BA |
|---|---:|---:|---:|---:|---:|---:|
| Full, old 2 views | 15.54% | 51.31% | 67.89% | 14.28% | 51.06% | 68.39% |
| Full, E51 3 views | 15.64% | 50.13% | 67.25% | 15.17% | 51.67% | 68.25% |
| Full, order/scale 3 views | 15.39% | 45.96% | 65.28% | 15.17% | 47.40% | 66.12% |
| Mean only, old 2 views | 12.39% | 61.40% | 74.50% | 11.52% | 58.72% | 73.60% |
| Mean only, E51 3 views | 12.81% | 59.44% | 73.32% | 12.52% | 58.52% | 73.00% |
| Mean only, order/scale 3 views | 12.94% | 58.36% | 72.71% | 13.31% | 58.88% | 72.78% |

All six fail the full research preservation guard. Mean-only variants improve pooled behavior but
lose individual AI sources. No winner is promoted. Conditional paired publisher bootstrap uses
20,000 draws and Bonferroni-adjusted intervals for four primary deltas per comparator; only 11
observed components and multiple-candidate selection limit inference. No pooled raw-score AUC is
reported across differently calibrated heads. Source counts and per-fold AUCs remain in evidence.
Existing two/three-view mean-one weighting changes total loss mass, so this is not a pure
augmentation-only causal comparison. `evidence/e53_source_held_out_result.json` preserves all arms.

Additional engineering check `evidence/e53_crop_dedup_benchmark.json`: stable exact crop dedup
on 60 fixed TRAIN parents/120 views cuts 360 crops to 316; max feature/score error=0, decision
flips=0, median elapsed 2.0802→1.8546 seconds across three warmed passes. This is roughly 10.84%
less elapsed time (1.1216x throughput), not 12.16% less elapsed time or a detection accuracy gain.
Native audit was running in the background; timings are local workload/device-specific.

Native expansion is fixed before its scores: 1,000 REAL each CSAFE S21/FODB/VISION, 500 AI each
CF/FLUX.2 Klein/GPT Image 1/Nano Banana/Qwen Image 2512, and all 152 eligible Nano Banana Pro.
Only matched-publisher FIT rows are added; original E53 CAL/validation are untouched. Three
original-based views, full/mean-only C=0.01, total weights normalized to the old three-view FIT
mass. This isolates native data addition more cleanly than changing both pool and calibration.
Original audit and complete E51 reserve closure precede the frozen expansion contract. Features
are extracting at this checkpoint; no extra-arm score or external final pass is asserted.

Four precommitted head controls completed on the same v1 folds, three views and weight mass:

| Arm | Clean REAL FPR | Clean AI recall | Clean BA | Q75 REAL FPR | Q75 AI recall | Q75 BA |
|---|---:|---:|---:|---:|---:|---:|
| Full, C=0.001 | 13.63% | 50.23% | 68.30% | 13.41% | 52.19% | 69.39% |
| Mean, C=0.001 | 13.36% | 63.00% | 74.82% | 12.34% | 61.19% | 74.43% |
| Full, row-L2 | 15.46% | 49.92% | 67.23% | 14.99% | 51.06% | 68.03% |
| Mean, row-L2 | 15.02% | 56.46% | 70.72% | 15.09% | 56.05% | 70.48% |

No preservation-guard survivor. `evidence/e53_head_controls_contract.json` precedes the scores;
`evidence/e53_source_held_out_controls_result.json` retains every comparison. Multiple-candidate
TRAIN selection is explicit. Stronger regularization is helpful descriptively in the mean arm;
row-L2 and additional order/scale transforms are not automatically improvements.

Weight-construction optimization is measured independently of quality: all 17,934 TRAIN view
weights are bitwise identical; 0.31790→0.00915 seconds median, ~34.75x for this preparation step.
No claim of a 35x full training speedup. New control fits use this verified helper without changing
their objective. Exact crop-dedup parity is extended prospectively to all 11,956 TRAIN clean/Q75
views, with fixed feature/score tolerances and zero permitted decision flips.

Coverage-v2 protocol, before its scores: metadata reveals two components (E36 REAL, E32 paired
FLUX/Qwen) never train in v1. Jointly choose the three inner CAL component sets with minimum
sum class-wise squared deviation from 20%, subject to every component appearing in FIT at least
once. Keep outer validation assignments exactly unchanged, whole-component separation, >=30
CAL parents/class and >=100 FIT parents/class. Repeat the same twelve arms including native
expansion. This is a more efficient TRAIN model-selection protocol, not an independent replication
or an excuse to discard v1 failures. No v2 result is claimed at this checkpoint.

### E53 closure — native replay, coverage-v2 and offline preparation (2026-09-09)

Native original-based features completed for 5,652 parents/16,956 views, SHA `82ab703a...82c1c5`.
The v1 native additions retain the existing folds and normalize total FIT sample-weight mass to
the original three-view baseline. Existing CAL/validation receive no new images. Results:

| Additional v1 arm | Clean REAL FPR | Clean AI recall | Clean BA | Q75 REAL FPR | Q75 AI recall | Q75 BA |
|---|---:|---:|---:|---:|---:|---:|
| Full, native-expanded 3 views | 23.74% | 77.61% | 76.93% | 22.97% | 75.76% | 76.39% |
| Mean, native-expanded 3 views | 16.78% | 72.52% | 77.87% | 16.70% | 71.23% | 77.26% |

Both fail the combined guard. Full native expansion preserves/increases supported-source AI
point recall versus both references, but worsens REAL false accusations. Mean expansion still
loses individual AI sources. Evidence: `evidence/e53_source_held_out_expanded_controls_result.json`.
No candidate is promoted by its higher balanced accuracy.

Coverage-v2 contract `354b3ec4...00a0c8` keeps all original outer-validation assignments and
requires every one of the eleven components to appear in FIT at least once across folds. It
repeats the same twelve arms, not a new C search. All 36 fits complete; together with v1 this
is 72 fold fits. The same consumed TRAIN validation is reused, so v2 is not independent evidence
of a v1-to-v2 model improvement. Compare arms inside the following v2 table:

| V2 arm | Clean REAL FPR | Clean AI recall | Clean BA | Q75 REAL FPR | Q75 AI recall | Q75 BA |
|---|---:|---:|---:|---:|---:|---:|
| Full, old 2 views | 12.66% | 43.75% | 65.54% | 11.43% | 43.08% | 65.83% |
| Full, E51 3 views | 12.76% | 43.18% | 65.21% | 12.09% | 45.45% | 66.68% |
| Full, order/scale 3 views | 11.90% | 39.99% | 64.05% | 11.05% | 41.17% | 65.06% |
| Full, native-expanded 3 views | 19.50% | 66.39% | 73.44% | 18.76% | 65.11% | 73.17% |
| Mean, old 2 views | 11.43% | 55.64% | 72.11% | 10.53% | 54.25% | 71.86% |
| Mean, E51 3 views | 10.93% | 52.86% | 70.96% | 10.48% | 53.17% | 71.34% |
| Mean, order/scale 3 views | 11.55% | 51.47% | 69.96% | 11.18% | 53.11% | 70.97% |
| Mean, native-expanded 3 views | 15.17% | 65.52% | 75.17% | 15.07% | 64.95% | 74.94% |
| Full, C=0.001 | 12.71% | 42.67% | 64.98% | 12.54% | 46.22% | 66.84% |
| Mean, C=0.001 | 13.73% | 59.91% | 73.09% | 12.89% | 59.34% | 73.23% |
| Full, row-L2 | 12.22% | 42.15% | 64.97% | 11.70% | 44.11% | 66.20% |
| Mean, row-L2 | 12.27% | 48.02% | 67.88% | 12.24% | 49.05% | 68.40% |

**V2 survivors: zero.** Mean/E51 and mean/old2 pass pooled REAL improvement and AI preservation
interval checks versus full/E51 but fail supported AI-source no-loss. This is exactly why pooled
improvement alone is insufficient. V1/V2 bootstrap intervals condition on eleven observed publisher
components and fitted heads; per-pair multiplicity correction is not simultaneous certification
of a winner selected from all configurations. Evidence: `evidence/e53_coverage_result.json`.

Frozen-prediction diagnostics (`e53_diagnostics.json`) report macro/worst-source rates, individual
rescues/new errors and per-fold AUC/TPR@FPR<=10%, without new model inference or threshold changes.
For the v1 RR-only held-out fold, full/E51→full/native clean AUC rises 53.29%→69.32%. Native's
CAL-selected cut catches 77.39% AI but falsely accuses 55.76% REAL. Its evaluation-label-derived
TPR at <=10% FPR is only 36.49%, ruling out a simple threshold rescue on those observations.
By contrast, v1 fold 1 native/full reaches 96.36% AUC and 89.17% recall at <=10% FPR. These are
different held-out sources, not contradictory results or proof of universal performance.

Mean/E51 versus full/E51 on the v2 clean E32 FLUX subset loses twelve previously detected images
and rescues none (69 parents); CF loses nine with none rescued (69 parents). Nano Banana Pro has
only eight held-out parents and remains under-supported, not silently passing. The diagnostics
retain all sources/configurations, not only these illustrative weaknesses. ROC-derived cuts are
optimistic diagnostics, never transferred to production. Differently fitted raw scores are never
pooled into a headline AUC.

Input-association audit (`e53_shortcut_audit.json`): E51 TRAIN exact-224 fractions are 57.35% REAL
(2,314/4,035) versus 18.17% AI (353/1,943); all added native inputs have short side >=512. The
combined inventory is 32.89% versus 7.68% exact-224. Crop-dispersion median L2 norms are
10.70 REAL/23.56 AI before additions versus 27.15/27.13 combined. Only one original REAL has
near-zero dispersion; exact-224 must NOT be equated with zero dispersion, because global and
local preprocessing differ. Native REAL is nonsquare, while 2,152/2,652 native AI are square.
These associations motivate class-identical processing and bounded adaptation, not a geometry-
based authenticity rule or a claim that dataset dimensions alone caused the failed gate.

Engineering: full TRAIN numerical check (`e53_crop_dedup_full.json`) reduces 35,868 crops to
30,103 unique crops across 11,956 views, with exactly zero feature/score error and zero decision
flips. Its scope is all current TRAIN original/Q75, not future uploads; the prior small benchmark
is the only timing comparison. Serving remains unchanged. Fast class/source/parent weights remain
bitwise-identical to the old implementation and are used in the new controls/v2 fits.

Resource-only preparation (`e53_adaptation_probe.json`), frozen before steps: eight hash-ranked
TRAIN parents, four per class; 24 clean crops; existing DINOv2-S last two blocks plus a copied
E51 binary head, AdamW LR=1e-5/weight decay=.01, one warmup plus three timed steps. This is not
an accuracy study or an eligible fold model. Differentiable mean/population-std and converted
head errors are 2.38e-7/6.71e-8 before steps. All gradients finite; 3,553,537 trainable parameters,
frozen backbone hash unchanged. Median MPS step 0.20565s; sampled driver allocation 1.222GB,
not true peak and excluding decoding/evaluation. No candidate weights, CAL/test scores or training
accuracy are saved. A real successor must start from its own pretrained backbone/FIT-only head.

Decision: retain current serving/full research artifacts. Native expansion with order/scale views
was not completed and must not be described as a full 2x2. Next is a separately preregistered
restricted-adaptation study using eligible existing data; download only after explicit permission
and an independent-source/byte-budget contract. No consumed validation/test is relabelled E52,
and Module 2 held-out data remains protected. No image/weight/dependency download in this slice.

Serialization closure: `e53_artifact_replay.py` verifies all 72 saved head hashes and reproduces
all 286,944 archived validation predictions from pinned cached features. Maximum score error=0;
decision flips=0 (`evidence/e53_artifact_replay.json`). No new external test or model-selection
evidence. Final suite: 574 Python tests passed; compilation and `pip check` passed. The user
requested an office-to-home pause before any successor study, leaving no active training/download.

## 2026-09-09 — E54: fixed restricted adaptation versus continued-head control (preregistered)

Home-network continuation/download permission received. No paid API budget inferred. This study
uses E53 coverage-v2 source folds and the same 5,978 base parents plus 5,652 FIT-only native
additions (11,630 inventory parents: 7,035 REAL/4,595 AI). Native additions enter only a fold
where their publisher is FIT. No external CAL/DEV/final rows are introduced into optimization.

Two arms, three folds each, seed 54, two complete epochs: (1) continued head-only AdamW control;
(2) last two DINOv2-S blocks plus head, with 0.1 cosine preservation to the frozen pretrained
3072-D features. Each starts from the hash-pinned FIT-only native fold head, not the E51 full-data
head used in the earlier resource probe. Class/source/parent weights have mean one; deterministic
shuffling visits every FIT view rather than discarding AI replay. Batch=8 views/24 crops, head
LR=1e-5, backbone LR=1e-6, weight decay=.01, clip=1.0. Fixed final epoch only, no validation-based
stopping. The head control isolates continued optimization, not equality to logistic-regression L2.

Original/assigned-transport/Q75 preprocessing is cached identically to the frozen feature recipe.
Check feature/head parity on hash-chosen FIT examples from every FIT source before updates;
fail above 5e-5. Verify that frozen parameters retain their hash. Save model/optimizer/next-batch
position every 200 steps and at epoch boundaries. CAL and VALIDATION are opened only after the
fixed final epoch; choose the threshold solely on inner CAL, then report source-held-out TRAIN
validation. Compare both arms with the two old refitted recipes AND native-replay baseline using
the same REAL-improvement/AI-preservation guard and conditional publisher intervals. Separately
compare adaptation versus head-only. Three folds are not three independent seeds or final tests.

Data contract v2 SHA `26a169921797043d0978b1fae8acf97146ac4a72538f5c54a8ddd955aaf8679e`;
training/data receipts in `evidence/e54_*contract*.json`. The first data implementation repeatedly
decompressed an entire NPZ member inside the parent loop; preparation was stopped before a cache
or E54 fit/score existed. Preserve its original contract, fix one-time array loading and bind v2.
This was an engineering correction before outcome inspection, not a change in scientific inputs.

No quality result yet at this registration point. Successful preparation/training alone will not
authorize serving promotion, a full-data model, or opening independent E52. MNW acquisition is
evaluation-only, separately frozen and unscored; no part of it is an E54 training input. Existing
E43/E51/served models and Module 2 protected data remain intact.

## 2026-09-09 — E54 results and E55 registered follow-up

E54 completes all six fixed fits. Last-two-block final steps by fold: 6,054/5,244/1,770; elapsed
including scoring 2,363.42/2,028.13/741.14s. Every final checkpoint retains its frozen-parameter
hash; pre-fit feature error <=4.77e-7 and score error <=5.97e-7. No validation-selected epoch.

| Same 5,978 TRAIN OOF parents | Clean AI recall | Clean REAL FPR | Clean BA | Q75 AI recall | Q75 REAL FPR | Q75 BA |
|---|---:|---:|---:|---:|---:|---:|
| Native linear baseline | 66.39% | 19.50% | 73.44% | 65.11% | 18.76% | 73.17% |
| Continued-head control | 66.65% | 19.48% | 73.58% | 65.11% | 18.64% | 73.23% |
| Last-two-block + cosine anchor | 68.76% | 17.50% | 75.63% | 66.80% | 16.85% | 74.98% |

Against continued-head control, adjusted paired publisher bootstrap deltas are: REAL clean
[-3.93,-1.17] percentage points, REAL Q75 [-3.59,-1.08], AI clean [+0.136,+4.324], AI Q75
[-0.136,+3.743]. Thus aggregate REAL improvement is supported under this conditional analysis,
but Q75 AI preservation is not established. Supported-source losses remain (including GPT Image 1,
FLUX Klein Q75 and some RR topics), and all absolute fold acceptance checks fail. Comparisons
against the two non-native reference recipes also fail REAL-improvement requirements. Neither
arm advances. This is not the full-model E49 benchmark, three independent random seeds, a new
final, or a universal generalization claim. Full source transitions and per-fold ranking diagnostics
remain in `evidence/e54_diagnostics.json`; evaluation-derived TPR@FPR10 cannot be deployed as a cut.

Colour audit (exploratory, `evidence/e54_color_audit.json`): near-monochrome means mean per-pixel
max-minus-min RGB <=2 on the exact clean global crop. REAL 230/4,035, AI 34/1,943 in the base
OOF population; 229 of those REAL rows are RR and fold-0 FIT contains very few monochrome REAL
examples. Native baseline clean errors 169/230 versus 618/3,805 other REAL; adaptation 153/230
versus 553/3,805. This is source-confounded association, not relabelling or a colour-based detector.

E55 preregistered before derivative extraction and fitting: all 11,630 admitted E54 TRAIN parents,
same roles and all three input views, frozen pretrained DINOv2-S, full 3,072 features. Pillow
RGB->L->RGB per exact crop. Two arms x three folds: duplicate-view control and grayscale derivative;
80% original /20% derivative loss mass for every class/source/parent. Total mass fixed to the
base FIT three-view count as in E53 native expansion. Weighted StandardScaler and C=.01 lbfgs
head, max_iter=1000, seed=53; convergence warnings are errors. No stochastic resampling, AI row
removal, CAL/validation grayscale substitution, hyperparameter sweep or label changes. CAL chooses
cuts exactly as before; only unchanged original/Q75 outer-validation views are detector-scored.
Require control parity <=5e-5 and zero decision changes versus native reference, exact artifact
replay, relative no-loss guard and absolute fold gates. Only a passing result may authorize a
separate full-data/CAL+DEV stage, not direct serving or E52. All new derivatives remain TRAIN
material, not another independent source or downloaded data. E55 receipt: `evidence/e55_contract.json`.

E54 post-fit integrity closure: all 23,912 predictions from six reconstructed checkpoints reproduce
exactly, zero decision flips. Separate acceptance rejects both arms for relative and absolute gate
failures despite successful replay. Full local Python suite now 598 passed, one existing warning.

E55 operational interruption before any fit: at 21:18 local the machine reports 3% remaining on
battery, not AC. Stop extraction to protect work. Seventeen complete chunks retain 816/34,890
derivative views, verified contract/order/shape/finiteness and hashes; no partial file. This is not
an E55 quality result, early stopping on validation, or hyperparameter change. Resume the same
feature extraction and six prescribed fits only after charging is verified. The paired temporary
throughput diagnostic was also interrupted and must not be reported as a completed speed comparison.

2026-09-10 operational follow-up: AC restored but LaCie is absent from mount, physical-disk and
USB inventories. E55 scientific code/contract, inputs and hyperparameters remain unchanged; no
new fit or inference result. A separate guarded resume wrapper and eleven fixture cases verify
power/storage checks, checkpoint hashes, unfinished-stage selection and scoped process cleanup.
Real preflight correctly blocks the missing volume. Overnight heartbeat may resume the original
study only after the disk returns; these engineering tests are not evidence of detector quality.

Verification: full Python suite 609 passed (one existing warning); real guarded `run` refuses
missing storage without creating substitute mount/work directories. No new E55 model score.

## 2026-09-10 — E55 result and frozen FIT-only numerical audit

Completed 34,890 derivative views (11,630 parents), six fits, exact saved-head replay for each.
Feature SHA `161213c4e6818ac967787a3da0a24a6d3a84eb51822a843a0597305e28b85ee8`.

| Same consumed TRAIN OOF population | Clean AI recall | Clean REAL FPR | Q75 AI recall | Q75 REAL FPR |
|---|---:|---:|---:|---:|
| Native baseline | 66.39% | 19.50% | 65.11% | 18.76% |
| Duplicate control | 66.34% | 19.50% | 65.05% | 18.76% |
| Grayscale 20% | 67.78% | 20.12% | 66.03% | 19.45% |

Both arms fail acceptance. Grayscale pooled REAL FPR worsens and supported AI source losses remain.
Duplicate-control parity also fails: fold max score errors .001179/.007679/.002377; fold 1 changes
one Qwen Image 2 Pro AI decision in each transport. No tolerance relaxation or promotion. This is
not an independent final or directly comparable to historical full-model E49 recall. Results,
source comparisons and gate details: evidence/e55_result.json. No new evaluation reserve opened.

Post-result audit registered in PLAN before fitting, code/input hashes frozen and committed aed70a6.
FIT features, labels, sources and parents are bitwise/order equal in all folds. Maximum collapsed
80/20 weight error 2.22e-16. Saved native/duplicate heads have float32 coefficients, tol=1e-4 and
objective gradients approximately 6.5e-5 to 1.0e-4 when evaluated in float64. Exact same mathematical
FIT objective is tested using two fixed float64/tol=1e-8 fits per fold, C=.01/max_iter=1000/seed=53,
two CPU threads. Maximum FIT score differences: 1.3581e-6, 2.2342e-6, 6.9823e-7; iterations
118/119, 167/186, 157/159. All complete without convergence warning. The observed gradient may
exceed tol when another optimizer stopping criterion fires; absence of warning is not proof of
zero gradient. See sklearn's [LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

This supports numerical sensitivity, not reversed labels/input corruption. Precision and tolerance
changed together, so no individual attribution. FIT agreement is not held-out quality evidence.
No CAL/validation prediction, saved candidate, threshold change or external inference in this audit.
Old E55 rejection is unchanged. Contract and results: evidence/e55_audit_contract.json and
evidence/e55_audit.json. Two unit tests cover objective/gradient duplicate identity, finite-difference
gradient and invalid weights.

Archived errors: grayscale versus native monochrome REAL clean 152/230 versus 169/230, Q75 159/230
versus 170/230; other REAL clean 660/3805 versus 618/3805, Q75 626/3805 versus 587/3805. AI
monochrome misses 16/34 versus 21/34 clean and 15/34 versus 20/34 Q75. These net subgroup counts
remain source/content-confounded, not a causal colour diagnosis. Reject global grayscale and plan
coverage/content-matched follow-up rather than reusing this validation to sweep mixture ratios.

## 2026-09-10 — E56 source/processing coverage audit (no model experiment)

Preregistered metadata audit, code commit c815439; input hashes and outputs in
`evidence/e56_coverage_audit.json`. Admitted parents 11,630 (7,035 REAL/4,595 AI), eleven existing
publisher components. Ten single-class components contain 9,270 parents (79.71%); RR is the only
mixed component under the existing grouping. No publisher crosses active fold roles. Whole-pool
monochrome REAL 238, AI 120; exact224 REAL 2,314, AI 353. These are descriptive counts, not quality.

Fold-0 REAL: FIT 9/5,314 monochrome (0.169%), VALIDATION 229/1,250 (18.32%); CAL 0/471. Fold 2
FIT has only RR publisher for both classes. Topic metadata field is absent, scene field present
only for 1,000 REAL parents; missing normalized metadata is unknown, not a guessed scene/category.
Three fixture tests cover counts/missingness, exact metadata joins and publisher-role leakage.

Inference: coverage/processing mismatch remains plausible after E55's aggregate REAL regression,
but not established as the unique cause. E40 already used FIT-only DINO cluster weights, so a repeat
without a changed hypothesis would be weak. [B-Free](https://github.com/grip-unina/B-Free) motivates
content-matched training; do not equate simple grayscale or clustering with its paired generation.
Its COCO release remains inadmissible under current project publisher protection.

FiveK source preparation is metadata-only: five official resources, 4,725,181 bytes, no images,
weights, new inference or fitting. Source index parses 5,000 original identities, six subject
categories, three lighting groups. V1 stops on filename-vs-stem licence schema mismatch; separate
v2 exact-stem reconciliation admits all metadata, never image TRAIN eligibility. Both failures and
correction hashes preserved. Two parser and four licence-reconciliation fixture cases pass. A
12-parent decoder/overlap pilot is planned separately, not started and not detector evaluation.

## 2026-09-10 — E56 RAW pilot outcome and E57 preregistration

E56 pilot contract committed c9e484a before image GET, SHA
`ca8a293f3b03fc9336eefc74e32b289dc1639d6c0cd436d3a6417f5f62aaa308`. Twelve parents, exactly two per
publisher subject category, deterministic hash rank including unknown. Actual raw 122,074,000 B;
derived PNG 166,400,845 B. Isolated rawpy 0.27.1/LibRaw 0.22.1/NumPy 2.5.1/Pillow 12.3.0;
fixed full-resolution AHD, camera WB, sRGB primaries, gamma(2.4,12.92), no auto-bright, bright=1,
clip highlights, 8-bit. See [rawpy parameters](https://letmaik.github.io/rawpy/api/rawpy.Params.html).
This is a local RAW rendition, not native-camera JPEG or FiveK expert output. All twelve replay
to exact RGB array equality, no new image fetch. Some highlight clipping persists; no post-hoc
per-photo adjustments or exclusions. Shape/pixel hash validation passes.

Frozen protected snapshot plus MNW/HDR+ gives 150,883 reference observations: zero cross matches
under exact/raw/pixel and existing dHash<=4 +pHash<=4 screening; zero internal pairs. No inference
on references, no model scores, no TRAIN admission. A one-rendering heuristic cannot guarantee
all differently edited/cropped versions are found. No authenticity accuracy is measured here.
Evidence `evidence/e56_fivek_pilot_{contract,download,audit}.json`. Six fixture cases plus real
acquisition/decode/replay checks; full suite 626 passed, one pre-existing warning.

E57 next hypothesis registered in PLAN before new image selection: 201 candidates from min(12,N)
per each of 18 subject x lighting cells, score-blind filename hash. Subject/light are publisher
annotations, not ideal coverage guarantees; retain small cells as 4/5, not artificial replication.
New source FIT-only, unchanged three outer folds and inner CAL. Two fixed float64 heads per fold
(native control and plus FiveK), C=.01/tol=1e-8/max_iter=1000/seed53; preserve all AI rows and total
loss mass. Compare all unchanged gates and historical references with exact replay; no validation
sweep, new reserve score or promotion. Freeze acquisition, feature and training contracts before
their respective stages. This source supplement is not a reproduction of B-Free paired alignment.

## 2026-09-10 — E57 pre-score RAW eligibility amendment

Frozen v1 acquisition rejects the 62nd selected DNG after 61 completed decodes. DCS460D
`a1854-kme_290.dng` has camera WB [0,1,0,0], Bayer RGB, rather than valid as-shot RGB gains.
No detector features/fits/scores or threshold choices precede this amendment. The operational
guard stops without completed acquisition receipts. Preserve failed v1; do not adjust its decoder.

Separate preregistered v2 (f840136) keeps selection, decoder, HTTP identity, caps and licences;
model-blind metadata preflight requires four finite WB values and positive first three, exactly
the existing decoder predicate. Exclude every unsupported case without replacement; preserve
RAW/hash/reason and report lost content/light coverage before admission. Do not substitute
daylight/auto-WB or remove images according to detector errors. This is an eligibility restriction
with potential camera-coverage bias, not improved accuracy or full camera support.

Six-fit protocol remains unchanged: float64 native control versus new REAL supplement,
three existing source folds, all existing AI views and their loss weights retained, original CAL
only, all old acceptance gates plus paired stable-control comparison. Source code is committed
before execution; model contract freezes admitted manifest and input hashes before features/fits.
633 unit/regression tests pass; actual acquisition and subsequent model experiment are still
pending completion at this checkpoint. Final reserves remain closed; serving unchanged.

## 2026-09-10 — E57 completed: reject REAL-only source supplement

199 admitted FiveK originals, 597 fixed views, pretrained parity max error 0; all six fits
finish and saved artifacts replay exactly. Two unsupported as-shot-WB originals excluded
pre-score without replacement. No overlap found against 150,883 references (heuristic, not
semantic guarantee). Model contract SHA
`87cf485974e8c53b5e93d15c0afb64767f3dc5961ca291aa67ab917b71aeb222`.

| Same 5,978 source-held-out research parents | Clean AI recall | Clean REAL FPR | Q75 AI recall | Q75 REAL FPR |
|---|---:|---:|---:|---:|
| Stable native64 control | 66.2378% | 19.5291% | 64.9511% | 18.7361% |
| Native64 +199 FiveK FIT parents | 66.4436% | 18.3147% | 65.8775% | 18.2404% |

Versus stable control, 20k paired publisher-bootstrap adjusted intervals in percentage points:
REAL clean [-2.2892,-0.8091], Q75 [-0.7028,-0.4239]; AI clean [-1.0732,+1.3237], Q75
[-0.9785,+2.4341]. Aggregate REAL improvement is supported conditional on these fitted folds,
but AI preservation is not. Seedream-5 clean/Q75 recall loses 5/7.5 points; GPT Image 2 loses
5/6.25; Qwen Image 2 Pro loses 2.5/6.25. Other complete source deltas and historical-reference
comparisons are retained in `evidence/e57_result.json`. Small-source limits are not waived.

Fold 0 RR remains a representation/transfer concern: supplement AUC clean/Q75 .7045/.6744,
REAL FPR 49.28/49.84%, AI recall 74.41/72.07%. Fold 1 AUC .9137/.9275 but recall
50.42/53.33%; fold 2 AUC .8965/.8991, recall 63.17/63.46%. Different CAL cuts and source
composition forbid pooling raw-score AUC or equating these results with E49 full-model scores.
Every absolute fold gate fails; neither arm is eligible. No deployment, independent final,
three-seed claim, new cut, post-hoc weight sweep or automatic acquisition enlargement.

E58 score-only diagnosis preregistered in PLAN and frozen after E57 rejection: paired
source/fold/transport error transitions and optimistic threshold envelopes at fixed FPR10 and
AI80/95. These consumed-label diagnostics cannot produce deployable cuts or certify success.
UnivFD's pretrained representation motivation remains relevant, but E47 already tested its
official CLIP detector; this is not an untried discovery. B-Free's paired real/fake generation
is materially different from our unpaired FiveK addition; protected COCO is still inadmissible.
Sources: [UnivFD paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html),
[B-Free authors](https://grip-unina.github.io/B-Free/).

## 2026-09-10 — E58 existing-score diagnosis completed; E59 engineering probe registered

Code 3c46ada, diagnostic contract36c7a25 bind all six E57 artifacts/score streams. No new
model scores, cuts, fits or test exposures. Four fixtures cover tie atomicity, permutation,
perfect/reversed rankings, invalid input and paired truth identity. Consumed-label envelopes
are optimistic diagnostics, not validation-selected deployable operating points.

Supplement rescues/new errors: REAL clean105/56, Q75 76/56; AI clean50/46, Q75 62/44.
Seedream Q75 loses six/80 detections and rescues none; GPT Image2 Q75 loses five/80 and
rescues none. Full source/fold/condition transitions retained, not only favourable examples.

| Frozen scores, optimistic AI recall at REAL FPR<=10% | Native clean | Supplement clean | Native Q75 | Supplement Q75 |
|---|---:|---:|---:|---:|
| Fold0 (RR) | 30.0901% | 37.2973% | 27.2072% | 34.5946% |
| Fold1 | 78.9583% | 75.8333% | 82.9167% | 80.0000% |
| Fold2 | 66.0057% | 66.2890% | 68.8385% | 68.2720% |

No AI95/FPR10 feasible cell; only fold1 Q75 reaches the weaker AI80/FPR10 diagnostic pair.
The supplement needs REAL FPR58.72/65.76% to reach AI80 in fold0 clean/Q75. This is evidence
against scalar threshold repair on these heads; it is not a proof against every architecture.
The weaker target is diagnostic only and never replaces AI preservation or the full gate set.

E59 preregistered resource-only probe: existing SHA-pinned CLIP ViT-L/14, official image
normalization on E54 cached RGB224, eight deterministic FIT-only parents (four/class),24views,
72crops twice after one warmup. Frozen eval, no head/gradients, 768-D finite embeddings,
replay<=1e-5; <=15min guarded AC/LaCie execution. Record runtime/sampled-not-peak memory,
not accuracy or full-scale guarantees. Selection fixture passes; full suite638 passed. Only
after this check freeze a complete existing-data CLIP-only/DINO+CLIP/control experiment.

E59 resource result: contract SHA
`8ad483b0c57be06a948c3afa8a3b544f5d25b9a528e6ad1cbac1c5f84f7ad68a`;
two MPS passes7.9457/7.8218s for24views each; raw768-D embedding replay max error0.
Sampled driver allocation2,194,358,272B, not measured peak. No classifier output, accuracy,
fit, new data/weights or saved candidate. Existing upstream tracked code unchanged. The small
sample supports implementing resumable full-scale extraction, not a quality claim or reliable
full-run time estimate. Entire protected CAL/DEV/final/Module2/reserves remain unchanged.

## 2026-09-10 — E59 full feature/pre-fit protocol registered and extraction started

Existing E54-only inventory11,630 parents/34,890views, no new FiveK, external CAL/test or
Module2 data. One parent keeps existing clean/assigned-transport/Q75 and three RGB224 crops.
Frozen cached CLIP ViT-L/14, official image normalization, float32 eval, three-crop batch;
aggregate raw768-D vectors as mean+population-std1536, no L2 or score-selected transform.
Save raw crop vectors and aggregate chunks with parent/binding/content hashes. Frozen contract
224382c10f2831a9a522bb668e952c30b3a1be890fa2219044f99e7679605954 (f533877) precedes
full extraction. Same-batch restart parity<=1e-5; first actual worker reports exact replay0.

Planned nine fits: DINO3072-only, CLIP1536-only, concatenated4608, three unchanged source folds.
Weighted FIT-only StandardScaler then float64 logistic regression C=.01/tol1e-8/max1000,
seed53/lbfgs/twoCPUthreads, convergence warnings fail. All AI replay and original base FIT
view-count total weight mass retained; same CAL-only cuts and clean/Q75 held-out observations.
Compare complete historical references, stable E57 native and within-E59 DINO control with
all original absolute/paired/source-wise AI-preservation gates. No best-mixture/feature sweep.
Separate actual training code/feature/artifact contract required before fitting. No fit yet.

Operational guard starts06:22, <=60min, one lock,2s AC/disk monitoring and owned-group cleanup.
External chunks support verified resume; interrupted completed archive may only be reused after
exact reconstruction matches. Code314593a, three new fixture cases; full suite641passed.
No classifier scores or new independent test exposure. Full feature completion remains pending.

## 2026-09-10 — E59 training implementation prepared before feature completion

While the original feature worker progresses past2,700 parents, implement only the registered
nine fits (three representations x three source folds). PLAN6e1a648 adds an explicit stable
DINO control parity gate against paired E57 native64 scores and CAL cuts: <=5e-5, no verdict
changes; a mismatch blocks interpretation/promotion, never retunes the tolerance. No fits yet.

Feature completion requires archive plus both bound receipts; only afterward can actual model
input/code/reference hashes freeze. Every arm uses identical AI rows and loss weights, verified
by per-fold weight SHA and comparison with prior AI row counts. Fixed weighted StandardScaler
on FIT only, float64 C=.01/tol1e-8/max1000/seed53; same CAL clean/Q75 cuts, OOF and all
historical references. All nine saved heads need exact artifact replay; source-wise/paired and
absolute gates plus control parity determine research eligibility, never automatic deployment.

Operational runner shares the feature lock, requires completed inputs and executes freeze ->
fit -> report with2s power/storage checks, <=60min and owned-child cleanup. Unreceipted saved
head stops for explicit audit rather than overwrite. Six fixtures cover fixed representation
dimensions/order, weight mass, control score/truth/decision parity, incomplete features, pending
nine-fit completeness and lock exclusion. Full suite646passed before the final added lock case;
focused six-test suite passes afterward. Real-data model contract and fits remain pending.

E59 operational continuation07:53: first feature invocation reaches its frozen60min deadline,
preserving4,465 parent chunks (173,197,992B, no pending partials). Old guard/worker absent;
power/real external storage checked. Resume unchanged code/contract in another bounded run,
revalidating every cached parent before reuse. Log e59_features_20260910T075334.log. No
scientific amendment, resampling, re-inference of verified completed chunks, fits or accuracy
measurement; completion receipts still pending. The filesystem count alone is not an audit pass.

09:24 continuation: second bounded run preserves8,308 chunks/342,619,493B, no partial file,
then exits at deadline. Resume same feature contract after AC/real-disk/absence-of-old-workers
checks. Operational-only waiter (plan80f8bff/code530513e) binds the observed guard PID,
requires completed archive+both receipts, and replaces itself with the existing training guard
using only remaining minutes of a60min combined budget. No feature/scientific code amendment,
new fit, score or test exposure. Full suite649passed; training remains conditional on complete
features, and any failed/incomplete handoff requires inspection at the next scheduled check.

User-requested pause before office execution: E59 extraction/queued handoff terminated cleanly,
9,599 parent chunks preserved (399,641,982B); no E59 model contract/fits or performance scores.
No feature protocol changed. Plan pivots to a frozen E43-S champion-preserving correction;
artifact SHA matches E49 binding. This remains a proposal requiring exact teacher exposure/
TRAIN-CAL-DEV protection audit and pre-fit hyperparameters. Warm-starting E43 does not permit
calling overlapping source-fold validation unseen; retention losses do not prove no AI loss.
No new download or experiment executed; scheduled continuation paused until user restarts.

### 2026-09-10 — E49-C gate-count erratum during documentation review

The earlier phrase **"FAIL 11/20"** means the overall experiment failed, but incorrectly
suggests 11 failed checks. The saved `evidence/e49_final_result.json` has **11 passed / 20
total, therefore 9 failed**: publisher-original passes 6/10 and social-Q75 passes 5/10.
The overall FAIL, consumed evaluation status, scores and promotion restrictions are unchanged.
This is a documentation correction from existing evidence, not a new metric opening or experiment.

## E60 — Frozen E43 preservation: offline reference and role audit (registered 2026-09-10)

User authorizes the champion-first plan with zero downloads. Stage 1 reconstructs the exact
E43 FIT parent/source membership from pinned feature metadata, compares admitted E51/E54
populations, checks modern protections and records possible local comparison populations.
No detector fitting or new validation scoring is authorized by this audit registration itself.
Freeze the resulting eligible data, one correction recipe and acceptance rules before any fit.
Keep E43 and all historical contracts unchanged; do not resume E59 or open final reserves.

Audit v1 stops before model scores on a case-sensitive TRAIN check. E54 contains 4,278 admitted
lowercase `train` rows plus 7,352 uppercase `TRAIN` rows; all labels are integer 0/1. Audit v2
normalizes role case, preserves population and all exclusions, and keeps the original v1 code
and contract. This is an engineering correction before experiment fitting, not a relaxed split.

Before audit-v2 freeze, inherited acquisition imports attempted Kaggle OAuth introspection
(no dataset download; request timed out). Remove the import side effect from the E60 path,
verify admission JSON/hash bindings directly, and deny network connections for new execution.

**Audit-v2 completed after interruption:** E43 FIT8,844 parents (5,041 REAL/3,803 AI),19,648
views. Current11,630-parent TRAIN retains4,278 teacher parents with identical encoded bodies;
7,035 REAL/4,595 AI.1,860 teacher AI are absent, including360 now-protected CAL. E54 validation
overlaps teacher FIT by2,360/480/1,438 parents. Of10,508 unused audited native parents,2,385
outside current recorded groups are AI-only; no balanced fresh group-disjoint DEV in this pool.
Evidence/e60_audit.json. Missing old rows are not restored without admission.

**Single correction registration, before fit:** original E43 backbone/scaler/head and two cuts
frozen. X=scaler_E43(features)/sqrt(3072), CPU float64 after original scaler transform;
delta=2*tanh(X*w/2), no intercept, w=0.200 full-batch Adam steps, lr.02, betas.9/.999, eps1e-8,
final step; L2 .01. All34,890 cached TRAIN views, including every4,595 admitted AI parent.
Supervised BCE uses logits relative to fixed AI cut; class/source/parent weights, teacher-false-
AI REAL views2x, re-normalize each class to.5. Retention penalty10/2 times weighted squared
negative delta on teacher-correct AI. True labels remain supervision; penalty is not a guarantee
on unseen AI. No sweep, validation early stop, threshold change or downloads; <=20min/two CPU threads.

Require exact zero-initialization E43 scores and saved-correction replay. After candidate hash
freeze, one consumed E49 regression uses unchanged4,000 observations/cuts, <=15min inference,
batch16, network connections denied. Baseline replay<=5e-5 and zero decisions changed. Freeze
raw scores before metrics. Existing20 absolute/selective gates plus REAL-FP reduction, pooled/
source AI no-loss and four familywise paired source-cluster intervals (20,000 resamples, seed60,
quantiles.00625/.99375). Consumed diagnosis only; no winner selection or fresh final/serving
claim. Archive failure; no automatic second fit or retuning on the report.

**FIT complete:**200 steps,9.12s, full11,630-parent/34,890-view pool and all4,595 AI parents.
Candidate SHA582d6c4f020ce309c8e44a88383ef9d69c559773d6332694bd203e1132415bc5; original E43
unchanged. Exact zero-init and saved-head replay errors0. Max observed TRAIN delta.171559;
the analytic global bound remains2. TRAIN clean/Q75 REAL FP15.0675->14.8543% /13.8024->13.5039%;
AI recall98.1066->98.0849% /97.3885->97.3232% (one/three new misses, zero AI rescues).
Assigned-transport TRAIN also has one new AI miss. No validation quality claim from these rows.
Final weights are frozen; no further optimizer steps or tuning based on E49.20 focused tests pass.

**Consumed E49 regression complete — insufficient improvement, no promotion.** All4,000 views
scored in183s, exact frozen E43 replay (max error0; zero binary/selective decision changes).
Full raw score receipt is locked before metrics (`evidence/e60_regression_scores.json`).

| Condition | E43 AI recall | E60 AI recall | E43 REAL FPR | E60 REAL FPR | REAL rescued/new errors | AI rescued/new errors |
|---|---:|---:|---:|---:|---:|---:|
| Publisher original |94.3%|94.3%|39.1%|39.0%|1/0|0/0|
| Social Q75 |95.5%|95.5%|49.0%|48.8%|2/0|0/0|

All supported AI sources preserve observed recall. REAL source-cluster familywise improvement
intervals[-.004,0]/[-.008,0] include zero; AI intervals[0,0] are conditional on these consumed
sources, not guarantees about unseen generators. Original/Q75 AUC.902425->.902086 /
.868850->.868826; the change does not establish a better ranking. Existing20 absolute/selective
gates still fail. Training and E49 results differ: the one/three TRAIN AI losses remain reported.
Keep E43 as reference and do not promote E60 or retune its coefficients/bound/cuts using this
report. No fresh independent DEV/final opened. Detailed results: `evidence/e60_regression.json`.


## E61 — Strict TRAIN replay retention gate (registered 2026-09-10)

**Engineering validation, not a new learned model or independent experiment.** Literature
review and interpretation: IMAGE_FORENSICS_REFERENCE 2026-09-10 addendum. GEM motivates
explicit preservation constraints but is not implemented here; average loss penalties do
not certify per-image recall. E60 already reported TRAIN losses 1/1/3 on clean/transport/Q75.

Freeze input/code hashes before executing `experiments.e61_replay_gate freeze` then `run`.
Use only the existing 11,630 admitted TRAIN parents ×3 conditions, all4,595 AI parents;
original E43 head/scaler and E60 weights remain unchanged. Fixed AI cut
0.07940196245908739. No optimizer, new features, evaluation/held-out reads, thresholds,
source downloads or serving changes. CPU two threads; existing feature cache only.

Acceptance test for the **software**: E43 against itself passes; frozen E60 must reproduce
known losses and be rejected. Zero newly missed AI views is mandatory for a future candidate
at this preliminary guard; AI rescues cannot cancel misses. Report each source/condition
and parent/view counts separately. Invalid/nonfinite/missing or protected inputs fail closed.
Passing is necessary only and never sets promotion_allowed=true; later independent REAL
improvement and pooled/per-source AI non-regression requirements remain unchanged.


## 2026-09-10 — E61 engineering outcome

Frozen contract SHA `0fd0c9ff216d7a915d5208fbab3f06d45df91cd18af389a8d387b057e040ba6a`.
One cached TRAIN replay completed: E43 self-comparison passes; unchanged E60 fails with
1 clean,1 assigned-transport and3 Q75 newly missed AI views, affecting5 distinct parents.
This reproduces known E60 TRAIN behavior; it is not independent model-quality evidence.
All11,630 admitted parents/34,890 views were checked, including4,595 AI parents. Zero fits,
zero evaluation rows read, zero data downloads; reference/candidate hashes and serving
unchanged. Evidence: `evidence/e61_replay_contract.json` and `evidence/e61_replay_gate.json`.

Focused verification:37 tests passed in1.39s (retention gate, E61 tamper/write-once/network
checks, E60, E49 evaluation and parent weighting). The new gate catches equal-pooled-recall
source swaps, exact-threshold losses, incomplete views, duplicate parents, invalid scores,
vacuous controls and protected roles. It is available to future experiment runners; historical
frozen runs and the serving path were deliberately not retrofitted. Any future candidate
must explicitly run it before independent evaluation; a pass cannot authorize promotion.


## 2026-09-10 — E62 preregistration: constrained correction after explicit continuation

User requests achieving the existing test targets without reducing AI recall. Register one
new TRAIN mechanism before fit: frozen E43 scaler/head, admitted E54 cached11,630parents ×3
views including4,595AI. PCA64 TRAIN-only randomized seed62 power3; transform explicitly after
fit and normalize by sqrt(explained variance), append intercept. Add linear correction to
E43 logit. Class/source/parent-balanced BCE at unchanged AI cut0.07940196245908739; E43 REAL
false alarms get2x within-class weight then each class mass.5. L2.01 including intercept.

SLSQP CPUfloat64,200 iterations,ftol1e-9,zero start,final solution only,2threads,20min ceiling
and30% battery floor. All reference-caught TRAIN AI have a linear inequality preserving their
decision with min(original margin,1e-7) safety margin; correct REAL corrections <=0. This
explicit constraint is a different TRAIN-motivated hypothesis from increasing E60's penalty.
The PCA projection may discard useful distinctions; infeasibility/no gain is a possible result.

Require solver success, max constraint violation<=1e-8, exact saved score replay, E61 zero new
AI misses, zero new REAL false alarms and strictly lower REAL FPR across clean/transport/Q75.
No tuning on evaluation rows. No new data, packages or weights; no protected role change.
If TRAIN checks pass, permit one separately frozen consumed E49 comparison, with all20 original
absolute/selective gates and stricter reference AI-retention checks. E49 remains consumed:
any measured benchmark pass is not independent final/generalization certification. No automatic
promotion and no further parameter selection from that comparison. Source/code hashes are
recorded in e62_contract.json before fitting; historical E60/E61 implementations stay frozen.


### E62 TRAIN result and consumed comparison registration (2026-09-10)

One fit completed in9.98s/22 SLSQP iterations; constraint violation1.17e-15. Candidate SHA
`7d56af6c721fda86284839b6f6d247f2f723f741ded88f14a8bd32ba95c9b557`, fit contract SHA
`a4bbd3ab7dbe7cfbfb91300a2b049024df16b40130ebf0d465aac99ea242454c`. PCA64 explains59.63%
of TRAIN feature variance. Exact zero initialization and saved replay passed. REAL FPR
clean15.0675->13.6887%, transport16.1336->14.4279%, Q7513.8024->12.5089%:97/120/91
rescued REAL observations, zero new REAL false alarms. All admitted AI replay retained;
zero new AI misses and one Q75 AI rescue. This is training behavior, not test success.

All preregistered TRAIN guards pass. Register one4000-view consumed E49 comparison of this
immutable candidate before scoring: original/Q75, unchanged cuts, baseline reproduction
<=5e-5 with zero decision drift, all20 absolute/selective gates plus pooled/per-source AI
no-loss and paired source-cluster intervals.20,000 bootstrap resamples,seed62,four-family
quantiles.00625/.99375; CPU2 threads, MPS extraction from cached weights, batch16,15min
ceiling, battery30%. Lock all raw scores before metrics; do not select parameters from them.
No new data or fresh-final claim. E62 regression code copies frozen E60 evaluator mechanics
into a new bound namespace; original code/artifacts remain unchanged.

Pre-fit17 objective/constraint/retention tests passed. A development-only script extraction
initially searched for a nonexistent main function; it failed before writing any runner or
contract, then used the actual module entrypoint boundary. No experiment output was altered.


### E63 preregistration before opening E62 benchmark metrics (2026-09-10)

E62 TRAIN residual REAL FPR remains13.69/12.51%, with RR real47.52/43.84%; TRAIN evidence
alone motivates a second, nonlinear hypothesis. E62 benchmark metrics remain unopened;
no E62 benchmark metrics have been used to select this recipe. This is an explicitly bounded
second candidate, not a threshold/rank/loss sweep. Keep E62/E63 comparisons separate.

Use the same admitted TRAIN population, frozen E43, PCA64(seed62,power3), balancing, fixed cuts,
linear decision-preservation constraints, SLSQP200/ftol1e-9/L2.01/CPU2/20min,zero initialization
and strict E61 guard. Replace the linear PCA correction features with64 Gaussian RBF features:
weighted KMeans64 on reference-false-positive TRAIN REAL views in whitened PCA space,
seed63,n_init3,max_iter100; sample weights are class/source/parent weights restricted to those
views. Sigma is the median positive distance of those TRAIN views to their assigned center.
RBF=exp(-distance_squared/(2*sigma_squared)); center and standardize each basis column on all
TRAIN views, append intercept. This permits curved corrections with the same65 coefficient
count; no raw test scores, sources or identities enter fitting or representation choice.

Apply all original TRAIN acceptance guards before any separately frozen benchmark comparison.
If TRAIN fails, archive and do not score E49. If it passes, one immutable consumed E49 comparison
uses unchanged20 gates and reference AI retention; no subsequent recipe selection from scores.
No source/model/package downloads, fresh-final claim or serving promotion. RBFs are learned
features, not a hand-written metadata/camera veto. Passing finite TRAIN constraints cannot
promise preservation on unseen images.


### E62 benchmark failed; E63 fit frozen and comparison registered (2026-09-10)

E62 consumed E49: REAL FPR39.1->35.8% /49.0->46.8%, with33/23 rescues and0/1 new REAL
errors. AI recall94.3->93.8% /95.5->95.1%, with5/4 newly missed AI observations. REAL paired
source intervals favor improvement, but pooled/source AI preservation fails; all20 benchmark
gates do not pass. Reject E62. Original E43 score reproduction is exact. No deployment.

E63 recipe/contract were frozen and fitting started BEFORE E62 benchmark metrics were opened;
E63 is not a response to the observed E62 test losses. E63 TRAIN fit completed in14.38s,
39 iterations, violation2e-15; no newly missed AI view or new REAL
false alarm. REAL FPR clean15.0675->13.7313%, transport16.1336->14.7832%, Q7513.8024->12.5657%
(94/95/87 rescued observations). TRAIN guards pass. Candidate SHA
`0d29e7a07e9d1c8d3c89adb5c87a27e251f24641967a3e5f26608f27240841dc`. Candidate remains research-only.

Register one E63 consumed E49 comparison with the unchanged protocol before its scoring:
4,000 original/Q75 observations, fixed cuts, exact E43 replay within5e-5/zero decision drift,
all20 existing gates and pooled/source AI no-loss,20,000 paired source-cluster bootstrap seed63,
four-family quantiles.00625/.99375. Complete raw score lock before metrics. CPU2/MPS,batch16,
15min and30% battery floors; source downloads0. No hyperparameter changes or winner selection
from these consumed reports. E63 copied the E62/E60 evaluator mechanics into its own bound
namespace; all prior code, contracts and outcomes remain immutable.


### E64 preregistration — isolate an unnecessarily strong TRAIN constraint (2026-09-10)

Inspection of the TRAIN optimization (E63 test metrics still unopened) identifies an avoidable
restriction: E62/E63 force every correctly classified REAL logit to never increase. The actual
user requirement concerns classifications/FPR, not unchanged confidence. E64 isolates this
mathematical distinction with exactly E62's PCA64/seed62/weights/L2/optimizer/cuts and AI
constraints. Change only correct-REAL upper bound from0 to
max(logit(AI_cut)-reference_logit-1e-7,0): preserve the REAL decision while allowing motion
inside its original safe margin. AI constraints and E61 zero-new-AI-miss rule remain unchanged.
No changes to E62/E63 code or outcomes; new code/input hashes frozen before a single fit.

Strengthen pre-test feasibility: solver success/violation<=1e-8/exact replay, zero newly missed
AI, zero new REAL false alarms, and REAL TRAIN FPR <=10% in ALL3 conditions. If this TRAIN-only
necessary screen fails, do NOT open E49 for E64. If it passes, permit one separate preregistered
consumed comparison with the original20 gates and E43 AI-retention criteria; no parameter
selection from prior benchmark errors or individual test identities. No downloads, role changes,
fresh-final claim or deployment. This is an isolated constraint ablation, not a bound/L2 sweep.


## 2026-09-10 — E62–E64 final checkpoint: targets NOT reached

| Candidate | TRAIN REAL FPR clean / Q75 | Consumed E49 REAL FPR original / Q75 | Consumed E49 AI recall original / Q75 | Decision |
| --- | --- | --- | --- | --- |
| Frozen E43 | 15.0675% /13.8024% | 39.1% /49.0% | 94.3% /95.5% | Retain reference |
| E62 constrained PCA64 | 13.6887% /12.5089% | 35.8% /46.8% | 93.8% /95.1% | Reject AI regression;11/20 gates |
| E63 constrained RBF64 | 13.7313% /12.5657% | 35.8% /46.7% | 93.8% /95.3% | Reject AI regression;11/20 gates |
| E64 decision-margin ablation | 11.8977% /11.3291% | Not scored | Not scored | TRAIN <=10% guard failed |

E63's paired REAL-improvement intervals are[-.05,-.02] original and[-.033,-.013] Q75;
AI intervals[-.009375,-.00092593] /[-.00520833,0] do not establish retention. Its original
AI losses span FLUX.2 Klein9B, GPT Image2, Midjourney7 and Seedreamv5.0; Q75 losses span
FLUX.2 Klein9B and Z-ImageTurbo. No REAL new false alarms,33/23 rescued REAL observations,
but5/2 newly missed AI observations. Both E62/E63 fail the user's no-AI-loss requirement.

E64 changed only the correct-REAL linear constraint from nonpositive shift to preserving
its original decision margin. Contract SHA
`f883fb815cd454e15ab7199f76b731c3f2a1c3cccae3afb0535d8f122bb12aee`; candidate SHA
`bcf8e7e6d0cba1905088cda34350256b8e953da812d9d0d314773e948b29224d`.
One fit10.33s/28iterations, maximum constraint violation4.69e-16. Zero newly missed AI,
zero new REAL errors;223/190/174 rescued TRAIN observations across clean/transport/Q75.
Transport FPR13.4328%; all3 TRAIN rates remain above the preregistered10% pre-test ceiling.
No E64 evaluation contract or scores created. Do not remove that failed guard after the result.

These are exactly three fixed recipes, not three seeds or an estimate of seed variability.
E63 was preregistered/frozen before opening E62 metrics; E64 before opening E63 metrics.
No test-derived threshold, rank or regularization sweep. E62/E63 each completed one4000-view
consumed comparison with exact E43 reproduction; their raw scores were locked before metrics.
E64 stopped at TRAIN. No fresh-final/generalization or v1-success claim is supported.

49 focused tests passed in1.23s: objective gradients, feasible/conflicting decision constraints,
REAL safe-margin motion, RBF behavior/invalid bandwidth, retention invariants, E60/E61/E49
and source/parent weighting. Diff whitespace check passed. No source/weight/package downloads;
all fitted arrays stay on the external volume. Original E43 hash and web serving unchanged;
E59/heartbeat remain paused. All experiment processes finished, locks released.

Interpretation: finite TRAIN constraints prevent replay regressions but did not generalize
AI preservation on these consumed external sources. The isolated REAL-margin correction
improves TRAIN flexibility but does not itself reach the necessary TRAIN target. These results
rule out these recipes as accepted candidates, not all possible DINO corrections. Next work
needs a new representation/data hypothesis with a valid development population, not another
rank/loss sweep against E49. The home metadata-only queue remains pending; do not claim that
those untested REAL-only datasets will necessarily solve the detection problem.


## E65 — score-blind camera/noise pilot registered (2026-09-13)

User now authorizes needed data downloads and explicitly requested starting while connecting
power; subsequent pmset confirmed AC/charging. WIFD full tree is101,432,609,110B/6,944 blobs,
not a sensible automatic full download. Its1,838 SDR JPEGs cover10 cameras (not all14 cameras
in the entire corpus). Select at most4 distinct exposure/aperture setting tuples at each
camera's min/max recorded ISO; hash-rank with fixedE65 seed, omit reference/burst/AEB files.
SDR scene independence is unknown; do not equate files or cameras with independent scenes.
Pinned revision3f577edf0b14c686aa08e8d0d8ae07a83ba44f26; MIT README/license blobs verified.

RawNIND official Dataverse metadata v1.0:2,845 files/120,172,531,162B, dataset-level CC-BY-SA-4.0.
Its permissions.txt contains per-scene exceptions, including research-only entries: therefore
select ONLY explicitly CC0 scenes with author attribution, excluding official test reserves
from both dataset.yaml and test_reserve.yaml. Four hash-ranked scenes per Bayer/X-Trans,
one clean RAW and one highest available ISO<=6400 RAW per scene, preserving filename SHA1.
No RAW/scene selection uses pixels/model scores. Known-field hashes and unrestricted flags
are required. This pilot is REAL-only DIAGNOSTIC_DEV, whole publishers reserved, never TRAIN
or balanced final; future separate decoding/overlap and scoring contracts required.

Freeze all metadata/code and selected file identities before pixels. Total payload ceiling2GiB,
two download workers,40min cap, external volume with10GiB reserve. Verify Git blob SHA1 for WIFD,
MD5 plus filename SHA1 for RawNIND, then record SHA256. Resume partial HTTP ranges; if the
server ignores Range, truncate/restart rather than append. No existing payload/receipt overwrite.
No source code, executable or full repository clone is downloaded. Originals stay outside git.

Metadata recovery: initial gh recursive-tree response was incomplete; curl retry obtained the
complete non-truncated tree. System Python lacked trusted CA setup; requests in the existing
venv succeeded with normal TLS verification (no certificate checking disabled). No image bytes
had been acquired at that point. E43/E59/serving and existing protected datasets remain unchanged.


E65 pre-payload schema correction: initial manifest freeze stopped before writing a contract
because RawNIND includes three nonnumeric ISO names (`ISOHi 3200`/`ISOUnknown`). Unknown or
extended ISO names are now excluded before numeric selection, without guessing effective ISO;
regression fixture added. No image or model score was acquired and no frozen contract altered.


### E65 payload complete / score-blind audit plan (2026-09-13)

83 files (67 WIFD JPEG,16 RawNIND RAW),1,009,998,251B downloaded and publisher-checksum
verified. Receipt SHA `0aab1b2fb544e415b9018084ab13af2cd0bd17761ee3a163b68cdf601c750fd1`.
User said start while connecting power; AC/charging subsequently confirmed. No model scores.
Next: fixed isolated E56 RAW decoder, pinned runtime, canonical E51 fingerprints against
150,483 protected reference bodies plus300 MNW,100 HDR+ and199 FiveK records. Failures and
cross-reference exact/confirmed perceptual matches quarantine; RawNIND quarantine propagates
across its scene pair. Internal similar observations remain linked descriptive repetitions,
never independent scene evidence. No score/brightness exclusions or replacement draws.
Whole WIFD/RawNIND publishers remain diagnostic-only; no TRAIN or final admission.


### E65 decode/identity audit complete; diagnostic preregistration (2026-09-13)

All83 originals decoded successfully, including16 RAWs with frozen E56/LibRaw0.22.1 runtime.
151,082 reference records screened;0 cross-reference matches under fixed exact/pHash+dHash
criteria. Eight internal matched pairs are recorded, not discarded or counted independent.
83 diagnostic observations remain, RawNIND8 scenes. Audit SHA
`db8b3134886b62dc31d4d39be5b7eb8015550260fa0359bd92af8e18a8316caa`.
Initial audit CLI omitted PIXELPROOF_DATA_ROOT and failed before opening any payload/contract;
correct external-root environment succeeded. No frozen code/manifest changed.

Separate diagnostic protocol, before any E65 model score: unchanged E43-S head and backbone,
AI cut0.07940196245908739 and REAL cut0.011505939625203613; all83 admitted observations in
original and E49-convention1080px/JPEG75 views (166 views). Global+2texture crops, four frozen
DINO blocks and mean/std aggregation. Network disabled, batch8,20min cap, no training.
Lock full scores/features before descriptive source/camera/CFA/capture/ISO and paired RAW
scene/transport reports. No threshold search, E49 read, AI-retention claim or promotion.
11 focused E65 tests passed: source eligibility, checksums/resume, identity screening, pair
quarantine, transport orientation and complete metric pairing/boundary conventions.


### E65 diagnostic complete (2026-09-13)

All166 original/Q75 views scored with frozen E43 in146.33s; scores locked before metrics.
WIFD REAL false AI11/67 (16.42%) original and13/67 (19.40%) Q75; RawNIND3/16 (18.75%) and
5/16 (31.25%). Transport creates8 WIFD and3 RawNIND errors, rescues6 and1. High-ISO scores
fall in5/8 and rise in3/8 RAW scenes; brightness/development/scene differences prohibit a
noise-causality claim.7D-6 soil/seedling pair visually inspected after score lock; no exclusions.
No new candidate or AI recall measurement; target remains unmet, E43 and serving unchanged.
Whole publishers and feature cache remain consumed diagnostic-only, never TRAIN/fresh final.
697 Python tests pass (11 new); compile/pip/diff checks pass. Source/code/hash details and
next valid TRAIN/DEV acquisition requirements are in `evidence/e65_diagnostic.md`.


## 2026-09-13 overnight continuation — E66 acquisition preregistration

User requests continuous active-session experiments/development and commit/push checkpoints,
explicitly no30-minute monitor or scheduled heartbeat. No automation created; E59 remains
paused. Work proceeds while the live session can execute; interruption/network/app limits
cannot be guaranteed away. AC power confirmed51%. E65 commit2c6524e is present on origin/main.
Its GitHub Python job passed; web lint/typecheck/tests passed, existing critical Next.js audit
failed again (run34722064922), independently of ML. No audit threshold weakened.

Next E66: acquire only official SIDD Small sRGB archive,6,615,978,508B, published
MD5 796971867583bf14677dcae510e52538 and SHA1 5a4aa6aa7abcf7b0b56c88ad578fea9a4ff77935.
MIT stated on official SIDD page. One stream,2h ceiling,30GiB reserve; resume verified ranges;
no executables or official benchmark. Codalab mirror HEAD403, direct official mirror supports
ranges. Keep full publisher out of TRAIN/final; no model scoring before decode/group/overlap
admission.160 noisy/GT pairs are at most10 underlying scenes,5 older phones; processed GT
is not an additional independent photo. Prefer160 noisy sRGB for a grouped DEV candidate.

E60's2,385 unused recorded-group AI candidates remain under audit: GPT Image1 (current source
card CC-BY4.0) and Nano Banana (MIT). Their groups mostly identify individual files, not known
prompts. Plan a hash-ranked80 per family paired with160 noisy SIDD observations only after
provenance, old-role/teacher/current-TRAIN and newer-reserve overlap checks. This may support
a limited balanced DEV screen, never unseen-publisher/independent-prompt or final claims.
Model recipe/feature contract is a subsequent step; no candidate fitted yet.


## E67 feature hypothesis registered before extraction (2026-09-13)

E65 motivates a processing-response branch, not a noise-based REAL veto. Use admitted E54
TRAIN only (11,630 parents,3 conditions,all4,595 AI replay parents). Preserve fixed3x224 crops
and apply Pillow GaussianBlur0.8 to each crop, identically across labels and conditions;
no crop reselection. Frozen DINOv2S blocks2/5/8/11,3-crop mean/std3072D response. This new
input can distinguish processing sensitivity from the original semantic representation.
It does not establish noise causality or guarantee improvement.

Before full extraction, hash-select one parent per source for original-crop E43 replay parity:
score error<=5e-5 and0 binary/selective changes. Freeze code, original TRAIN data/features,
crop index and model bindings first. Verify every crop body even on resume.8 parents per GPU
batch,128-parent immutable chunk receipts,CPU2 threads,4h ceiling,power30% floor unless AC.
No E49/DEV reads, no classifier fit at this stage; SIDD acquisition proceeds independently.
E54 q75 is full-source JPEG75 before2048 cap, NOT E49's1080px social transport. Future DEV
must apply and name the intended serving conditions explicitly.3 crop-transform/identity
tests pass. Candidate fit protocol will separately bind original+response representation,
AI constraints and evaluation rules after the new DEV admission is resolved.


E67 pre-extraction correction: freeze rejected the existing manifest's historical role casing
(7,352 TRAIN +4,278 train), before writing any E67 contract or running model inference.
Normalize case consistently with E60-E64 admission checks; no role reassignment, no frozen
file modified. All11,630 rows remain admitted training, original manifest hash unchanged.


E66 local AI admission protocol frozen before materialization/scoring:80 per GPT Image1/Nano
Banana, seedE66, chosen only from the2,385 E60-unused recorded groups after all E53 old-role
and duplicate exclusions. Explicitly reject E43 FIT/current TRAIN identity/body overlap.
The150,483-body snapshot includes the native candidates themselves: remove only selected
exact self-body entries after those eligibility checks, preserve every other hash/pixel and
all newer MNW/HDR+/FiveK/E65 references. Current TRAIN has100% body coverage in that snapshot.
Any confirmed cross-reference or internal overlap quarantines selected endpoints, with no
replacement draws. Reserve the entire2,385 unused pool from future TRAIN. Local materialization
and source-checksum verification only; no new AI downloads or inference. Unknown prompt
independence and previously seen families prohibit a fresh-final claim. Balanced DEV remains
pending SIDD. Native selection and archive range-resume tests pass; E67's30-source original
crop replay passed with exact0 score error and0 decision changes; blur extraction is running.


### E66 AI audit complete; SIDD group audit registered (2026-09-13)

160 selected local AI originals materialized and verified;160 admitted,0 cross-reference or
internal matches. Report SHA914c5eb4a4db7bea693965cca3e1e0f076dd34986cae71ae47be0ebb1cba51b7.
151,005 references after160 exact candidate self-body snapshot exemptions plus later protected
sources. No AI image downloaded or scored. All2,385 unused candidates now reserved from TRAIN.

SIDD remote ZIP directory required68,753 metadata bytes (no image members extracted),484
entries:320 PNGs,160 scene-instance directories,2 text metadata files and container directories.
160 noisy observations span10 underlying scenes; camera countsIP54,S635,GP33,N622,G416.
This directory is bound before full archive decode. Archive selection unchanged: complete
Small sRGB only, official benchmark untouched. Await whole-archive published checksum/SHA256
receipt, then exact directory validation and160 NOISY PNG decodes; GT stays unextracted.
Cross-reference overlaps or cross-scene internal matches quarantine complete underlying scenes,
including other cameras/settings; same-scene internal similarity stays explicitly dependent.
Only complete160 REAL+160 AI, no shared id/body/cross-label match, can freeze limited DEV.
No replacement draws or model-score-based exclusions.10 new E66/E67 pipeline tests pass.


## E67 candidate recipe fixed in code before any DEV score (2026-09-13)

Retain the E64 original TRAIN PCA64 basis (seed62) and add a second64-dimensional basis of
standardized original-minus-blurred crop features (seed67); whiten both, append intercept,
129 additive-logit coefficients. Existing E43 backbone/scaler/head remain frozen. This tests
additional processing-response information and capacity jointly, not a clean attribution
of effect solely to blur. No PCA rank/seed/regularization sweep is planned.

Use unchanged E64 operating-cut BCE, class/source/parent weighting, hard REAL2x, L2.01 and
SLSQP200/ftol1e-9; preserve correct REAL decision margins and every caught TRAIN AI. Zero-init
and serialized replay must be exact. E67 guard retains E64's10% REAL TRAIN FPR ceiling in ALL3
conditions plus0 new AI misses/0 new REAL errors, successful solver and violation<=1e-8.
Only after E66's complete unscored DEV manifest and blur features are frozen may the separate
fit contract be created and one fit run. DEV labels/scores do not enter PCA/scalers/optimizer.
If TRAIN passes, register one original/1080pxQ75 E66 DEV comparison with all20 absolute/selective
gates and per-image/source AI retention. Only a DEV-passing candidate may receive a separately
registered consumed E49 regression. Neither E66 nor E49 certifies independent final success.
Three new projection/zero-init/serialized-replay tests plus existing decision-constraint tests
pass. Fit has NOT been run; acquisition and TRAIN blur features are still in progress.


### E67 TRAIN feature extraction complete (2026-09-13)

11,630x3x3,072 fixed-crop blur features frozen in1,091.53s,91 checked chunks. Feature SHA
f48474422616c0b3776d6749c0b03796493f8b349a97ce355749011e1b715b4b; contract SHA
51ea9318a9142970f259004e18f2fa0386232403e984543ba07961a68c702237. Thirty source-selected
original-crop parity parents reproduce E43 scores exactly (0 maximum error,0 decisions).
No DEV/final image/score read and no candidate fitted. External arrays remain TRAIN only.
Protocol/code checkpoint f325b7c is pushed and remote main verified; work continues.

E67 DEV runner prepared and four tests pass: blur transform parity, exact pair/source/role
coverage, rejection of offsetting AI loss despite equal pooled recall, and hard denial of
DEV contract creation after a failed TRAIN guard. It reuses original20 numeric gate logic
without invoking E49's population validator or bootstrapping dependent SIDD views. Additional
criteria:0 newly missed AI in each transport/source and non-increased pooled REAL FPR both
conditions. Report all10 SIDD scene groups descriptively. No score-based selection or new
threshold; failed TRAIN candidates remain unscored on DEV. The runner has NOT been executed.


Full Python verification after the E67 DEV runner: 714 tests passed in 12.65 seconds,
with one existing Starlette/httpx deprecation warning. No model-quality claim follows
from software tests. SIDD archive final integrity verification is still in progress.


### E66 SIDD acquisition complete (2026-09-13)

All 6,615,978,508 archive bytes verified against published MD5 and SHA1. Recorded SHA256:
855c375ae20312386cd961e7fdbbeaeb33efdecf0f6f115c01e09f45ec471fdf. No members decoded or
scored at acquisition completion. Starting the already frozen 160-NOISY-image scene and
protected-overlap audit; acquisition alone does not admit DEV. Raw archive stays external.
E67 feature/DEV-runner checkpoint 61f96f85f6039a6e88ee18e05ac37967e4544af7 is pushed,
with identical local HEAD and remote main verified. No fitted candidate yet.


### E66 limited DEV admitted; E67 fit can start (2026-09-13)

All 160 SIDD NOISY images decoded. Against 151,165 protected/reference records: zero
cross-reference matches, zero cross-scene or cross-label matches, no quarantine. The
223 internal matched pairs all stay within their known scene and are explicitly dependent.
Ten underlying REAL scenes across five cameras plus the 160 audited AI observations form
the frozen 320-observation development manifest, SHA bf4c3586126edf91588cc72b375504fa089c3d235abdce7f660686a82c2a6301.
No model scores exist for this population. This limited DEV is not independent final data.
Starting the already preregistered E67 fit: freeze exact artifact/code/DEV identity bindings,
then one constrained TRAIN fit with the unchanged 10% REAL ceiling and per-image AI guard.


## E67 completed: TRAIN guard failed, no DEV/test scoring (2026-09-13)

The one constrained original-plus-blur-response fit converged in 49 iterations / 36.53s;
maximum constraint violation 6.66e-16. Exact serialized replay passed. Zero newly missed
AI views across all 4,595 AI parents and three conditions; zero new REAL errors. REAL
false AI fell from 1,060 to 721 / 7,035 clean (15.07% -> 10.25%), 1,135 to 797 assigned
transport (16.13% -> 11.33%), and 971 to 667 TRAIN q75 (13.80% -> 9.48%). The unchanged
10% ceiling fails in the first two conditions. E67 is rejected at TRAIN; do not run its
DEV scorer, read E49 for it, alter its frozen recipe or promote its candidate. E66 remains
unscored. Original E43 and serving remain unchanged; target not achieved.

TRAIN source diagnosis from the locked fit report: RR real pool contributes 510 / 721
remaining clean errors, 537 / 797 assigned-transport errors, and 468 / 667 q75 errors.
Its own FPR remains 40.80% / 42.96% / 37.44%, versus SCIMD-17 7.06% / 8.06% / 7.06%;
other sources are much lower. RR has 1,250 / 7,035 REAL parents yet contributes roughly
67-71% of remaining errors. This is a TRAIN-only concentration finding, not a licence
to remove RR, relabel errors, tune a source-specific threshold or infer external recall.
Original PCA64 explains 59.63% and response PCA64 40.31% of their respective TRAIN variance.
Next: examine source-weighted objective/constraint diagnostics before choosing a distinct
new hypothesis. No PCA-rank/L2/threshold sweep; preserve all failed E67 artifacts.


## E68 source/condition minimax hypothesis (registered before fit, 2026-09-13)

E67 TRAIN-only diagnostic confirms RR's remaining mean operating-cut BCE is 1.0598
assigned transport, 1.0191 clean and 0.9459 q75, versus the next REAL group's 0.2133.
Yet RR receives only 6.30% of total E67 loss mass across its three conditions. The
worst AI group is GPT Image 1 q75 at BCE 0.2008. These are existing-candidate TRAIN
measurements; the diagnostic fits no candidate and reads no DEV/test rows. Its additive
branch means exclude the shared intercept and are not causal ablations.

E68 tests a distinct training objective motivated by this concentration: minimize one
half the worst REAL group BCE plus one half the worst AI group BCE, plus unchanged
L2 0.01 on the correction coefficients. Groups are ALL pre-existing label/source/condition
combinations, equally weighted parents within each group. No RR-only gate or weighting
parameter search; no source identity is needed at inference. Remove E67's hard-REAL
2x weighting because E68 explicitly optimizes worst-group mean loss. This changes the
objective, not the acceptance criteria. Group-DRO motivation: Sagawa et al., ICLR 2020
(https://arxiv.org/abs/1911.08731); our convex frozen-feature epigraph solver is not a
reproduction of that neural-network training pipeline or a generalization guarantee.

Reuse the exact E67 TRAIN-fitted original/response bases (64+64 and intercept), discard
E67's learned correction weights and initialize all 129 coefficients to zero (exact E43).
Add two scalar epigraph bounds, one per class; each group's BCE must lie below its class
bound. Keep E64's per-image caught-AI/correct-REAL linear decision constraints, including
its margin convention, without change. CPU float64 SLSQP, 200 iterations, ftol 1e-9,
final iterate only, two CPU threads, 30-minute ceiling. No PCA-rank/L2/threshold sweep.

Before fit, test analytic gradients, worst-group epigraph constraints and synthetic hard
AI/REAL retention; bind code, original TRAIN inputs, E67 basis artifact and unscored E66
identity. Accept TRAIN only on successful solver, all constraint violations <=1e-8,
exact serialization, zero new AI misses/REAL errors and <=10% REAL FPR in all three
conditions. Only then separately freeze one E66 DEV comparison at unchanged E43 cuts,
all 20 numeric gates plus per-image/source AI retention and non-increased REAL FPR.
Only a DEV pass permits separately registered consumed E49 regression. No independent
final or deployment claim. E67 remains failed and immutable.


E68 pre-fit verification: all 719 Python tests passed in 12.22 seconds, with the
existing Starlette/httpx warning. No DEV/test scores read. Starting the one frozen
source/condition minimax fit; unchanged TRAIN guard determines subsequent access.


## E68 completed: minimax loss improved; TRAIN acceptance failed (2026-09-13)

One fit converged in 40 iterations / 43.75 seconds. Maximum total constraint violation
1.41e-10; zero newly missed TRAIN AI views and zero new REAL errors, exact saved replay.
Remaining REAL errors: clean 761 / 7,035 (10.82%), assigned transport 817 / 7,035 (11.61%),
q75 719 / 7,035 (10.22%). All three exceed the unchanged 10% ceiling; E68 is TRAIN-stopped
and must not score E66 or E49. E66 remains unscored. No accepted model or serving change.

Worst REAL group BCE fell from 1.4045 at E43 and 1.0598 at E67 to 0.9819; worst AI BCE
fell to 0.1721. RR error counts improved versus E67 (489/508/458 versus 510/537/468),
but other-source rescues decreased, especially SCIMD (149/168/148 errors versus 120/137/120).
The minimax surrogate therefore improved while pooled threshold error worsened. This
rejects this objective recipe; it does not establish that no objective or nonlinear
representation could work. Do not retune group weights, regularization or thresholds.
Next examine a different content/texture representation, keeping the unscored DEV and
all failed recipes intact. E68 preregistration checkpoint cf36565 is pushed; local and
remote main identity verified. All 719 Python tests passed before the fit.


## E69 fixed patch-shuffle representation (registered before extraction, 2026-09-13)

E67/E68 show the original-plus-blur representation still fails the fixed TRAIN error
ceiling, even after a different source-risk objective. This motivates testing texture
information under disrupted global layout, not further tuning the failed objectives.
The existing SFLD reference explicitly uses 28/56/224-pixel patch scales with CLIP and
ten shuffled test views per scale (https://arxiv.org/html/2502.17105v1). E69 is a bounded
DINOv2S adaptation, not a reproduction or an assumed improvement: one fixed 28-pixel
patch permutation (NumPy default_rng seed69) applied to each cached 224-pixel crop.
28 is also aligned to DINO's 14-pixel token grid. Preserve all RGB pixels and all three
fixed crops/conditions; apply the identical permutation to both labels and every crop.
No random test-time views, input-name/metadata dependence, crop reselection or rescaling.
Synthetic patch boundaries are a limitation, despite being identical across labels.

Extract on the complete admitted E54 TRAIN population only: 11,630 parents, three
conditions, 3,072-D frozen DINO features. Same restartable 128-parent chunks, eight-parent
batches, two CPU threads, four-hour cap, AC/battery and disk checks. Recheck a hash-selected
parent from each of 30 sources for exact original-crop replay (score tolerance5e-5 and
zero binary/selective changes). No DEV/test reads or classifier fitting at extraction.

Fix the subsequent candidate recipe now: original E67 PCA64 plus TRAIN-standardized
shuffled-feature PCA64 (seed69, randomized power3), both whitened, one intercept, 129
coefficients. Replace the blur-response branch with shuffled features; retain E67's
class/source/parent-balanced BCE, hard REAL2x, L2.01, SLSQP200/ftol1e-9 and original
per-image E64 constraints. Start all correction weights at zero, preserve the E43 head
and thresholds, require exact serialization. Same <=10% REAL FPR in ALL three TRAIN
conditions, zero new caught-AI misses/REAL errors and successful feasible solver.
No patch-size/permutation/rank/L2/threshold sweep. E66 stays unscored until a passing
TRAIN candidate receives a separately frozen 20-gate DEV comparison. No promotion or
independent final claim; all E67/E68 failures remain immutable.


E69 extraction implementation prepared. Three transform tests pass: whole-patch RGB
identity, deterministic invertibility with no pixel loss or input mutation, and invalid
shape/dtype rejection. The 28-pixel permutation is class/condition/crop independent.
No full extraction, new fit or DEV score has run at this checkpoint.


E69 extraction preregistration verified after 722 Python tests passed in 11.87 seconds
(existing Starlette/httpx warning only). Starting the complete TRAIN-only shuffled-crop
feature pass; the original-model parity check runs before transformed extraction.


E69 original-crop parity passed on all 30 source-selected parents with maximum score
error exactly zero and zero binary/selective changes. Full shuffled extraction continues.
Its prepared candidate code uses standardized shuffled features directly for the second
PCA64 branch (not original-minus-shuffled), retains 129 coefficients and the fixed E67
objective. Four model tests pass, including independent branch inputs, exact zero-init
and serialized replay. No candidate fit contract has yet been created.


E69 fit and conditional DEV runners are prepared before extraction completes. Four
additional DEV tests pass: transform parity with TRAIN, exact two-condition coverage,
source/role integrity, and rejection of a failed TRAIN candidate or offsetting AI losses.
All 730 Python tests passed in 14.21 seconds with the existing warning; compile and
whitespace checks pass. No E69 fit/DEV contract or candidate has been created yet.
Review of prior E54/E55/E58 records confirms backbone adaptation, grayscale augmentation
and scalar threshold repair have already failed their respective gates; do not present
those as untried or restart E59's parked CLIP experiment automatically.


### E69 TRAIN patch-shuffle features complete (2026-09-13)

All 11,630 parents x three conditions x 3,072 features completed in 871.99s
across 91 checked chunks. Original parity: zero maximum score error and zero
decision changes on 30 source-selected parents. Feature SHA fb1028dc15e7c4ef7b85c54253edf9e3a9e04a184a60c4da07e8aa9cbc7a1819;
contract SHA abb66f00bacf05bfd290d3318ec4cf273b2ad8ea6d39341212ba976ecb2b4dc0. No DEV/final rows read and no candidate yet.
Proceed to bind and run the already fixed 129-coefficient E69 recipe.


## E69 complete: fixed patch shuffle rejected at TRAIN (2026-09-13)

Feature extraction finished in 871.99 seconds; the single candidate fit in 38.50 seconds.
Solver converged, exact serialized replay passed, zero newly missed TRAIN AI views and
zero new REAL errors. REAL FPR: 740 / 7,035 clean (10.52%), 828 assigned transport (11.77%),
713 TRAIN q75 (10.14%). All three fail the unchanged <=10% TRAIN ceiling. E69 is rejected;
no E66 DEV or E49 scores may be created for it. E66 remains entirely unscored.

The fixed shuffled-texture branch does not improve pooled threshold errors versus E67's
blur-response branch. This rejects this bounded DINO adaptation, not the published SFLD
CLIP ensemble. Do not sweep patch sizes/permutations/ranks or modify frozen artifacts.
Three distinct frozen-feature recipes have now failed under the same finite decision
constraints. Next investigate whether content and processing response need interactions,
rather than another transform/weighting sweep. Any new candidate remains TRAIN-only
until it passes the same guard. E43 and serving are unchanged; target not achieved.


## E70 original/blur interaction hypothesis (registered before fit, 2026-09-13)

E67, E68 and E69 failed fixed TRAIN acceptance. Test a distinct nonlinear feature map
using the already frozen E67 original64 and blur-response64 coordinates: bilinear
interactions let the correction depend jointly on image representation and processing
sensitivity. This hypothesis is not established by the preceding failures. The larger
parameter count is a joint capacity change, not a clean attribution solely to interactions.
E63 already tested RBF features of the original representation and failed external AI
retention; E70 is not a repeat of that original-only kernel or a kernel-width sweep.

Use two independent CountSketch hash/sign arrays from NumPy default_rng(seed70), each
64->128, and the real inverse FFT of their FFT product to sketch all original-by-response
products into 128 coordinates. This is an explicit bilinear TensorSketch; verify exact
agreement with direct hashed outer products on fixtures. Method reference: Pham/Pagh,
KDD 2013, https://www.rasmuspagh.net/papers/tensorsketch.pdf . Standardize sketch coordinates
using all and only admitted TRAIN views. Keep the original 128 E67 coordinates, append
128 standardized interactions and one intercept: 257 trainable correction coefficients.
No new pixels, backbone, PCA fit, patch transform, or hash/dimension/seed search.

Discard E67's correction weights; zero-init all 257 (exact E43). Fit one final-iterate
CPU float64 SLSQP200/ftol1e-9 recipe, two threads, 30-minute ceiling. Use unchanged E67
class/source/parent-balanced operating-cut BCE with hard REAL2x and L2.01 on all correction
coefficients. Retain exact E64 caught-AI/correct-REAL linear decision constraints and
margin conventions. No L2, weight, threshold or post-fit scale sweep. Require solver
success, violation<=1e-8, exact serialized replay, zero newly missed TRAIN AI/REAL errors
and REAL FPR<=10% in ALL three conditions before any E66 DEV access.

Bind every source feature/basis/code hash, the failed E69 report and unscored E66 identity
before fitting. Only a passing TRAIN candidate permits a separately registered E66 DEV
comparison at unchanged cuts, all20 numeric gates plus per-image/source AI retention and
non-increased REAL FPR. Only then may a consumed E49 regression be separately registered.
No independent final or serving promotion; E67/E68/E69 remain failed and immutable.


E70 implementation verification: five new tests pass, including direct outer-product
versus FFT equality, bilinearity/input immutability, exact zero-init with discarded prior
weights, serialized replay/batch partition stability, and invalid input rejection. All
735 Python tests passed in 12.29 seconds (existing warning), compile/diff checks pass.
The prior E69 result checkpoint 486f826 has identical verified local/remote identity.
Now freeze the E70 fit contract and run its single preregistered TRAIN fit.


## E70 TRAIN guard passed; register first E66 DEV comparison (2026-09-13)

The single bilinear-map fit completed in 88.82 seconds, successful feasible solver and
exact saved replay. REAL errors are 568 / 7,035 clean (8.07%), 615 assigned transport
(8.74%), 513 TRAIN q75 (7.29%). All three meet the unchanged <=10% ceiling. Zero newly
missed TRAIN AI views and zero new REAL errors; no protected AI is traded away. This
passes TRAIN feasibility only, not external recall or the target benchmark.

Now separately freeze the first E66 comparison: all 320 admitted observations in publisher
original and 1080px/JPEG75 views, unchanged E43 cuts, original+fixed GaussianBlur0.8 crops
with the frozen E70 interaction map. All 640 scores are locked before metrics. Same 20
numeric gates, zero newly missed E43-caught AI per source/condition, non-increased pooled
REAL FPR both conditions. Report all ten dependent SIDD scenes descriptively; no independent
view confidence intervals or final claim. No threshold/map/weight adaptation after scores.
Only a passing DEV screen permits separately registered consumed E49 regression.
E59 remains parked; the old feature/training plan is not needed for this accepted TRAIN
candidate. No serving change and no DEV score yet at this registration entry.


E70 DEV runner verification: four new tests pass (TRAIN/DEV blur identity, complete
paired/source/role coverage, no offsetting AI loss, failed-TRAIN access denial). Full
Python suite: 739 passed in 11.88 seconds with the existing warning. Whitespace checks
pass. Checkpoint 1f3bc59 is pushed and remote identity verified. Freeze the separate
DEV contract now; no E66 model score exists yet.


### First E66 DEV score stream locked (2026-09-13)

E70 completed all 640 paired DEV scores in 274.49 seconds. Full score SHA
9cf625b6f2123c5a6bd421f37d6415030cc386ccdc7cd8b809a6dba36b879521. Scores are now locked before the first metric report. E66's
admission manifest is an immutable historical unscored snapshot; the population has
NOW been exposed to model scoring and must be treated as consumed DEVELOPMENT in
future plans. Never infer current freshness from its original model_scores_created=0
field. No fitting, data exclusion, threshold change or E49 access took place in this step.


## E70 rejected on first E66 DEV comparison (2026-09-13)

All 640 scores locked in 274.49 seconds before metrics; unchanged candidate and cuts.
Original REAL FPR worsens from 68/160 (42.50%) to 75/160 (46.88%): eight rescues and
15 new errors. Social-Q75 REAL FPR improves from 68/160 to 49/160 (30.63%): 19 rescues,
zero new errors. AI recall declines from 156/160 (97.50%) to 153/160 (95.63%) original,
and from 154/160 (96.25%) to 152/160 (95.00%) social Q75. Paired AI losses are four
original and three Q75 views, partially offset by one rescue in each condition; both
GPT Image 1 and Nano Banana lose previously caught examples. Do not mask losses by net
counts. Original AUC .96668 -> .95379; Q75 .94430 -> .95031. Both reference and candidate
pass only 12/20 numeric gates on this limited DEV. E70 fails all additional original
checks and Q75 AI retention, so no E49 regression is permitted and no promotion occurs.

The TRAIN pass does not generalize to the new SIDD rendering domain or every new AI
observation. Its ten scene groups and unknown AI prompt relationships remain explicit
limitations. Preserve E70 candidate, full scores and failed report. E66 is now consumed
DEVELOPMENT, never TRAIN or a fresh final; its admission snapshot stays immutable. No
post-score threshold, interaction-map or source-specific route adjustment is allowed.
Next assess a complementary pretrained representation with the same E43-preserving
TRAIN/DEV rules, reusing eligible cached work where possible. Target remains unmet.


## E71 complementary CLIP representation and cached-feature reuse (2026-09-13)

E70's TRAIN-to-DEV failure motivates a different pretrained representation rather than
a sweep of DINO transforms or thresholds. The current active-session overnight request
authorizes continued experiments and needed acquisition. Reuse eligible unfinished work
through a new E71 contract; E59's original nine fits, queued handoff and automation remain
paused. Do not modify E59's frozen contract/code/chunks or issue its training commands.
This updates the next-work decision after the observed E70 failure, not a silent automatic
restart of the office-paused experiment.

Inventory: exactly 9,599 numeric E59 parent chunks, 399,641,982 bytes; 2,031 parents missing
from the same 11,630 admitted E54 TRAIN population. Files prefixed `._` are macOS sidecars,
not extra examples (an initial broad glob counted them; numeric-name audit corrected this
before any contract). No completed E59 feature archive or receipt exists. No experiment
worker remains active. Cached CLIP ViT-L/14 weights are already local and hash-pinned; no
weight/image download is needed. CLIP itself and its official detector were investigated
previously; the new hypothesis is a constrained complement to E43 using our TRAIN features.

E71 feature stage: verify all legacy bound inputs, every cached chunk's parent/binding,
raw/aggregate array hashes and exact aggregation, plus the associated cached crop body.
Freeze a hashed external inventory and new runner/CLIP code identities before new extraction.
Use a safe standalone adapter, avoiding historical acquisition-module import side effects.
Re-encode one cached parent per represented source; require raw-vector agreement<=1e-5
with historical chunks. Only then compute the 2,031 missing parents into owned E71 chunks,
with identical official CLIP preprocessing, frozen float32 eval, three-crop batches and
three conditions. Retain raw768 crop vectors and mean/std1536 aggregates, no L2 normalization.
Bind all 11,630 parents in original order; no subsampling, missing-AI omission or DEV input.
Use two CPU threads, AC and20GiB reserve, bounded15min inventory /60min extraction, immutable
chunk receipts and verified resume. Old E59 outputs remain untouched; no automatic model fit.

Fix the E71 candidate before features complete: retain E67's original TRAIN PCA64 basis,
add TRAIN-standardized CLIP1536 PCA64 (seed71, randomized power3), whiten both, one intercept,
129 correction coefficients initialized to zero (exact E43). Same E67 class/source/parent
BCE with hard REAL2x, class mass.5 each, L2.01, SLSQP200/ftol1e-9, two CPU threads/30min;
same E64 per-image caught-AI/correct-REAL margins, solver feasibility and exact serialization.
Require <=10% REAL FPR ALL3 TRAIN conditions and zero new AI misses/REAL errors. No rank,
seed, mixture, weight, L2 or threshold sweep; no warm-started source-fold independence claim.

Only a TRAIN-passing candidate may receive one separately registered E66 comparison. E66
is now CONSUMED DEVELOPMENT, not unscored/fresh: bind E70's exposure report and score stream,
keep it excluded from TRAIN/final and preserve its initial admission snapshot. Retain all20
numeric DEV gates, zero new E43-caught AI per source/condition and non-increased REAL FPR.
No E49 regression until that screen passes; no independent final or serving promotion.

E71 engineering checkpoint: complete9,599-parent legacy inventory/crop bodies verified and
feature contract frozen,30 source parity parents registered,2,031 parents left to encode.
Extraction started, no completion claim. Six feature tests and745-test suite passed;
four model tests additionally passed. Batch probabilities allow1e-7 for unchanged float32
reference rounding; coordinate batch parity1e-14 and same-batch serialization exact.
New fit/DEV code remains unfrozen and unexecuted; DEV explicitly consumed by E70.

E71 encoder parity result:30/30 cached source representatives, raw maximum error0.0.
Missing-parent completion active. Conditional consumed-DEV runner prepared with historical
pixel identity and <=1e-6 E43 score replay plus zero changes at either frozen decision cut;
TRAIN/DEV CLIP aggregation and failed-TRAIN denial are tested. No DEV scoring yet.

E71 code checkpoint validation complete:752 Python tests passed in13.11s, including13
new E71 tests; compile/diff checks passed. No candidate score exists at this checkpoint.

E70 post-fit margin diagnostic registered in plan while E71 features run: fixed TRAIN and
already-consumed DEV score margins only, quantiles0/1/5/25/50/75/95/99/100%, near-boundary
counts1e-6/.01/.1/.5. Keep every previously caught AI, disclose existing new DEV misses;
no new fit/view/threshold search or E49/E71 score access. TRAIN margin erosion, if observed,
is descriptive and does not alone explain generalization.

E70 margin diagnostic complete:24/40/45 previously caught TRAIN AI views sit within1e-6
logit units of the fixed cut; zero training misses. Lower1% margins before→after are
0.88384→0.37102 clean,0.83336→0.05094 assigned transport,0.50470→0.00605 Q75.
The109 boundary views are not109 independent parents. Existing consumed DEV misses4/3
include original E43 margins up to1.5608/1.1134, so this is not merely float rounding.
Potential next mechanism: protect confidence away from the boundary, subject to unchanged
REAL/AI gates; not implemented or selected here, no causal guarantee. E71 unchanged.

Margin diagnostic code verification:754 Python tests passed in14.82s, compile/diff clean.

SIDL data lead metadata audit only:1605 DNG records/253 filename scenes, all iPhone12Pro;
release count/split reconciliation pending. This changes no model/data role, no images scored.

Literature/data refresh during E71: PiD and MPFT recorded as distinct unimplemented methods;
QuAD already E46. MIDD release availability found, packaging/terms not admitted; no image
acquisition or model inference from these sources. Existing E71 extraction continues unchanged.

Current-status documentation refreshed for E71 feature completion; historical E70 and older
results remain unchanged. This is a documentation update, not a new experiment outcome.

MIDD bounded expansion metadata plan is separate from E71: four vendor/sensor packages,
TRAIN-original members only after per-archive license/identity and scene/overlap audit.
ISOCELL_3P9 directory/license inspected successfully; image members0. No new candidate fit.


E71 CLIP TRAIN extraction completed:11,630 parents x3 conditions x1536 features;
9,599 verified legacy chunks plus2,031 new E71 chunks,1538.79s. Thirty source-selected
raw encoder replays matched exactly (max error0). Feature SHA
`dc0c4ab80d3be56495a6c5f19ee2c5a4c63a7f0e9eacf97856d7f529fb5a0603`;
contract SHA`9ffd471967c6b85cdb618ed6714e21844420d4c4b30bd093e295e5ff7ccf26d5`.
No DEV/final feature rows read, classifier scores0, downloads0, E59 outputs unchanged.
Fit contract is now frozen and the single preregistered fit is running. No quality result
or DEV permission is implied by feature completion. Full arrays remain external under e71.


### E71 single frozen TRAIN fit passed (2026-09-13)

28.22s, solver success, zero newly missed AI views and zero new REAL errors across all3
TRAIN conditions. REAL false AI: clean492/7035=6.9936%, assigned transport546/7035=7.7612%,
Q75 480/7035=6.8230%, versus E43 15.0675%/16.1336%/13.8024%. All3 <=10% gates passed.
Candidate SHA`f3e15446623b533c7519def877f855cc13d5e8a0175f43eb6c2daa36e855ef5b`, contract SHA`72c381457d619fd9331023d5cd91bc9f9fcc696aae9d30662cff72aa9954d088`.
This is constrained TRAIN feasibility, not external retention. Proceed to the separately
registered one640-view consumed E66 comparison, with all20 numeric gates and zero newly
missed AI per condition/source. E70 already consumed this set; never call it fresh/final.
No E49 access or promotion from this TRAIN result. Fixed recipe, no parameter sweep.


MIDD four-vendor metadata verification completed: Hynix_SL846868, ISOCELL_3P9842,
Sony_IMX258485 and OmniVision_OV32A763 official TRAIN originals (2,958 total).
Exact archive identities and directory digests are in `evidence/midd_catalog_lead.json`;
all four embedded LICENSE.txt bodies match CC BY-NC-SA4.0. Total directory/license
range transfer874,873B; image members read0. Separate official test and denoised paths
are excluded from any proposed TRAIN acquisition. Next: bounded score-blind original
selection, per-member integrity/resume and scene/overlap audit before TRAIN admission.


### E72 bounded MIDD research TRAIN acquisition plan (2026-09-13)

Freeze128 SHA256(E72|filename)-ranked publisher-original TRAIN JPEGs per preselected
Hynix_SL846/ISOCELL_3P9/Sony_IMX258/OmniVision_OV32A sensor:512 members,
3,821,581,226 uncompressed original bytes. No reserve/refill, official test or denoised
partners. Selection uses only filenames/metadata; no detector scores or content filtering.
Two workers,90min execution budget, AC and20GiB free reserve. Pinned archive ETags,
If-Match exact206 ranges, member ZIP CRC/length and SHA256 receipts, verified member resume.
Bulk fallback rejected before reading its body. Code's8 focused tests passed, including
wrong range/ETag, truncation, excess transfer, entry identity and test/denoised/path exclusion.
All images initially QUARANTINE; separately freeze canonical decode/overlap audit including
E52/later reserves/E65 and consumed E66 before TRAIN admission. Unknown filename scene links
are not independence; whole MIDD publisher excluded from DEV/fresh final. Research license
CC BY-NC-SA4.0, no automatic serving eligibility. E71 recipe/population stays frozen.

E72 acquisition contract frozen (`4bec68d8…b1fd0f1`), bounded download active.
Full Python suite762passed in14.02s; only existing Starlette/httpx deprecation.


### E71 rejected on consumed E66 DEVELOPMENT (2026-09-13)

640 scores locked in412.36s; historical E43 pixel/score/decision replay is exact (max score
error0). Original REAL false AI68/160=42.50% ->55/160=34.375%; social Q75 68/160 ->61/160=38.125%.
Original AI156/160=97.50% and Q75 154/160=96.25% remain equal in aggregate, but original loses
2 previously caught GPT images while rescuing2; Q75 loses1 GPT while rescuing1 Nano Banana.
Thus aggregate recall masks per-image regression. Both conditions fail zero-new-AI-miss and
absolute gates (12/20 total). No E49 regression, no serving promotion. Candidate rejected.
Scores SHA`971b6e130b51f8ef741e4a194782e4b9c02e6ba18b68582db82f8eea07be8db0`. E66 remains consumed DEV; target unmet.

Next isolated mechanism E73: retain the exact frozen E71 original64/CLIP64 coordinates,
discard E71 weights, and replace caught-AI decision-only protection with non-decreasing
reference logits for ALL13,785 AI TRAIN views, including currently missed AI. Keep E64
correct-REAL decision constraints, objective/class-source-parent weights, L2.01, zero start,
SLSQP200/ftol1e-9, fixed cuts and all3 REAL FPR<=10% guards. No fractional margin/penalty/rank
sweep. One fit on unchanged E54 TRAIN; MIDD is a separate acquisition. Only complete TRAIN
pass permits a separately registered consumed DEV comparison. No external AI-retention claim
from TRAIN constraints; E70/E71 failures motivate testing confidence erosion explicitly.


E72 admission code prepared: canonical whole-body/RGB plus dHash/pHash overlap against
all frozen references and explicit320-row consumed E66; verify all2,385 reserved AI bodies
and current TRAIN are covered by protected snapshot. Internal similarity and exact same-sensor
EXIF capture-second links form transitive components. Propagate protected matches through the
whole component; keep one deterministic representative, no replacement. Scene independence
remains unverified. Three focused transitivity/time-link tests passed; full suite765passed.
Audit freeze waits for complete acquisition receipt. E71 DEV failure evidence is now checkpointed.


E73 confidence-preservation code prepared and4 focused tests passed: caught and missed AI
both constrained; a feasible toy fit reduces REAL errors while keeping AI logits; protected
REAL handling and exact copied-basis/zero-weight reset verified. Freeze and run one fixed fit.
E72 first download execution hit a transient Hynix connection timeout after verified members;
ISOCELL/Sony128 each completed while the worker pool settled. Resume the identical frozen
selection/code using existing CRC/SHA receipts; no quota or sensor substitution.

E73 contract frozen; one full-confidence TRAIN fit active. Full suite769passed/12.32s,
with the same existing deprecation warning. No MIDD or DEV data enters this fit.


### E73 confidence constraint: TRAIN failed (2026-09-13)

Single27.70s fit,23 SLSQP iterations, solver success, max violation5.13e-16; all13,785 AI
view logits preserved within numerical tolerance (minimum shift-4.72e-16), zero new AI/REAL
classification errors. REAL FPR14.5700%/15.8351%/13.3475%, versus reference15.0675%/16.1336%/
13.8024%: all3 fail10%. Candidate SHA`147551945c7fdb9843cd93c4262a8b07381f15f738b897412fb9cfd2354f88b9`. No DEV/E49 scores permitted.
This shows the fixed linear correction under the stronger constraint achieved little REAL
improvement; it does not prove global infeasibility or external retention. Do not loosen
confidence protection or sweep its fraction. Preserve the failed fit.

Next E74 isolates representational capacity: exact E71 original64/CLIP64 coordinates plus
one128D bilinear TensorSketch (same tested E70 hash/sign construction, seed70), TRAIN-standardized,
257 zero-initialized coefficients. E73 full-AI-logit constraints, E64 objective/L2.01/REAL guards,
SLSQP200/ftol1e-9 and fixed cuts unchanged; same E54 population, no MIDD yet. One fixed fit,
no sketch-size/seed/rank sweep. E70 tested original/blur interactions with weaker constraints;
this is original/CLIP interaction under full-confidence protection, not a rerun of E70.
Only all TRAIN guards permit a separately registered consumed E66 comparison.

E74 bilinear model implementation prepared;4 focused tests pass for exact E71 basis preservation,
TRAIN interaction normalization, no old-weight leakage, serialized/batch replay and invalid maps.
Freeze the fixed257-coefficient map before one fit; E73 constraints are imported unchanged.

E74 protocol frozen and its single fit active. Full Python suite773passed/12.53s;
no changed cuts, confidence constraints, data population or failed-experiment reruns.


### E74 nonlinear confidence-preserving fit failed TRAIN (2026-09-13)

One104.39s/31-iteration fit, solver success, max violation9.44e-16, all13,785 AI logits
preserved within tolerance; zero new AI/REAL decisions. REAL FPR13.2907%/14.3568%/12.1393%
improves on E73 but fails all3 <=10% guards. Candidate SHA`cadf88f4fc3e3030495bc9cbdc0c74a6d2ffb205091fcebaac063dbf731eb05a`.
No DEV/E49 access. Do not sweep sketch rank/seed or weaken confidence constraints.

E72 acquisition completed512 originals/3,821,581,226B with pinned ETags, CRCs and body SHAs.
Second execution resumed after Hynix connection timeout; its1,781,971,093 range bytes are
only that execution's transfer, not total across both attempts. All four sensors128 each;
no test or denoised image extracted. Receipt SHA`1b7e28056104d4b8ef2a913448a09b6026f01c67b141c3f1a5a12dae76cb12a2`.
Still quarantine, no classifier scores. Freeze and run the prepared protected-role/component audit.

Next E75 prepares features only for admitted MIDD TRAIN representatives. Copy exact E54
source-resolution JPEG75-before2048-cap and existing global+2texture224 crops; frozen DINOv2S
3072D and CLIPViT-L/141536D, same3 conditions. First replay the30 E67-registered old TRAIN
source representatives against old DINO reference scores and E71 CLIP features, requiring
reference error<=5e-5, no decision changes at either cut, CLIP error<=1e-5. No new MIDD
classifier scores, no DEV/final pixels, no old-feature modifications or automatic model fit.
Feature contract waits for complete audited admission; separately decide the next fit.

E75 prepared feature code passed3 focused tests: pixel-identical E54 transport/crop replay
for all4 assignments on an EXIF-rotated2177px source, role/duplicate denial and source-bound
chunk corruption detection. Historical function tested by extracting only its pure AST,
without importing old acquisition code. E72 audit is active; no E75 contract/encoding yet.


### E72 score-blind MIDD TRAIN admission complete (2026-09-13)

All512 originals decoded;151,485 protected references, zero cross-reference matches,
zero decode failures. One internal perceptual pair formed a component; retain one hash-ranked
representative,511 admitted (Hynix127, other3 sensors128). No refill. Native publisher
JPEG dimensions:256 at3264x2448,128 at2320x1744,128 at4208x3120. EXIF make/model and capture
timestamps absent in all512, so sensor provenance comes from publisher packaging and scene
independence remains unverified. Whole MIDD publisher is research TRAIN only, never fresh
DEV/final; no detector scores yet. Audit SHA`23021f21e0ff5fcb72c972c8293a0c18188ddf18cdf8fdfb33ec3046ac5d8e04`;
TRAIN manifest SHA`a258b4362e45016762ce1d557cb2ac87f82fe29f78d598559caf216b551e0d8e`.
E75 feature contract may now freeze for all511 admitted representatives. Existing E54 TRAIN
remains immutable; a future expanded fit would have12,141 parents /7,546 REAL /4,595 AI.

E75 feature contract frozen for511 audited MIDD TRAIN images; old-source DINO/CLIP
parity and extraction now active. Full Python suite776passed/13.33s. E72 complete
acquisition/admission and E74 failed-fit evidence are being committed together.


### E76 isolated MIDD TRAIN expansion planned before new image scores (2026-09-13)

After complete E75 features, one fit expands unchanged E54 TRAIN with all511 admitted MIDD
representatives (12,141 parents;7,546 REAL/4,595 AI). Keep the exact E74 original64/CLIP64/
128-interaction map and its old-TRAIN scaling, discard correction weights; no map refit.
This isolates data contribution from capacity changes. Same E73 full-AI-logit constraints,
E64 correct-REAL guards/BCE/class-source-parent weighting/hard REAL2x/L2.01 and SLSQP200.
Zero257 coefficients, same cuts, no mixture/penalty/rank/sensor weighting sweep.
All old AI retained. Require solver/serialization/decision guards, old-TRAIN and expanded
pooled REAL FPR<=10% in all3 conditions. Also require new-MIDD pooled REAL FPR<=10% and
worst admitted MIDD sensor<=20% in each condition, preventing old/new population dilution.
No DEV/E49 after any failure; TRAIN pass permits a separately frozen consumed DEV screen.
No confidence/decision protection relaxation. Features currently active; no fit contract yet.


E75 historical encoder parity passed30/30 TRAIN source representatives in all3 conditions:
E43 maximum score error0, CLIP maximum feature error0, both decision cuts unchanged.
New511-parent feature extraction active. E76 prepared fit code passed3 focused tests:
new-camera failures cannot hide in large old populations, poor individual sensors cannot
hide in pooled camera results, and full AI/unique-body/role expansion checks fail closed.
Read-only metadata join verified12,141 unique bodies/parents, all4,595 old AI retained.
No new MIDD model score, E76 contract or fit yet.

E76 prepared implementation verification complete:779 Python tests passed/14.92s.
Commit its fixed protocol/code before E75 feature completion; do not freeze inputs or fit early.


### E75 MIDD TRAIN features complete (2026-09-13)

All511 admitted parents x3 conditions completed: DINO3072 and CLIP1536,778.72s.
Thirty historical source-parent replays in all3 conditions had exactly zero E43 score
and CLIP feature error, zero decisions changed. New MIDD classifier scores0, DEV/final
rows read0, downloads0. Feature SHA`99de60c680acefea301fb43c468d71b43912b025411da57067a1218e07ae28cb`;
contract SHA`d6bea64e830b758cd8f44a06a4b147b43fda647d40049778d8280dda16624e3e`.
Full arrays/511 immutable source-bound chunks stay external under e75. Proceed to freeze
and run the already prepared single E76 data-expansion fit, preserving all AI and every gate.

E76 expanded TRAIN fit contract frozen for12,141 parents. The single fixed-map fit
is active; old and new REAL slices have separate gates. No thermal/performance warning
recorded, AC80%. E75 completion evidence checkpointed; no detector improvement claimed yet.


### E76 isolated camera-data expansion failed TRAIN (2026-09-13)

One126.62s/39-iteration fit; solver success/violation8.33e-16. All13,785 AI logits preserved;
zero new AI misses/REAL errors. Newly caught AI11/28/19 views across the three conditions.
Old REAL FPR13.2765%/14.3710%/12.2530%; expanded12.6425%/13.6496%/11.6221%: all fail10%.
New MIDD REAL FPR3.9139%/3.7182%/2.9354% passes, versus reference7.4364%/5.0881%/4.5010%;
all new sensor gates pass. Adding cameras did not solve old difficult REAL examples under
this fixed map/constraint. No DEV/E49 permission. Candidate SHA`dd6fd99607f674cafa3a4f9982002ce2c688d564ebd4b94bcd57e4ac3106a117`.

### E77 REAL-only feature manifold adaptation planned (2026-09-13)

Motivated by E76 data-only failure and the primary Attribution Consistency paper, learn
an additional nonlinear REAL feature residual instead of sweeping existing maps. This is
our explicit adaptation, not paper reproduction (paper architecture/epochs not fully specified).
Use all12,141 TRAIN parents, old and MIDD; retain all4,595 AI for downstream correction.
REAL-only CLIP1536 StandardScaler weighted by source/parent, followed by a1536-256-64-256-1536
ReLU autoencoder (linear output), initialized with seed77. Minimize weighted per-feature L1
on all22,638 REAL views, Adam2e-4/no weight decay, batch256,100 fixed epochs, no early stop,
no DEV selection. Freeze a deterministic per-epoch order; bounded AC execution and bound
optimizer/model checkpoints for resume. Opposite-class examples never enter autoencoder
normalization or loss. No pixel/backbone training or downloads.
Afterward freeze the autoencoder; absolute standardized-feature residual1536 -> unweighted
TRAIN StandardScaler/PCA64(seed77,randomized power3), whiten. Append these64 coordinates to
exact E74 original64/CLIP64/bilinear128 map, yielding321 zero-initialized correction weights.
Same E73 full-AI-logit and E64 correct-REAL constraints, same BCE/source-parent balancing/
hard REAL2x/L2.01/SLSQP200. All E76 old/expanded/new-MIDD/sensor guards retained. Separate
representation and fit contracts, exactly one trained AE and one fit, no capacity/epoch/seed
sweep. Full TRAIN guards alone permit one separately frozen consumed DEV comparison.


### E77 representation engineering verification (2026-09-13)

Implemented the fixed REAL-only weighted CLIP autoencoder plan below. Five focused tests
verify AI exclusion from normalization, equal REAL-source weight mass, deterministic complete
epoch permutations, exact serialized residual replay with bounded batch variation, checkpoint
contract/weight/epoch binding, and exact CPU Adam continuation. Checkpoints include optimizer
state every10 epochs; no checkpoint selection. Data remain TRAIN-only; no detector result yet.


### E77 REAL-only representation complete; constrained head prepared (2026-09-13)

The fixed100-epoch AE finished in40.96s on MPS. Weighted REAL L1 fell
0.79261052→0.54065642; no AI views entered its normalization/loss. The frozen
all-TRAIN residual PCA64 explains18.035% variance; this is descriptive, not a rank-selection
criterion. Exact serialized residual and coordinate replay passed. Manifold SHA256
`21de374098fb71b5b691eb1010d35aed33cbfc6d7192416ec738719a53a33c2b`;
features `[12141,3,64]` SHA256
`aef74e52d4d0f14f49d94905e86ee20ffb472f0e0da75341813633aa351a40dc`.

Prepared the single321-coefficient fit with unchanged full-AI-logit and correct-REAL guards,
objective and all E76 population gates. Three new model tests verify exact preservation of
old coordinates, no old correction-weight leakage, serialized/batched predictions and
invalid alignment rejection. All10 focused E77/E76/E73 model/guard tests pass. No DEV or E49
read, no detector-quality claim from reconstruction loss.


### E77 constrained reconstruction residual: TRAIN FAILED (2026-09-13)

One fixed fit completed in176.22s/42 iterations; solver success, max violation9.99e-16,
minimum AI shift-9.99e-16. All13,785 AI views protected; zero newly missed AI and
zero newly wrong REAL views. Old-REAL FPR10.9595%/12.3810%/10.3909% fails all three
10% guards. Expanded FPR10.2968%/11.6486%/9.7535%; new MIDD1.1742%/1.5656%/.9785%.
The pooled Q75 pass cannot hide the old-REAL failure. This improves E76 but is rejected;
no DEV/E49 read or promotion. Candidate SHA256 `f4be42e1cb0a7582b38f363cb66bfa043a5741dfae875651e6d1d587558f4b5d`.
No AE epoch/rank/architecture sweep or threshold repair follows this result.

Next preparation: E78 acquires only the pinned94,372,114-byte official DEAR-r checkpoint
for research. Its aligned-pair forensic backbone offers a different signal from the frozen
semantic DINO/CLIP coordinates. Selection precedes all local DEAR image scores; no c/r sweep.
Code/weight terms and inference source are reviewed. Strict exact size/SHA, bounded streaming
and no third-party execution in acquisition. Three downloader tests pass; synthetic
inference/parity/resource registration must precede any execution. No new image download.


### E78 acquisition complete; synthetic inference probe prepared (2026-09-13)

Official DEAR-r94,372,114 bytes downloaded in13.96s; pinned SHA256
`430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97` matches.
No image bytes or scores, no third-party execution during acquisition. Local strict
torchvision equivalent uses stride0=1 and fixed820/2048 binary channel gate; weights-only
load requires every state entry. Three focused tests pass: exact official normalization,
missing tensors/invalid gates refused and digest failure before deserialization.

Next fixed synthetic probe compares reviewed author ResNet source and local implementation
on96/113x157/224/383x511 synthetic RGB, plus CPU/MPS parity at224. Numerical and
resource gates are registered before execution; projected full extraction must be<=7200s
and MPS driver memory<=6GiB. No real image or detector performance is measured here.


### E78 synthetic parity passed; batch3 throughput guard FAILED (2026-09-13)

All author CPU feature outputs equal exactly; max logit difference5.96e-8. MPS
feature/logit max differences2.98e-7/7.75e-7, zero sign changes. Driver allocation1.22GB.
However .06970s/crop projects7,615.60s for full TRAIN, exceeding the predeclared7,200s
limit. No image extraction permitted by that probe. Preserve failure receipt.

Engineering-only follow-up: one batch9 synthetic probe, same nine crops/parent, normalization,
float32, model/gate and numeric/resource limits. Compare CPU/same-device batch3 vs batch9
first, then time full encode including transfers. This is a throughput repair, not a detector
recipe or quality threshold change. No image scores used to choose batch size.


### E78 batch9 unchanged numerics, no speed gain; E79 explicit resource revision (2026-09-13)

Batch9 vs MPS batch3 features/logits are exactly equal; CPU differences3.58e-7/1.31e-6,
zero sign changes. Driver allocation3.36GB. .06984s/crop projects7,631.24s; the original
7,200s throughput guard still FAILS. Both failed resource probes remain immutable.

User explicitly authorized sustained overnight work. Before any DEAR image feature/scoring,
E79 allocates9,000s wall time, retaining all numeric and6GiB memory gates. This is a recorded
resource-budget revision based only on synthetic throughput, not a changed quality threshold
or an E78 pass. No additional batch/precision/model sweep.

E79 prepares all12,141 admitted TRAIN parents, all3 conditions and global+2texture224 crops,
fixed DEAR-r820 active channels and per-condition mean/std1640. Existing verified E54
crops and exact E75 MIDD pixels/transports; no new image bytes. The classifier head is not
executed. Immutable per-parent chunks support resuming without excluding any failures.
One source representative per source will be replayed under the same batch layout.
Three tests verify view/crop ordering and population statistics, chunk body/parent/source
binding, and feature-only encoding without classifier execution. No DEV/E49 access.


### E79 active; E77 frozen TRAIN operating-point diagnostic planned (2026-09-13)

E79 all-parent feature contract is frozen and extraction active, observed MPS driver2.29GB.
While it runs, audit the already failed E77 candidate at both unchanged cuts on TRAIN only.
Apply the pure generic E49 metric function to old/expanded TRAIN, including selective
coverage/accuracy and worst-source errors; do not read E49 scores/pixels or run its final
validator/bootstrap. Verify FPR replay matches the frozen fit receipt before reporting.
No fit, threshold selection or image reading. Existing source transition counts show RR
remaining false positives548/602/508 of1250 across the three conditions; no row is removed.


### E77 full TRAIN diagnostic complete; E80 head prepared (2026-09-13)

Frozen E77 candidate replay exactly matches its FPR receipt. Old TRAIN passes7/10 fixed
numeric checks per condition; expanded TRAIN7/7/8. Automatic coverage92.3–93.2% and
uncertainty6.8–7.7% are adequate, but covered accuracy92.05–93.30% fails95%; RR
source false AI43.84%/48.16%/40.64% fails20%. No new fit/cut, protected data or image
file read. Pure generic metric reuse is not an E49 measurement. Compact evidence:
`evidence/e77_train_operating_point.json`.

Prepared E80, not yet frozen/fitted: exact E77 original64/CLIP64/bilinear128/REAL-residual64
plus new DEAR64 (all-TRAIN StandardScaler, randomized PCA seed80/power3, whiten).
385 zero-initialized coefficients; no previous correction-weight leakage. Unchanged
E64 objective and E73 all-AI logit constraints, all E76 population guards. Additionally
require all10 fixed numeric checks on both old and expanded TRAIN for all3 conditions
before DEV. This strengthens screening to expose source/covered-accuracy failures before
spending any DEV data. No rank/seed/cut/weight sweep. Freeze only after complete E79 receipt.

Four new tests pass: exact old-coordinate/teacher replay, orthonormal new basis, serialized
and batch score parity, hidden small-source failures and excessive uncertainty rejected.
An initial test wrongly required exact unit variance from approximate randomized SVD;
corrected to its orthonormal-basis property, without changing the model or recipe.


### E80 optional consumed DEV implementation prepared; E79 still active (2026-09-13)

Prepared a gated E80 DEV implementation without freezing or reading its pixels. It
requires the complete TRAIN pass, reuses tested E71 pair/reference/transition helpers,
and adds padded fixed-batch9 DEAR feature inference with padding removed before
three-crop aggregation. Two focused tests pass: padding cannot add/change observations,
and failed TRAIN refuses DEV registration. All20 numeric and per-source AI-retention
gates remain; this cannot make consumed E66 fresh.

Consolidated the living overnight overview to remove obsolete current-state claims
while retaining append-only history here. Completed GitHub CI34732736214 has successful
Python, web lint/type/tests and the same existing failed web dependency audit.


### E77 TRAIN-only visual diagnostic selection planned (2026-09-13)

While E79 computes, inspect three E77 clean false-AI and three correct REAL examples
from the RR TRAIN pool only. Within each bin choose the first three by
SHA256(`E77_visual|parent_id`), not visual appeal or closeness to a desired narrative.
Replay the full frozen E77 TRAIN feature matrix and verify its published FPR slices
before selecting. Record selection first, then view original source images. No labels
change, no exclusion, no DEV/final pixels and no new fit or threshold. Visual appearance
cannot establish provenance or prove an image authentic.


### E77 six-image visual review; E80 runtime parity guard prepared (2026-09-13)

Viewed the six previously locked RR TRAIN examples: the false-AI bin has two posed color
portraits and one crowded stage event; the non-AI bin has a monochrome close-up, an
outdoor portrait on grass, and a low-resolution indoor portrait with a bird. Both bins
contain portraits. This small outcome-conditioned sample cannot establish cause, prevalence
or provenance and supports no portrait/monochrome exclusion rule. Labels/data unchanged;
no person identities used. Selection/review receipts and reproducible selection code saved.

Last non-AI example scores .07940195514935858, only7.31e-9 below AI cut; it is still
research-uncertain, not a confident REAL decision. Before E80 freezes, add runtime checks
after a provisional TRAIN pass: all views in batch8, max score error<=1e-6, zero decision
changes at both cuts and all-AI minimum correction shift>=-1e-8. New test ensures tiny
numeric error cannot hide a cut crossing. No cut or candidate recipe changed.


### Aggregate overnight figure prepared and visually verified (2026-09-13)

Rendered PNG/SVG directly from committed fit/development receipts. Upper panel fixes
the7,035 old REAL population even for expanded E76/E77 fits; lower panels show consumed
E66 REAL errors and newly missed AI separately. Marks TRAIN-passing but DEV-failed E70/E71.
No new score or image read. Figure footer states E79 active/E80 unmeasured at this checkpoint.
Source/output hashes and plotted values: `evidence/overnight_progress_plot.json`; reproducible
renderer: `ml/tools/plot_overnight_progress.py`. Visual inspection confirmed legible labels
and no clipped chart content. This plot is descriptive; no independent-quality claim.


### E80 pre-fit revision: preserve the official DEAR head direction (2026-09-13)

Before any E80 contract, candidate, fit or DEAR image classifier score exists, code review
identified that unsupervised PCA64 can discard the pretrained DEAR-r decision direction.
Add one fixed scalar alongside those64 PCs: official gated fc weights applied to the820
mean crop features plus official bias, with zero weights on820 std features. Compute in
float64 and standardize on all TRAIN without labels. This is a local mean-crop linear
response, not native full-image official inference. Pin the verified E78 checkpoint directly.
The final map now has386 zero-start coefficients (E77 coordinates320 + PCs64 + scalar1 +
intercept1). Earlier385-coefficient preparation remains historical; it was never frozen/fitted.
E79 extraction and all objectives, constraints, cuts, absolute/source/selective/runtime gates
are unchanged. No score/seed/rank/weight sweep; no partial-feature fit or DEV/final read.

Tests verify preservation of a decision direction outside the PCA subspace, correct active
mean-channel selection, checkpoint digest rejection, serialized prediction and exact old-map
reuse. The first null-space test had a scalar/array shape mismatch in its assertion; corrected
the expected scalar without changing implementation or model inputs. Full Python suite:
805 passed with the existing Starlette/httpx deprecation. E79 remains active and resumable.


### E77 constraint-slack diagnostic planned while E79 runs (2026-09-13)

Replay the frozen failed E77 on all12,141 TRAIN parents/all3 conditions and require exact
published REAL population metrics. Describe all-AI logit shifts and protected-REAL slack
by source/condition, near-active at fixed1e-7, plus remaining REAL false-AI margins and
positive/negative shifts. This is no fit, sweep, new candidate or threshold change; no
E79/DEV/final features or image files. Near-active counts alone are not KKT multipliers
or proof that constraints cause errors or that the target is infeasible. Views are dependent.
Register code/input hashes before execution and save the immutable aggregate report.


### E77 frozen constraint-slack diagnostic completed (2026-09-13)

Exact published TRAIN population replay passed. In clean/assigned/Q75,60/73/84 AI views
have correction shift<=1e-7;19/35/23 protected REAL views have slack<=1e-7. These are
near-active counts, not multiplier-based causal attribution. Remaining REAL false-AI counts
are777/879/736;359/435/335 of these have a positive correction>1e-7 despite no newly
misclassified REAL images. The correction rescued321/282/258 baseline REAL errors.
RR remains dominant:548/602/508 errors,295/339/273 with increased logit; median remaining
margin above AI cut is2.210/2.119/2.131 logits. Thus the observed residual error includes
substantial incorrect confidence, not only numerical cut ties. This does not prove map
infeasibility, require weaker AI constraints or justify changing labels/cuts. E80's distinct
forensic representation remains the next measured test. No new fit, image, DEV/final or
E79 feature read. Immutable receipts: evidence/e77_slack_contract.json and
 evidence/e77_slack_diagnostic.json; reproduction: e77_constraint_diagnostic.py.


### E81 objective helper prepared conditionally, no TRAIN fit (2026-09-13)

E77's frozen slack audit found increased logits on359/435/335 already-wrong REAL views,
with RR dominating remaining errors. Prepare a conditional objective alternative while
E79 runs: minimize worst REAL source/condition cut-centered BCE with parent-balanced
within-group means. Keep the prior REAL coefficient .5 and L2.01; remove only the AI BCE
reward. All13,785 AI TRAIN logits must still not decrease, including previously missed AI;
correct-REAL decision guards stay unchanged. This focuses the loss on the user's REAL
error objective while AI protection remains a hard constraint. It is a local constrained
worst-group adaptation, not a reproduced literature result or a guarantee of generalization.
Unlike E68's two class epigraphs and decision-only guards, this has one REAL epigraph and
the stronger E73 full-AI-confidence bounds. SLSQP200/ftol1e-9/zero correction unchanged.

Only the generic optimizer and synthetic gradient/conflicting-class tests are prepared.
No E81 TRAIN representation, contract, scores or fit yet. Finish E80 first; consider the
new objective only if actual E80 results still show the relevant REAL/source failure.
If used, register one fit in the unchanged complete E80 map, no rank/seed/L2/cut sweep,
and retain every E80 population,10-metric,selective and runtime gate. No DEV/final access
unless the complete preceding gates pass. Prior failed candidates remain frozen.

E81 preparation validation: three synthetic tests pass (finite-difference derivatives,
worst-REAL improvement under all-AI guards, and refusal to sacrifice even missed AI in
conflicting features). Full suite808 passed, existing Starlette/httpx warning. No production
TRAIN fit/score was created. Current overview updated; earlier checkpoints remain historical.


### RR upstream-source clarification during E79 (2026-09-13)

[RRDataset's primary paper, sections3.2.1–3.2.2](https://arxiv.org/html/2509.09172v1)
names news photography sources and COCO/CC3M-val/Unsplash for its broader REAL collection,
with multiple generators and some Chameleon AI samples. This is broader-corpus provenance,
not an established per-file mapping for our1,250 legacy RR TRAIN REAL rows. The
[inspected official repository README](https://github.com/ChunXiaostudy/RRDataset)
describes directory structure but supplies no such per-file mapping. The existing E33
filename audit already established pooled real_* names without camera/site identities.

Record a limitation of treating the outer RR publisher label as full lineage: exact-body
matches alone cannot rule out upstream COCO derivatives or establish upstream independence.
No specific new overlapping image has been identified; do not invent a match, relabel/drop
files by appearance, or rewrite frozen candidates. Existing legacy TRAIN research continues
with this limitation; no new COCO dataset is admitted and no unseen-COCO/fresh-final claim
is permitted. E80's pre-fit limits now record this explicitly; representation, objective,
AI guards and acceptance thresholds unchanged. Future clean benchmark claims need audited
upstream membership or a demonstrably independent source. No authors contacted, new images
read/downloaded, protected scores opened, or data-role reassignment performed in this review.


### E79 first attempt hit9000s; same-contract resume (2026-09-13)

The first bounded extraction stopped at its9000s deadline with12061 immutable chunks
and80 remaining parents. Last progress was12,050 at8983s. No image or numeric
failure was reported. Frozen E79 validation passed. Resume the identical recipe/population,
validating every existing chunk and computing only missing parents. No partial-feature fit,
budget rewrite, source/label/threshold change. Final total cost includes both attempts and
must not be reported as within2.5h. Receipt: evidence/e79_attempt1_deadline.json.
The first metadata-count assertion included exFAT AppleDouble sidecars and wrote no receipt;
strict five-digit NPZ-name counting fixes the audit. The extractor's numeric paths already
exclude those sidecars; its code/contract/pixels are unchanged.


### E79 complete; E80 one fit registered (2026-09-13)

All12,141 TRAIN parents/all3 conditions are complete: shape12141x3x1640,34 source
representative replays exactly equal (max error0), MPS driver peak2,290,827,264 bytes.
First attempt stopped at9000s with12,061 chunks; same-contract resume validated all old
chunks and created exactly80 missing parents in412.623s. Extraction-loop cost
is at least9412.623s across both attempts, plus setup/validation and the
interruption interval; the final receipt's seconds field is resume-only. Full extraction
did not fit the initial2.5h allocation. Numeric/memory checks pass; resource failure remains
recorded. No upstream classifier head execution, new image download or DEV/final access.
Feature SHA256:b357ad637640c3c34e04ece4fa7eb949906fb07d97f3e6fe751bb92ccff39a02; compact receipt: evidence/e79_features.json.

E80's single386-coefficient fit is now registered against complete verified features:
contract SHA256:5a1165b2895d2a31c0a655eee1a73cfb36d3e6b057d293d0487906a9f041b1a9. Exact E77 map320 + DEAR PCs64 + fixed official
mean-crop head scalar1 + intercept1, zero correction start; unchanged E64 objective and
full-AI confidence constraints. All E80 population/source/selective/runtime guards apply.
No candidate or detector-quality result yet. E81 remains a conditional prepared objective,
not a TRAIN fit. No E49 access, deployment or independent-final claim permitted.


### E80 failed full TRAIN screen; E81 objective now justified (2026-09-13)

The one386-coefficient fit completed in248.664s/52 iterations, solver success. Minimum
AI shift -8.88e-16, zero new AI misses/REAL errors, exact saved-model replay. Old-REAL
FPR8.7989%/9.7939%/8.6141%; expanded8.2428%/9.2102%/8.0705%; new MIDD
.5871%/1.1742%/.5871%. All population10% and new-sensor20% guards pass.
However, both old/expanded TRAIN fail worst REAL source and covered accuracy in all3
conditions (8/10 numeric checks per cell). RR FPR35.44%/40.48%/34.96% still exceeds20%;
expanded covered accuracy94.4376%/93.7688%/94.5167% falls below95%. Full TRAIN fails.
Runtime batch replay skipped by its predefined guard; no DEV/E49 scoring. Candidate
SHA256:136be4ca5ef6aeac683f604a4e3820870df3436802e6fb924846e09ee9f58d22. Full immutable report: evidence/e80_fit.json.
This is a TRAIN improvement, not an external-quality pass or promoted model.

E80 confirms the conditional E81 trigger: pooled REAL passes while the dominant source
fails. Register one E81 worst-REAL source/condition objective using the exact complete E80
map, discard E80 correction weights and start386 at zero. Keep .5 REAL loss coefficient,
L2.01, SLSQP200/ftol1e-9, all13,785 AI logit non-decrease constraints and correct-REAL
guards. Remove only the AI BCE reward; preserve every E80 acceptance and runtime check.
No rank/seed/weight/cut sweep, data change or new image extraction. No DEV/final until
complete prior gates pass. This tests an objective change after measured source failure.


### E81 implementation precision before fit freeze (2026-09-13)

The objective uses equal-parent mean BCE within each REAL source/condition and minimizes
the worst of those groups. This replaces E80's source-averaged, hard-REAL2x weighted loss;
there is no hard-REAL multiplier in the new group means. The earlier shorthand "remove
only the AI BCE reward" described the AI-side change, not an otherwise identical loss.
Keep .5 REAL coefficient and L2.01, and all full-AI/correct-REAL constraints. This is one
registered objective design, not a single-factor causal ablation or a weight sweep.

Code review caught a copied E80 evidence-output path before any E81 freeze or TRAIN fit.
Corrected it to E81 and added a test that preserves the existing predecessor report and
refuses a second write. No old candidate/report was modified. E81 also tests exact map
copy with discarded prior correction/artifact bindings and rejects an inapplicable predecessor.


E81 single-fit contract is now frozen: `21dc8c72c4f9085a8cf08d2dbda7fa7e8e19e47d8c5b133a67f01037b4c1fe57`. Full Python suite811
passed (existing Starlette/httpx deprecation). No E81 candidate or quality result yet. Run
once from zero correction; no basis refit. Exact output namespaces and prior-report
immutability are tested. All E80 acceptance/runtime gates retained.


### E81 failed full TRAIN screen; nonlinear representation next (2026-09-13)

One fixed-map worst-REAL fit completed in330.585s/50 iterations. Solver success; maximum
constraint violation4.393e-10, minimum AI shift -5.329e-15, zero new AI/REAL errors, exact
serialized replay. Old-REAL FPR8.5430%/8.5430%/8.3298%; expanded8.0042%/8.0705%/7.8187%.
New MIDD.5871%/1.5656%/.7828%. Every population budget passes, but RR32.24%/33.52%/31.84%
still fails20%, and expanded covered accuracy94.5844%/94.5498%/94.6857% fails95%.8/10
numeric gates per old/expanded condition. No runtime/DEV/E49 permitted. Candidate SHA256
`66d6cab0785946ec029307f710eec1bb16ab9de6c0986ba72c32e000190d87ec`; immutable report evidence/e81_fit.json.

Relative to E80, TRAIN AI recall rises on clean/Q75 but falls slightly on assigned transport
(99.0424% to98.9336%); E81 still loses zero E43-caught AI and exceeds the E43 reference.
E80 is a rejected candidate, not a newly adopted reference; retain both results and do not
hide that difference. The target and reference definition are unchanged.

The source objective lowers RR errors but does not meet the gates. This does not prove
linear-map infeasibility or justify a loss/rank/threshold sweep. Prior E63 already tested
Gaussian RBFs in DINO-only PCA64; E70/E74 tested bilinear maps; E77 learned a REAL-only
reconstruction residual. Next investigate a supervised nonlinear feature learner on the
complete frozen E80 multimodal coordinates, followed by the same constrained E81 head.
This differs from those prior mechanisms and needs its own representation/fit contracts.


### E82 supervised nonlinear representation planned (2026-09-13)

E81's measured RR31.84–33.52% and covered-accuracy failure motivate a learned class-
discriminative feature map. Prior E63 RBFs, E70/E74 bilinear maps and E77 REAL-only AE are
not this experiment. Use all12,141 admitted TRAIN parents/36,423 views in the exact frozen
E80 coordinates, excluding intercept (385 inputs). No new image inference/download, no
E66/E49 or other protected features. StandardScaler fits all TRAIN inputs without labels.

Train one385→256→64→1 ReLU MLP with binary REAL/AI labels, CPU seed82 initialization,
AdamW2e-4/betas(.9,.999)/eps1e-8/weight_decay1e-4/foreachFalse, MPSfloat32 if available.
Fixed100 epochs/batch256, NumPy SeedSequence[82,epoch] permutations, no dropout, early
stopping, architecture/seed/epoch/weight sweep. Use existing class/source/parent-balanced
weights, E43 baseline-false REAL2x then class mass.5, global mean weight1; never renormalize
within mini-batches. Classifier logits are used for TRAIN BCE only; they are not a deployed
detector or an AI-retention claim. This is a local engineering hypothesis, not a paper reproduction.

Save complete model/optimizer checkpoints every10 epochs, validate binding/weights/trace
on same-recipe resume. Require final weighted TRAIN BCE improvement and finite weights.
Freeze the64 latent ReLU features; their full affine span already contains the learned
classifier logit, so do not append a redundant scalar. Evaluate saved float32 weights in
CPUfloat64 with float64 input normalization; fit latent StandardScaler on all TRAIN. This
inference precision is specified before training and checked for exact saved replay plus
all-view batch8 feature error<=1e-10; it does not fix or claim parity for the older AE stage.
No PCA/rank selection. Output12141x3x64 with parent/role/contract/input-map bindings.
Budget3600s per execution with AC/mount/free-space checks; no intermediate detector selection.

Only complete verified E82 features permit a separate E83 fit: exact E80 coordinates385
plus new64 plus intercept =450 zero-start coefficients, same E81 worst-REAL objective,
all13,785 AI logit and correct-REAL guards, all E80 population/numeric/selective/runtime
checks. No new data/cut and no DEV/final before preceding gates pass. A supervised feature
learner can overfit TRAIN; only subsequent external screening can assess transfer.


E82 representation contract frozen before learning: `64a1d4fd2c8efaad5792d731345cc3795454d1374e1625f68292087bed5a0fa1`. Four
new tests cover global class/hard-REAL weighting, latent affine-span/logit preservation,
CPUfloat64 saved/batch replay, fixed epoch order/checkpoint binding, and exact next-step
AdamW checkpoint restoration. Full suite815 passed, existing Starlette/httpx deprecation.
Input-map replay must match frozen E80 population metrics before training. No E82 trained
artifact/quality result yet; prepare/run only this registered100-epoch representation.


### E82 complete; E83 single constrained fit registered (2026-09-13)

E82 completed100 fixed epochs in54.910s on36,423 TRAIN views. Weighted BCE fell
from0.6951443 to9.598565e-8. This is training fit, not generalization evidence; the very
low loss may reflect overfitting. No DEV/final rows or new image inference were used.
Saved64-coordinate features reproduce exactly, including fixed-input batch8 (max error0).
This new layer check does not certify the older E77 float32 autoencoder stage.

E83 is frozen before its one fit: `4d21e5357416ceb5d6f768a40204e9b2d86fc59adeda249f4b6a6f4fc726ae24`. Exact E80 coordinates385 plus frozen E82
latent64 and intercept produce450 zero-start weights. E81 worst-REAL objective, all-AI
logit protection, correct-REAL protection, all60 TRAIN metric checks and full batch8
runtime replay remain unchanged. No rank, seed, weight, margin or cutoff sweep. Only a
complete TRAIN/runtime pass permits a separately frozen consumed E66 DEV comparison.
817 Python tests pass; one existing Starlette/httpx deprecation. E49 remains unopened.


### E83 full TRAIN/runtime pass; consumed DEV registered (2026-09-13)

One53-iteration fit completed in138.586s including replay. All60 fixed TRAIN numeric
gates, population/sensor budgets and zero-new-AI/REAL guards pass. Old-REAL FPR
0.113717%/0.113717%/0.071073%; expanded0.106016%/0.106016%/0.066260%; new MIDD0%.
AI recall99.847661%/99.825898%/99.738847%. All13,785 AI logits protected; minimum
fit shift2.78e-16. Runtime batch8 on all36,423 views: max score error5.67056e-7, zero
decision changes at both cuts, minimum AI shift-8.88e-16, pass. Solver violation1.38e-10.
These are resubstitution results after supervised E82 learning, not unseen-image accuracy.

Candidate SHA256 `d4c7d644de0e59cda8dda0899e30860d56991530ab759f5525ea9c708b0dbd1a`. No E49 reads or serving change.
One consumed E66 development comparison is now frozen: `08a2451de42642d8163738e5a96d16153a3c3f480025d4a13b316c80f9c90c9b`. Same640 image/transport
views, all20 fixed numeric gates, zero new E43-caught AI losses per source/condition and
non-increased REAL FPR; no training or cut adjustment. E70/E71 already consumed this set.
DINO/CLIP/DEAR features will be saved externally with image/order/contract/DEVELOPMENT
bindings to avoid repeated expensive extraction; never eligible for TRAIN. 819 tests pass.


### E83 analysis tools prepared while DEV runs (2026-09-13)

Prepared a separate aggregate-only E83 progress plot, preserving the historical E77
snapshot. It will render only after the complete DEV report exists. Also prepared a
conditional read-only component diagnostic: only a completed rejected E83 DEV report
can freeze it. It verifies all640 cached DEVELOPMENT identities/roles and batch8 locked
score replay before grouping the eight additive logit components by source/condition and
paired transition. No block-removal ablation, image selection, fit or cutoff choice.
Role/order/hash rejection test passes. Neither tool has yet read unfinished DEV results.


### E83 consumed DEV rejected despite substantial REAL improvement (2026-09-13)

All640 views completed in575.982s. Frozen E43 reference scores and both cuts replayed
exactly. Original: REAL FPR42.5% to0% (68 rescues), AI97.5% to98.75%, AUC0.999765625,
10/10 numeric gates. However one previously caught GPT image is newly missed; three
previous misses are rescued, so pooled gain cannot satisfy the zero-new-loss condition.
Q75: REAL FPR42.5% to13.75% (46 rescues,22 remaining false alarms), AI96.25% to99.375%,
zero newly missed AI; AUC0.988828125.7/10 numeric gates: pooled REAL13.75%>10%, worst
SIDD:GP24.2424%>20%, covered accuracy92.4342%<95%. Overall rejected. No E49 or promotion.
SIDD remains consumed DEVELOPMENT;160 observations represent only10 dependent scenes.

Read-only component diagnosis reproduced all locked scores exactly from cached features.
The supervised component contributes roughly-5.6 to-6.2 logits to original REAL rescues.
For remaining Q75 REAL errors its group means range+0.375 to+3.960. The one newly missed
original AI has total shift-2.854, supervised component-1.370; several old blocks also
contribute negatively. Correlated-basis contributions are descriptive, not causal ablations.
Complete receipts: e83_dev_scores.json, e83_development.json, e83_component_diagnostic.json.
Updated PNG/SVG compares matched TRAIN and consumed DEV populations; visually checked.
820 Python tests pass. GitHub Python job passes; existing web dependency audit still fails.

### Next bounded direction: E84 social-transport TRAIN feature extension

The measured transport gap and known pipeline mismatch motivate one data-view change.
Retain all12,141 admitted TRAIN parents and all three old conditions. Add exactly one
1080px-long-side-then-JPEG75 condition to every parent, both labels, with the same E65
social transport and E83 crop/encoder operations. TRAIN's old Q75 encodes before2048 cap;
it is not that condition. No SIDD/other DEV/final image admission, new data download,
threshold adjustment or failure-specific image selection. Audit/bind complete TRAIN identities,
freeze code/weights/resource budget and resumable chunks before extraction. Verify encoding
and numerical parity on fixed source representatives; report failure rather than alter recipe.

Only complete E84 features will justify separately registering a supervised representation
with the same E82 architecture/optimizer/initialization and all four TRAIN conditions, then
one constrained head. Preserve every old and new AI-view E43 confidence, correct-REAL
protection, all absolute gates and runtime replay. Additional transport coverage is a
hypothesis; it is not guaranteed to recover the original AI miss or generalize beyond E66.
No representation/head has yet been frozen or fitted for this extension.


### E84 transport extraction frozen before probe (2026-09-13)

Contract `41604833e4b2d809d8906949c973227e26bc29ca0696627b8190d51b79161dad`. All12,141 original TRAIN parents, one exact social1080/JPEG75 view
each, preserving original three-condition caches. Fixed eight-parent windows; last five
views padded to eight during encoding then padding removed. DINO/CLIP/DEAR frozen.
No new-view classifier scores. Before full run:34-source old-clean parity,16 hash-selected
windows/128 parents for measured cost, one repeated-window parity,6GiB MPS budget.
Full encoding estimate must fit10,800s; AC/20GiB reserve and immutable restartable chunks.
Interrupted probe may replay the same128 views for cost measurement; existing chunks
require exact body/features and are never overwritten. Full run only fills missing chunks.
Three transport/order/role/corruption tests and all823 Python tests pass. Probe not yet run.


### E84 probe interrupted by native source schema; source repair registered (2026-09-13)

E84 completed8 probe windows/64 parents, then raised KeyError:path on a native TRAIN
source-key row. Original contract/code/chunks are preserved; full extraction is denied.
This is an engineering input error, not a detector quality result. Receipt
`evidence/e84_probe_attempt1.json` verifies all64 chunks. Do not omit affected rows.

5,652 admitted native rows have no direct path:4,652 use the existing E32 loose-file
resolver;1,000 are selected Parquet cells (500 CommunityForensics,500 Nano). Source
contract `6750838f3a3fe0fa8e0f12d76874d18b2277235035bffa40874672ba0c32c23f` freezes all12,141 original-body checks and only the selected TRAIN
Parquet materialization. Row-group I/O can include other encoded cells; only admitted
indices are converted to byte payloads, with no unselected label extraction/image decode.
No standardized224JPEG fallback. Original SHA must match every parent.

E84B is prepared as a source-resolution-only revision, reusing the exact E84 encoder,
transport, parity and chunk functions. Same population, hash-selected probe windows,
numeric/memory/time guards. It will bind the complete verified source index and require
exact replay of the64 previously encoded views. No B contract or encoding yet.
825 Python tests pass, including selected-cell extraction and original-body/role rejection.


### E84 original bodies verified; E84B frozen (2026-09-13)

All12,141 TRAIN source bodies verified in518.833s. Materialized1,000 selected native
Parquet payloads (976,266,728 bytes) locally; no download/image decode/model inference.
Source index SHA256 `d722b16400d11afec05813a69273d4409ab40f4dcaf41e162406ac664795b0e8`.
E84B contract `6ca86aad24fdb89c91d711e35c24846e500cb044bdb76020488489ed1474de0c` preserves original E84 probe windows and all scientific/resource
settings. It reuses frozen E84 operations and requires exact replay of the64 old probe
views. A pre-freeze code review caught the renamed hash prefix changing probe windows;
corrected to read the original frozen plan directly, with a regression test. No B feature
or score had been computed. E84B probe follows; full extraction remains conditional.

E85 four-condition data/representation code is prepared, not frozen/fitted. Old three
feature tensors must remain exact when appending social Q75. Same E82 architecture,
seed82, optimizer and100epochs; input/latent scalers learn only the complete four-view
TRAIN data. Before learning, old E80 population replay and old E83 decisions under the
new input layout must pass. Source/order/class-mass and tiny decision-flip tests pass.
827 full Python tests passed before two additional focused E85 tests (both passed).


### E84B probe passed; full transport extraction starts (2026-09-13)

E84B completed the same128-parent probe.34-source old-clean reference score/cut replay
and CLIP/DEAR features are exact; repeated new window exact; all8 predecessor windows
(64 views) reproduced exactly. Source schema repair changed no measured encoder output.
Encoding/source-read/chunk-write time72.781s projects6,903.391s (~1h55m) for12,141
parents, within10,800s budget. Setup16.349s and parity31.946s are separate; total probe
125.907s. MPS driver peak5,518,344,192B, below6GiB. Full extraction is permitted and
will preserve probe chunks, fill missing fixed windows and verify complete coverage.
Receipt `evidence/e84b_probe.json`. No new-view classifier scores or DEV/final image reads.

E85 preparation now verifies original E83 artifact hash and condition metadata; E86 four-
condition gate helper matches old three-condition results exactly and rejects an old-
condition failure even if the new condition passes. Focused tests pass. No E85/E86 fit
or representation freeze yet; complete E84B features are required.


### E85/E86 complete preparation while E84B runs (2026-09-13)

E85 representation and E86 single-fit/consumed-DEV code are prepared, with no new
representation/head contract or model scores. Same E82 seed82/architecture/optimizer/
100epochs on four TRAIN conditions; E86 starts zero450 coefficients and keeps E81's
worst-REAL objective. All18,380 AI views retain E43 logits; correct-REAL guards apply
to all four conditions.80 numeric TRAIN checks cover old/expanded populations independently,
plus separate MIDD budgets and full48,564-view runtime replay. Old-condition failures
cannot be averaged away by the new condition.

Only full TRAIN/runtime success may freeze E86 consumed E66 scoring. It verifies and
reuses the exact complete E83 DEVELOPMENT feature cache, replays E43 scores and both
cuts, locks scores before metrics, checks all20 numeric gates and zero new E43 AI losses,
and also reports paired changes against E83. No new image inference or cache role change.
832 Python tests pass. Checks include four-view parent/class weights, old decision-flip
rejection, gate parity, immutable predecessor reports and stopping before DEV cache access
when TRAIN fails. E84B feature extraction continues under its frozen contract.


### E84B metadata-only transport coverage (2026-09-13)

The hash-bound original TRAIN manifests contain12,141 parents, all with dimensions.
REAL:3,091 at most1080px long side,426 at1081–2048,4,029 above2048.
AI:3,549 /1,045 /1 respectively. The social1080 cap potentially affects59.04% of REAL
and22.76% of AI parents. Identical policies do not imply equal transformation exposure.
Declared REAL formats:7,034 JPEG,1 PNG,511 unknown (MIDD metadata); AI:786 JPEG,3,809 PNG.
Do not substitute filename extensions for unknown manifest fields. This class/format
association is a descriptive coverage limitation, not a causal explanation or score result.
For6,640 inputs at most1080, the new view is expected to duplicate old source-JPEG75
under unchanged helpers; no complete pixel-equality test was performed or claimed.

Tool `ml/tools/train_transport_coverage.py` executed once; receipt
`evidence/e84b_transport_coverage.json` pins both manifests and tool SHA. Zero image reads,
model scores, parameter choices or E49 reads. No change to active frozen E84B extraction;
E85/E86 remain prepared, not frozen or fitted. Latest GitHub242dfd7 Python checks pass;
web lint/typecheck/tests pass, existing vinext dependency audit remains failed.


### SID metadata and archive inventories complete (2026-09-13)

Pinned six split lists and README/license total373,748 bytes; after a local stdlib TLS
certificate-store failure, the same HTTPS requests succeeded with certifi verification.
5,094 listed short captures map to424 camera/filename scene groups and424 unique long
RAW references. Sony TRAIN/VAL/TEST groups161/20/50; Fuji135/17/41. All listed members
exist in the official archives. Remote ZIP directories required472,283 bytes total;
no image member was opened. These filename groups are not verified independent scenes.

Sony archive26,926,662,016B; Fuji55,409,370,853B. Their official TRAIN long references
alone require2,533,425,149+5,007,113,383 compressed bytes (296 originals); no full archive
is necessary if exact206/ETag range verification remains available. Inventory receipts:
`evidence/sid_metadata_inventory.json`, `evidence/sid_archive_inventory.json`.
Both cameras' pinned author training code renders original long RAW through RAWpy with
camera white balance, full size, no automatic brightness and16-bit output. Network outputs
are separate and must not be labeled authentic captures. No RAW decoder is currently in
our experiment environment. No pixels, role admission, model scores or change to E84B.

Next bounded preparation: select64 original long-exposure TRAIN references per camera
by a fixed filename hash, one per published scene; freeze exact members/source/terms
and a4GiB compressed/5GiB raw budget before downloading. Acquisition stays quarantined
for possible future research TRAIN, with the whole SID publisher reserved from fresh
DEV/final. Existing E85/E86 data and recipe remain unchanged. Decode/protected-overlap
and scene checks must be separately registered before any TRAIN admission. Selection
uses no image content or detector score; no refill after duplicate/quality quarantine.
This adds a documented non-JPEG capture route, not a promised improvement.


### E87 bounded SID RAW acquisition frozen (2026-09-13)

Contract SHA256 `748dc0b3e18c93d9e09441fcc8f8f7a64ca02aef5484512b93e1dcfad1e4d5b2` selects64 Sony and64 Fuji original long-exposure
references from the publisher's TRAIN scene lists by SHA256(E87|filename).128 RAW bodies
require4,851,194,880 bytes; exact member ranges plus directory allowance3,384,272,676 bytes.
One worker, AC,30GiB free reserve,90min per execution, HTTP206/If-Match, ZIP CRC/length/SHA,
verified completed-member resume, no full-archive fallback. Five focused tests pass for
order/repeated-burst invariance, test-split rejection, repeated-scene rejection and source URL.

Role remains QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN. No decode or classifier scoring in
acquisition; no refill and no automatic TRAIN admission. Entire SID publisher is reserved
from fresh DEV/final. Research source/license limitations remain as recorded. RAW originals
may themselves use camera compression; future lossless RGB refers to our output encoding,
not a claim of pristine/unprocessed sensor data. Active E84B and prepared E85/E86 are
unchanged. Separately register fixed rendering and protected overlap/scene audit next.

E87 implementation verification: full Python suite837 passed in18.00s; one existing
Starlette/httpx deprecation warning. No experiment runtime dependency was changed.


### E87 download active; E88 RAW audit prepared, not frozen (2026-09-13)

E87 fixed128-member download is active; Sony64 complete, Fuji in progress at this
checkpoint. CRC/source SHA/range checks run on every member; still no RAW decode/model
score. E88 implementation now requires the complete E87 receipt before freezing.

E88 will reuse `e56_raw_decode.py` with the existing `ml/work/e56_decoder` runtime,
exactly matching E65's frozen versions: Python3.13.5, RAWpy0.27.1/LibRaw0.22.1,
NumPy2.5.1, Pillow12.3.0. Existing full-size/as-shot-WB/sRGB8/gamma2.4,12.92/PNG6
recipe, no auto brightness or learned enhancement. This differs from the SID paper's
16-bit/default-gamma rendering and is explicitly a local existing convention.
A redundant2,058,253B official RAWpy wheel was downloaded/verified and installed only
under external `source_research/rawpy0271`; SHA
`878b16434cebe66f2a575dae27ebca673be2b4b537e1f593072218bb820a79db`.
That copy has not decoded any images and will not be used; experiment/site packages
were unchanged. The already available isolated runtime supplies the required decoder.

Prepared E88 checks original RAW identity, canonical rendered body/RGB/perceptual matches,
and original RAW hashes retained by protected derivative references. It includes all
E72-protected snapshots/AI reserves/consumed E66 plus511 admitted MIDD fingerprints.
Cross-camera internal matches form transitive components; any protected overlap excludes
the whole component, then one fixed-hash representative survives. No brightness or
model-score selection/refill. Filename groups are not verified independent scenes;
no GPS/serial extraction or full EXIF dump. Two focused tests cover original-RAW overlap
through a differently rendered reference and cross-camera transitive rejection.
No E88 freeze/audit or use in E85/E86 yet; E84B extraction continues unchanged.

Full verification after E88 preparation:839 Python tests pass in18.18s; only the
existing Starlette/httpx deprecation warning. E88 is still not frozen or executed.


### E87 complete; E88 fixed RAW audit registered (2026-09-13)

All128 RAW originals completed,64 per camera.4,851,194,880B verified bodies;
3,376,363,519B transferred through pinned206 ranges. No decoder/model scoring yet.
Download receipt SHA256`60c5563122a1ae7aaa76a13a0a6ac4c316a8a0d9daa03f7e455d1fa62e0dedff`.
The original acquisition contract remains`748dc0b3e18c93d9e09441fcc8f8f7a64ca02aef5484512b93e1dcfad1e4d5b2`.

E88 audit contract now frozen: `f4bab7531e2859e7a33540f59f5994a18526f75ee7ddbedd62476e70c771c32a`. Same existing isolated RAW
renderer/runtime verified against E65. Complete original-body and rendered-image overlap
screen, transitive internal grouping, no score/brightness filtering, no refill. One-hour
stage with120s per RAW subprocess; all128 accounted for. No TRAIN admission until audit
receipt/manifest completes. E84B extraction and the prepared E85/E86 sequence stay fixed.


### Conditional E86 consumed-regression implementation prepared (2026-09-13)

`e86_regression.py` now prepares freeze/score/report stages, but none has run. It first
validates immutable E86 TRAIN/runtime and consumed E66 DEV passes before any E49 manifest,
score or image access. A failed/new-AI-loss DEV state is blocked by focused tests.
Only after passing does it bind the existing4,000-view E49 manifest and E43 baseline.
It reuses exact E84/E83 encoders, E43 crops on already-realized original/Q75 images,
and frozen E86 runtime batches8; no second social transform or model refitting.

Immutable per-window chunks retain complete DINO/CLIP/DEAR arrays, image/order/role/
contract bindings and float64 score byte hashes. Resume requires original image SHA,
finite feature/hash checks and exact model/reference replay. E43 tolerance5e-5 still
requires zero changes at both cuts. Fixed two-hour/6GiB resource ceiling, no downloads.
All scores lock before metrics. All20 fixed numeric gates, lower REAL FPR, pooled/source
AI no-loss plus zero newly missed E43 AI per source/condition remain mandatory. Same
E63 paired source-cluster interval construction,20,000 draws/label, seed86 fixed,
Bonferroni4 bounds. This remains consumed regression, never independent final/promotion.

Five focused tests pass: two pre-access DEV failures, tiny cut crossing, float64 hash
integrity below float32 precision and paired interval direction. No E49 data was read;
only implementation mechanics used synthetic values. E85/E86 remain unfitted while
E84B extraction continues. E88 RAW audit continues without any decoder failure so far.

Two fixed E87 hash-order examples (first per camera) were viewed solely for rendering QA:
Sony00108 and Fuji00120. Both open as recognizable scenes; no claim of authenticity
from appearance, no brightness selection, settings change, filtering or scores. Their
actual rendered sizes4256x2848 and6032x4032 differ from commonly quoted cropped SID
sizes; the pinned full-size LibRaw output is the audited input, not an inferred crop.

Full verification after conditional E86 regression preparation:844 Python tests pass
in20.28s; one existing Starlette/httpx warning. No E86 regression stage executed.


### E88 SID audit complete:128 additional research TRAIN originals (2026-09-13)

All128 RAW originals decoded,64 per camera, in860.226s including overlap checks.
Against151,996 protected/reference records: zero original-RAW hash matches, zero rendered
body/RGB/perceptual cross-matches, zero internal similar pairs, zero failures/quarantine.
All128 admitted to a separate research TRAIN manifest; no quota refill. Full-size output
64x4256x2848 Sony and64x6032x4032 Fuji. No detector score, independent-final admission,
model fitting or serving change. Filename groups are not proven independent scenes;
whole SID publisher stays exclusively research TRAIN, with recorded source/rights limits.

Audit SHA256`49c2e9fa930d0c85eff5d1b871351aa62f73c6ba79317bf19a7aa3384bdef6af`;
manifest SHA256`73545e7a9404e3c4f83ee2f6affd46c6ae2e2bf75ee37db0e8fd56dab11c5d83`.
External `e88/training_manifest.json` keeps original RAW identities and derived PNG hashes.
Receipt `evidence/e88_audit.json`. This dataset is available for a separately registered
future coverage extension; active E84B and E85/E86 still use their original12,141 parents.
Next prepare a bounded, no-new-score four-condition encoder cache for the128 SID parents,
reusing exact existing transforms/encoders. Do not compete with active E84B for GPU memory;
finish the existing feature/model comparison before launching a second extraction/fit.


### E89 new-cohort feature cache prepared, not frozen or started (2026-09-13)

Prepared all128 audited SID parents x4 existing conditions,512 views. Reuse exact E75
clean/assigned/source-Q75 and E84 social1080->JPEG75 helpers, plus identical frozen
DINO/CLIP/DEAR encoders. Two parents per8-view batch, float32 features; verify original
RAW and decoded PNG identities. Same34-source old-clean E43/CLIP/DEAR parity and first
new-batch repeat<=1e-5. No classifier score for SID, no fitting or E49 image access.
Immutable resumable2-parent chunks bind ordered parents/roles/conditions/source/RAW/
derived-crop/feature hashes. Complete archive would be128x4x3072/1536/1640.

A30min/6GiB ceiling is planned from the existing probe cost. The extraction entry point
rejects incomplete E84B and requires the fixed E85/E86 fit plus any permitted E66
comparison to finish first. Operational priority also remains with any permitted E49
regression; do not launch competing GPU stages. E89 freeze is deliberately deferred
until the existing comparison is complete. This is preparation for possible subsequent
coverage work, not a model change or a promise to fit on SID regardless of results.

Three focused tests verify byte-identical old3 crops with a distinct social resized view,
pre-GPU rejection while E84B is incomplete, and exclusion of DEVELOPMENT feature roles.
No E89 image/encoder operation has run on the new cohort; tests use synthetic pixels.

Full verification after E89 preparation:847 Python tests pass in18.45s; one existing
Starlette/httpx warning. E89 remains unfrozen/unexecuted. External reserve347GiB;
macOS reports no recorded thermal/performance warning. Only E84B extraction is active.


### Planned E90 read-only E82-head/E83-decision agreement diagnostic (2026-09-13)

The E82 supervised network retains its trained scalar BCE head, although E83 uses the64
latent features and an independently constrained correction head. Inspect their agreement
on all640 already-consumed E83 DEV views, using complete saved encoder caches and exact
E83 runtime replay. Fixed native BCE sign boundary0, no threshold choice, new training,
new detector candidate, score repair, image reads or E49 access. This is a new diagnostic
read of the existing trained scalar head, not a fresh validation or causal attribution.

Before reading these logits, register/hash one E90 diagnostic: report their fixed sign and
min/median/max within every existing source/condition/E43-to-E83 transition bin. Preserve
all E83 scores unchanged. This can distinguish raw-head disagreement from agreement on
known errors, without claiming which representation caused the error. Any subsequent
model change must be separately justified, frozen and pass TRAIN/DEV guards. Do not
change the currently fixed E84B/E85/E86 transport experiment based on this diagnostic.


### E90 diagnostic registered before scalar-head reads (2026-09-13)

Contract SHA256`dd075be24a426cbc0c782f70a51a7d9503138763395e9e2dd11f0c2213cf3d37` binds all640 consumed DEV views, exact E83
replay and frozen existing E82 scalar-head sign0. No diagnostic run yet. Initial focused
and full suites exposed a test fixture exporting float64 network weights, whereas the
frozen inference validator correctly requires saved float32 weights. Fixed only the test:
export original float32 state before converting the independent reference network to
float64. Diagnostic implementation/contract unchanged; two focused tests now pass.
The registration command had already completed while the first test result was pending;
no scalar-head data was read and no diagnostic execution started before the repair.
A complete post-repair suite must pass before execution. Existing E84B/E85/E86 unchanged.

E90 post-repair full verification:849 Python tests pass in17.80s with the existing
Starlette/httpx warning. Diagnostic may now run under its unchanged frozen contract.


### E90 scalar-head agreement diagnosis complete (2026-09-13)

All640 E83 scores replay exactly (max error0, both-cut changes0);9.308s CPU diagnostic.
The sole newly missed original GPT AI has existing E82 scalar-head logit-2.59427408:
the native BCE sign also calls it REAL. All68 original REAL rescues have negative scalar
logits. Of46 Q75 REAL rescues, six have positive scalar logits (G4 three, GP/IP/N6 one
apiece). Therefore a simple proposed disagreement fallback that restores E43 whenever
this scalar head says AI would not recover that original AI loss and would undo six
Q75 REAL rescues. This is a logical implication of the measured sign/transition bins,
not an executed candidate, fitted threshold or causal proof. Do not implement/sweep that
fallback from these observations. No new detector candidate or E49 access occurred.

Receipt `evidence/e90_head_agreement.json`; unchanged E83 score hashes and all source/
condition bins preserved. The failure is not merely a disagreement with the raw BCE
head; a representation/data-coverage change remains the current hypothesis. E84B's
exact transport extension and the fixed E85/E86 learner/head sequence remain unchanged.
Separately audited SID data stays available for later justified coverage work.


### Read-only source/visual review of E83's sole newly missed AI (2026-09-13)

The complete set of newly missed E43 AI in E83 contains one original view, GPTIMG_431.png.
Source/E66 body SHA matches`32e1ccf2c9cb523c119f01afd3b7461b3db241a27d837be729b8d22e70a07d93`.
Existing E43 score0.08128665 is just above fixed AI cut0.07940196; E83 lowers it to
0.00507187. The source image is a close portrait with skin texture, sheen and haze/grain-
like detail. Appearance does not prove authenticity, edit type or a labeling mistake.
No relabeling, exclusion, TRAIN admission, model fit or new score was performed.

The [publisher card](https://huggingface.co/datasets/a3xrfgb/gpt-image-mega-4k) describes
curated GPT-IMAGE1 images captioned by Qwen3-VL. The neighboring.txt is descriptive
caption text; it must not be treated as a verified original generation prompt. No edit
mask, input photograph or generation log appears in the reviewed records. Full-generation
versus editing provenance remains unresolved. Keep the existing publisher-provided AI
label and unknown prompt-dependence limitation. Do not infer population-wide bias from
this single case. Receipt `evidence/e83_known_ai_miss_review.json` records both image and
caption hashes; no image or full caption is copied into git. Current E84B/E85/E86 unchanged.


Existing paired E83 scores for that same GPTIMG_431 parent show transport sensitivity:
original E43/E83=0.08128665/0.00507187, socialQ75=0.31710792/0.81932999. The Q75
version is already detected by both; the loss is on the original. These are previously
locked scores, not a new transform or inference. This supports examining transport
stability but does not prove that E84B augmentation will fix the original error.


### Read-only consistency-mechanism review during E84B extraction (2026-09-13)

Added a primary-source and mathematical review to `IMAGE_FORENSICS_REFERENCE.md` under
"Paired prediction consistency". A synthetic fixed-feature gradient check returned
exactly0 parameter-gradient difference; no real images or candidate were used. The
review separates an unverified implementation ambiguity from a possible future local
objective. E84B continues unchanged; E85/E86 remain the next fixed experiment. No new
loss, threshold, feature rank, seed or dataset mixture is selected from this review.
Any later consistency experiment must have its own complete preregistration and keep
all existing quality/retention/runtime guards. This is a research note, not a result.


### E86 aggregate visualization prepared (2026-09-13)

Prepared `ml/tools/plot_e86_checkpoint.py` to compare immutable E43/E83/E86 consumed
DEVELOPMENT aggregate rates and newly missed E43 AI counts. It requires complete
320-view condition reports and writes PNG/SVG plus hashes without image/model reads.
No figure has been rendered and no E86 result exists yet. Python compilation passes;
visual verification is deferred until the real E86 comparison is complete. Existing
E83 historical graphics are preserved. This reporting utility changes no experiment.


### E84B complete; E85 four-condition representation registered (2026-09-13)

All12,141 existing TRAIN parents now have the exact social1080->JPEG75 view. Complete
DINO/CLIP/DEAR shapes12141x3072/1536/1640,1518 immutable chunks. This execution created
12,013 parents and reused128 probe parents;7296.694s, peak MPS5,520,441,344B within
budget. No new-view classifier score, DEV/final image read or download. Archive SHA256
`8f2a6a9c381bbfc3400ccb408680ae17da9e0bc9af50a290bc531c4ead9a2f21`; full receipt SHA256
`79d098324cc3cb60b0db33dd3d02e02b9890cde6aaf7f024b15f5bcdee97485b`. Original3-condition caches are unchanged.

E85 contract SHA256`5af47a4051c2ddc2babe1b36a26f878e6d42121181e688b9529960b7ef5352c9` freezes the previously
prepared architecture/seed82/AdamW recipe for48,564 TRAIN views (30,184REAL/18,380AI),
100 fixed epochs, same385->256->64->1 network, final-only export. No warm start or sweep.
Only its input/latent scalers are refitted on the four TRAIN conditions. The complete
E84B archive hash/receipt and predecessor artifacts were verified at registration.
Old-three score replay and saved/runtime latent checks run during training. No E85
training result exists at this checkpoint. Next execute the frozen representation,
then separately freeze E86's450-weight constrained head only after successful export.
All80 absolute TRAIN metric checks and AI/runtime guards precede any consumed E66 read.
SID's separately audited128 parents remain outside this experiment. E49/serving unchanged.


### E85 complete; E86 constrained head registered (2026-09-13)

E85 completed100 fixed epochs in75.342871s. Weighted BCE0.6953957208 to
6.20674856e-9; this near-memorization TRAIN loss is not evidence of generalization.
All48,564 views exported as12141x4x64. Saved latent/coordinate replay is exact; fixed-input
batch8 error0. Original-three predecessor score replay error3.33066907e-16, zero changes
at both fixed cuts. No DEV/final rows, new image inference or download.
Map SHA256`12c3a8c24de8f6fc3be4b9572937c84810689d450d50b95db082f504be483e07`;
feature SHA256`6a32d56aa0c860050947e1b3c12d257992741f44e2ccbff04ec6e4d32fc91c21`.

E86 contract SHA256`d9c6410739b8b7dbce2159293f1745ef82743c2a7e15f7e5bcc2385540255e9f` now binds the complete E85
export and unchanged E81 worst-REAL objective. One zero-initialized450-coefficient fit,
all18,380 AI-view logit guards, correct-REAL constraints and all80 numeric TRAIN checks.
Full48,564-view runtime batch8 replay follows only if provisional TRAIN guards pass.
No E86 fit result yet; no coefficient, margin, regularization or cut sweep. A separate
consumed E66 cache comparison is permitted only after the complete TRAIN/runtime pass.
Current reference/serving and all protected data roles remain unchanged.


### E86 TRAIN/runtime passed; consumed E66 comparison registered (2026-09-13)

One49-iteration fit completed in235.522530s. All80 numeric TRAIN checks, old/expanded/
MIDD population guards and per-source AI/REAL retention passed in all4 conditions.
Old REAL FPR0.12793%/0.11372%/0.07107%/0.08529%; expanded REAL0.11927%/0.10602%/
0.06626%/0.09276%, in clean/assigned/source-Q75/social-Q75 order. MIDD has0 errors
in old3 conditions and1/511 social-Q75 error (Sony1/128). AI recall99.93471%/99.82590%/
99.80413%/99.80413%. These are TRAIN outcomes, not independent performance estimates.

All18,380 AI logits protected; solver max violation2.43646775e-10, minimum AI shift
5.66213743e-15. Full48,564 runtime batch8 score replay error5.95048665e-7, zero changes
at both cuts, minimum runtime AI shift4.88498131e-15. Candidate SHA256
`c5747880d46d78da4bcec9061cc30564aad2b0ce14d98cd8619398214ec51380`. No E49 read or serving change.

Consumed E66 contract SHA256`88b017e94b7225f6225aef772e15542661434124863ac3049aa5f9a6b36ff27c` is now registered.
Use exact prior640-view encoder cache and fixed E86 runtime batch8, lock all scores
before metrics. Require all20 numeric gates and zero newly missed E43 AI per source/
condition, report paired E83 changes too. No new encoder/images/download/fit. This set
is previously consumed DEVELOPMENT, not a new final. No E86 DEV result at registration.


### E86 consumed DEV rejected; E89 SID features registered (2026-09-13)

All640 cached DEV views scored in6.631617s; E43 reference replay exact0/cut changes0.
Original REAL FPR42.5% reference ->0.625% (1/160), AI97.5% ->99.375% (159/160).
All10 original numeric gates pass, but the same GPTIMG_431 original is newly missed
relative to E43: E43/E83/E86=0.0812866464/0.0050718705/0.0082836705. Four other
original AI rescues cannot mask this loss. Versus E83: one additional AI rescued, no
new AI loss, but one N6 REAL rescue lost. Original AUC0.999765625, BA0.99375.

Social-Q75 REAL FPR42.5% ->10.625% (17/160), versus E83's13.75% (22/160). Five
additional E83 REAL errors corrected (GP1/IP3/S6 1), no E83 REAL/AI regression in this
condition. AI99.375% unchanged from E83, no new E43 AI miss. Social AUC0.99203125,
BA0.94375, automatic coverage96.875%, covered accuracy94.19355%, worst REAL source
GP7/33=21.21212%. Same3 numeric failures remain: pooled REAL<=10%, worst REAL<=20%,
covered accuracy>=95%. Overall17/20 plus failed original AI guard -> REJECT. No E49
regression/promotion. This is consumed development, never a fresh independent final.

The transport-only data-view extension helped five Q75 REAL cases but did not resolve
all quality/retention failures. Preserve E86 and its locked scores. The next distinct
coverage hypothesis uses all128 SID REAL captures already selected/audited before this
E86 result, without selecting examples by score. E89 contract SHA256
`020a3a475b5fda3b50047b2b4f2bbaa38e4b9d5d36eef3b9749317ec90aab469` freezes512 four-condition encoder views, same exact
DINO/CLIP/DEAR operations, source/RAW hashes,34-source parity and30min/6GiB budget.
It does not fit a model or score SID with a classifier. Run this bounded extraction
then separately register one expanded-data learner/head retaining the old recipe and
all existing AI/REAL/transport checks, plus explicit new-SID camera/population checks.
Do not simultaneously add a new consistency loss or sweep settings. Camera/scene and
publisher-lineage limitations persist; this is a coverage hypothesis, not a promised fix.

`ml/tools/plot_e86_checkpoint.py` rendered frozen aggregate PNG/SVG and hash receipt.
Visual review found a target-label overlap in the first draft; moved only the annotation
and showed three decimals for REAL rates. The first draft is preserved externally in
`report_qa/e86_attempt1`; the corrected plot is visually verified. No scores/thresholds
changed. The initial ad-hoc text summary used a wrong dictionary key and exited before
printing rates; corrected to the actual report schema, with no experiment mutation.


### E89 startup failure and unchanged-contract restart (2026-09-13)

Initial extraction stopped during DINO model initialization: the hub attempted an
online metadata check and the existing socket guard rejected it. Offline environment
flags were assigned after heavy imports, so library constants had initialized earlier.
Zero SID feature chunks/output/report exist. Preserve the frozen E89 implementation,
weights and recipe. Restart with HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and both
OMP/OPENBLAS thread limits2 set in the process environment before Python imports.
This implements the already registered offline runtime; it changes no transform,
encoder, sample, tolerance or scientific condition. Receipt `e89_attempt1_startup.json`.


### E89 complete; E91/E92 prepared; user steers to GitHub repair/report (2026-09-13)

With offline flags set before imports, E89 completes128 SID parents/512 views in
565.928863s,64 chunks.34-source old-clean parity and first new-batch repeat are exact0;
peak MPS5,520,441,344B. No new SID classifier scores, DEV/final image reads or fitting.
Archive SHA2563422cb43781810ac97342801a559b84d647c5eeeb84612d433cb41b8222570e0;
report SHA2566d77447d44457d4ee2cc31e5416a51a02e9b074663d46e308e1c5807dec2669b.

Prepared E91/E92 (not frozen or executed): append all128 audited SID REAL parents to
12,141 previous TRAIN parents, preserve all4 condition arrays.49,076 views,30,696REAL
and18,380AI; same E85 weighting/network/seed/order/optimizer100 epochs, no warm start.
Exact old E86 metric replay and bounded input-layout score replay precede learning.
Same450-column E92 constrained head; retain all earlier population guards and add SID
pooled/camera guards plus all10 numeric gates on legacy/previous/expanded populations
in4 conditions (120 checks). Separate consumed DEV keeps the E83 encoder cache and
reports paired E86 transitions. No consistency-loss, rank, margin or threshold change.

Seven focused tests pass; full suite856 tests pass in15.79s with the existing
Starlette/httpx warning. Tests cover duplicate/role admission rejection, old feature
preservation, population/condition masking, immutable separate receipts and stopping
DEV before failed TRAIN. Review caught copied predecessor receipt names before any
registration/run and corrected them; no frozen artifact was altered. User now requests
GitHub failures fixed and a report. No ML job remains active; E91/E92 stay prepared
until a later continuation, rather than launching another experiment during this repair.


## 2026-09-13 — GitHub CI security repair; no new ML run

Failed GitHub run34748481696 passed Python and web functional checks but failed npm audit:
11 alerts (1critical,8high,2moderate). Update Next.js16.3.2→16.3.5,
vinext0.0.50→1.0.0-beta.9, plugin-rsc0.5.34, Cloudflare Vite plugin1.54.8,
Wrangler4.131.1, workers-types5.20260911.1 and eslint-config-next16.3.5.
Resolve the workers-types peer requirement normally; no force/legacy-peer bypass.
Refresh compatible transitives with npm audit fix; audit now reports0 vulnerabilities.

The new vinext output moves hashed assets from /assets to /_next/static. Initial
post-migration test correctly failed its obsolete path assertion. Update that contract
and additionally verify rendered JS/CSS files exist in the build. All6 web tests,
Sites build, lint and typecheck now pass locally. CI security threshold is strengthened
from critical to high, with existing registry retries preserved. Per-ref concurrency
cancels superseded runs. Remote verification is pending push at this checkpoint.

E91/E92 remain prepared, not frozen/fitted; no new DEV/final scoring or model promotion.
Last complete ML suite:856 passed. See rapor/GELISTIRME_RAPORU_2026-09-13.md and
evidence/ci_dependency_repair_2026-09-13.json for the repair/report record.


### 2026-09-13 — Post-repair verification and report

Repair commit57aaf61 and preparation804a07f are pushed to origin/main. Clean npm ci
installs491 packages and reports0 vulnerabilities. The complete Python suite was
rerun after repair:856 passed in17.01s, one existing Starlette/httpx deprecation warning.
All report-local links resolve and E83/E86 rates match recorded JSON receipts.
GitHub CI run34749429187 has been created but remains queued at this checkpoint;
local passing checks are not presented as remote success. The Turkish development
report is in rapor/GELISTIRME_RAPORU_2026-09-13.md. No further ML run was started.


### 2026-09-13 — Remote CI success verified

Commit bca8a1e0283c41e562228fa83be9b32547ab7f14 passes GitHub Actions CI34749560334:
https://github.com/EfeHanKeles346/ai-image-detector/actions/runs/34749560334
Web28s: clean npm ci, lint, typecheck, build/tests and npm audit all pass.
Python2m54s: installs, pytest, compileall, pip check and serving-lock pip-audit all pass.
Machine-readable run/job/step evidence is recorded in
evidence/ci_dependency_repair_2026-09-13.json; Turkish report and current README/PLAN
now distinguish verified CI from the unmet ML target. This final checkpoint changes
only documentation/evidence; no frozen recipe, candidate, threshold or dataset changed.


## 2026-09-14 — English report consolidation and offline continuation

User policy: ongoing Markdown updates belong only in PLAN.md, HISTORY.md and
ml/EXPERIMENTS.md, in English. Remove the standalone Turkish report
rapor/GELISTIRME_RAPORU_2026-09-13.md after preserving its findings here. Other legacy
Markdown files remain historical references; no new separate report is created.
Mobile data: no dataset/weight/package downloads or environment upgrades. Use existing
verified disk caches, with outbound sockets denied in the experiment processes.

### Latest completed detector result: E86, limited consumed DEVELOPMENT

This is the latest overall improvement over E83, not a winner on every measure and
not an independent final success. There are160 REAL and160 AI parents, two conditions
and640 views. REAL consists of only10 dependent SIDD scenes across5 cameras; AI comes
from2 previously seen families with unknown prompt linkage. Consumed DEV features and
labels remain excluded from TRAIN. 17/20 means17 acceptance criteria passed, not85%
image accuracy and not17 independent test datasets. Original passes10/10 numerical
criteria; social-Q75 passes7/10. E86's TRAIN screen passed80/80, a different evaluation.

| Consumed DEV measure | E43 reference | E83 | E86 |
|---|---:|---:|---:|
| Original REAL false AI | 42.500% (68/160) | 0% | 0.625% (1/160) |
| Social-Q75 REAL false AI | 42.500% (68/160) | 13.750% (22/160) | 10.625% (17/160) |
| Original AI recall | 97.500% (156/160) | 98.750% (158/160) | 99.375% (159/160) |
| Social-Q75 AI recall | 96.250% (154/160) | 99.375% (159/160) | 99.375% (159/160) |
| Original newly missed E43-caught AI | — | 1 | 1 |
| Numeric gates passed | — | 17/20 | 17/20 |

Relative to E43, E86 reduces original REAL errors by67/68 (98.53%) and social REAL
errors by51/68 (75%). Relative to E83, it rescues5 social REAL errors and1 original AI
miss, while losing1 original REAL rescue. Aggregate AI recall increases, but one
E43-caught GPT original remains missed. Separate per-source/condition AI retention
fails in the original condition; gains elsewhere do not offset that failure.

Remaining three numerical failures are all social-Q75:
- Pooled REAL FPR10.625% (17/160), requirement<=10% (at most16/160).
- Worst REAL source GP21.2121% (7/33), requirement<=20% (at most6/33).
- Covered-decision accuracy94.1935%, requirement>=95%.
These are close observed gaps, not proof that a single change will fix all three or
generalize. Together with the separate original AI loss, they reject E86. Keep fixed
AI cut0.07940196245908739 and REAL cut0.011505939625203613. No threshold, gate, label
or test membership changes to manufacture a pass. E20 serving and E43 reference stay
unchanged. E49 was not opened in this overnight chain; no independent-final claim.

### Completed cached-data lineage and costs

Earlier linear/multimodal candidates were rejected on TRAIN gates or DEV REAL/AI
retention failures. DEAR and the fixed supervised representation led to E83; near-zero
training loss was not treated as generalization. E79 feature work exceeded9412s across
attempts; do not relabel it as meeting its original single-run resource target.

E84B adds the matching social transform (1080 cap, then JPEG75) for all12,141 TRAIN
parents. This differs from source-resolution JPEG75. Native/source parity was checked;
no224px fallback. Four-view dataset:48,564 views. Feature extraction7296.694s.
E85 uses the fixed385→256→64→1 network, seed82, batch256, AdamW2e-4, weight decay1e-4,
100 epochs from scratch; training75.343s. No hyperparameter sweep or warm start.
E86 fits the450-column constrained head with the same worst-REAL objective and
per-view AI protection. Fit/runtime235.523s; all80 TRAIN checks and runtime replay pass.
Separate cached DEV scoring6.632s yields the rejected result above.

E87 selects64 Sony and64 Fuji SID long-exposure RAWs score-blind from official TRAIN,
before E86 results. Transfer3,376,363,519 compressed bytes; decoded archive members
4,851,194,880 RAW bytes. Whole large archives were not downloaded. E88 decodes all128
with the pinned camera-WB/sRGB8/no-autobright recipe in860.226s; no failed decodes,
internal duplicates or matches against151,996 protected references. Filename/camera
groups are not proven independent scenes; whole SID publisher remains research TRAIN.
E89 completes all128x4 DINO/CLIP/DEAR feature views in565.929s,64 chunks. First launch
failed before feature work because offline flags were set too late; same frozen recipe
succeeded with process-level offline flags. Old34-source parity and first-batch repeat
are exact0. Peak MPS allocation5,520,441,344 bytes. No SID classifier scoring, no new
DEV/final reads. All raw data and weights stay on the external volume.

### Next offline experiment, pre-result decision

E91/E92 code and focused role/replay tests already exist. Append all128 admitted SID
REAL parents to12,141 previous TRAIN parents:12,269 parents (7,674 REAL/4,595 AI),
49,076 views (30,696 REAL/18,380 AI), all4 conditions. Preserve every old feature row.
E91 keeps E85 architecture/seed/order/optimizer/100 epochs; refit only the TRAIN
input/latent scalers, then verify old E86 metrics and numerical replay. E92 keeps the
same zero450-weight constrained head objective. Protect all18,380 AI views, previous
populations and MIDD sensor checks; add SID pooled/camera checks and120 numerical
TRAIN checks across legacy/previous/expanded populations and4 conditions. Full49,076
runtime batch8 replay must pass before separately freezing any consumed DEV scoring.
Report E43 retention and paired E86 transitions. No E49 access on failed DEV.

This tests coverage from data already on disk, without a simultaneous consistency-loss,
rank, seed, margin, threshold or regularization sweep. Consistency research is only a
future lead; no new loss is claimed implemented. RR upstream REAL lineage is unresolved;
MIDD/DEAR research/noncommercial limits persist. WIFD/RawNIND remain diagnostic-only,
SIDD consumed DEV and other protected/final pools are not recycled into training.
Large disk volume alone does not establish usable independent training coverage.

Evidence: evidence/e86_fit.json, evidence/e86_development.json,
evidence/e86_progress_2026-09-13.png, evidence/e87_download.json,
evidence/e89_features.json and their immutable contracts. This consolidation does not
alter historical experiment scores, candidates or earlier receipts.


### 2026-09-14 — E91 registered offline before training

Verified all frozen ancestor/feature hashes and registered E91 for49,076 TRAIN views.
Contract SHA256 5660da2ba072eab7f5ce55a9713b33735aa29c6d353ef502934647b536635477.
No dataset/model/package download. AC power is now observed at100%; external disk
has347GiB free. Start the one fixed100-epoch cached-feature fit; no new feature
extraction, test-set access or recipe change. E92 is still unregistered pending E91.


### 2026-09-14 — E91 complete offline; register E92 next

E91 completed100 fixed epochs on49,076 TRAIN views in77.851s.
Weighted TRAIN BCE 0.695792679099→5.34038716076e-09.
Previous E86 four-view input replay: {'decision_changes_by_cut': {'0.011505939625203613': 0, '0.07940196245908739': 0}, 'max_score_error': 0.0, 'passed': True}.
Saved latent/coordinates exact; fixed-input batch8 error0.
Feature shape12269x4x64, artifact SHA256340cf7e305a18a2acf8abf587066d1af669d0bf8d1e7b822b59a84a29e593ac0.
Downloads0, new image inference0, DEV/final rows read0. Training loss is not a quality
or generalization result. Proceed to separately freeze the unchanged-objective E92
constrained head; require all120 TRAIN metrics, population/retention/runtime checks.


### 2026-09-14 — E92 frozen before fit

E92 registered for12,269 parents/49,076 views, all18,380 AI views protected.
Contract SHA25637a630a00a4693d1c5fa581dffcc8a8c17569784ca8fb196c43b66d197f7f972. No input/scaler/network refit in this stage,
zero450 head, unchanged E86 minimax objective/optimizer/thresholds. Offline fit starts
from verified E91. A TRAIN or runtime failure forbids DEV scoring.


### 2026-09-14 — E92 TRAIN/runtime pass; DEV registration permitted

Fit and full runtime checks completed in217.573s. All120 numeric TRAIN
checks pass across legacy/previous/expanded populations and4 conditions; all previous
and new SID population guards and AI/REAL retention pass.
Expanded TRAIN rates (fractions):
assigned_transport: REAL FPR0.00130310138129, AI recall0.998258977149
clean: REAL FPR0.00143341151942, AI recall0.999129488575
q75: REAL FPR0.000651550690644, AI recall0.998258977149
social_q75: REAL FPR0.00104248110503, AI recall0.998041349293
Runtime batch8 all49,076 views: {'batch_size': 8, 'decision_changes_by_cut': {'0.011505939625203613': 0, '0.07940196245908739': 0}, 'executed': True, 'max_score_error': 5.677039128237915e-07, 'minimum_ai_logit_shift': 1.9984014443252818e-15, 'passed': True}.
Candidate SHA2563a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35. No DEV/final feature use in fit, no downloads.
This is TRAIN success only. Register one separate consumed-E66 comparison now, using
unchanged E83 encoder caches and fixed cuts; keep all20 numeric criteria plus per-source
E43 AI retention, and report paired E86 changes. Do not read E49.


### 2026-09-14 — E92 DEV registration bug stopped before scoring; repaired pre-freeze

First DEV freeze attempt failed with ValueError: DEV coverage differs. The prepared
code passed640 E83 scored views as the manifest argument to validate_pairs, which
expects320 unique parent rows. No DEV contract or new score was written; E91/E92
TRAIN recipes, trained candidate and all predecessor artifacts remain unchanged.
Fix only the as-yet-unfrozen E92 DEV code: validate E86 and E83 view lists against the
actual parent manifest, then verify exact ordered parent/condition/SHA/role/source/label
identity. Add a successful-freeze integration test and rejection of reordered,
substituted or duplicate views. The failed-TRAIN cache-access guard is retained.
See evidence/e92_dev_registration_attempt1.json. Retry the same evaluation only after
tests pass; no threshold, dataset membership or model change.


### 2026-09-14 — E92 DEV contract frozen after pairing repair

Nine focused expansion/DEV tests pass in1.69s. Successful DEV registration validates
all prior receipts/cache identities and freezes contract SHA256f29fa1c917e07028cf879231c23d193bec14834d2aa2d49e8b448ec9a6019e90.
One640-view consumed DEVELOPMENT scoring pass is now permitted, runtime batches8
from existing features, no image inference/download. Scores must be locked before
metrics; all20 fixed gates plus source/condition AI retention and E86 paired changes.


### 2026-09-14 — E92 DEV scores locked; engineering suite858 passes

All640 cached views scored in6.674s; reference max score error0, decision
changes0, image reads0. Scores SHA2568de19298e0e62d982a6bae04ef7e922bcb12748cd68056b510ab832d49bb77ae. No metrics used before
locking. The full local test suite passes858 tests in16.46s with one existing
Starlette/httpx deprecation warning. No installation or dependency download.
Open the fixed report once; do not alter the locked candidate or scores.


### 2026-09-14 — E92 achieves20/20 DEV numeric gates; reference AI guard still fails

E92 is the strongest latest numeric result on the same consumed640-view E66 DEV.
Both original and social-Q75 now pass10/10 numeric gates,20/20 combined. This is not
100% classification accuracy, an independent final, or a full acceptance pass.

| Measure | E43 reference | E86 | E92 |
|---|---:|---:|---:|
| Original REAL false AI | 42.500% (68/160) | 0.625% (1/160) | 0% (0/160) |
| Social-Q75 REAL false AI | 42.500% (68/160) | 10.625% (17/160) | 8.750% (14/160) |
| Original AI recall | 97.500% (156/160) | 99.375% (159/160) | 99.375% (159/160) |
| Social-Q75 AI recall | 96.250% (154/160) | 99.375% (159/160) | 99.375% (159/160) |
| Numeric gates | — | 17/20 | 20/20 |
| New original AI misses vs E43 | — | 1 | 1 |
| New social AI misses vs E43 | — | 0 | 0 |

E92 rescues1 original REAL (N6) and3 social REAL (GP/IP/N6 one each) relative to E86.
No newly missed E86-caught AI or newly wrong E86-rescued REAL in either condition.
All prior E86 paired AI/REAL successes are preserved at the fixed AI decision cut.
Original binary balanced accuracy/accuracy99.6875%, AUC0.9998046875.
Social binary balanced accuracy/accuracy95.3125%, AUC0.9941796875.
Original automatic coverage98.75%, covered accuracy99.6835%, uncertain1.25%.
Social automatic coverage98.4375%, covered accuracy95.2381%, uncertain1.5625%.
The previously failing social gates now pass: pooled REAL8.75%<=10%, worst REAL
source18.75%<=20% (G4,3/16; GP is6/33=18.1818%), covered accuracy95.2381%>=95%.

The separate AI guard remains failed for the same original GPTIMG_431 parent
(e32:e77611a7fff7f7093b154b244e8c8359e00e93246043745e1056626b7259623b).
E43/E86/E92 scores0.08128664642572403/0.008283670495089164/0.00857941528582679;
fixed AI cut0.07940196245908739. This identity is reported diagnostically, not admitted
into TRAIN or used for a sample-specific rule. Do not change its label, threshold,
source or test membership. E43's original4 other AI misses are rescued, but those
gains cannot offset this one new miss. E92 therefore fails the complete DEV acceptance
contract; no E49/regression registration, serving change or promotion is permitted.
Candidate SHA2563a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35.
Evidence: evidence/e92_development.json, evidence/e92_dev_scores.json,
evidence/e92_fit.json, evidence/e91_representation.json and frozen contracts.

Scientific interpretation: the one fixed-recipe, already-audited128-parent SID coverage
extension improved this consumed DEV checkpoint. It does not prove the causal source
of improvement or unseen generalization. Only10 dependent REAL scenes and previously
seen AI families remain. Preserve E86/E92 immutable results and report both the new
20/20 milestone and the unmet AI guard together.

Next pre-experiment work is an offline TRAIN-only group/transport stability diagnosis
using existing cached DINO/CLIP/DEAR features. Verify which recorded groups are actually
disjoint before any cross-fitting; do not claim unidentified scenes/prompts independent.
Audit weak-reference AI stability across the four conditions without selecting or
training on the known DEV miss. Register a bounded protocol before any new model fit;
keep all E92 paired gains and the stricter E43 retention requirement visible. No
threshold/rank/seed sweep or sample-specific fallback; no further data download.


## 2026-09-14 — Critical audit of E92 evidence, bias and MVP readiness

### Scope and preserved conclusion

User requests an end-to-end critique, not another fit or a deployment. Preserve E92's
20/20 numeric milestone together with its failed separate reference-AI guard. This
review does not relabel a failure, change thresholds, refit a candidate or open final
images. Mobile-data restriction remains: no dataset, checkpoint or package downloads.
Only small primary-source text pages were consulted for methods/data/licensing.

Reproducible local audit: `ml/tools/audit_e92_readiness.py` reads locked score JSON,
existing manifests and hashes; outgoing sockets are denied. Receipt:
`evidence/e92_readiness_audit_2026-09-14.json`. Model inference, image reads and training
runs are all0. The audit reproduces the recorded metrics exactly, then independently
checks confusion counts, balanced accuracy, selective coverage/accuracy and pairwise
Mann–Whitney AUC, without relying on the benchmark helper for those formulas.
The candidate and score hashes match the frozen E92 report; TRAIN/DEV roles remain.

### What the20 conditions are, and where they came from

They are10 project acceptance criteria applied to original and social-Q75 views of
the same parents, not20 independent datasets or a100%-accuracy certificate. Git
history verifies GATES and both cuts are unchanged since b173d5b,2026-09-04, before
E92. The criteria appear in the earlier E45/E46 protocols and are inherited by E49
and the subsequent DEV screens. They are project engineering budgets; the records
do not establish an external certification standard or a market-specific cost study
justifying10% pooled REAL false positives or20% worst-source false positives.
The fixed AI/REAL cuts derive from the earlier E50 CAL selective-policy work, not
from E92's DEV results. Reusing them prevents post-result threshold manipulation;
it does not make E92 scores calibrated probabilities.

| Criterion, per condition | Requirement | E92 original | E92 social-Q75 |
|---|---:|---:|---:|
| Valid score coverage | 100% | 100% | 100% |
| ROC-AUC | >=0.90 | 0.9998047 | 0.9941797 |
| Balanced accuracy | >=85% | 99.6875% | 95.3125% |
| Pooled REAL false AI | <=10% | 0% | 8.75% |
| Worst REAL source/camera false AI | <=20% | 0% | 18.75% |
| Pooled AI recall | >=80% | 99.375% | 99.375% |
| Worst AI source recall | >=60% | 98.75% | 98.75% |
| Automatic decision coverage | >=80% | 98.75% | 98.4375% |
| Accuracy among automatic decisions | >=95% | 99.6835% | 95.2381% |
| Uncertain fraction | <=20% | 1.25% | 1.5625% |

Coverage and uncertainty are exact complements, so those two criteria count the same
constraint twice. Balanced accuracy is determined by REAL specificity and AI recall.
AUC and the thresholded measures share observations and are correlated. Conditions
are paired transformations, not independent samples. Passing20 criteria is useful
accountability but its count overstates independent evidential breadth if presented
without these relationships. The same applies to120 TRAIN checks: three nested
populations and four correlated views do not create120 independent experiments.

The separate zero-new-E43-AI-miss check was fixed before this result. It still fails
on the original GPTIMG_431 parent. E92 preserves all E86 decisions that were previously
correct and improves four REAL decisions, but that does not repair the E43-relative
failure. This stringent per-image retention requirement is an engineering choice, not
a universal definition of a better detector. A future product may use a prospectively
specified cost/non-inferiority criterion, but changing it after seeing this miss cannot
retroactively pass E92 or support a zero-AI-loss claim. Do not chase a one-image rule.

### Strong evidence and limits of leakage checks

The current12,269 TRAIN parents and320 DEV parents have zero observed parent-ID and
file-SHA intersections, including hashes from the640 scored views. All TRAIN rows
have TRAIN roles and all DEV parents DEVELOPMENT roles. Prior admissions independently
recorded exact/perceptual screening: E66 AI160 admitted,0 cross-role matches against
151,005 references; SIDD160 admitted,0 cross-role/cross-scene matches,223 same-scene
internal match pairs retained transparently as dependent; SID128 admitted,0 matches
against151,996 references. Retention, serialized model replay, fixed batch8 numerical
checks, score-before-metrics locking and failure receipts are real engineering strengths.

However, zero intersections is not proof of universal non-contamination. In this fresh
metadata-only audit, canonical pixel hashes exist for only639/12,269 TRAIN parents;
original-RAW hashes have no counterpart in DEV. No new image-level perceptual audit
was performed. Different encodings, semantically repeated scenes, shared prompts and
unknown source ancestry can evade exact identity checks. The large pretrained DINO,
CLIP and DEAR training corpora have not been shown fully disjoint from every evaluation
image. DEAR's card explicitly includes LSUN/COCO and aligned LDM/SD1.5 material;
unresolved RR upstream lineage prevents claiming arbitrary COCO data as unseen.
No affirmative direct TRAIN/DEV identity leak was found, but independence is unproven.

### Adaptation and bias risks

E66 has been reused by E70, E71, E83, E86 and E92. Labels/features were not added to
TRAIN, but failed results influenced subsequent representation, transport and data
coverage decisions. Freezing each individual run helps reproducibility; it does not
undo selection/adaptive overfitting across the overall research loop. E66 is consumed
DEVELOPMENT, suitable for diagnostics and rejection, not an independent final.
[Adaptive holdout reuse research](https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html)
explains why the distinction matters even without a gradient on test rows.

Dataset/source bias is plausible and not ruled out. E84B metadata recorded REAL7,034
JPEG/1 PNG/511 unspecified versus AI786 JPEG/3,809 PNG. Resizing affected classes
unequally. Fixed crops and shared JPEG augmentation do not erase previous compression,
texture, camera processing, semantics or resampling history. The old HISTORY statement
that fixed tiles make a biased dataset "perfectly safe" is too strong and is explicitly
superseded by this audit; a crop can retain these shortcuts. Neither removing EXIF nor
hiding a filename proves pixel-level dataset bias is gone.
[Fake or JPEG?](https://arxiv.org/abs/2403.17608) documents this failure mode in generated-image detection.
The model consumes visual features, not a known-source filename shortcut; source labels
still shape sampling/loss weights, and source fingerprints may be present in pixels.

The terminal E92 head minimizes worst REAL source loss subject to TRAIN AI constraints.
This is a deliberate tradeoff, not a proof of AI retention outside TRAIN. E91's final
TRAIN BCE~5.34e-9 after100 epochs means near-perfect TRAIN separation; it is neither
a generalization guarantee nor, by itself, proof of harmful memorization. A single
seed does not establish stability. Adding128 SID parents also changes the shuffle
trajectory, normalization and number of optimizer steps (190 to192 batches/epoch),
so this one result cannot isolate SID content as the sole causal explanation.

### Data freshness and sample dependence

Training is not exclusively old: the current manifest contains GPT Image2, Nano
Banana2.0, FLUX.2 Max, Qwen Image2.0 Pro, GLM-Image and Seedream5.0 source labels,
alongside older families. Such labels do not prove exhaustive coverage of current
provider versions. The E92 DEV AI population contains only80 GPT Image1 and80
nano-banana-local examples from already seen families; unknown prompts and ambiguous
full-generation versus editing provenance remain. It does not validate all current
models, unseen generators or local inpainting/localization.

REAL DEV is160 SIDD instances from10 dependent scenes and5 older phones (Pixel,
iPhone7, GalaxyS6 Edge, Nexus6, LG G4). SIDD is a2018 denoising benchmark; its provided
sRGB images are gamma-corrected RAW renderings without tone mapping, not necessarily
consumer phone JPEG processing. SID is also a2018 low-light RAW research source with
two cameras. Downloading these in2026 does not make the captures current. Modern HDR,
HEIC, denoise/sharpen, screenshots, repeated platform recompression, scans, illustrations,
AI-upscaled photos and mixed edited images need separately defined coverage.
[Official SIDD description](https://abdokamel.github.io/sidd/) and
[official SID description](https://cchen156.github.io/SID.html) support those dataset boundaries.

49,076 TRAIN views are12,269 parents under4 conditions, not49,076 independent images.
Similarly640 DEV views are320 parents, with REAL effective scene diversity far smaller
than160. Naive per-image binomial/bootstrap intervals would be too optimistic for a
population claim. There are no valid independent deployment confidence intervals in
the E92 evidence. Demographic fairness was not measured; no bias-free claim is supported.

### New diagnostic: concentration and fragility hidden by20/20

Reaggregating the unchanged locked scores by recorded REAL scene reveals:
- Social-Q75 scene002:6/15 false AI,40%.
- Social-Q75 scene006:7/19 false AI,36.8421%.
- Scene001:1/28; other seven scenes0.
Thus13/14 social REAL errors concentrate in two scenes. Worst-camera18.75% passes,
while worst-scene40% was never a registered criterion. This is a diagnostic finding,
not a post-hoc alteration of E92's20 numeric gates or evidence of the visual cause.

Social covered accuracy is300/315=95.2381%, just over95%. At unchanged coverage, one
additional error makes299/315=94.9206%, a failure. One additional error in G4 turns
3/16 into4/16=25%, failing the worst-camera20% budget. Margins are small; the result
should not be described as robustly clearing every requirement. Original0/160 REAL
errors is a finite dependent sample observation, not proof of zero population risk.
In each condition the one missed AI score is below REAL_CUT, so it is an automatically
wrong negative in the internal selective policy, not merely an abstention. A user-facing
"proven real" claim would be particularly misleading.

### Why market performance cannot be read from balanced accuracy

DEV is50% AI. If the observed social sensitivity99.375% and FPR8.75% transferred
unchanged to a population with10% AI, Bayes' rule gives AI-positive precision55.7895%;
at1% AI,10.2913%. These are algebraic scenarios, not measured user-population results.
They show why high recall and balanced-test accuracy do not establish trustworthy
accusations in a mostly-real photo gallery. Original observed FPR0 mathematically
produces100% precision in that formula, but that finite dependent sample rate must
not be treated as a zero deployment error guarantee. No calibrated p(AI) or error-cost
study supports marketing raw E92 scores as authenticity probabilities.

### Product and licensing verdict

There is a trained research candidate, not a drop-in standalone production model.
E92 combines pretrained DINOv2/CLIP/DEAR features, the earlier E43 head, learned
projections and E91 supervised features, then a450-column constrained correction.
Our contribution is the learned adaptation, protocol, data work and integration design;
it is not all encoders trained from scratch. The small correction.npz is not the whole
runtime. Cached640-view scoring6.674s excludes decoding and all encoder extraction.
No E92 native-image end-to-end serving parity, latency percentiles, concurrency, memory
budget or deployment device qualification has been established. The web still labels
E32 R1b; serve.py uses the E20/older research flow. E92 performance cannot be attributed
to that interface.858 software tests do not establish ML generalization; the pairing
bug that survived the earlier856 tests also shows test count alone is insufficient.

DEAR weights are explicitly CC BY-NC4.0 plus OpenRAIL use restrictions; the locally
pinned NOTICE also discloses upstream sources without explicit licences. MIDD carries
CC BY-NC-SA4.0, and other source restrictions persist. MIT/code availability does not
override weight/data terms. A free website is not automatically non-commercial if its
purpose is commercial advantage. No commercial clearance for this complete chain has
been established. Confirm rights or replace restricted components and revalidate.
[DEAR model card](https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints) and
[CC BY-NC4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/) are primary references.

A student research presentation can honestly show E92, exact population/limits, the
20/20 numeric milestone and the remaining AI failure. An E92 live research prototype
would require a separate scoped, non-commercial runtime plan and licence compliance;
it is not deployed by this review. A public or paid general-purpose detector is not
ready today. Demand, willingness to pay and a defensible advantage over alternatives
have not been measured. The suitable near-term positioning is an experimental visual
AI-evidence assistant, not an authenticity certificate or automated accusation system.

### Decision after critique

Do not start another DEV-driven sweep or fix only GPTIMG_431. Preserve E92 as a research
checkpoint and prioritize an evaluation/rights/runtime plan over chasing the headline.
Use the existing disk to inventory genuinely unexposed source/scene/prompt groups;
unused files from consumed publishers are not automatically fresh final data. Reserve
and lock a target-use benchmark before any further model choice; if no qualifying local
pool exists, document that limitation and postpone a generalization claim until access
conditions permit suitable acquisition. E49 was already consumed historically (E43
11/20, REAL FPR39.1% original/49% social) and is not a fresh final merely because E92
has not run on it. Those E43 numbers are not E92 measurements.

Before a new fit: define target use, acceptable false accusations/AI misses, realistic
prevalence and abstention costs; retain old contracts and prospectively specify any new
criteria. Use TRAIN-only source/scene/transport diagnostics and matched-processing
controls to test shortcuts; fit preprocessing/scalers inside each training fold.
Evaluate a compact baseline and E92 on the same legally usable locked populations.
Require representative, cluster-aware uncertainty and worst-group reporting. Probe
robustness with class-matched transforms and explicit mixed-edit scope. Then audit
portable runtime, accurate model naming, image privacy/retention, operational limits
and commercial rights. No conclusion from this review authorizes deployment or changes
serving, protected roles, thresholds, labels or the remaining AI guard.


## E93 registration — guarded E92 internship demo and offline source-shift diagnostic (2026-09-14)

The user explicitly requests a local internship demo and reliability work, with no
new downloads. Frozen E92 remains numeric20/20 with a failed E43 AI-retention guard;
this separate engineering wrapper neither changes weights/cuts nor promotes E92.
The exact prospective sequence and `e92-stability-v1` rule are registered at the top
of PLAN.md. E66 is consumed development; E65 is previously exposed REAL-only
WIFD/RawNIND diagnostic data. Neither can establish independent-final performance.
Freeze runtime/guard code, deterministic source representatives and file identities
before new encoder scoring. Report all input failures, every admitted parent, AI
coverage costs and scene/source errors; no threshold search or new fit follows.

Research motivation (primary text sources, no data/weights acquired):
- [Fake or JPEG? (2024)](https://arxiv.org/abs/2403.17608) demonstrates JPEG/size
  shortcuts. Applying the same later JPEG does not erase previous encoding history;
  matched processing, source-held evaluation and TRAIN-only controls remain necessary.
- [B-Free (CVPR2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf)
  motivates content/format alignment. No B-Free retraining or acquisition is proposed.
- [ReSIDe preprint (May2026)](https://arxiv.org/abs/2605.08574) shows that ordinary
  logit confidence can fail under image covariate shift. Its learned confidence
  method is NOT implemented here; our fixed disagreement/stability abstention is a
  separately measured heuristic, not a reproduction or guarantee.
- [Cross-dataset detector benchmark preprint (Feb2026)](https://arxiv.org/abs/2602.07814)
  reports unstable rankings across sources. Our inference: a limited DEV success
  cannot substitute for untouched source/generator evaluation.


### E93 initial startup failure and pre-score repair (2026-09-14)

The explicit local-only DINO loader initially omitted timm's official checkpoint
filter; strict state loading rejected positional-embedding1370/257 shape mismatch.
No image prediction, parity result or E65 score existed. Preserve attempt1 contract
and startup receipt; apply the same installed checkpoint_filter_fn used by the frozen
pretrained path, then re-register the adapter before any image scoring. This is a
loading repair, not model selection. Frozen E92 and all experiment helpers are unchanged.


## E93 result — exact native E92 parity and existing REAL source-shift diagnostic (2026-09-14)

Frozen contract SHA a8f46c5106d30f1057f4474bb70e2d3c2919626656ce1dbdc039d2de8d5d8216.
Locked166-view diagnostic score SHA83d387d181e2f1d77ed1d174a8310b2ded3205da637054577191d6312c66d95e.
No change to E92 weights/cuts, no fitting, no downloads, no E49 access. All83 admitted
E65 parents retained (67 WIFD,16 RawNIND); two fixed transports per parent. Outgoing
socket connections denied in the diagnostic process. Scores locked before reporting.
Initial loader failure and repair remain separately recorded above.

### Native pipeline parity

Before E65 inference, the SHA-sorted first parent per E66 source was fixed:7 parents,
14 original/Q75 views. In-memory social processing matched historical decoded pixels.
Every E92 and E43 score matched cached values exactly (max error0; allowed1e-5); no AI
or REAL cut crossing. This confirms native DINO3072/CLIP1536/DEAR1640 inference plus
frozen E92 mapping on this MPS environment, not new model generalization. CLIP3 and
DEAR9 microbatches/crop conventions remain unchanged. Seven paired image runs took
1.13–1.89s after17.59s weight initialization; cached E92's old6.67s/640-view score time
must still not be quoted as native-image serving latency.

### Raw E92 source transfer (all native inputs, before UI restrictions)

| REAL source | Parents | E43 original false AI | E92 original false AI | E43 Q75 false AI | E92 Q75 false AI |
|---|---:|---:|---:|---:|---:|
| WIFD | 67 | 11 (16.42%) | 2 (2.99%) | 13 (19.40%) | 1 (1.49%) |
| RawNIND | 16 | 3 (18.75%) | 0 | 5 (31.25%) | 0 |
| Pooled descriptive | 83 | 14 (16.87%) | 2 (2.41%) | 18 (21.69%) | 1 (1.20%) |

These sources are outside admitted E92 TRAIN but were previously consumed as E65
project diagnostics. WIFD scene linkage is unknown; RawNIND includes paired dependent
captures. Do not infer83 independent scenes or publish naive independent-view CIs.
There are no AI observations here, so no AI-recall/accuracy/AUC gate pass is claimed.
No fresh unseen-generator benchmark was validated. Preserve the new errors as future
diagnostic evidence; do not retune a cut or create source-specific exceptions.

### Strict E93 stability experiment and rejection as the default display

The preregistered rule required both E92 original/Q75 scores above the AI cut to show
AI; both below REAL plus no E43 AI indication to show no-clear; otherwise abstain.
On consumed E66: AI158 indications/2 uncertain (versus raw original159/160); REAL0 AI,
71 no-clear/89 uncertain. Total non-abstaining229/320=71.5625%, below the historical
80% target. Apparent error-free non-abstaining outputs do not establish perfect model
accuracy; the guard withheld difficult cases and used already-consumed development.
On E65:1 AI/58 no-clear/24 uncertain among83 REAL, showing that two consistent model
views can still be wrong. Stability does not establish authenticity or detect all OOD.

This strict positive suppression is rejected as the default user-facing decision
because it hides a correct raw E92 AI indication. Its rule, prototype API, inputs and
results are preserved unchanged; no score, threshold or E92 acceptance waiver follows.
The known GPT original missed by E92 becomes uncertain through reference disagreement,
not correctly detected AI; the original E43 AI-retention failure therefore remains.

## E94 — original-alert-preserving presentation, not another classifier fit (2026-09-14)

To satisfy the user's existing AI-retention instruction, the demo displays every raw
original E92 AI indication. The frozen E93 guard supplies a separate review flag:
when an original positive is unstable, show both the AI indication and a prominent
inconsistency warning. For original negatives, retain the guard's no-clear/uncertain
output. No image/source lookup, fitted confidence, new cut, recipe search or new score.
`demo_policy.display_result` has the invariant:

`display outcome == ai_signal` if and only if `original E92 score >= unchanged AI_CUT`.

This is not a claim that an alternative OOD classifier passed testing. It is a product
presentation constraint verified on all403 already-locked parent pairs plus boundary
and randomized unit tests. No original alert was added or removed; false alerts are
retained too. E94 does not overwrite E93 and does not claim FPR reduction from its flag.

- E66 AI159 indications/1 uncertain; one positive has a review warning. REAL71 no-clear,
  89 uncertain,0 indications. Non-abstention230/320=71.875%, below80%. The raw model's
 20/20 numeric result and failed E43 AI guard remain separate from this display policy.
- E65 REAL2 indications (one warned),58 no-clear,23 uncertain. A stable false positive
  remains. Source-shift reliability is improved operationally by honest bounds and
  uncertainty handling; generalization of the frozen classifier was measured, not newly
  optimized in this task.
- API input eligibility was measured separately: all160 original SIDD PNGs exceed12MiB;
  E66's160 AI originals pass. E65 accepts23 originals, rejects24 above12MiB and36 for
  geometry limits. The accepted23 yield1 AI indication/17 no-clear/5 uncertain. These
  selected subsets cannot inherit whole-benchmark accuracy or balanced class coverage.
  No silent resize, format conversion, omitted denominator or relabeling fixes this gap.
- Six targeted real HTTP cases match the locked expected policy responses, including
  E92 identity, browser-origin CORS and unstable-positive warnings. Times1.11–2.56s
  on this machine are integration observations, not production latency percentiles.
  Selection intentionally covers known outcome combinations and is not a new test set.

Current serving entrypoint: pixelproof.internship_serve:app. E93's research_serve.py
stays a frozen prototype. Input/error tests cover bad bytes, streamed oversized bodies,
geometry, transparency/animation, missing weights, failure cleanup, slot retention
past response timeout, explicit origins and schema rejection. No external API/model
calls, permanent upload storage or upload-based training are performed.

## Model 2 local evidence audit and evaluator critique (2026-09-14)

No classifier, extraction or training. Existing CocoGlide:512 edited images,512 valid
binary/nondegenerate same-geometry masks,512 linked authentic files with unique exact
file hashes. Metadata names an original `.png`; the compilation physically stores
`.png.png`. The initial audit's512 missing-link report was an auditor normalization
error, not missing images. Preserve attempt1 and the corrected exact-resolution audit;
never fuzzy-match ambiguous paths or rewrite source files.

Area bins:93 masks1–5%,193 masks5–20%,142 masks20–50%,84 masks50–100%; median15.974%.
No sub1% edited masks exist in this local subset, so tiny-edit sensitivity remains
unmeasured. Authentic decoding, near duplicates and scene/prompt/role separation are
still pending. E17/E18's first120 potential exposures and consumed CocoGlide source
status remain explicit; the remaining392 files do not establish an untouched final.

Review of e17_module2_first_measurement.py reveals two evaluation limitations:
1. The truth-dependent coverage filter retained only35/120 images, suppressing difficult
   small/poorly covered masks. A stride64 dense evaluator must retain/describe all rows.
2. `np.percentile(scores, 100 * (1 - mask_frac))` uses the true mask's area to choose each
   image threshold. Its IoU and random-area comparison measure oracle-assisted ranking,
   not a deployable threshold policy. Never present that IoU as Model2 product accuracy.

Retain tileAUC0.648/imageAUC0.721 and IoU margin+0.155 only with these sampling/oracle
caveats. Next plan: grouped pointer/provenance/decontamination audit, repaired coverage,
CAL-only thresholds, image-macro pixel AUC/AP, mask-area strata, random/centre baselines
and authentic false-localisation. Then one registered frozen-DINO dense linear head if
eligible groups and a baseline signal justify fitting. No E92 heatmap proxy, classic-
splice promise, new generation/download or fresh-final claim. PLAN contains the ladder.


## E95 registration — consumed owner gallery with current E92 (2026-09-14)

User requests inspecting the project gallery and justified development from its
failures. The gallery was previously moved to PixelProof Workspace/Samples. Freeze
the known210 REAL stills/206 unique byte parents using historical identity390e3c21…ac09;
exclude the separately protected reserve and MOV. Score fixed E92 original+Q75 once,
no gallery TRAIN/CAL or threshold tuning. Report every file, exact duplicates, camera/
format groups, UI eligibility and alert-preserving display; no independent-final or
AI-retention claim from this REAL-only sample. Full results remain private in ignored
ml/work/e95_owner_gallery; Git receives aggregate evidence only. No downloads or E49.
Decide a justified follow-up after locked diagnostic results; preserve E92 artifacts.


## E96 registration — support native24MP gallery inputs without changing E92 (2026-09-14)

Score-blind file-header audit found73/210 gallery images are5712x4284 (24.47MP),
rejected by the demo's16MP cap. All210 files fit12MiB. Add a general32MP local-photo
input profile, keeping12MiB, one concurrent request, dimension/aspect limits and all
format/animation/alpha/timeouts. Decode without resizing/recompression; frozen E92
features/cuts stay unchanged. This is input coverage work, not a gallery-trained model.
Before accepting: verify decode pixel parity for all210 images, old-policy rejection
versus new-policy admission,32MP overflow rejection, native HTTP replay of three newly
admitted24MP images plus the existing AI outcomes, and full tests. Keep E95's frozen
16MP admission report intact; report E96 separately. No filename/camera-based REAL veto.


## E95 result — current E92 on the consumed owner gallery (2026-09-14)

Contract SHA4753e1877ac8b89a9ec9440ee396eae5807b9198dce7040aa94e99f72dcb85a0;
private locked score SHA82223943ff259365c1aa9dc2d6679cc491dfc6ad0f4086e618b7d8c5dbee2b1a.
210 files,206 unique byte parents,412 native predictions; fixed original and social-Q75
views. Historical REAL labels and gallery identity retained; reserved still/MOV excluded.
No refit, cutoff change, downloads, E49 or fresh-final claim. Outgoing sockets denied.

| Population / measure | Original | Social Q75 |
|---|---:|---:|
| E92 false AI, all210 files |9/210 (4.29%)|18/210 (8.57%)|
| E92 false AI,206 unique byte parents |9/206 (4.37%)|18/206 (8.74%)|
| E43 false AI, all210 files |15/210 (7.14%)|28/210 (13.33%)|
| E92 new errors relative to E43, unique parents |7|2|
| E92 rescued E43 errors, unique parents |13|12|

Original-submission display:165 no-clear-signal/36 uncertain/9 AI indications;3 AI
indications are warned as inconsistent. Deduplicated display161/36/9. No-clear is not
an authenticity certificate; do not turn201 non-positive or165 no-clear outcomes into
an accuracy claim. The original-to-Q75 perturbation produces12 new false indications
and removes3, leaving18. The gallery is dominated by one phone and correlated scenes:
203 files from iPhone15Pro,4 iPhone16e,3 without camera metadata. These are grouping
metadata, not truth evidence or a permissible automatic REAL veto. Unique camera counts
are199/4/3. Two PNG application-screen inputs are not a natural-photo population.

The prior16MP API profile admitted137 files and rejected73 high-resolution originals.
Its admitted subset has7 E92 original false indications versus5 E43: the pooled source
improvement must not hide this reversal. E95 preserves that historical admission policy;
the changed E96 policy is evaluated separately. All rows, including screenshots and
rejected large inputs, remain in their original diagnostic denominator.

Posthoc visual review of three high-scoring examples found an application screenshot,
water ripples and a textured night street scene. This motivates distinct scope/context/
processing hypotheses, not causal attribution or sample-specific correction. No private
image, screenshot text or identifying filename is included in aggregate evidence.

## E96 result — input coverage repair and future paired regression protection (2026-09-14)

The header audit identified73 5712x4284 phone files (24.47MP), all below12MiB. A general
32MP local-demo decoder now admits210/210 versus137/210; this is +73 accepted inputs,
not73 newly correct predictions. Native RGB pixel equality is exact on all210 images,
including agreement with the old decoder wherever previously admitted. No resize,
JPEG save or E92 encoder/head/cut change. One request slot and all other input bounds
remain. Unit checks reject images above32MP and unsupported animation/transparency.

After source-hash registration and pixel parity, three new24MP HTTP examples were
chosen by file SHA, not scores. Along with three old AI outcome/warning representatives,
all six returned the exact frozen expected display. There is a false AI among the new
photo cases, explicitly retained as an inference-parity success and classification
failure. No generic AI-recall improvement is inferred from three selected examples.
No E92 alert policy changed; input handling cannot certify an unseen-source model.

New `development_regression.compare_development` and `tools/gallery_regression.py`
freeze a private732-view baseline:206 unique owner REAL+160 consumed E66 AI, two
conditions. Both classes, full paired identities, labels/sources/body hashes and finite
scores are required. Any new false REAL indication or lost reference-caught AI fails,
even if overall counts are unchanged. Failed CLI comparisons exit2. Reference replay
passes; synthetic single-REAL-score corruption fails. Applying this diagnostic to old
E43 versus E92 reveals9 newly wrong REAL views and1 lost AI view, correctly failing.
This is a consumed-DEV guard, never a TRAIN population, independent final or permission
to fit thresholds against the gallery. Future candidate artifact/scoring contracts must
be independently verified before these numeric reports are compared.

Raw E95 scores and the732-view baseline remain private in ignored ml/work; Git has
only hashes/aggregate reports. The local demo's scope now says photograph-like images,
not documents/application screens, following the observed screenshot failure; this
is a scope disclosure, not an automatic screenshot classifier. The nine original false
indications remain unresolved ML errors. Proposed next hypothesis and preservation gates
are recorded in PLAN. Validation:898 Python tests,11 web tests, lint/typecheck/build,
210 decode parity checks and6 targeted real HTTP checks pass.


## 2026-09-14 — E97: explain uncertain decisions and expose honest raw percentages

The user proposed an extra reviewer and arbitrary±5 score movement, and explicitly
requested percentages in the local UI. This supersedes the earlier no-percentage UI
preference, not the prohibition on unsupported probability claims. No threshold, model
weight, calibration, averaging or label rule changed. Frozen demo_policy/e92_demo and
their historical evidence remain intact. New demo_scores wraps the exact same paired
inference and alert policy; internship_serve now uses schema3 and bounded measured
original/social-Q75 scores with calibrated=false. Missing/nonfinite/conflicting scores
fail closed. Unscored tiny images return null, never a fabricated zero percentage.

The result card shows raw scores×100 to two decimals with the actual AI-alert cutoff
7.9401962459% (displayed7.94%), explains that50% is not the boundary and decisions use
unrounded scores. These are neither p(AI) nor correctness confidence. It explains the
three observed uncertain branches in plain Turkish. No arbitrary±5 adjustment, forced
binary claim, hidden AI alert, or real-authenticity certification was introduced.

Locked E95 diagnostic:210 owner-declared REAL files still yield165 no-clear,36 uncertain,
9 AI alerts. The36 split into12 original-negative/social-positive crossings,17 E92-both-
low/E43-veto cases and7 remaining E92 borderline cases. Categories explain the existing
policy, not causal texture/JPEG mechanisms. The prior E66 missed AI with an E43 warning
is why reference disagreement cannot safely be removed simply to make answers definite.

Presentation replay covers530 pairs (210 gallery+320 E66) with0 decision changes and
exact raw-score preservation. Seven targeted real HTTP scored cases cover five gallery
branches and stable/unstable AI; every score equals its locked value. One tiny input
returns no score. All8 real payloads pass the frontend parser. The gallery AI case is
an existing false indication, explicitly not a detection success. Reports:
evidence/e97_score_presentation.json and evidence/e97_score_http.json. Private paths,
filenames, individual gallery scores, image bytes and live payloads stay in ignored work.

Validation:909 Python tests pass in18.37s (one existing Starlette/httpx warning),13 Node
contract/SSR tests pass, lint/typecheck and Sites build pass. Local E92 API restarted
on8800; existing localhost:3002 preview remains. No dataset/weight/package download,
final-set read, model fit or model promotion. Current9 original gallery errors and36
abstentions are unchanged. Remote CI is not claimed verified by these local checks.

Research reviewed: Guo et al., ICML2017, https://proceedings.mlr.press/v70/guo17a.html,
separates neural scores from calibrated confidence. ReSIDe, May2026 preprint,
https://arxiv.org/abs/2605.08574, studies learned confidence from intermediate features
for selective synthetic-image detection under shifts; it does not establish success
for our E92 model. PLAN registers an eligible TRAIN/grouped-CAL audit and complementary
reviewer hypothesis, with calibration, risk/coverage and per-source AI/REAL retention
checks. No universal/no-error detector claim and no guarantee that calibration solves
OOD. Gallery-derived thresholds or a calibrated probability are not delivered here.


## 2026-09-14 — Home-network authorization and acquisition ledger reactivated

User lifts the mobile-data restriction and explicitly requests DATASETS.md logging for
every acquisition and its rationale. Ongoing docs now include that inventory alongside
PLAN/HISTORY/EXPERIMENTS. E98 begins with a metadata-only role audit of existing E92
TRAIN and old CAL manifests, plus targeted source/licence research. No model fitting,
new scoring, protected-final opening or new image-download success is claimed here.


### E99 MIDD coverage acquisition registration — 2026-09-14, before pixels

Preselect four previously unused sensor packages from the official MIDD listing:
ISOCELL_GN1, ISOCELL_HM3, OmniVision_OV64B and Sony_IMX766. These extend camera-pipeline
coverage beyond the four E72 sensors; they do not make a new independent publisher.
Inspect central directories and source terms first. Select64 publisher TRAIN/original
JPEGs per sensor by SHA256(E99|filename), no score/content-based choice or refill.
Exclude denoised partners and all upstream test rows. Cap selected bodies at4GiB,
32MiB per file,64MiB aggregate directory/header allowance,30GiB free disk reserve,
AC power and two workers. Require exact206/ETag ranges, ZIP length/CRC and stored SHA256;
no whole-archive fallback. Preserve every completed file/receipt for verified resume.
Initial role is QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN, never automatic TRAIN/CAL/final.
Whole MIDD collection retains prior research-TRAIN-only designation. Full native decode,
protected reference/gallery/DEV overlap and connected scene-group checks are a separate
registered admission step; no model may score selected images before that check.


E99 source inventory and selection completed:256 original TRAIN JPEGs across four new
MIDD sensors,3,323,883,735B. Contract0f67891068521533355ec39a90e3d4b396c14bb77b42fff9fc29a77745703729.
Two selection tests pass for order invariance, split exclusions, duplicate/path/capacity
rejection and no-refill overflow handling. Transfer now starts; no completion/admission
or detector-quality claim. Full acquisition record is in DATASETS.md as requested.


### E100 admission protocol — registered while E99 transfers, before image audit

Require complete256-file E99 receipt and source-code/hash validation before decoding.
Reuse fixed E65 canonical RGB/dHash/pHash fingerprint convention, native image decode,
100MP cap and224px floor. Preserve original JPEG bytes. Compare against the exact E88
protected reference closure plus128 admitted SID originals. All206 unique owner-gallery
body hashes must be present in that closure; all existing reserves/consumed E66 remain
protected. No protected pixels or classifier scores are read. Reject protected byte/RGB
matches or dHash<=4 AND pHash63<=4 matches; internal perceptual and same-sensor/capture-
second links form transitive groups. Reuse frozen E72 resolve rule/seed, propagate rejection
and keep one deterministic representative per surviving component, no refill. Mark scene
independence unverified. Pinned Pillow/NumPy runtime, one CPU decode worker, AC and30GiB
reserve. Downloaded256 is never treated as256 independent scenes or admitted examples.
Output separate E100 audit and TRAIN manifest only if checks complete; no E92 training,
CAL reassignment, AI-retention claim or demo promotion in the admission operation.


### E98 completed; E99 transferring; E100 admission prepared (2026-09-14)

Metadata-only E98 confirms12,269 E92 TRAIN parents (7,674REAL/4,595AI). The old C3 CAL
has4,534 rows and R1b CAL5,333; they overlap and must not be summed. Byte-only mapping
finds0 TRAIN/CAL collisions but resolves only5,652 inputs. A separate original-ID/body
lineage supplement resolves all6,619 E32-derived TRAIN parents, all to original TRAIN,
with0 shared source-scoped role/parent/available-scene groups against R1b CAL. No role
was changed; earlier CAL remains previously consumed model-selection data, not fresh
validation or new TRAIN. Missing scene metadata is not proof of independence.
Receipts: evidence/e98_role_audit.json and evidence/e98_lineage_audit.json.

E99 download is active with per-image CRC/length/SHA receipts. Added a DATASETS.md status
updater that counts exact contract-selected receipt paths and current sizes, not arbitrary
files. An initial ad-hoc filesystem glob encountered an AppleDouble metadata sidecar;
no data was changed. The updater excludes such sidecars/partial files by construction
and keeps incomplete transfer distinct from final completion. Original hashing occurs
in the downloader and will be repeated at admission.911 Python tests pass in18.05s
with the existing Starlette/httpx warning. E100 native admission code is prepared;
no image audit, new inference, candidate fit or model promotion has run yet.


### E99 download complete — 2026-09-14

All256/256 original MIDD JPEGs verified,64 per registered sensor. Stored source bodies
3,323,883,735B; exact compressed ZIP ranges received3,310,702,089B in the completed run.
No whole archives or denoised/test partners downloaded. All archive size/ETag checks,
member CRC/length checks and stored SHA256 receipts passed. Receipt SHA256
`6dab93cf4bddac6f670f927e8d53ff217efd9ba4c5bf80f6a1f9261f8f5bfacc`;
public summary `evidence/e99_download.json`. These256 files are still quarantined at
this transfer checkpoint, not256 admitted training examples or correct predictions.
E100 next repeats body hashing, native decoding and protected/scene-group checks.


### E100 native admission complete — 2026-09-14

All256 downloaded originals decoded successfully and retained native resolution. Against
152,124 protected/reference records, including every206 unique owner-gallery image:
zero byte/RGB/declared perceptual overlaps, zero internal perceptual links and zero
quarantined/removed images. All256 admitted to separate research TRAIN,64 per sensor;
137.118s. These heuristic checks do not establish256 independent scenes. No classifier
scores, new fit, calibration change or independent-final evaluation. Audit SHA256
0a403cb8c9acf755622f625c42e0a91a67b187ec77e63c999bbebbeabc9c031c;
TRAIN manifest SHA256c5abc769d1b430bba52e8e5ae8c7ae90fdeb1cbbc56c7a40d4bc01f458ee496e.
Public receipt: evidence/e100_audit.json. Local bodies remain under e99/images;
admission records under e100/training_manifest.json. Training availability is not a
model-quality improvement. The earlier quarantined snapshot describes acquisition only;
this later admission receipt supersedes that role status for all256 originals.


### E101 four-condition cache registration, before new features

Use all256 E100-admitted original JPEGs. Reuse exact E89 four-view crops and frozen
DINO3072/CLIP1536/DEAR1640 encoders; no global-crop change or new classifier score.
Two-parent immutable chunks;34-source old-clean parity and first-batch repeat<=1e-5;
90min/6GiB process memory guard. Verify native source hashes and all saved feature/role/
parent/condition bindings. Existing E92 source/map/cuts and demo policy remain unchanged.
This stage prepares1024 TRAIN views; only a later separately frozen E102 experiment may
fit a correction. It is the data-coverage experiment, not yet the learned-reviewer or
calibrated-probability proposal. Prior E92 losses and all preservation gates remain.


E101 frozen and started:256 parents/1024 TRAIN views, contract SHA2560a4dd9a88cb40810412647df6123aec01827ef6cd0635036577f74dd92020e9d.
Two targeted tests pass for exact E89 transform parity and rejection of CAL/changed-
original chunk identities. The local E92 API is temporarily stopped to avoid concurrent
GPU memory use; frontend remains available and the same API must be restored after the
encoder stage. No E92 weights/cuts/policy changed. Feature completion is not yet claimed.


### E102 correction protocol — registered before new TRAIN scores or fit

One data-coverage correction on the frozen E92 representation, not a new confidence
calibrator. Append the complete256-parent E100 cohort after its E101 cache; retain all
12,269 old TRAIN parents and4 conditions (50,100 views;18,380 AI views). No representation
refit or source/score filtering. Initialize450 delta weights to zero; baseline logits are
E92 logits. Reuse E81 worst-REAL-source epigraph loss and all-AI nonnegative logit-shift
constraints, preserving every reference-correct REAL binary decision. SameL2=.01,
SLSQP200 iterations/ftol1e-9, no hyperparameter sweep; CPU2 threads,1-hour fit budget.

Replay all frozen E92 TRAIN numeric reports before optimization. Preserve every previous
source/population and numeric gate; additionally require new-sensor pooled REAL FPR<=10%,
worst new sensor<=20%, combined REAL<=10% in each condition, all10 numeric gates on the
expanded population, solver success/constraint error<=1e-8, no lost E92-caught AI/new REAL
errors and full batch8 runtime replay<=1e-6/zero cut crossings. If training fails, lock
the failed result and do not score DEV/gallery. If it passes, separately register one
consumed E66/gallery paired comparison with E92 and E43 retention; no automatic promotion.
No existing CAL, DEV, gallery, E49 or final data in fitting. No calibrated probabilities.


E101/E102 engineering checkpoint:915 Python tests pass in19.26s (existing warning only).
Additional conditional-DEV test confirms failed TRAIN denies cache access. E102 uses
frozen E92 representation coordinates and zero delta initialization; all original AI
views protected relative to E92, old gate replay required before optimization. The
prepared consumed-DEV runner additionally requires zero lost E43/E92 AI and zero new
E92 REAL errors per source/condition. No current candidate fit/DEV score exists yet.


Final prepared-code checks:916 Python tests pass in19.81s, one existing warning.
Remote GitHub Actions34860324077 also passes for55afb6a4ce507e88c3c067db4ef45203756a5719.
E101 extraction continues with numeric/source parity intact; E102 still unexecuted.


### E101 complete — four-condition TRAIN cache, no quality claim

All256 new parents/1024 views encoded in867.662s. All34 fixed old-source checks have
exact0 score/CLIP/DEAR difference and0 threshold changes; first new-batch feature repeat
error0. Peak MPS driver memory5,520,441,344B. Saved shapes256x4x3072 DINO,
256x4x1536 CLIP and256x4x1640 DEAR. Cache SHA256
`a834e0517f9663b2f3856f17272e107f87148709095ccf300affe3a3a04e25f1`; contract
`0a4dd9a88cb40810412647df6123aec01827ef6cd0635036577f74dd92020e9d`. All128 parent chunks and
source/role/condition hashes verified. No new-image classifier score, CAL/DEV/final
read or fit. Source originals unchanged. Local E92 API restart initiated after extraction;
E102 fit registration now validates the completed cache and baseline before any fitting.


E102 fit frozen and started; contractSHA2565d5b0075f8b46a3fd7b7e5d93e06ebfa72ad4f230c843029b51f6394fb024185.
Complete12,525 TRAIN parents/50,100 views, including all256 new MIDD originals; no
CAL/DEV/gallery rows in fitting. Baseline E92 map is fixed; only one constrained450D
correction is fitted with the registered objective. Existing E92 API is again READY
on127.0.0.1:8800 with the same candidate SHA and raw-score UI. No demo promotion.


### E102 TRAIN correction complete — 2026-09-14

The registered frozen-map correction passes all TRAIN/source/numeric/retention and full
batch8 runtime guards in221.595s. All18,380 AI TRAIN view logits are protected relative
to E92 and no reference-correct REAL binary decision is lost. Candidate SHA256
`73ad43fbd74f1d1f94ca45f49c1ff662cfea89ed301af123057e7260277bd7de`. This only permits the conditional
consumed E66 comparison, not serving or an independent performance claim.

Important negative/neutral finding: E92 already had0/256 false AI indications in each
of the four newly acquired MIDD conditions, and E102 remains0/256. These new captures
expand audited pipeline coverage but did not expose the owner's gallery failure mode.
Do not call the zero new-source rate an improvement caused by this fit, or treat this
TRAIN-designated, already-used publisher as independent source-OOD evidence. More MIDD
volume alone is not established as a way to solve the gallery errors. Existing untouched
checks remain protected until the staged acceptance conditions are met.


E102 consumed-E66 comparison registered only after the completed TRAIN/runtime pass.
Contract SHA25612a92512c39614a9c4588739ef677f7c15819d8ef1f7b656dbe497872be8ba93;640 fixed cached views,
no image acquisition or encoder run. Same20 numeric gates, zero newly missed E43/E92
AI and zero new E92 REAL errors per source/condition. Lock all scores before metrics.
No gallery or E49 opening unless this comparison passes. This remains consumed DEV.


### E102 consumed DEV complete — numeric pass, full acceptance failure (2026-09-14)

One frozen candidate, no sweep. The locked 640-view comparison uses the previously
consumed E66/E83 cache; no new image read or independent evaluation. Reference replay
error is 0, and scoring took 6.386s. Public evidence: evidence/e102_development.json;
score-lock receipt: evidence/e102_dev_scores.json.

| Condition | E92 REAL false AI | E102 REAL false AI | E92 AI caught | E102 AI caught |
| --- | --- | --- | --- | --- |
| Publisher original | 0/160 | 0/160 | 159/160 | 159/160 |
| Social Q75 | 14/160 (8.75%) | 12/160 (7.50%) | 159/160 | 159/160 |

All 20 numeric gates pass. Both conditions retain every E92-caught AI and introduce
zero new E92 REAL false alerts. Social Q75 rescues two REAL observations, one SIDD:GP
and one SIDD:IP. Nevertheless, the original condition still misses one AI caught by
E43, so `passes_limited_dev_screen=false`, `consumed_regression_may_be_registered=false`
and `independent_final_passed=false`. This is the unresolved reference-retention deficit,
not an additional lost E92 detection. The candidate is preserved but not promoted.

Binary decisions and scientific selective metrics are not the demo guard's displayed
outcomes. Original selective uncertain rows fall 4 to 1; social rows remain 5. No claim
about fewer owner-gallery abstentions follows: E102 gallery/E49 scoring was not opened
because the staged DEV condition failed. The local E92 API remains ready and unchanged.
The 160 REAL observations comprise only 10 dependent SIDD scenes across five cameras;
AI observations come from two previously seen families. No universal, independent-OOD,
calibration or deployment-readiness claim follows from these results.

Next plan: investigate eligible TRAIN global-context and matched processing features
before registering another fit; retain the failed candidate and every retention rule.
The four new sensor sources remain TRAIN only. Download/admission totals are unchanged.


### E103 planned — joint worst-group objective on identical E102 TRAIN (2026-09-14)

Before another feature acquisition, code/receipt review identifies a narrower test:
E102's E81 loss directly optimizes only REAL risk; AI has nondecreasing-logit constraints
but no loss incentive. E102's complete TRAIN report retains 1/4/2/3 missed AI views in
clean/assigned/Q75/social-Q75, involving GPT-Image and Flux TRAIN sources. This diagnosis
uses existing aggregate TRAIN evidence, not the identity or features of the DEV miss.

Register one 450-dimensional zero-initialized delta from the frozen E102 candidate,
using exactly the same 12,525 TRAIN parents / 50,100 views, all four conditions and the
same representation. Objective: .5 times worst REAL source/condition cut-BCE + .5 times
worst AI source/condition cut-BCE + .5*.01 times squared delta norm. Parent-balanced
means inside each group. Use E68 group risks with the stronger E73 all-AI-confidence
constraints; E68's earlier decision-only constraints are NOT reused. Every E102 AI
logit stays nondecreasing and every correct E102 REAL binary decision stays protected.
SLSQP, 200 iterations, ftol 1e-9, CPU float64/two threads, final iterate only; no sweep.
All previous source/numeric/new-sensor gates and full batch8 parity remain required.

This is a local constrained objective experiment, not a reproduction or a universal
method. [Sagawa et al., ICLR2020](https://arxiv.org/abs/1911.08731) motivate measuring
predefined worst groups and warn that training group performance alone can generalize
poorly. Keep L2 fixed; no DEV-driven regularization search. The prior global-context
reviewer remains a later hypothesis, not an implemented feature of this experiment.

Only a full TRAIN/runtime pass permits one separately frozen consumed E66 comparison.
Require zero lost E43/E102 AI and zero new E102 REAL errors per source/condition, plus
all 20 numeric gates. E102 already preserves E92 detections and correct REAL decisions;
validate that predecessor relation. No gallery/E49 access after a DEV failure. No direct
promotion, calibration, private-gallery training, protected-final read or new download.
Freeze code and artifact identities before fitting; preserve the outcome even on failure.


E103 contract frozen before fitting: SHA256 `10f308eadc24602a2a7bf317e4ac432d013cc0edd8a16da77f73ab4a76f757db`.
All 921 Python tests pass in 18.28s with the existing Starlette/httpx warning.
One E103 fit started on AC power, two CPU threads, offline; no new image features or
downloads. Previous commit65e33c2 passed both GitHub Actions jobs (run34862651261).


### E103 TRAIN/runtime completed — 2026-09-14

One fit passed all TRAIN and full batch8 runtime gates in 195.537s; candidate SHA256 `c78bbeae1ecd9328a0955c27eb378d221d1b3243a9f7c3f9bbd94c6edaa4d174`.
No newly misclassified E102 REAL observations or lost AI; all 18,380 AI logit constraints
remain satisfied within the fixed tolerance. New MIDD sensor false AI stays 0/256 in
all four conditions. TRAIN confusion (FP/FN) by condition:
- assigned_transport: 0 REAL false AI / 7,930 REAL; 0 missed AI / 4,595 AI.
- clean: 2 REAL false AI / 7,930 REAL; 0 missed AI / 4,595 AI.
- q75: 0 REAL false AI / 7,930 REAL; 0 missed AI / 4,595 AI.
- social_q75: 0 REAL false AI / 7,930 REAL; 0 missed AI / 4,595 AI.

These are fitting-population results, not external accuracy. Register one consumed DEV
comparison next, without modifying the trained candidate or any acceptance gate.
921 local Python tests and both GitHub jobs pass for1269a66 (run34864358329).


E103 consumed DEV comparison frozen after TRAIN/runtime pass, contract SHA256 `ffefc0e08626a37bc45d5d9dad67f2bd01d4741b2be027fa78a34a8ed762f569`.
Score all 640 existing cached views once and lock them before metrics. No encoder/image
read, no fresh-final claim and no threshold/weight changes. Later stages remain gated.


### E103 consumed DEV complete — TRAIN improvement did not change binary DEV outcomes

The locked 640-view consumed comparison has exact E43 score replay. E103 retains all
E102-caught AI and all previously correct E102 REAL decisions. Original REAL false AI
is 0/160, social Q75 is 12/160, and AI caught is 159/160 in each condition: identical
binary outcomes to E102. All 20 numeric gates pass, but one E43-caught original AI is
still missed, so the full DEV screen FAILS. No gallery/E49 or final scoring, no promotion.
Candidate SHA256 `c78bbeae1ecd9328a0955c27eb378d221d1b3243a9f7c3f9bbd94c6edaa4d174` is preserved.

Original scientific selective uncertain rows increase from 1 to 2; covered accuracy
rises to 100% only on the 318 covered observations. This is NOT 100% overall accuracy,
a restored binary AI detection, or a measurement of the demo's guarded display. The
social selective result is unchanged. No claim of fewer gallery abstentions. Despite
zero missed TRAIN AI and improved TRAIN risks, this fit establishes no binary DEV gain.
Do not repeat objective/threshold sweeps against the same consumed item. E92 stays served.

### E104 planned — score-blind crop-role/cache audit

Code inspection finds that the three crop vectors (resized-center context plus two
texture crops) are pooled by mean/std. This is symmetric in crop order and cannot
retain which vector came from context. Further, the historical name "global" refers
to a short-side resize followed by a center crop/JPEG90; it is not a complete frame.
These are implementation properties, not established causes of the known model errors.

Before expensive encoder extraction, audit one SHA256(E104|parent_id)-first TRAIN parent
per label/source from the complete legacy 11,630-parent E71 cache, all three existing
conditions. Selection ignores scores. Verify bound chunk/raw/aggregate hashes; record
center/local-swap pooling roundoff and the change in separate center plus signed
center-minus-local coordinates. No head, image, DEV/gallery/final access or fit.
Synthetic tests verify pooling's information loss, local-slot invariance, center-crop
blindness to distant frame edges, score-blind TRAIN selection and invalid-input rejection.

This prepares a separate crop-role feature helper; it is not yet a trained reviewer.
The old three-condition CLIP raw cache can be audited without new downloads or encoders.
Complete social/new-cohort raw coverage is not established. Any future context model
must keep the same eligible population and both labels/conditions, register its recipe,
and pass retention/independent evaluation; a new feature is not an accuracy guarantee.


### E104 completed — recoverable crop-role signal, no classifier claim (2026-09-14)

All 30 hash-selected TRAIN label/source representatives / 90 existing views pass exact
raw-to-saved aggregate replay. Swapping center-context and one local crop changes none
of the pooled float32 features (maximum difference 0), while separate crop-role
coordinates change on every selected view (minimum L2 11.0002). Center-minus-local L2
ranges 5.9195–19.2166, median 12.6098, in unnormalized CLIP coordinates; these are feature
differences, not accuracy, confidence or significance statistics. Receipt:
evidence/e104_context_audit.json, contract SHA256
ef9b38d42dbb14c7a62b1ae7c806a603d0dbde03a1c42c244b040a538d1405a7.

The new context_coordinates helper retains resized-center features and their signed
contrast to the mean of the two local crops. Historical encoders/aggregates and all
models stay untouched. This supplies a concrete next representation hypothesis rather
than another objective sweep, but cannot establish the cause of the DEV/gallery errors.
No image, encoder, detector, new fit or protected-data operation occurred in E104.
Only 30 parent chunks were inspected; complete current-cohort raw availability still
requires inventory. This is not a new full-frame model or a trained second reviewer.


E103/E104 engineering verification: full Python suite925 passed in16.89s; one existing
Starlette/httpx deprecation warning. No web/serving change. E103 code commit1269a66
passed both GitHub jobs (34864358329). Final evidence/documentation commit follows.


### Remote dual-model research programme registered — 2026-09-14

The current instruction authorizes research/acquisition/implementation for both modules
without manual computer interaction. This supersedes the historical Model2-waits-for-E49
execution order, while retaining all scientific data/retention gates. PLAN.md now contains
the primary-source comparison, present gaps, prospective finite-sample proof targets,
Model2 evaluation repair, and P0–P7 implementation ladder. Broader targets are aspirational
project criteria, not an industry certification or an amendment to frozen20-check contracts.
E105 selects512 author DiffSeg30k TRAIN pairs by fixed identity hash before images/scores;
E106 audits existing raw TRAIN caches; E107 evaluates only120 historically exposed CocoGlide
parents after evaluator repair. No new model selection on protected final data.

## 2026-09-14 — E106/E107/E108: evidence repair before model expansion

E106 verified **all 11,630 legacy TRAIN parents / 34,890 views**, including each raw
CLIP chunk body, internal array digests and exact saved mean/std aggregation (489,358,434
bytes). Crop-role coordinates are finite throughout. The full E103 population still
requires **15,210 raw views** (legacy social views and all four views of appended cohorts)
before the proposed full-population representation comparison. No classifier was fitted,
no DEV/final image was read, and this inventory is not an accuracy improvement.
Receipt: `evidence/e106_context_inventory.json`.

E107 preregistered the historical first120 CocoGlide edits and their exact metadata-linked
authentic parents. The old crop128 checkpoint is SHA256
`fd504af03f223557d42f2fce6bbd5e2e93493a34adba15a884828a439446f11f`.
Replaced the old truth-dependent example filter and percentile threshold with full native
128px tiles, stride64, final-edge coverage, overlap mean, and fixed diagnostic threshold0.5.
All **120/120 pairs** completed (2,160 tiles, 11.58seconds). No mask-area oracle, image
resizing or tile cap; tiny masks retained. This is a consumed single-generator diagnostic.

Image-macro pixel AUC **0.57238** (parent bootstrap95% [0.53656,0.61085]), AP **0.36330**,
IoU **0.19963**. Center-only baseline AUC **0.72208**; crop128 IoU difference versus center
**-0.01076** (95% [-0.05921,0.03702]), versus always-positive **-0.07432** (95%
[-0.10291,-0.04814]), versus seeded random tiles **-0.00091**. Authentic controls have
mean **27.40%** flagged area; **65.83%** exceed5% flagged area. Image detection AUC at fixed
p95 aggregation is **0.61521**. These fail the proposed Model2 requirements; the old tile
map must not be presented as validated localization. The center baseline result exposes
spatial bias in this benchmark and motivates semantic/random-mask controls. These numbers
are not directly comparable to E17's filtered35-image tile metrics. No web model changed.
Full strata and all baseline comparisons: `evidence/e107_model2_replay.json`.

E108 adds exact one-sided binomial limits and an evidence-assumption gate. Consumed,
dependent or incomplete observations cannot certify an error-rate target. The E103
0/160 original REAL errors would have a **1.855%** one-sided95% upper limit even under an
unjustified iid assumption; it does not establish FPR<=1%. Its160 REAL photos represent
only10 scenes. A separate scene-any-error endpoint with0/10 has a **25.89%** conditional
iid upper limit; this is explicitly not a photo FPR. Social12/160 has an11.868% conditional
upper limit. All are descriptive because these data were consumed and independence is
unestablished. Zero errors require299 fresh iid units for one1% claim at95%, or597 for
20 prespecified simultaneous one-sided claims using Bonferroni. The old20 gates mean
2 conditions x10 engineering criteria, not20 independent test datasets. Receipt:
`evidence/e108_evidence_limits.json`. E92 remains the serving model.

### E110 preregistration — missing raw-context feasibility

Before full extraction, freeze one SHA-first TRAIN parent per cohort/label/source,
legacy social view and each appended cohort's four conditions. Reuse the pinned local
CLIP encoder, exact historical batch3/preprocess, compare reconstructed aggregates to
existing SHA-bound caches at max absolute error1e-5, and retain raw ordered crop vectors.
No subset head fitting, classifier scores, DEV/gallery/final inputs or downloads during
encoding. Bound runtime20minutes and GPU allocation6GiB; stop the known demo API for
memory and restore the same E92 after the probe. Estimate full extraction cost, then
register a full-population run separately. Protocol/code: `e110_context_probe.py`.

### E105/E109 — completed acquisition, unexpected task mixture

Downloaded the full preregistered512 DiffSeg30k TRAIN image/mask pairs. E109 native
schema audit:512 decoded,201 partial/224 full-positive/87 empty masks, zero internal
body/RGB duplicate groups. No mask>127 conversion or refill. A single arbitrary pair
was inspected before the full audit and found empty; this is disclosed in its receipt.
The dataset combines cases that cannot share one within-image localization metric.
Keep quarantine; missing authentic ancestry blocks independent role assignment and
empty masks do not imply authentic photos. Details and rationale are in DATASETS.md.

### E110 source-path failure and E110B engineering revision

The frozen E110 run reproduced two selected legacy-social CLIP aggregates exactly, then
stopped before the third image with KeyError(path). Some native legacy records use a
source_key rather than a materialized path. No raw cache/report/model fit was completed.
Preserve E110 code/contract; E110B resolves the same parent identities through the already
verified E84 source index, preserving the same40 parents/70 conditions, selection salt,
encoder, tolerance and no-classifier policy. Bind the source index/receipt/code and assert
that the selected parent order is identical before encoding. No error-driven refill or
changed metric. This is an engineering failure, not negative representation evidence.

### E110B completed — raw crop-role extraction is feasible

The same40 selected TRAIN parents /70 missing views completed in36.10seconds including
encoder load. All historical CLIP mean/std aggregates reproduced **exactly (max error0)**
across the four cohorts. Peak MPS allocation2,194,358,272bytes (~2.04GiB).
Raw cache SHA256 `79043f2a4b6144b88acf968069483fbf247a56fd9a8de16077e89c7a7d074094`.
The rough15210-view estimate is7,843seconds (~2.18hours), source-dependent and including
amortized loading. This verifies an implementation path; it does not measure improved
classification or authorize fitting on a subset. Full ordered extraction, cache admission
and a separately preregistered context-feature head remain next. Restored E92 demo API
after the GPU probe. Receipt: `evidence/e110b_context_probe.json`.

### E111 preregistration — Model2 overlap before role assignment

Audit all512 CocoGlide edits and512 linked authentic originals plus all512 E105 images
against the9 frozen reference files used/extended by E100. Canonical body/RGB or joint
dHash<=4 and pHash63<=4 overlap, also internal cross-parent comparisons. No classifiers
or masks for selection, no protected image decoding, and no automatic role admission.
A match-free derivative cannot establish original COCO/prompt/generator independence.

### E111 completed — 45 CocoGlide parent groups require overlap quarantine

All1,536 selected images decoded; compared with152,380 existing reference fingerprints.
Found252 joint perceptual match observations affecting **45 CocoGlide parent groups**:
14 among the historical first120 and31 among the other392. These were joint dHash/pHash
matches, **not exact body/RGB duplicates**; conservatively flag each complete parent and
its derivatives pending review. No DiffSeg30k image matched this snapshot and no internal
cross-parent matches were detected by the frozen criterion. Neither negative result
proves unseen COCO/prompt ancestry. Local authentic pointers use numeric compiled names,
so original COCO identifiers are not recovered by filename alone. No training roles
assigned; no fresh evaluation claim. Runtime14.82seconds. Receipt:
`evidence/e111_model2_lineage.json`. These findings reinforce that E107 is consumed
research diagnosis and its bootstrap intervals are not independent deployment evidence.

## 2026-09-14 — E112 preregistration: complete crop-role feature extraction

Implement full15,210 missing CLIP raw views for12,525 admitted TRAIN parents; concatenate
with immutable34,890 legacy views to preserve all50,100 conditions/parents. Exact E110B
operations and source-key materialization; verify source bodies, historical aggregates,
parent/role bindings and every resumed chunk. Per-parent atomic NPZ plus immutable receipt,
source/body/array hashes; no refill or score-driven selection. Four-hour execution bound,
6GiB MPS ceiling and existing AC/disk guards. No classifier fit or independent performance
claim. The subsequent paired context/control fit remains separately registered.

### E112/E113 unattended execution — 2026-09-14T17:39:24+00:00

Started the fixed full E112 extraction; temporarily stopped the verified E92 API for GPU memory. Durable chunks and stage logs are local. This is feature extraction, not a completed model improvement.

### E113 implementation — fixed paired comparison queued after E112

One all-TRAIN StandardScaler/PCA128 per branch (std floor1e-6, variance floor1e-12,
randomized seed113 power3); no held-out fitting statistics. Compare1536-dimensional
pooled mean/std inputs with1536-dimensional center/center-minus-local-mean inputs using
the same latent capacity. Retain frozen E103450 coordinates and logits, fit578 delta
coefficients from zero. Reuse E103 joint worst-group loss, L2.01, SLSQP200/ftol1e-9,
CPU2, all-AI nonnegative logit shifts and correct-REAL decision constraints. One-hour
budget per branch, no sweep. Run all historical/expanded gates and complete batch8 replay;
retain both successes and failures. The executable registers actual feature/artifact
hashes only after E112 passes; it cannot fit an incomplete cache. No DEV scores or
new performance results exist at this implementation checkpoint.

The unattended runner logs each stage and appends completion/failure summaries to the
English records. Code checks confirm deterministic saved representation/batch replay,
no use of evaluation statistics, corrupt/role-swapped resume rejection and stopping on
child failure. API E92 is temporarily stopped for computation and restored in the runner's
finalizer. Training results will appear as later dated entries, not be fabricated here.

## 2026-09-14 — E114 implementation, no DEV execution yet

Added the paired consumed-DEV gate, exact predecessor/transport identity checks, CLIP
raw/aggregate parity and all-source individual retention comparisons. Six focused tests
pass, including rejection of failed/incomplete/nonfinite TRAIN permissions and equal-count
swapped AI/REAL errors. Registration must occur after E113 completes; code preparation
does not access DEV pixels or fit scores. E112 continued with zero aggregate error at
10,300 parents. Its eventual result remains recorded by the existing unattended runner.

### E115–E117: recovered Model2 ancestry reveals hidden cross-role overlap

E115 found the author table/prompts/licences with167,412 metadata bytes, no images.
E116 then acquired the123MB original package and joined all512 local triples by exact
canonical authentic/edit/mask pixels. This recovered true COCO val2017 identities rather
than assuming compilation row numbers correspond to original filenames. All512 original
licences and prompts are now retained locally; licence counts are in DATASETS.md.

E117 metadata-only comparison against the frozen DDA-COCO protected manifest finds
506 matching COCO originals, including every historical first120 pair. The protected
manifest has7 records per matching parent (original plus generated variants). This adds
461 flags beyond E111's45 perceptual matches. Therefore E107 remains a consumed diagnostic
and CocoGlide is unsuitable for newly admitted Model2 TRAIN/CAL under the current role
policy. Do not describe this as demonstrated historical weight contamination; exact
parent overlap does not by itself establish what the crop model's weights saw.

An initial E117 parser accepted generated member paths but rejected the protected REAL
members (`val2017/<id>.jpg`). Before any report was written it was extended to both exact
published layouts; parent/member numeric IDs must still agree, and non-val2017 paths
are rejected. No threshold, cohort selection or protected image content changed.

Implementation checkpoint:955 Python tests pass with one existing Starlette/httpx warning; pip check and whitespace checks pass. E112 remains active, at11,800/12,525 parents with zero aggregate error. This checkpoint does not report a completed fit or DEV pass. E114 awaits the complete paired TRAIN reports.

## E118/E119 preparatory hypothesis

CocoGlide exact ancestry overlap motivates a traceable TRAIN-only generation pilot, not reusing protected COCO parents. SD1.5 inpainting is selected for a bounded512px Mac feasibility probe, not novelty or universal coverage. Pin weights, original roles, masks and generation configuration; compare authentic/traditional-edit controls with generated composites and preserve raw output for out-of-mask drift measurement. No detector result exists at this preparation stage. Keep current Model1 dependencies immutable during E112/E113/E114.

E118 asset acquisition completed in344.59s; all four safetensors match publisher SHA256 and small files match Git blobs. E119 preparation completed16/16 groups with10.10–16.99% mask coverage, preserving original TRAIN roles and source hashes. Tests cover exact unchanged composite background, identical masking of AI/non-AI edits, deterministic selection, rejection of evaluation-role originals, corrupted asset bodies and dependency/API failure gates. Full suite962 passed; no detector/generator scores at this preparation checkpoint. Fixed generation is queued after Model1 completion; no claim of model improvement from data preparation.

E119 environment completion before generation: isolated pip check detected missing importlib-metadata even though class import succeeded. Added only importlib-metadata9.0.1 and zipp4.1.0 (38158wheel bytes), checked publisher hashes, recorded an isolated hash-locked dependency file and supplemental receipt. Isolated pip check now passes; all frozen E119 inputs remain unchanged. All16 classic controls preserve every out-of-mask pixel exactly;97.89–100% of their masked pixels change;3/16 masks cover the image center. No detector/generator inference performed. CI for6b399ac passed both jobs (34886187832).

### E112/E113 unattended execution — 2026-09-14T19:32:52+00:00

E112 completed 12525 parents / 50100 TRAIN views; maximum aggregate error 0.0. Registering the prespecified paired E113 fit; no DEV/final opened.

### E112/E113 unattended execution — 2026-09-14T19:40:03+00:00

E113 paired TRAIN fits completed. {"ordered_context": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}, "pooled_control": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}} These are TRAIN guards only; serving/promotion and independent evaluation remain unchanged. Full reports are in evidence/e113_context_fit.json.

### E112/E113 unattended execution — 2026-09-14T19:40:23+00:00

Pipeline final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E112/E113 unattended execution — 2026-09-14T19:41:02+00:00

Started separately registered E114 consumed-DEV scoring for pooled_control, ordered_context. Frozen640-view comparison; no training or final/gallery access.

### E114 automatic consumed-DEV result

{"ordered_context": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}, "pooled_control": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}}

Full report: evidence/e114_context_development.json. This is consumed development; no independent proof or serving change. Only passing branches may proceed to a separately registered gallery regression. Failed branches require a new hypothesis, not relaxed gates.

### E112/E113 unattended execution — 2026-09-14T19:46:07+00:00

E114 finalizer E92 restoration ready=True.

### E119 one-shot execution — 2026-09-14T19:46:11+00:00

Starting registered16-parent E119 local inpainting engineering probe after Model1 completion. No detector fitting or evaluation data. E92 is temporarily stopped for GPU memory.

### E119 automatic engineering pilot result

{"contract_sha256": "80721e3928d94f618ce4525e50f7f3b51de04bdd15ec31f0c36217120ea85e80", "detector_scores": 0, "limits": "16 already-used TRAIN parents, one old editor, correlated sensor scenes and fixed prompt. This is an engineering pilot. No calibration, independent evaluation, universal claim or automatic training/serving admission.", "mean_raw_background_changed_fraction": 0.9998254416167689, "parents": 16, "passed": false, "passed_parents": 1, "peak_mps_bytes": 4143857664, "promotion_allowed": false, "result_sha256": "bba363a9c710dd094c09e9717a648b4a5b3024acd7d4c3002a482e227ef61abf", "seconds": 273.0622565409867, "state": "E119_inpainting_engineering_pilot_complete", "training_admission": false}

This measures generation feasibility only. Spatial-head training requires a new protocol; no independent detector evidence or serving change.

### E119 one-shot execution — 2026-09-14T19:51:35+00:00

E119 pipeline final state: {"E92_restored_after_pilot": true, "failure": null, "state": "complete"}

## E114 rejection and E119 numerical failure review

Complete E113 training became error-free in all four views, yet both fixed branches reproduce the E103 DEV binary counts and fail the original E43-AI retention check. Thus this context extension is not a validated improvement, despite20/20 numeric DEV gates. E119 has15 nonblank/safety failures; only parent0 passes. Invalid-value cast warnings precede the black outputs, so semantic safety conclusions are unsupported for those15. E120 is an instrumented fixed two-parent /three-branch numerical probe; finite tensor checks must fail before rendering/safety evaluation. Registered full correction remains conditional on measured probe results.

E120 completed the fixed precision comparison: fp16+sliced reproduces a UNet nonfinite failure on parent1 before rendering/safety; fp16+SDPA and float32+sliced both pass parents0/1. Follow the prespecified preference: E122 registers complete16-parent fp16+SDPA replay, unchanged weights/images/masks/seeds/prompt and all acceptance checks. Keep finite boundary guards and safety checker; no downloads or detector fitting. E121 initially stopped on4278 lowercase legacy train roles; preserved frozen code/contract and ran E121B with the existing upstream case-insensitive TRAIN rule. E121B reproduces all DEV guards: neither branch changes any of640 AI-cut decisions; each changes two REAL-cut decisions. All12525 TRAIN dimensions are present. The center window spans~28.04–76.56% of source area; this describes only the center branch, not union-of-crops coverage or proof of the missed example cause.

E121B source-level audit confirms all12525 dimensions, reproduces E114 guards exactly and finds zero binary AI-cut changes between each E113 extension and E103. Registered E123 to probe uncropped available-frame features without fitting. All160 source/condition cases must preserve old center pixels and CLIP<=1e-5 and repeat full-frame vectors<=1e-5. Same JPEG90 preprocessing on old/new semantic views; geometric warp distortion and source-aspect confounding remain explicit limitations.

### E122 complete numerical replay result

{"branch": "fp16_sdpa", "contract_sha256": "353836702bcf414116aa3f2669de92a90081ba1f930b1cd85858afe3c8746789", "detector_scores": 0, "limits": "Engineering correction of numerical failure, not improved detector performance or a semantic-quality audit. Only after16/16 pass may a separate TRAIN-only spatial learning experiment be registered. All original research-only licences and ancestry roles remain.", "mean_valid_raw_background_changed_fraction": 0.9994174467463207, "parents": 16, "passed": false, "passed_parents": 11, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "report_sha256": "cb0c713de246cb20e8d4822948aa5dd34303d845d85deefdffeb6fb9c54cd373", "seconds": 487.82744045800064, "state": "E122_complete16_numerical_replay_finished", "training_allowed": false}

Original E119 remains failed; this is a separately registered correction. No detector performance or independent evidence is claimed.

E122 confirms why the two-case numerical probe was insufficient for full-pilot acceptance: all16 finite pipelines complete, but5 outputs are filter rejected and only11/16 pass. Preserve both original E119 and corrected E122 failures; no semantic interpretation or bypass of the filter, no training on black outputs. E123 passes160 source/transport feature operations exactly. Next E124/E125 protocol changes field of view with a matched center-only capacity control, not optimizer/cut selection; all original TRAIN and old AI-retention requirements remain.

### E124/E125 unattended execution — 2026-09-14T20:33:54+00:00

Started the fixed full E124 extraction; temporarily stopped the verified E92 API for GPU memory. Durable chunks and stage logs are local. This is feature extraction, not a completed model improvement.

### E126 prepared, not yet registered — 2026-09-15

Hypothesis: the fixed full-frame representation may transfer differently from its equal-capacity center control; no result is assumed. Code is ready for both E125 branches subject to complete TRAIN/runtime permission. The consumed 640-view comparison preserves all E114 gates, separately replays old center and duplicate full-frame features, locks branch scores before metrics, and keeps E43/E92/E103 individual AI retention. Known limitations: repeated DEV selection, 10 correlated SIDD scenes and limited seen AI families. E124 feature extraction is still active; E125 has not produced candidates and E126 has not read/scored DEV.

E126 implementation verification: 991 Python tests passed; compile/whitespace checks passed. Started the one-shot dependent runner, currently waiting for E124/E125. This is operational readiness, not an evaluated detection result.

### E127 pixel audit design — 2026-09-15

Prespecify pixel-error magnitudes and mask-boundary gradients to interpret the E122 background-change statistic and investigate compositing shortcut risks. Selection is fixed by the previous E122 acceptance receipts, not by these new measurements. All 16 originals/classical controls remain counted; rejected black safety placeholders will not enter generated-content metrics. Seven synthetic tests validate exact boundary counts, unchanged backgrounds, one-level changes and invalid-mask rejection. No pixels have been measured by E127 yet; freeze the input/metric contract before execution.

### E127 completed — pixel magnitude and compositing controls (2026-09-15)

Frozen contract `8b5e1efb01339e92d7450d7ae5e04fc9a24c8bc580c3bc362a33cdae81bf082d`; all 16 attempts accounted for, 11 accepted pairs measured and 5 safety-filter/blank placeholders explicitly excluded from generated-content metrics. CPU-only audit finished in 1.73s. All 16 classical controls preserve background exactly and have zero AI targets; all 11 accepted composites reproduce the exact registered hard-mask operation.

On the matched 11-parent subset, raw generator outside-mask MAE is 0.038381 in [0,1] (about 9.79/255 channel levels), with per-parent MAE range 0.015261–0.115276. Outside changed-pixel fraction remains 99.9417%, exactly replaying E122. These measure different properties: many changed pixels do not by themselves imply large or semantic changes, while the measured magnitude still prevents labelling raw background untouched. Composite outside MAE is exactly 0.

Mean mask-crossing RGB gradient is 0.030659 in authentic inputs, 0.042073 in classical controls, 0.028218 in raw generator outputs and 0.044219 in hard composites. Composite gradient exceeds the corresponding classical control in only 5/11 pairs. Classical controls therefore also expose the boundary shortcut; these small correlated-sample summaries do not establish separability, semantic quality or actual learned detector bias. Statistics are equal-parent summaries, not independent population estimates.

Decision: retain classical-edit negatives and require boundary-versus-interior localization evaluation in a future spatial-head protocol. Keep raw outputs diagnostic, and preserve exact mask/provenance/encoding matching. The failed E122 16/16 gate remains failed; no output is newly admitted to training and no safety filter is disabled. No detector score, GPU operation, download or serving change. Aggregate receipt: `evidence/e127_pixel_audit.json`; per-case measurements stay local.

After E127, all 998 Python tests passed. E124 checkpoint: 4,050/12,525 parents extracted, pilot replay error 0; E125 not yet completed and E126 still waiting. No new Model1 accuracy result exists at this checkpoint.

### E128/E129 proposed input-preparation ablation — 2026-09-15

Use the exact16 original E119 TRAIN research parents, without selecting replacement survivors. Change native512 crop to full-short-side center-square resized512; retain masks/seeds/SDPA/prompt/steps/checker and all16/16 quality checks. E128 prepares matched authentic/classical inputs; E129 will be registered separately after preparation. The hypothesis changes both scene context and resampling, is motivated by the native-crop diagnostic and the generator's512px aesthetic-image training, and makes no assertion about safety-filter correctness. No generated output, detector improvement or corpus admission is claimed yet.

E128 completed16/16 preparations in5.25seconds. Source center squares cover69.23–75.29% of full source area; authentic/classic/mask/zero-target provenance is bound. Mean texture proxy0.038142→0.032809 is an input-description result, not detection improvement. E129 is now separately frozen; full16 generation awaits the Model1 GPU lifecycle. Twelve focused geometry/ancestry/outcome tests plus four dependency tests passed. Preserve E12211/16 result unchanged and compare all16 paired attempts.

E128/E129 implementation verification:1,014 Python tests passed; no detector accuracy is implied. E129 remains in `waiting_for_Model1_GPU_release`.

### Model2 boundary-aware evaluation implementation — 2026-09-15

Added `pixelproof.spatial_evaluation.evaluate_triplet` for a future registered localization experiment. It accepts matched authentic, traditional-edit and exact-composite score maps, with a strictly boolean partial intended mask. The mask is evaluation input only. A prespecified integer Chebyshev boundary width partitions every in-image pixel into interior, inner boundary, outer boundary and background; image edges do not invent mask boundaries. The calling protocol must freeze width and threshold before scoring.

Report per-parent regional mean scores/flagged fractions, full-image confusion/IoU/AUC/AP and interior-versus-background ranking. Authentic and traditional edits always have zero AI truth; their intended masks only stratify false accusations. Empty interiors/backgrounds retain null metrics and explicit availability, avoiding artificial success on thin edits. No pixel-independence confidence interval, threshold optimization or pooled-pixel dominance. The module cannot itself certify source identity or exact compositing; experiment input hashes and provenance checks remain necessary.

Synthetic verification includes a perfect localizer, a seam-only detector that misses every interior pixel, a seam-sensitive traditional-edit false positive, thin masks, image-border geometry, score/shape/type failures and fixed-threshold ties. This is evaluation infrastructure, not detector improvement. Existing E107 remains immutable; E129's complete16/16 generation gate and a separate learning contract are still required before new spatial-head training. No new images, GPU work, training admission or serving change.

### E130 implementation checkpoint — 2026-09-15

Implemented a pending mask-blind patch-drift diagnostic for the complete E129 population; see PLAN.md for fixed settings, primary references and limitations. Thirty-nine targeted tests passed across numerical primitives, boundary-aware metrics, complete-attempt identity/role guards and mocked encoder geometry. A rejected generation retains authentic/classical controls without fabricated positive metrics. E130 registration waits for actual E129 results; no new detector result or training admission. Prior metrics/source-access commit80ebce0 passed all1033 Python tests and GitHub CI34904064735.

### E131 implementation/metadata checkpoint — 2026-09-15

Implemented conservative source/prompt/family component closure and FIT-only transforms/fresh convex heads for a pending internal source-holdout diagnostic. Forty class/source names collapse to10 declared components, only3 AI-bearing. Exact grouped folds and method are recorded in PLAN.md. No actual new head fit, held-out scores or role reassignment yet. E130 remains implemented but unfrozen, waiting for E129 outputs. E130 implementation commit64d63f9 passed1053 Python tests and GitHub CI34904736843.

### Immediate registered-stage runner and encoder smoke — 2026-09-15

Added `ml/tools/run_registered_research.py` for an immediate, single already-frozen E130 or E131 stage. It does not schedule or wait for future work. It refuses incomplete E124/E125/E126/E129 resource lifecycles, checks the frozen contract and exact E92 artifact before stopping only the verified8800 listener, uses one resource lock, records the result and restores the base E92 environment with exact health identity. One targeted lifecycle test covers completed, scientific-skip, running and restoration-failure states.

A CPU-only smoke test loaded the existing SHA-verified DINOv2-S weights at448px and scored one synthetic constant RGB image through the actual E130 adapter. Patch grid32x32, token shape32x32x384, duplicate-token error0, maximum half-cosine numerical drift2.682209e-7,0.88seconds including setup. Zero real images and zero downloads. This checks technical tensor/weight compatibility only; it is not a localization result. The main E124 GPU extraction remained separate. E131 implementation commit7961b14 passed1070 Python tests and GitHub CI34905706666.

### E124/E125 unattended execution — 2026-09-14T22:51:28+00:00

E124 completed 12525 parents / 50100 TRAIN views; maximum full-frame pilot replay error 0.0. Registering the prespecified paired E125 fit; no DEV/final opened.

### E131 pre-registration scope clarification — 2026-09-15

Before any E131 contract or fit, rechecked the existing E79 provenance note and [RR's primary paper, sections3.2.1–3.2.2](https://arxiv.org/html/2509.09172v1). RR explicitly includes FLUX, DALL-E and SD-family generations, COCO/CC3M-derived prompts and Chameleon material at corpus level. The local admitted rows lack the file-level mapping needed to assign those origins. CommunityForensics is also a mixed-generator corpus. Therefore the10 declared row/publisher/prompt components **do not establish generator-family independence**. Conservatively connecting every possible mixed-corpus relation could collapse the apparent AI-bearing groups; their count is not a certificate of unseen families.

E131 remains useful as a publisher-group internal transfer diagnostic with explicitly incomplete upstream ancestry. Its contract and report now state `generator_family_holdout_supported:false` and retain the corpus-level limitation. Do not describe its grouped folds as unseen-generator proof, a fresh source benchmark or a qualified deployment candidate. This correction precedes registration/scoring; no result or threshold was selected. A true generator-held-out experiment requires file-level generator/prompt lineage or independently acquired clean families.

E131 is now separately frozen before fitting at contract SHA256 `6db23ea917a7428d015e0453efe99b16bbcbe3fa2822e6b540e8c31b645ea421`. Exact12525-parent/10-declared-component/three-fold allocation is retained; generator-family independence is explicitly unsupported. Execution waits for resource release after E129/E130. No E131 scores yet.

### E130 pre-registration baseline completion — 2026-09-15

Before E130 freeze or any real-image localization score, added matched Haar pixel-response RMS, radial-center and constant ranking controls. Keep the identical448px input/32x32 grid and reflect3x3 median for the pixel-response control, then compare on exactly the same accepted parents/masks. Pixel RMS receives no transferred classification threshold. These controls test whether a drift map only reflects image texture or a location prior. All authentic/classical negatives and fixed drift-cut diagnostics remain; nothing is selected from scores. Twenty targeted numerical/encoder tests passed. The prior synthetic CPU smoke remains a technical check only. E131's already frozen inputs are unchanged.

### E124/E125 unattended execution — 2026-09-14T22:59:13+00:00

E125 paired TRAIN fits completed. {"center_control": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}, "full_frame": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}} These are TRAIN guards only; serving/promotion and independent evaluation remain unchanged. Full reports are in evidence/e125_fullframe_fit.json.

### E124/E125 unattended execution — 2026-09-14T22:59:31+00:00

Pipeline final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E126 dependent execution — 2026-09-14T23:00:11+00:00

Started separately registered E126 consumed-DEV scoring for center_control, full_frame. Frozen640-view comparison; no training or final/gallery access.

### E125 paired TRAIN completion — 2026-09-15

Both center_control and full_frame pass complete TRAIN, source/sensor, retention and runtime guards. Each catches4595/4595 AI and falsely flags0/7930 REAL in every one of four conditions. Center/full runs took211.60/206.25seconds. Saved batch8 maximum score errors2.383225e-7/2.383701e-7 stay below the frozen1e-6 tolerance, with zero decision changes at either cutoff. Candidate SHA256 values are04b56392e57a320c0a1aae2bc4ab939a1459c042159f2aa2b93a51ceaecf69bf andb8d7d38950e47aca3c05ba7d51dd71bc1a4aa50fa5225bcd32d25b17bb0412aa respectively.

E92 was restored successfully. E126 has started its separately gated640-view consumed-development procedure. No DEV outcome or serving promotion is implied yet. Full result: evidence/e125_context_fit.json. E131 was frozen before this completion and remains pending resource release.

### E126 automatic consumed-DEV result

{"center_control": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}, "full_frame": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}}

Full report: evidence/e126_fullframe_development.json. This is consumed development; no independent proof or serving change. Only passing branches may proceed to a separately registered gallery regression. Failed branches require a new hypothesis, not relaxed gates.

### E126 dependent execution — 2026-09-14T23:05:34+00:00

E126 finalizer E92 restoration ready=True.

### E129 one-shot execution — 2026-09-14T23:05:42+00:00

Starting registered same16-parent E129 resized-context inpainting ablation after Model1 completion. No detector fitting or evaluation data. E92 is temporarily stopped for GPU memory.

### E129 fixed input-context replay completed

{"comparison": {"context_passed": 16, "full16_engineering_gate_passed": true, "new_failed_attempts": 0, "previous_passed": 11, "rescued_attempts": 5}, "contract_sha256": "15524ea7f66a1e52fd7a539b53f2b218968cf8899d383623aba4b18025a7116a", "detector_scores": 0, "downloads": 0, "limits": "TRAIN research engineering, one old generator and correlated MIDD parents. Resampling and scene context change together. A pass is not semantic-quality, localization-accuracy or independent-OOD evidence; a new spatial-learning contract remains necessary.", "parents": 16, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "report_sha256": "b07aaf1b361c6603ebcee78d3e6054198ab6d6d4f3022deaac94128f1106e76e", "seconds": 458.7882085829042, "spatial_protocol_may_be_registered": true, "state": "E129_context_generation_complete", "training_admission": false}

This is paired generation engineering, not detector accuracy. Preserve E122 and all failures; no automatic training/serving admission.

### E129 one-shot execution — 2026-09-14T23:14:08+00:00

E129 pipeline final state: {"E92_restored_after_pilot": true, "failure": null, "state": "complete"}

### E130 registered-stage execution — 2026-09-15T10:43:08+00:00

Starting the registered audit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E130 patch-drift diagnostic completed

{"accepted_composites": 16, "contract_sha256": "b9eb9ed0e1f1240e5f846cf7deec635b6db57fff48529f6d03e1d6f3a51aea78", "excluded_generations": 0, "limits": "Small correlated TRAIN research, one old editor, simple masks, conditional accepted-generation subset. This is an adapted edit-response baseline, not an exact TRAIL reproduction, calibrated AI probability, semantic annotation or independent detector evidence.", "max_duplicate_token_error": 0.0, "measurements_sha256": "c235cb22f5000405974063b951bb1c7a93f87d435acb3d455332d19bb1776bb0", "parents": 16, "peak_mps_bytes": 1219198976, "promotion_allowed": false, "scoring_sha256": "0b54f326768b00ce90029c2d87ef088acdd4a42825bf9d9edf796049c95616df", "seconds": 25.685287708998658, "state": "E130_patch_drift_audit_complete", "summary": {"jpeg75": {"accepted_parent_ranking_baselines": {"constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.5291163098010457, "radial_center_auc": 0.7975643577862963}, "accepted_parent_spatial_contrast": {"AI_minus_authentic_auc": 0.12430492180855968, "AI_minus_classical_edit_auc": -0.18437926993454085, "interpretation": "Within-image aligned-region ranking differences, not REAL-versus-AI classification accuracy.", "parents": 16}, "ai_composite": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.459066698329999}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.50054931640625}, "interior_background_auc": {"available_parents": 16, "mean": 0.7056244594884324}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.7439364244365543}, "pixel_auc": {"available_parents": 16, "mean": 0.6973202554169273}}, "parents": 16}, "authentic": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.4412181806683595}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.45697021484375}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.5513152973102817}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}, "classical_edit": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.48944667454250673}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.5535888671875}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.925899659695963}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}}, "original": {"accepted_parent_ranking_baselines": {"constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.47224683746742646, "radial_center_auc": 0.7975643577862963}, "accepted_parent_spatial_contrast": {"AI_minus_authentic_auc": 0.1468308237980383, "AI_minus_classical_edit_auc": -0.20524348577118698, "interpretation": "Within-image aligned-region ranking differences, not REAL-versus-AI classification accuracy.", "parents": 16}, "ai_composite": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.5346556152692453}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.57806396484375}, "interior_background_auc": {"available_parents": 16, "mean": 0.7321641102415646}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.8157760800408127}, "pixel_auc": {"available_parents": 16, "mean": 0.7259450807081305}}, "parents": 16}, "authentic": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.5161144963519939}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.53826904296875}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.6631707335390167}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}, "classical_edit": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.6090485126157333}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.66693115234375}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.991369405449905}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}}}, "training_admission": false}

No trained localizer, calibrated probability, independent proof or training admission is claimed.

### Continuation checkpoint and next diagnostic — 2026-09-15

Reviewed the completed E126/E129 receipts after the user requested continuation. E126's center and full-frame branches both retain the same binary consumed-DEV counts: original REAL false alarms0/160 and AI159/160; social-Q75 REAL false alarms12/160 and AI159/160. Both pass the20 numeric checks but fail the original-image E43 individual-AI retention guard. Neither branch proceeds to gallery regression or serving. E129 improves the fixed generation population from11/16 to16/16, rescuing five attempts with no new failures,458.79seconds and6,788,562,944 peak MPS bytes. This is input/generation engineering, not detector accuracy or semantic-quality certification.

E130 is now frozen against the complete16-case E129 receipt and running its prespecified mask-blind localization/control audit. E131 remains frozen and will run after E130 releases shared memory. The prior automatic E125 journal's filename evidence/e125_fullframe_fit.json was incorrect; the actual report is evidence/e125_context_fit.json. Preserve that original journal and this correction.

The previous overnight window has elapsed. Process completion timestamps establish execution, not nine hours of continuous agent reasoning. Continue from existing evidence rather than restart experiments.

### E130 registered-stage execution — 2026-09-15T10:43:53+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E131 registered-stage execution — 2026-09-15T10:44:14+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E131 internal source-holdout result

{"contract_sha256": "6db23ea917a7428d015e0453efe99b16bbcbe3fa2822e6b540e8c31b645ea421", "corpus_level_overlap_limit": "RR paper sections3.2.1-3.2.2 explicitly name FLUX, DALL-E and SD families, COCO/CC3M prompts and Chameleon. Local RR rows do not map each file to those origins. CommunityForensics also mixes generators. The10 components preserve declared row/prompt links but do not isolate all corpus-level possible families; this is publisher-group transfer only, not unseen-generator evidence.", "downloads": 0, "fold_counts": [{"0": 2467, "1": 2916}, {"0": 2875, "1": 1110}, {"0": 2588, "1": 569}], "generator_family_holdout_supported": false, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence.", "locked_scores_sha256": "e3dbcbc1fa28d26e58e48805c1d97eb6575b60462b7d3894b8ec2a9d86b45121", "new_pixels_read": 0, "paired_changes_fullframe_vs_center": {"assigned_transport": {"0": {"new_errors": 84, "rescued_errors": 78}, "1": {"new_errors": 67, "rescued_errors": 152}}, "clean": {"0": {"new_errors": 79, "rescued_errors": 58}, "1": {"new_errors": 55, "rescued_errors": 116}}, "q75": {"0": {"new_errors": 77, "rescued_errors": 46}, "1": {"new_errors": 71, "rescued_errors": 162}}, "social_q75": {"0": {"new_errors": 81, "rescued_errors": 66}, "1": {"new_errors": 68, "rescued_errors": 152}}}, "parents": 12525, "promotion_allowed": false, "seconds": 42.25651224993635, "source_components": 10, "state": "E131_paired_internal_source_holdout_complete"}

Full component/condition metrics: evidence/e131_source_holdout.json. This is consumed TRAIN analysis using separately fitted fold heads, not an independent final or serving candidate.

## E132 — paired patch learning with source exclusion (planned 2026-09-15)

E130 completed in25.69seconds with zero duplicate-token error. Original/JPEG75 mean composite pixel AUC is0.72595/0.69732, below the same-mask radial-center baseline0.79756. The transferred diagnostic cut falsely flags53.83%/45.70% of authentic area and66.69%/55.36% of classical-edit area. AI-versus-classical aligned-region AUC contrasts are negative (-0.20524/-0.18438). Thus this drift map cannot be presented as an AI-specific detector. No threshold sweep or deployment follows.

Next implement a separately registered E132 internal learning diagnostic using the existing E130 original patch-token caches (384 dimensions,32x32). Admit only the fixed16 E129 hard composites and their16 authentic/16 same-mask Gaussian-blur controls to this scoped TRAIN research experiment; global dataset roles remain unchanged. Their16/16 engineering gate passes. A contact-sheet inspection of all16 pairs found scene-preserving local reconstructions and some visible object/detail changes, not a semantic-mask annotation or a guarantee of realistic edits. Keep all16; do not cherry-pick attractive outputs.

Use leave-one-declared-sensor-group-out folds, joining any known scene/source-body ancestry across sensors first. Fit normalization and a zero-initialized regularized logistic patch head solely on the remaining parents. No position, true-mask, difference-to-original or drift map enters inference. Train on fully interior AI patches as positives and equal-mass authentic, classical-edit and composite-background negatives; discard composite boundary patches using the fixed8px band. Both original and JPEG75 views of a parent stay together. Fix BCE+0.005||w||², zero-start L-BFGS and0.5 diagnostic cut before fitting, without held-out tuning. Reuse the tested finite/convergence guards.

Save all fold heads and lock every held-out map before metrics. Report authentic/classical false-flag area, full and interior/background localization, per-parent paired contrast and the same E130/radial/constant baselines, including every parent and condition. This compares internal held-out sensors within one small, already inspected MIDD research population and one SD1.5 editor; it cannot demonstrate unseen-editor, new-dataset or universal performance. Scores are uncalibrated. No final full-data head, web heatmap or deployment is authorized by this diagnostic's outcome.

### E131 registered-stage execution — 2026-09-15T10:45:18+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E131 source-transfer interpretation — 2026-09-15

E131 completed all six fresh-head fits in42.26seconds. On clean held-out publisher groups, center/full-frame AUC is0.91815/0.92284, AI recall80.04%/81.37%, and REAL FPR14.74%/15.01% at the preregistered0.5 diagnostic cut. Social-Q75 AUC is0.89184/0.89889, AI recall75.91%/77.74%, and REAL FPR16.20%/16.39%. The full-frame input slightly improves aggregate AI recall but also creates new REAL and AI errors; it is not a dominance or promotion result.

The largest clean REAL failure is the excluded RR group:997/1250 (79.76%) false alarms for center and1010/1250 (80.80%) for full-frame. The declared E32/E36 AI component's recall is70.85%/73.11% clean and64.09%/67.04% social-Q75. MIDD has0/767 REAL flags in both clean branches. These are separately refitted simple heads using different fold models, not measured deployed-E92 rates. The large group variation supports investigating source/content/processing shortcuts and calibration transfer. It does not identify a causal shortcut, prove bad labels, or establish unseen-generator independence. No threshold tuning, reserve opening, gallery scoring or serving change follows.

E132 implements the previously described negative-control patch-learning test. Targeted checks cover transitive source/scene/body grouping, boundary-cell exclusion, exact class/parent/condition/negative-variant weight mass, FIT-only normalization and a real small end-to-end logistic fit. The end-to-end check ensures every held-out map is locked before metrics and refuses rerunning a started experiment. Runtime restoration now supports E132 only after E130/E131 complete with exact E92 restored.

### E132 registered-stage execution — 2026-09-15T10:51:22+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E132 internal patch-learning result

{"contract_sha256": "c5f07c1e38b8b932c3e63b1ff16ce3ffbb9d529d5013a348d4a8af85dd7a4363", "downloads": 0, "folds": 8, "limits": "Sixteen inspected MIDD parents, one SD1.5 editor, hard composites, simple masks and two views. Global token context may carry synthetic cues into background. Intended generation masks are not semantic-change annotations. Unknown scene ancestry and encoder pretraining remain. Fold heads are not one deployable model; raw scores are not calibrated probabilities.", "locked_scores_sha256": "65329652dca7ee42648170b86788f02769bd6d0ef793d8d38398b113684182aa", "measurements_sha256": "18bdb2a002574c21c5da148d5d38d4a0d9a54403ed063b75a9ad11f1e3955f46", "parents": 16, "promotion_allowed": false, "seconds": 25.856737582944334, "state": "E132_internal_patch_learning_complete", "summary": {"jpeg75": {"AI_minus_authentic_aligned_auc": 0.07112007203130111, "AI_minus_classical_aligned_auc": 0.13904100934913213, "ai_composite": {"mean_flagged_area": 0.1927490234375, "mean_interior_background_auc": 0.7491296090659942, "mean_iou": 0.20270705884133267, "mean_pixel_auc": 0.7342234512939676}, "authentic": {"mean_flagged_area": 0.174560546875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "classical_edit": {"mean_flagged_area": 0.17108154296875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "reference_rankings": {"E130_drift_mean_pixel_auc": 0.6973202554169273, "constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.5291163098010457, "radial_center_auc": 0.7975643577862963}}, "original": {"AI_minus_authentic_aligned_auc": 0.07381122188737699, "AI_minus_classical_aligned_auc": 0.17412843479263335, "ai_composite": {"mean_flagged_area": 0.1910400390625, "mean_interior_background_auc": 0.7570250862785984, "mean_iou": 0.22249568870308084, "mean_pixel_auc": 0.7422432027316962}, "authentic": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "classical_edit": {"mean_flagged_area": 0.16009521484375, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "reference_rankings": {"E130_drift_mean_pixel_auc": 0.7259450807081305, "constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.47224683746742646, "radial_center_auc": 0.7975643577862963}}}}

No serving change, calibrated probability or independent generalization claim.

### E132 registration and validation — 2026-09-15

Frozen E132 before fitting at contract SHA256 `c5f07c1e38b8b932c3e63b1ff16ce3ffbb9d529d5013a348d4a8af85dd7a4363`:16 parents,8 declared sensor/known-ancestry components. The common16-set mask audit found usable pure interior and background patches for every parent. Runtime dependency versions, source helpers and complete inherited evidence are bound. Only this scoped patch-learning experiment admits the paired research examples; no global role or independent-final claim changes.

All1079 Python tests passed (one upstream Starlette/httpx deprecation warning); compileall succeeded. The small end-to-end test performs actual convex fitting and confirms held-out source exclusion, locked predictions before metrics, and fail-closed reruns. E132 is executing with exact E92 restoration in its finalizer; no new data download is required.

### E132 registered-stage execution — 2026-09-15T10:52:08+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E132 interpretation and next steps — 2026-09-15

All eight sensor/known-ancestry fold heads completed in25.86seconds; every source pair was scored only by a head that excluded its component. Original/JPEG75 mean composite pixel AUC is0.74224/0.73422 (E130:0.72595/0.69732), interior/background AUC0.75703/0.74913, and IoU0.22250/0.20271. AI-minus-classical aligned-region AUC contrast is now positive0.17413/0.13904 (E130 negative). This supports a limited learned distinction between these synthetic composites and Gaussian blur in this pilot.

Authentic flagged area at the E1320.5 diagnostic cut is17.11%/17.46%, and classical-edit flagged area16.01%/17.11%. Both remain far too large to present as a reliable AI heatmap. These cut-specific false-area rates cannot be compared as matched-operating-point improvement over E130: its external drift cut and positive coverage differ. The threshold-free composite ranking remains below the radial-center baseline0.79756 in both conditions. The pilot has not defeated a location prior or demonstrated robust localization. No full-data final head, gallery score, web change or deployment follows; exact E92 is restored.

Next Model2 mechanism test: pre-register masks with matched area but varied location/shape independent of image content, including off-center edits; keep authentic and same-mask classical controls. Hold complete source/known-ancestry groups out and compare frozen E132 fold heads before any retraining. Record generator/checker failures without replacements, and separate interior from boundary success. Do not infer causal center bias merely from the present baseline gap. A second editor and independent source population are required later for editor/source transfer evidence. The present16-set SD1.5/MIDD pilot cannot provide that proof by further threshold tuning.

Next Model1 investigation: use the locked E131 training-source predictions to characterize publisher/content/processing differences and per-source score shifts before choosing another representation or grouped calibration experiment. The excluded RR error rate is a diagnostic, not permission to tune on consumed DEV or protected reserves. Any calibration experiment needs a nested source split, retaining outer held-out sources untouched by normalization, fitting and cutoff selection. Preserve individual AI non-regression as well as REAL errors; do not accept a higher aggregate AUC as a replacement for those safeguards.

## E133 — frozen-head mask-location challenge (planned 2026-09-15)

Test E132's location sensitivity without retraining or choosing another cutoff. Reuse the exact16 E128 authentic inputs and E132 source-excluding fold heads. Translate each original binary mask's tight bounding box, unchanged in shape and pixel area, to a fixed corner with16px image margin. Corner assignment is index modulo4 (top-left, top-right, bottom-left, bottom-right), four parents per corner, fixed independently of pixels and scores. This isolates the mask-position intervention more narrowly than simultaneously varying shape; varied shapes/editors remain later work. Each parent receives one new placement, not a fully crossed parent-by-corner experiment.

Keep source images, seeds119000+i, prompt, SD1.5 weights,30 steps,7.5 guidance, fp16+SDPA and enabled safety checker exactly as E129. Generate one attempt per parent, retaining all failures and no rerolls/refills. Rebuild matching GaussianBlur6 classical negatives using the new mask and retain authentic negatives. Locally synthesized descendants are TRAIN research only; no download, new independent source, global training admission or deployment.

Freeze preparation, generation, head binding and scoring before any new generation or score. Score all16 authentic/classical controls and only accepted composites; explicitly count rejected attempts. Encode with the exact E130 DINO adapter and require authentic-token replay<=1e-5 and frozen-head authentic-score replay<=1e-6 against the previous cache. The head for each image must still exclude its original source/known-ancestry component. Preserve the0.5 diagnostic cut. Lock every map before new-mask metrics.

Report all-negative false-flag area; composite full and interior/background AUC/IoU; aligned AI-authentic/classical contrast; radial-center and constant ranking baselines on the same accepted new masks; paired old/new composite metrics on exactly the accepted parents; and every failure. A position intervention also changes edited semantic content and diffusion output, so any score difference is not pure causal proof of positional bias. Sixteen previously inspected parents, one old editor and unknown upstream ancestry cannot establish independent generalization.

### E133 registration and preflight — 2026-09-15

E133 is frozen before preparation/generation/scoring at SHA256 `74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037`. Targeted28 tests and the complete1104-test Python suite passed. Tests cover exact shape/area-preserving translation for all16 assignments, no clipping/nonbinary admission, consistent failed/successful attempt accounting, exact classical-control background and authentic identity, accepted-parent-only paired summaries, and predecessor resource restoration.

The first freeze invocation used the base environment and failed on the deliberately isolated diffusers dependency before writing a contract; the premature preparation attempt consequently failed on the absent contract before writing images. Re-running freeze with the existing pinned model2-runtime PYTHONPATH succeeded. No dependency installation, parameter change, generation attempt or contract rewrite occurred. All subsequent generation uses that isolated runtime; E92 restoration uses the base environment.

### E133 preparation complete

{"contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "parents": 16, "prepared_sha256": "37f6855fb56972980d35bf1753aaf3c0e86755a9bb9608459b3119ca53cde3c0", "training_admission": false}

## E134 — locked Model1 source-score audit (planned 2026-09-15)

While E133 generates images, run only a lightweight, separately registered audit of already locked E131 predictions. No raw features, new image pixels, encoder, fit, cutoff selection, calibration or GPU allocation. Bind E131's contract, report, locked-score receipt and complete12525-parent/four-condition/two-branch score arrays. Retain TRAIN/INTERNAL_HELD_OUT roles and the existing source components/folds.

For every declared component and class, report fixed score quantiles0/10/25/50/75/90/100%, mean and the fraction above the existing0.5 diagnostic cutoff. Compare each processing condition with clean on matched parents, reporting score shifts and new/rescued decisions. Include the complete roster, not only RR. These raw-score distributions characterize source-dependent offsets and processing sensitivity; they are not calibrated probabilities or causal evidence about which visual feature creates errors. Scores were previously inspected in E131; this is a retrospective descriptive diagnostic, not a new predictive test or independent validation.

### E133_GENERATION registered-stage execution — 2026-09-15T13:26:28+00:00

Starting the registered generate stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E134 locked source-score audit complete

All88 component/class/condition/branch summaries are retained in evidence/e134_source_scores.json. No fits, new pixels, threshold changes or serving change. Retrospective consumed TRAIN/INTERNAL_HELD_OUT analysis, not a new predictive test or deployed-E92 evaluation. Different folds use different heads. Unknown generator/prompt/scene ancestry remains; no calibration, threshold selection or causal feature explanation.

### E134 source-score findings — 2026-09-15

Frozen retrospective audit SHA256 `685a172b1171345dd24ab2ab7abd195268728c1eb8004eb704681b5b7367655b`; all88 component/class/condition/branch summaries are retained. Eight targeted distribution/paired-direction/invalid-input tests passed. No raw features or images, GPU, fitting, calibration or cutoff search were used.

In E131's full-frame head excluding RR, the1250 clean RR REAL scores have median0.81744, lower quartile0.56913 and upper quartile0.93320;1010 exceed the fixed0.5 cut. This is a broad distribution shift, not merely a handful of outliers. RR AI median is0.99184, with1038/1110 above0.5. Social-Q75 shifts the RR REAL median to0.85479 and produces67 new errors versus18 rescues, for49 net extra false alerts. These are raw scores of a source-excluding diagnostic head, not E92 deployed rates or calibrated probabilities.

For the separate E32/E36 AI component, clean-to-social-Q75 causes298 new misses versus121 rescues (177 net losses); recall drops73.11% to67.04%. This preserves the distinction between source offsets and processing sensitivity. The component uses a different fold head from RR; cross-component score differences cannot by themselves establish a single-model calibration solution or causal feature failure. Retain nested source-separated calibration as a future controlled experiment rather than choosing a cutoff from these observed distributions.

### E133 generation complete

{"contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "full16_engineering_gate_passed": true, "generation_sha256": "c9f75a64948f0364a4e0990cb33d3344997b4c1d6fcba38754ae2dd0c0efd369", "parents": 16, "passed": 16, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "seconds": 555.461868665996, "state": "E133_location_generation_complete", "training_admission": false}

### E133_GENERATION registered-stage execution — 2026-09-15T13:36:04+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E133_LOCALIZATION registered-stage execution — 2026-09-15T13:36:40+00:00

Starting the registered score stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E133 frozen-head location result

{"accepted_composites": 16, "contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "limits": "Consumed sixteen-parent MIDD research with one old editor and known source-excluding heads. New placement changes edited semantic content and diffusion output; not pure causal position attribution or a fully crossed corner experiment. Intended masks are not semantic-change annotations. No fresh-source/editor proof, calibration or serving promotion.", "locked_scores_sha256": "db0cb8362cfd34ee32ab6e81efe31eaf27ae81c257cf48ce4b5c1e736192b79e", "max_authentic_score_error": 0.0, "max_authentic_token_error": 0.0, "measurements_sha256": "6edfdcd3c03b92b7c71dc5d3c708c22da2dc12b4fa7f75819670d534167bca91", "parents": 16, "peak_mps_bytes": 1219198976, "promotion_allowed": false, "rejected_attempts": 0, "seconds": 18.19654425000772, "state": "E133_frozen_head_location_challenge_complete", "summary": {"jpeg75": {"accepted_parents": 16, "ai_composite": {"mean_flagged_area": 0.183349609375, "mean_interior_background_auc": 0.5696380503109616, "mean_iou": 0.09461821008149006, "mean_pixel_auc": 0.5618493555000308, "parents": 16}, "all_parents": 16, "authentic": {"mean_flagged_area": 0.174560546875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "classical_edit": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "matched_accepted_comparison": {"AI_minus_authentic_aligned_auc": 0.08085072641406385, "AI_minus_classical_aligned_auc": 0.16855089642605717, "constant_auc": 0.5, "new_radial_center_auc": 0.4421728665547357, "old_E132_mean_pixel_auc": 0.7342234512939676, "parents": 16}}, "original": {"accepted_parents": 16, "ai_composite": {"mean_flagged_area": 0.182373046875, "mean_interior_background_auc": 0.5746209190475555, "mean_iou": 0.10165574918839779, "mean_pixel_auc": 0.5658123794602998, "parents": 16}, "all_parents": 16, "authentic": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "classical_edit": {"mean_flagged_area": 0.16778564453125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "matched_accepted_comparison": {"AI_minus_authentic_aligned_auc": 0.09192518625126159, "AI_minus_classical_aligned_auc": 0.1851799676281684, "constant_auc": 0.5, "new_radial_center_auc": 0.4421728665547357, "old_E132_mean_pixel_auc": 0.7422432027316962, "parents": 16}}}, "training_admission": false}

### E133_LOCALIZATION registered-stage execution — 2026-09-15T13:37:08+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E133 completed location challenge — 2026-09-15

All16 generation attempts passed the unchanged engineering checks,555.46seconds and6,788,562,944 peak MPS bytes. No rejected/repeated/replaced attempt. Prepared32 PNG payloads occupy7,413,473 bytes (16 masks,16 classical controls);32 raw/composite generation PNG payloads occupy15,646,384 bytes. Authentic inputs are reused. macOS AppleDouble sidecars are excluded from image counts. Downloads remain0; these are local research derivatives, not new independent parents.

Frozen E132 heads on the new locations produce original/JPEG75 composite mean pixel AUC0.56581/0.56185, down from0.74224/0.73422 on the exact matched16 original placements. IoU is0.10166/0.09462, down from0.22250/0.20271. The new radial-center baseline is0.44217; fixed heads beat that weak corner ranking but only modestly exceed the constant0.5 baseline. AI-minus-classical aligned AUC contrast remains positive0.18518/0.16855. Thus some edit-related discrimination persists while absolute localization degrades substantially.

Authentic token and score replay errors are both exactly0. Authentic flagged area is unchanged17.11%/17.46%. The measurement does not identify pure causal positional bias because moving the mask also changes edited semantic content and the generated output. It does expose failure under the registered location/content intervention. All maps were locked before mask metrics; scorer took18.20seconds with1,219,198,976 peak MPS bytes. E92 restored exactly. No serving promotion.

## E135 — learn from both registered mask placements (planned 2026-09-15)

Apply a bounded development response to E133: train the same E132384-D frozen-token logistic head with both the original and corner placements, equal weight per parent/placement/condition. Preserve exact E132 source/known-ancestry folds; all variants/placements of a source component stay together. Use both authentic and same-mask classical negatives, the fixed8px pure-patch exclusion, weighted FIT-only normalization, zero-start BCE+0.005||w||², solver guards and0.5 cut. No hyperparameter or threshold search.

Cache only the missing E133 classical/composite patch tokens with the exact E130 adapter; replay unchanged authentic tokens<=1e-5. Reuse the original E130 tokens and authentic entries; duplicate bookkeeping across the two placements does not create independent authentic observations. Only the E129/E13316/16 accepted hard composites enter this scoped TRAIN research. No role changes outside E135, new editor or image download.

Fit eight new fold heads, each excluding all data from its original held-out sensor component. Lock all192 view maps (16 parents x2 placements x3 variants x2 processing conditions) before held-out metrics. Report original/corner separately, authentic/classical false-flag area, full/interior-background AUC and IoU; retain paired E132/E133 frozen-head comparisons and new/rescued parent-level outcomes where applicable. This is an adaptive internal development comparison after seeing E133, not fresh independent validation, and no model is promoted from its outcome.

### E135 registration and validation — 2026-09-15

Before new feature extraction or fitting, E135 froze at contract SHA256 `0e240bbe58c5672b5a1568be315d509c4c49196cb68f0d7d594aa5b65e24c98c`:16 unique parents,32 placement records, the same8 source-excluding folds,192 view maps. New controls explicitly verify identical authentic ancestry across placements, unchanged authentic training-weight mass, separate matched comparisons per placement and cache/resource restoration before fitting. Eleven targeted tests and all1116 Python tests passed. The single warning is the existing upstream Starlette/httpx deprecation. No frozen E132/E133 implementation was modified.

The prior implementation push430d385 passed GitHub CI34975306116, including1112 Python tests and web checks. E135 token caching is now running; results remain unknown at registration. Subsequent metrics will remain adaptive internal development evidence, not independent confirmation.

### E135_CACHE registered-stage execution — 2026-09-15T13:43:55+00:00

Starting the registered cache stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E135 cache complete

{"contract_sha256": "0e240bbe58c5672b5a1568be315d509c4c49196cb68f0d7d594aa5b65e24c98c", "downloads": 0, "max_authentic_error": 0.0, "peak_mps_bytes": 1219198976, "seconds": 27.1236343330238, "state": "E135_two_placement_tokens_complete", "token_sha256": "a684dcc96b6540fdfc842e106a3418515bc92e1778bcbcdf5d92feb9d737416b", "unique_parents": 16, "views": 192}

### E135_CACHE registered-stage execution — 2026-09-15T13:44:42+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E135_FIT registered-stage execution — 2026-09-15T13:45:14+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E135 learning result

{"contract_sha256": "0e240bbe58c5672b5a1568be315d509c4c49196cb68f0d7d594aa5b65e24c98c", "downloads": 0, "folds": 8, "limits": "Adaptive internal development after observing E133; sixteen previously inspected MIDD parents, one SD1.5 editor, two masks per parent. Source exclusion prevents declared parent/scene leakage, not unknown ancestry or adaptive validation bias. More views are not more independent parents. Raw scores uncalibrated; no deployment or universal/editor-transfer proof.", "locked_scores_sha256": "6872c17f604ed874c1a0d4d0c97a17db30d81926784f012ed1bb786e49d27962", "measurements_sha256": "2434046ea207c1ca450d80e6f6e1f647b01ee7ce08e200dce2803de85c41de11", "placement_records": 32, "promotion_allowed": false, "seconds": 48.97825333289802, "state": "E135_two_placement_internal_learning_complete", "summary": {"corner": {"jpeg75": {"ai_composite": {"mean_auc_change": 0.07067495358935735, "mean_flagged_area": 0.28759765625, "mean_flagged_area_change": 0.104248046875, "mean_interior_background_auc": 0.6476919664012324, "mean_iou": 0.17331742888558427, "mean_pixel_auc": 0.6325243090893882, "parents_higher_auc": 12, "parents_lower_auc": 4}, "authentic": {"mean_flagged_area": 0.26605224609375, "mean_flagged_area_change": 0.09149169921875, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.2734375, "mean_flagged_area_change": 0.102294921875, "mean_iou": 0.0}, "unique_parents": 16}, "original": {"ai_composite": {"mean_auc_change": 0.07221693225085765, "mean_flagged_area": 0.28076171875, "mean_flagged_area_change": 0.098388671875, "mean_interior_background_auc": 0.6538335879592014, "mean_iou": 0.18619654171877348, "mean_pixel_auc": 0.6380293117111575, "parents_higher_auc": 13, "parents_lower_auc": 3}, "authentic": {"mean_flagged_area": 0.2574462890625, "mean_flagged_area_change": 0.0863037109375, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.25482177734375, "mean_flagged_area_change": 0.0870361328125, "mean_iou": 0.0}, "unique_parents": 16}}, "original": {"jpeg75": {"ai_composite": {"mean_auc_change": -0.0440433199883205, "mean_flagged_area": 0.29193115234375, "mean_flagged_area_change": 0.09918212890625, "mean_interior_background_auc": 0.7017308809755072, "mean_iou": 0.18385901662410248, "mean_pixel_auc": 0.690180131305647, "parents_higher_auc": 3, "parents_lower_auc": 13}, "authentic": {"mean_flagged_area": 0.26605224609375, "mean_flagged_area_change": 0.09149169921875, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.2701416015625, "mean_flagged_area_change": 0.09906005859375, "mean_iou": 0.0}, "unique_parents": 16}, "original": {"ai_composite": {"mean_auc_change": -0.04322624339088158, "mean_flagged_area": 0.28692626953125, "mean_flagged_area_change": 0.09588623046875, "mean_interior_background_auc": 0.7096534175195824, "mean_iou": 0.19590418623135836, "mean_pixel_auc": 0.6990169593408145, "parents_higher_auc": 3, "parents_lower_auc": 13}, "authentic": {"mean_flagged_area": 0.2574462890625, "mean_flagged_area_change": 0.0863037109375, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.24737548828125, "mean_flagged_area_change": 0.0872802734375, "mean_iou": 0.0}, "unique_parents": 16}}}, "unique_parents": 16}

### E135_FIT registered-stage execution — 2026-09-15T13:46:15+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

### E135 result and disposition — 2026-09-15

Completed192 token views in27.12seconds with exact authentic replay (maximum error0) and1,219,198,976 peak MPS bytes. All eight fold heads completed in48.98seconds. Post-run receipt review confirms each fold excluded its expected parent/source component, all16 parent IDs are accounted for exactly once per fold's FIT/held union, saved-head maximum score error0, and identical authentic maps across placement bookkeeping. All192 maps were locked before evaluation. Downloads0; E92 is restored exactly.

The corner original/JPEG75 mean composite AUC improves0.56581->0.63803 /0.56185->0.63252. Per-parent corner AUC improves for13/16 original and12/16 JPEG75 cases. Corner IoU improves0.10166->0.18620 /0.09462->0.17332. However, original-placement AUC drops0.74224->0.69902 /0.73422->0.69018, with13/16 parents worsening in each condition. Original-placement IoU also falls0.22250->0.19590 /0.20271->0.18386.

Authentic falsely flagged area rises17.11%->25.74% original and17.46%->26.61% JPEG75. Classical-negative flagged area also rises:24.74%/27.01% for original placement and25.48%/27.34% for corners. Thus the candidate trades corner improvement for original-placement loss and substantially more false alarms. It is not a successful non-regressing replacement and is rejected for serving. Raw-cut comparisons are descriptive; no matched-coverage calibration claim is made.

Next investigate two mechanisms separately: richer location-independent low-level/residual features with exactly matched classical and authentic controls; and a nested source-separated calibration diagnostic to determine whether score offsets can be corrected without destroying localization. The present experiment does not prove that either will work or that the encoder alone caused the failure. Keep the current folds, previous failures and per-placement reports; any new experiment must freeze its features, fitting population and evaluation before scoring. More independent parents and a second editor remain necessary for transfer evidence; repeated tuning on this small pilot cannot supply it. Do not promote E135 or alter the demo's score/decision policy.

Implementation commit123d626 passed GitHub CI34976937916 (Python and web). The full local1116-test suite and compilation checks passed before execution. These software checks establish implementation health, not detector reliability.

### User priority update — 2026-09-15

The user requests primary focus on Model1 and broader generalization, while Model2 is presented as testing-stage work. Updated the active plan accordingly. E92 remains serving; no Model2 jobs are scheduled or running and no additional Model2 experiments will start by default. Preserve all earlier protocols/results and keep Model2 separate from the Model1 decision. Universal detection remains an unproven research objective; fresh family/source evidence and AI-retention guards remain necessary.

## E136 — Model1 paired-processing consistency (planned 2026-09-15)

E134 found substantial clean-to-social-Q75 losses in the declared E32/E36 AI component. Isolate one training change: add a quadratic within-parent logit-variance penalty to the exact E131 fresh-head objective. Existing four processing views remain together, and both center-control/full-frame geometries remain separately reported. Use the frozen E131 FIT-only normalization/PCA maps; verify all old held-out predictions before comparing new heads. Do not alter the encoders, data population, source folds, BCE/class-component weights, L2 coefficient, cutoff or existing serving model.

The penalty is0.5*0.1*sum_i w_i*(z_i-mean_parent(z))², where the original E131 weights are equal across a parent's four views. Its coefficient0.1 is a fixed first-pilot choice, not a searched optimum. Precompute the FIT-only weighted within-parent feature covariance, then fit a zero-start convex head with the existing500-iteration/finite-gradient guards. Intercept has no variance penalty. No labels, features or weights from the held-out source fold enter this covariance or head fitting. Six candidate fits (three folds x two geometries), one run, no threshold search.

Lock all candidate held-out scores before computing metrics. Report source/condition REAL FPR, AI recall, AUC, worst-group rates, per-image new/rescued errors versus E131 and changes in within-parent score range. Explicitly check zero new AI misses and zero new REAL false alarms in every condition; aggregate improvements cannot replace those checks. No E92/full-TRAIN candidate, consumed-DEV/gallery step or deployment follows automatically. This is adaptive internal TRAIN analysis; three AI-bearing components with unresolved corpus-level family overlap do not provide fresh unseen-generator evidence.

Research context: [Sagawa et al., ICLR2020](https://arxiv.org/abs/1911.08731) show that low worst-group TRAIN loss alone does not establish group generalization and study regularization. The indexed official abstract of [Li et al., CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html) includes consistency learning for compressed AI images. Direct page fetching returned403; only the indexed abstract was used. E136 is our simple convex paired-logit experiment, not a reproduction of its phase-spectrum architecture or difficulty-aware loss. No paper-reported improvement is transferred to our model.

### E136 registered-stage execution — 2026-09-15T16:36:58+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E136 Model1 consistency result

{"contract_sha256": "a94d48cbfe540464698b7c555feca35468ce14a6de12aaa2fc262c20ff6cf77d", "downloads": 0, "four_view_score_stability": {"center_control": {"0": {"baseline_mean_four_view_score_range": 0.07215121479782, "candidate_mean_four_view_score_range": 0.058017973338941625}, "1": {"baseline_mean_four_view_score_range": 0.14756244758067666, "candidate_mean_four_view_score_range": 0.11504566218030453}}, "full_frame": {"0": {"baseline_mean_four_view_score_range": 0.07063796249434952, "candidate_mean_four_view_score_range": 0.057147595881528526}, "1": {"baseline_mean_four_view_score_range": 0.14259441615715251, "candidate_mean_four_view_score_range": 0.10958736853573167}}}, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E134, not an E92 serving-model evaluation. Consistency can reduce useful discrimination or create new errors; no automatic candidate promotion.", "locked_scores_sha256": "06cb526a3ee7f2c5dfe7f99da6feb8072117095998e441a77972078b3c1201a7", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control": false, "full_frame": false}, "promotion_allowed": false, "seconds": 31.591304500005208, "state": "E136_Model1_consistency_complete"}

Full source/condition and individual-transition results: evidence/e136_transport_consistency.json. E92 remains unchanged.

### E136 registered-stage execution — 2026-09-15T16:37:51+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

## E136 interpretation and Model1-first handoff — 2026-09-15

The six fits completed in31.59seconds on12525 existing TRAIN parents (7930 REAL,
4595 AI), with four processing views per parent. All old-head and saved-candidate
score replays were exact; both center and full-frame branches fail the registered
individual non-regression screen. No coefficient search, rerun, new acquisition,
consumed-DEV/gallery access or E92 replacement followed.

Full-frame pooled internal results (different source-excluding heads per fold):

| Condition | E131 AI recall → E136 | E131 REAL FPR → E136 | New/rescued AI misses | New/rescued REAL alerts |
| --- | --- | --- | --- | --- |
| Clean |81.37% →83.44% |15.01% →15.27% |54 /149 |61 /40 |
| Social-Q75 |77.74% →79.56% |16.39% →16.07% |30 /114 |17 /43 |

Social AUC rises0.898886 to0.904752, but clean AUC falls0.922836 to0.921709.
Mean four-view score range falls0.070638 to0.057148 for REAL and0.142594 to0.109587
for AI. These are raw score ranges, not probability confidence intervals. The largest
full-frame REAL component error remains81.20% clean and83.60% social (RR), compared
with80.80%/84.72% previously. Processing consistency therefore improves stability and
some aggregate internal rates without resolving source transfer or preserving each
previously correct image. This is a useful mechanism result, not a production upgrade.
Unknown mixed-corpus generator ancestry, only three AI-bearing components and adaptive
reuse of these folds prevent an independent or universal-generalization claim.

The active priority in PLAN now gives Model1 the development/compute budget. Next is a
separately registered expert-ablation diagnostic, then provenance-qualified independent
coverage; no implicit E137 fit or download has been started. Model2 remains research/test
only; no further Model2 job is queued by default. The existing local demo explicitly
labels Model1 active and Model2 test-stage and explains that local-edit research does
not contribute to the displayed Model1 decision. The E92 artifact and decision policy
are unchanged and API health confirms the exact reference artifact is ready.

Validation:1122 Python tests and13 web tests passed; web lint/typecheck and the prescribed
Sites build helper passed. Only the existing upstream Starlette/httpx deprecation warning
was reported by Python tests. This is software verification, not detection accuracy.
Sites building/hosting guidance was reviewed for the existing manifest; the documented
local internship-demo scope is retained, with no external Sites registration/publication.

## E137 plan — Model1 expert ablation (2026-09-16, before registration/scoring)

Follow the Model1-first priority and E136 failure with one fixed leave-one-expert-out
refit experiment. Reuse all12525 TRAIN parents, four views, both center/full-frame
geometries and all three E131 publisher-group outer folds. Drop DINO, pooled CLIP,
DEAR or context separately;24 fits in total. No combinations, weight search, threshold
search, E136 penalty or adaptive selection among omissions. E131 is the comparator.
Reuse its exact FIT-only PCA/whitening maps; each remaining linear head starts at zero
with unchanged BCE+.005||w||^2 and class/component/parent/view weights. The omission
changes dimensionality (256 coordinates, or192 without context); this is deliberate,
and does not isolate an architecture-independent causal effect of an expert.

Replay frozen baseline predictions, save each candidate and require exact-cut/1e-10
numeric parity. Omitting context leaves identical inputs across the geometry branches;
require their resulting predictions to agree within1e-10. Lock all24 fit outputs before
any held-out metrics. Report all eight branch/omission combinations, source/fold/condition
rates, AUC and individual new/rescued REAL/AI errors at the unchanged0.5 diagnostic cut.
Any new error fails the corresponding individual screen, even if aggregate rates improve.

This is adaptive, consumed TRAIN diagnosis using separate fold heads. It cannot certify
unseen-generator transfer, calibrated probabilities, causes of dataset bias or a deployable
E92 replacement. Only three components contain AI and mixed-corpus ancestry is unresolved.
No consumed DEV, owner gallery or unopened reserve is accessed. No Model2 work, downloads
or new image reads; feature artifacts remain on the external data volume. Use the existing
AC-power/resource guard and restore exact E92 after this bounded offline stage.

### E137 registered-stage execution — 2026-09-15T21:49:54+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E137 Model1 expert-ablation result

{"context_omitted_geometry_max_error": 0.0, "contract_sha256": "63c2993345b5d4181988da87212ce7c2fadf26bf9062567f8a26be4173089545", "downloads": 0, "fits": 24, "generator_family_holdout_supported": false, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E134, not an E92 serving-model evaluation. Consistency can reduce useful discrimination or create new errors; no automatic candidate promotion. Adaptive repeated-fold expert-ablation diagnostic. Refit differences measure conditional utility of an expert plus changed capacity, not causal dataset bias, standalone expert quality or independent selection evidence.", "locked_scores_sha256": "3ed493830780064a951ca737d27c50b0bdc11e4591e54c1b6b44677479e911b1", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control_without_clip": false, "center_control_without_context": false, "center_control_without_dear": false, "center_control_without_dino": false, "full_frame_without_clip": false, "full_frame_without_context": false, "full_frame_without_dear": false, "full_frame_without_dino": false}, "promotion_allowed": false, "seconds": 39.361219875048846, "state": "E137_Model1_expert_ablation_complete"}

All eight branch/omission reports: evidence/e137_expert_ablation.json. No serving change.

### E137 registered-stage execution — 2026-09-15T21:50:55+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

## E137 interpretation — 2026-09-16

All24 registered refits completed in39.36seconds. All baseline and artifact replays are exact; both context-omitted geometries give identical scores. All eight branch/omission screens fail individual non-regression. E92 was restored, no deployment, DEV/gallery access, reserve use, data download or Model2 job.

Full-frame adaptive internal comparisons (AI denominator4595; REAL denominator7930):

| Representation | Clean AI caught | Clean REAL alerts | Social AI caught | Social REAL alerts |
| --- | --- | --- | --- | --- |
| All experts (E131) |3739 |1190 |3572 |1300 |
| Without dino |3755 |1153 |3554 |1343 |
| Without clip |3722 |1161 |3608 |1273 |
| Without dear |4023 |1332 |3942 |1392 |
| Without context |3602 |1189 |3467 |1319 |

No omission solves the broad RR REAL shift: full-frame clean RR FPR remains75.04%–86.16%
across omissions. Removing DINO reduces clean REAL errors but increases social errors;
removing DEAR increases aggregate AI recall while adding REAL alarms. Omitting context
loses137 clean and105 social AI detections on net relative to full-frame E131. These are
conditional refit effects with changed dimensions, not proof of a defective encoder or
causal dataset bias. No expert is discarded from E92 or selected for production.

Next bounded mechanism: preserve all experts and compare a fixed smooth worst-source/class
training objective with E131's equal source/class mean. Train weights may depend only on
FIT-group loss; held-out sources never set weights or cutoffs. A separately registered
E138 pilot will retain both geometries, all folds, maps, cutoffs and paired non-regression
checks, with no temperature/penalty search. This targets training imbalance in difficulty;
it cannot manufacture missing generator ancestry or guarantee unseen-source robustness.
Validation:1127 Python tests passed, compilation passed, one existing upstream warning.

## E138 plan — fixed source/class entropic training risk (2026-09-16, before registration/scoring)

E137 did not identify a safe expert removal. Retain all four E131 experts and isolate
the training objective. For each outer FIT partition, compute mean BCE for each
source-component/class intersection using uniform parent/view weights. Minimize
0.5*tau*sum_class(log(mean_group(exp(group_mean_BCE/tau)))) +0.005*||w||^2,
with tau=0.1 chosen as one fixed pilot before scores, not claimed optimal. The gradient
allocates exactly half the mass to each class and a softmax share to its harder FIT groups.
No source-specific serving threshold, label use during inference or E136 consistency term.

Use exact E131 FIT-only maps, three folds, both geometries and six zero-start320-coordinate
heads. Keep the prior L-BFGS limits/tolerances and all four conditions. Lock all predictions
before pooled/per-fold/per-component metrics and paired AI/REAL transition checks at0.5.
Retain both baseline and candidate FIT group-loss/weight diagnostics. No temperature/L2
sweep, adaptive winner selection, final full-TRAIN head or E92 replacement follows.

Research basis: [Sagawa et al., ICLR2020](https://arxiv.org/abs/1911.08731) and the
[authors' repository](https://github.com/kohpangwei/group_DRO), accessed2026-09-16. Their
work cautions that reducing worst-group training loss alone can still generalize poorly,
and studies the role of regularization. E138 is our smooth entropic convex adaptation;
it does not reproduce their neural training algorithm or borrow their measured gains.
Fixed L2 is retained to isolate one mechanism, not advertised as sufficient regularization.

This remains adaptive internal diagnosis after E131/E134/E136/E137 on previously exposed
TRAIN folds; source-group exclusion does not establish unknown generator independence.
No gallery/DEV/reserve reads, new pixels, dataset/package/weight downloads or Model2 work.
Run only on AC power with external disk reserve, and restore exact E92 at completion.

### E138 registered-stage execution — 2026-09-15T21:55:43+00:00

Starting the registered fit stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.

### E138 Model1 source-risk result

{"contract_sha256": "a98371486c6423918dfbaf68dc7cbd9e14df3380a3fcf350addb7f023a70edd6", "downloads": 0, "four_view_score_stability": {"center_control": {"0": {"baseline_mean_four_view_score_range": 0.07215121479782, "candidate_mean_four_view_score_range": 0.07523635573952739}, "1": {"baseline_mean_four_view_score_range": 0.14756244758067666, "candidate_mean_four_view_score_range": 0.15987321354486975}}, "full_frame": {"0": {"baseline_mean_four_view_score_range": 0.07063796249434952, "candidate_mean_four_view_score_range": 0.07306581163107048}, "1": {"baseline_mean_four_view_score_range": 0.14259441615715251, "candidate_mean_four_view_score_range": 0.15570828479970486}}}, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E137, not an E92 serving-model evaluation. Entropic source risk is a fixed convex adaptation, not a reproduction of neural group DRO or a guarantee for unseen groups. No automatic promotion.", "locked_scores_sha256": "1603c1a77424c61a47bb34e9e749e5d0bb7846387c4748d6768300ca09d83336", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control": false, "full_frame": false}, "promotion_allowed": false, "seconds": 31.994000499951653, "state": "E138_Model1_source_risk_complete"}

Full source/condition and individual-transition results: evidence/e138_source_risk.json. E92 remains unchanged.

### E138 registered-stage execution — 2026-09-15T21:56:35+00:00

Stage final state: {"E92_restored": true, "failure": null, "state": "complete"}

## E138 interpretation and Model1 continuation checkpoint — 2026-09-16

All six fixed source-risk fits completed in31.99seconds. Baseline and saved-head score
replays are exact; both geometries fail individual non-regression. The optimization
converged and FIT entropic risk decreased in all six fits, yet transfer got worse in
both clean and social conditions for both geometries. This separates a successful
implementation/optimization from an unsuccessful generalization mechanism.

Full-frame internal comparison with E131 (different models across held-out folds):

| Condition | AI recall E131 → E138 | REAL FPR E131 → E138 | New/rescued AI misses | New/rescued REAL alerts |
| --- | --- | --- | --- | --- |
| Clean |81.37% →79.26% |15.01% →15.62% |109 /12 |53 /4 |
| Social-Q75 |77.74% →74.73% |16.39% →17.06% |148 /10 |59 /6 |

Clean/social AUC falls0.922836/0.898886 to0.911235/0.883008. The worst REAL component
(RR) remains81.60%/85.28% false alerts. The center branch also loses109 clean and157
social AI detections on net and adds46/44 REAL alarms. This particular tau0.1 fixed-L2
pilot is rejected; do not generalize this to all group-DRO methods or claim its training
risk improvement is an OOD improvement. Predeclared locked FIT diagnostics are also
exported to evidence/e138_fit_diagnostics.json for inspection without local artifacts.

Across this continuation: E13724 ablation fits and E1386 source-risk fits completed,
all outcomes retained. No new image data, feature extraction, consumed DEV/gallery or
protected reserve use, Model2 job, candidate promotion or UI score change. Exact E92
API restoration succeeded after each stage. The current bottleneck remains unsolved
source transfer with incomplete upstream ancestry. Before another adaptive coefficient
or expert sweep, prioritize lineage auditing and independent evaluation coverage: resolve
per-file generator/prompt/scene metadata from release manifests, conservatively join
unknown overlaps, and distinguish development acquisitions from an unopened confirmation
reserve before admission/scoring. Existing observed groups cannot serve as fresh evidence.

Validation:1134 Python tests passed in21.68seconds; compilation and diff checks passed.
One existing upstream Starlette/httpx deprecation warning remains. Web source is unchanged
in this continuation; the prior13 web tests are historical validation, not newly rerun
local tests. GitHub CI will validate the exact committed revision. E137/E138 are research
results, not improved serving-E92 scores or evidence of universal detection.

## E139 plan — offline provenance and reserve-status audit (2026-09-16)

Audit only the frozen active E131 TRAIN contract and four already-public E54 MNW/HDR+
manifest/overlap summaries. Record field population by source/class, exact stored body
and pixel identities in separate namespaces, declared scene links and component/fold
crossings. Missing fields do not erase upstream provenance; field presence does not
certify it. Do not decode/rehash images, load features/weights, score or refit any model,
read individual protected reserve records or turn TRAIN into a fresh test. Historical
reserve summaries remain historical; their screening scope cannot automatically certify
later-added training sources. This audit creates no new independent evidence.

Mobile-data constraint remains active: no dataset, weight, package or website download.
Independently check local demo launch/error handling with engineering-only inputs; do
not measure detector accuracy or use the owner's gallery. Preserve exact E92 and Model2
experimental status. Record software checks separately from scientific results.

### E139 metadata audit result

{"AI_bearing_components": 3, "components_crossing_folds": 0, "declared_components": 10, "independent_test_parents_created": 0, "mixed_corpus_AI_rows_without_explicit_generator_field": 1179, "model_scores_created": 0, "parents": 12525, "pixels_read": 0, "stored_identity_links": {"body": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 0, "repeated_identity_groups": 0}, "declared_scene": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 1000, "repeated_identity_groups": 114}, "pixel": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 0, "repeated_identity_groups": 0}}}

Full report: evidence/e139_provenance_audit.json. No new independent test or serving change.

## E139 interpretation and offline demo recovery plan — 2026-09-16

All12525 active parents remain TRAIN (7930 REAL,4595 AI). Stored body identities cover
all parents; stored pixel digests cover only895 REAL parents. No repeated body or pixel
identity was found in the covered fields.114 repeated declared-scene groups span1000
parents, with zero cross-component/fold links. All10 declared components stay inside
their outer fold, with only3 AI-bearing components. These checks neither rehash images
nor test perceptual/semantic duplicates; zero collisions is not independence certification.

The active manifest retains explicit model_name values for500 CommunityForensics AI
rows.69 other CommunityForensics rows and1110 RR AI rows lack an explicit generator/model
field:1179 mixed-corpus AI rows in total. Other source names themselves identify declared
generators, so empty fields must not be interpreted as absence of all upstream provenance.
No AI row stores a prompt/prompt_id/prompt_sha256 in this active manifest. Known shared
prompt batches remain conservatively grouped. Upstream manifests may recover additional
metadata; no family-independent fold or new test is certified here.

Historical aggregate records describe MNW300 AI and HDR+100 REAL as protected, unscored
reserves at their audit checkpoints; both have balanced_final_admitted:false. They are
not automatically a new balanced final test, not300/100 certified independent scenes,
and not available for adaptive tuning. The MNW screen names11630 then-current TRAIN
parents; extending its assurance to today's12525 requires following later admission
checks, not assuming either leakage or safety. No individual reserve identities or
pixels were opened during E139.

Add ml/tools/serve_local_demo.py for explicit external-root preflight, offline flags,
port3002 CORS, exact E92/policy readiness and non-destructive existing-listener handling.
It starts only the local API; the existing web server stays separate. Check-only succeeded
against the running E92. Next engineering check: stop only the verified existing E92
process, launch via this wrapper, and use invalid uploads plus an undersized synthetic
PNG to verify HTTP handling without detector inference. No accuracy estimate or model
selection follows, and no frozen scientific module or serving policy is changed.

## Offline demo recovery and readable Model1 failure map — 2026-09-16

The verified E92 listener was stopped once and restarted using serve_local_demo.py.
Exact artifact/guard/display policy and port3002 CORS returned ready. The existing web
server returns HTTP200. Live engineering checks passed: unsupported MIME415, malformed
image415, oversized declared body413, undersized32x32 synthetic PNG200/uncertain with
no model score and guard not_run, then CORS preflight200. Error cases did not leave the
analysis slot blocked. No actual photograph or native model inference was used.
The first manual malformed-image check incorrectly expected400; the existing API/test
contract specifies415. Corrected that test expectation, not the API, and retained the
initial mismatch in evidence/local_demo_recovery_20260916.json.

Reproducible local API launch (existing virtualenv/dependencies; no installer):
`ml/.venv/bin/python ml/tools/serve_local_demo.py --data-root /Volumes/LaCie/pixelproof-datasets`
Add `--check-only` for preflight/readiness without starting anything. This starts/reuses
only the API; the existing frontend at localhost:3002 runs separately. A foreign or
wrong-identity listener is never stopped. Missing data files fail without downloads;
actual model loading retains E92's hash verification. Leave the launcher running; Ctrl-C
stops only its own child. The successful launch is still running at this checkpoint.

Historical measured error map, with no new scoring in this review:

| Model and population | Condition | AI caught | REAL alerts | Meaning/limit |
| --- | --- | --- | --- | --- |
| E92 consumed DEV | Original |159/160 |0/160 | Limited familiar sources; REAL observations share10 scenes |
| E92 consumed DEV | Social-Q75 |159/160 |14/160 | Compression/resizing sensitivity remains |
| E92 E65 REAL-only diagnostic | Original |Not measured |2/83 | Raw scientific threshold scores across WIFD/RawNIND; consumed, not independent final |
| E92 E65 REAL-only diagnostic | Social-Q75 |Not measured |1/83 | No AI-retention evidence; not the current UI result count or an API-eligible subset |
| E131/E137/E138 separate research heads | Excluded-source diagnostics |See locked reports |Substantial source shift | Not serving-E92 accuracy; inspected repeatedly, no universal claim |

E65 figures are copied from the existing E93 raw_conditions summary, not rerun. Its
older guarded display differs from today's preserve-alerts UI and is not substituted
for raw E92 metrics. This distinction prevents a lower guarded-alert count from hiding
original AI alerts or turning abstentions into correct predictions.

Validation:1140 Python tests passed in22.19seconds; one existing upstream Starlette/httpx
warning. No web source change or fresh local web build was needed. Metadata audit and
runtime engineering receipts are separate from model quality. No download, training,
threshold change, reserve opening or Model2 work. Next: trace later TRAIN-admission
checks against protected reserves and recover upstream prompt/generator-family metadata
where already stored before registering any new independent evaluation.

## E140 plan — later-admission reference-chain reconciliation (2026-09-16)

Follow the preliminary E139 metadata review by reproducing the admission evidence chain.
Bind the current E131 TRAIN contract, E54 base contract, E72/E88/E100 contracts/audits/
TRAIN manifests and public receipts. Hash their declared reference JSON documents as
opaque metadata; do not parse individual protected identities, images, features or scores.
Verify each stage references the exact MNW/HDR+ receipt identities, accepted rows agree
with its report and active TRAIN identities, and all current parents are accounted for
once. Fail on missing references, identity/role drift, overlap or uncovered membership.

The preliminary review found511 MIDD +128 SID +256 later MIDD admissions, whose contracts
name both protected reserves. E140 tests those bindings rather than treating their mention
as sufficient. This is retrospective metadata verification, not a blind performance test
or a new pixel/perceptual audit. A complete chain still cannot prove semantic scene or
base-generator independence, balance the protected evaluation automatically or authorize
score-driven reserve use. Mobile-data constraint remains: no downloads, fit, scoring,
threshold change, Model2 work or runtime interruption.

### E140 admission-chain audit result

{"active_parents": 12525, "balanced_final_admitted": false, "base_parents": 11630, "contract_sha256": "c0c121dfb96ac82ed65588e6cd30fab45aeec09bf60ee1611dd406ef4877dfe0", "downloads": 0, "generator_independence_proven": false, "historical_admission_coverage_complete": true, "image_reads": 0, "later_admitted_parents": 895, "limits": "Historical exact/perceptual screening lineage, not fresh pixel comparison, semantic deduplication, generator independence, balanced final admission or accuracy evaluation.", "model_scores": 0, "new_independent_test_admitted": false, "next": "Membership/reference gap for the895 later additions is closed under stored admission policy. Prompt/scene/base-generator ancestry, protected evaluation protocol and candidate gates remain separate; do not score/tune on reserves automatically.", "stages": {"e100": {"admitted": 256, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 8}, "e72": {"admitted": 511, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 6}, "e88": {"admitted": 128, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 7}}, "state": "E140_admission_chain_complete", "verified_metadata_files": 32}

## E140 interpretation — admission coverage closed, final readiness still separate (2026-09-16)

Verified32 metadata files and joined all12525 current TRAIN parents exactly to the11630
E54 base plus511 E72 MIDD,128 E88 SID and256 E100 MIDD admissions. Each later contract
binds the same protected MNW/HDR+ receipt identities. Its accepted manifest agrees with
its audit and the current parent/body/pixel-original/source/class identities. Historical
reference JSON hashes still match, and the recorded cross-match lists are empty. No
uncovered, duplicated-cohort or identity-drift parent was found. This closes the specific
895-parent admission-chain question raised in E139; it does not newly compare image bytes
or prove semantic non-overlap. Protected reference documents were hashed without parsing
individual records. No scores, images, feature arrays or model weights were read.

Current evaluation-readiness distinctions:

| Requirement | Status | Consequence |
| --- | --- | --- |
| Later TRAIN additions include both protected reserves | Verified through E140 receipt/reference chain | No longer an unresolved895-parent membership/reference gap |
| Stored exact/perceptual screens | Recorded and hash-bound, with no reported overlap | Historical heuristic assurance; not exhaustive scene/prompt deduplication |
| Per-file upstream generator/prompt ancestry | Partial; E139 gaps remain | Six MNW folder names are not six certified unseen generator families; FLUX/OpenAI families already occur in TRAIN |
| Balanced multi-source E52 final | Not admitted | MNW300 AI +HDR+100 REAL is not the existing >=2000-parent balanced multi-source protocol; do not silently substitute it |
| Fresh candidate qualification and frozen one-time evaluation | Pending | Serving E92/consumed DEV success does not automatically authorize adaptive reserve use |

Next local work is upstream metadata recovery for eligible TRAIN records and a concrete
coverage matrix separating declared model names from verified base-family/prompt lineage.
Keep unknowns explicit; no model-name substring heuristic may certify family independence.
No additional download, fit or scoring job was started here. Model2 stays secondary and
E92's serving artifact/cutoffs are unchanged. The existing reserve bodies need not be
re-downloaded. Any future independent evaluation requires its own candidate/protocol and
admission review, preserving the established gates or explicitly documenting a different
research question without calling it an E52 pass.

Validation:1148 Python tests passed in22.20seconds with the existing upstream warning;
compilation and diff checks passed. No web/model-serving source changed. All E140 output
is evidence of record integrity and coverage, not an accuracy improvement.


## Local demo uncertainty explanation and private-photo replay (2026-09-16)

The user reported an uncertain result with displayed E92 scores of approximately
0.01% / 0.39%. Under the frozen policy, two E92 scores below REAL_CUT can still be
uncertain when at least one E43 reference score reaches AI_CUT. The screenshot's
explanation describes this branch; the displayed percentages omit the reference
scores and are not calibrated probabilities. No actual reference values were returned
by the API, so none are invented here.

One raw-byte replay of the user-named desktop JPEG against the existing loopback
E92 API returned no_clear_signal / limited_negative_evidence, review_required=false,
and raw scores 0.00041586352881194593 / 0.00012823884321173178 (0.04% / 0.01%).
Artifact SHA remained 3a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35.
The replay did not reproduce the screenshot. The supplied path disappeared before
an attempted second replay, which failed before sending any bytes. No duplicate
input identity, deterministic repeat, screenshot-input identity, stale-browser bug,
or authenticity ground truth has been established. Keep this discrepancy open;
a future exact-byte replay requires the original available input. Do not describe
this as a repaired model error or an accuracy improvement.

Presentation changes: the primary result now explains the specific uncertainty
cause rather than burying it below the scores; its title is "Sonuç belirsiz".
The result identifies the selected filename and decoded dimensions, and the score
panel explicitly identifies the main model. Details describe both the compression
check and the reference-model check. Small unscored images have separate guidance.
Original AI alerts, guard rules, thresholds, weights, percentages and response schema
are unchanged. This is a consumed user diagnostic, not a benchmark or new training
admission. No photo, filename, body hash, metadata or image-derived feature is added
to Git. No datasets, weights or packages were downloaded; local-only demo scope stays.

The preceding E140 commit 21989ea474fa11eee510dc6eeccce4caea25f914 passed GitHub CI
run35072285478. New UI validation is recorded below when complete.

Validation: Sites build, all14 web tests, TypeScript check, ESLint and git diff
checks passed. The retained localhost:3002 route returned HTTP200, and the API health
reported the same exact E92 artifact ready with downloads disabled. No browser
interaction/visual test or second successful photo inference was performed.


## E141 plan — recover existing TRAIN lineage (2026-09-16)

Recover E32 AI metadata already stored on the external disk into an additive overlay.
Bind current parent ID/source/class to original c3 TRAIN records and native body SHA;
processed inputs also require the original R0 input receipt SHA. Bind each original
source key/body to its realization audit, already referenced by eligibility metadata.
Recover declared model names, source-family declarations, dataset revisions and stored
prompt hashes. Exclude the empty-string hash from prompt links. Reject conflicting or
ambiguous identities; do not guess generator ancestry from model names. Report source
coverage and prompt links across current components/folds, without changing those folds.

Only active E32 AI TRAIN rows join to the private output. No photos, features, weights,
model scores, protected reserve records or new data are opened. Schema/role exploration
preceded registration and is not a blind experiment. Roles, serving E92 and Model2 stay
unchanged. No downloads, training or threshold search. The scientific purpose is to find
missing provenance and possible evaluation dependence before further candidate fitting.


## E141 result — recoverable lineage and remaining coverage gaps (2026-09-16)

Contract SHA: ee668f66549618d6c3a9251c56db33c3195a363fdfbfb5df833ae45695a2653b.
All3005 active E32 AI parents joined without identity conflict through the historical
c3 TRAIN/native-body/processed-receipt/audit chain. The separate overlay is stored only
on the external volume at e141/train_lineage_overlay.json; its SHA is
 d0d636ec405eb0ecee77437cdc8036b84f736e9f544446249b9c89fcd08d5e1e.
No old manifest, fold assignment, artifact or guard was edited.

| Declared source | Active AI parents | Recovered prompt hashes | Declared model identity |
| --- | ---: | ---: | --- |
| CommunityForensics |569|569|569 populated rows,240 distinct names;69 previously omitted names recovered |
| FLUX.2 Klein 9B Base |569|569|Source declaration only |
| GPT Image1 |569|569|Source declaration only |
| Qwen Image2512 |569|569|Source declaration only |
| Gemini2.5 Flash Image Preview / Nano Banana |569|0|Source declaration only |
| Nano Banana Pro |160|0|Source declaration only |
| E36 consumed TRAIN AI |480|Not recovered in this stage|Six declared source labels; scope does not inspect their upstream records |
| RR TRAIN AI |1110|Not recovered in this stage|Per-file generator ancestry remains unresolved |

Recovered2276 nonempty stored prompt hashes among4595 active AI parents.2319 remain
without a recovered hash in this overlay, not necessarily absent upstream.418 repeated
hash groups involve1050 parents;380 groups cross source labels. None crosses the current
frozen E131 component or fold. Existing conservative grouping covers these observed
links. Different hashes do not establish different semantics; preprocessing/normalization
may differ, and the check does not audit all hidden common training ancestry.

The4026 rows without a recovered explicit model_name field are NOT4026 wholly unknown
generators: many have explicit source-level declarations shown above. Conversely, the240
CommunityForensics names are NOT240 verified independent base-generator families. Dataset
revision fields refer to release metadata, not model weight revisions. CF's historical
revision is a local E31-pinned description, not a recovered upstream commit in this stage.

Interpretation: E139's claim that no prompt IDs were stored applies to the old active
manifest. E141 recovers metadata from upstream records without rewriting that evidence.
This closes69 missing CommunityForensics name fields and quantifies known prompt links;
it does not improve AI recall, REAL false positives, abstention coverage or calibration.
The mixed-corpus explicit-model gap is now1110 RR AI parents in the new overlay, down
from1179 when the69 older CF rows lacked names. No independent test parents created.

Next: inspect eligible E36 consumed-TRAIN metadata for the remaining480 known-source AI
parents and document whether local RR metadata can map the1110 mixed-generator parents.
Keep unavailable prompt/model ancestry explicit. A later candidate/evaluation design must
use this coverage evidence and independent groups; do not run another adaptive sweep on
the consumed source folds or score the protected reserve to fill a dashboard.

Downloads, fits, model scores and image reads:0. All existing source roles and licences
are retained; no acquisition/admission or additional use rights are implied. E92 stays
serving with unchanged thresholds, and Model2 remains experimental. The prior UI commit
85590ea61410d8695d64ed48d6278b92866175ad passed GitHub CI run35072884465.

Validation: all1160 Python tests passed in21.05seconds (one existing upstream
Starlette/httpx deprecation warning); compileall and diff checks passed. No frontend
or inference code changed in E141. These checks establish implementation integrity,
not improved detection performance.


## E92 threshold origin and isolated reference-veto audit (2026-09-16)

The user questioned why E92's low paired scores can still show uncertainty. The newly
provided screenshot identifies a different input from the previously replayed desktop
file. The previous same-input comparison was unsupported; the displayed filename now
explains why those two results must not be compared as repeatability evidence. No image
identity/hash or new inference was obtained here, and screenshot provenance is not proved
beyond its visible filename. Do not keep describing those two different filenames as a
confirmed same-file contradiction. No private filenames or screenshot pixels enter Git.

Threshold trace: E48's bounded raw-E43 diagnosis selected0.07940196245908739 using the
lowest CAL threshold satisfying pooled REAL FP<=10% and worst-camera FP<=20%.
E50 then froze that threshold for its development evaluation. E49's low-score cutoff
0.011505939625203613 was chosen from consumed CAL/DEVELOPMENT selective-policy analysis.
E64 and the subsequent E92 correction preserved these constants. Neither cutoff is a
calibrated probability or a proven universal/E92-specific optimum. The old safety budgets
are historical selection criteria, not promises about future photos or desired current
product error rates. E92 itself is an E43 logit plus learned feature correction, not a
wholly independent detector; the separate E43 veto is an additional presentation rule.

The E93/E94 notes motivate guarding original negatives because E92 missed a known AI
that E43 detected. A new read-only comparison isolates the reference veto from the paired
compression guard, using exactly bound, previously consumed E92/E93 scores. Both policies
preserve every original E92 positive and require both E92 views below the low-score cut
for no-clear-signal. Only the E43-negative requirement is removed in the counterfactual.

| Population | Current uncertain | Uncertain without reference veto | Extra reference-only abstentions | AI indications before/after |
| --- | ---: | ---: | ---: | ---: |
| Consumed E66 REAL160 |89|21|68|0/0 |
| Consumed E66 AI160 |1|1|0|159/159 |
| Consumed E65 REAL83 |23|3|20|2/2 |

The known missed AI has E92 original score0.00857941528582679 and social-Q75 score
0.6113029556660701; it therefore remains uncertain from the compression guard alone.
The earlier verbal rationale must not imply the E43 veto uniquely protected this case.
Observed additional AI protection from the reference-only veto is zero in these limited
consumed populations. Reference-only abstention costs are42.5% of E66 REAL and24.10% of
E65 REAL. These are not overall app uncertainty rates: the populations are dependent,
consumed, source-limited and not filtered by current API eligibility. E65 has no AI.
Absence of observed incremental benefit does not prove the veto is useless on all OOD.

Aggregate evidence: evidence/reference_veto_audit_20260916.json, with exact input-score
and policy-code hashes. No new inference, fit, acquisition, threshold search or serving
change. Model1's score and its UI review policy remain distinct. Recommended next product
candidate: E92 paired result as primary, reference disagreement as a separate advisory
rather than an automatic veto. Register and compare that display policy explicitly before
adoption, preserving original AI alerts and compression/borderline uncertainty. Any future
threshold/probability calibration requires a separate source-disjoint calibration design
and locked independent evaluation; do not optimize on this private photo or consumed tests.

E141 commit5ee957682767e9906ecbcc14f7d39d602be34666 passed GitHub CI run35073605408.


## E92 primary display v2 implementation plan — 2026-09-16

The user explicitly authorizes making the demo use E92 as primary with the E43
reference warning advisory, and requests a report. Preserve the frozen weights,
preprocessing, cutoffs and historical E93/E94 policies. Add a new versioned display
adapter: original E92 >=AI_CUT stays an AI indication; both E92 views <REAL_CUT yield
no_clear_signal regardless of reference; all other original-negative combinations
stay uncertain. E43 >=AI_CUT on either view yields a separate reference_ai_warning.
Original positives with compression disagreement keep their review warning.

Bump the live API/client to schema4 and e92-primary-reference-advisory-v2 with
paired guard e92-paired-v2. Reject stale schema/policy payloads. Keep missing/invalid
inputs and failed inference fail-closed. Test cut boundaries and advisory invariance,
replay all403 bound consumed pairs, then restart only the verified E92 listener and
perform loopback integration. The now-available user-named screenshot input can be
replayed before/after with a private byte-hash binding; publish aggregate results only.
The site remains local-only; no data/package/weight acquisition or publication.


## E142 plan — E36 TRAIN prompt identity recovery (2026-09-16)

After making the demo current, resume Model1 provenance work without downloads.
Existing E36 consumed CAL metadata retains prompt_id fields omitted by the current
TRAIN manifest. Register a source/parent/body join restricted to currently active E36
AI TRAIN rows. Recover local declared prompt IDs and count cross-component/fold links.
Reject role/body/ID mismatches and duplicate keys. A shared numeric prompt ID is only
meaningful within this publisher namespace, not proof of identical text across corpora.
Do not open E36 FINAL, protected reserves, pixels, model scores or weights. Preserve all
frozen manifests and create only a separate private overlay and aggregate public report.


## E92 primary display v2 — implemented and running locally (2026-09-16)

The user-approved display policy is now e92-primary-reference-advisory-v2, schema4,
paired guard e92-paired-v2. The new primary_demo_policy adapter leaves historical
E93/E94/demo_policy/demo_scores code intact. E92's model artifact, native preprocessing,
raw scores and AI/low-signal cutoffs are unchanged. E43 disagreement appears as a
separate advisory and cannot veto the E92 paired decision. Positive instability and
borderline/compression uncertainty remain. The frontend rejects old schemas, and a hot
update hides retained results from the previous policy until a new analysis completes.
The launcher requires the new policy before reporting readiness. Legacy frozen smoke
scripts remain historical and must not be used to assert the live schema4 contract.

Replay on all403 previously locked consumed pairs (primary_demo_v2_replay.json):

| Population | Before uncertain | After uncertain | Before/after AI indications |
| --- | ---: | ---: | --- |
| E66 REAL160 |89|21|0 /0 |
| E66 AI160 |1|1|159 /159 |
| E65 REAL83 |23|3|2 /2 |

All88 changed results are reference-veto uncertainty becoming no_clear_signal with a
separate advisory. No score or original E92 AI indication changes. These are dependent,
consumed diagnostic populations, not new OOD evidence, current app-wide coverage or an
increase in classifier accuracy. The two E65 false AI indications remain false alerts.

Same-byte private-photo loopback check: the screenshot input is now available. Under
schema3 it reproduces uncertain with scores0.0001069483959563185 and0.003934696542018971.
After restarting only the verified port8800 E92 API, schema4 returns no_clear_signal
and reference_ai_warning=true, review_required=false, with exactly identical scores
and artifact identity. The UI formats these as0.01% /0.39%. CORS for localhost:3002
passes. The actual HTTP payload passes the TypeScript parser; its old response is
rejected. The input hash and detailed private responses stay on the external disk;
only sanitized outcome evidence is committed in primary_demo_v2_http.json. This is
engineering verification, not proof of the photograph's authenticity.

The frontend remains at localhost:3002 and the API is ready on127.0.0.1:8800. No public
hosting, dependencies, datasets or weights were downloaded. Browser interaction/visual
QA was not performed; built SSR, HTTP and response-contract checks are reported instead.


## E142 result — remaining E36 prompt metadata recovered (2026-09-16)

Main Model1 work resumed after the demo update. Contract SHA
 d183adbda1fa99e989300859b06d12179915a7f6e2330a3240f5eddfea627511 binds current E131
TRAIN and historical E36 CAL metadata.480 existing AI TRAIN parents (80 per each of
six declared sources) match source/parent/body/old-role identities. Recovered100
publisher-scoped prompt groups:28 groups have6 parents,37 have5,24 have4,9 have3,
and2 have2. No group crosses a registered component or fold. The private overlay SHA
is a349aca026ec85e14c9a658be51689424de6b4713da56841a9195f43c7040895; no inherited
manifest or role was rewritten. The original consumed-CAL role is historical; eligibility
comes only from the current TRAIN contract, not a new admission made by this audit.

Together with E141,2276 active AI parents now have recovered stored prompt hashes and
another480 have publisher-scoped prompt IDs. These are different kinds of evidence and
cannot be compared across namespaces without prompt text or authoritative mappings.
1839 active AI parents remain without either recovered link in these overlays:729 Nano
source parents and1110 RR parents. Base-generator ancestry remains unverified. Next is
local RR per-file metadata availability, without inferring model families from filenames
or consuming protected test reserves. No images, fits, scores or downloads in E142;
Model2 remains experimental and E92 stays serving under the new display-only policy.

Validation: final1169 Python tests passed in21.12seconds (one existing upstream
Starlette/httpx warning);14 web tests, Sites build, TypeScript, ESLint, compileall and
diff checks passed. Actual schema4 photo response passed the frontend parser and the
old schema3 response was rejected. The retained frontend returned HTTP200; the local
launcher confirms exact E92 and the current advisory policy with downloads disabled.


## E143 plan — local RR TRAIN metadata availability (2026-09-16)

Resume Model1 work after the site update. Existing RR train/validation inventory records
3000 images and no non-image sidecars; this is a historical inventory, not a new archive
scan. Verify its acquisition/extraction metadata chain and bind only currently active
RR TRAIN parents to their original train member/source/class/body identities. Read those
bounded local bodies to recheck SHA/size, then inspect lazy image headers and EXIF already
present in the header. No pixel decoding, model/feature extraction, gallery, official
validation/test image or protected reserve access. Use a separate private record file;
only counts/hashes reach Git. No datasets, weights or packages downloaded.

Selected PNG text keys and EXIF software/camera/description fields are untrusted
self-declarations. A camera Model tag is not an AI generator. Empty/missing metadata
cannot certify authenticity, and header-only inspection may miss later container fields
such as PNG post-IDAT text. Count failures without removing parents. Freeze code and
input metadata before this image-header pass; do not modify TRAIN roles/folds or infer
generator families from topic/filename strings. This audit must lead to an explicit
local-provenance availability conclusion, not repeated unbounded searches or a new
claim of unseen-generator evaluation.


## E143 completed — RR local provenance availability (2026-09-16)

Verified all2360 active RR TRAIN bodies (1250 REAL,1110 AI), totaling1,721,292,228
bytes, against their original extraction/acquisition identities. Header/EXIF parse
errors:0. Selected fields occur in525 REAL and135 AI parents. The original containers
are REAL:1249 JPEG/1 PNG; AI:113 JPEG/997 PNG. These strongly unequal distributions
motivate a bounded processing-confound diagnostic; they do not prove that a model uses
container identity or that generator/content differences are unimportant.

A separately recorded post-hoc parser finds110 exact DALL-E producer declarations and
one model-name declaration in generation settings. These111 strings remain unverified
file self-declarations;999 RR AI parents lack either parsed convention. Camera and
Matplotlib software tags are not generator identities. No verified checkpoint/base-family
ancestry was recovered, no folds/labels changed, and1839 active AI parents still lack a
recovered prompt hash or publisher-scoped prompt ID. Header-only inspection can miss
later container fields; this closes the bounded local-header availability question,
not all provenance uncertainty. Do not repeat unconstrained metadata searches.

Public aggregate receipts: evidence/e143_rr_metadata.json and
 evidence/rr_header_declarations_20260916.json. The frozen E143 contract SHA256 is
9bb8135d43ef4c5c0848f689406fc55e27a982717aa9fa649b2335d85eddb4f6.
Raw metadata text and per-parent overlays remain on the external disk, outside Git.
No downloads, pixel decoding, model scoring, fits, protected-image access or promotion.
E92 remains the live primary demo; Model2 remains experimental. The preceding site/code
commit b1ac736 passed GitHub CI35076273276.


## E144 plan — container strata in locked RR source-held-out scores (2026-09-16)

Before viewing subgroup scores, register a bounded diagnostic joining all2360 E143 RR
TRAIN header identities to the already locked E131 predictions. Preserve complete parent,
class, source, fold, condition and score hashes. Report every JPEG/PNG x REAL/AI stratum,
both center-control/full-frame branches and all four existing processing conditions.
Use E131's fixed0.5 diagnostic cut and complete raw-score distributions, denominators,
error counts and paired clean-to-processed new/rescued errors. No subgroup selection,
threshold search, inference, pixels, fits, downloads or protected-reserve access.

This is retrospective internal TRAIN evidence, not deployed E92 performance. E92's
7.94% warning cutoff is unrelated to this head's0.5 cut. Only one REAL PNG prevents a
meaningful balanced within-PNG comparison. Container, content and generator are
confounded; existing decoded-pixel scores cannot establish causal extension reliance.
Use the result to specify a later matched processing intervention, not to promote a
classifier or infer authenticity from metadata/file format.


## E144 completed — format does not explain away RR false alarms (2026-09-16)

Contract65b619fec1272f1d1cad209c803cc2b59a7e71530a16c7dcf89a7bb6469cbc10 froze the
complete2360-parent join before subgroup metrics. All32 branch/class/format/condition
summaries are retained in evidence/e144_rr_format_scores.json. No missing parents,
new image reads, inference, fits, downloads, cutoff changes or serving promotion.

Full-frame E131, fixed0.5 diagnostic cut:

| Original class/container | Parents | Clean AI-alert count/rate | Social-Q75 AI-alert count/rate |
| --- | ---: | ---: | ---: |
| REAL JPEG | 1249 | 1009 /80.78% | 1058 /84.71% |
| REAL PNG | 1 | 1 /100% | 1 /100% |
| AI JPEG | 113 | 111 /98.23% | 111 /98.23% |
| AI PNG | 997 | 927 /92.98% | 941 /94.38% |

For REAL, an AI alert is a false positive; for AI it is a true positive. The center-control
branch also has severe REAL JPEG FPR79.74% clean and84.31% social. Full-frame social
processing introduces67 new REAL JPEG false alarms and rescues18; AI PNG processing
introduces5 misses and rescues19. AI JPEG has one new miss and one rescue despite the
unchanged aggregate count. Thus aggregate gains can conceal individual regressions.

The simple claim that this held-out head merely calls original PNG AI and original JPEG
REAL is inconsistent with these predictions: most REAL JPEGs are also called AI. A
broader source-dependent score shift remains the actionable problem. This does NOT
rule out visual codec/processing-history shortcuts or prove a causal mechanism; generator,
content and processing remain confounded. Only one REAL PNG makes that row descriptive,
not a reliable population-rate estimate. One already consumed RR component cannot
support source-general confidence intervals or unseen-generator claims.

These are E131 source-excluding research heads, not E92 serving measurements. No new
candidate has passed qualification and no detector-accuracy improvement is claimed.
Keep E92 and the updated primary/advisory demo policy unchanged.


### E143–E144 implementation verification — 2026-09-16

All1178 Python tests passed in21.26seconds, with one pre-existing upstream
Starlette/httpx deprecation warning. The nine new tests cover bounded lazy header
inspection, untrusted declaration parsing and strict TRAIN/header identity joins.
Compilation and whitespace checks passed. The local launcher confirms the exact E92
artifact and current primary/advisory policy with downloads disabled; localhost:3002
returns HTTP200. No web source changed in this work, so the prior successful web build
is retained. These checks validate software/identity handling, not universal detection.


## E145 plan — nested source-split feasibility (2026-09-16)

Inspect only the current bound E131 TRAIN roster and declared components/folds. Enumerate
all six ordered assignments of the existing three folds to FIT/CAL/EVAL. Require unique
TRAIN parents, both classes in every role and no component crossing; preserve all four
views of each parent. Publish class/component support and source names, never parent IDs.
No score, image, fit, download or reserve access. This tests engineering feasibility;
three AI-bearing components and unknown RR/CF ancestry cannot support independent
unseen-generator calibration. The available sources are already development-consumed.


## E145 result and E146 registered design — 2026-09-16

The E145 metadata-only check finds all six ordered FIT/CAL/EVAL assignments structurally
feasible. Existing folds0/1/2 contain5383/3985/3157 parents: REAL2467/2875/2588 and
AI2916/1110/569. Each has one AI-bearing component; REAL component counts are2/3/3.
Known components remain intact. These small source counts and unknown RR/CF ancestry
limit the experiment to adaptive internal development; no independent validation claim.

Proceed with one bounded E146 experiment using existing raw feature caches only. Fit
three fresh full-frame heads, one on each FIT fold. Refit every standardizer and PCA on
that FIT fold exclusively; never reuse the E131 maps that included the new CAL fold.
Retain E131's DINO/CLIP/DEAR64 PCs each plus full-frame128 PCs, whitening seed131,
class/component/parent/view weights, logistic L2 .01 and convergence guards. Three fits,
no hyperparameter search; all four processing views stay with their parent/source.

For each of six assignments, select one common CAL cutoff meeting pooled REAL FPR<=10%
and each REAL-component FPR<=20% in every condition. These are fixed empirical diagnostic
budgets, not a claim of acceptable product risk or statistical guarantees. Choose the
smallest feasible value by exact order statistics, handling >=cut ties with nextafter.
Reject it if any condition loses a .5-caught CAL AI or falls below95% CAL AI recall.
Increasing the cutoff cannot repair those failures; no EVAL-driven fallback or retuning.

Lock all three non-FIT prediction arrays before CAL selection, then all six selections
before EVAL metrics. Retain every assignment, including rejected CAL cuts, and compare
accepted cuts with the same fresh head's .5 baseline. Report per-condition/component
counts, AI recall, REAL FPR and individual new/rescued errors. EVAL acceptance requires
the same absolute budgets/recall plus zero new AI misses and zero new REAL alerts in
all conditions. Do not pool repeated parents from the six assignments. Less FIT data
than E131 means comparing its earlier heads would conflate training size and cutoff.

No downloads, new image pixels, protected-reserve reads, gallery/DEV access or E92
serving change. Local AC power and external disk reserve are available. Raw artifacts
remain external; only aggregate evidence/code enter Git. Maximum run3600seconds,
CPU thread limit2, no GPU encoder allocation. Model2 remains experimental.


### Current-versus-historical runtime review — 2026-09-16

Inspection found that MODEL_CARD.md, ml/SERVING.md and PRESENTATION_EVIDENCE.md could
still be read as current E20/E26/E31 guidance. Added explicit historical notices and the
actual E92 local launch/health/API contract; retained their historical results unchanged.
No new presentation/report is claimed. Canonical ongoing records remain PLAN.md,
HISTORY.md, ml/EXPERIMENTS.md and DATASETS.md. Frozen scientific reference implementations
and their historical receipts remain untouched.

Also closed a launcher readiness gap: exact artifact/policy/CORS checks now additionally
require schema_version4. A listener with old schema3 cannot be reused as the current
frontend-compatible service. This does not alter scores or model weights. The focused
three launcher tests and actual --check-only pass; the existing E92 process was reused.

Pre-run/full implementation verification:1183 Python tests passed in21.75seconds with
one pre-existing Starlette/httpx warning. The five new nested-calibration tests include
synthetic end-to-end checks that all12 PCA fits see only one FIT fold, no FIT predictions
are emitted, every role permutation is retained and EVAL metrics follow both locks.
Cutoff tests cover ties, infeasible AI-retention constraints, worst-source constraints
and every processing condition. The previous b86ccab GitHub CI35078418953 passed.
E146 is registered under contract2786f38e8404e94d2e7c26c6d3fe1b93f4ad663f0211bd517489acab03fc843d;
execution is in progress and no measured detector improvement is yet claimed.


## E146 result — CAL-only thresholds fail source transfer (2026-09-16)

Completed three fresh full-frame heads in39.306seconds after input verification. All
three saved-artifact score replays are exact0; solvers converge in21/29/24 iterations.
Each standardizer/PCA/head used only its FIT fold; all non-FIT predictions were locked
before all six CAL selections, and those selections were locked before EVAL metrics.
Contract2786f38e8404e94d2e7c26c6d3fe1b93f4ad663f0211bd517489acab03fc843d;
full aggregate evidence: evidence/e146_nested_calibration.json.

Three of six CAL assignments are accepted, but zero of six assignments pass the complete
EVAL gate. The three rejected CAL cuts are not applied to EVAL. At FIT0/CAL1 the necessary
REAL-budget cut0.97347 loses255/383/307/309 previously detected AI views (by condition);
FIT2/CAL1 cut0.81802 also loses AI. FIT1/CAL0 cannot reach95% AI recall while meeting the
REAL budgets. These are separate condition counts, not independent unique-parent totals.

Accepted CAL assignments, evaluated only on the remaining source fold:

| FIT/CAL/EVAL | CAL cut | EVAL clean REAL FPR, baseline -> cut | EVAL clean AI recall, baseline -> cut | New clean REAL alerts |
| --- | ---: | ---: | ---: | ---: |
| 0/2/1 | 0.471254 | 49.67% ->51.17% | 95.68% ->96.04% | 43 |
| 1/2/0 | 0.250869 | 8.67% ->20.31% | 65.05% ->78.19% | 287 |
| 2/0/1 | 0.261082 | 20.24% ->31.62% | 91.35% ->96.58% | 327 |

All three accepted CAL cuts are below the same-head0.5 baseline, so they preserve caught
AI while adding REAL alerts. Social-Q75 REAL FPR becomes51.83%,20.75%,37.11%, respectively.
Every condition/component and transition is retained, including the failures. Results
from repeated EVAL parents under different heads are not pooled into one accuracy.

Interpretation: the registered lowest-budget-feasible CAL rule fails to transfer in these
six development assignments. The rule prioritizes recall subject to a REAL budget; this
is not proof that every possible threshold-selection objective is impossible. It does
show that success on one calibration source cannot be treated as a new-source guarantee.
Do not retrospectively choose a different cut from these exposed EVAL scores, present
this as independent validation, or replace E92. The next scientific change should target
representation/coverage, with a separately specified control, rather than another cut
sweep on these folds. The one-AI-component-per-role limitation remains fundamental.

No downloads, new image pixels, new data admission, protected reserve access, serving
weights or UI decision changes. E92 remains live; Model2 remains experimental. This is
an implemented and falsified calibration proposal, not a detector-accuracy improvement.


### E146 independent receipt/count verification — 2026-09-16

Verified public/private report equality, locked-score receipt/NPZ hashes, locked CAL
receipt hash, complete parent order and TRAIN role. An independent direct Boolean recount
of60 condition records (24 CAL,24 baseline EVAL,12 accepted-cut EVAL) matches every REAL
FP, AI TP, newly missed AI and new REAL alert count. No new inference or image read.
Compilation and whitespace checks passed. After the launcher schema fix, all three
focused launcher tests passed; exact E92 --check-only succeeds and localhost:3002 returns
HTTP200. The full1183-test run preceded only that one readiness check change, which the
focused tests cover. No experiment is still running and no candidate was promoted.


## Deep project audit — scope and reproduced failures (2026-09-16)

User requested a broad internal search for overlooked weaknesses before more model
changes. Review covers current serving/input lifecycle, artifact identity, frontend result
binding, training/DEV parent separation, source-fold normalization, metric/retention
counting, preprocessing parity and the distinction between numerical gates and independent
validation. No new fitting, data download or protected-reserve access is authorized by
this audit plan; existing user-authorized development continues locally.

Confirmed failure1: cancellation while awaiting decode_photo released the API semaphore
although the decoder thread was still running. A deterministic synthetic ASGI cancellation
regression reproduced a second HTTP200 while the first decoder remained blocked, instead
of429. Repair makes a single shielded worker own decode plus inference and release the
slot only on completion; timeout/validation/error paths preserve release semantics.

Confirmed failure2: the historical E92 loader verified files against a mutable manifest
without pinning the manifest itself. An empty altered manifest reached the weight
loader, bypassing the intended identity checks. This does not show that installed weights
were actually altered; it demonstrates a reproducibility/integrity gap. Add a separate
VerifiedE92Engine wrapper with the original manifest SHA pinned before deserialization.
Preserve the historical loader/receipts. The active primary engine uses the wrapper;
health and launcher require the verification revision and manifest digest.

Next bounded read-only check: verify264 manifest-bound code files and five data artifacts,
reconstruct12269 historical E92 TRAIN parents from bound manifests, compare them and all
12525 current TRAIN parents against320 consumed E66 DEV parent/body/pixel identities,
then recount the640 locked DEV views and current paired-policy outcomes. Report missing
hash coverage explicitly. Compare historical/runtime social processing on four synthetic
JPEG inputs including EXIF rotations, a high-resolution image and a narrow aspect ratio.
No classifier inference or dataset image pixels; raw metadata/parent details stay outside
Git. This is an engineering/scientific audit, not another independent benchmark.


## Deep project audit — results and development priorities (2026-09-16)

Aggregate evidence: evidence/project_audit_20260916.json and
 evidence/representation_weighting_audit_20260916.json. No new detector training,
dataset download, protected-reserve access or model promotion in this review.

### Confirmed current-runtime defects, fixed

1. Cancellation during image decoding prematurely released the one-worker admission slot.
   Reproduced before repair: the second request received200 while the cancelled request's
   decoder remained blocked. Decode and inference now share one shielded worker/slot;
   cancellation or response timeout cannot admit another heavy job until it finishes.
   Normal decode errors still return their original4xx status and free the slot.
2. E92's historical loader trusted the identity manifest's own expected hashes. Replacing
   its file lists with empty dictionaries reached deserialization without verification.
   No installed weight corruption was found. The active wrapper now pins the original
   manifest SHA6bf3e29c8c4bada93d975513f31e8acf8b615d6076b52a53239ccbe06d48a553
   before invoking the unchanged historical loader. Health/launcher identify and require
   e92-manifest-pinned-v1. This closes accidental/self-declared manifest substitution;
   it is not a security boundary against someone who can rewrite trusted application code.

### Newly quantified research assumption, not a proven cause

E131's classifier balances classes/components in its loss, but holdout_linear.fit_map
uses unweighted mean/std and unweighted PCA. The REAL fraction entering those maps is
76.49%,59.19%,57.02% across the three outer-FIT populations, versus50% REAL loss mass in
every fitted head. Individual source/component masses also differ. E136-E138 reused
these maps; E146 refitted maps but retained uniform row weighting. Consequently, their
loss reweighting experiments did not test source-balanced representation fitting.

This is consistent with the frozen implementation and protocol, not test leakage or a
newly discovered arithmetic error. It is a plausible representation bottleneck to isolate,
not proof that majority-source variance caused the failures. Next priority is one fixed
FIT-only weighted normalization/covariance-PCA control at unchanged dimensions, folds,
classifier objective and cutoff, with the original unweighted baseline and per-image
AI/REAL non-regression checks. Do not change historical maps, choose weights from held-out
results or promise improved accuracy. This control takes priority over an additional
native-residual feature extraction until its narrower hypothesis has been tested.

### Integrity/measurement checks that passed, with coverage limits

-264 frozen code files and five E92 data artifacts match the pinned runtime manifest;
  the current fitted artifact matches the declared E92 SHA. This is actual-file hashing,
  not a successful health response alone.
-Reconstructed all12269 E92 TRAIN parents and matched their source/label/body identities
  to the current12525-parent roster. Neither TRAIN population intersects the320 consumed
  E66 DEV parents by stored parent IDs or body digests. Stored pixel digests also show no
  intersection, but only639 E92 TRAIN /895 current TRAIN parents have them, compared with
  all320 DEV parents. Most TRAIN encodings therefore lack that comparison. No fresh image
  hashing, semantic/perceptual matching or pretrained-corpus audit was performed. Do not
  turn this into a complete decontamination or independence claim.
-All640 stored E92 DEV views reproduce the published metrics exactly. The20 numeric
  gates still pass; the separate original-view E43 AI-retention guard still fails on
  one parent. Originals:0/160 REAL false alerts,159/160 AI detected. Social-Q75 alone:
  14/160 REAL false alerts,159/160 AI detected. Those are per-view score measurements,
  not the paired web verdict. Current paired UI: REAL139 no-clear/21 uncertain/0 AI;
  AI159 AI/1 uncertain/0 no-clear. These are consumed development observations, not an
  independent success rate.20 gates are20 conditions, not20 independent datasets.
-Four synthetic JPEG inputs, including EXIF6/8 rotation,3000x2600 resolution and224x4100
  narrow geometry, give exact historical/runtime social-transform pixel parity. This
  finds no skew for these cases, not all formats or inputs. The narrow image becomes
  59x1080 after social resizing: derived-view upsampling/limited detail remains a support
  caveat for future robustness tests, not evidence of a transformation mismatch.
-Reviewed current client request gating, stale-schema rejection, paired decision checks,
  upload bounds, failed-inference handling and FIT-only maps. No new confirmed failure
  was found in those inspected paths beyond the two runtime defects above. This is a
  scoped code audit, not a claim that all code is bug-free.

Known unresolved scientific limits remain: repeated development-source use, only three
AI-bearing source components, unverified mixed-corpus generator ancestry, and no independent
balanced final pass. Model2 remains experimental. E92 weights/cuts/decision policy remain
unchanged; the repairs improve correctness and reproducibility, not measured detection.


### Audit completion, Model2 scope and live verification — 2026-09-16

Reviewed the Model2 patch split/weighting and spatial evaluation helpers as well:
source/scene/body components remain together, masks are supervised targets/evaluation
inputs rather than inference inputs, classical edits are AI-negative, and unavailable
positive/interior endpoints remain explicit. Existing small-parent/single-editor/location
failures remain unresolved; no new localization success or candidate is claimed.

All1189 Python tests passed in20.44seconds (one existing Starlette/httpx deprecation
warning); compilation and whitespace checks passed. Six added tests cover cancelled and
timed-out decoders, manifest substitution before deserialization, exact manifest acceptance,
cross-encoding body/pixel overlaps and missing identity coverage. Prior CI35079701049
for80ea13c is successful. No web source changed in this audit.

Restarted only the verified local E92 worker to activate the fixes. The new worker loads
the exact unchanged artifact, serves schema4 and e92-manifest-pinned-v1 with the pinned
manifest digest, and passes launcher/CORS checks. Real loopback requests give415 for a
malformed body followed by200 for a223x224 synthetic PNG with image_too_small and no
score, proving error-path slot release without running detector inference. Frontend
localhost:3002 returns200. Receipt: evidence/project_audit_runtime_20260916.json.
No ML experiment is running; the next weighted-representation control is planned, not
trained. The local E92 site remains running and Model2 remains experimental.


## E147 registration — FIT-weighted representation control (2026-09-16)

Hypothesis: E131's unweighted normalization/PCA may preserve the largest source/class
population disproportionately even though its classifier loss is balanced. The audit
found REAL PCA row mass76.49%/59.19%/57.02% versus50% loss mass. This motivates a control,
not a causal finding. E147 changes only the FIT representation weighting bundle: weighted
population mean/std and covariance PCA use the existing class/component/parent/view
weights. Covariance divides by1-sum(w²), recovering the original sample convention for
uniform weights; this is not a claim that related views are statistically independent.

Before fitting, freeze all12525 existing TRAIN parents, four conditions, three E131
source folds, cache/artifact/code hashes, seed131, ranks64/64/64/128, whitening floors,
randomized SVD power3/oversamples10 and the unchanged weighted convex classifier objective,
solver and fixed0.5 diagnostic cut. Six fits only; no cut/rank/weight search. Replay old
E131 held-out predictions and new serialized shared/branch maps plus heads within1e-10
with identical decisions. Lock complete predictions before metrics. Report every
condition/fold/component, worst groups and paired new/rescued errors. Both zero new AI
misses and zero new REAL alerts across all conditions are required for individual
nonregression; even a pass does not authorize deployment.

Use existing caches only, CPU2threads,7200-second budget, AC/disk guards and denied network
connections. No dataset/weight/package downloads, image-pixel extraction, gallery/DEV or
protected-reserve scoring. Six targeted synthetic checks passed before registration,
including direct weighted-covariance eigenvalues, uniform-weight parity and held-out
perturbation proving fitted-map isolation. These are implementation checks, not model
quality results. E92 serving stays unchanged. This adaptive internal control still has
only three AI-bearing components and unresolved upstream/pretraining overlap; it cannot
prove universality. Public evidence will contain aggregates only; private artifacts stay
under the external disk's e147 directory.


## E147 result — weighted PCA does not pass preservation (2026-09-16)

Completed all six registered fits on12525 TRAIN parents (7930 REAL/4595 AI), four views
and three existing source folds in46.34 seconds after input verification. All old E131
and new serialized-map/head replays have maximum score difference0 and zero decision
changes. Every FIT representation now has exactly50% REAL weight. Both branches fail
individual nonregression; no weights, thresholds or serving policies were promoted.

Percentages below describe fresh research heads pooled across three different outer-fold
models, not the active E92 model. Counts are paired per-condition errors, not independent
images across views. The diagnostic cutoff remains0.5.

| Branch / condition | REAL FPR old → new | AI recall old → new | New / rescued REAL errors | New / rescued AI misses |
|---|---:|---:|---:|---:|
| center_control / clean | 14.74% → 15.42% | 80.04% → 81.35% | 82 / 28 | 62 / 122 |
| center_control / assigned_transport | 15.78% → 16.28% | 79.04% → 80.61% | 88 / 48 | 81 / 153 |
| center_control / q75 | 14.87% → 15.37% | 75.73% → 77.65% | 68 / 28 | 71 / 159 |
| center_control / social_q75 | 16.20% → 16.81% | 75.91% → 77.61% | 80 / 32 | 71 / 149 |
| full_frame / clean | 15.01% → 16.05% | 81.37% → 82.05% | 99 / 16 | 80 / 111 |
| full_frame / assigned_transport | 15.85% → 16.75% | 80.89% → 80.54% | 116 / 45 | 119 / 103 |
| full_frame / q75 | 15.26% → 16.05% | 77.71% → 78.72% | 84 / 21 | 81 / 127 |
| full_frame / social_q75 | 16.39% → 17.33% | 77.74% → 78.52% | 101 / 27 | 87 / 123 |

Interpretation: the center branch increases pooled AI recall in every condition but
increases REAL false alarms in every condition and introduces new AI misses. The full-
frame branch also loses pooled AI recall on assigned transport. Balancing PCA is therefore
not a free correction for source shift. This rejects this fixed weighting bundle under
our acceptance requirements; it neither proves PCA weighting useless generally nor
isolates normalization versus eigenspace effects.

The RR component remains the worst REAL source group: clean FPR79.76%→82.96% for center
and80.80%→84.72% for full-frame. These are within-group REAL denominators, not all2360
RR component parents (which include AI). Do not hide this under pooled improvement.
Three AI-bearing components, adaptively consumed TRAIN folds, unknown RR/CF ancestry and
unknown frozen-encoder pretraining overlap prevent independent universality claims.

Historical-mechanism review before the next representation proposal: E31 forensic68
features had weak AI recall and ensemble tradeoffs; E51 residual/DCT32 additions did not
solve cross-source transfer (CAL-selected A had no residual branch); E53 mean-only/native
and transform controls lost individual sources; E69 fixed28px shuffled DINO crops missed
the TRAIN FPR ceiling. None justifies renaming another transform or cutoff sweep as a
new method. Next prepare a genuinely distinct native-scale processing/residual control,
first confirming source-native versus already-resized body coverage without score-based
selection. Freeze extraction, matched class processing, source folds and all acceptance
rules before fitting. Current data cannot supply independent new-family proof by itself.

Evidence: evidence/e147_weighted_representation_contract.json and
evidence/e147_weighted_representation.json. Contract SHA256
6d046984df6ff82fd0d387e770673f27db8719da5a695e638ce96c413715a24d. Private maps/heads/scores stay on LaCie.
Validation:1195 Python tests passed in21.18s (one existing Starlette/httpx warning);
live frontend HTTP200 and API ready with E92/schema4/current pinned-manifest policy.
Zero downloads, new image pixels, DEV/gallery or protected-reserve scores.


## E148 registration — source-body geometry and processing inventory (2026-09-16)

After E147's rejected weighted maps, inspect whether a native-scale residual proposal is
actually supported by the current inputs. Code review establishes that E54's native flag
identifies its expansion cohort; false does not mean resized and true does not certify
untouched camera/generator pixels. Explicit SCIMD publisher-224-resize and E32 fixed-replay
provenance mark known224 derivatives; SID path/original-hash/decode records mark local RAW
renders. All other admitted source bodies retain unknown upstream processing history.

Freeze all12525 E131 TRAIN parents before the audit. Verify every full body SHA256 and
available recorded size, then inspect only lazy format/dimensions and already-present
EXIF orientation. Do not decode pixels, extract metadata text, infer labels/generators,
read feature/prediction archives, score models or touch gallery/CAL/DEV/protected reserves.
Report each class/source/fold, known processing groups, header errors, declared-dimension
mismatches,224/256/512/1024 source-pixel crop capacity and how many exceed the existing
2048px local-texture cap. Small/derived/error cases remain counted, not dropped. Capacity
is geometry only, not proof of native upstream resolution. No fitting or promotion.

The preliminary stat-only inventory covers27.09GiB, maximum52.45MiB/body;5652 inherited
records have no bytes field but all have a bound body SHA. The registered run allows at
most100MiB/body,100MP/header,1800seconds, requires AC/disk reserve, and denies network
connections. Source historical code snapshots are hash-bound; where E131 already pins a
source-code digest it must still match. This inventories recorded provenance, not a new
independent upstream certification. Six synthetic checks passed before freeze: no pixel
load, no PNG post-IDAT EXIF scan, restored parser limits, honest unknown-state treatment,
complete error denominators and local TRAIN-only admission. JPEG may internally parse its
already-present EXIF for DPI during lazy open; no metadata text is retained.


## E148 complete — processing and class imbalance exposed (2026-09-16)

Verified all12525 active TRAIN bodies,29,083,486,690 encoded bytes (27.09GiB), in443.84s.
All body SHAs matched; no available size mismatch, header error, EXIF parse error,
declared-geometry mismatch or explicit processing/geometry conflict. Full pixel decoding
was not performed, so successful headers do not certify a complete decode. No training,
feature extraction, detector scores, acquisition or serving change occurred.

| Observable property | REAL (7930) | AI (4595) |
|---|---:|---:|
| Known224 derived inputs | 2314 (29.18%) | 353 (7.68%) |
| Long side over existing2048px cap | 4413 (55.65%) | 1 (0.02%) |
| JPEG bodies | 7801 (98.37%) | 786 (17.11%) |
| PNG bodies | 129 (1.63%) | 3809 (82.89%) |

Known224 inputs comprise1700 publisher-resized SCIMD images and967 local E32 JPEG90
replays. Another128 SID images are explicitly recorded full-resolution RAW renders.
The remaining9730 are admitted source bodies whose upstream processing is unverified,
not certified untouched originals. This does not overturn their publisher class labels.
E54 native=True/False only records expansion versus base cohort:5652/5978 parents, with
895 newer parents lacking that field. Do not infer original/derived from this flag.

The clean E42 path already caps the long side at2048 before texture crops. Consequently
4413 REAL but only1 AI source body requires this downsampling. Combined with JPEG/PNG
and224-derivative imbalance, this is a plausible processing shortcut and an experimental
confound, not proof that a detector learned it or that it caused RR errors. Matched new
JPEG encoding cannot erase upstream resize/codec history. E143/E144 remain valid prior
RR-only observations; this inventory adds whole-TRAIN coverage without score reuse.

Source-pixel crop capacity:12524 parents support224,9855 support256,9513 support512 and
7685 support1024. One RR REAL body is500x183. A universal224 crop would require explicit
padding/upsampling or exclusion, so it must not silently claim native-pixel coverage.
The recorded and verified minimum short side is183; a proposed128 source-pixel probe
can preserve clean-body membership. This alone does not establish all transported-view
geometry; check those before any full four-condition extraction.

Next freeze a score-blind engineering probe across source/class/processing strata:
center128 source pixels and a downsample64/upsample128 control of exactly the same patch,
using adjacent quantized residual-pair distributions. This differs from E31 scalar68
forensic and E51 DCT/residual32 summaries, but is currently a proposal, not an implemented
or validated representation. Fix filters, quantization, selection and budgets before
pixel reads. No classifier, threshold selection or small-image exclusions at the probe.
Do not silently substitute larger ancestors for E32 replays: that would change admitted
body identities and require a separate ancestry/protected-overlap audit. Keep E92 intact.

Historical features.py comments about generated images lacking sensor/CFA behavior are
mechanistic hypotheses, not universal authenticity guarantees. Do not attribute such
guarantees to the active model or rewrite hash-bound historical feature code.

Validation:1201 Python tests passed in21.52s; compilation and pip check passed. Existing
Starlette/httpx deprecation warning only. Local frontend HTTP200; E92 API ready with
schema4/current display policy/pinned runtime manifest. Public aggregate receipts:
evidence/e148_processing_inventory_contract.json and evidence/e148_processing_inventory.json.
Private per-parent geometry records stay under e148 on LaCie; no paths, metadata text,
image bytes or parent records were added to Git. Contract SHA256
e1de07735818604f10310dae60e121d0dfd1d6534cac908843663dbec3c5873a.
