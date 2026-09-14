# AI Image Detector — Decision History

> **APPEND-ONLY ARCHIVE SINCE 2026-08-24.** The original history from day one through E21 is
> preserved verbatim as the source for the internship report. It is never silently rewritten;
> later completed development phases are appended as dated entries at the end. The living plan
> remains [`PLAN.md`](PLAN.md), while measured scientific results remain append-only in
> [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md). Paths in older entries describe the repository as it
> existed at that time; finished experiment scripts have since moved to
> `ml/experiments/archive/` and retired modules to `ml/src/pixelproof/archive/`.

Goal: build a model that decides whether a photo is **AI-generated** or a **real photograph**, starting with a small CNN and iterating toward stronger models, then serving it through the web app in this repo.

---

## 1. The Data

| Dataset | Location | Contents | Role |
|---|---|---|---|
| CIFAKE (`archive`) | `~/Desktop/archive` | 100k train + 20k test images, 32×32×3, balanced `REAL`/`FAKE` folders. Real half comes from CIFAR-10, fake half from Stable Diffusion. | Main training + held-out test set |
| External set (`archive1`) | `~/Desktop/archive1` | High-resolution images in `Ai_generated_dataset/` and `real_dataset/`, organized by category (animals, city, food, nature, people). | **Out-of-distribution (OOD) evaluation only** — never used for training. **Audited 2026-07-27, see §1b** |
| GenImage (`genimage_split`) | `~/Desktop/genimage_split` | 9,917 train / 1,742 test. REAL = ImageNet nature photos; FAKE = balanced across 7 generators. Perfectly balanced. | Training set for the ResNet **and** the feature model |
| Defactify | `~/Desktop/defactify_test` | 16,875 images: 2,851 real MS-COCO + ~2,800 each from SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney v6. Both classes JPEG. | Modern-generator test set — never trained on |

Why keep evaluation sets out of training? A model can score very well on data that looks like its training set while failing on anything else. Evaluating on a dataset from a completely different source answers the real question: *does the model generalize, or did it just memorize the training statistics?*

## 1b. Dataset auditing — a rule learned the hard way

`archive1` was this project's out-of-distribution benchmark from E1 through E6 before anyone inspected it. When we finally did (E10, 2026-07-27) the result was stark:

| | real (745) | AI (250) |
|---|---|---|
| format | 100% JPEG | 100% PNG |
| distinct sizes | 138 | 2 |
| square | 2% | 100% (512×512) |

**A logistic model on width/height/aspect alone separates the classes at AUC 1.000.** The dataset carries a perfect shortcut. (The CNNs turned out to be immune — see E10 — but that was luck, not design.)

**Rule adopted for every dataset from now on:** audit before use, and record the verdict. The checks are mechanical:
1. Do the classes differ in **file format**? (JPEG vs PNG is the classic trap)
2. Do they differ in **shape**? (all AI square, all real rectangular)
3. Do they differ in **resolution**? (median side ratio > 2)
4. Do they differ in **compression** (bytes per pixel)?
5. Is the class balance sane?

Tooling: `ml/tools/audit_datasets.py` runs all five on any folder, reading inside parquet, zip and tar without extracting. Output: `DENETIM.md`.

**The crucial nuance — a flaw is a *usage condition*, not a disqualification.** A resolution or shape shortcut only exists if the model can perceive it. Feed whole images and it can; feed fixed-size native crops (§9b) and the information never reaches the model at all. So a dataset flagged "AI is 100% square 1024px" is **unusable for whole-image training and perfectly safe for tile-based training**. Every entry in §1c is labelled accordingly.

## 1c. Dataset registry

255 GB acquired 2026-07-28/29, stored on the external SSD at
`/Volumes/LaCie/pixelproof-datasets/`.

**The full inventory, per-dataset audit verdict, and the module assignment live in
[`DATASETS.md`](DATASETS.md)** — kept there rather than duplicated here, so there is one
authoritative answer to "which data may I train on, and in which mode".

Summary of what it decides:

| | |
|---|---|
| Module 1 training | CommunityForensics-Small (228 generators) + AI-vs-Real-balanced, both clean; two larger sets usable in tile mode only |
| Module 1 testing | `defactify_test` (established benchmark) and `julienlucas` (Nano Banana Pro, cleanest modern set) |
| Module 2 | the 78 GB manipulation compilation — 13 forensic datasets with pixel-level masks |

**Label convention (important):** everywhere in the code, `1 = AI-generated`, `0 = real`. `torchvision.datasets.ImageFolder` sorts folders alphabetically (`FAKE`=0, `REAL`=1), so we invert its labels (`invert_label` in `data.py`) to keep the convention consistent.

## 2. Project Layout (`ml/`)

```
ml/
├── configs/                     # one YAML per experiment — no magic numbers in code
├── src/pixelproof/
│   ├── data.py                  # datasets, transforms, train/validation split
│   ├── models.py                # CNN definitions + registry
│   ├── train.py                 # training loop, checkpoints to artifacts/
│   ├── evaluate.py              # test-set metrics + external (OOD) evaluation
│   ├── predict.py               # classify arbitrary image files from the CLI
│   ├── features.py              # 68 resolution-independent statistics + tiling (§9)
│   ├── feature_model.py         # trains/serves the two feature models
│   ├── feature_experiment.py    # the E8 harness
│   ├── serve.py                 # FastAPI: four methods, user-selected (§11)
│   ├── prepare_genimage.py      # dataset prep — GenImage
│   ├── prepare_defactify.py     # dataset prep — Defactify
│   ├── analyze.py / classical.py / embeddings.py / learning_curve.py   # Phase 2 (E2–E4)
│   └── ela.py                   # Phase 7a baseline — written, not yet evaluated
├── experiments/                 # one script per numbered experiment (E7–E11)
├── tools/                       # dataset acquisition + auditing (§1b)
├── tests/                       # fast sanity tests
└── artifacts/
    ├── best.pt, best_genimage.pt, feature_*.joblib    # the four served detectors
    ├── experiments/             # E4/E5 checkpoints, kept as evidence
    └── features/                # cached feature matrices (14 MB)
```

Design principles worth remembering for any ML project:
- **Config-driven experiments.** Changing the learning rate or epochs means editing YAML, not code. Each checkpoint stores the config it was trained with, so results are reproducible.
- **Fixed random seed** for the train/validation split and weight init — reruns are comparable.
- **Model registry** (`MODEL_REGISTRY` dict): adding a new architecture later is one entry, and the config's `model.name` selects it.

## 2b. What was built in the 2026-07-27 → 07-29 session, and why

Everything below is new since commit `6a65a86`. Roughly **2,300 lines** of code and documentation, two new trained models, and 255 GB of audited data. This section is written as a decision chain rather than an inventory: each thing exists because a specific measurement made the previous approach untenable.

### 2b.1 — The chain of reasoning, in order · 07-27

**Starting point.** After Phase 6 the project had two CNNs behind resolution routing, and one headline OOD number: ResNet-18 at AUC 0.888 on `archive1`. Two things about that number were unexamined: the generators it was measured against were all 2021–22 vintage, and `archive1` itself had never been inspected.

**Step 1 — build a harder test set.** We downloaded Defactify (MS-COCOAI): 16,875 images, five generators (SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney v6), *every one newer than anything in GenImage*, with real MS-COCO photos alongside. Both classes JPEG, so no format shortcut. `prepare_defactify.py` was written for this — and it writes **raw JPEG bytes** rather than decoding and re-saving, because re-encoding would have overwritten the compression history that half our features read.

**Step 2 — the result reframed the project.** AUC fell 0.888 → **0.760**, which was expected. What was not expected: the per-generator scores lined up almost perfectly with *source resolution*, and in the direction opposite to a shortcut — the **smallest** images (DALL-E 3, 270px) scored best, the largest (SD 3, 1024px) worst. A model exploiting resolution would show the reverse. The only mechanism that produces this ordering is our own preprocessing: `Resize(224,224)` downscales a 1024² image 4.6×, and generation artefacts live in exactly the high frequencies a downscale removes. **We were deleting the evidence before the model saw it.** (§4b)

**Step 3 — confirm the mechanism, and fail informatively.** We fed the existing model native-resolution 224px patches instead of downscaled whole images. Discrimination improved exactly where predicted (SD 3 0.672→0.776, SDXL 0.725→0.800), *and* the false-positive rate on real photographs went from 44% to 96%. The model had never seen a sharp native pixel in training, so sharpness itself read as "AI". This was the **third** independent instance of the same law (E5, E6, here) — see §4a. Conclusion: the idea is right, but it cannot be bolted onto a model trained the old way.

**Step 4 — the reframing.** If the problem is that resizing destroys the signal, the fix is not a better network — it is a representation that never resizes. That produced the two models below.

### 2b.2 — Model 3: the feature detector (`feature_full.joblib`) · 07-27

**Where the idea came from.** Rather than feeding pixels to a network, reduce each image to a fixed-length vector of numbers and hand that to classical ML.

**Why this dissolves the problem.** A statistic is scale-free by construction. "Average high-frequency energy per pixel" means the same thing on a 300×200 image and a 4000×3000 one, and both produce *one number*. Resolution changes how many pixels we average over — never how many numbers come out. So a 300px image and a 4000px image both produce the same 68-length vector, with no resizing, no cropping, and no pixel skipped. The scale mismatch that had dominated E5, E6 and Step 3 simply has nowhere to appear.

**Why it was cheap to try.** E2 had already built the classical-ML machinery (LogReg / SVM / RF / HistGB on embeddings) and established that *the representation, not the classifier, is the bottleneck*. This experiment is E2 run again with a different representation — one built from physics instead of learned from CIFAKE.

**Which 68 numbers, and why those.** Every feature is a ratio or a per-pixel average, never a total: a total grows with pixel count and would smuggle resolution back into the vector, recreating the shortcut we are trying to escape.

| Group | Physical justification |
|---|---|
| Per-channel moments (mean/std/skew/kurtosis × RGB) | generated images occupy a measurably different colour regime |
| **Cross-channel correlation + Bayer sub-lattice variance** | the strongest idea in the set. A real sensor measures **one** colour per photosite and interpolates the other two from neighbours (CFA/demosaicing), leaving a structured, periodic inter-channel dependency across the whole frame. A latent-diffusion image never passed through a sensor and has none. This encodes **camera physics**, not a generator's quirks — so it should survive a new generator shipping, which is exactly where supervised detectors fail |
| Noise-residual statistics | real photos carry sensor shot/read noise everywhere; diffusion output carries whatever the VAE decoder invented |
| 16-band radial FFT spectrum | upsamplers and VAE decoders leave periodic spectral traces; normalised by total power so it describes a *shape*, not an amount |
| Local-variance percentiles | diffusion suppresses local high-frequency variance relative to optical imaging |
| 8×8 JPEG-grid blockiness | compression history — the basis of ELA and double-JPEG analysis |
| HSV statistics | saturation distributions differ systematically |

**The controlled-comparison decision.** We trained it on the *identical* GenImage split the ResNet uses. Same images, same split, same test sets — the only variable is the method. Without that, "features vs CNN" would be an anecdote.

**Result: a specialist, not a replacement.** Overall it loses (0.717 vs 0.760 on Defactify) but wins by +0.09 to +0.15 AUC on precisely the three high-resolution generators the CNN handles worst, and collapses on small heavily-compressed ones (DALL-E 3: 0.377, *below chance* — the model systematically calls those "real"). The two methods fail in disjoint places, which is the complementarity the literature's "RGB branch + low-level branch" architectures are built to exploit.

### 2b.3 — Model 4: the tile detector (`feature_crop128.joblib` + 6×6 grid) · 07-29

**The failure that produced it.** Testing a ChatGPT-generated image (1122×1402) in the demo: the ResNet said 48% ("uncertain" — wrong), the whole-image feature model said 94% (correct), and the 128px-crop model said 47% (wrong). Digging into the third: a 128×128 centre crop of that image is **1.04% of its pixels**, and in this photograph the centre is the subject's plain navy t-shirt — grey-level std 0.027 against 0.283 for the full image, i.e. **10.6× flatter**. We had handed the model a featureless patch of fabric and it correctly answered "no idea".

So the fixed-crop idea was sound and its implementation was blind sampling: a single centre crop of a large image is an arbitrary 1% window that may land on sky, a wall, or clothing.

**The fix.** Cut the image into a **grid** of native-resolution tiles, score every one, aggregate. Coverage goes from ~1% to ~100%, and the fixed-size property that kills the shortcut is preserved.

**Why no retraining was needed** — this is the non-obvious part. Having just learned the preprocessing law three times, the instinct was to retrain. But `feature_crop128` was fitted on 128×128 native crops, and **every tile is a 128×128 native crop**. Same input distribution, evaluated several times per image instead of once. No mismatch exists. (The mild caveat: training used *centre* crops while tiles include edges, so content statistics differ slightly — far milder than the resize-vs-native gap that caused the earlier failures.)

**Why the grid is 6×6 and the rule is top-3, both measured rather than chosen:**

| Grid | Best AUC (high-res generators) |
|---|---|
| 2×2 | 0.760 |
| 3×3 | 0.799 |
| 4×4 | 0.801 |
| 5×5 | 0.807 |
| **6×6** | **0.821** |

And the aggregation rule matters as much as the grid. A plain mean scores 0.781; the mean of the **top 3** tiles scores 0.821. The reason is the t-shirt problem again at scale: flat tiles score around 0.5, and averaging them in drags a confident image toward "no idea", drowning the tiles that carry evidence. Measured: 21% of all tiles fall below the texture floor. (Dropping flat tiles explicitly by texture threshold was also tested and gave no advantage over top-3.)

**Result — the project's best numbers:**

| Generator | Source | Tiled | CNN | Δ |
|---|---|---|---|---|
| SDXL | 1024px | **0.948** | 0.717 | +0.231 |
| SD 3 | 1024px | **0.894** | 0.670 | +0.224 |
| SD 2.1 | 768px | **0.863** | 0.696 | +0.167 |
| Midjourney | 436px | 0.580 | 0.821 | −0.241 |

0.948 exceeds E6's 0.888 headline. The crossover — **~700px** — is now measured, which is what lets `serve.py` replace its invented `128px` routing threshold with an evidence-based one.

**The bonus nobody planned.** The per-tile scores *are* a localisation map: "which tiles look synthetic" is the same question as "where was this image tampered". Module 2's core machinery was therefore built as a side effect of solving a Module 1 problem (§9c).

### 2b.4 — Ideas that were tested and lost · 07-27 → 07-29

These are as valuable as the ones that worked, and they were all resolved by measurement rather than argument.

**"Train only on AI, call everything else real."** Proposed on the grounds that this dataset holds 5× more AI than real, so learn the majority class. Tested as a third setup on identical features (one-class SVM / Isolation Forest). On `archive1` — the most out-of-distribution set — the result was decisive:

| Setup | AUC |
|---|---|
| one-class on **real** | **0.688** |
| supervised | 0.505 |
| one-class on **AI** | **0.358** |

Learning "what AI looks like" scored *below chance*, i.e. systematically inverted. The asymmetry has a physical cause: "AI" is an expanding set that changes every few months, so a boundary drawn around today's generators expires; "real photograph" is fixed by sensor physics and does not change when a new generator ships. And the failure mode runs the wrong way — a model that knows only today's AI meets a new generator, finds no match, and stamps it **authentic**, which is the worst possible error for a misinformation detector. Learning "real" fails safe instead. This is Phase 5's thesis, now with a number attached.

**Blending the CNN and the feature model.** Since they fail in disjoint places, an ensemble should win. Eight rules tested (mean, weighted, max, min, and rank-normalised variants, the ranks specifically to remove the probability-scale mismatch between a neural net and gradient boosting). The best beat the ResNet alone by **+0.002** — noise. It *relocates* accuracy rather than adding it: Defactify +0.036, `archive1` −0.036, because the feature model is near-random on `archive1` (0.505) and averaging a random signal into a good one costs what the gains are worth. A fixed-weight blend cannot exploit a specialist; a conditional one needs a reliable "is this model trustworthy here?" signal we do not have. **Hence the demo shows both scores separately and flags disagreement rather than hiding it inside a mean.**

**Suspicion about `archive1`.** The feature model scoring exactly 0.505 there, and logistic regression scoring 0.217 (systematically inverted), prompted an audit of a benchmark that had gone uninspected since E1. It turned out to be maximally confounded — width/height/aspect alone separate the classes at **AUC 1.000**. Two controls followed, changing one variable at a time: re-encoding the PNGs to JPEG, then also centre-cropping both classes square. CNN performance moved by **+0.008** — *upward*. Both networks were immune, for a mechanical reason: `PIL` decoding discards the container format and `Resize((N,N))` discards dimensions and aspect ratio, so neither could perceive the leak. **E1's 77.1% and E6's 0.888 stand.** The irony is worth recording: the aggressive downscaling criticised throughout §4 for destroying signal also, accidentally, destroyed the shortcut. And the immunity does **not** transfer — the feature model reads native pixels and its shortcut probe predicts image width at 92.6% accuracy, so any native-resolution method must control for this explicitly.

### 2b.5 — Questions answered along the way · 07-27 → 07-28

These came up as design questions and the answers shaped the code; full treatment in `IMAGE_STRUCTURE_NOTES.md`.

- **Why were we downscaling at all?** Networks need fixed-size batches; ResNet-18 was pretrained at 224; and 1024² is 21× the pixels. But the deciding point is that the standard recipe was written for *semantic* classification — a cat survives downscaling, a generation artefact does not. The error was adopting a recipe built for a different problem.
- **Is image structure constant across resolutions?** Channel count is: RGB is 3 channels at 32×32 and at 4000×3000; only height and width change. That invariance is precisely why the feature approach yields a fixed-length vector.
- **Does it change by format?** By colour mode, yes: grey=1, RGB=3, RGBA=4. And **JPEG does not store RGB** — it stores YCbCr and usually subsamples the two colour channels to half resolution, so inside a single JPEG the channels are not even the same size.
- **Are the channels independent?** No — and the dependency is one of the most valuable signals available (the CFA trace, §2b.2). Analysing channels in isolation throws it away.
- **Can one model handle all of them at once?** Yes, by concatenation: per-channel features + cross-channel features + alternative colour spaces in a single vector. No fusion logic required.
- **Does normalisation lose information?** Some kinds destroy exactly what we measure — resizing, grayscale conversion, per-image contrast standardisation, re-encoding. The rule adopted: **normalise after extraction, in feature space; never before extraction, in pixel space.**
- **Would training at full resolution fix it?** Measured on the M3 Pro: feasible (80 min vs 4 min) but wrong, for three reasons unrelated to speed — see §4c.

### 2b.6 — The models

| Artifact | Size | What it is | Trained on |
|---|---|---|---|
| `artifacts/feature_full.joblib` | 0.5 MB | StandardScaler + HistGradientBoosting over 68 statistics of the **whole image** at native resolution | GenImage train (9,917) |
| `artifacts/feature_crop128.joblib` | 0.5 MB | Same pipeline on a **128×128 native crop**; also the model the 6×6 tile scorer calls once per tile | GenImage train (9,917) |

Gradient boosting was chosen deliberately: with ~68 columns and ~10k rows this is tabular data, where boosted trees are the standard strong baseline and a neural network would not be expected to win. Both models are ~85× smaller than the ResNet checkpoint and refit in seconds once features are cached, which is the real benefit — it turns an experiment cycle from minutes into seconds and makes ablations cheap.

The project now holds **four detectors**: SmallCNN (32px), ResNet-18 (GenImage), feature-full, feature-tiles.

### 2b.6b — How to run any of it

All paths relative to `ml/`. Feature extraction caches to `artifacts/features/` (14 MB), so everything after the first run is seconds rather than minutes.

```bash
# --- reproduce the feature experiment (E8): extract, train 3 setups, per-generator table
PYTHONPATH=src .venv/bin/python -m pixelproof.feature_experiment
#   both modes (whole-image + 128px crop), supervised / one-class-real / one-class-AI,
#   the shortcut probe, and the table that decides whether the resolution ordering survived

# --- refit and save the two feature models (reads the cache, writes artifacts/*.joblib)
PYTHONPATH=src .venv/bin/python -m pixelproof.feature_model

# --- rebuild the Defactify test set from downloaded parquet shards
PYTHONPATH=src .venv/bin/python -m pixelproof.prepare_defactify \
  --source ~/Desktop/defactify/data --output ~/Desktop/defactify_test
#   optional: --per-generator N   to build a class-balanced subset

# --- serve all four detectors (returns CNN + feature-full + 6x6 tile map)
PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799
npm run dev          # from the repo root, UI on :3000

# --- audit any dataset folder before using it (§1b)
.venv/bin/python /Volumes/LaCie/pixelproof-datasets/audit.py [folder]
#   no argument = audit everything on the SSD; writes DENETIM.md

# --- the training pool: index every source, then resolution-balance the index
PYTHONPATH=src .venv/bin/python -m pixelproof.build_pool          # -> pool_index.csv
PYTHONPATH=src .venv/bin/python -m pixelproof.make_balanced_pool  # -> pool_balanced.csv
#   --bands 128,256,384,512,768,1024,1536   closes the class gap to 1.00x (use this for new pools)
#   --min-side 128    drop images smaller than one tile — below this features.py
#                     reflection-pads, i.e. the model is shown a synthetic pattern
#   --verify-against artifacts/pool_balanced.csv    check a reconstruction

# --- extract the 68 statistics for a pool index (caches to pool_features.npz)
PYTHONPATH=src .venv/bin/python -m pixelproof.pool_features --index artifacts/pool_balanced.csv

# --- frozen-backbone embeddings for the same pool (E16)
PYTHONPATH=src .venv/bin/python -m pixelproof.backbone_features --backbone dinov2

# --- Module 2 data: unpack the manipulation compilation (§12)
PYTHONPATH=src .venv/bin/python -m pixelproof.prepare_manipulation --list
PYTHONPATH=src .venv/bin/python -m pixelproof.prepare_manipulation
#   default: every sub-dataset except OpenForensics, 1 tar per split -> ~/Desktop/manipulation_test
#   idempotent; --force re-extracts, --tars-per-split N pulls more
PYTHONPATH=src .venv/bin/python experiments/e17_module2_first_measurement.py
PYTHONPATH=src .venv/bin/python experiments/e18_ela_vs_tiles.py
```

⚠️ Everything in this block from `build_pool` down needs the SSD mounted at
`/Volumes/LaCie`; the scripts hard-code that path.

Existing artifacts: `best.pt` (SmallCNN/CIFAKE), `best_genimage.pt` (ResNet-18/GenImage),
`feature_full.joblib`, `feature_crop128.joblib`, plus `best_10k/20k/50k.pt` from the E4 learning curve.

### New ML modules (`ml/src/pixelproof/`)

| File | Lines | Purpose |
|---|---|---|
| `features.py` | 316 | The feature extractor. 68 resolution-independent statistics per image — channel moments, cross-channel/CFA traces, Bayer sub-lattice variance, noise residuals, 16-band radial FFT, local-variance percentiles, JPEG-grid blockiness, HSV. Also `extract_tiles()` + `tile_positions()`, which cut an image into a grid of native-resolution tiles and report each tile's texture so flat ones can be discounted. |
| `feature_experiment.py` | 252 | The E8 harness: parallel feature extraction with on-disk caching, three learning setups (supervised, one-class on real, one-class on AI), the shortcut probe that predicts image width from features alone, and the per-generator table. |
| `feature_model.py` | 137 | Trains and persists the two feature models; `score_image()` and `score_tiles()` are what `serve.py` calls. Holds the measured constants: `TILE_PX = 128`, `TILE_GRID = 36`. |
| `prepare_defactify.py` | 104 | Unpacks the Defactify parquet shards into image folders. Writes **raw JPEG bytes** rather than decoding and re-saving — re-encoding would rewrite the compression history, which is part of what we measure. |

### Rewritten

| File | Change |
|---|---|
| `serve.py` | +122 lines. Three named signals instead of one verdict, input-dependent `primary`, score-based agreement, `enough_evidence` floor, per-tile map in the response. |
| `app/page.tsx` + `globals.css` | +79 lines. "Which model said what" panel: every model named, its own bar and verdict, agreement badge. |

### Dataset tooling (`/Volumes/LaCie/pixelproof-datasets/`)

| File | Lines | Purpose |
|---|---|---|
| `fetch.py` | 333 | Unattended downloader. One subprocess per dataset (a poisoned HTTP client cannot cascade), exponential backoff with longer waits on 429, multiple passes over the queue, disk-cap guards. |
| `audit.py` | 259 | Standalone auditor. Reads inside parquet, **zip and tar** without extracting; samples across the whole shard range rather than the first few files; applies the five §1b checks. Writes `DENETIM.md`. |
| `watchdog.sh` | 21 | Outermost net: restarts `fetch.py` if it dies for any reason. |

### New documentation

| File | Lines | Purpose |
|---|---|---|
| `IMAGE_STRUCTURE_NOTES.md` | 192 | How an image is physically structured and what that means for feature design: channel counts by format, JPEG's YCbCr with chroma subsampling, why channels are **not** independent (the CFA trace), and which normalisations destroy the evidence. |
| `STATUS.md` → `DATASETS.md` | 131 | Started as a one-page status summary; on 2026-07-30 its findings were folded into §2b and the file was re-purposed as the dataset inventory and module assignment. |
| `EXPERIMENTS.md` | +90 | E7–E10 in the project's pre-registered-hypothesis format, including the two negative results. |

### 2b.7 — The dataset acquisition, and how the auditing rule was born · 07-28 → 07-29

**Why more data was needed.** Two things forced it. `archive1` turned out to be confounded (§2b.4), so the project's OOD benchmark needed replacing. And Defactify's newest generator is Midjourney v6 (~2024) — nothing in the project had seen a 2025–26 model.

**What was collected.** 255 GB to an external SSD across two overnight runs, chosen by value-per-GB rather than raw size: modern generators (Nano Banana, Nano Banana Pro, FLUX.1-dev, GPT Image 4K), balanced real/AI sets, an 18-generator benchmark, CommunityForensics (228 distinct models with per-image metadata), and — the most consequential item — a compilation of 13 forensic datasets **with pixel-level manipulation masks**. Full registry with per-dataset verdicts in §1c.

**The first run failed, instructively.** It died after 31 minutes: an anonymous HTTP 429 from the Hub put the download client into a bad state, and because there was no retry logic anywhere, every remaining dataset then failed instantly with "Previous task error" — ten of them in two seconds. The rebuild addressed each cause separately: **one subprocess per dataset** (a poisoned client dies with it and the next starts clean), **exponential backoff** with longer waits on 429, **multiple passes** over the queue so early failures get retried later, **lower concurrency** (8 workers was what triggered the throttling), and a **shell watchdog** outside the process. The queue was also reordered by *file count*, not size — the first run began with an 8,002-file dataset, i.e. 8,002 API calls, which is what provoked the rate limit in the first place.

**Where the auditing rule came from.** `archive1` had been the OOD benchmark through six experiments before anyone looked inside it. The lesson generalises: **audit before use, and write the verdict down.** `audit.py` (259 lines) runs five mechanical checks — format split, shape split, resolution split, compression split, class balance — reading inside parquet, zip and tar without extracting anything. Every downloaded set carries a verdict in `DENETIM.md`.

The most important refinement is in §1b and bears repeating: **a flaw is a usage condition, not a disqualification.** A shortcut only exists if the model can perceive it. "AI images are 100% square 1024px" makes a dataset unusable for whole-image training and perfectly safe for tile-based training, because a 128×128 tile carries no information about the size of the image it came from.

### 2b.8 — Bugs found in this session's own code · 07-28 → 07-30

All three would have quietly corrupted conclusions rather than raising errors — which is the kind worth cataloguing.

1. **AppleDouble stubs.** ExFAT makes macOS write a 4 KB `._name` file beside every real file. Sorted alphabetically these come *first*, so the auditor opened them instead of the parquet and reported "corrupt file" for perfectly good data. Fix: filter `._*` everywhere. Filesystem-specific, invisible on APFS.
2. **Shard-sorted labels.** `theminji/ai-vs-real-200k` stores class 0 in shards 0–133 and class 1 in 134–267. Sampling the first 8 shards reported **"single class" for a perfectly balanced dataset** (4,930 vs 4,926 once sampled properly). Fix: spread the sample across the whole shard range. This is the dangerous kind: it produced a confident, wrong verdict about data quality instead of an obvious failure.
3. **Agreement computed from labels instead of scores.** The demo called 0.48 and 0.94 "in agreement" because one of them fell inside the uncertainty band, so only one *non-uncertain* verdict existed. Technically consistent with the rule as written, visibly absurd to anyone looking at the screen. Fix: compare the raw score spread (<0.20 agree, <0.40 partial, else conflict) — what the eye actually reads.

A fourth, in method rather than code: **the datasets were audited after downloading rather than before.** The correct procedure is to pull a single shard, audit it, and only then commit to the full download. Recorded here so the next acquisition does it in the right order.

## 3. Phase 1 — Baseline CNN ✅ (2026-07-20 → 07-21, commit `4c149ec`)

### Architecture (`SmallCNN`)
Three convolutional blocks (Conv → BatchNorm → ReLU, ×2 per block) with max-pooling in between, then **global average pooling** and a single-logit linear head. ~300k parameters — small, fast, and hard to overfit on 100k images.

- **Why one output logit instead of two classes?** Binary classification only needs one number; `sigmoid(logit)` is the probability of "AI". Trained with `BCEWithLogitsLoss` (numerically stabler than sigmoid + BCE separately).
- **Why BatchNorm?** Stabilizes training and lets us use a higher learning rate.
- **Why global average pooling?** No giant fully-connected layer → far fewer parameters, and the network technically accepts any input size.

### Training setup
- Split: 90k train / 10k validation (10%), stratified by the seed-fixed shuffle.
- Augmentation: **horizontal flip only.** We deliberately avoid color jitter / blur / JPEG-style augmentations, because subtle color statistics and generation artifacts are exactly the signal that separates AI images from real ones — destroying them would hurt the model.
  > ⚠️ **Status 2026-07-29: this is an assumption, not a result.** The literature says the opposite — CNNSpot's central finding is that JPEG/blur augmentation is *the* lever for cross-generator generalisation (`IMAGE_FORENSICS_REFERENCE.md` §4.4), and §4.5 there states the trade-off must be measured rather than assumed. It has never been tested here. Listed as open in §13.
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

## 4. The central finding: preprocessing, not architecture

What began as a footnote about the 32×32 baseline turned out to be the dominant factor in this project's performance. Three separate experiments converged on the same law, and a fourth quantified its cost.

### 4a. The law

> **Whatever the model will be shown at test time is what it must be shown during training.**

| Experiment | Trained on | Given at test | Result |
|---|---|---|---|
| E5 | blurry 32→224 upscales | sharp native photos | called 984/995 images "AI" |
| E6 | native high resolution | 32×32 CIFAKE | 50% — total collapse |
| E7 patch trial | downscaled crops | native-resolution patches | 96% of real photos called "AI" |

Every one of these looked like a model failure and was a preprocessing mismatch. Model capacity never entered into it.

### 4b. The cost of downscaling, measured

`eval_transform` resizes every input to 224×224. Generation artefacts live in fine texture; downscaling is a low-pass filter. On Defactify — five generators the model had never seen — the results ordered themselves almost perfectly by *source resolution*:

```
DALL-E 3     270px  → AUC 0.896   (barely downscaled)
Midjourney   436px  → AUC 0.821
SD 2.1       768px  → AUC 0.696
SDXL        1024px  → AUC 0.717
SD 3        1024px  → AUC 0.670   (downscaled 4.6x)
```

Note the direction: the *smallest* images are detected best. A resolution shortcut would produce the opposite ordering, so this is not the model cheating — it is the model being starved of evidence before it ever sees the image.

**Root cause in our own config:** `configs/genimage.yaml` sets `crop_augmentation: true` → `RandomResizedCrop(224, scale=(0.7,1.0))`. For a 1024² training image that is a 3.8–4.6× downscale, every epoch. **The ResNet has never seen a native-resolution pixel.**

### 4c. Compute was never the constraint

Measured on the M3 Pro (ResNet-18, forward+backward):

| Input | Throughput | 10k images × 5 epochs |
|---|---|---|
| 224×224 | 193 img/s | 4 min |
| 512×512 | 41 img/s | 20 min |
| 1024×1024 | 10 img/s | 80 min |

Training at full resolution is 20× slower but entirely feasible. It is still the wrong answer, for three reasons that have nothing to do with speed:
1. **You still resize.** A 4000×3000 photo fed to a 1024² model is still downscaled 3.9×; the problem shrinks, it does not vanish.
2. **Global average pooling dilutes.** At 1024 input the final feature map is 32×32 = 1024 positions, all averaged into one vector. A small local trace is divided by 1024 instead of 49. Bigger input makes local evidence *weaker*.
3. **Pretraining scale breaks.** ImageNet weights were learned at 224; at 1024 the transfer benefit degrades.

The right answer is to stop resizing altogether — §9b.

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

## 6. Phase 2 — Hybrid ML Experiments ✅ (2026-07-21, commits `1b20eeb` · `ee21fa3` — E2, E3, E4)

A core learning goal of this internship project: combine deep learning with classical ML instead of treating them as rivals. All three experiments reuse the trained CNN, so none of them require expensive retraining.

**A note on terminology first.** This project is **supervised learning** (we have REAL/FAKE labels and train a classifier on them), *not* unsupervised learning. Clustering algorithms like k-means only enter the picture below as *analysis tools* on top of the supervised model — except for Phase 5, which is a genuinely unsupervised formulation of the problem.

### 6a. CNN as a feature extractor + classical classifiers
- Take the 128-dim embedding from the CNN's penultimate layer for every image.
- Train classical models on those embeddings: Logistic Regression, SVM, Random Forest, Gradient Boosting.
- Compare all of them against the CNN's own classification head on the same test sets.
- What this teaches: deep nets as representation learners; strengths/weaknesses of each classical algorithm; a clean comparison table for the report.

### 6b. Embedding analysis with clustering & projection
- Run k-means (k-means++ init) on the embeddings; project to 2D with t-SNE/UMAP, color by true label and by cluster.
- Questions to answer: do real/AI images separate cleanly? Where do the misclassified images live? Do clusters align with semantic categories (animals, city, food…)?
- What this teaches: what the network actually learned, communicated visually — and the correct role of clustering: *exploration and error analysis*, not classification.

### 6c. Learning-curve experiment (data-size ablation)
- Retrain the same CNN on subsets (e.g. 10k / 20k / 50k / 90k) and plot accuracy vs. training-set size.
- Answers empirically: "how much does more data matter?" Expected: logarithmic gains and a widening train/val gap at small sizes.

## 7. Phase 3 — Transfer Learning + Ensemble ✅ (2026-07-21, commit `08e9c76` — E5)

- Fine-tune a pretrained backbone (ResNet-18 → EfficientNet-B0) at 224×224 input.
- Compare against the baseline on the *same* test sets — this is why the metrics pipeline came first.
- Ensemble idea: average/vote the CNN (pixel domain) with a gradient-boosting model trained on frequency-domain features (FFT/DCT) — diffusion models leave periodic fingerprints in the frequency spectrum that pixel-space models can miss.
  > ⚠️ **Status 2026-07-29: done, and it did not work.** This is exactly what E8 + E9 built and tested. The frequency/gradient-boosting model exists and is genuinely complementary (§9a), but eight blending rules all failed to beat the CNN by more than noise (§9d). Do not re-propose a fixed-weight blend without reading E9 first.
- Concepts to learn here: freezing vs. full fine-tuning, discriminative learning rates, pretrained normalization statistics, why diverse ensembles beat their members.

## 8. Phase 4 — High-Resolution / Patch-Based Inference ✅ (2026-07-21, commit `1fd4ea5` — E6)

- Instead of downscaling a large image, crop several patches at native resolution, classify each, and aggregate (mean or max probability).
- Needs a higher-resolution training dataset (e.g. GenImage or a scraped SD/Midjourney set) — CIFAKE alone can't teach high-res artifacts.

> **Status 2026-07-29: this phase is done, and it turned out to be the most important idea in the project — but not in the form written above.**
> - The GenImage retraining happened (E6), and it fixed E5's collapse.
> - Patch inference on the *CNN* was tried and failed for a specific reason: the ResNet had only ever seen downscaled crops, so native patches were a fresh preprocessing mismatch (§4a). It is not enough to patch at inference time.
> - Patch inference on the **feature model** worked, because that model was already fitted on native 128px crops — so tiles are the same input distribution, not a new one. This is §9b, and it produced the project's best scores (SDXL 0.948).
> - "Mean or max probability" was measured rather than assumed: **top-3 mean** beats both, because flat tiles score ~0.5 and dilute an ordinary average.

## 9. Phase 4b — Resolution-independent detection (2026-07-27/29)

The answer to §4 came from a question raised during the session: *instead of feeding pixels to a network, reduce the image to a fixed-length vector of numbers and classify that.* Two things follow from it, and both were measured.

### 9a. Hand-crafted statistics (`features.py`, E8)

68 numbers per image, computed over **every pixel at native resolution** — nothing resized, nothing cropped, nothing skipped. All of them are ratios or per-pixel averages, never totals, which is what makes the vector the same length and the same meaning at any image size.

| Feature group | What it reads |
|---|---|
| Per-channel moments | global colour/tone behaviour |
| **Cross-channel correlation + Bayer sub-lattice variance** | the **CFA/demosaicing trace** — a real sensor measures one colour per photosite and interpolates the other two, leaving a structured inter-channel dependency. Latent-diffusion output never passed through a sensor and has none. |
| Noise-residual statistics | sensor shot/read noise |
| 16-band radial FFT spectrum | upsampler / VAE decoder periodic traces |
| Local-variance percentiles | texture consistency |
| 8×8 JPEG-grid blockiness | compression history |
| HSV statistics | generated images occupy a different saturation regime |

Trained on the *identical* GenImage split as the ResNet, so the comparison is controlled. Result: a **specialist, not a replacement** — worse overall (0.717 vs 0.760 on Defactify) but better by +0.09 to +0.15 AUC on exactly the three high-resolution generators the CNN handles worst, and far worse on small heavily-compressed ones.

Concepts covered here: why tabular data wants gradient boosting rather than a neural net; why normalisation must happen **after** extraction in feature space and never **before** in pixel space; why physics-based features (sensor noise, CFA, compression) should outlive generator-specific ones.

### 9b. Tiling — the general answer to the scale problem

The second idea: **cut the image into a grid of fixed-size native tiles, score every tile, aggregate.**

This dissolves the whole §4 problem. The model always sees 128×128 native pixels; resolution changes only *how many tiles come out*, never what a tile looks like. No resizing anywhere in the pipeline, and — as a side effect — image dimensions can no longer act as a shortcut (§1b).

No retraining was needed: the crop128 model was fitted on 128×128 native crops, and every tile is one.

**Measured optimum: 6×6 grid, aggregating the mean of the top 3 tiles.**

| Grid | Best AUC |
|---|---|
| 2×2 | 0.760 |
| 3×3 | 0.799 |
| 4×4 | 0.801 |
| 5×5 | 0.807 |
| **6×6** | **0.821** |

Top-3 beats a plain mean (0.821 vs 0.781) because flat tiles — sky, a wall, plain clothing — score around 0.5 and drag an ordinary average toward "no idea", drowning the tiles that carry evidence.

Against the CNN on Defactify's high-resolution generators:

| Generator | Source | Tiled | CNN | Δ |
|---|---|---|---|---|
| **SDXL** | 1024px | **0.948** | 0.717 | **+0.231** |
| **SD 3** | 1024px | **0.894** | 0.670 | **+0.224** |
| **SD 2.1** | 768px | **0.863** | 0.696 | **+0.167** |
| Midjourney | 436px | 0.580 | 0.821 | −0.241 |

**0.948 is the highest score this project has produced**, above E6's 0.888 headline.

The crossover is measured, not guessed: **above ~700px the tile model wins decisively, below it the CNN does.** That replaces the invented `128px` routing threshold in `serve.py` with an evidence-based one.

### 9c. Why this also builds Module 2

The per-tile scores *are* a localisation map. "Which tiles look synthetic" is the same question as "where was this image tampered". One implementation serves both modules — Module 2 no longer starts from zero, and with the mask-annotated compilation in §1c it can now be **scored at pixel level** rather than argued about.

Open caveat: the tile model was trained on image-level labels ("is this whole image AI"), never on region labels. For a fully-AI image every tile lights up, which is correct but carries no localisation information. The interesting case — a real photo with a pasted AI region — is a well-founded hypothesis that is **still unvalidated**. The data to validate it is now on disk.

### 9d. Ensembling: a negative result (E9)

Since the CNN and the feature model fail in disjoint places, blending them should win. Eight rules were tested (mean, weighted, max, min, and rank-normalised variants). The best beat the ResNet by **+0.002** — noise. It *relocates* accuracy rather than adding it: Defactify +0.036, archive1 −0.036, because the feature model is near-random on archive1 (0.505) and averaging a random signal into a good one costs what the gains are worth.

Conclusion: a fixed-weight blend cannot exploit a specialist. A conditional combination needs a reliable "when is this model trustworthy?" signal, which we do not have. **Decision: the demo reports both scores side by side and flags disagreement instead of averaging.**

## 10. Phase 5 — Unsupervised Track: Anomaly Detection (parked; partially probed 2026-07-27 in E8)

Train a model **only on real photographs** and flag anything that deviates as suspicious (one-class SVM on embeddings, or an autoencoder with reconstruction error).

**Why this matters.** Every supervised detector has a built-in blind spot: it learns the artifacts of the generators it was trained against. When a new generator ships (Midjourney v7, Flux, whatever comes next), those artifacts change and supervised accuracy silently collapses — our own CIFAKE→`archive1` drop (96.75% → 77%) is a small-scale preview of exactly this failure mode. An anomaly detector inverts the question: instead of "what does AI look like?" it learns "what do real photos look like?" — and real photos don't change when a new generator is released. This is the closest thing the field has to future-proofing, which is why it deserves a dedicated phase even though it comes last.

## 11. Phase 6 — Serving in the Web App ✅ (2026-07-21 commit `3b02c53`; rebuilt 2026-07-28 → 07-30)

`serve.py` + the Next.js UI, rebuilt 2026-07-28/29 around the findings above.

**Three named models, reported side by side, deliberately unblended** (§9d showed a fixed blend adds nothing):

| Signal | Which model |
|---|---|
| CNN | SmallCNN (<128px) or ResNet-18 (≥128px) |
| Statistics — whole image | feature model, `full` variant |
| Tiles — 6×6 grid, top-3 mean | feature model on native 128px crops |

Design decisions that came out of measurement rather than taste:

- **`primary` follows the input, not a fixed favourite.** Above `TILE_RELIABLE_PX = 700` the tile model leads; below it the CNN does. Both numbers are measured (§9b).
- **Agreement is computed from the scores, not the verdict labels.** An earlier version called 0.48 and 0.94 "in agreement" because one of them fell in the uncertainty band — technically true, visibly absurd. Now: spread <0.20 agree, <0.40 partial, else conflict.
- **`enough_evidence` floor at 48px.** Below that no method has the pixels to measure texture; the honest output is "insufficient evidence", not a confident guess.
- Disagreement between the two families is surfaced, not hidden — it is a real uncertainty signal.

Still open: calibration. 44% of real photographs are still called "AI" at threshold 0.5.

## 12. Phase 7 — Manipulation Detection (Module 2) — approved 2026-07-23, unblocked 2026-07-29

On 2026-07-23 the mentor approved the two-module architecture: **Module 1** (done — the existing real-vs-AI classifiers behind resolution routing) and **Module 2** — a separate detector answering "does this photo contain a locally tampered region?", with optional localization. Rationale: manipulation leaves local traces, not global ones (the Sunak-photo case: a mostly-real photo fools any whole-image classifier). Transfer learning was also explicitly approved.

Planned steps, mirroring the Phase 1→3 methodology (cheap baseline first, learned model second):

- **7a — ELA baseline (hand-written, no training):** Error Level Analysis re-saves the image as JPEG and maps how much each region changes; regions with a different compression history (pasted/inpainted) light up. Known limits (fails on PNG/screenshot pipelines, false-positives on sharp edges) are accepted — it is the SmallCNN of Module 2: quick to build, measurable, and the yardstick the learned model must beat.
- **7b — Learned detector:** fine-tune a pretrained backbone on a manipulation dataset with ground-truth labels (e.g. CASIA v2). Input-representation ablation planned: raw image vs. ELA map vs. both.
- **7c — Localization:** patch-based inference producing a "where was it tampered" heatmap.
- **Integration:** third model in `serve.py` + combined verdict logic (single user-facing verdict: real / fully AI / real-but-tampered / uncertain).

### Status update 2026-07-29 — 7c is half-built and now measurable

Two things changed:

1. **The machinery exists.** The tile scorer from §9b already produces a per-tile probability map at native resolution. "Which tiles look synthetic" is the localisation question. No new architecture is required for a first heat-map.
2. **The ground truth exists.** `ductai199x/image-manipulation-dataset-compilation` (§1c) ships 13 forensic datasets with **pixel-level masks**, including CASIA 2.0 (the set 7b planned to use) and CocoGlide (diffusion inpainting). Until this landed, "the tiles will show where the manipulation is" was an untestable hypothesis. It can now be scored with pixel F1/IoU against real masks.

### First measurement, 2026-08-04 (E17/E18)

Run, and the answer splits cleanly by manipulation type:

| sub-dataset | type | tile model | ELA |
|---|---|---|---|
| **CocoGlide** | diffusion inpainting | **0.648** tile / 0.721 image | 0.339 |
| CASIA 2.0 | classic splice | 0.606 tile / **0.481 image** | 0.468 |
| hand-made JPEG splice (control) | classic splice | — | **0.719** |

**Module 2 needs two detectors, not one.** The tile model asks an *absolute* question — "does
this region look generated" — which is right for an AI-inpainted region and wrong for a splice,
where the pasted pixels are camera output from a different camera. On CASIA it scores 0.481 at
image level: manipulated 0.760 against authentic 0.755, i.e. it cannot see the edit at all, and
is answering its own question correctly.

The splice case needs a *relative* question — "is this region inconsistent with the rest of this
image" — because the donor differs in sensor noise, demosaicing signature and JPEG history. ELA
reads the last of those, and the controlled test confirms it works (0.719) even though it fails
on the compilation (0.468). The cause is in the data: every image was converted to PNG, and that
uniform re-encode flattens exactly the differential compression history ELA depends on
(`IMAGE_FORENSICS_REFERENCE.md` §4.3). Splitting CASIA by original extension shows it directly —
`.tif` originals 0.578, `.jpg` originals 0.338.

So the two-detector design is supported, and **cannot be validated on this dataset**. That needs
manipulation data preserving JPEG history, or splices constructed here.

**Widened to nine sub-datasets, 2026-08-04.** Scripting the extraction
(`prepare_manipulation.py`) took the measurement from 2 usable sub-datasets to 9. The CocoGlide and
CASIA numbers reproduced exactly, and the wider run added two things:

- **CocoGlide is still the only one that works, and now against eight controls rather than one.**
  Every classic-manipulation set sits between 0.326 and 0.548 at image level — chance. The
  absolute-vs-relative split is no longer an argument from two data points.
- **The IoU column was measuring mask size, not skill.** The experiment flags exactly `mask_frac`
  of the tiles, so random flagging already scores `f/(2-f)`. NIST2016's 0.864 — the best-looking
  localisation number this project has produced — is **+0.041 over chance**, while CocoGlide's
  0.419 is +0.155. Ranking by raw IoU nearly inverts the true ranking. The baseline is now printed
  next to every IoU.

A third observation belongs to §12b rather than here: on the classic photographic sets the model
scores manipulated and authentic images alike at 0.97–0.99. Nine fresh forensics datasets, and it
calls all of them AI. That is E13's false-positive rate again, and it is *why* those image-level
AUCs are at chance — both classes are pinned to the ceiling.

**Honest limit before anyone over-claims:** the tile model was trained on *image-level* labels only. It answers "does this tile look like AI-generated texture", not "was this tile edited". Those coincide for a pasted synthetic region and diverge for everything else. The first experiment must therefore be a measurement, not a demo.

**Taxonomy correction carried over from `IMAGE_FORENSICS_REFERENCE.md` §4.2:** ChatGPT-family edits re-render every pixel, so at the pixel level they are *generated*, not *locally tampered*. `real-but-tampered` is recoverable only for classic edits and AI-spliced images. For fully-regenerated edits the honest verdict is "AI-regenerated"; promising localisation there would be a claim the field cannot currently support.

## 12b. The narrow-real-class finding (2026-07-30) — reframes everything above

E13 and E14 changed what this project believes it was measuring. Recorded here rather than
buried in the experiment log, because it invalidates the reading of several earlier results.

### What was found

Manual testing surfaced a real photograph the tile model scored at 99% AI. Measuring it properly
(E13) gave this:

| real-photograph set | called AI | median p |
|---|---|---|
| GenImage (ImageNet) — **the training source** | 45.3% | 0.461 |
| Defactify (MS-COCO) | 93.3% | 0.935 |
| archive1 (Instagram) | 99.3% | 0.939 |

The model scores its own training real source near the middle and every unseen real source at
0.93–0.94. Real photographs sit at 0.935; SDXL sits at 0.993. **There is no threshold that
separates them** — 5% false positives costs 73 points of AI recall.

E14 isolated the cause. Five arms, each trained on real photographs from one source, the AI half
held identical, real budgets equalised so only *diversity* varies:

| training real source | own source FP | held-out FP |
|---|---|---|
| CommunityForensics | 0.3% | **99.9%** |
| GenImage | 23.7% | 91.7% |
| AIGC-Benchmark | 64.0% | 88.6% |

| arm | AI recall | AUC |
|---|---|---|
| single source | 99.5–100% | 0.548–0.661 |
| **all five sources** | 99.8% | **0.884** |

### What it means

The detectors were never learning "what generated images look like". They were learning **"what
my training set's real photographs look like"**, and labelling everything outside that manifold
AI. The AI class in training was diverse (7, then 300+ generators); the real class was narrow. The
model took the easier boundary.

`IMAGE_FORENSICS_REFERENCE.md` §4.1 names the correct target: a detector should read **camera
traces** — PRNU, CFA correlation, compression history — which are physics, and therefore
source-independent. "Unlike my training set" is source identity, not physics.

### What it explains

| Earlier observation | Explanation |
|---|---|
| Tile model calls 79% of real photographs AI (E13) | Narrow real class |
| CNN catches real but misses AI; statistics models do the opposite | Same disease, opposite expression — downscaling makes unfamiliar input look smooth (CNN defaults to "real"), native texture unlike ImageNet triggers the statistics models (default to "AI") |
| Tenfold training data did not help (E12) | Volume rose; real-class **diversity** did not |
| Calibration collapses on archive1 (E6) | Instagram-processed reals are an unseen camera pipeline |
| E12's compression gap (0.9 vs 0.12 bytes/pixel) | A second axis of the same problem: the training real class is narrow in compression history too |

### What it changes

**Real-class diversity now precedes any backbone upgrade.** A stronger network — ConvNeXt,
DINOv2, CLIP — trained on the same narrow real class would answer the same wrong question more
sharply. Widening the real class costs nothing measurable: AI recall stayed at 99.8% in every arm
of E14.

Two methodology rules follow, and they apply to every experiment from here:

1. **Report in-distribution and out-of-distribution separately for BOTH classes.** Every
   experiment to date asked whether the AI class generalises to an unseen *generator*. None asked
   whether the real class generalises to an unseen *camera pipeline*. That is where the models
   were breaking.
2. **A detector's claim must be an operating point, not an AUC.** E11 reported 0.948 and stopped;
   E13 showed the same model has no usable threshold. Ranking quality and deployability are
   different claims and need to be stated separately.

## 13. Target architecture and the ordered plan (2026-07-30)

### 13a. Where the system is heading

```
                         ┌─────────────────────────────────────────┐
   image ───────────────▶│  TILING  — 128px native crops, no resize│
                         └────────────────┬────────────────────────┘
                                          │  one feature vector per tile
                   ┌──────────────────────┼──────────────────────┐
                   ▼                      ▼                      ▼
        ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
        │  hand-crafted    │   │  frozen backbone │   │  per-tile map    │
        │  68 statistics   │   │  CLIP / DINOv2   │   │  (no classifier) │
        │  physics-based   │   │  learned, strong │   │                  │
        └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
                 └──────────┬───────────┘                      │
                            ▼                                  ▼
                 ┌────────────────────┐            ┌────────────────────────┐
      MODULE 1   │ is it AI-generated?│  MODULE 2  │ where was it edited?   │
                 │ aggregate: top-3   │            │ spatial pattern of the │
                 └────────────────────┘            │ same tile scores       │
                                                   └────────────────────────┘
```

The important structural point: **tiling is shared infrastructure, not a Module 1 trick.**
Module 1 aggregates the tile scores into one number; Module 2 reads their spatial pattern. Both
modules consume the same per-tile output, which is why Module 2's core machinery already exists.

**How the three cases separate** — from the tile-score distribution, not from the aggregate:

| Case | Tile scores | Variance | Spatial |
|---|---|---|---|
| Fully AI-generated | all high | low | uniform |
| Authentic photograph | all low | low | uniform |
| Locally manipulated | most low, few high | **high** | **clustered** |

This is the answer to "won't a fully-AI image confuse Module 2" — it will not, provided we read
variance and clustering rather than the mean. Matches `IMAGE_FORENSICS_REFERENCE.md` §4.1.

**How tiles get labelled for Module 2:** from the ground-truth masks, mechanically. For each tile,
compute the mask's coverage of it — >50% inside ⇒ tampered, 0% ⇒ authentic, in between ⇒ ambiguous
(drop, or soft-label). We never tell the model where the edit is; the mask does.

⚠️ Note the two tasks are not the same question. Module 1 asks *"does this tile look generated"* —
absolute. Module 2 asks *"does this tile differ from the rest of this image"* — relative. A
tampered region is betrayed by inconsistency with its surroundings, which suggests a second,
almost-free formulation: normalise each tile's features against that image's own tile distribution
(a within-image z-score) and flag outliers. Worth testing alongside the supervised version, since
the features are already computed.

### 13b. Replacing ResNet-18 — and why "a newer CNN" is the wrong answer

ResNet-18 is a 2015 architecture and the obvious move is a modern one. But every failure measured
in this project (E5, E6, E7) was a **preprocessing mismatch, not a capacity limit**. ConvNeXt or
EfficientNetV2 would still resize to 224 and still delete the same evidence. Swapping the backbone
without fixing the input would buy very little.

The recommendation is therefore two-part:

1. **Representation — a frozen CLIP-ViT or DINOv2 backbone with a linear probe.**
   > ✅ **Tested twice. Falsified on 2026-08-04, and the falsification itself was wrong — see E16
   > and E19c.** The first run scored DINOv2 at **0.480 on Defactify** (chance) and this section
   > struck the recommendation out on that basis, with a three-part argument about semantic
   > encoders and content control. Re-run on 2026-08-05 with the label column fixed (E19b), the
   > same probe on the same images scores **0.764 on Defactify** — the highest whole-image number
   > this project has produced, above ResNet-18's 0.760 — and **40.4% AI recall at a 10%
   > false-positive budget**, against the statistics model's 33.7%. Per generator it is 0.71–0.81,
   > nothing near chance.
   >
   > So this recommendation stands, and stands stronger than when it was written. Its caveat is
   > calibration, not discrimination: at threshold 0.5 the probe calls 71.8% of archive1's real
   > photographs AI. Best ranking, worst operating point — the E11→E13 pattern in the other
   > direction.
2. **Applied to native tiles, not whole images.** A ViT resizes to 224 like everything else, so a
   whole-image CLIP probe would inherit the §4 penalty. Feeding it 128–224px native tiles combines
   the strongest known representation with the measured fix.

Cost is low: the backbone stays frozen, so this is feature extraction plus logistic regression —
the same shape as the existing feature model, and cacheable the same way.

If a fine-tuned CNN is still wanted for comparison, the modern equivalent is **ConvNeXt-Tiny** —
but trained with native crops, or it repeats ResNet's history.

### 13c. Ordered plan

> ⚠️ **Superseded on 2026-08-05 by §13d.** Steps 0, 2 and 3 were executed (E15, E17, E16) and two
> of them changed what the rest should be: Step 3's frozen-backbone probe was falsified, and E13/E14
> reframed the whole target. Kept here because the reasoning is still the reasoning — the ordering
> is what expired.

Sequenced by dependency and by value-per-hour, not by ambition. Each step's output decides whether
the next one is still the right move.

**Step 0 — Widen the real class** *(~10 min, features already cached)*
Retrain on a class-balanced, multi-source real half (§12b). Measured effect: AUC 0.55 → 0.884 at
no cost to AI recall. This precedes everything else — every other improvement is measured against
a model that currently answers the wrong question.

**Step 1 — Native-crop retraining** *(1 training run, ~1h)*
Change `RandomResizedCrop` to `RandomCrop` in `configs/`, retrain the ResNet. This is the fix §4b
points at and it costs no extra compute (§4c). Unblocks everything else: until training and tile
inference use the same input, no comparison between them is clean.

**Step 2 — Module 2's first measurement** *(no training, ~2h)*
Run the existing tile model over CASIA and CocoGlide, score the heat-map against the ground-truth
masks (pixel F1 / IoU), **reported per sub-dataset**. Expected outcome: works on CocoGlide
(diffusion inpainting), fails on classic splices. That asymmetry is the useful result and it costs
nothing to obtain. Also implements the variance/clustering readout from §13a.

**Step 3 — CLIP/DINOv2 tile probe** *(no fine-tuning, ~3h)*
The §13b recommendation. Extract frozen features per tile, fit a linear probe, evaluate on
Defactify. Direct comparison against the 68 hand-crafted features on identical tiles. Likely the
single largest Module 1 gain available.

**Step 4 — Train on the new data** *(several runs)*
CommunityForensics-Small + AI-vs-Real-balanced, tile mode (`DATASETS.md`). Only now, because
steps 1–3 decide *what* to train: which input pipeline, which representation.

**Step 5 — Leave-one-generator-out** *(k runs)*
CommunityForensics carries `model_name` for 228 generators. Hold one out, train on the rest, test
only on the held-out one. This is the only honest way to claim generalisation, and no other
dataset we hold makes it possible.

**Step 6 — Compression robustness** *(no training)*
Re-encode the test sets at JPEG-75 and WEBP and re-measure everything. Every image on the internet
is recompressed and none of our numbers account for it. Cheap, and it recalibrates every claim.

**Step 7 — Calibration**
44% of real photographs are still called "AI" at threshold 0.5. E6 showed thresholds do not
transfer across domains, so this is a research task, not a constant to tune.

**Running throughout:** ≥3 seeds on any comparison we intend to report (§5), and an audit of every
dataset *before* training on it (§1b, §2b.8).

### 13d. The phase plan (2026-08-05) — current

§13c was written before E12–E18. Four things it could not have known now set the agenda:

| Discovery | Consequence for the plan |
|---|---|
| **E13/E14** — detectors learn "what my training set's real photographs look like", not "what generation looks like" | The metric changes. Every comparison is now judged on **false positives on unseen real sources**, not AUC |
| **E16** — a frozen semantic backbone on whole images scores 0.480 on content-controlled data | §13b's Step 3 is dead as written. But its *second* half — apply a strong backbone to **native tiles** — was never tested, and becomes the centrepiece |
| **E17 extended** — only CocoGlide (diffusion inpainting) carries signal; eight classic-manipulation sets sit at chance | Module 2 is scoped to **AI-manipulated regions only**. The ELA/splice line is closed |
| **Measured 2026-08-05** — the tile pipeline sees 4.8% of a 12 MP photo, and a CNN scores a tile 20× faster than the statistics model | Full coverage is affordable. The architecture choice and the cost problem are the same decision |

**Standing rules for every phase below**
1. The headline metric is the **operating point** — AI recall at a fixed false-positive budget on
   *unseen real sources*. AUC is reported alongside, never alone (E11 → E13).
2. ≥3 seeds on anything we intend to report (§5). A single seed is a number, not evidence.
3. Audit before training (§1b), and audit the **merged** pool, not just each source (E12).
4. One commit per phase, code and documentation together.

---

#### ✅ Phase 0 — Reproducibility *(done 2026-08-05)*

Two pipeline steps existed only as artifacts on disk; the scripts that produced them were never
committed, so E12–E18 could not be rebuilt from the repo.

- [x] **0.1** `make_balanced_pool.py` — the resolution-balancing step recovered. Rule fitted against
      the surviving CSV: four bands, `min(n_real, n_ai)` each. Corrected two figures in E12 ("six
      bands" → four; "1.08× residual gap" → 1.68×) and measured that the band grid **saturates** at
      seven cut points (gap 1.00×, and finer grids buy nothing).
- [x] **0.2** `prepare_manipulation.py` — the Module 2 extraction recovered, and the data moved off
      `/tmp`, where a reboot would have destroyed the only copy. 3 → 10 sub-datasets. Found that
      three tars name their internal folder differently and were silently extracting zero files.
- [x] Bonus: E17's IoU column was measuring **mask size, not skill** — flagging at random already
      scores `f/(2-f)`. A baseline is now printed next to every IoU.

#### ▶️ Phase 1 — Pool hygiene *(~1.5 h · needs the SSD)* — see E19

Three measured defects sat in the pool everything else is about to be built on. Fixing them exposed
a fourth that nobody predicted.

- [x] **1.1** Minimum-side floor. `ai_vs_real_balanced` has a **median longest side of 32 px**;
      `features.py` reflection-pads anything under one tile, so the model is shown a synthetic
      pattern rather than a photograph. 27,153 rows dropped.
- [x] **1.2** `communityforensics` → `whole_image_safe=False`. Class 0 is **entirely** 1024², class 1
      **entirely** 512² — zero overlap, a perfect shortcut for any native-resolution model. The flag
      is now read from `SOURCES` at balancing time rather than from the index, because the index is
      a snapshot and a stale one silently restores the shortcut (39,990 rows were carrying it).
- [x] **1.3** Auditor threshold 2.5× → **2.0× inclusive**, plus a **non-overlap check** on p10–p90.
      The split above is a ratio of exactly 2.0, and a ratio cannot tell "overlapping distributions"
      from "two disjoint constants" — only the second is a perfect shortcut.
- [x] **1.4** Regenerate `DENETIM.md` for **every** dataset — 16 sets, 660 bytes → 17.9 kB. The new
      check fired on the first run: CommunityForensics now reports both a resolution trap and
      `ÇÖZÜNÜRLÜK AYRIMI (KESİN)` with ranges (512,512) vs (1024,1024).
- [x] **1.5** Rebuild the pool. → `artifacts/pool_tile_v2.csv`, **48,066 rows** (24,033 / 24,033),
      balanced on **resolution × compression jointly**.
- [x] **1.7 — the label bug (E19b).** Two of five sources declare `0 = AI` in their own metadata
      while this project uses `0 = real`, and `build_pool.py` read the raw value: **47% of the index
      was inverted**, poisoning E12/E14/E15/E16. `SOURCES` now carries `label_map` + `label_names`,
      `to_project_label()` raises on an undeclared source, `verify_labels()` raises if a dataset is
      re-exported with swapped classes, and the auditor gained a sixth check for it. The index was
      rebuilt rather than patched (a CSV that might hold raw *or* mapped labels is a double-inversion
      waiting to happen); the old one is kept as `pool_index_BOZUK_etiket.csv.bak`.
- [x] **1.8** The metadata probe moved **into** `build_pool.audit()`. The five threshold checks
      printed "none detected" for a pool the probe separates at **0.924** — medians can coincide
      while distributions differ, and a boosted tree finds that instantly.

**1.6 — the unpredicted one.** A standing **metadata probe** (predict the class from width, height,
aspect, bytes/pixel and squareness alone — archive1's AUC 1.000 test) showed that the 32px floor
*created* a compression shortcut where E12 had explicitly measured none:

| pool (labels corrected) | probe (all) | compression alone |
|---|---|---|
| raw index, 169,668 | 0.924 | 0.701 |
| class balance only, 122,772 | **0.956** | 0.730 |
| compression bands only, 53,022 | 0.912 | 0.616 |
| resolution bands only, 91,270 | 0.916 | 0.700 |
| **both, 48,066** ← shipped | **0.801** | **0.578** |

Keeping 122k rows would have meant a pool whose class is predictable at 0.956 from metadata alone.
48k clean beats 122k dirty. Note also that with the labels fixed the *raw* class resolution gap is
**1.02×**, not the 3.41× E12 measured — **most of the "3.4× merged-pool resolution gap" that E12
spent 40% of the data correcting was the label bug**, not a real bias. The bands are still needed:
matching medians does not match distributions.

Compression is the **only** metadata axis that survives into a 128 px tile — size, aspect and
squareness do not — so it is the one that had to be fixed, and joint balancing costs 2,702 rows.
The residual 0.750 is carried by axes a tile-trained model cannot see: **this pool is clean for tile
training and still unfit for whole-image native-resolution training** (§1b: a flaw is a usage
condition). Caveat: "size does not survive tiling" is true of metadata, not texture — E8's probe
predicted image width from the 68 features at 92.6% *in crop128 mode*. Re-run it in Phase 2.

**Decision taken:** 43k clean rows over 102k dirty ones. E12 (ten times the data did not help) and
E14 (diversity beats volume) both argue for clean, and the metadata probe now says the dirty pool
was 0.916-exploitable.

```bash
PYTHONPATH=src .venv/bin/python -m pixelproof.make_balanced_pool \
  --index artifacts/pool_index.csv \
  --bands 128,256,384,512,768,1024,1536 --bpp-bands 0.15,0.3,0.5,0.8,1.2,1.8 \
  --min-side 128 --dedupe --output artifacts/pool_tile_v1.csv
```

#### Phase 2 — Rebuilding the tile pipeline *(~7 h)*

The project's most visible model, and the only one the E15 real-class fix never reached.

**2a — Geometry and cost** *(no training, ~2 h)* — fix the input before training on it.

- [x] **2a.1** Remove the 36-tile cap. A 4032×3024 photo yields 713 tiles and we score 36 — **4.8%
      coverage**. The "~100% coverage" claim in §9b and the README holds only up to 768 px.
- [x] **2a.2** Edge anchoring — flush the last tile to the far edge. Loss is `C ≈ 2/k` where k is
      tiles per axis: **41% at 500 px**, and every real photograph in GenImage is exactly 500 px.
      The loss lands on the border, so edge manipulations are systematically invisible.
- [x] **2a.3** Cache the grey residual. The 3×3 median filter is **63% of a tile's cost** and the
      grey channel is filtered **twice** per tile.
- [x] **2a.4** Texture prefilter. Measuring texture costs **1/307** of a full extraction, and flat
      tiles score ~0.5 — they can never enter the top-k. ~21% saved at no accuracy cost.
- [x] **2a.5** Parallelise `score_tiles`. The demo path is a serial loop; the pool machinery already
      exists elsewhere in the repo.
- [ ] **2a.6** *(optional, deferred)* `cv2.medianBlur` — 8.9 ms → ~3 ms per tile, if border
      behaviour matches. Not needed yet: threading already brought a 12 MP photo to 2.0 s.

**Measured after 2a** — full coverage, and faster than the capped version was:

| image | before (cap 36) | after (full coverage) |
|---|---|---|
| 500×500 | 9 tiles, 59% | **16 tiles, 100%** · 0.06 s |
| 1280×1280 | 36 tiles, 36% | **100 tiles, 100%** · 0.26 s |
| 4032×3024 | 36 tiles, **4.8%** · 0.35 s | **768 tiles, 100%** · 2.00 s |

21× the tiles for 5.7× the time. Threads, not processes: the median filter and FFT release the GIL,
so 6 threads give 3.3× with no pickling and nothing to break inside a web worker.

**Two things the work surfaced.** `extract_tiles` now returns tile positions instead of letting
callers re-derive them — E17 and E18 were calling `tile_positions()` separately and zipping by
index, which is only correct while nothing filters, and the texture floor filters. And the floor
needed a `min_tiles` guard: a mostly-flat photograph lost 19 of 20 tiles, and a top-3 mean over one
survivor is one tile's opinion wearing an average's clothes.

**2b — Three models on identical tiles** *(training, ~4 h)* — one variable: the model.

| arm | model | rationale |
|---|---|---|
| A | 68 statistics + gradient boosting | the incumbent, retrained on the clean pool |
| B | **ResNet-18 @128 px**, fine-tuned on native tiles | §13b's untested second half. Already in `MODEL_REGISTRY`; fully convolutional |
| C | **SmallCNN @128 px** | `AdaptiveAvgPool2d(1)` already accepts any input size — a data change, not an architecture change |
| D | *(deferred — see below)* **Noiseprint++** | the only pretrained model that genuinely encodes camera traces — self-supervised on real photographs (§4.4) |

**Arm D — Noiseprint++: parked 2026-08-05 with everything needed to pick it up.**
The most interesting arm and the only one with real integration risk, so A/B/C build the harness
first and D slots in afterwards as "one more feature extractor".

| | |
|---|---|
| weights | `https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip` |
| size | **249 MB**, md5 `7bee48f3476c75616c3c5721ab256ff8` |
| licence | GRIP-UNINA — *"informational and nonprofit purposes"*, which fits this project |
| code | `github.com/grip-unina/TruFor`, network definition under `TruFor_train_test` |

⚠️ **Noiseprint++ is not packaged separately.** It is the low-level encoder inside the full TruFor
model, so using it alone means downloading the bundle, taking the network definition from the repo
and extracting that submodule's weights. Estimate 1–2 hours, with a genuine chance of getting
stuck on the third step.

**Why it stays on the list:** every other backbone here learned objects from ImageNet.
Noiseprint++ is trained self-supervised on **real photographs only** and is sensitive to camera
model — the one pretrained representation that encodes the camera traces §4.1 names as the correct
target, instead of "unlike my training set" (§12b). It is also the honest answer to "a backbone
that already knows camera traces", which an ImageNet CNN is not.

**✅ Run 2026-08-06 — see E20. ResNet-18 wins, and by a lot.**

| AI recall @ 10% FP budget | statistics | **ResNet-18 @128** | SmallCNN @128 |
|---|---|---|---|
| Defactify, all five generators | 39.0% | **55.5%** | 30.5% |
| Midjourney alone | 5.0% | **51.5%** | 24.5% |
| Defactify AUC | 0.603 | **0.770** | 0.655 |

**55.5% is the best operating point the project has produced**, from 39.0% on identical tiles — and
above E19c's whole-image DINOv2 probe (40.4%). The recommendation in §13b had two halves; the first
(a strong pretrained representation) was tested on whole images in E16 and the second (apply it to
**native tiles**) was never tested until now. It is the half that mattered.

SmallCNN losing to both is the informative part: 0.3M parameters from scratch does worse than
hand-crafted physics on the same data. So the win is **ImageNet pretraining**, not "a CNN".

**And the bottleneck moved.** At threshold 0.5 all three arms call 86–99% of real photographs AI —
including 86.5% of 2,314 authentic photographs from ten forensics datasets. Ranking is now good and
deciding is still impossible. It is no longer the data (Phase 1) and no longer the representation
(this experiment): **it is calibration.** DALL-E 3 also stays broken in all three arms (4–9.5%,
AUC at or below chance) — 270px at 16 KB has no texture to read, and that needs a different route
rather than a better tile model.

Throughput on 713 tiles (full coverage of a 12 MP photo), measured 2026-08-05:
statistics serial **6.3 s** · statistics parallel **0.6 s** · SmallCNN **0.40 s** · ResNet-18
**0.31 s**. The CNN is faster because it batches, not because it is smaller — so choosing the
architecture also dissolves the cost problem.

⚠️ ImageNet backbones do not know camera traces; they know objects, and E16 charged 0.480 for
assuming otherwise. The difference here is that a 128 px native tile rarely contains a recognisable
object, so the model is shown texture. Expect the early layers to earn their keep and the semantic
layers to be the risk.

**2c — Aggregation and scale** *(~2 h)*

- [ ] **2c.1** Extend the grid sweep past 36 (49 / 64 / 100). E11 stopped at 36 while the score was
      **still climbing** (0.760 → 0.821); the cap was never justified.
- [ ] **2c.2** Sweep `k`: fixed count vs fraction vs adaptive vs soft texture weighting. `3` was the
      best of a five-item menu — **2, 4 and 5 were never tried** — and once the cap rises a fixed 3
      silently degenerates toward `max`, which already measured worse.
- [ ] **2c.3** Adaptive threshold `median + λ·MAD`, so `k` becomes a measurement rather than a
      hyperparameter. It also separates the two cases Module 2 cares about: everything high (fully
      AI) versus a few outliers (local manipulation).
- [ ] **2c.4** Multi-scale training (128/192/256), which unlocks tiling at `B' = A/round(A/B)` —
      zero remainder loss and zero overlap.

#### Phase 2c-bis — Calibration and source transfer *(revised 2026-08-06)*

E20 moved the bottleneck, but protocol v2 then located it more precisely. The ResNet ranks
Defactify reasonably (AUC 0.770), yet a threshold fitted to 10% FP does not even hold on an
untouched half of the same source (19% FP), and collapses across other real pipelines (45% macro,
96% worst-source FP). This is **calibration plus source-domain shift**, not a global scalar that
temperature scaling alone can repair.

- [x] **Why the scores pile up high.** The top-3 aggregate is an upper-tail statistic over a
      variable number of tiles, so it is biased upward by construction and the bias grows with tile
      count. Protocol v2 compared top-10%, p90, mean and fixed-16 controls. They reduce false
      positives but remain unusable; DSO-1 has more tiles than the worst source and far fewer false
      alarms. Arithmetic contributes, but source/pipeline shift dominates.
- [x] **Test threshold transfer before fitting a fancier calibrator.** A Defactify-calibrated 10%
      threshold yields 19% on its untouched half and 45% macro / 96% worst on ten forensic sources.
      A single global threshold is empirically falsified for the present representation.
- [ ] **Source-balanced real calibration/training**, but only after external pretrained baselines.
      Temperature and isotonic transforms are monotonic: they can rename a score but cannot repair
      a source whose authentic images rank above synthetic ones. Any calibrator must be fitted on
      multiple disjoint real pipelines and tested on held-out pipelines.
- [ ] **An explicit "insufficient evidence" band** rather than a forced call. §11's floor is 48px;
      the principled version keys on measured high-frequency content, and DALL-E 3's collapse
      (4–9.5% recall at AUC ≤0.36 across all three arms) is exactly the population that should fall
      into it rather than receive a confident guess.

**Protocol hardening implemented; checkpoint diagnostic complete; full retrain deferred.** The E20
script now saves every per-image/per-tile score and makes the aggregation rule part of the measured
contract. Defactify real and each generator are split into stable calibration/evaluation halves;
aggregation and threshold see calibration only. Top-3 is compared with top-10%, p90, mean and a
fixed-16-tile control. The selected threshold is then transferred unchanged to all ten forensic
real sources, reported as macro and worst-source FP. Runs intended for the report default to three
seeds, and checkpoints store the full inference/provenance contract.

The existing ResNet seed-42 checkpoint was evaluated before retraining: selected top-3 gives
61.4% untouched AI recall and AUC 0.770, but 19% Defactify FP, 45% forensic macro FP and 96%
worst-source FP. Mean/fixed-16 controls still leave 58%/61.5% worst-source FP. The model is useful
as a research baseline and **not deployable**. Full three-seed retraining stays pending, but a
frozen external baseline now has higher information value than repeating the same failure nine
times.

- [x] **2c-bis.0** Implement raw score persistence, disjoint calibration/evaluation, aggregation
      controls, source-transfer metrics, three-seed default and protocol tests.
- [x] **2c-bis.1a** Evaluate the existing winning ResNet checkpoint under v2 as a go/no-go gate;
      result is no-go for deployment.
- [ ] **2c-bis.1b** Compare frozen **B-Free first, then CLIP**, under the same saved-score/source-
      transfer protocol. Noiseprint++ remains the camera-trace specialist after those lower-risk
      integrations. No external model earns an API route from its paper's own benchmark.
- [ ] **2c-bis.2** Run three seeds for the best surviving representation/configuration. Do not pay
      for nine confirmatory runs before the model family passes the cross-source specificity gate.

#### Phase 2.5 — Module 2, AI-only *(~3 h)*

Scope decision of 2026-08-05: Module 2 targets **AI-manipulated regions**, not manipulation in
general. Of nine measured sub-datasets, only CocoGlide (diffusion inpainting) carries signal; the
eight classic sets sit between 0.326 and 0.548 at image level. They earned their keep once, as the
specificity control that proves the model reads AI texture rather than "something was edited".

- [ ] **2.5.1** Loosen E17's mask-coverage filter. Only **35 of 120** images survive it today; a
      half-tile step yielded 152 AI-filled tiles in a probe. **~10× more data, zero download.**
- [ ] **2.5.2** Re-run E17 on the Phase 2 tile model — Module 2 rests entirely on it.
- [ ] **2.5.3** Test the noise-energy signal. Measured on 120 image/original pairs: noise energy
      **halves** inside AI-filled regions (0.0164 → 0.0088), exactly as §2 predicts.
- [ ] **2.5.4** Close the ELA/splice line (E18). It answered a question we have decided not to ask.
- [ ] **2.5.5** TGIF test split, when on Wi-Fi. §5 names it "the closest match to this project's
      Module 2", and it is the only source that separates **spliced from fully-regenerated** —
      the distinction that decides whether localisation is honest at all.

⚠️ The CFA measurement came out opposite to theory on CocoGlide, most likely because its
"authentic" images are MS-COCO — already-downscaled web photographs. **We cannot measure camera
traces on data whose camera traces are already gone.** Phase 4 is the fix.

#### Phase 3 — Two measured gaps *(~3 h)*

- [ ] **3.1** Compression augmentation (JPEG q30–q95). Pool sits at **0.43–1.59 bytes/pixel**,
      Defactify at **0.14–0.16** — a 3–10× gap. E12 named this fix and skipped it.
- [ ] **3.2** A deployment-matched validation set: 30 phone photographs and 30 AI images at native
      size, both pushed through one identical encoder. What the demo actually does on real input is
      currently unknown.
- [ ] **3.3** An honest "insufficient evidence" threshold based on measured high-frequency content
      rather than pixel count. Where compression has removed the evidence, no method can recover it,
      and the correct output is a refusal rather than a guess.

#### Phase 4 — New data axes *(~4 h · needs the SSD)*

- [ ] **4.1** Add a personal photo library as a **sixth** real source — the pool's only intact,
      full-resolution camera output. As a sixth source, never as the only one (E14).
- [ ] **4.2** Fold in `gpt-image-mega-4k` and `nano-banana-pro`: downloaded, audited, never used.
- [ ] **4.3** Resolution augmentation (`native + ÷2 + ÷4`).
- [ ] **4.4** Run `shortcut_probe()` after every addition — measure the leak, do not assume it away.

#### Phase 5 — Documentation alignment *(~1 h)*

- [ ] README's 0.948 headline against the E13/E15 reality
- [ ] §14 checklist, frozen at E11
- [ ] The **`v3` name collision**: `feature_full_v3` (E12, 256 px floor) is not the docs' "v3"
      (E15, balanced multi-source) which is `feature_full_v4`
- [ ] `feature_full_v4` loads but no API route reaches it
- [ ] The "~100% coverage" claim (true only ≤768 px)
- [ ] "Every experiment is single-seed" — no longer true

#### Dependency chain

```
Phase 0 ✅
  └─> Phase 1   pool hygiene            SSD
        └─> Phase 2a  geometry + cost
              └─> Phase 2b  three models        ← the decisive step
                    └─> Phase 2c  k rule, multi-scale
                          ├─> Phase 2.5  Module 2
                          └─> Phase 3    compression, deployment set
                                └─> Phase 4    SSD
                                      └─> Phase 5
```

## 14. Progress Checklist

- [x] Datasets inspected (CIFAKE 100k/20k + external OOD set)
- [x] Python env with PyTorch + MPS
- [x] Baseline CNN, config-driven training pipeline
- [x] Evaluation script (test metrics + OOD) and CLI predictor
- [x] Baseline training run finished (20 epochs, MPS): best val accuracy **96.8%**
  - Held-out test (20k): **accuracy 96.75%, F1 0.967, ROC-AUC 0.995**
  - External OOD set (995 high-res images): **accuracy 77.1%, F1 0.570, ROC-AUC 0.800** — the expected resolution-driven generalization gap; motivates Phases 2–3
- [x] Phase 2a: embeddings + classical classifiers — all four match the CNN head (±0.2 pts) → representation is the bottleneck, not the classifier (see `ml/EXPERIMENTS.md` E2)
- [x] Phase 2b: k-means + t-SNE analysis — test embeddings nearly linearly separable (purity 0.965); structure collapses on OOD (purity 0.749, ARI 0.013); errors concentrate in the borderline band |p−0.5|<0.1 (see E3)
- [x] Phase 2c: learning curve — accuracy linear in log(data) (93.8% @10k → 96.75% @90k, each doubling ≈ +1 pt); overfitting gap shrinks 5.1 → 1.2 pts; not saturated at 90k (see E4)
- [x] Phase 3 (transfer learning): ResNet-18 fine-tune — best test acc **97.66%**, but OOD collapsed to 25.2% due to 32→224 upscale domain shift; control experiment (32px bottleneck) recovers 72% → strongest motivation for Phase 4 (see E5)
- [ ] Phase 3 (remaining): frequency-domain ensemble; EfficientNet comparison (deprioritized until Phase 4 fixes the data domain)
- [x] Phase 4: ResNet-18 retrained on natively high-res GenImage — best OOD discrimination so far (archive1 ROC-AUC 0.80 → **0.888**, AI recall 60% → 89.6%); calibration under distribution shift identified as the remaining gap; CIFAKE forgotten (catastrophic forgetting) → motivates dual-model routing (see E6)
- [ ] Phase 5: unsupervised anomaly-detection track (train on real only)
- [x] Phase 6: web demo — FastAPI inference service (dual model, resolution routing, uncertainty band) + upload/analyze UI, verified end-to-end locally
- [x] Mentor decisions (2026-07-23): transfer learning approved; two-module architecture approved

### Session 2026-07-27 → 07-29

- [x] **E7 — modern-generator stress test.** Built the Defactify test set (16,875 images, 5 generators all newer than training). AUC 0.888 → **0.760**. Discovered results order by *source resolution* → downscaling is destroying the evidence (§4b)
- [x] **E7 control — native patches.** Confirmed the mechanism (high-res generators improved) but exposed the mismatch: 96% of real photos called "AI". Third instance of the preprocessing law (§4a)
- [x] **E8 — resolution-independent features.** 68 hand-crafted statistics + gradient boosting, trained on the same GenImage split as the ResNet. A specialist: wins by +0.09…+0.15 on high-res generators, collapses on small compressed ones (§9a)
- [x] **E9 — ensemble.** Negative result: best rule beats the ResNet by +0.002. Relocates accuracy rather than adding it (§9d)
- [x] **E10 — archive1 audit + two controls.** Dataset is maximally confounded (metadata alone → AUC 1.000), but both CNNs are immune because `Resize()` destroys format and dimensions before they see anything. **E1's 77.1% and E6's 0.888 stand.** Immunity does NOT transfer to native-resolution methods
- [x] **Tiling.** 6×6 grid + top-3 mean, no retraining. **SDXL 0.948 · SD 3 0.894 · SD 2.1 0.863** — the project's best scores. Measured crossover at ~700px replaces the invented 128px threshold (§9b)
- [x] **Demo rebuilt.** Three named signals side by side, score-based agreement flag, evidence floor at 48px (§11)
- [x] **255 GB of audited datasets** on the LaCie SSD, with an automated auditor and a per-dataset verdict (§1b, §1c)
- [x] **Module 2 unblocked.** Mask-annotated manipulation data acquired; localisation is now measurable rather than hypothetical (§12)

### Session 2026-07-30 → 08-04

- [x] **Pool infrastructure.** `build_pool.py` indexes five sources into one manifest and runs three checks before any training. The contamination check caught **38 pool images that were perceptual-hash matches for Defactify test images** — without it we would have trained on our own test set
- [x] **E12 — ten times the data.** 9.9k → 101k. archive1 +0.133, DALL-E 3 −0.189, a wash elsewhere. Two hypotheses tested: low-resolution contamination **refuted**, compression domain gap (7×) **supported**
- [x] **E13 — the operating point.** The tile model calls **79% of real photographs "AI"**. Real photos sit at 0.935, SDXL at 0.993 — no usable threshold exists. E11's 0.948 is genuine ranking information and simultaneously undeployable
- [x] **E14 — the cause.** A narrow real class. One source → 88–99.9% false positives on other sources; five sources → AUC 0.55 → **0.884** at no cost to AI recall (§12b). Reframes every earlier result
- [x] **E15 — Step 0 applied.** Class-balanced multi-source real half: archive1 0.706 → **0.904**, false positives 30.1% → 19.8%. Defactify untouched (0.717 → 0.692) — **E14's gain was within-pool and does not transfer**
- [x] **E16 — DINOv2 falsified.** Frozen ViT-S/14 at 518 px: **0.480 on Defactify**, chance. It is a semantic encoder and Defactify is content-controlled. Carries a warning backwards: GenImage is *not* content-controlled, so every GenImage number inherits the doubt
- [x] **E17/E18 — Module 2 measured.** Tile localisation works on diffusion inpainting (CocoGlide 0.648 / 0.721) and not on classic splices (CASIA image-level 0.481). ELA's positive control passes (0.719) while the compilation's PNG pipeline removes its input

### Session 2026-08-05

- [x] **Two lost pipeline steps recovered** (§13d Phase 0). `make_balanced_pool.py` and `prepare_manipulation.py`; the Module 2 data rescued from `/tmp` hours before a reboot would have deleted the only copy; 3 → 10 sub-datasets
- [x] **E17 extended to nine sub-datasets**, and its IoU column shown to be measuring **mask size rather than skill** — NIST2016's 0.864 is +0.041 over random flagging, while CocoGlide's 0.419 is +0.155
- [x] **Tile-pipeline cost and coverage measured.** 4.8% coverage on a 12 MP photo; `C ≈ 2/k` remainder loss (41% at 500 px); the 3×3 median filter is 63% of a tile's cost and runs twice on the grey channel; ResNet-18 scores a tile **20× faster** than the statistics model
- [x] **Module 2 scoped to AI-manipulated regions only** — eight classic-manipulation sets measured at chance; they served as the specificity control and that job is done

### Next — the plan lives in §13d

- [ ] **Phase 1** Pool hygiene — 32 px floor, CommunityForensics shortcut, auditor threshold, full audit report
- [ ] **Phase 2a** Tile geometry and cost — remove the cap, anchor the edges, cache, prefilter, parallelise
- [x] **Phase 2b** Three models on identical tiles — **ResNet-18 wins ranking, but fails source transfer** (E20-v2: 45% macro / 96% worst real FP)
- [ ] **Phase 2b-2** Noiseprint++ arm — parked with download URL, size and licence in §13d
- [ ] **Phase 2c-bis** Source-robust representation — B-Free then CLIP under E20-v2; calibrate only a survivor
- [ ] **Phase 2c** Aggregation rule and multi-scale tiles
- [ ] **Phase 2.5** Module 2, AI-only — loosen the filter, re-measure, test noise energy, fetch TGIF
- [ ] **Phase 3** Compression augmentation + a deployment-matched validation set
- [ ] **Phase 4** New data axes — intact camera photographs, 4K generators, resolution augmentation
- [ ] **Phase 5** Documentation alignment

Also open, not on the critical path:
- [ ] Leave-one-generator-out over CommunityForensics' 228 models — still the only honest generalisation claim available to us
- [ ] Calibration as a research task (E6: thresholds do not transfer across domains)
- [ ] Build a small ChatGPT/Gemini test set by hand, both classes through one identical pipeline — the only uncontaminated route to 2026-era editing models
- [ ] Verify the CommunityForensics slice we hold (47 of 260 GB) is representative before training on it
- [ ] Seed variance on E1–E11, which predate the ≥3-seed rule (§5)

## 15. Session 2026-08-24 — repository hardening and the model-first milestone

The E20-E27 research line had produced meaningful measurements, but a full repository audit found
that the serving, browser contract, reproducibility and documentation did not yet enforce the same
discipline as the experiments. The correction was executed as H0-H6, with every phase planned,
measured, recorded in `PLAN.md` and committed separately:

| Phase | Commit | Durable result |
|---|---|---|
| H0 | `ef9edaa` | Ordered hardening roadmap recorded before implementation |
| H1 | `6509ebf` | Web lint, typecheck, build and current-product tests restored |
| H2 | `18ab632` | Deployment API origin, runtime response validation and stale-request cancellation |
| H3 | `364d9f0` | Bounded image decoding/inference, normalized inputs, restricted CORS and truthful health |
| H4 | `d0d856d` | E27 evaluation leak corrected; calibration-only rerun failed G1 and the arm was removed |
| H5 | `dbafd05` | Locked Python serving environment and hash-verified artifact registry |
| H6 | `9830d31` | Documentation alignment, CI/Dependabot and real local end-to-end verification |

The most important scientific outcome was a refusal, not a higher score. E27's original union
procedure could consult evaluation halves while choosing its threshold. After the threshold was
fitted from calibration sources only, the candidate moved from 15.38 to 21.71 and GPT-probe recall
fell from 40.5% to 14.5%, below the pre-registered 40% gate. The failed arm was removed from serving
and the correction was appended to `ml/EXPERIMENTS.md`; the earlier entry was retained as history.

Repository verification at the end of H6 passed 6/6 web tests and 25/25 Python tests, plus lint,
TypeScript, production build, Python dependency checks, artifact hashes and a real API/web smoke
run. Two high-severity `vinext -> image-size` advisories remain explicit dependency debt because
npm's available remediation crosses into the breaking vinext beta line.

### Model-first transition

The immediate project goal was then narrowed deliberately: before production deployment or another
research branch, make one project-owned model easy to start, test and present. Commit `774520b`
recorded M0-M6 in `PLAN.md` before implementation. The canonical checkpoint is the E20 native-tile
ResNet-18, seed 2024 (`artifacts/tile_resnet18_seed2024.pt`): 128 px tiles, ImageNet normalization,
0.04 texture floor, `top3` aggregation and calibration-only threshold 0.9894907.

This is a working project result, not an authenticity certificate. Its three-seed result is
Defactify evaluation AUC 0.751 +/- 0.033 and recall 49.9% +/- 6.1; worst-source authentic false
positives remain 86.2% +/- 3.1. M1-M6 therefore focus on a verified artifact, one canonical API/UI
path, repeatable folder evaluation, one-command local demonstration and traceable presentation
evidence, with the limitation displayed rather than hidden.

### M1 — the E20 model becomes a verified runtime artifact

The canonical checkpoint was previously a local experiment output: it existed and carried a good
internal inference dictionary, but the runtime artifact registry did not know its identity and no
owned module enforced the dictionary. M1 added it as `e20-tile-resnet18-seed2024`, with SHA-256
`b9f39eda10ba3de54b706d6448b67d93ce8e4c7bae97a685f3c1b57ebfd65adf`, E20-v2/seed-2024
provenance, label direction and project-local licence boundary.

`pixelproof.project_model` now verifies that hash before `torch.load`, then rejects a checkpoint
unless its arm, state dict, seed, validation metadata, tile size, normalization, texture floor,
selected aggregation, threshold and calibration split are compatible. The same module owns
batched normalized tile scoring, checkpoint-selected aggregation and threshold comparison, so M2
does not need to copy experimental constants into the API.

Measured locally: the real 44,789,451-byte artifact loaded on CPU, reproduced seed 2024,
`top3`, threshold 0.9894907 and validation AUC 0.909627, then completed a three-tile score and
aggregation. Tests cover valid, missing, tampered and incompatible artifacts. The full Python
suite passed 29/29; compileall, `pip check` and all six default artifact hashes passed. This phase
does not claim image-level API availability yet — that is M2's separately testable boundary.

### M2 — one image reaches the project model through API and CLI

M2 made `project_model` the FastAPI and CLI default and connected it to the single verified E20
loader from M1. One shared image scorer now selects native 128 px tiles, applies the checkpoint's
0.04 texture floor, scores every retained tile once in batches, aggregates with stored `top3` and
compares against stored threshold 0.9894907. Spatial sampling remains capped at 256 tiles.

The response is deliberately evidence rather than marketing: raw score, experimental threshold,
trigger state, `research_only=true`, the measured 86.2% worst-source FP limitation, artifact id,
full SHA-256, E20 revision, seed, aggregation and tile count travel together. A negative result is
`uncertain`, never “real”. The older methods and external decision can still be returned for
comparison, but their readiness is independent; a missing retired artifact cannot disable the
canonical project model.

Measured on the real MPS runtime with `ml/artifacts/figures/generators.png`: 51 texture-qualified
tiles produced 0.2409 through the common scorer and exactly 0.2409 through HTTP. The CLI, invoked
from the repository root without a path override, reported the same rounded 0.241 against 0.990.
Health reported `project_model_ready`, legacy `core_ready` and external `decision_ready`
separately. Automated coverage includes padded 64 px input, a 2304 px input held to exactly 256
tiles, unavailable artifact and project-only readiness. Python passed 33/33; web lint, typecheck,
build and 6/6 tests remained clean. The model works end to end; M3 now makes that path visible as
the primary web experience.

### M3 — the web demo becomes model-first

M3 removed the four-method chooser from the main interaction without deleting the earlier research
paths. `project_model` is now the initial selection and the primary action explicitly runs the E20
ResNet-18. Its result card keeps the raw score beside the stored experimental threshold, trigger
state, E20 revision, tile count, `top3` aggregation and verified artifact hash prefix. An
under-threshold result says only that the experimental threshold was not crossed and explicitly
warns that this does not prove the image is real.

The measured limitation is part of the result rather than a footnote: E20's 86.2% +/- 3.1
worst-source authentic false-positive result is visible in the interface. The frozen E26 decision
layer remains available as a separately titled external comparison, including its own provenance
and caveats. The old `auto`, `cnn`, `stats` and `tiles` paths moved behind an optional research
disclosure and their output is labelled as a legacy score that does not replace the canonical
model.

The browser contract now validates every project-model response field before publishing it,
including score ranges, the 64-character checkpoint SHA-256, integer seed, positive tile size and
positive tile count. The existing request gate still prevents an older response from replacing a
newer one, and upload controls retain keyboard-visible button semantics and an accessible file
label. Verification passed `git diff --check`, ESLint, TypeScript, the production Vinext build and
all 6/6 web tests. Public hosting remains deferred: this phase completes the local, testable model
experience without changing the deployment boundary.

### M4 — repeatable labelled-folder evaluation

M4 added the installed `pixelproof-evaluate-project` command around the same verified loader and
image scorer used by API and CLI. It accepts distinct `real/` and `ai/` roots, recursively keeps
their source folders, applies the serving decoder limits, scores each supported image once and
writes a self-contained `results.json` plus row-level `predictions.csv`. The report carries the
full model metadata and SHA-256, input/configuration limits, Python/platform/library/device state,
exact command, working directory and git revision/state.

Failure accounting is part of the data contract. Read, decode and inference failures retain a row,
are counted per class and source, and make a partially completed command exit non-zero after its
evidence is written. The evaluator refuses nested class roots, empty classes and non-empty output
directories. Successful rows produce ROC-AUC, recall, false-positive rate, accuracy and TP/FN/FP/TN
at the checkpoint's stored threshold; no new threshold is fitted to the user's evaluation set.

Tests with controlled images proved once-only scoring, exact metrics and schema, per-folder
breakdown, malformed-image retention and overwrite refusal. The suite rose from 33 to 36 tests and
passed 36/36. The newly installed command then loaded the real canonical checkpoint on MPS and
processed the pinned B-Free checkout's two real and two AI demo images: 4/4 succeeded, AUC 0.500,
recall 1.000, FP rate 1.000, TP=2/FN=0/FP=2/TN=0. Both real examples crossed the threshold. The
exact scores are preserved in `ml/EXPERIMENTS.md`; the result verifies the runnable evaluator and
again demonstrates E20's source-transfer weakness rather than claiming performance from four
samples.

### M5 — one-command local model demonstration

M5 replaced the manual two-terminal startup recipe with executable
`./tools/pixelproof-demo`. Its `check` command verifies the active venv and imports, `pip check`,
the canonical artifact registry entry and hash, installed prediction/evaluation CLIs, Node version,
npm dependency tree and requested loopback ports. Every failure exits with the missing component
and a concrete repair instruction. `smoke` checks a running API's health identity, submits a real
multipart image and refuses responses that omit the research-only flag, verified hash, bounded
score/threshold, positive tile count or asymmetric `ai`/`uncertain` result.

`start` composes those operations: preflight, loopback API, readiness wait, real smoke inference,
loopback web UI and readiness check. It owns both child process groups, treats an early exit as an
error and shuts both down on one `Ctrl+C`. A new `PIXELPROOF_RUNTIME_PROFILE=project` mode prevents
the primary demo from loading retired research and optional external models; the default runtime
profile remains `full` for comparison work. Tests cover both profiles, invalid profile rejection
and the smoke response invariants.

The final live run verified Python 3.13.5, Node v25.2.1, both dependency graphs, both installed
project CLIs, ports 8799/3000 and the complete E20 artifact. On MPS the tracked generators figure
returned 0.2409 against threshold 0.9895 from 51 tiles with the canonical hash. The Turkish E20 web
shell then returned HTTP 200 at `127.0.0.1:3000`; one interrupt stopped both services cleanly with
exit code zero. Python passed 41/41. Exact operational evidence is also appended to
`ml/EXPERIMENTS.md` so the internship report does not depend on memory or an unrecorded terminal.

### M6 — model card and internship evidence freeze

M6 completed the runnable-model milestone with two current documents outside the historical
`rapor/` snapshot. `MODEL_CARD.md` freezes the canonical artifact identity, exact five-source
48,037-tile training inventory, architecture and preprocessing, calibration-only inference
contract, three-seed and deployed-seed measurements, intended uses, prohibited claims and known
failure modes. `PRESENTATION_EVIDENCE.md` turns M0-M5 into a presentation-ready commit/test ledger,
maps each claim to its durable and raw source, and gives a six-step live-demo sequence. The
historical report boundary now points to both files rather than implying it contains M1-M6.

The presentation disagreement is also machine-readable. `evidence/demo_disagreement.json` binds
the input path, upstream real label, byte size, dimensions, SHA-256 and pinned B-Free checkout
revision to runtime commit `95fe2b2`, device/profile, E20 artifact hash and both displayed result
blocks. Tests cross-check those identities against `artifacts.manifest.json`, verify the optional
local input bytes when present and require every M0-M5 commit in the presentation ledger.

The example was reproduced through real `POST /predict` on the full MPS runtime. On upstream-real
`img0000.png`, E20 returned 1.0000 against 0.9895 from 69 tiles — a false positive — while the
external CF-ViT comparison returned -2.4631 against 0.6617 and therefore `insufficient`. This is
the intended teaching example: the project model demonstrably runs, yet its source-transfer
failure is visible in the same evidence package. Python passed 43/43 after adding the evidence
integrity tests. M0-M6 are now complete; deployment and stronger-model research remain explicitly
deferred rather than being confused with the achieved runnable-model goal.

### N0 — source-robust model v2 is pre-registered before code

With the runnable E20 path complete, development returned to its largest measured model failure:
seed 2024's 83.2% worst-source and 43.3% macro authentic false-positive rates. The next candidate
is an independently implemented Stay-Positive linear head over the frozen E20 ResNet18 backbone.
The published algorithm, exact E20 comparator, data boundary, single-seed advancement gate and
three-seed integration gate were recorded in `PLAN.md` and `ml/EXPERIMENTS.md` before any training
code changed. The official research repository is not vendored because its reviewed page exposed
no explicit software licence; only the paper's described method is being reimplemented. This entry
starts a new experiment line and does not alter the currently served model.

### N1 — the constrained-head candidate is implemented without touching serving

The new `pixelproof.stay_positive` module independently implements the paper's small algorithm:
E20's feature extractor is frozen, normalized tile embeddings are made explicitly non-negative, a
linear head starts from zero, and feature weights are clamped to non-negative values after every
optimizer step while the bias remains free. An installed experiment command writes only to the
ignored `artifacts/e28/` candidate area; the registered E20 checkpoint and all API/web behavior are
unchanged.

Five focused tests and a real 120-tile CPU smoke established the mechanical contract. The smoke
checkpoint loaded back into ResNet18, selected epoch 1 at validation AUC 0.9000 and had no negative
feature weights. The complete Python suite rose from 43 to 48 tests and passed 48/48; compileall and
`pip check` also passed. This phase proves the experiment can run, not that the model generalizes.
Only N2's already-frozen seed-2024 evaluation may answer that question.

### N2–N4 — E28 is rejected and never reaches serving

The full seed-2024 experiment used all 48,037 E20 tiles and kept epoch 1 solely from its
source-stratified training validation slice (AUC 0.8947). Under the untouched E20 protocol,
calibration selected `top3`; evaluation measured AUC 0.7290, recall 48.9%, Defactify FP 12.7%,
forensic macro FP 44.6% and worst-source FP 85.0%. Against the pre-registered advancement gate,
the AUC, recall and Defactify conditions passed, but macro <=35% and worst-source <=70% failed.

The tempting lower-FP aggregations are retained as diagnostics, not substituted after the fact:
`p90`, for example, reached 29.4% macro and 59.0% worst-source FP but only 35.6% recall, below the
frozen 42% floor. Seeds 42/1337 were therefore not run, the E28 candidate was not added to the
artifact manifest, and API/web serving remained on the verified E20 checkpoint. Compact exact
evidence and hashes live in `evidence/e28_seed2024_rejection.json`; the large candidate and raw
scores remain ignored local artifacts. The outcome narrows the next problem: E20's representation
or data composition must change, because constraining only its final head did not cure source
shift.

### O0 — the next line changes representation, not the rejected threshold

After E28's rejection, the next feasibility candidate was recorded before implementation: RINE's
intermediate CLIP encoder-block representation with learned block importance. The official ECCV
2024 repository is Apache-2.0, but code licence, checkpoint terms, transitive CLIP weights and
training-data rights will be audited separately before anything is installed or downloaded. If
that audit passes, the candidate remains isolated from serving and must meet PixelProof's own
source-wise gate before a project-trained head is attempted. No external code, dependency, weight
or runtime change was made in this planning phase; E20 remains the working model.

### O1 — RINE is conditionally admissible only as an isolated local benchmark

The provenance audit pinned RINE revision `9b7fd585...620` and its Apache-2.0 code, the official
25.3 MB 4-class trainable checkpoint, OpenAI CLIP revision `d05afc4...35f6` and the official
932.8 MB ViT-L/14 base hash. RINE's save path excludes all CLIP parameters, but a separate licence
for redistributing the CLIP base bytes was not found, so those weights may remain local only.

The audit also refused the upstream environment recipe: it is unpinned, assumes Python 3.9,
Torch 2.1.1 and CUDA, and dynamically assigns checkpoint fields with `exec`. None of that entered
the locked serving environment. `ml/RINE_FEASIBILITY.md` records the full PASS/FAIL matrix, exact
input normalization, score direction, storage/memory estimate and O2 boundaries. The result is a
conditional GO for a pinned, hash-verified, strict CPU/MPS research adapter—not for integration.

### N/O development commit ledger

| Phase | Commit | Archived outcome |
|---|---|---|
| N0 | `e5fd980` | Stay-Positive method, data boundary and stop/go gates pre-registered |
| N1 | `d488a75` | Independent constrained-head command, tests and real-checkpoint smoke added |
| N2–N4 | `2ade393` | Full seed-2024 E28 gate failure frozen; candidate excluded from serving |
| O0 | `c601e25` | Representation-first RINE/CLIP feasibility direction recorded before work |
| O1 | `8d31806` | Pinned licence, provenance, runtime and redistribution audit completed |

This table is an index into Git's immutable record. The detailed measured values remain in the
preceding append-only entries, `ml/EXPERIMENTS.md` and the compact E28 evidence JSON.

### P0 — the owner's gallery exposes an iPhone MPO decoder gap

A local, authentic-only gallery test stopped before model comparison: 187 of 210 still images use
iPhone's two-frame MPO JPEG container, which Pillow reports as `MPO`; PixelProof allowed only the
literal `JPEG`, `PNG` and `WEBP` identifiers. Twenty-three files decoded, but that format-selected
subset cannot support a fair model conclusion.

The correction was scoped before implementation: accept only the primary MPO frame through the
same bounded decoder, keep MOV unsupported, then rerun every still once without training,
calibration or threshold changes. Only aggregate results will enter the repository; personal
images, GPS, filenames and per-image hashes remain outside it. RINE O2 pauses until this direct
product blocker and its complete real-camera measurement are resolved.

### P1 — iPhone MPO primary frames now enter the shared bounded decoder

PixelProof now accepts `MPO` as a JPEG-family container and explicitly decodes only frame zero;
auxiliary iPhone frames are never scored. Existing byte, geometry, decompression, orientation and
color rules remain in the same shared path. A new automated contract test raised the complete
Python suite to 50/50; compileall and dependency checks passed.

On the real gallery, default acceptance rose from 23/210 to 137/210. The other 73 files are now
truthfully rejected for exceeding the unchanged 16 MP product ceiling (5712x4284), rather than
being mislabeled as unsupported formats. P2 will report that product boundary and separately score
all stills under a declared 26 MP local-only evaluation ceiling, without changing API policy.

Before the final P2 report, the owner also requested the newly developed model. The rejected E28
Stay-Positive checkpoint was therefore added to the gallery comparison as a diagnostic under its
already-frozen N2 threshold. This does not reopen its failed gate or authorize integration.

### P2 — every current arm was measured on the complete private gallery

The completed run accounted for 210 still-image instances, four of which were exact duplicate
bytes, plus one unsupported MOV. At the unchanged product boundary, 137/210 stills are accepted
and 73 high-resolution 24.47 MP stills are rejected by the 16 MP safety ceiling. A separately
declared 26 MP local-only measurement admitted all 206 unique stills with zero decode or inference
failures. This did not change the API policy. No personal image, filename, EXIF/GPS field or
per-image hash was committed, and the photos were not used for training, calibration or threshold
selection.

The camera-pipeline result is decisive. Canonical E20 falsely triggered on 178/206 authentic
photos (86.4%). The rejected E28 Stay-Positive candidate triggered on 170/206 (82.5%), only eight
fewer, so its earlier source-wise rejection is independently confirmed and it remains outside the
manifest and serving. Legacy ResNet-18 produced 100 AI, 18 uncertain and 88 real labels; full-image
statistics produced 134/40/32; tile statistics and therefore `auto` called all 206 AI. The
external CF-ViT comparison triggered once and abstained as `insufficient` on 205. Its 0.49% false
alarm result is excellent specificity on this real-only slice, but abstention is not proof of
authenticity and this run says nothing about AI recall.

This closes the iPhone compatibility/measurement phase with two separate truths. PixelProof now
decodes the primary frame of ordinary iPhone MPO photographs through its shared bounded input
path, so the demo is materially more testable. The project-owned model genuinely runs end to end,
but its decision is not trustworthy on the owner's camera pipeline. More threshold adjustment on
these private photos would contaminate evaluation; the next justified model line must change the
representation and then face both unseen real pipelines and held-out modern AI.

### P0–P2 development commit ledger

| Phase | Commit | Archived outcome |
|---|---|---|
| P0 | `703908b` | iPhone MPO correction and private-gallery protocol recorded before code |
| P1 | `b4cc339` | bounded primary-frame MPO support, regression test and 50/50 suite completed |
| P2 amendment | `dfe6993` | rejected E28 diagnostic frozen before reading gallery scores |
| P2 result | `48f9696` | complete anonymous model comparison and practical conclusion archived |

The gallery itself remains outside Git. This ledger links the append-only narrative to the
immutable implementation and measurement record needed for the internship report.

### Q0 — a current, sub-100 MB AI-only probe is frozen before download

Internet research selected the MIT-licensed SANEval sample at pinned Hugging Face revision
`e9e188f6018b3d491708f29e7a387f5043dc8841`. Its commercial API outputs include the requested
2025 generation families. The planned subset contains 100 images: 20 each from GPT Image 1,
Imagen 4, Imagen 4 Ultra, Nano Banana and Seedream 3, balanced over five structured prompt types
and simple/hard difficulty. Imagen 3 is excluded as outside the requested date boundary.

Selection is fixed by source row id before any detector score is read. Downloaded image bytes must
remain below 100,000,000, stay in ignored local data and pass count, balance, uniqueness, decode
and revision checks. The row service exposes cached JPEGs although the source card describes raw
PNGs, so the result will explicitly be a web-recompression diagnostic. CF-ViT keeps threshold
0.6617392; this AI-only slice can measure recall but cannot measure false positives, specificity,
accuracy or AUC. No training, tuning or serving change is part of Q0–Q2.

### Q1 implementation — the capped probe is ready before network data enters

The new E29 command turns the pre-registration into an enforceable contract: bounded network
retry, pinned revision and schema checks, deterministic balanced selection, 100,000,000-byte
preflight/stream guard, JPEG/decode/geometry checks, unique file hashes and atomic local writes.
It stores the third-party subset only under ignored `ml/data/` and reuses the existing verified
CF-ViT adapter and frozen threshold rather than adding or tuning a model.

Two new contract tests cover the exact 100-row balance and strict size failure. The complete
Python suite passed 52/52, compileall and dependency checks passed. The internet interruption
occurred before any image download, so no partial dataset existed; the pinned dataset endpoint was
reachable again with the expected revision before this implementation checkpoint was closed.

The first invocation then confirmed that a nominally reachable endpoint can still stall on a row
chunk. It was stopped before preflight and before any image existed. E29 now persists only complete,
revision-matched metadata chunks atomically, refuses signed URLs near expiry, prints each fetch
boundary and reuses fully validated image files after interruption. This converts another network
cut from a restart-from-zero risk into an explicit, auditable resume path; focused tests passed 3/3.

### Q2 — the best real-gallery arm sees only 19 of 100 current AI images

The resumed run completed all six pinned source chunks, preflighted the selected assets at
11,546,660 bytes and downloaded exactly 100 unique 1024x1024 JPEG cache representations. Including
resumable metadata, manifest and scores, the ignored local folder occupies 12,092,513 bytes—far
below the requested 100 MB maximum. All images decoded and all CF-ViT inferences completed on MPS
without failure; no third-party image entered Git.

At the unchanged 0.6617392 decision threshold, CF-ViT detected only 19/100 AI images. GPT Image 1
was 2/20, Imagen 4 and Imagen 4 Ultra were each 4/20, Nano Banana was 4/20 and Seedream 3 was 5/20.
Hard prompts were 7/50 and simple prompts 12/50. This is consistent with the earlier GPT-family
blind spot and demonstrates the central tradeoff directly: the same external arm that falsely
flagged only 1/206 owner camera photos also misses 81/100 images in this current AI-only slice.

The result is intentionally not called accuracy. There is no real class, the sample has only 20
items per generator, and the dataset-server JPEGs are recompressed derivatives of documented raw
PNGs. It nevertheless falsifies the useful claim that CF-ViT's strong owner-gallery specificity
makes it the project's successful universal detector. It remains a valuable external comparison;
the project still needs a representation evaluated jointly on unseen camera pipelines and modern
AI families.

### Q0–Q2 development commit ledger

| Phase | Commit | Archived outcome |
|---|---|---|
| Q0 | `075df5f` | source, balanced row selection, 100 MB ceiling and frozen threshold recorded before download |
| Q1 | `32c770c` | reproducible capped downloader, CF-ViT probe and contract tests added |
| Q1 correction | `d96dde5` | interrupted metadata retrieval converted into revision/expiry-checked resume |
| Q2 | `5c9bd4d` | 100-image result, limitations and compact evidence archived |

The local subset can be reproduced by the E29 command but its image bytes remain ignored. The
ledger and `evidence/e29_cfvit_2025_probe.json` are the presentation-safe scientific record.

### E30/A0–A1 — the dataset idea becomes a role-enforced scientific system

The next step began from the owner's correct diagnosis that the project lacked a trustworthy,
current real-vs-AI test environment. The correction to the proposed “one real dataset plus one AI
dataset” is structural: two collections can be separated perfectly by format, geometry or
compression without learning authenticity, exactly as `archive1` demonstrated at metadata-only
AUC 1.000. E30 therefore records five non-interchangeable roles—TRAIN, CALIBRATION, DEVELOPMENT
TEST, LOCKED FINAL TEST and chronological FUTURE TEST—and makes the role part of each row rather
than a folder naming convention.

Plan commit `c55be75` froze the working-v1 gates, the owner's exposed gallery boundary, the need for
a new native multi-phone vault, the low/full-bandwidth profiles and three pinned public sources
before implementation or download. MLLMGenSet supplies matched GPT Image 2, Nano Banana 2 and real
development cells; LAION-Mobile supplies only web-laundered real-phone regression; Qwen Image
Bench supplies an independent sealed current-generator collection. Existing project training and
E22/E24 calibration data keep their old roles, and FUTURE remains honestly empty until a later
generator release.

A1 turns those words into refusal conditions. A manifest cannot carry an ambiguous label,
unpinned revision, unsafe path, unnamed generator/camera pipeline, derived image in another role,
or exact/underlying content shared across roles. A TRAIN request for test rows raises instead of
loading them. Downloads are deterministic, capped, atomic and resumable; decoded geometry,
SHA-256 and dHash enter ignored local manifests. A metadata-only classifier measures whether
format/shape/resolution/compression can solve the merged classes, and a locked-final receipt cannot
be overwritten after a run.

Twelve focused tests and the complete 65-test Python suite passed; compileall and dependency checks
were clean. This checkpoint still contains zero E30 images and zero E30 model scores. That ordering
is deliberate evidence for the internship report: the decision rules existed before the result.

The first acquisition attempt then exposed a library-boundary bug before downloading a byte:
Hugging Face Hub's installed client is httpx-based, not requests-based, so it rejected the stream
argument at call construction. The correction uses the client's native streamed send path, handles
both iterator APIs, closes responses and leaves the frozen selection unchanged. As with E29's
interruption, the failed attempt is retained because proving that recovery did not reselect data is
part of the scientific record.

### E30/A2 — compact development data is real; an incompatible source is not disguised

The fixed downloader completed the frozen MLLMGenSet development slice: 180 JPEG parents across
all nine cells (120 current AI and 60 matched real) in 4,419,610 bytes. It then generated q90,
q75, q50 and resize256-q90 descendants for every parent without further network use. The 720
derivatives add 14,029,255 bytes, retain the parents' DEVELOPMENT role and content identity, and
produce 900 unique byte hashes overall. All five metadata-only shortcut probes passed the frozen
0.65 AUC ceiling; the worst was q50 at 0.6362. These results make the battery eligible for model
diagnosis, not a universal benchmark and not evidence about native generator files.

LAION-Mobile exposed two separate failure modes. The first run lost connectivity after accepting
one iPhone 11 URL. The preflight was changed to persist a `source_incomplete` manifest with every
pipeline count and rejection category; an automated regression test prevents future silent
rebalancing. The connected retry filled all four Apple cells but only 9/10, 5/10, 1/10 and 0/10
in the frozen Samsung/Xiaomi cells. Of 361 rejections, 287 were over the 375 KB per-file contract,
33 lacked content length, 40 failed HTTP/network checks and one was not an image. The smallest
reachable ten-per-cell combination itself would be about 45.96 MB, so relaxing the cap would break
the pre-registered low-bandwidth budget. No LAION image was downloaded, the 55-row partial result
was not called a dataset, and no easier phone group replaced a failed one.

This is the intended behavior of A2's scientific line: the complete local battery remains
18,448,865 image bytes, while an unsuitable source becomes an archived negative result rather
than an invisible distribution change. `evidence/e30_development_realization.json` preserves the
aggregate counts, hashes and audits needed for the internship report; detailed third-party URLs
and all images remain ignored. A larger full-internet attempt must be a separately pre-registered
version, preferably with a native multi-phone vault, rather than mutating this result.

### E30/A3 seal — the independent frontier scout is fixed before download

The first Qwen Image Bench metadata call tried to paginate large directories and was stopped after
90 silent seconds, before producing a selection or downloading an image. A bounded pinned-revision
tree query replaced it and now fetches only enough metadata to apply the unchanged numeric-first
rule. A regression test fixes the endpoint revision, encoded directory, 20-item page and 15-second
attempt timeout.

This pre-download audit also corrected the project's all-PNG assumption. The selected source rows
are 21 PNG and 19 JPEG, so acquisition now preserves the original mixed encoding under one
`native_source` transport instead of silently excluding four generators' early rows or converting
them. The sealed result is 40 paths, five for each of eight 2026 generator families, totaling
37,907,745 declared bytes under the 70 MB ceiling. Selection SHA is `50e3fec1...eeb`; every exact
path and prompt id is archived in `evidence/e30_qwen_sealed_selection.json` with
`detector_scored=false`.

The small sample is still only a pipeline scout. Its existence does not authorize threshold
tuning or a pass/fail statement, and the locked images remain unread until this seal is committed.

After that seal commit, acquisition completed without interruption: 40/40 files, 37,907,745 exact
bytes, 21 PNG plus 19 JPEG, 40 unique byte hashes and zero decode failure. A network-free q90 JPEG
view was then created for every parent. The 9,449,715 derivative bytes retain LOCKED FINAL role,
prompt identity and parent linkage; all 80 byte hashes remain unique. Native and combined
content-set hashes are archived in `evidence/e30_qwen_realization.json`. No detector score was
computed, so the locked-final boundary remains intact at the end of A3.

### E30/A4 implementation — scoring rules exist before development output

The next tool can read only the frozen 900-row MLLM DEVELOPMENT manifest. It verifies the complete
content-set hash and each individual byte hash, binds resumable score rows to a hash of the model
contract, and rejects a changed E20 checkpoint or CF-ViT weight file. E20 retains its checkpoint
threshold and binary semantics; CF-ViT retains its E24 threshold and honest `insufficient` band.
Reports include exact 95% binomial intervals, macro/worst groups and transport deltas. Two focused
tests passed before inference, and the Qwen LOCKED FINAL path is absent from the evaluator. Thus
the implementation commit remains a genuine pre-result checkpoint rather than documentation
written after seeing performance.

### E30/A4 result — high specificity is not a working detector

Both existing arms completed all 900 DEVELOPMENT inferences without failure, but neither earned
access to the locked Qwen set. On the independent standardized view, project-owned E20 falsely
triggered on 8/60 real images (13.3%) and detected only 15/120 current AI images (12.5%). Across all
five correlated transports its descriptive FP/recall was 9.33%/7.67% and AUC 0.463. Its apparent
signal came mainly from texture: AI texture recall was 20%, but matched-real texture false alarms
were 24%, the same shortcut pattern the project is trying to escape.

CF-ViT made zero false alarms on 60 standardized real images but detected only 2/120 AI (1.67%);
across all views it triggered on 6/600 AI and abstained on 99.33% of inputs. Nano Banana 2 recall
was exactly zero. This strengthens the earlier E29 conclusion: CF-ViT is the safest real-gallery
comparison, not a usable current-generator detector. E20 is neither selective enough nor sensitive
enough on this battery.

The report does not pretend that 900 derived views are 900 independent photographs. Pooled rates
are marked descriptive and exact intervals are limited to one transport or one source cell. The
pre-registered cells contain 20 items rather than the formal 40 minimum, but aggregate recall misses
the 50% target so widely—and worst AI point recall is 0%—that consuming the final set would add no
scientific value. `evidence/e30_development_benchmark.json` records the contracts and outcome;
Qwen remains sealed and unscored. Work must return to representation/training, not threshold tuning.

### E31/B0 — the attached disk changes the options, not the scientific rules

The LaCie disk was attached after the existing E30 arms failed on current-generator DEVELOPMENT.
The proposed recovery had three parts: inspect the large local datasets, retrain if old data had
poisoned the models, and connect several models behind one decision. The proposal was evaluated
against the archive before implementation. E20 did **not** use the old inverted-label pool: it was
trained after E19's correction on 48,037 native tiles, and its three seeds reproduced the same
source-shift failure. E28 had also shown that replacing only its head was insufficient. Repeating
the identical E20 run would therefore create another checkpoint, not new evidence.

The ensemble idea was kept but narrowed. E9 had already tested eight fixed blends and gained at
most 0.002 AUC. E20 and CF-ViT make disjoint positive decisions on E30, but their simple OR catches
only 52/600 correlated AI views and falsely triggers on 28/300 real views. This proves only that the
current two-arm union is unusable; it does not rule out a later ensemble of genuinely heterogeneous
and independently useful representations. `PLAN.md` now freezes E31 in the order SSD audit →
source-aware TRAIN v2 → representation ladder → out-of-fold calibrated fusion → frozen E30 gate.
The plan was committed first as `a929901`, before an E31 model score, checkpoint or ensemble fit.

### E31/B1 checkpoint — 270.91 GB inventoried without writing to the source disk

`e31_ssd_audit.py` was added as a reusable read-only audit instead of treating one mount path as
project configuration. It requires `--root`, refuses to write its report under that root, ignores
exFAT AppleDouble/cache artifacts, verifies known upstream label orders and reports aggregate data
only. Ten registered sources occupy 173,576,436,217 bytes and seven inventory-only sources another
97,337,151,271 bytes. Complete registered Parquet metadata covers 603,991 rows.

The old CommunityForensics estimate was wrong in a useful direction: the 44,884 local rows contain
11,972 AI, 32,912 real and **300** distinct AI model names—not 228 and not merely a few generators
from an ordered partial shard set. Yet the pixels cannot be used natively without repeating the
project's oldest mistake. Native metadata alone separates CommunityForensics at AUC 1.000, AIGC at
0.967 and ai-vs-real-200k at 0.841. After an identical fixed 128 RGB/JPEG probe those values fall to
0.636, 0.540 and 0.552. AI-vs-Real-balanced reaches native AUC 0.549 but still has different format
sets; its fixed probe is 0.586. The Julien Lucas modern test set, previously described as the
cleanest modern set, shows native AUC 0.974 in the deterministic shard-spread sample and must not
support a naive pooled native claim; its fixed probe is 0.560.

The final bounded run decoded 3,000/3,000 images and found no sampled exact match against all 980
E30 parent and derived protected hashes. This is deliberately recorded as **sampled evidence**, not
full decontamination. B2 must choose exact TRAIN-v2 rows and hash every one against calibration,
owner gallery, E30 DEVELOPMENT/LOCKED and named test-only data before training. The compact evidence
is `evidence/e31_ssd_audit.json` with SHA-256
`2f7399bed965a8a428b4180aab059405fbcc4d4aa4d3754a5295ee4e97021f29`.

Two local audit attempts were stopped rather than hidden. The first allocated 600 samples across
every shard; Parquet then decompressed enormous embedded-image row groups for even one requested
row. Limiting the range to 12 shards was still wasteful after a single CommunityForensics shard was
measured at 4.09 GB. The final tested rule selects the lexical first, middle and last shard and 300
rows per registered source, while full row/generator counts still read all metadata. Six focused
tests pin root safety, AppleDouble exclusion, label behavior, bounded shard spread, implicit folder
labels and inclusion of derived E30 manifests. Internet was not required, no external-disk byte was
changed, no E31 training began and the Qwen LOCKED FINAL scout remains unscored.

### E31/B2 selection — breadth is frozen before opening 11,300 images

The first TRAIN-v2 contract selects 5,650 AI and 5,650 real parents rather than consuming every
available row. CommunityForensics contributes eight rows from each of its 300 AI model identities
plus 2,400 real rows. AI-vs-Real-balanced contributes 2,000 AI and 3,250 real; the extra real rows
provide the matched counterweight for 500 Flux, 500 Nano Banana and 250 Nano Banana Pro rows. AIGC
and ai-vs-real-200k remain deferred: their fixed-view metadata probes passed, but adding another
84 GB before measuring the smaller candidate would test volume instead of the hypothesis.

Groups—not individual rows—receive one of five folds. Fold zero is CALIBRATION and the other four
are TRAIN. An initial metadata-only dry assignment used hash modulo; by chance none of Flux's seven
shards entered CALIBRATION. Because no image byte or score had been opened and nothing was committed,
the rule was corrected to stable within-source rank plus round-robin folds. The frozen result has
8,561 TRAIN and 2,739 CALIBRATION rows; all five collections support both roles and none of the 383
generator/shard groups crosses roles.

The exact source shard and row ids, maps, source fingerprints and counts are committed in
`evidence/e31_train_v2_selection.json`. Selection SHA is
`5907c14ba3e173c125c024a30658fb8e7e56788a469614808ad4ef5519a5fbfb`; evidence-file SHA is
`59f95563da578c8274518ae0394b00064bd1b0109ad652077a68ad3967ff5620`. This checkpoint exists
before realization by design. Next, every frozen row must reproduce the source fingerprints,
decode, avoid exact/dHash overlap with every protected role and yield one deterministic native
128 px tile. No E31 model or embedding has read these rows yet.

Before realization, one boundary was strengthened: protected loose directories and E30 manifests
were already supported, but named test-only images stored inside Parquet were not. The reader now
streams those embedded bytes too, so all 12,695 Julien Lucas rows and the separate 8,000-real /
8,000-fake CommunityForensics probes can reject exact or dHash overlap. This amendment happened
before the frozen selection opened an image and does not change its SHA.

The first full realization then stopped for a different reason. All accessed rows decoded, but
3,534 of 11,300 could not produce the required native 128 px texture-qualified tile. The program
did not write a partial archive or let those losses silently rebalance sources. This is the E19
small-image floor returning in the much larger balanced corpus. The rejected selection remains
archived. A new eligibility command is committed before use: it may inspect only decode success,
width/height and the unchanged 0.04 texture rule across the balanced source, freezes eligible row
keys by SHA and cannot read a model score. Only after that mechanical set exists may selection v2
be frozen.

The full balanced scan found the hidden distribution precisely: 47,233 AI and 50,000 real rows are
smaller than 128 px; 24,301 AI and 21,532 real rows are eligible, while only one AI and three real
rows fail solely for texture. The eligible key set is archived with SHA `91089e22...eb2`. Selection
v2 preserves the 11,300 total, every source cap, label balance and group role; 7,767 rows stay and
3,533 balanced rows are replaced by their next deterministic eligible candidates. Its frozen SHA
is `5355e430...9b2`. Because v1 rejected 3,534 total, the remaining possible one-row failure outside
balanced is not inferred; v2 realization must expose it mechanically. Both eligibility and exact
v2 row ids are committed before that run.

One operational lesson from v1 was also kept: a long rejected run needs a receipt, not just a
traceback. Before v2 reads bytes, realization now persists only rejected record ids/reasons and
aggregate protected-scope counts, returns a non-zero status and still refuses to write tiles. This
changes observability, not the frozen selection or acceptance rule.

Selection v2 then completed the expensive scan and was correctly rejected again. It produced
11,299 candidate tiles but found one too-flat Nano Banana Pro row, 74 exact protected-test matches
and 83 dHash matches; 74 rows were in both overlap sets, leaving nine re-encoded/content matches.
The exact collisions were 8 balanced AI, 10 Flux, 10 Nano Banana and 46 Nano Banana Pro rows—mostly
the predictable consequence of E25 having already sampled those modern collections. Nothing was
written to the tile archive. The ignored rejection receipt has SHA `55364ed2...b9c8`.

Replacing only those selected ids would turn leakage discovery into an iterative lottery. The next
screen is therefore committed before use and processes every candidate row in balanced, Flux,
Nano Banana and Nano Banana Pro against the same complete protected exact/dHash set plus the input
floor. It records eligible keys, not protected hashes or model scores. Selection v3 may use only
that set. CommunityForensics remains unchanged because its selected v2 rows had zero overlap and
zero input failure.

### E31/B2 checkpoint — the protected pool is screened and selection v3 is frozen

After the office restart, the LaCie scan resumed locally without downloading anything. It inspected
all 163,777 candidate rows from balanced, Flux, Nano Banana and Nano Banana Pro. The safe pool has
65,650 rows; 97,982 exact protected matches, 137 additional dHash-only matches and six flat rows
were excluded before selection. The unusually large balanced exact count is explained by the
120,000-image historical project archive being protected: old training content cannot quietly
return as new E31 evidence. The committed aggregate is `evidence/e31_protected_screen.json` with
SHA `e1a3f811...122ff`; detailed eligible keys remain ignored with SHA `16ff5f14...bad10`.

The deterministic v3 freeze preserves all scientific controls: 11,300 balanced parents, 383
indivisible groups, 303 AI identities, and unchanged TRAIN/CALIBRATION/source counts. Exactly the
84 v2 rejects are replaced source-for-source while 11,216 rows stay fixed; CommunityForensics is
unchanged. Selection SHA is `1a3a5c98...df2e`, evidence-file SHA `c6748b12...d98c`. This checkpoint
is committed before the independent realization opens v3 bytes. It does not yet claim a usable
training archive or model result.

### E31/B2 accepted — the first decontaminated source-aware tile archive exists

The independent realization then opened the frozen v3 rows and passed every hard gate. It created
11,300/11,300 native 128 px tiles: 5,650 AI / 5,650 real, split into 8,561 TRAIN and 2,739
CALIBRATION exactly as pre-registered. There were zero decode failures, zero small/flat losses,
zero exact protected overlaps and zero dHash protected overlaps. Every tile hash is unique.

The image-bearing NPZ remains ignored local data (395,082,960 bytes, SHA `508330c2...9f2b`); its
compact committed receipt is `evidence/e31_train_v2_realization_v3.json` (SHA
`5bff123c...619d`). This closes B2 without touching the E30/Qwen final test. It is a useful project
output by itself: unlike earlier corpora,
the next model will train on a balanced, source-capped, generator-broad, group-disjoint and fully
decontaminated contract. Whether its representation transfers remains the B3 question, not an
assumption.

### E31/B3 pre-score checkpoint — the representation ladder is executable

`ml/experiments/e31_representation_ladder.py` now pins the accepted tile/selection SHA values and
implements the planned R0 E20, R1 frozen DINOv2 and R2 68-feature arms. R1/R2 use group-disjoint
TRAIN out-of-fold scores for threshold selection, then fit all TRAIN and open CALIBRATION once.
The real operating point requires <=5% source-macro and <=10% worst-source FP; transfer requires
>=50% macro recall and >=30% weakest-source recall across Flux, Nano Banana and Nano Banana Pro.
Three focused tests pin source-aware threshold ties, zero-FP behavior and the acceptance gate.
This implementation and protocol are committed before extracting a B3 feature or producing a
model score; E30 remains unopened.

### E31/B3 result — frozen DINOv2 is the first new arm to beat E20 cleanly

On untouched E31 CALIBRATION, the old E20 control already performs well: 0.960 AUC, 4.49% source-
macro / 6.70% worst-source real FP and 84.49% macro recall across Flux, Nano Banana and Nano Banana
Pro. The frozen DINOv2 linear probe improves this to 0.966 AUC and **90.72% current-AI macro recall**
while holding 4.67% / 6.70% real FP; its weakest current source is Nano Banana Pro at 84%. The
68-feature specialist holds slightly lower FP (4.24% / 5.51%) but only 56.24% current recall.

Seeds 42, 2024 and 2026 reproduce the same DINOv2 and feature metrics—the expected outcome for the
converged convex linear head. No backbone was fine-tuned and E30 was not opened. DINOv2 therefore
passes B3 independently and is the leading project candidate. The forensic arm is not promoted by
its standalone floor alone; B4 must prove complementary row-level errors before any fusion. Compact
evidence is `evidence/e31_b3_representation_screen.json`; the ignored feature cache SHA is
`f59e1fb6...c4c49`.

### E31/B4 pre-score checkpoint — fusion can win only on paired evidence

The B4 runner is committed before a fusion score. It assigns source-stratified CALIBRATION groups
to five meta-folds, cross-fits both stacking coefficients and real-FP thresholds, and compares only
DINO alone, DINO+E20 max/stack and DINO+68-feature max/stack. A fused rule needs at least +5 points
of current-source macro recall, both real-FP budgets, and a positive paired group-bootstrap 95%
lower bound. Otherwise DINO remains the single winner. Three focused tests pin group containment,
source-macro metrics and paired bootstrap behavior. No E30 row is opened.

### E31/B4 result — the ensemble idea helps, but not enough to earn deployment

The best fusion, DINO+E20 max, raises current-source macro recall from 90.72% to 93.77%; its paired
group-bootstrap interval is positive (+1.87 to +4.20 points), so the complementarity is real. It
still fails both decisive requirements: the gain is below +5 points and macro real FP rises to
5.34%. DINO+68-feature max gains only +1.86 points at 5.08% macro FP; stacking is weaker. In raw
terms E20 catches 12 of DINO's 24 current-AI misses but creates 50 additional real false positives;
R2 catches eight and creates 42.

The honest outcome is therefore **ensemble rejected, single DINOv2 selected**. The candidate
artifact embeds its fitted head plus the DINO model/input/weight contract, uses full-CALIBRATION
threshold `0.7090073824`, and has SHA `99901219...4d860`. Compact B4 evidence is
`evidence/e31_b4_ensemble.json`; row scores remain ignored with SHA `04f5dc59...d5bc5`. This is the
first candidate allowed to approach B5. E30 has still not been read.

### E31/B5 pre-score checkpoint — candidate, crop and final lock are frozen

The B5 scorer verifies candidate SHA `99901219...4d860` and DINO weight SHA `04d27f34...0081` before
loading. It uses exactly one native 128 px texture-qualified tile, chosen from a parent `content_id`
so transport derivatives share the selection key, then the fixed 224 px DINO view and threshold
`0.7090073824`. This preserves the one-tile training/calibration distribution instead of inventing
an uncalibrated top-k rule. A synthetic-noise smoke image scored successfully; no E30 image was
opened.

The existing 900-row MLLM DEVELOPMENT manifest remains pinned to content set `7634755c...24b8` and
the old scorer now exposes the E31 arm without altering E20/CF caches. The Qwen one-shot runner is
also committed but requires a committed `development_passed` receipt for this candidate before it
will open 40 native + 40 standardized LOCKED rows. Focused tests pin crop reproducibility, role
gate rejection, generator/transport reporting and every DEVELOPMENT stop/go condition.

### E31/B5 result — the new model runs, but independent real photos falsify it

The frozen candidate processed 897/900 DEVELOPMENT views; three resize derivatives could not meet
the tile texture floor. It retained strong AI sensitivity—80.67% macro recall, about 80.6% on both
GPT Image 2 and Nano Banana 2—and was stable under JPEG/resize. Yet it called **83.63% of real
groups AI**, reached 100% FP on the worst real group and ranked the task backwards at AUC 0.385.
Even standardized JPEG real FP is 81.67%; the problem is not just the resize view.

A diagnostic-only threshold search makes the failure sharper: satisfying <=5% macro / <=10% worst
real FP requires threshold 0.999986 and leaves 0.33% macro AI recall with a 0% worst AI group. No
threshold or retry was adopted. The scientific conclusion is that E31 fixed the data contract and
found a strong in-contract DINO representation, but the real class still does not span independent
MLLM/web photographic pipelines. The model is runnable, not deployable.

Because DEVELOPMENT failed, `evidence/e31_b5_development.json` records `development_failed` and the
Qwen runner cannot pass its gate. The 40 native + 40 standardized LOCKED rows remain unscored. B6
must not replace the current serving model with E31; the next research step is real-pipeline data
breadth or a camera-trace representation, followed by a new pre-registered candidate—not tuning on
this DEVELOPMENT result.

### E31/B6 closure — runnable research scorer preserved, serving remains unchanged

Because B5 failed, B6 did not touch FastAPI, the web UI or the verified E20 runtime. A separate
`e31_score_folder.py` tool preserves the exact DINO candidate for labelled research: immutable
artifact/weight hashes, one content-keyed tile, fixed threshold, asymmetric verdict and a mandatory
83.63% DEVELOPMENT FP warning in every JSON. It refuses to overwrite an earlier evaluation.

The end-to-end CPU smoke used scikit-learn's bundled real `china.jpg`; the candidate ran without
error and scored it 0.999826 above threshold 0.709007—another false AI signal consistent with B5,
not a success example. The output correctly declared `rejected_for_serving_after_E30_DEVELOPMENT`
and never claimed authenticity. Model card, README, serving boundary, dataset ledger, experiment
log, presentation evidence and internship report now tell the same story. E31 ends with a working
experimental model and a successful safety gate, not a deployable replacement.

### E32/C0 — in-the-wild rebuild is frozen before acquisition

E31's failure was converted into a new data-and-representation experiment before a new image byte
or model score existed. The attached LaCie disk was inspected read-only and has about 651 GiB free.
Its existing holdings already include 10,000 FLUX.1-dev, 9,457 Nano Banana, 1,250 registered Nano
Banana Pro and about 2,122 physical GPT-image files, so E32 will audit and cap these sources rather
than redownload volume blindly. No E32 dataset directory, API-generated image, manifest, embedding
or checkpoint exists at this checkpoint.

The authentic pool now targets 10,000–20,000 parents from at least three independent collections.
VISION, Forchheim FODB and a selective CSAFE subset are the preferred candidates because they
provide explicit device/scene structure; modern web/professional photos fill semantic rather than
camera-count coverage. SOCRatES remains conditional on its signed agreement. ForensiCam-215K is not
admitted while its Baidu-only access and dataset licence remain unresolved. Every realized source,
revision, byte count, term and limitation must enter `DATASETS.md`; candidate counts in the plan
are availability evidence, not a claim that data was downloaded.

The model direction is deliberately not “train DINO again.” E32 first compares the rejected E31
DINO-S contract with a feasible frozen PE-Core linear probe motivated by 2026 SSAFE, a DINOv2-L
intermediate-block/global-plus-texture candidate motivated by ITW-SM/RINE, and EfficientNet-B4 as
the narrow public Hive architecture ablation. All see the same frozen data roles and input-view
ablation. Fine-tuning and ensemble work are earned only by transferable single-arm evidence.

The Champions League test remains a battery, not one pooled accuracy: historical E30 and owner
gallery DEVELOPMENT, new API-current AI, unseen-camera/web authentic content, untouched ITW-SM and
conditional Qwen locked arms. Existing gallery content cannot become a fresh final merely by being
renamed; only never-scored additions may be sealed. `PLAN.md` records exact role, label, leakage,
metric and stop/go rules. This C0/history checkpoint must be committed before C1 acquisition.

### E32/C1a — authentic sources frozen before the first image transfer

The source registry and tested acquisition runner froze three complementary authentic collections.
VISION contributes exactly 3,500 `/images/nat/` JPEG parents over 35 devices; the selection rejects
flat fields, videos and all social encodes mechanically. FODB contributes three fixed archives
totalling 22,940,347,533 declared bytes and an expected 3,851 `orig` parents; its five social
copies will inherit scene/device parent identity. CSAFE is bounded to the 17,588,803,163-byte
Galaxy S21 archive with published MD5 `5c5f79e3e508a5cbf7a19e75846091d8`, rather than mirroring the
132.7 GB collection.

The detailed external receipt is 1,166,007 bytes with SHA-256
`200a7aeb23d9c303d880dff76a08b21e38efe666531a6552ffe4bdd5841eca4d`; compact evidence is
`evidence/e32_real_acquisition_selection.json`. Freeze created no image, archive, embedding,
checkpoint or score. Eight focused tests pin registry uniqueness, native-only VISION filtering,
path containment, completed-file reuse and receipt-state rejection.

One operational fault was caught before bulk transfer: Python Requests rejects FODB's incomplete
certificate chain while macOS system `curl` verifies it successfully. The implementation now uses
TLS-verifying `/usr/bin/curl` for metadata and image bytes, never `verify=False`; `.partial` resume,
retry, declared archive sizes and a 100 GiB free-space floor remain mandatory. SOCRatES and
ForensiCam decisions are unchanged. C1a freezes availability, not eligibility: no source becomes
TRAIN/CALIBRATION until full decode, grouping, duplicate and shortcut audits pass.

The first VISION transfer smoke exposed a terminal-operability problem rather than a data failure:
four concurrent curl progress meters interleaved into a large unreadable PTY stream. The run was
stopped explicitly after 95 completed JPEGs plus resumable partials; no completed file was removed.
The downloader now uses curl's quiet-success/error-visible mode and retains one aggregate message
per 100 completions. A focused test pins the quiet TLS curl and atomic `.partial` destination.
Successful VISION destinations have no published per-file size but exist only after curl exits
cleanly and atomic rename succeeds; they are now reused directly, while only `.partial` files are
resumed. This prevents a reconnect from redownloading every completed native JPEG.

### E32/C2a — volume is not provenance: only three modern AI families are admitted

A complete physical metadata inventory corrected the earlier rough file counts. The partial GPT
Image 1 checkout contains 1,060 PNG images and 1,061 text files—2,122 physical files did not mean
2,122 images. The Nano-Banana-150k archive contains 127,835 image members, not the card's claimed
more-than-150k quantity. These corrections happened before C2 row selection or training.

Only three present families currently have both an explicit generator statement and dataset
licence: 9,457-row Nano Banana (MIT), 200-image Nano Banana Pro (MIT), and the 1,060-image partial
GPT Image 1 snapshot (CC BY 4.0). The 10,000-row FLUX.1-dev and 1,250-row second NBP holdings have
prompts but their cards expose no dataset licence and insufficient generation narrative. The
127,835-member Nano editing archive also lacks a dataset licence and manifest. They remain on disk
but do not count toward the five-family gate.

CommunityForensics remains a licensed, high-breadth diversity anchor under its already audited
fixed-input conditions; it is not relabelled as a new current family. AIGC Benchmark, Julien,
MLLMGenSet and Qwen retain their protected test roles. Compact evidence
`evidence/e32_ai_inventory.json` records every revision, physical byte/count and decision. The next
C2 action is to find at least two licensed, explicitly sourced modern families—not to download
more volume from ambiguous holdings.

### E32/C2b — the two-family gap is filled on paper, with a decoder gate before bytes

Source research selected Qwen Image 2512 and FLUX.2 Klein 9B Base instead of weakening C2a's
licence/provenance rules. Their pinned cards declare CC BY-SA 4.0 and CC BY 4.0 respectively and
describe the exact generation settings. Qwen exposes 3,936 outputs across 984 four-variation
prompt groups; FLUX exposes 4,072 generated outputs across 1,018 groups. FLUX's 160 reference
inputs are explicitly non-generated and excluded.

For each source, deterministic category round-robin froze 750 whole prompt groups / 3,000 images.
Qwen contributes 7,108,445,821 selected image bytes and FLUX 4,400,537,141. All four variants of a
prompt remain indivisible for future splitting. Detailed receipt SHA is
`b871004f381a16e54d30122004a5ffc81b0cdb4811cbad755e6a6531dc068ecc`; no image byte was fetched
by freeze. These are separate training candidates and do not open or relabel Qwen LOCKED FINAL.

Both sources use JPEG XL, which the current Pillow build does not register. macOS `sips` advertises
native JPEG XL support, but availability is not assumed to equal a successful model decode. The
bulk 11.51 GB selection is blocked until exactly one frozen JXL per source downloads and decodes.
Five focused tests pin complete four-image grouping, balanced category selection, reference
rejection and destination containment.

### E32/C2b decoder smoke — paths say JXL, bytes say PNG

The authorized smoke downloaded one selected image plus its prompt sidecar from each gap source:
2,579,073-byte Qwen and 1,215,314-byte FLUX images. Both load directly with Pillow. Qwen is RGB
1328x1328 and FLUX RGB 1024x1024, matching the cards' dimensions. However `file`, `sips` and Pillow
all identify the payloads as PNG despite `.jxl` suffixes. No JPEG XL plugin or bulk conversion is
needed; actual decoded format must override path metadata.

Expected dimensions were added to the receipt schema without changing selected groups/assets,
superseding detailed receipt SHA `b871004f...8ecc` with `e9c3d3da...af7a` (2,349,078 bytes). The
smoke evidence stores exact image hashes and the new selection SHA. Bulk code now refuses to run
unless `evidence/e32_ai_gap_decoder_smoke.json` says `decoder_smoke_passed` for that exact receipt.
Six focused tests pass, including missing/stale gate rejection. The format mismatch is a new
shortcut warning: E32 must infer format from bytes and normalize REAL/AI identically.

### E32/C1-C2 realization gate — downloaded bytes still have no training role

The completion gate was implemented while the frozen VISION, Qwen and FLUX transfers continued,
so the arrival of files cannot silently become a TRAIN manifest. `e32_source_realization.py`
binds each source report to its exact frozen selection SHA, requires every expected file to exist
without a `.partial` sibling, decodes content by its byte signature, and records SHA-256, dHash,
format, mode, dimensions, EXIF/orientation and bytes-per-pixel. It compares candidates with all E30
manifest hashes and every previously passed E32 source audit. Detailed per-row reports stay on the
LaCie E32 audit root; Git receives only compact, hash-bound evidence after a real source run.

The AI contract adds an indivisible prompt-group check: four selected images, four declared-byte
sidecars, valid non-empty UTF-8, one identical prompt across the four variations, and the frozen
source dimensions. A live four-sidecar Qwen check reproduced the same prompt hash on all variants.
The gate also preserves the discovered extension mismatch as measurement—`.jxl` may decode as
PNG—but never treats an extension as truth or a class feature. Eighteen focused C1/C2 tests pass,
including byte-format sniffing, missing-member rejection and REAL device/EXIF accounting. No
production E32 source has passed this gate yet, no role was assigned and no model was trained.

### E32/C2c — the 15K AI allocation is fixed before remaining selection

A column-only read of the already inventoried local candidates established the available structure
without decoding or scoring images. Nano Banana has 9,457/9,457 unique ids and uniform declared
PNG RGB 1024x1024 metadata. CommunityForensics has 11,972 raw-label-1 AI rows, 32,912 real rows and
300 AI model identities; only its AI rows are candidates for this C2 allocation. The licensed NBP
holding has 200 PNGs. GPT Image 1 remains a partial 1,060-image/1,060-matching-prompt local snapshot
of the pinned 4,000-image source. An initial ad-hoc scan hit an exFAT `._*.parquet` AppleDouble
stub; the already documented filter was reapplied and the 31 Nano plus 15 Community real Parquets
then reproduced the counts above. No source data were damaged or changed.

Before exact row selection or the missing GPT transfer, the nominal pool is fixed at 15,000:
3,000 each from Qwen Image 2512, FLUX.2 Klein 9B, Nano Banana and GPT Image 1; all 200 licensed Nano
Banana Pro images; and 2,800 CommunityForensics AI rows selected across its model identities. The
largest source share is exactly 20%, not above it. CommunityForensics is a diversity anchor and
does not count toward the five-current-family claim. If GPT's remaining 1,940 selected pairs cannot
be obtained from the pinned source, the run stops for a documented replacement rather than
silently oversampling another source or opening a protected final arm.

The selection engine was then implemented without fabricating the unavailable GPT listing. On
real local metadata it deterministically reproduced 3,000/9,457 Nano rows (source fingerprint
`65dfa0a3...ee60f`), all 200 NBP files (`fdbe22b1...6c58`), and 2,800/11,972 Community AI rows
across all 300 model identities with a maximum of ten rows per model (`375b8b66...e8055`). The two
already-frozen gap receipts each contribute exactly 3,000 rows. Stable hashing makes Nano/GPT
selection independent of filesystem and partial-download order; Community round-robin prevents
large model identities from dominating.

The full receipt remains unwritten by design. A pinned GPT repository-info request was reset while
the three authorized transfers saturated the mobile link. Rather than infer filenames or let the
local first 1,060 pairs dictate selection, the command requires the exact upstream revision,
CC-BY-4.0 tag and 4,000 complete PNG/TXT pairs before selecting 3,000. Twenty-six E32 tests pass,
including input-order invariance, model-spread selection and GPT local-availability independence.

### E32 acquisition pause — mobile link stopped without discarding bytes

The user reported that internet access was ending, so all three active transfers were stopped
before starting another source. Thread-pool shutdown required three interrupt signals for VISION
and Qwen; FLUX stopped on the first. The tracebacks were operator interrupts during executor
shutdown, not decode or data-integrity failures. A process audit afterwards found no remaining E32
Python downloader or curl process.

The preserved checkpoint contains 1,715 completed VISION files / 5,096,745,086 bytes, 1,891
completed Qwen image-or-prompt assets / 2,367,125,910 bytes, and 2,096 completed FLUX
image-or-prompt assets / 1,577,354,843 bytes. Total completed payload is 9,041,225,839 bytes. Eleven
VISION, five Qwen and two FLUX `.partial` files remain intentionally in place. No file was deleted,
no source was claimed complete, no realization audit was run and no model role or score changed;
the existing commands will reuse completed files and resume these partials when connectivity
returns.

### E32/C2c exact freeze — 15,000 AI parents are now immutable

After connectivity returned, the pinned GPT metadata gate reproduced revision
`bba366cb...4825`, the `license:cc-by-4.0` tag, and exactly 4,000 PNG plus 4,000 TXT files. The
metadata-only selector then froze all six source allocations into a 4,752,567-byte external
receipt with SHA-256 `3230f026...80b7`; normalized record selection SHA is `2a31e792...0ef7` and
the inherited Qwen/FLUX gap-receipt SHA remains `e9c3d3da...af7a`. Freeze downloaded zero image
bytes and opened no image or model score.

The exact GPT result corrected the earlier arithmetic bandwidth estimate without rewriting that
history: stable selection from all 4,000 upstream pairs overlaps only 795 of the 1,060 locally
present pairs, so 2,205 selected pairs—not 1,940—require download. This is the intended evidence
that local availability did not choose content. Counts remain Qwen 3,000, FLUX.2 3,000, Nano
Banana 3,000, GPT Image 1 3,000, NBP 200 and CommunityForensics AI 2,800; five current families,
15,000 total parents and no source above 20%. No row has yet earned TRAIN/CALIBRATION.

Before fetching a missing GPT image, a receipt-bound acquisition runner was added. It recomputes
the 15K record-selection SHA, reuses any exact-size selected pair already present in the pinned
local checkout or E32 download root, and writes only missing assets through the existing TLS,
retry, free-space and `.partial` machinery. One deterministic selected missing image/prompt pair
must decode with a non-empty UTF-8 prompt before bulk is enabled; stale or missing smoke evidence
hard-fails. Eleven focused GPT/pool tests plus the existing E32 suite pass. No GPT image byte was
downloaded by this implementation checkpoint.

The authorized GPT smoke then fetched selected missing pair `GPTIMG_852`. Its image is a
3,486,339-byte RGB PNG, SHA-256 `8f30398f...6e96`, decoding at 1024x1536; its 1,341-byte UTF-8
prompt is non-empty with normalized SHA `e4f291e3...09c28`. Evidence binds both to 15K selection
SHA `2a31e792...0ef7`. The decoder/prompt gate passes and bulk may start; this does not make the
source eligible or alter the 3,000 selected GPT parents.

The role-free realization gate was extended to the remaining 15K sources before their production
rows were opened. Nano Banana and CommunityForensics now verify the frozen Parquet-file
fingerprints, seek only exact selected row indices and decode the embedded `image`/`image_data`
bytes. The licensed NBP arm verifies all selected loose-file byte counts. GPT resolves every exact
selected image/prompt pair from either the untouched local checkout or the isolated E32 root and
rejects partials, size changes, invalid UTF-8 or empty prompts. All four then use the same
SHA-256/dHash, protected-E30, passed-peer, format/geometry and duplicate gates as VISION/Qwen/FLUX.
Two additional byte-level fixtures bring the E32 suite to 34 passing tests. This checkpoint added
capability only; no production pool source was opened or accepted before commit.

The first production local-pool realization then passed. All 200/200 licensed Nano Banana Pro
images decoded as PNG: 136 RGB and 64 RGBA, with no EXIF. All 200 SHA-256 and all 200 dHash values
are unique; there is no exact or dHash overlap with the 980 exact / 382 perceptual hashes across
four protected E30 manifests, and no passed E32 peer existed yet. The 91,762-byte detailed external
receipt SHA is `bfc217f0...d17b`; compact evidence is
`evidence/e32_nano-banana-pro-ash-local_realization.json`. State is explicitly
`source_realization_passed_candidate_only`: the source is usable for later pool construction but
still has no TRAIN/CALIBRATION role.

The first 3,000-row Nano Banana realization attempt decoded its selected Parquet images but stopped
before writing evidence when the peer-audit directory scan tried to parse exFAT's binary
`._nano-banana-pro-ash-local.json` AppleDouble sidecar as UTF-8 JSON. This is an audit-tool path
filter defect, not a source decode failure. The generic peer scanner now excludes `._*` receipts
and tolerates undecodable/non-JSON filesystem debris; a regression fixture recreates the binary
sidecar. The failed attempt claims no Nano result and will be rerun independently after this fix.

The unchanged Nano rerun then produced a real methodological finding rather than a corrupt row:
3,000/3,000 images decoded and all SHA-256 values were unique, but five unrelated images shared
dHash `0f0f0f0f0f0f0f0f`. A bounded diagnostic showed an enchanted forest, flamingo lake, urban
portal, fireworks and cave angel—same dark-edge/bright-centre layout, different content. Their
independent 64-bit DCT pHashes are 24–32 Hamming bits apart. Treating exact dHash as a duplicate was
therefore a false positive; the source correctly remained rejected under the old gate and no role
was assigned.

Realization schema v2 is frozen before another rerun: SHA-256 exact matches remain definitive;
dHash is only a cheap candidate bucket, and a candidate pair must also have DCT-pHash Hamming
distance <=5 to become a confirmed perceptual duplicate. New E32 peer receipts use the same pair.
Legacy protected E30 manifests contain only dHash, so those hits remain conservative hard
exclusions rather than weakening final-role isolation. This changes no selected row and uses no
model label/score. A regression test proves both a far-pHash collision and a close-pHash match; 36
E32 tests pass.

The independent schema-v2 rerun passed the unchanged Nano selection. All 3,000/3,000 images decode
as RGB PNG, with 3,000 unique SHA-256 and 3,000 unique pHash values. There is no exact or confirmed
perceptual duplicate and no exact/dHash overlap with protected E30 or the passed NBP peer. The
five-row equal-dHash bucket remains visible as one candidate collision rather than disappearing
from evidence. Detailed external receipt is 1,767,170 bytes with SHA-256 `8cb04e52...fe2f`;
compact evidence is `evidence/e32_nano-banana-local_realization.json`. Nano is now a role-free
candidate, not TRAIN/CALIBRATION.

Qwen's first complete source realization was then rejected for genuine duplicate content. All
3,000/3,000 selected outputs and 750 prompt groups decoded as RGB PNG, with zero protected-E30 or
passed-E32-peer overlap. However, `composition_00038` duplicates all four
`architecture_00058` variants byte-for-byte and `composition_00039` duplicates all four
`architecture_00059` variants. Separately, variants 1 and 2 of `style_00053` are confirmed
perceptual duplicates. Totals are 2,992 unique SHA-256, 2,990 unique pHash, eight exact duplicate
groups and nine confirmed perceptual groups.

The rejected receipt is preserved rather than repaired in place: external detail is 2,020,166 B /
SHA `fbdc34d4...ad57`, with compact `evidence/e32_qwen-image-2512_realization.json`. Parent-group
integrity means the later eligibility overlay must remove the two redundant composition groups and
the affected style group in full—12 rows—not individual convenient variants. It may also trim
other top sources deterministically to retain the <=20% cap, but it cannot add post-audit
replacement rows or change the immutable 15K selection.

FLUX.2 also failed intact-source realization despite complete decode and clean protected-role
boundaries. All 3,000 images / 750 prompt groups decode as RGB PNG and have zero protected-E30 or
passed-E32-peer overlap, but only 2,964 SHA-256 and 2,932 pHash values are unique. The audit found
28 exact duplicate groups and 41 confirmed perceptual groups. Duplication is concentrated in
repeated `diffusiondb_orig` prompts and several editing variants; the combined conflict set touches
98 image keys across 32 prompt groups.

External rejected receipt is 2,045,961 B / SHA `53c0793b...1451`; compact evidence is
`evidence/e32_flux2-klein-9b_realization.json`. The later eligibility overlay will operate on a
prompt-group conflict graph: keep a deterministic canonical group between cross-group copies and
drop any group containing an internal duplicate. Exact counts remain deliberately unfrozen until
Community and GPT complete their own audits. No new/unseen FLUX row may be added as replacement.

The first full GPT realization was preserved as rejected rather than normalizing unexpected text
silently. Transfer itself completed all 3,000 selected image/prompt pairs (6,000 assets), reusing
1,703 exact-size local/E32 assets and downloading 4,297. The UTF-8-only prompt gate rejected 107
sidecars, so 2,893/3,000 images reached hashing. Those images are RGB PNG with 2,893 unique SHA,
2,887 unique pHash, zero protected/peer overlap and five confirmed perceptual duplicate pairs.
External rejected receipt is 1,792,420 B / SHA `9ce487a2...5184`; compact evidence is
`evidence/e32_gpt-image-1_realization.json`.

A bounded byte audit established the cause before any code change: every one of the 107 files
decodes as Windows-1252, and their non-ASCII characters consist only of em dashes, curly quotes,
`é` and `ç`; none fails that codec. The next committed method change will therefore use UTF-8 first
and Windows-1252 as an explicit, counted fallback while preserving original bytes and byte hashes.
It will not hide the five duplicate pairs or change the immutable 3,000-row selection.

VISION's first complete realization was also preserved as rejected. All 3,500/3,500 native JPEGs
decode as RGB, contain EXIF and retain exact device balance: 100 parents for each of 35 camera
pipelines. All 3,500 SHA values are unique and protected/peer overlap is zero. Four dHash candidate
buckets reduce to three confirmed perceptual duplicate pairs, leaving 3,497 unique pHash values;
therefore the intact source receives no role. External rejected receipt is 1,939,155 B / SHA
`3312c774...e6b1`; compact evidence is `evidence/e32_vision-base-native_realization.json`. A later
receipt-bound eligibility overlay may keep one stable canonical parent per pair and exclude the
other; it may not replace selected parents after decode.

The GPT encoding repair was implemented as a method checkpoint before rerunning production bytes.
Prompt decoding now tries UTF-8 first and exactly one declared fallback, Windows-1252; undefined
Windows-1252 bytes still fail rather than being replaced. Each realized GPT row will retain the
SHA-256 of original prompt bytes, the SHA-256 of normalized UTF-8 text and its chosen encoding, and
the compact report will publish encoding counts. Two new regression cases prove fallback behavior
and rejection of undefined byte `0x81`; 20 focused E32 selection/acquisition/realization tests pass.
No production receipt was overwritten in this commit.

CommunityForensics then completed the schema-v2 realization cleanly. All 2,800 selected embedded
PNG rows decode as RGB; all 2,800 SHA-256, dHash and pHash values are unique, with zero protected
E30 or passed-E32-peer overlap. The source-cap selection still represents all 300 frozen model
identities, normally nine or ten rows each. External detailed receipt is 1,980,274 B / SHA
`cb4bffe2...76b2`; compact evidence is
`evidence/e32_communityforensics-ai-local_realization.json`. State is role-free
`source_realization_passed_candidate_only`.

Nano Banana Pro was independently rerun to replace its schema-v1 audit artifact with the common
schema-v2 pHash receipt. The result remains unchanged in substance: 200/200 PNGs decode (136 RGB,
64 RGBA), all 200 SHA/dHash/pHash values are unique and all protected/peer overlap counts are zero.
The new detailed receipt is 98,924 B / SHA `55ec23ec...eb8e`; the compact evidence path is unchanged.
This is a method-version refresh, not a new selection or role assignment.

The unchanged 3,000-row GPT selection was rerun after the independently committed decoder fix.
All 3,000 RGB PNG images and all prompts now realize: 2,893 prompts decode as UTF-8 and exactly the
previously diagnosed 107 use Windows-1252. All 3,000 image SHA values are unique, all
protected/peer overlap counts remain zero, and every original prompt byte hash is retained.

The intact GPT source still remains correctly rejected: complete visibility reveals six confirmed
perceptual duplicate pairs (one more than the incomplete first audit could observe), leaving 2,993
unique pHash values. External rejected receipt is 2,239,691 B / SHA `48945f7f...73d5`; compact
evidence at `evidence/e32_gpt-image-1_realization.json` supersedes the first compact artifact while
HISTORY retains both audit outcomes. The later eligibility overlay will keep one stable canonical
row per pair and exclude six losers; it cannot download replacements.

Before implementing the combined overlay, its scope was tightened to close a peer-order hole.
Source realization compares against previously *passed* peers, so internally rejected Qwen, FLUX
and GPT receipts have not necessarily been cross-compared with one another. The precommitted
overlay will therefore recompute exact SHA and frozen dHash+pHash duplicate components globally
across all 15,000 AI records plus 3,500 VISION records. Same-label components retain one stable
parent unit; any REAL/AI component excludes every affected unit on both sides as label-ambiguous.
The subsequent <=20% AI source cap must respect four-row prompt groups and use only selection-bound
stable hashes. This checkpoint changes no data and assigns no role.

The combined eligibility-overlay implementation was then completed without opening production
receipts. It validates exact equality between every frozen selection and audit row set, permits
only duplicate-related source rejections, recomputes global exact/perceptual connected components,
preserves parent groups, removes both labels from cross-label components, and maximizes retained AI
rows subject to an exact <=20% source share. Stable-hash cap trimming is bound to the immutable AI
selection SHA; Qwen/FLUX four-row units remain indivisible. The detailed output will retain every
eligible key and exclusion reason while Git receives only aggregates. Three focused overlay tests
plus 29 related E32 tests pass. No production overlay was emitted by this method commit.

The independently run production overlay then froze a clean role-free subset. It bound immutable
AI selection SHA `3230f026...80b7`, REAL selection SHA `200a7aeb...a4d` and all seven detailed audit
SHAs, and globally compared 18,500 realized rows. It found 59 duplicate row components but no
REAL/AI component and no new cross-source collision beyond the recorded within-source findings.
Thirteen parent units contain an internal duplicate and 20 noncanonical same-label units lose
their components; no replacement is added.

The eligible AI pool is 14,786/15,000: Qwen 2,956, FLUX.2 2,916, Nano Banana 2,957, GPT Image 1
2,957, Nano Banana Pro 200 and CommunityForensics 2,800. Maximum source share is 19.998647%, below
the exact 20% cap; Qwen/FLUX remain divisible into intact four-row prompt groups. VISION retains
3,497/3,500 parents after three canonical-loser exclusions. Detailed external overlay is 913,980 B
/ SHA `b6c2101f...32e4`; compact evidence is `evidence/e32_eligibility_overlay.json`. State is
`eligibility_frozen_role_free`: this is not yet TRAIN/CALIBRATION and does not authorize evaluation
on protected roles.

The next C1 archive step was precommitted while the frozen FODB transfer continued. Archives will
not be blindly expanded: a reproducible gate must reject traversal/absolute paths, symlinks,
encryption, duplicate names, declared-size drift and implausible expansion before extraction.
FODB must prove exactly 3,851 `orig` JPEG parents under 27 device roots and link its five social
copies by device/scene index; only `orig` candidates may be atomically extracted. CSAFE `s21.zip`
must first pass its published MD5 and a separate hierarchy inventory, after which internal rows can
be selected—never before. This planning checkpoint writes no archive member.

The ZIP safety/inventory implementation was committed before the remaining archives completed.
It validates physical archive size and CRC, rejects unsafe paths, backslashes, symlinks, encrypted
members, duplicate names, oversized members and >4:1 expansion. FODB parsing requires matching
device/transport identifiers, exactly one `orig` plus five named social derivatives per parent,
3,851 parents and 27 roots; extraction can write only `orig` members atomically and records each
SHA. CSAFE repeats the published MD5 before inventory and leaves all internal rows unselected.
Fifteen focused acquisition/archive tests pass. This method checkpoint extracted nothing.

The FODB realization path was also implemented before production extraction. It requires the
Git/external extraction-receipt SHA binding, checks missing/partial/size/SHA drift, decodes every
original through the common format/EXIF/dHash/pHash gate, and records camera pipeline, device,
scene group and `orig` state. It then uses the same protected-E30 and passed-peer boundaries as the
other E32 sources and still assigns no role. A receipt-bound fixture brings the focused archive and
realization suite to 16 passing tests. No FODB image was opened by this method checkpoint.

The first production FODB inventory correctly stopped without emitting evidence or extracting a
member. Every device-root file matched the frozen six-transport contract, but `part03.zip` also
contains 4,004 JPEG / 2,834,597,196 bytes under an undocumented-by-our-parser `inspection/` root:
3,861 `check_devices` and 143 `compare_devices` helpers. The latter count equals reported scenes
and the former approximates the original-parent count, so treating them as independent parents
would create derived duplicates and inflate REAL. The repair will explicitly exclude only the
whole `inspection/` root, publish its counts/bytes, and continue rejecting every other unknown
path. The failed run produced no inventory receipt and changed no archive.

The repair now excludes exactly the precommitted top-level `inspection` root, removes it from the
27-device-root assertion, and publishes excluded member/root/byte totals in the receipt. Every
other unrecognized root or device-member pattern remains a hard failure. A regression locks the
single allowed root; 17 focused archive/realization tests pass. Production archives were not
reopened by this repair commit and will be rerun independently.

The untouched three-archive FODB rerun passed the corrected safety gate. Declared physical bytes
total 22,940,347,533; archive SHAs are `c719cac3...517c`, `271e07da...e5f1` and
`a3c2d69f...2a6d`. CRC, path, symlink, encryption, duplicate-name and expansion checks pass.
Exactly 3,851 device/scene parents span 27 camera pipelines and 143 scene groups, and every parent
has one `orig` plus Facebook, Instagram, Telegram, Twitter and WhatsApp derivatives. The 4,004
`inspection` JPEGs / 2,834,597,196 B are published as excluded nonparents.

The detailed external inventory is 5,356,810 B / SHA `d378573f...9631`; compact evidence is
`evidence/e32_fodb_archive_inventory.json`. State is
`archive_inventory_passed_orig_parents_unextracted`. No archive member was extracted by this
checkpoint; the next independent action may extract only the 3,851 `orig` members.

FODB original-only extraction then completed from the committed inventory. Exactly 3,851 JPEG
parents / 15,416,129,383 B were written atomically below `e32/real/fodb/orig`, with per-file SHA,
camera pipeline, device and scene group in the external receipt. No Facebook, Instagram, Telegram,
Twitter, WhatsApp or `inspection` member was extracted. Detailed extraction receipt is 1,311,414 B
/ SHA `a1626b0b...8b05`, bound to inventory SHA `d378573f...9631`; compact evidence is
`evidence/e32_fodb_orig_extraction.json`. State is `orig_extraction_complete_role_free`, not a
training role or data-quality pass.

FODB's independent full realization passed. All 3,851/3,851 originals decode as RGB JPEG and carry
EXIF; all 3,851 SHA values are unique, all protected/passed-peer overlap counts are zero and all 143
scene groups / 27 camera pipelines are retained. Seven same-scene cross-camera pairs share dHash,
but none meets the frozen pHash <=5 confirmation rule, so there is no confirmed perceptual
duplicate. This is expected evidence that different devices photographed the same scene, not a
reason to inflate or delete parents.

Detailed external audit is 2,588,737 B / SHA `dcbf8b55...fd11`; compact evidence is
`evidence/e32_forchheim-fodb_realization.json`. FODB passes only as
`source_realization_passed_candidate_only`; scene/device-disjoint role freezing remains pending.

Before admitting FODB to the role-free combined pool, the global overlay extension was
precommitted. It will bind the original extraction receipt as a third selection input and compare
all 3,851 FODB rows against the existing 15,000 AI plus 3,500 VISION rows using the unchanged
SHA/dHash+pHash rule. Scene metadata remains available for later group-disjoint folds. Any REAL/AI
component loses both sides; absent a new collision, the already-frozen AI source-cap subset must
remain identical. This planning checkpoint changes no overlay.

The FODB overlay extension was implemented before reopening production receipts. It requires the
exact role-free extraction receipt, derives the same camera-parent unit IDs as realization, binds
the receipt SHA alongside AI/REAL selections and demands exact row equality with the FODB
schema-v2 audit. Thirteen focused overlay/realization tests pass. This method commit does not yet
replace the 18,500-row overlay evidence.

The independent 22,351-row global overlay rerun passed. Adding all 3,851 FODB parents created no
new exact/perceptual component and no REAL/AI ambiguity; the 59 known components and all AI
eligible keys therefore remain unchanged. AI stays 14,786 with maximum source share 19.998647%.
REAL is now 7,348 role-free parents: 3,497 VISION plus all 3,851 FODB.

The new detailed overlay is 1,179,329 B / SHA `510e94eb...fc3b`, bound additionally to FODB
extraction SHA `a1626b0b...8b05` and audit SHA `dcbf8b55...fd11`. Compact evidence at
`evidence/e32_eligibility_overlay.json` supersedes the 18,500-row aggregate while HISTORY retains
both. The C1 acceptance floor is not yet met; CSAFE remains necessary for >=10,000 REAL parents.

CSAFE's single-stream Figshare/S3 transfer later degraded to roughly 9 MB/min after preserving a
contiguous ~4.70 GB prefix. A one-byte diagnostic request confirmed HTTP 206 and exact byte-range
support. Before interrupting that resumable prefix, a four-range recovery path was precommitted:
each remaining range must download to an independent partial, validate `Content-Range` and length,
assemble beside—not over—the original prefix, reproduce the published full MD5, and only then
promote atomically. Any failure leaves the original prefix and range partials recoverable.

The four-range CSAFE recovery was implemented and tested before production use. It partitions only
the missing suffix, resumes each independent range file, requires HTTP 206 plus exact
`Content-Range`, assembles prefix+ranges into a separate file while computing MD5, and promotes
only on the published `5c5f...91d8`. Prefix/ranges are deleted only after verified promotion; a
failure preserves them and the assembled artifact for review. Three new range/assembly regressions
bring the focused acquisition/archive suite to 19 passing tests. Production recovery remains a
separate action.

The production CSAFE range recovery completed successfully. It preserved the 4,723,834,880-byte
single-stream prefix, fetched four disjoint exact ranges, assembled all 17,588,803,163 bytes beside
the prefix and reproduced published MD5 `5c5f79e3e508a5cbf7a19e75846091d8`. Only after this full-file
verification was `s21.zip` promoted and the prefix/range temporaries removed. This completes every
frozen C1 transfer without modifying an upstream archive; internal CSAFE rows remain unselected
until the independent ZIP inventory passes.

CSAFE's independent archive inventory then passed published MD5, CRC, path/symlink/encryption,
member-size and expansion checks. The 17,588,803,163-byte ZIP has SHA `54a7193c...25df` and 7,996
JPEG members under ten physical devices (`s21_1` through `s21_10`, 798–800 each). Its hierarchy is
scientifically decisive: 4,000 `blank` flat-field images and 3,996 `natural` images, each divided
across front, telephoto, ultra and wide pipelines. Blank fields must not define REAL or inflate the
parent count.

Detailed external inventory is 1,306,218 B / SHA `77a88649...fd8d`; compact evidence is
`evidence/e32_csafe_archive_inventory.json`. State remains
`archive_inventory_frozen_internal_rows_unselected`. A separate committed selector must choose
natural rows only and preserve device/lens identity before extraction.

The CSAFE natural-only path was precommitted before opening an internal JPEG. The selector will
take all 3,996 inventory-declared `natural` members, bind physical device and lens pipeline and
exclude every one of the 4,000 `blank` fields mechanically. Only that frozen list may be extracted
atomically with per-file SHA. A receipt-bound realization command must exist before production
extraction and must apply the shared decode/protected/duplicate gate. Selection, extraction and
realization remain separate commits and none assigns TRAIN/CALIBRATION.

The CSAFE natural-only method was implemented before production selection. The parser permits only
the ten frozen device IDs, `blank|natural`, four known lenses and JPEG suffix. Metadata selection
requires exact 4,000/3,996 counts and reads no member bytes; extraction verifies inventory
size/CRC, writes only natural rows atomically and records SHA; realization binds that receipt and
applies shared decode/protected/duplicate checks while publishing device/lens counts. Twenty-three
focused archive/realization tests pass. No production member was selected or opened by this commit.

The metadata-only CSAFE selection then froze all and only the 3,996 natural JPEG members. Ten
physical devices contribute 398–400 rows each; front and telephoto contribute 998 each, ultra and
wide 1,000 each. All 4,000 blank fields are excluded. No ZIP member byte was opened. Detailed
selection is 1,193,310 B / SHA `3a24bd50...ad1c`, bound to inventory SHA `77a88649...fd8d`;
compact evidence is `evidence/e32_csafe_natural_selection.json`. State is
`natural_selection_frozen_no_member_bytes_read` and assigns no role.

CSAFE natural-only extraction completed from the committed selection. Exactly 3,996 JPEG parents /
13,219,178,988 B were written atomically below the isolated E32 root with per-file SHA, archive
member, device, lens and camera-pipeline metadata. No `blank` member was extracted. Detailed
receipt is 1,775,854 B / SHA `32acdfb3...d7e4`, bound to selection SHA `3a24bd50...ad1c`; compact
evidence is `evidence/e32_csafe_natural_extraction.json`. State remains
`natural_extraction_complete_role_free` pending independent realization.

CSAFE full realization passed. All 3,996/3,996 natural members decode as RGB JPEG and carry EXIF;
all 3,996 SHA and pHash values are unique. One cross-device wide-lens pair shares dHash but is not
pHash-confirmed, and every protected/passed-peer overlap count is zero. The balanced ten-device /
four-lens structure remains intact.

Detailed external audit is 2,521,737 B / SHA `3ea951ec...b701`; compact evidence is
`evidence/e32_csafe-mcsidb-s21_realization.json`. State is
`source_realization_passed_candidate_only`. C1 now has enough source-audited volume to exceed the
10,000 REAL floor, subject to the final global cross-source/cross-label overlay rerun.

Before adding CSAFE to the combined pool, its overlay extension was precommitted. The rerun must
bind the exact natural-extraction receipt and schema-v2 audit, compare 15,000 AI plus 11,347 REAL
selected rows globally, exclude both labels from any ambiguous component and preserve the existing
AI eligible keys if no new collision exists. Only this result may close the >=10,000 REAL gate;
source realization alone is insufficient.

The CSAFE overlay input was implemented before production rerun. It requires the exact
natural-extraction state/SHA, maps each selected camera parent without inventing rows and demands
exact equality with the CSAFE schema-v2 audit. The selection hash joins AI, VISION, FODB and CSAFE
bindings. Fourteen focused overlay/realization tests pass. The previous 22,351-row production
overlay remains unchanged by this method commit.

The production CSAFE-inclusive global overlay then closed the E32 candidate-pool volume gates.
Exactly 26,347 immutable selected rows were compared together: 15,000 AI and 11,347 REAL. No new
global duplicate component and no REAL/AI ambiguity appeared. The eligible AI subset therefore
remains bit-for-bit 14,786, while eligible REAL reaches 11,344: VISION 3,497, FODB 3,851 and CSAFE
3,996. The 59 already-known duplicate components remain globally visible; exclusions still total
20 same-label noncanonical units and 13 within-parent rows.

The detailed overlay is now 1,431,190 B / SHA `45830283...78b6`, with the combined selection bound
to AI SHA `3230f026...80b7`, VISION SHA `200a7aeb...ca4d`, FODB extraction SHA
`a1626b0b...8b05` and CSAFE extraction SHA `32acdfb3...d7e4`. Compact evidence remains
`evidence/e32_eligibility_overlay.json`; state is `eligibility_frozen_role_free`. C1 and C2 now
meet their 10K–20K eligible-volume targets, three-collection REAL floor and five-modern-family AI
floor. This is deliberately not a trained model: the next irreversible boundary is a committed,
group-aware TRAIN/CALIBRATION manifest followed by the cheapest frozen DINOv2-S control.

Before opening an eligible image for representation training, the C3 role transition was
precommitted. The model pool will be exactly balanced at 11,344 parents per class. Every eligible
REAL parent remains; AI is deterministically reduced to Qwen 2,232, FLUX.2 2,232, Nano 2,227, GPT
2,227, Nano Banana Pro 200 and CommunityForensics 2,226. No model score, pixel statistic or image
byte may select these rows.

CALIBRATION targets about 20% inside every source using stable group assignment. Qwen/FLUX prompt
groups, Community generator identities, VISION/CSAFE devices and FODB scenes cannot cross roles.
FODB is a crossed 27-camera-by-143-scene design: enforcing both camera- and scene-disjoint roles
connects the whole collection and makes a split impossible. The contract therefore prioritizes
scene leakage prevention, reports device overlap honestly and keeps this limitation out of any
unseen-camera claim. The method, tests and production role receipt are separate later commits.

The metadata-only C3 role freezer was implemented before production use. It binds the final
eligibility-overlay byte count/SHA and all nine audit receipts, makes the exact precommitted AI
downselection while keeping Qwen/FLUX prompt groups indivisible, then solves a deterministic
nearest-target subset assignment over each source's protected role groups. It fails on changed
inputs, impossible exact counts, duplicate IDs, empty source-role cells or any role-group overlap.
Eight focused role/overlay tests pass; no production role manifest or image byte was opened by this
method commit.

The production C3 role freeze succeeded over metadata only. It retained all 11,344 eligible REAL
parents and the precommitted 11,344 AI subset. TRAIN contains 18,154 rows (9,081 AI / 9,073 REAL);
CALIBRATION contains 4,534 (2,263 AI / 2,271 REAL). Every one of nine sources appears in both
roles. Device, scene, prompt and generator-identity group intersections are exactly zero under the
declared per-source contract, and no DEVELOPMENT or LOCKED row exists.

The detailed manifest's record-list SHA is `568e8e26...d887`; compact evidence is
`evidence/e32_c3_role_manifest.json`. The manifest remains a parent-role contract, not a feature
archive or fitted model. It now authorizes a byte-identical preprocessing audit and the R0 frozen
DINOv2-S screen without opening any protected final.

The first runnable E32 model contract was precommitted before derived inputs or embeddings. Every
C3 parent will receive one identical EXIF-aware RGB transform: short side 256, center crop 224,
then JPEG q90/4:4:4. This prevents the classifier API from reading container type, alpha channel,
native geometry or path, while honestly retaining the possibility of content and earlier codec
bias. Every output byte must be hashed back to C3.

R0 is the already-cached frozen DINOv2-S final embedding plus a standardized, class-weighted
logistic head. Only TRAIN fits the head. CALIBRATION chooses C from {0.01, 0.1, 1, 10} by AUC and
then the lowest threshold meeting <=10% authentic source-macro FP and <=20% worst-source FP. The
saved artifact must bind model ID, input receipt, feature archive and threshold. This is a
group-held-out, source-stratified prototype screen—not unseen-source or final evidence.

The R0 input realizer and trainer were implemented before production bytes. Loose JPEG/PNG/JXL
payloads and the two Parquet-backed sources share one decoder and one fixed transform; source SHA
is rechecked before an atomic derived write. Reruns accept an existing output only when its bytes
exactly reproduce the transform. The trainer binds the complete receipt, rechecks every derived
hash, caches record-aligned frozen DINO features, fits only TRAIN and saves a preprocessing/model/
threshold-bound joblib artifact. Thirteen focused role/input/training tests pass. No production R0
input, feature or model was created by this method commit.

Production R0 input realization completed without losing a parent. All 22,688 C3 rows were
source-SHA verified and transformed to the same 224x224 RGB JPEG q90/4:4:4 input. The result is
exactly balanced at 11,344/class and preserves TRAIN 18,154 / CALIBRATION 4,534. Logical payload is
487,845,683 bytes; exFAT allocation is larger because 22,688 small files occupy full allocation
units. Record-list SHA is `f9424d3d...f14b`; detailed receipt is 9,021,504 B / SHA
`2255b123...5199`, with compact evidence `evidence/e32_r0_input_receipt.json`. This freezes the
only input set the R0 feature extractor may open.

The first E32 model is now genuinely fitted and runnable. Frozen DINOv2-S produced a 22,688x384
feature archive; a class-weighted logistic head selected C=0.1 from the preregistered grid. On the
4,534-row group-held-out CALIBRATION role it achieves AUC 0.996404, AP 0.996769, AI recall 99.07%,
REAL recall 90.14%, balanced accuracy 94.60% and F1 94.82%. Authentic source-macro FP is 9.97%; the
worst source is FODB at 13.84%, below the 20% ceiling. Current-family macro AI recall is 99.14% and
the weakest current family, Nano Banana Pro, is 97.5%. All five preregistered R0 screen checks pass.

The cached feature archive is 33,439,283 B / SHA `716df956...be3b`; the 12,720-byte fitted artifact
has SHA `7f170340...a85e`. Compact evidence is `evidence/e32_r0_dinov2s.json`, state
`r0_screen_passed_runnable_candidate`. This is the first successful E32 model but not yet a final
generalization result: every CALIBRATION source also has TRAIN groups, and DEVELOPMENT/LOCKED arms
remain untouched. The next step is a stable one-image inference boundary plus independent smoke
tests, then an actually unseen-source evaluation before any universal claim.

Before re-opening the already-consumed owner gallery, the R0 inference boundary was precommitted.
The CLI must verify artifact and DINO weight hashes, reproduce the complete in-memory R0 transform
and return machine-readable score/threshold/verdict per supported still image. Unit tests precede
gallery use. The 210 JPEG/PNG gallery images remain DEVELOPMENT because earlier models already
scored them; the MOV file is out of this still-image contract. Its result may diagnose practical
false positives but may not change the artifact or become a locked claim.

The E32 serving boundary was implemented before gallery use. `pixelproof-predict-e32` hard-verifies
the 12,720-byte head artifact and cached DINO weight SHA, reproduces the exact JPEG round-trip and
supports batched files/directories with one JSON verdict per image. A separate gallery runner
stores only aggregate DEVELOPMENT evidence plus the ten highest basenames; it does not refit or
change threshold. Ten focused candidate/input/trainer tests pass. Gallery pixels remain unopened
by this method commit.

The frozen E32 R0 candidate then failed its first owner-real DEVELOPMENT stress badly. Of 210
supported still images, 159 crossed the untouched 0.141444 threshold; REAL recall is only 24.29%
and median AI score is 0.6806. One MOV was excluded by contract. This directly contradicts the
90.14% REAL recall seen inside source-stratified CALIBRATION and shows severe authentic-pipeline
shift/source shortcut learning. The artifact was not refit and the threshold was not changed.

Evidence is `evidence/e32_owner_gallery_smoke.json`, bound to artifact SHA `7f170340...a85e` and
gallery identity SHA `390e3c21...ac09`. The success and failure are both retained: E32 achieved a
working end-to-end model and an excellent within-collection group holdout, but not a trustworthy
real-world detector. It is blocked from service and locked finals. The next scientific correction
must change validation/data/representation—not cosmetically recalibrate on these 210 known photos.

Final engineering verification reproduced the hash-checked one-image CLI on owner photo
`IMG_8540.jpeg` (score 0.699661, false AI trigger), then passed 174 Python tests, six web tests,
the production web build and TypeScript typecheck. This separates two facts cleanly: the E32
pipeline, model artifact and inference program operate correctly; the learned decision does not
generalize to the owner's authentic camera pipeline. That distinction is now presentation-grade
evidence rather than an unrecorded failure.

A source-held-out postmortem was precommitted next because it reuses the frozen 22,688x384 cache
and cannot tune against the owner gallery. Nine LOCO rounds will remove one complete collection
from both fitting and threshold selection, reproduce C=0.1 plus the authentic FP budgets on the
remaining roles and measure only the held-out collection. The result is diagnostic: it may decide
the next data/representation action but cannot alter or rescue the already-frozen R0 artifact.

The LOCO runner was implemented and tested before reading results. It hard-binds the input receipt
and feature-cache SHAs, removes the held-out source from both TRAIN fitting and CALIBRATION
thresholding, refuses any arm that loses a class and reports the absent source separately. Five
focused LOCO/threshold tests pass. The accepted artifact and owner-gallery evidence are read-only.

All nine LOCO rounds completed. Generator transfer is not the immediate bottleneck: held-out AI
macro recall is 98.34% and the weakest absent source, CommunityForensics, still reaches 95.78%.
Authentic transfer fails: held-out CSAFE FP 15.74%, VISION FP 19.82% and FODB FP 34.85%, for 23.47%
macro / 34.85% worst-source FP. These results explain why the source-stratified split looked
excellent and why a genuinely new owner pipeline collapsed even harder.

Evidence is `evidence/e32_r0_loco.json`; the original artifact remains byte-for-byte unchanged.
The next priority is therefore not a model ensemble or expensive full fine-tune. First add a
licensed, diverse REAL complement with a matched storage/input route and enforce a source-held-out
authentic gate. The already-local CommunityForensics REAL half is the cheapest candidate because
its AI half is already admitted and its matched container route can suppress a major source
shortcut; it still requires independent metadata selection, licence/provenance confirmation and
full decontamination before use.

Metadata inspection rejected that apparent shortcut. CommunityForensics-Small has 32,912 REAL
rows, but all identify as FFHQ/Real and therefore add only a face domain—not diverse owner-like
camera content. The already-local `34data__communityforensics-real` has 8,000 embedded JPEGs and
useful geometry, but it is an unofficial repack with no local/upstream dataset card, licence or
source identity. The `theminji` real parquets have the same provenance/licence gap and were already
implicated in earlier shortcut failures. None is admitted merely to inflate volume.

The next representation screen was precommitted without new downloads. R1a will pass the exact
22,688 standardized R0 inputs through the pinned MIT Community-Forensics ViT-S and fit a fresh
regularized binary head on its frozen CLS embeddings under the same C grid, roles and authentic FP
budgets. This is motivated by measured complementarity: the frozen CF decision made only one
false alarm on 206 unique owner images, while its old fixed head missed most current AI. A new head
on the modern E32 pool tests whether the representation can keep real specificity and learn the new
families. Owner pixels remain closed until an internal artifact is independently frozen.

The R1a implementation was completed before feature extraction. It hard-verifies the pinned
CF-ViT revision and weight SHA, every R0 input byte and record order, caches only aligned CLS
embeddings, then applies the unchanged C-grid and CALIBRATION threshold budgets. A separate
artifact prevents any accidental replacement of R0. Four focused CF-head/threshold tests pass; no
production CF feature, head result or owner image was opened by this commit.

R1a completed its internal screen successfully. The frozen CF-ViT CLS matrix is 22,688x384; C=0.01
won the preregistered grid. CALIBRATION AUC is 0.998222, AP 0.998351, AI recall 99.91%, REAL recall
90.05%, balanced accuracy 94.98% and F1 95.20%. Current-family macro AI recall is 99.95% and its
weakest family is 99.77%. Authentic macro FP is 9.97%, with 12.77% worst-source FP. Every screen
gate passes.

The feature archive is 33,436,875 B / SHA `c170a1f6...bc6b`; the 12,703-byte artifact SHA is
`6288acba...d670`; compact evidence is `evidence/e32_r1a_cfvit.json`. This artifact is now frozen
before owner-gallery use. Its strong source-stratified result is still not a generalization claim;
the next separate gate is the already-consumed, refit-forbidden owner-real DEVELOPMENT stress.

Only after freezing R1a was its owner-gallery stress precommitted. A separate scorer must verify
the artifact, CF model revision/weight and exact R0-style JPEG round-trip before applying the new
CLS head. It will score the same 210 supported stills once at threshold 0.118110, exclude MOV and
make no parameter change. The comparison is diagnostic and fixed in advance: R1a versus R0's
24.29% REAL recall and the original CF decision's historical 99.51% on 206 unique gallery images.

The R1a inference boundary was then implemented and tested while owner pixels remained closed.
`pixelproof-predict-e32-cf` verifies the frozen 12,703-byte artifact, pinned CF revision and cached
weight SHA; it reproduces R0's JPEG round-trip before the official CF processor and emits stable
JSON scores/verdicts. The separate gallery runner can only write aggregate DEVELOPMENT evidence
and cannot alter the head or threshold. Four focused candidate/input/trainer tests pass. This
method checkpoint is committed before the one allowed gallery run.

The frozen R1a gallery run failed. At the untouched 0.118110 threshold, 154/210 authentic stills
were labeled AI: only 26.67% REAL recall, versus R0's 24.29%; median AI score was 0.4892. One MOV
was excluded. The gallery identity remained `390e3c21...ac09`, proving this is the same evaluation
population, and neither the head nor threshold changed. Evidence SHA is `2e242ef5...b3a`.

This closes the encoder-only hypothesis: both generic DINOv2-S and a forensic CF-ViT trunk achieve
near-perfect source-stratified CALIBRATION yet collapse on a new authentic camera pipeline. R1a is
rejected from serving and LOCKED FINAL. The next admissible correction is a licensed,
provenance-complete fourth REAL camera source plus a source-held-out REAL gate; the consumed owner
gallery stays DEVELOPMENT and may not become training or threshold data. E26 remains the working
demo while E32 R0/R1a remain reproducible rejected controls.

Final verification reproduced the hash-checked R1a CLI on `IMG_8540.jpeg` (score 0.016521, correctly
below threshold), passed all 178 Python tests, all six web tests, the production web build and
TypeScript typecheck. The isolated correct example does not override the 210-image aggregate
failure; it only confirms that the inference path itself is operational.

The corrective data move was frozen next, before any new selected image byte. CSAFE's official
Figshare API exposes a CC-BY-4.0 iPhone 14 archive at 20,428,338,922 bytes with published MD5
`dfc01c89...946c`; it is a training-side candidate only after natural-only inventory and audit.
IPN-NFID v3 independently links twelve CC-BY-4.0 smartphone articles containing exactly 960
natural JPEGs / 3,889,897,594 bytes; that whole source is reserved as DEVELOPMENT and forbidden
from fitting or threshold selection. This pairing adds a relevant Apple training pipeline while
retaining a genuinely absent-source gate. Drift in API identity/licence/size/MD5 is a hard stop.

The R1b acquisition method was implemented before selected-byte transfer. It separately freezes
official metadata, downloads IPN files with four bounded workers, downloads the CSAFE archive with
resumable curl, preserves `.partial` state and verifies every completed file against its published
size and MD5 before promotion. A 100 GiB free-space floor is enforced. Four focused selection and
drift tests pass; no selected image byte was fetched by this method checkpoint.

The production R1b metadata freeze then reproduced all official contracts without downloading a
selected image: 960 IPN natural JPEGs / 3,889,897,594 bytes across twelve devices and the
20,428,338,922-byte CSAFE iPhone 14 archive. Detailed selection SHA is `c807d140...1c7f`; this
receipt is frozen before either transfer begins.

The first CSAFE iPhone 14 single stream was intentionally stopped after preserving a 92,159,662-byte
contiguous prefix because mobile throughput made a 20.4 GB serial transfer impractical. No byte was
deleted or promoted. Before code changes, the S21-proven recovery contract was reused: four exact
remaining ranges, strict HTTP 206/Content-Range/length checks, prefix-plus-ranges assembly to a new
temporary file, whole-archive published MD5, then atomic promotion. IPN remains an independent
transfer and continues unaffected.

Append-only correction: the filesystem `stat` captured immediately after the preceding checkpoint
is authoritative; the preserved prefix is **92,274,688 bytes**, not the mistyped 92,159,662. Range
planning reads the live prefix size and binds 92,274,688.

Four-range iPhone recovery was implemented against that live prefix. Range planning is exhaustive
and disjoint; every response must be HTTP 206 with the exact requested interval and total. Assembly
uses a distinct temporary path and whole-file MD5 before atomic promotion. Eighteen combined
R1b/original acquisition tests pass. Production ranges remain unopened by this method commit.

IPN-NFID transfer then completed independently: 960/960 natural JPEGs, twelve devices and exactly
3,889,897,594 bytes, with each published MD5 verified before promotion. Before any pixel decode or
model score, a separate audit was precommitted: bind selection+download receipts, retain shared
scene identities across phones, reject decode/exact/protected-peer overlap and record perceptual
candidates. The audit is data-quality only; DEVELOPMENT images remain forbidden from fitting.

The IPN audit implementation binds both receipts, derives landscape/portrait scene ids across
devices, decodes and hashes without importing a detector, distinguishes legitimate same-scene
cross-camera similarity from cross-scene collision and checks protected/passed-peer overlap.
Twenty focused audit/acquisition/realization tests pass. Pixels remain unopened by this method
commit.

Production IPN realization passed without loading a detector. Every one of 960 files decoded as RGB
JPEG with EXIF and had a unique SHA; the dataset contains 80 shared scene groups (50 landscape, 30
portrait) across twelve devices. There is zero protected E30 overlap, passed E32 peer overlap or
cross-scene perceptual collision. Detailed report SHA is `f5827dce...243b`. IPN is now a clean,
source-held-out DEVELOPMENT gate and stays unscored until an R1b artifact is frozen.

CSAFE iPhone 14 range recovery also completed: 20,428,338,922 bytes and published MD5
`dfc01c89...946c` passed before promotion; the 92,274,688-byte prefix and four range files were
removed only afterward. Central-directory-only inspection (no member pixel decode) exposes 7,996
JPEGs across ten device roots: 4,000 blank and 3,996 natural, with front/telephoto/ultra/wide
lenses. A receipt-bound safe inventory, natural-only freezer and atomic extractor were
precommitted before member bytes are opened.

The iPhone archive handler was implemented without reading member payloads. It reuses the proven
ZIP traversal/symlink/encryption/CRC/expansion gates, binds acquisition receipts, accepts only the
exact iPhone14/device/content/lens/JPEG hierarchy, freezes natural rows from metadata and extracts
through size/CRC/SHA-checked atomic writes. Twenty-four combined archive/acquisition tests pass.

Production iPhone 14 inventory passed every CRC and safety check. The 20,428,338,922-byte archive
SHA is `22f04a95...8cbb9`; all 7,996 members are expected JPEGs, split exactly into 4,000 blank and
3,996 natural across ten devices and four lenses. Detailed inventory SHA is `8931a535...912e`.
No member was selected/extracted, so the inventory can now be frozen before natural selection.

The frozen inventory then produced an exact natural-only metadata selection: 3,996 members at
398-400/device and 998-1,000/lens; all 4,000 blank captures are excluded. Detailed selection SHA
is `88dc326e...7b74`. No member payload was opened and no role assigned; this receipt is committed
before extraction.

iPhone 14 natural extraction completed all 3,996 frozen members / 12,914,703,500 bytes with
size+CRC checks, atomic writes and per-file SHA; no blank member was extracted. Detailed receipt
SHA is `46b36e56...09de`. Before decoding pixels, the realization gate was precommitted to bind this
receipt, check format/EXIF/SHA+dHash+pHash and compare against protected E30/passed peers plus stored
IPN hashes. The already-consumed owner gallery may contribute raw exact hashes only after its
identity reproduces `390e3c21...ac09`; no gallery/IPN model score or threshold access is allowed.

The iPhone realization gate was implemented before decoding its pixels. It hard-binds extraction,
decodes and hashes every candidate, applies the existing protected/peer duplicate gates, consumes
only stored IPN hashes and verifies the owner population through raw exact-file identity. It has no
detector import or score path and assigns no role. Eighteen focused iPhone/realization/identity
tests pass.

The resumable open-component binder then cached all ten eligible Commons metadata populations and
froze 1,100 REAL plus 240 local StyleGAN2 reserve identities. Its budget check exposed a second
pre-transfer problem: the selected Commons files alone total 4,140,590,955 bytes, so adding 800
Datapoint images could not respect the global 4 GiB ceiling. No image or model was opened. Contract
`c6f2cfb0...f794` is archived as a failed feasibility bind rather than overwritten. A model-blind
scan of the same frozen metadata found that capping each original at 4 MiB still fills all ten
110-row reserves and lowers Commons to roughly 2.52 GiB, motivating an explicit V2 contract.

The size-aware V2 was implemented, regression-tested and committed before rebinding. It preserves
all ten 100-photo targets and freezes 1,100 Commons reserves under 4 MiB each at exactly
2,706,581,778 bytes, leaving 1,588,385,518 bytes for Datapoint under the original global stop. The
same 240 local StyleGAN2 coordinates are bound. V2 contract `1d4e184c...82aa` and reserve identity
`31c0e420...e171` remain untransferred and unscored. Commons transfer is deliberately deferred
until the gated component exposes exact sizes; this avoids spending 2.71 GB on an impossible final.

The free local work then continued with the frozen StyleGAN2 coordinates. A committed, tested
realizer read only the required Parquet row groups, rechecked label=AI/generator=14, decoded all
240 candidates and compared them against 15 protected-role manifests plus internal exact/dHash
sets. There were zero decode failures and zero overlaps; the first 200 ranks froze at manifest
`150ed354...ec99`, totaling 20,111,615 bytes. Every selected payload is 256x256 PNG, so the record
now explicitly warns that format/geometry may be a shortcut and forbids treating this source alone
as final evidence. No detector was loaded and no score was created.

Production realization decoded all 3,996 iPhone parents as RGB with EXIF and found zero overlap
with protected E30, passed peers, stored IPN or the exact 210-photo owner population. It still
stopped, correctly, on one confirmed duplicate burst: byte-distinct `IMG_1290.JPG`/`IMG_1291.JPG`
from the same iPhone14_5 telephoto pipeline have equal dHash+pHash and the same visible scene.
Another key observation is 3,945 MPO payloads versus 51 JPEG despite `.JPG` suffixes. Rejected audit
SHA is `8325aaf4...05fd`; no role was assigned.

Before correction code, the rule was frozen: exclude the entire two-row perceptual component,
never select one side, preserve the raw audit and freeze the remaining 3,994 as role-free only.
MPO/JPEG differences must be removed by the existing standardized JPEG input contract before any
head fitting so container type cannot define REAL.

The corrective overlay was implemented as a receipt transformer, not a file mutation. It accepts
only the exact single failure/component above, marks both members ineligible, retains every source
byte and assigns no role. Two focused component/owner-identity tests pass. Production eligibility
remains unopened by this method commit.

Production overlay passed exactly as frozen: both duplicate-burst rows excluded, 3,994 role-free
parents retained, detailed SHA `a71c4a06...57bf`; no source file was removed. R1b's controlled
data test is now fixed: preserve all old C3 roles, append only these iPhone parents, split complete
devices 8/2 between TRAIN/CALIBRATION by stable hash, add no AI rows and rely on class-weighted
heads. This isolates authentic Apple coverage rather than silently changing two variables.

The controlled R1b role extension was implemented as an append-only manifest transformation. It
hard-binds the C3 and iPhone eligibility receipts, preserves every old record and role in order,
splits exactly eight/two complete iPhone devices by the existing stable subset rule, refuses id or
group leakage and intentionally does not rebalance. Six focused role tests pass; production roles
remain unopened by this method commit.

Production R1b roles froze 26,682 rows while preserving the 22,688-row C3 prefix exactly. Class
counts are 11,344 AI / 15,338 REAL; role counts TRAIN 21,349 / CALIBRATION 5,333. iPhone contributes
3,195 TRAIN rows from eight devices and 799 CALIBRATION rows from iPhone14_4/iPhone14_8, with zero
device overlap. Detailed manifest SHA is `16deb276...750f`, records SHA `263af46b...5611`. No
DEVELOPMENT or LOCKED row was included.

The R1b input extension was implemented before derived bytes. It binds the R1b manifest and old R0
receipt, reuses all 22,688 old standardized files byte-for-byte and materializes only the 3,994 new
iPhone rows through the identical EXIF-transpose/RGB/short-256/center-224/JPEG-q90-4:4:4 function.
Seven focused input tests pass; IPN/owner paths are absent.

Production standardization completed all 26,682 inputs / 568,959,891 logical bytes; detailed
receipt SHA is `400a990d...6af8`, record-list SHA `3e51f87a...1395`. Every old standardized row was
reused and only iPhone rows were added, neutralizing MPO versus JPEG before the encoder.

The controlled trainer was implemented next. It verifies and reuses the frozen 22,688-row DINO and
CF feature archives, extracts only 3,994 iPhone embeddings, merges strictly by record id and runs
the unchanged class-weighted C grid plus authentic-source FP budgets. Six focused merge/head tests
pass. Before results, external-arm selection was fixed: among passing arms choose higher CAL AUC;
exact tie chooses smaller selected C, then DINO. IPN/owner model scores remain closed.

Both controlled internal arms passed. DINO: CAL AUC 0.996860, current-AI macro/worst recall
99.18%/97.5%, REAL macro/worst FP 9.97%/15.91%, artifact `aca41dd8...8e86`. CF: CAL AUC 0.998079,
current-AI macro/worst 99.82%/99.55%, REAL macro/worst FP 9.97%/12.64%, artifact
`68a54aa2...701c`. The preregistered higher-AUC rule therefore freezes CF at C=0.01 and threshold
0.125935. A deterministic selection receipt binds both evidence files; external scores remain
unopened and DINO is no longer eligible for outcome-based rescue.

After CF selection, the external R1b gate was precommitted before scorer code. It will verify the
selected artifact/CF weights/0.125935 threshold, bind the clean 960-row IPN realization and exact
210-photo owner identity, reproduce the same JPEG round-trip and score CF only once. Passing needs
IPN worst-device FP <=20% and owner FP <=20%; internal current-AI macro is already >90%. No refit,
threshold change, DINO fallback or test-derived policy is allowed.

The selected R1b inference boundary was implemented while external pixels remained model-closed.
`pixelproof-predict-e32-r1b` verifies selection/artifact/CF weights and reproduces the exact JPEG
round-trip. A separate runner binds IPN and owner identities, reports per-device/aggregate metrics
and contains only the frozen gate. Three focused candidate/gate tests pass; the method commit
precedes the one external run.

The single frozen R1b external DEVELOPMENT run failed both authentic gates and changed no model
parameter. IPN-NFID produced 249/960 false positives: REAL recall 74.06%, macro-device FP 25.94%
and worst-device FP 40.0% (iPhone SE 2020 instance 1). The already-consumed owner gallery produced
144/210 false positives and 31.43% REAL recall, modestly above R0's 24.29% and R1a's 26.67% but far
below the required 80%. Internal current-AI macro recall remained 99.82%, so the AI gate passed
while both authentic gates failed. Evidence is `evidence/e32_r1b_external_development.json`.

R1b is rejected from serving; the threshold was not changed, DINO was not substituted and no
LOCKED AI arm was opened. The experiment establishes a stronger conclusion than the earlier
encoder controls: adding nearly four thousand clean iPhone images repairs the represented Apple
domain but does not create camera-source invariance. The next step is an objective/decision-layer
redesign evaluated with leave-one-real-source-out calibration, followed by a genuinely new
multi-camera authentic gate. IPN and the owner gallery are now permanently consumed DEVELOPMENT
sets and cannot tune that redesign.

Final engineering verification after the R1b rejection passed 203 Python tests, the production web
build and six web tests, TypeScript typecheck, ESLint, `pip check` and the canonical artifact
registry check. The compact external evidence SHA-256 is
`8752699f643609588d8725f34c469c44791f6712cffcca7c03677de30b1c8d48`. A hash-verified R1b CLI
smoke on one owner still loaded the pinned CF model and emitted score 0.313247 at threshold
0.125935 (incorrect AI verdict), reproducing that the failure is model generalization rather than
a broken runtime. Disk audit found no real `.partial` acquisition file or active transfer; E32
occupies 137 GB and the LaCie volume retains 514 GiB free.

R1b was then connected to the local web demo under a new, precommitted non-promotion contract. The
API's `demo` profile loads E20 and E26, and optionally R1b only when an explicit data root is passed.
R1b reuses E26's already-loaded, pinned CF-ViT model/processor but independently verifies the R1b
head, selection and CF weight hashes. Its typed payload is permanently `research_only=true` and
`affects_decision=false`; load or inference failure removes only the optional card and is exposed
in health. It cannot vote in E26's OR rule, readiness or the canonical registry.

The page was simplified to one upload and one analysis action. Results now appear in honest order:
E26 measured decision, optional R1b experimental second opinion, then expandable E20 technical
baseline. The R1b card prints its raw score/threshold as non-probability plus the measured 40.0%
IPN worst-device and 68.57% owner-gallery false-positive warnings. Legacy method selectors were
removed from the lightweight browser flow while their API compatibility remains. Responsive,
keyboard/touch focus and reduced-motion behavior were retained; a project-specific social card was
added without changing inference.

The real local end-to-end run used the LaCie R1b artifact and one owner still. E26 returned
`insufficient` (CF logit -8.6586), R1b returned 0.3132 at 0.1259 (`ai_signal`) and E20 returned
0.9988 at 0.9895. This deliberately visible disagreement demonstrates why R1b is diagnostic rather
than a decision vote. Focused verification passed 20 Python tests, production web build, six web
tests, TypeScript and ESLint. Final regression then passed all 207 Python tests, the dependency
graph and canonical artifact registry. The local API/web processes were stopped after the smoke;
no model endpoint or external disk was published.

The demo hierarchy was then corrected after the primary R1b answer proved too easy to confuse with
the older E26/E20 layers. R1b is now the only primary result card; E26, E20, artifact identity and
external false-positive measurements are behind one collapsed technical-details control. Its bar
shows the frozen threshold and the result sentence now says exactly how many percentage points the
raw score sits above or below it. The percentage is still explicitly a model signal, not a
calibrated probability. A local owner-still request rendered 31.3% versus the 12.6% threshold.
This changed only information hierarchy: R1b stays research-only and non-voting, and no threshold,
artifact, API decision rule or external-development result was altered. Final verification passed
207 Python tests, the production web build and all six web tests, TypeScript, ESLint, `pip check`
and the six-entry canonical artifact registry.

A full result review then separated R1b's representation from its operating threshold. Without
writing a new artifact or changing serving, the unchanged model was rescored on the already
consumed IPN and owner DEVELOPMENT populations and its frozen internal AI CAL features. The
post-hoc frontier revealed a potentially useful conservative region: threshold 0.863312 would have
reduced owner FP from 68.57% to 20.0% and IPN worst-device FP from 40.0% to 15.0%, while internal
current-family macro/worst recall remained 90.01%/80.0% (91.00% across all six AI sources). At
0.95, owner FP was 9.52%, IPN worst 7.5% and internal current-family macro/worst was 83.28%/65.0%
(85.13% across all six). These are diagnostic, test-derived numbers—not a repair—and neither
threshold may become a candidate.

The next plan was therefore narrowed from a broad architecture sweep to E32/R1c threshold-first
recovery. It will preserve the R1b backbone/head and select a new conservative threshold only on a
new, provenance-complete multi-pipeline calibration population, then pass consumed DEVELOPMENT and
one untouched real+modern-AI final. If clean transfer fails, the next controlled change is a compact
paired-content/frequency-aligned dataset and source-held-out CF adapter, following B-Free/DDA; new
spectral/global architectures and ensembles remain later gates. This ordering reflects the local
evidence and 2025–2026 Community Forensics, B-Free, DDA, SPAI, GAPL, NTIRE and GlobalForge results,
while avoiding another large blind download.

The first GitHub CI run on `main` exposed a portable-environment defect rather than a model or web
regression. Run `33070433088` passed the web job but could not collect Python tests: the workflow
installed only serving dependencies and exposed `ml/src`, while the full suite also imports the
repository-owned `ml/experiments` package and `pyarrow`. Pytest configuration now owns both import
roots, and CI installs the existing `experiments` and `test` extras declared in `ml/pyproject.toml`
instead of duplicating an incomplete dependency list. Verification without a caller-provided
`PYTHONPATH` passed all 207 Python tests; compileall, `pip check`, the six-entry artifact registry,
ESLint, TypeScript, the production web build and all six web tests also passed. No model artifact,
threshold, dataset, inference rule or measured result changed.

Replacement GitHub Actions run `33070928471` then completed green on the pushed `main` commit:
both the `web` and clean-environment `python` jobs passed. Only after that evidence existed, GitHub
branch protection was enabled for `main`: both checks are required with strict up-to-date status,
force-push and branch deletion are disabled, and administrator enforcement remains off so the
repository owner retains recovery access. This closes the first shared-repository CI failure and
replaces the unprotected-main warning with an enforceable test boundary.

Repository simplification was then preregistered before moving active code. A dependency/read-only
audit separated the live product circuit, reproducible research and frozen history. It also showed
that the apparent 5.1 GB `ml/` size was ignored local datasets/artifacts rather than tracked source
bloat, so no scientific byte, model, evidence receipt or append-only record was deleted.

The web circuit was rewired without changing output: `app/page.tsx` now owns only file/request
lifecycle, while four result-only components live together in `app/result-panels.tsx`; response
validation remains in `analysis-contract.ts`. The page fell from 411 to 224 lines, and 38 lines of
unowned CSS for the retired method picker, tile overlay, legacy cards and probability meters were
removed. The obsolete Claude launcher—which silently acknowledged B-Free's restricted licence—an
empty Next configuration and three unreferenced starter SVGs were deleted. Sites/Vinext hosting,
worker, PostCSS, favicon/social card and lockfile paths were deliberately preserved. README and the
experiment index now distinguish served code from E20–E32 reproducibility and frozen archives.

No dataset, model or API image was downloaded, and no threshold, model artifact, API field,
decision rule or measured result changed. Local verification passed 207 Python tests (one upstream
Starlette/httpx deprecation warning), compileall, `pip check`, all six registered artifact hashes,
ESLint, TypeScript, the production Sites/Vinext build and all six web tests. Generated pytest,
Python bytecode, Vinext and `dist` caches were removed after validation; they are rebuildable and
contain no project evidence.

Pushed commit `700e49c` then passed protected GitHub Actions run `33073029567`: the clean `web`
job completed in 37 seconds and the clean `python` job in 2 minutes 7 seconds. The repository
simplification phases S0–S3 are therefore closed. S4 remains intentionally open and stops before
network image bytes: it will freeze R1c CAL/LOCKED metadata, allocation, licence and acquisition
receipts before a later explicitly authorized transfer.

The user then authorized the next data/evaluation slice and a narrowly scoped Desktop cleanup.
A read-only inventory distinguished PixelProof material from personal forms, screenshots,
academic files and unrelated EOE/rork projects. Twenty-one proven PixelProof items were moved—no
copy was deleted—into `~/Desktop/PixelProof Workspace`: twelve legacy dataset directories plus
`archive.zip` under `Legacy Datasets`, the owner gallery/empty AI staging directory/verified
ChatGPT sample under `Samples`, and five closed project documents under `Documents`. The active Git
checkout stayed at its existing path. The open `PixelProof_Sunum.pptx` and its PowerPoint lock file
were deliberately left on the Desktop until the application is closed. Eight live legacy path
defaults were redirected through the consolidated root; syntax compilation and exact destination
existence checks passed.

A primary-source benchmark audit found no honest universal company-standard “pass score.” NIST
GenAI Image-D is the strongest future blind authority, but requires registration/data terms,
forbids test inspection/tuning, reports ROC-AUC/EER/TPR@FPR/Brier metrics and explicitly cannot be
presented as NIST endorsement. NTIRE 2026 supplies the most current public competitive reference
(42 generators and real-world transformations), but its public validation card has no declared
dataset licence, so PixelProof downloaded none of its 3.99 GB image payload. The project instead
selected ICCV 2025 RRDataset: official Zenodo record `14963880`, CC BY 4.0, original plus repeated
internet transmission and physical re-digitization conditions. The paper's best reported detector
overall accuracy is 89.59%, a comparison point rather than a certification cutoff.

Before any RRDataset archive byte, E33 froze both official Zenodo assets, URLs, roles, exact sizes
and MD5 values. The 2,163,176,547-byte original train/validation archive is the only possible new
R1c calibration source; the 20,117,869,400-byte test archive is locked until the threshold-only
candidate is frozen. The detailed selection receipt is 1,166 B / SHA-256
`ad6fc31f...3519`; compact evidence is `evidence/e33_rrdataset_acquisition.json`. Ten focused data
contract/acquisition tests passed and status confirmed zero archive/partial bytes at the freeze.

E33's measurement and archive-safety code was then implemented before opening or scoring a
production member. The shared metric layer enforces explicit `0=REAL, 1=AI`, preserves failed rows,
and reports ROC-AUC, EER, TPR@FPR, separate target/non-target Brier diagnostics, thresholded
confusion/balanced accuracy and per-source/condition rates. Its CAL-only selector chooses the
lowest threshold satisfying frozen macro/worst authentic false-positive budgets and rejects it if
AI family floors fail. The RR acquisition tool now rejects traversal, links/devices, duplicate
members, undeclared class paths, implausible files and excessive expansion before atomic
validation-only extraction. Nine focused metric/acquisition tests pass. No production image was
scored and the locked test archive remained unopened at this checkpoint.

The R1c-T production path was also committed before its manifest or first score. It derives the
seven declared RRDataset scenario groups from strict filenames, preserves explicit class labels,
records that scenarios are not camera-pipeline identities, and freezes a role-only manifest before
loading R1b. Batch inference falls back recursively to isolate a bad file rather than dropping a
whole batch. The single-run score receipt binds the manifest, unchanged R1b contract and score-file
hash; the candidate freezer can change only the threshold and refuses to rerun over an existing
score/candidate. Fifteen focused archive/metric/manifest tests pass. DEVELOPMENT and the locked RR
test remained unopened.

The licensed RR calibration transfer then completed resumably and reproduced its published byte
count and MD5. Archive inventory passed 3,000 images with the official 1,250+1,250 train and
250+250 validation class counts; validation-only extraction produced 500 logical images. macOS
created exFAT AppleDouble sidecars, which the manifest now rejects by name and cross-checks against
the extraction receipt so they cannot silently double the sample count.

Pre-score filename inspection corrected one planned assumption without reading a model output:
the 250 AI validation files expose seven scenario prefixes, but every authentic filename is only
`real_*`. RRDataset therefore cannot support a camera-pipeline or even per-site calibration claim.
R1c CAL was honestly frozen as 250 pooled REAL plus 250 AI across seven scenario groups, with
minimum reportable group size 20; IPN per-device and owner-gallery DEVELOPMENT remain mandatory
transfer gates. The unscored detailed manifest is 124,960 B / SHA-256 `5d575a08...b521`, and compact
evidence is `evidence/e33_r1c_cal_manifest.json`. Sixteen focused tests passed before the first
production score.

R1c-T then scored all 500/500 frozen RR validation rows with zero decode/inference failure and the
unchanged R1b artifact. The clean result rejects the threshold-only hypothesis: ROC-AUC 0.80728,
EER 0.276 and TPR@FPR=10% 0.52. R1b's old 0.125935 cut retains 96.4% AI recall but falsely flags
82.8% of RR reals. The first pre-specified REAL-safe cut is 0.998400 at exactly 10.0% REAL FP; it
falls to 52.0% pooled AI recall, 60.52% scenario-macro and 26.88% worst-scenario recall. Both the
working AUC tier and 80%/60% AI gates fail. Detailed rejected candidate SHA is
`b521edbc...7538`; compact evidence is `evidence/e33_r1c_threshold.json`.

Because CAL failed, IPN/owner DEVELOPMENT was not reopened, no threshold was promoted, and the
20.12 GB RR locked test was neither downloaded nor inspected. Research then identified official
NeurIPS 2025 DDA-COCO as the next causally relevant input rather than another volume-only dataset:
Apache-2.0, 4,301,452,066 B, revision `8c9330a3...68fb`, Xet SHA-256 `8cd60077...9c24`, with
MS-COCO reals and semantically/frequency-aligned VAE reconstructions. E34 preregisters it as a
pair-preserving TRAIN/CAL source; using it for fitting permanently forfeits a DDA benchmark claim.

E34 acquisition code was implemented and tested before transfer. It hard-binds the official
revision/licence/file size/Xet SHA, preserves resumable partial bytes and a 100 GiB disk floor, and
adds ZIP traversal/symlink/encryption/duplicate/CRC/expansion gates. Live metadata freeze produced
a 916 B detailed receipt / SHA-256 `f0bc21a7...5184` and compact
`evidence/e34_dda_acquisition.json`; status confirmed zero complete/partial archive bytes. Sixteen
focused E33/E34 metric/manifest/acquisition tests passed before download authorization was used.

The first DDA single-stream transfer was intentionally stopped without deleting its verified-size
prefix after throughput settled near 2–3 MB/s. Before resuming, E34 gained the same fail-closed
parallel recovery pattern proven on CSAFE: split only the exact remaining interval, require HTTP
206 and exact `Content-Range`/length for each resumable part, assemble prefix plus ordered ranges
into a new file, verify whole-file size/SHA-256, atomically promote, then remove temporary pieces.
No DDA archive member was opened and the preserved prefix could not be mistaken for a completed
archive.

A follow-up read of the official DDA repository corrected the E34 role before any member was
opened: DDA-COCO is explicitly an evaluation benchmark. The actual official training release is a
~112.97 GB split ZIP, so using the 4.30 GB benchmark as a convenient training subset would be test
contamination. PixelProof therefore keeps DDA-COCO locked, defers the full training release to
home internet, and preregisters the official Apache-2.0 DDA checkpoint as the next compact
candidate (1,255,621,296 B, revision `4390d902...16c`, SHA-256 `b27a31d3...e3e`). This correction
changes no measured result and preserves the already-downloaded partial only as unopened benchmark
bytes.

The official DDA checkpoint metadata was then frozen independently before transfer: exact model
repository/revision, Apache-2.0 licence, 1,255,621,296-byte `DDA_ckpt.pth` and Xet SHA-256. The
detailed E35 selection is 962 B / SHA-256 `7bdbe886...3fd9`; compact evidence is
`evidence/e35_dda_model_acquisition.json`. Seven focused DDA archive/model contract tests passed,
and status confirmed zero checkpoint bytes while the unopened benchmark transfer continued.

The unopened DDA-COCO range transfer was then paused without deleting any part so the smaller
official checkpoint could take network priority. A receipt-bound resumable checkpoint command was
committed first; it preserves a 100 GiB floor and promotes only the exact 1,255,621,296-byte file
after whole-file SHA-256 verification. Benchmark transfer state remains recoverable.

The official DDA checkpoint transfer completed and reproduced its exact Xet SHA-256. Before any
production image score, PixelProof implemented a minimal Apache-attributed adapter: offline timm
DINOv2-L/14 architecture, the official rank-8 LoRA target layout, strict `weights_only` checkpoint
loading, published RGB center-crop-336/normalization, sigmoid score direction and untouched 0.5
cut. The full 537-tensor state loaded strictly; the only compatibility shim retains torch.hub's
unused `mask_token` parameter that timm omits. Nine focused LoRA/preprocessing/acquisition tests
passed. No RR/IPN/owner or DDA-COCO member was scored at this checkpoint.

The DDA DEVELOPMENT runner was then frozen before its first image. It binds the 500-row consumed
RR manifest, 960-row IPN realization and exact 210-still owner identity; scores the published 0.5
cut only; stores a hashed local row stream; and gates RR AUC/REAL FP/scenario recall plus IPN
worst-device and owner FP simultaneously. No threshold-fit, retry arm or DDA-COCO read path exists.

The first invocation stopped before model loading because the owner folder now held 211 supported
stills rather than the frozen 210. A model-free identity comparison proved the sole addition is
`WhatsApp Image 2026-08-25 at 17.14.51.jpeg` (206,418 B, SHA-256 `e04755bf...57e3`); removing only
that row exactly reproduces protected identity `390e3c21...ac09`. The file remains untouched and
unscored as a possible future reserve. The runner was amended to accept only the exact old set plus
this exact optional reserve—any other membership change still fails closed.

### Desktop maintenance — PixelProof material isolated without touching other work

The Desktop inventory was classified conservatively by exact PixelProof names and prior documented
paths. Twenty-one proven project items were moved—not copied or deleted—under `PixelProof Workspace`:
13 retired dataset/artifact roots in `Legacy Datasets`, three sample/gallery items in `Samples`, and
five report/presentation/reference files in `Documents`. The active Git repository remains at its
original path. Eight live/archive path defaults now resolve the new legacy root through
`LEGACY_DATA_ROOT` instead of relying on Desktop clutter. Unrelated academic, EOE, Rork, Erasmus,
form and screenshot files were untouched. `PixelProof_Sunum.pptx` and its Office lock file remain on
Desktop because Microsoft PowerPoint has the deck open; moving an open document was intentionally
refused.

### E35 — DDA works as a ranker but the published cut fails real-camera transfer

The first DDA scoring attempt produced no scientific output because the offline timm DINOv2-L
adapter retained a fixed 518px assertion while the official DDA preprocessing supplies a 336px
center crop. PixelProof did not label that runtime mismatch as a model result. Enabling timm's
dynamic positional-grid interpolation reproduces torch.hub DINOv2 behaviour; three focused tests
and a real one-image inference passed, and the compatibility change was committed before the only
production run.

That one run scored all 1,670 declared rows without failure. On 250 RR REAL + 250 RR AI, official
DDA is strong: AUC 0.978192, EER 0.08, TPR@FPR10 0.92, balanced accuracy 92.4%, REAL FP 6.4%, AI
recall 91.2% and worst scenario recall 81.72%. The same untouched 0.5 cut, however, falsely marks
36.25% of the worst IPN phone and 34.76% of the owner gallery as AI. The preregistered DEVELOPMENT
gate therefore fails; DDA is not added to serving and the almost-complete DDA-COCO benchmark stays
unopened.

A separately recorded post-hoc curve explains the next move without laundering the failure. The
first observed boundary satisfying all consumed real gates is 0.901156, retaining 82.4% RR AI
recall, but it was discovered from RR/IPN/owner outcomes and is permanently barred from deployment.
E36 will test the hypothesis once on a newly acquired, role-separated CAL: >=6 unseen native camera
pipelines and >=6 pinned modern generator families. Only a clean CAL pass can freeze a threshold,
complete/open DDA-COCO and unlock a one-shot unseen FINAL. A failure opens paired-data training at
home internet; it does not justify another arbitrary ensemble or benchmark-derived threshold.

The slice closed with 236 Python tests, compileall, dependency integrity, the six-entry canonical
artifact registry, production web build plus six web tests, TypeScript and ESLint all passing. The
only warning is an upstream Starlette/httpx deprecation already present before E35. The result,
post-hoc diagnostic, dataset ledger and next-data plan are committed together so the internship
archive can reproduce both the promising RR result and the authentic-transfer rejection.

### E36-A planning — balance is enforced by disjoint sources, not by moving one threshold blindly

The next-stage source audit rejected two easy-looking shortcuts before bytes. VISION/FODB/S21/
iPhone14 are already training-side sources and cannot become new evidence by renaming them. Zenodo
SCIMD-17 is compact and licensed but explicitly resizes every phone image to 224×224, so it does not
answer the native-gallery false-positive problem. The selected 2026 Zenodo SNS dataset instead
separates nine device archives and normal/QQ/Weibo parent views. Five previously unseen phone
archives enter CAL; four author-held-out device archives remain inaccessible FINAL.

For AI balance, the pinned Apache-2.0 Qwen Image Bench contains 18 current generator cells with
1,000 aligned prompts each. Six families/prompt 101–200 are assigned to CAL and six completely
different families/prompt 1–40 to FINAL before download. Threshold selection will weight device and
generator macros equally and must satisfy authentic FP and AI recall floors together. Thus E36
cannot appear to fix real photos by predicting REAL for everything, and the final set cannot tune
the candidate it evaluates.

The fail-closed acquisition implementation was committed before live metadata. Four focused tests
cover source identity/licence/checksum drift, CAL/FINAL family disjointness, exact selection counts
and hostile ZIP paths. Live freeze then bound 600 CAL AI rows (468,420,944 B), 240 locked FINAL AI
rows (311,236,195 B), five CAL REAL archives (2,052,606,020 B) and four locked FINAL REAL archives
(2,038,841,380 B). Detailed selection SHA-256 is `01eec03e...2dcc`; no image byte moved. The old
unscored Qwen scout was superseded by a recorded role amendment before any E36 score.

The resumable CAL transfer completed all five REAL archives and 600 AI blobs with exact published
MD5/SHA-256 and zero FINAL bytes. CRC/safety inventory then exposed one model-free feasibility fact:
device 009 has 71 normal originals, below the planned 80 floor, while the other four provide at
least 100. Before extraction or model loading, E36 recorded an amendment to accept >=70 and retain
device 009 rather than dropping the difficult source or substituting an easier device. The 100-row
per-device cap and every performance gate remain unchanged.

The restart-safe realization pass then decoded and audited all 1,071 CAL parents: 471 native REAL
and 600 modern AI. Every AI family contributes 100; REAL contributes 100 from devices 001/002/003/
005 and all 71 available originals from device 009. Exact and perceptual overlap with prior passed
E32 realizations is zero, as are within-CAL exact and cross-label dHash collisions. Manifest SHA is
`4ed1b734...2e03`. It was committed unscored with zero FINAL bytes, preserving a clean boundary for
the one-threshold DDA calibration.

### E36 result — threshold-only repair is closed; FINAL remains clean

The preregistered DDA calibration scored every one of the 1,071 frozen CAL parents exactly once,
with no decode/inference failure and without downloading a FINAL byte. The published 0.5 cut was
already unbalanced: 16.61% REAL device-macro and 35.0% worst-device false positives while detecting
only 38.0% of the average AI family and 6.0% of FLUX.2 Max.

Moving to the first authentic-safe threshold did not solve the joint task. At `0.756332`, REAL
device-macro/worst FP reached the required 9.36%/20.0%, but AI family-macro/worst recall fell to
27.67%/1.0%. ROC-AUC was 0.58753, TPR@FPR10 0.285, EER 0.4267 and balanced accuracy 0.5895. This is
not an almost-pass and cannot be repaired by choosing another value from the same scores. E36 is
recorded as `calibration_failed`; no candidate artifact exists, DDA-COCO remains unopened and all
preregistered FINAL device/family cells remain untouched.

The next experiment changes one controlled component instead of contaminating FINAL. E36's rows
are role-amended to consumed adaptation data, and E37 will reuse the existing frozen DINOv2-S
representation plus old E32 TRAIN features to fit a lightweight source-balanced logistic head.
Every E36 parent must receive an out-of-fold score from a model that saw neither its REAL device nor
its AI family; one threshold must then pass both authentic FP and modern-AI recall gates. Only that
pass permits final refit and one-shot FINAL acquisition/evaluation. This preserves the project's
central lesson: fewer REAL accusations are not progress if the detector achieves them by missing
AI images.

### E37 result — ranking recovered; historical volume diluted current adaptation

E37 reused the existing 26,682-row DINOv2-S feature cache and extracted only 1,071 new embeddings.
Five complete source-held-out folds ensured every modern AI family and native REAL device was
scored by a head that had not seen that source. The fixed head produced ROC-AUC 0.94811,
TPR@FPR10 0.82 and EER 0.12976—clear evidence that the compact representation can separate the new
domain, unlike DDA on E36.

It still failed the frozen joint decision gate. The first REAL-safe threshold kept device-macro/
worst FP at 4.14%/19.72%, but AI family-macro/worst recall reached only 57.5%/42.0% and balanced
accuracy 0.7716. No artifact was written. A read-only DEVELOPMENT diagnostic then identified the
mechanism: 21,349 historical TRAIN rows numerically dominate 1,071 current adaptation rows. A
uniform adaptation multiplier plus stronger regularization improved the joint frontier; simple
DDA/DINO min, max, product and linear blends did not, so an ensemble was rejected.

Because that diagnostic inspected E36 outcomes, it is tuning, not new evidence. E38 therefore
freezes one simple setting (`C=0.0003`, every E36 row weight 100, every old TRAIN row weight 1),
retains all sources/rows and the same five held-source folds, and may earn only the right to face
the already-preregistered untouched FINAL. The FINAL result—not another E36 number—will decide
whether this candidate is successful.

### E38 checkpoint — the first jointly balanced development-selected candidate

The fixed E38 reproduction applied the same uniform weight to every modern adaptation row and did
not select sources or examples. All nine gates passed together: AUC 0.98062, TPR@FPR10 0.975, EER
0.06162, balanced accuracy 0.8955, 4.34% REAL device-macro and 19.72% worst-device FP, plus 82.5%
AI family-macro and 77.0% worst-family recall. Coverage was 1,071/1,071. This is the first stage in
the current recovery line that improves authentic safety without becoming AI-blind.

The final head was fitted on the unchanged 21,349 historical TRAIN rows plus all 1,071 modern
adaptation rows. Its 13,078-byte artifact SHA-256 is `fddbe475...4067`; its source-held-out OOF
threshold is frozen at `0.896190`. The archive explicitly retains the caveat that C and the uniform
weight were chosen from consumed DEVELOPMENT diagnostics. E38 has therefore earned a test, not a
success claim: only the four reserved REAL devices and six family-disjoint AI cells may now be
acquired and scored once.

The candidate-gated FINAL acquisition path was committed before transfer and refused to operate
unless both E38 evidence and artifact hashes matched. It then verified 2.04 GB of four reserved
REAL device archives and 311.24 MB of 240 family-disjoint AI blobs. CRC/safety inventory, native
extraction, decode and decontamination all completed before model scoring. The resulting unscored
manifest has 400 REAL (100 each from devices 004/006/007/008) and 240 AI (40 each from six held-out
families), zero overlap with prior roles and SHA-256 `cad71ff5...66e6`. This is now a one-use FINAL;
no row, candidate setting or threshold may change after the next command.

### E38 FINAL result — working conservative detector, strict gate not passed

The hash-bound scorer processed all 640 FINAL parents exactly once. E38 retained excellent
source-disjoint ranking: AUC 0.98185, TPR@FPR10 0.95 and EER 0.075. More importantly for the
original false-accusation defect, every one of the 400 unseen native camera originals was correctly
kept below the frozen AI threshold: 0% FP on all four devices.

The same conservative operating point missed too many synthetic images. It detected 162/240;
family recall was 72.5% FLUX.2 Pro, 77.5% GPT Image 1.5, 72.5% Hunyuan 3, 57.5% Imagen 4 Ultra,
50.0% Seedream 4.5 and 75.0% Nano Banana Pro. AI macro/worst recall was 67.5%/50.0% and balanced
accuracy 0.8375, so three frozen gates failed. The recorded state is `final_failed`, not a softened
pass. E38 is a genuinely functioning conservative prototype but is not promoted to the product
verdict.

A post-hoc curve was inspected only after sealing that result. At `0.270069`, the same frozen
scores would have satisfied every joint source gate—10% REAL macro, 17% worst REAL, 95% AI macro
and 90% worst AI—showing the remaining defect is OOF-to-refit threshold scale, not representation.
Because the value comes from FINAL it is permanently ineligible for E38. The 640 rows are now
consumed E39 development/calibration data; a corrected threshold may face only a newly sourced,
fully disjoint FINAL.

The E36–E38 slice closed with 251/251 Python tests, bytecode compilation and dependency integrity;
the six-entry canonical runtime artifact registry also passed. The production web build, all six
web tests, TypeScript and ESLint passed unchanged. One known upstream Starlette/httpx deprecation
warning remains unrelated to inference. Acquisition, manifests, experiment contracts, failed and
passed intermediate gates, the one-shot FINAL result and the E39 boundary are all committed and
pushed; no local scientific result is left undocumented.

### E39-A checkpoint — decision layer corrected, independent proof still required

Before changing a threshold, the complete 640-parent E38 FINAL was explicitly reclassified as
consumed `E39_CALIBRATION`. The role amendment binds the failed E38 evidence, original manifest,
score stream and byte-identical DINOv2-S logistic artifact. It forbids retraining, row selection,
crop/preprocessing changes, score reversal and any reinterpretation of the failed E38 claim.

One source-robust threshold was then selected from every consumed score under the unchanged joint
budgets. The frozen value is `0.27006945014`. On calibration it gives 10.0% REAL device-macro and
17.0% worst-device false positives, 95.0% AI family-macro and 90.0% worst-family recall, balanced
accuracy 0.925, AUC 0.98185, TPR@FPR10 0.95 and EER 0.075. All calibration gates pass with 640/640
coverage. These are development measurements, not a new model result.

The E39 candidate is a small decision-layer contract pointing to the unchanged artifact SHA-256
`fddbe475...4067`; candidate JSON SHA-256 is `7d497929...2cef`. Compact calibration evidence is
`evidence/e39_calibration.json` (`5f9a000e...4b0e`). No E39 FINAL image has been transferred or
scored. The next valid claim therefore requires a source-frozen, licensed and genuinely disjoint
FINAL collected without model access.

### E39-B checkpoint — new FINAL sources fixed before bytes

Primary-source research replaced the tentative CID2013 plan with a stronger native-phone source.
FloreView exposes individually downloadable camera-native outdoor JPEGs, explicit CC BY-SA 4.0
terms and device/EXIF metadata. Four source-new devices from four brands contribute 40 parents
each. This directly targets the real-photo false-accusation risk without reusing VISION, FODB,
CSAFE, IPN or owner-gallery evidence.

The AI side uses the 2026 AIGenImages2026 benchmark release and seven model-version cells absent
from all prior PixelProof roles, again capped at 40 each. The publisher's pinned 11.14 GB archive
is larger than the selected 280 rows but is the only authoritative package; its complete LFS hash
binds the future extraction. The combined 440-row allocation, licence receipts, catalog/archive
hashes and score-blind selection rules were committed while E39 FINAL still had zero local image
bytes. Acquisition may now proceed without loading the detector.

The live metadata preflight then reproduced both licence receipts, the FloreView catalog hash, the
exact four-device 160-URL selection, Hugging Face revision and the 11.14 GB archive LFS/Xet
identity. Detailed external selection SHA-256 is `4253497a...7be4`; compact evidence is
`evidence/e39_source_preflight.json`. The receipt still records zero downloaded FINAL image bytes.

Acquisition then completed resumably without model access. FloreView supplied all 160 individual
native JPEGs; the AIGenImages2026 archive matched its exact 11,138,511,098-byte size and published
SHA-256 `67c60427...c498`. Safety inventory found 10,905 tar members/10,879 regular files and
11,197,937,530 expanded bytes. Every selected family exceeded the 40-row floor (150–307 eligible),
and the deterministic ranking froze exactly 40 per family. Detailed inventory SHA-256 is
`bf6a6ecf...40dfb`; compact unscored evidence is `evidence/e39_ai_inventory.json`.

The selected 280 AI members were then extracted in one archive pass and audited alongside all 160
REAL parents. Every row decoded; every REAL parent was at least 2 MP and retained EXIF; all 280 AI
rows linked to publisher prompt metadata. Exact and dHash overlap with earlier roles were both
zero, as were exact/dHash duplicates inside E39 FINAL. The 412,914-byte detailed unscored manifest
SHA-256 is `1076df20...7306`; compact evidence is `evidence/e39_final_manifest.json`. This commit is
the irreversible boundary before the first and only E39 prediction.

### E39 FINAL result — modern AI transfer succeeds, native REAL safety collapses

The one-shot scorer completed all 440 parents without decode or inference failure. E39 detects
261/280 new AI images: 93.21% family-macro recall, 90.0% worst-family recall and 90–97.5% per-family
recall across all seven unseen generators. This is the strongest evidence so far that the frozen
DINO representation carries useful modern-generator signal.

The same threshold falsely accused 85/160 genuine phone photos. Device FP is 55.0% iPhone 13 mini,
55.0% DOOGEE S96 Pro, 60.0% Pixel 5 and 42.5% OnePlus 8T; macro is 53.13%. Overall balanced
accuracy is 0.7004, AUC 0.90033, TPR@FPR10 0.7714 and EER 0.1933. REAL safety, balanced accuracy,
TPR and EER gates fail. E39 is `final_failed` and is not promoted to the web/API verdict.

A post-hoc threshold diagnostic confirms this is not another scale-only error. The first REAL-safe
threshold (`0.622942`) gives 10%/15% REAL macro/worst FP but only 77.14% AI macro, 67.5% worst AI
and 0.8357 balanced accuracy; TPR@FPR10/EER remain threshold-independent failures. No joint
threshold exists. E39 now becomes consumed E40 adaptation data, and E40 will use source/content-
balanced replay rather than another threshold retry or ensemble.

### E40-A role boundary — E39 becomes development, never evidence again

Before E40 feature extraction or fitting, all 440 sealed E39 FINAL parents were reclassified as
consumed `E40_ADAPTATION_DEVELOPMENT`. The amendment binds the E39 manifest, complete score stream,
compact failed result and decision-contract hashes. E39 remains `final_failed`; neither these rows
nor unused rows from the same FloreView/AIGenImages2026 sources may be reused as independent FINAL.

The source-held-out protocol was also corrected before scores existed: every E39 source will be
predicted only by a head that did not train on that source. Frozen-embedding content clusters may
balance weights, but cannot define folds or remove rows—the four FloreView phones share a scene
catalog, making a simultaneous device-and-cluster holdout impossible without discarding evidence.
The next operation is therefore a model-free, all-row DINOv2-S feature cache.

### E40 fixed protocol — three heads, no hidden sweep

Before extracting E39 features or generating any E40 score, the complete implementation contract
was frozen. E40 will compare exactly three DINOv2-S logistic heads in simplest-first order:
uniform, source-balanced, then source-and-content-balanced. Every head uses the same seven E39
source-held-out folds, C=0.01 and all E36 development rows plus a deterministic 1,067-row E32 TRAIN
replay. The content arm fits 16 KMeans cells on each training fold only; held sources do not enter
clustering, and clusters influence weights rather than row selection.

Primary seed 42 chooses the first complete-gate pass. Its one REAL-safe threshold is then frozen
unchanged for seeds 41 and 43. DDA/CF-ViT features, ensembles, per-source thresholds, row removal
and further hyperparameter sweeps are prohibited. The implementation and six focused unit tests
pass; the complete Python suite passes 264/264 with dependency and bytecode checks. No E40 model
score, feature cache or new FINAL byte exists at this commit.

### E40-A feature checkpoint — all 440 parents represented exactly once

The unchanged, locally cached DINOv2-S backbone processed every consumed E39 parent successfully.
The resulting 440x384 float32 archive is 642,070 bytes, SHA-256 `ec050171...94e68`; all values are
finite and all 440 record IDs are unique. Counts remain 160 REAL and 280 AI, with fold sizes
80/80/80/80/40/40/40 under the frozen seven-source contract. No row was removed, clustered,
weighted, fitted or scored by a classifier. Compact evidence is `evidence/e40_features.json`.

### E40-B result — the simple head clears every development gate

All three preregistered source-held-out heads passed without retry. Uniform weighting reached AUC
0.99464, TPR@FPR10 0.98929, EER 0.03661 and balanced accuracy 0.94464. At its frozen threshold
0.170801, REAL device-macro/worst FP is 10.0%/17.5% and AI family-macro/worst recall is
98.93%/97.5%. Source-balanced was slightly stronger (AUC 0.99699), while source+content-balanced
also passed (AUC 0.99071); the fixed simplest-first rule correctly selected uniform rather than
optimizing after seeing scores.

Seeds 41, 42 and 43 then reproduced the selected head and unchanged threshold exactly, each passing
all nine gates. A 12,690-byte development draft was refit on the 1,067-row historical replay, all
1,071 E36 rows and all 440 E39 rows; SHA-256 `72b8d8cd...819c0`. It is explicitly not packaged or
served: E39 is consumed development, and E40-C local robustness plus a genuinely new FINAL are
still required. Detailed development evidence SHA-256 is `93f98f07...ebcc`.

### E40-C robustness protocol — frozen before reopening local pixels

The passed E40 draft is now bound to one non-tunable robustness gate before its first owner-gallery
or transport score. All 440 consumed E39 parents will be viewed natively, at JPEG q50 and after
75% resize plus q50; the latter two remain parent-linked derivatives, never 880 new observations.
Each derivative must retain fixed AUC/TPR/balanced/source floors and at least 80% decision agreement
within both classes. The unchanged threshold must also keep the hash-bound 210-photo owner gallery
at or below 20% false positives. Passing packages a research candidate only; it cannot serve or
claim generalization until a new FINAL. Two focused robustness tests pass before pixel access.

### E40-C result — transport passes, personal-gallery safety does not

All 1,320 parent-linked E39 views and all 210 declared owner-gallery stills scored once at the
unchanged 0.170801 threshold. JPEG q50 and resize75+q50 are stable: AUC 0.99951/0.99949, balanced
accuracy 0.96696/0.96071, REAL macro/worst FP 6.25%/15.0% and 7.5%/15.0%, AI macro/worst recall
99.64%/97.5% in both, and per-class decision agreement 98.13–100%.

The owner gallery instead produces 146/210 false AI decisions, **69.52% FP** and only 30.48% REAL
recall. This essentially reproduces the previous 68.57% warning: E40 learned the new FloreView phone
distribution but not the broader casual personal-photo manifold. E40 is therefore
`robustness_failed`; no research candidate was packaged, no product code changed and no FINAL data
was downloaded. Score stream SHA-256 is `a126e814...0e3b`; sealed report/evidence SHA-256 is
`74e23b66...a463`.

After sealing the failure, a diagnostic combined the 440 native E39 draft scores with the 210
gallery scores. Its first REAL-safe boundary is 0.619554: REAL macro/worst 4%/20%, AI macro/worst
92.14%/90%, AUC 0.97218, TPR@FPR10 0.90714, EER 0.10 and balanced accuracy 0.90396. This proves a
broad-real threshold candidate exists, but the number is contaminated and cannot rescue E40. It is
recorded only to justify an explicit E41 calibration-transfer candidate and a wholly new FINAL.

### E41 protocol — broad-real calibration without another fit

Before packaging, all 440 native E39 draft rows and all 210 owner-gallery rows were explicitly
assigned `E41_BROAD_REAL_CALIBRATION`; compressed derivatives remain robustness-only. The frozen
E41 operation changes only threshold 0.170801 -> 0.619554 on the existing E40 uniform head. Learned
scaler/logistic arrays receive their own numeric checksum before and after serialization. Any fit,
second threshold, row exclusion, product promotion or FINAL access is prohibited. Implementation,
two focused tests and the role/contract are committed before artifact creation.

### E41 candidate checkpoint — broad-real threshold frozen, head unchanged

The E41 packager produced a 13,064-byte research artifact, SHA-256 `9bcc021e...ab65`, at threshold
0.619554. The learned StandardScaler and logistic arrays are numerically identical to the E40 draft
under checksum `4211d8d8...f49f`; there was no fit, feature, crop or preprocessing change. The
consumed 650-row calibration frontier remains REAL macro/worst 4%/20%, AI macro/worst 92.14%/90%,
AUC 0.97218, TPR@FPR10 0.90714, EER 0.10 and balanced accuracy 0.90396.

State is `candidate_frozen_awaiting_independent_final`, not validated or served. No E41 FINAL image
has been selected, transferred or scored. Work stops at the planned data boundary: the next action
must first bind genuinely new real devices/sessions and generator/model versions, then acquire them
without model access. Existing E39, FloreView, AIGenImages2026 and owner-gallery bytes are forbidden.

Final checkpoint verification passes 268/268 Python tests, bytecode compilation and dependency
integrity. The unchanged web product passes its production build, all six tests, TypeScript and
ESLint. One known upstream Starlette/httpx deprecation warning remains non-functional. All E40/E41
role, method, result and stop decisions are recorded and pushed; the working product remains on its
previous validated served model until E41 earns independent evidence.

## 2026-08-28 — E41 external-proof route selected from current evidence

The new objective was stated plainly: achieve success, not merely add another experiment. The
entire E1–E41 chain and current primary literature were re-audited before any new dataset byte.
Three findings changed the order of work. First, the MAD 2026 ITW-SM study reports that a DINOv2-L
RINE variant reaches 0.9823 AUC on 10,000 real social-media images only when in-the-wild training,
texture-aware crops and realistic augmentations are combined; SPAI reaches 0.9810 with a distinct
spectral route. Second, NTIRE 2026's winner reaches 0.9723 robust AUC with millions of training
images, multiple DINOv3 backbones and hierarchical degradations—evidence that data and transport
coverage dominate a simple backbone swap. Third, a 2026 23-detector/12-dataset comparison finds no
universal winner; even its best released ensemble averages only 0.780 accuracy and current
commercial generators defeat most frozen detectors.

This evidence does not justify throwing away E41. Its learned head already separates seven recent
families well and its only change from E40 is the broad-real threshold. The shortest honest route
is therefore fixed: score the unchanged E41 candidate first on two open external surfaces. The
B-Free viral set supplies a difficult, parent-grouped web-propagation stress test; RRDataset's
still-unopened 20.12 GB CC BY 4.0 test archive supplies a larger clean/transmission/redigitization
robustness transfer. Neither may tune E41. ITW-SM remains the preferred stronger social-media final,
but its 3.57 GB release is manually gated, the machine has no Hugging Face identity and the user
has not yet personally accepted its non-commercial terms; no silent access request or download was
made.

The failure branch is also frozen before scores: if E41 misses, preserve both external tests and
open one E42 line combining DINOv2 global features, texture-rich multi-crop aggregation, symmetric
JPEG/WebP/resize/blur augmentation and source-held-out calibration. Do not train on failed external
test rows or sweep an ensemble after seeing them. Existing success gates remain unchanged, so the
project cannot create progress by lowering the bar. `PLAN.md` and `DATASETS.md` hold the complete
pre-byte contract and exact source receipts.

### E42 external acquisition method — committed before production transfer

The B-Free URL collector now fails closed on the pinned CSV hash/schema, unsafe paths, unknown or
cross-label parent events, invalid MD5/geometry and changed population counts. It maps labels only
through the explicit invariant REAL=0/FAKE=1, downloads each URL independently, verifies the
authors' per-row MD5 and decoded dimensions, and keeps every failed/dead URL visible in coverage.
All versions retain their 34 source-event parents so repost volume cannot masquerade as independent
sample size. Three focused synthetic tests and the pinned 1,111-row registry validation pass before
any production URL is opened. `evidence/e42_external_contract.json` binds the unchanged E41
artifact, B-Free registry, RR test receipt, ITW-SM access limitation and no-retuning rules.

### B-Free external bytes — every parent event survives, URL attrition stays visible

The committed collector opened all 1,111 pinned registry URLs without model access. It verified
811 rows / 162,894,149 bytes against the authors' exact MD5 and dimensions: 278 REAL and 533 FAKE.
There are 191 dead/request-failed URLs and 109 live URLs whose returned bytes no longer match the
published MD5; the latter were discarded instead of silently relabelled. Despite 72.9973% row
coverage, every one of the 17 REAL and 17 FAKE source events retains at least one valid web version,
so the effective 34-event parent structure remains complete. Detailed acquisition SHA-256 is
`e95f514...b221d`. No E41 score, embedding or threshold was accessed; decontamination and the
unscored parent manifest remain the next gate.

The first manifest command exposed a path-safety defect before scoring: `PIXELPROOF_DATA_ROOT` was
not set, so the verified B-Free bytes had landed under the ignored project data root and the prior
E32/E33/E36/E39 manifests appeared empty. The resulting zero-prior manifest was rejected and
preserved as `evidence/e42_bfree_manifest_rejected_wrong_root.json`; no model was loaded. All
verified bytes and the acquisition receipt were moved intact to the declared LaCie destination,
avoiding a redownload. The manifest code now hard-fails when it finds zero protected prior files,
with a regression test, before decontamination can be accepted.

The next dry manifest attempt found a second relocation edge before writing output: the acquisition
receipt correctly preserved original absolute paths, but the manifest reader followed those stale
paths after the byte-preserving move. It now derives every live path from the pinned safe relative
filename plus the configured data root; receipt paths remain provenance only. This correction also
has a focused relocation test and still precedes every model score.

With the corrected LaCie root, the production manifest found 14 protected earlier-role files and
screened all 811 verified B-Free children. There is no exact or dHash overlap with any prior role,
no cross-event duplicate group and therefore no hash-driven source exclusion. The frozen unscored
population remains 278 REAL +533 FAKE versions under 17+17 parent events. Detailed manifest
SHA-256 is `338a2f2...f37ca2`. E41 still has not been loaded; this closes the B-Free pre-score gate.

### B-Free score method — frozen before the candidate sees a pixel

The one-shot scorer is now bound to the E41 artifact SHA-256, its unchanged 0.619554 threshold and
the exact 811-row unscored-manifest SHA-256. It verifies every image hash during inference, records
every surviving URL version, but gives each of the 34 original source events equal decision weight
so a heavily reposted image cannot dominate the result. The pre-registered stress pass requires
parent-weighted balanced accuracy >=0.80 and both REAL and AI parent-weighted recall >=0.75;
10,000 event-level bootstrap draws expose the uncertainty of only 17 parents per class. The score
stream is write-once: an existing result blocks a retry. Two focused method tests and the complete
seven-test E42 external suite pass before the model is loaded. No B-Free score has yet been seen.

### E41 external stress result — perfect AI recall hides another real-photo collapse

The frozen E41 candidate scored all 811 manifest rows once with full inference coverage. Every AI
parent is detected, but the unchanged decision calls 81.59% of equally weighted REAL-parent web
versions AI: REAL parent recall is 18.41%, AI parent recall 100% and balanced accuracy 59.20%
(95% event bootstrap interval 52.26–67.75%). Event-mean AUC is only 0.76125, TPR@FPR10 0.35294 and
EER 0.35294. The pre-registered 0.80 balanced /0.75 per-class stress gate therefore fails on both
balanced accuracy and REAL safety. Version-weighted results tell the same story, so URL attrition
or parent weighting did not manufacture the diagnosis.

There is no threshold retry and E41 is not promoted. The external result confirms that the broad-
real threshold repaired a consumed gallery but did not repair the representation's dependence on
the authentic source domain. Because one mandatory gate already failed, the unopened 20.12 GB RR
test cannot make E41 pass and remains untouched for the eventual E42 winner. The frozen failure
opens only the preregistered E42 route: texture-aware multi-crop evidence, symmetric transport
augmentation and source-held-out calibration, selected without B-Free or RR-test tuning. Score
stream SHA-256 is `83783551...c33fc`; tracked evidence is
`evidence/e42_bfree_result.json`.

### E42 recovery design — representation and data diversity change together

The recovery line is now specific enough to execute and is frozen before RR train extraction or
E42 feature access. It binds 4,638 base-training parents from the existing E32 replay, E36 CAL and
RR official train split, plus 2,250 already-consumed source-held-out DEVELOPMENT parents spanning
E36/E39, 12 IPN devices and the owner gallery. B-Free and RR test are excluded from every adaptive
decision. This directly addresses the observed defect: previous global DINO heads could rank
their known AI families but treated unfamiliar authentic pipelines as evidence of generation.

Only two representations may compete. Both aggregate normalized CLS tokens from four DINOv2
blocks over a global view and two deterministic texture-rich native crops; one uses DINOv2-S and
one DINOv2-L. The large backbone will reuse its original frozen tensors already present in the
hash-pinned official DDA checkpoint, so no duplicate gigabyte download is justified. Each parent
gets clean plus one class-symmetric transport view for fitting, while DEVELOPMENT is measured on
clean and all four fixed transports. Source-held-out OOF selects one threshold; the smaller model
wins if both pass. Failure stops the ladder, while a pass permits exactly one unopened RR-test
transfer. Exact bindings and gates are machine-readable in `evidence/e42_fixed_contract.json`.

The first combined-manifest attempt stopped before writing a manifest or loading a model because
the provisional parent key was not unique. E36 deliberately reuses each of 100 prompt/content IDs
across six distinct generator families, so keys such as `qwen-bench:101` appeared six times even
though the image hashes differ. This is an identifier collision, not duplicate imagery. E42 now
qualifies those image-parent keys with the source family while retaining the original content ID;
a regression test proves two generators sharing a prompt cannot merge. The completed 2,500-row RR
train extraction remains valid and is reused; the failed manifest produced no accepted output.

The corrected-key rerun exposed a second pre-feature data fact: the declared 210-file owner-gallery
identity contains four exact duplicate pairs (`IMG_8335` through `IMG_8338`, each also saved with a
` 2` suffix). They are 206 unique image parents, not 210 independent observations. The manifest
again stopped before output/model access. E42 collapses each exact pair deterministically and
amends DEVELOPMENT from 2,250 file rows to 2,246 unique parents; the original 210-file identity is
still preserved as provenance. This reduces duplicate weighting instead of hiding it.

The third manifest pass completed before model access. RR official train contributes all 2,500
declared rows /1,860,689,134 decoded image bytes. The full E42 manifest binds 6,884 unique parents
under 63 sources: 4,638 TRAIN (2,335 REAL, 2,303 AI) and 2,246 consumed DEVELOPMENT (1,726 REAL,
520 AI). There is no cross-role exact SHA-256 or exact dHash group. The 3,737,406-byte detailed
manifest has SHA-256 `15124d93...3e238`; the RR receipt has SHA-256 `ba8f4ab1...41813`. This closes
the data gate. No feature extractor, classifier, B-Free row or RR-test byte participated.

### E42 feature method — executable before any real feature cache

The fixed extractor now implements the preregistered 20,506-view population: TRAIN receives clean
plus one hash-assigned transport; DEVELOPMENT receives clean plus JPEG, WebP, resize+JPEG and mild
blur. Every view becomes one global center crop plus two deterministic, preferably non-overlapping
highest-texture native crops after a 2048px safety cap. Normalized CLS tokens from four fixed DINO
blocks are reduced only by per-block crop mean and standard deviation. Four unit tests cover view
symmetry/cardinality, deterministic crop geometry, all transports and aggregation math. Synthetic
224px smoke inference confirms the small 4x384 and large 4x1024 intermediate-token contracts; no
project image, label outcome or classifier score was accessed. Feature code is committed before a
real E42 cache may be created.

The first DINOv2-S cache run was interrupted at 4,464/20,506 views before any output file existed.
Timing exposed that JPEG/WebP/blur was being computed on full 12–48 MP inputs and only then capped,
even though the frozen inference contract requires a 2048px safety cap before 224px crops. The cap
is now applied once before every transport, with a regression assertion for all five conditions.
This changes no declared view, crop, label or model feature; it removes discarded computation. No
partial feature archive/evidence exists, so the optimized run begins from a clean write-once state.

The capped rerun was also interrupted before output at 4,416 views after timing showed sequential
PIL preparation leaving the accelerator idle. Hash verification remains single-pass and ordered,
but the independent transport/crop preparation is now mapped over six worker threads before each
unchanged tensor batch. This is an execution-only correction: the same functions, arrays, ordering
and model inputs are used. A focused deterministic-feature test still passes and no partial cache
exists.

The optimized DINOv2-S run completed every planned row: 20,506 views from 6,884 parents, three
crops per view and four intermediate blocks, producing a 20,506x3,072 finite float32 matrix. The
compressed cache is 235,605,776 bytes / SHA-256 `452fec98...69ac5a`, bound to DINOv2-S weights
`04d27f34...20081` and manifest `15124d93...3e238`. No classifier or score was produced. The fixed
smallest-pass rule now short-circuits redundant computation: S is evaluated first, and DINOv2-L is
needed only if S misses a mandatory development gate because a passing L cannot replace a passing
S.

### E42 decision method — fixed before OOF scores

The consumed DEVELOPMENT population contains 34 whole source families. A deterministic greedy
assignment keeps each source intact across five folds (7/7/7/6/7 sources), balanced separately by
class. Each fold fits on all base TRAIN views plus only clean and hash-assigned transport views from
the other development sources, then scores all five conditions of the held sources exactly once.
The resulting OOF population is fixed at 11,230 rows; 2,246 clean rows alone select the first REAL-
safe threshold, while 8,984 transformed rows can only pass/fail that unchanged cut.

The head is fixed to StandardScaler + LogisticRegression C=0.01 with equal class mass and equal
source mass inside each class. Clean must pass all nine standing success checks; combined robust
views must additionally reach AUC 0.85 and balanced accuracy 0.80 with full coverage. Eight focused
tests plus a real-cache structural dry run pass before any classifier fit. A full S pass packages
S immediately; only an S miss authorizes the already-fixed L representation. B-Free and RR test
remain inaccessible to this decision.

### E42-S DEVELOPMENT result — the real-photo collapse is repaired on consumed sources

All five source-held-out folds completed with 11,230/11,230 scored views and no failure. At the
first REAL-safe clean OOF threshold 0.660046, DINOv2-S reaches AUC 0.99287, TPR@FPR10 0.98462, EER
0.04047 and balanced accuracy 0.95477. REAL macro FP is 1.23%; the weakest source is device_004 at
the exact 20% ceiling, while all 12 IPN devices are 0% and the deduplicated owner gallery is 3.88%.
AI macro/worst-family recall is 92.69%/75%. This is the first project head to satisfy every clean
gate while retaining high modern-AI recall on whole held-out sources.

The unchanged threshold also passes all transport checks. JPEG/WebP/resize+JPEG/blur combined AUC
is 0.99338, balanced accuracy 0.93923, REAL macro/worst FP 0.84%/13.5% and AI macro/worst recall
88.99%/68.13%; each condition stays above 0.992 AUC and 0.928 balanced accuracy. All 12 fixed checks
pass. The smallest-pass rule selects S without spending compute on L. One refit on all consumed
fit-eligible views produced the 87,977-byte research candidate SHA-256 `6768466a...9062e7` at the
same threshold. This is a major DEVELOPMENT success, not an external-final claim: B-Free was never
used for E42 decisions and RR test remains unopened. OOF stream SHA-256 is `0fbd15d5...dd32ff`;
tracked report is `evidence/e42_development_small.json`.

### E42-S external contract — candidate bound before RR test transfer

The one-shot external candidate is now fixed at artifact SHA-256 `6768466a...9062e7`, 87,977 bytes
and threshold 0.660046. Its next source is only Zenodo 14963880's CC BY 4.0 RR test archive, exactly
20,117,869,400 bytes /MD5 `13c3ff3d...cd4b`. Transfer may reveal archive structure for safe
inventory, but the model cannot open a test pixel until a decoded, parent/condition-aware unscored
manifest and overlap audit are frozen. Original images must pass the full internship gate; every
sufficient robust condition must retain AUC 0.85 and balanced accuracy 0.80 at full coverage.
Partial transfer, row removal, threshold repair and B-Free-informed tuning are forbidden. The
machine contract is `evidence/e42_rr_final_contract.json` and is committed before network bytes.

### E42 RR execution method — frozen while transfer is still incomplete

The official 20,117,869,400-byte archive transfer now runs as a resumable `.partial` file on LaCie;
an interrupted connection cannot create a completed receipt or expose the model to test pixels.
Before archive completion, the project added a fail-closed extractor that accepts only the three
official RR conditions and explicit REAL/AI paths, verifies extracted row and byte totals against
the tar inventory and records that extraction itself decoded/scored nothing.

The external evaluator is likewise executable before any result. It decodes and hashes every
extracted image, maps `transfer_`/`redigital_` filenames back to their original parent, preserves
the seven declared AI scenario families and pooled REAL limitation, and rejects duplicate
parent-condition rows, cross-label parents, exact cross-parent copies or exact/dHash overlap with
E42 development/training and the B-Free stress set. Only after that zero-score manifest exists may
a second contract bind its SHA-256 to the unchanged E42-S artifact and threshold. The scorer then
uses the same global-plus-two-texture-crop, four-intermediate-block feature path and reports
original, transfer and redigital separately. Original must pass the full project gate; both robust
conditions must pass AUC 0.85/balanced accuracy 0.80 with complete coverage. Ten focused tests pass.
No RR model score has been created at this checkpoint.

The first full inventory stopped safely before producing a receipt because the published archive
root is `RRDataset_final`, whereas the archive filename/initial acquisition assumption used
`RRDataset_test`. A read-only member sample also established that the actual archive uses
`{condition}/{real,ai}`, despite the repository README documenting `real_images/ai_images`. The
inventory/extractor contract is narrowed to the observed archive layout and recommitted before a
second inventory. This is a packaging correction only: no image was decoded and E42 stayed closed.

The corrected inventory and extraction then completed. The pinned 20,117,869,400-byte archive
matches MD5 `13c3ff3d...cd4b` and contains 50,999 declared images /20,354,797,721 expanded bytes:
8,500 in each condition/class cell except redigital REAL at 8,499. This differs from the paper's
description of 10,000 REAL +10,000 AI parents per condition and is preserved as a public-package
limitation, not silently filled or resampled. Compact acquisition evidence is
`evidence/e42_rr_acquisition.json`.

The first complete decode/hash pass also stopped before writing a manifest. It found 35 same-label
cross-parent exact duplicate components, 13 original REAL files exactly overlapping protected E42
roles, and one AI parent whose original/transfer dHash matches protected data. This is precisely why
the manifest precedes model access. The corrected decontamination rule excludes a protected parent
across all available conditions, propagates exclusion through exact-copy components and retains
only the lexical canonical parent in otherwise clean same-label exact components. Cross-label exact
copies, repeated conditions or label-crossing parents remain fatal. Official-to-selected coverage
and every excluded parent/reason remain in the detailed manifest. No threshold/model change and no
RR score occurred.

The revised write-once manifest passed. Whole-parent removal excludes 47 contaminated or redundant
parents /141 derivative rows, leaving 50,858 images from 16,953 parents and 20,341,312,914 image
bytes: original 8,454 REAL +8,499 AI, transfer 8,454 +8,499 and redigital 8,453 +8,499. Official
row coverage after decontamination is 99.7235%. Exactly 16,999 official parents have all three
conditions and one REAL parent lacks only redigital before exclusion. The 31,091,691-byte detailed
unscored manifest SHA-256 is `b2d815af...30c98`; its compact tracked receipt reports zero scores.

A second immutable score contract now binds that manifest to E42-S artifact `6768466a...9062e7`,
threshold 0.660046 and exactly 50,858 declared rows. Contract SHA-256 is `a5387eb9...de658`.
Threshold change, row removal, test-informed fitting and retry after a completed stream remain
forbidden. The model has still not accessed an RR pixel at this checkpoint.

### E42 RR one-shot result — useful ranking, final gate failed

The frozen scorer completed its first and only stream over all 50,858 declared rows with 100%
inference coverage. E42-S did not pass the external gate. On original images it achieved AUC
0.94448, TPR@FPR10 0.85139, EER 0.12434 and AI recall 93.54%, but the unchanged 0.660046 threshold
marked 2,052/8,454 authentic images as AI: REAL FP 24.27% and balanced accuracy 0.84634. Those two
REAL checks and the 0.85 balanced-accuracy check fail. Transfer remains above its working gate at
AUC 0.92582 /balanced accuracy 0.83993. Redigital retains AUC 0.85629 but falls to balanced accuracy
0.78756, so that condition also fails. The 14,572,649-byte score stream SHA-256 is
`c065957e...68434`; the identical local/tracked result is 17,498 bytes /SHA-256
`516c6d92...6252e`.

A read-only post-hoc threshold audit shows why E42 cannot be repaired and rerun. Original's best
balanced threshold would be 0.92704 (balanced 0.87686, REAL FP 9.25%, AI recall 84.62%), but
redigital's own best threshold reaches only 0.78943 balanced accuracy. No single threshold satisfies
the declared original, transfer and redigital gates. This is therefore a representation/transport
generalization miss, not merely the old low-threshold mistake. E42 stays research-only and is not
promoted into the API or web demo.

RR labels and scores are now consumed. Any E43 work must declare RR as DEVELOPMENT, change the
representation or realistic redigitalization training coverage, and use a genuinely untouched
final such as manually authorized ITW-SM or registered NIST Image-D. Calling a retuned RR result a
new final would invalidate the project record and is prohibited.

### E43 final source decision — ITW-SM selected, access pending

The next untouched final is now explicitly selected as **ITW-SM (In The Wild – Social Media)**,
the 2026 MAD benchmark distributed through the gated Hugging Face repository
[`dkarageo/itw-sm`](https://huggingface.co/datasets/dkarageo/itw-sm). It contains 10,000 balanced
examples (5,000 REAL /5,000 AI) collected from Facebook, Instagram, LinkedIn and X while preserving
native resolution, platform compression and real social-media semantics. This is materially closer
to the project's intended user input than another clean generator benchmark.

The choice directly follows the E42 failure mechanism. RR showed useful ranking but excessive REAL
false positives and a redigital robustness ceiling that no threshold could repair. ITW-SM is
therefore reserved as a never-trained-on, never-calibrated-on external final that can test whether
E43 transfers to uncontrolled social-media imagery. A pass will be strong independent evidence,
not a universal authenticity certificate or NIST approval.

Access is individual, non-commercial-research-only and gated by explicit terms: no redistribution,
no identity/re-identification attempts, respect third-party rights and cite the associated paper.
The student access form has been prepared honestly for an internship research project. At this
checkpoint access approval, authenticated download, local bytes, file counts and hashes are all
still pending; **zero ITW-SM image bytes have been downloaded or exposed to a model**. The dataset
must remain unopened until the E43 artifact, threshold, manifest rules and one-shot score contract
are frozen.

On 2026-09-02 the student accepted the ITW-SM terms and completed local OAuth authentication. The
authenticated repository inventory is now frozen before image transfer at commit
`3060094fb576669927134193de3f517d7e64af86`: 10,004 files /3,573,691,324 bytes, including exactly
5,000 REAL and 5,000 AI images. A fail-closed, revision-pinned LaCie downloader and focused tests
were added before acquisition. It preserves partial Hugging Face state across connection loss,
requires 100 GiB free after the expected payload, rejects missing/extra/wrong-sized files and emits
no receipt until the complete local snapshot matches the remote inventory. No model score is part
of acquisition.

**Access-state correction.** The first frozen content request established that successful OAuth and
visibility of the 10,004-file metadata inventory do not equal dataset approval. Hugging Face
returned HTTP 403 with `awaiting manual author review`; the earlier wording “access accepted” meant
that the student submitted/accepted the terms, not that the authors had granted file access. The
attempt stopped without an image payload or receipt. Only approximately 6.3 MB of resumable local-
dir cache/tree/lock metadata exists on LaCie. The acquisition method now probes one non-image file
before scheduling the image pool, and E43 final remains blocked until the authors approve the
individual request. No retry is useful while that external state is unchanged.

### E43 work continues locally — data alignment before a larger backbone

The manual ITW-SM wait does not block DEVELOPMENT. A record audit rejected the tempting
DINO-plus-forensic sidecar: E8/E9 and E31 already measured real complementarity but insufficient AI
gain and increased authentic false positives. E42's RR post-hoc ceiling also rules out another
threshold-only repair. The next smallest scientific change is therefore an E43-S head learned on
real transport/redigital examples while preserving E42's successful multi-crop intermediate DINO
representation.

Before selecting a row, E43 fixes a score-blind RR population of 1,960 complete REAL parents and
1,960 complete AI parents balanced as 280 from each of seven scenarios. Separate deterministic
hashes select parents and split every stratum 50/25/25 into TRAIN/CAL/DEVELOPMENT, keeping each
parent's three conditions together. TRAIN changes the 3,072-dimensional logistic decision boundary;
CAL alone selects a REAL-safe threshold; DEVELOPMENT tests the frozen local candidate. Only an S
failure can unlock the already-local DINOv2-L arm. Passing this consumed benchmark creates a better
candidate, never a final claim; ITW-SM remains unopened and mandatory.

The E43 RR role freeze then passed on the real manifest without loading the old score stream. It
selected the declared 3,920 complete parents /11,760 linked condition rows and produced exact
TRAIN/CAL/DEVELOPMENT parent counts 1,960/980/980, with both labels balanced in each role. Every
condition has 3,920 rows. The 7,645,807-byte detailed manifest SHA-256 is
`29dd9b56...4b16`; tracked evidence explicitly reports zero score files read and zero model scores.
This closes the data-role gate and permits E43-S feature extraction, but not fitting or final access
before their own contracts.

The E43-S RR feature pass then covered all 11,760 rows without a decode/hash failure. It reused the
unchanged DINOv2-S weights and E42 global-plus-two-texture-crop, four-intermediate-block mean+std
representation, producing shape 11,760x3,072. The compressed 134,777,581-byte archive SHA-256 is
`fdc5d4c8...a4aa4`. No head, threshold or score exists at this checkpoint; DEVELOPMENT has not been
used. Compact evidence is `evidence/e43_rr_features_small.json`.

The next committed method fitted the single E43-S head before opening DEVELOPMENT. It combined
13,768 consumed E42 fit views with 5,880 RR TRAIN triplet views and used fixed `C=0.01`
source/parent-balanced logistic learning. On the permitted 980 RR CAL originals, the frozen
REAL-safe threshold is `0.8712875247`; calibration AUC is 0.97369, balanced accuracy 0.92551,
REAL FP 10.0% and AI recall 95.10%. The 87,916-byte candidate SHA-256 is
`a3aec445...47390`. This promising calibration result is not a pass: the candidate and threshold
are now immutable for the separate consumed DEVELOPMENT run, and both RR DEVELOPMENT and ITW-SM
still have zero scores at this checkpoint.

The separately committed one-shot evaluator then opened the 2,940 RR DEVELOPMENT rows exactly
once. E43-S passed every frozen local gate: original AUC/balanced accuracy are 0.98194/0.93265 with
7.96% REAL FP and 94.49% AI recall; transfer is 0.97826/0.92755; redigital is
0.95186/0.88673. This repairs the old E42 redigital balanced-accuracy failure from 0.78756 to
0.88673 while also reducing original REAL FP from 24.27% to 7.96% on the selected score-blind
development slice. The `everyday_life` redigital subgroup remains a visible weakness at 48.57%
recall. Historical clean/robust regression checks pass, but are explicitly not independent because
that consumed population is partly replayed during fit.

The immutable local stream contains 14,170 rows /4,192,797 bytes with SHA-256
`8398f763...1ccc4`; tracked report SHA-256 is `eda98604...5319`. The pass locks DINOv2-L rather
than spending a larger model after S succeeded. E43-S is now a research candidate awaiting the
untouched ITW-SM final, not a promoted detector. ITW-SM still has zero payload images and zero
scores until manual author approval arrives.

With ITW-SM still awaiting manual review, the project opened its preregistered NIST Plan B without
touching evaluation data. The official GenAI Image portal still offers participant registration
through Login.gov and requires a completed data agreement before resources or submissions become
available. However, the visible Image-D round-3 schedule is already historical: D-Testset-3 was
released on 2026-02-23, outputs closed on 2026-04-03 and results followed on 2026-04-10. The portal
is therefore paused at the user-controlled Login.gov boundary so the authenticated account can be
checked for a late or future Image-D round. No team registration, agreement, NIST byte, system
submission or score exists yet; the project will not reinterpret an expired round as access.

The user then completed Login.gov authentication and capped any future NIST transfer at 4 GB. The
authenticated dashboard did not expose data: it requires every individual to act for a legally
registered/incorporated organization, allows foreign organizations subject to possible IAAO
approval, and shows no organization `site` associated with this account. The truthful profile form
requires country, full name, affiliation and affiliation type before site creation/joining, track
registration and licence upload can begin. Work stopped before submitting personal/affiliation
data because the exact official institution and authority to register under it must come from the
student. No site, track, agreement, NIST byte or score was created.

The access delay triggered the preregistered open-data fallback rather than more tuning on consumed
RR. A record audit found that the official NeurIPS 2025 DDA-COCO benchmark is still untouched by
every E43 decision and already sits on LaCie as five unopened multipart files. A fresh byte count
corrected the old interrupted-transfer note: the 212,860,928-byte prefix plus four disjoint ranges
sum exactly to the official 4,301,452,066-byte archive, so no further network transfer should be
needed. Whole-file SHA, safe ZIP inventory, member structure, decontamination and scores remain
absent.

Before assembly, Plan C binds the unchanged E43-S candidate `a3aec445...47390`, threshold
`0.8712875247`, DDA-COCO revision `8c9330a3...68fb`, Apache-2.0 archive hash and one-shot gates.
DDA-COCO directly tests whether the detector survives content/frequency alignment; it cannot prove
social-media robustness or replace ITW-SM. Any overlap will remove the entire real/reconstruction
pair before a score, and any failed archive hash stops without silently exceeding the user's 4 GB
download ceiling.

The assembled DDA-COCO archive matched the official SHA-256 and passed safe ZIP plus full CRC
inspection without a model load. It contains 29,969 synthetic images across six—not five—variant
folders, with counts 5,000/5,000/5,000/5,000/4,998/4,971. The release does not bundle its MS-COCO
REAL source, so Plan C now freezes one 815,585,330-byte official COCO val2017 companion download.
Its S3 size, Last-Modified and multipart ETag are bound before transfer; the resulting SHA-256 and
5,000-file schema must be sealed before decode/decontamination. This correction changes no model
or score and keeps total new transfer well below the user's 4 GB ceiling.

The official COCO val2017 companion then downloaded exactly once: 815,585,330 network bytes and
SHA-256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`. Its 5,000 JPEGs passed
the frozen member schema and full ZIP CRC. Cross-archive structure found 4,969 parents shared by
REAL and all six DDA synthetic variants, giving 34,783 candidate rows. No pixel was decoded for
selection and no model score exists; the next immutable checkpoint is parent-group decode/hash and
protected-role decontamination. This immediately available open benchmark advances the project
while ITW-SM and NIST remain access-gated, but it does not replace their real-world final scope.

The pre-score DDA-COCO pixel audit then decoded all 34,783 candidate images. Nineteen dHash hits
against 17 protected manifests touched four parent IDs; following the frozen pair rule removed all
28 real/synthetic views belonging to those parents. The final unscored manifest holds 4,965
complete parents /34,755 rows /5,080,919,889 source bytes, with zero exact duplicate groups, zero
cross-label exact groups and zero cross-parent dHash diagnostics. Its SHA-256 is
`e663d679f86ba69a545659203e11528d8998c9a362198a19f5f269a1ef97a3db`. No model was loaded and no
threshold or result changed; this is the immutable population for the next one-shot score.

The DDA-COCO one-shot score contract was then frozen with SHA-256
`a414e5005da69ea55ca2f7376421d4eae6956ff9a52c23a5cb7d8c06cb9b69da`. It binds the unchanged
E43-S artifact, threshold `0.8712875247`, manifest `e663d679...a3db`, 4,965 parents /34,755 rows,
seven conditions and all eight pass gates before model access. It explicitly forbids threshold
repair, post-score row removal, test-informed fitting and retry after a completed stream. Model
scores at this checkpoint remain zero.

The unchanged E43-S candidate then completed the DDA-COCO test once with 34,755/34,755 successful
scores. It failed decisively: pooled AUC 0.54178, TPR@FPR10 0.11712, EER 0.47051, balanced accuracy
0.51114, REAL FP 14.44% and AI macro/worst-variant recall 16.67%/12.77%. Every performance/safety
gate failed; only coverage passed. Score-stream SHA-256 is `1eefbdb7...42dd` and report SHA-256 is
`b91f4a52...c844b`.

The result is not a calibration accident. A post-hoc read-only diagnostic found a maximum pooled
balanced accuracy of only 0.53159 at threshold 0.32859, where REAL FP rises to 43.26% and AI TPR is
49.58%. The model therefore lacks a transferable representation for DDA's content/frequency-
aligned reconstruction regime. The first result stays immutable; DDA-COCO becomes consumed
DEVELOPMENT for a future E44 paired/adapter design, while ITW-SM or a future NIST round remains the
required untouched final. This negative result materially narrows the next research question and
prevents another ineffective threshold-only cycle.

Before downloading another representation or generating weaker VAE-only pairs, E44-A froze a
comparative screen for the already-pinned official DDA detector. Exactly 700 complete DDA parents
and all 4,900 real/synthetic views were selected only by a namespaced SHA-256 rank; the selected-
parent-list hash is `b1ac6bb2...1990`. Contract SHA-256 `df256498...5ce9` binds the consumed manifest,
official checkpoint, published threshold 0.5 and seven pass gates. No score was produced at this
checkpoint. This screen can isolate useful aligned-reconstruction expertise, but cannot become a
new independent final because DDA-COCO was already consumed by E43.

The frozen E44-A screen then completed all 4,900 rows without failure. The official DDA specialist
passed every preregistered gate: AUC 0.99006, balanced accuracy 0.93917, REAL false-positive rate
0.86%, core-four macro recall 98.61%, all-six macro recall 88.69% and worst-variant recall 64.57%.
Its strongest recalls were 99.71% on both SD VAE variants and SD 2.1; SDXL reached 95.29%, while
FLUX.1 and SD 3.5 Large were weaker at 64.57% and 73.14%. Score-stream SHA-256 is
`3d24d1c1...31d75`; detailed report SHA-256 is `a57e001d...090e`.

This resolves the low-score mystery: E43-S did not fail because the disk or manifest was broken;
its compact scalar representation omitted the content/frequency-aligned reconstruction evidence
that the official DDA representation learned. The official model still cannot be served alone
because earlier E35/E36 work found unsafe smartphone and modern-generator transfer. E44 therefore
keeps it as a specialist and proceeds to conservative fusion/adaptation with existing real-camera
safeguards. No replacement VAE dataset will be downloaded at this branch.

E44-B was then planned before creating the missing joint scores. It will use only the two frozen
model probabilities, source/parent/device-disjoint roles and a regularized logistic decision layer.
The 210 owner-gallery images remain development-only. Success requires simultaneous aligned-DDA,
RR, IPN and owner safety gates; failure preserves separate experts rather than manufacturing one
universal score. This stage requires no new image download.

The E44-B contract then verified all 1,670 frozen E35 image identities locally and joined their
existing DDA stream with the unchanged E43-S artifact, while binding the 4,900-row E44 aligned
sample to its immutable E43 stream. Detailed contract SHA-256 is `25681b62...3fb4`; E35 identity
SHA-256 is `919a0586...6b10` and the DDA role-map hash is `330000bf...8547`. The contract contains
zero new scores and transferred zero image bytes.

E43-S then produced the missing companion probability for all 1,670 E35 rows with 100% coverage.
The 455,631-byte stream SHA-256 is `35d9d2c2...ad5af`. Every input passed its bound byte hash; no
fusion head or DEVELOPMENT result existed at this checkpoint.

The fixed two-logit logistic head then fit 3,657 FIT rows and selected its single threshold on
1,307 CAL rows. Threshold `0.3423850493` yielded CAL AUC 0.96936, balanced accuracy 0.91637, REAL
FP 7.22%, AI group-macro recall 95.38% and worst recall 68.00%. Candidate SHA-256 is
`19fd7bbc...b100`; no DEVELOPMENT score had been created when it was frozen.

E44-B then completed its one-shot 1,606-row DEVELOPMENT score. It achieved AUC 0.97165, balanced
accuracy 0.91099, DDA macro/worst recall 91.33%/74.67%, RR AI macro/worst recall 99.29%/95.00%
and IPN worst-device FP 1.25%. However, the preregistered gate failed 2/10 checks: RR REAL FP was
12.00% (6/50, one image above the allowed count) and owner-gallery FP was 20.48% (43/210, also one
image above). Score-stream SHA-256 is `ce10c43f...67969`.

The near miss is not rewritten as success and its threshold is not repaired after inspection.
E44-B remains consumed DEVELOPMENT and the experts remain separate. E44-C may use this failure to
set a more conservative successor, but must bind and score a different already-local population
before judging it; ITW-SM or a future NIST round remains the independent final requirement.

A read-only E44-B margin diagnostic found that cut `0.3477933653`, only 0.00541 above the frozen
cut, would meet both missed real-image budgets on the consumed rows while retaining every other
gate. This does not change the failed E44-B record. E44-C instead freezes that value as a successor
hypothesis before scoring a different local population: 2,940 E43 RR views plus 2,160 E42 E36/E39
clean/robust views whose official-DDA scores do not yet exist. IPN/owner repeats are excluded.

The E44-C contract then verified 5,100 rows across 4,020 unique local files, with zero exact-byte
overlap against the E35 fusion population. Detailed contract SHA-256 is `b3c399e9...e1152` and
population SHA-256 is `ac79ea36...89aa3`. It binds successor cut `0.3477933653`, the unchanged
fusion artifact and all gates while `dda_scores_created` remains zero. No network byte was used.

The official DDA arm then completed 5,100/5,100 E44-C views with 100% coverage. Its 1,793,353-byte
stream SHA-256 is `3618b158...d3108`. The run survived the internet interruption because every
input and model weight was local. No fused E44-C metric existed at this checkpoint.

E44-C then completed its frozen 5,100-row comparison and passed 20/22 gates. Pooled AUC was
0.98181, balanced accuracy 0.92731, AI recall 96.85%, REAL FP 11.39% and EER 0.07373. RR-original
AI macro/worst recall reached 99.80%/98.57%; E42 clean and robust balanced accuracy reached
0.95501 and 0.96538. Fused score-stream SHA-256 is `0507cc4d...3d6bd`.

Two camera-safety checks failed: RR-original REAL FP was 16.33% against a 10% limit, and although
E42 clean REAL macro FP passed at 6.94%, `e36:device_004` reached 31% against the 20% worst-device
limit. The result remains failed. Since ranking and AI recall are already strong while residual
errors are concentrated in DDA-triggered camera pipelines, the next architecture is selective
classification (`AI`/`REAL`/`UNCERTAIN`), not another post-hoc scalar-threshold repair.

A read-only selective-risk diagnostic then combined the consumed E44-B/C score streams without
rerunning either model. Requiring group-level REAL false-AI macro/worst <=5%/10% and AI false-REAL
macro/worst <=10%/20% produced hypothesis cuts REAL `<0.2545712170` and AI
`>=0.6938513176`, with the middle marked `UNCERTAIN`. On the same consumed 6,706 rows this covers
87.40%, abstains on 12.60% and is 96.47% accurate among covered rows. These numbers design E44-D;
they do not validate it. A new independent population remains mandatory.

On 2026-09-03 the authenticated ITW-SM acquisition was retried after the student reported several
new emails. The pinned `.gitattributes` content preflight again returned HTTP 403 with Hugging
Face's explicit state `awaiting a review from the repo authors`. The downloader stopped before
scheduling any image: zero payload file, no acquisition receipt and zero model score exist; only
the previous 6.3 MB resumable cache scaffolding remains. The blocker is still author approval, not
local OAuth authentication or internet connectivity.

The open replacement route was then frozen before downloading a byte. MediaEval's official 2026
SID repository publicly links `itw-sm-sid-val.zip`, declaring the same scientific target: 10,000
in-the-wild images, 5,000 REAL and 5,000 synthetic. The live archive identity is 3,553,693,205 bytes,
ETag `"68555a02-d3d10e15"` and Last-Modified 2025-06-20, with byte-range resume. E45 assigns it once
as the untouched final for the already-fixed E44-D policy. It preserves the accepted research-only
and no-redistribution boundary and explicitly refuses to count this public distribution and the
gated Hugging Face snapshot as two tests unless later identity evidence proves they differ.

The success rules are fixed before transfer: complete scores; binary AUC >=0.90, balanced accuracy
>=0.85, pooled REAL false-AI <=10% and AI recall >=80%; source worst REAL false-AI <=20% and AI
recall >=60%; plus selective automatic coverage >=80%, covered accuracy >=95% and uncertainty
<=20%. The unchanged E44-D cuts remain REAL below 0.2545712170, AI at/above 0.6938513176 and
UNCERTAIN between them. A resumable, identity-bound, LaCie-only downloader and fail-closed ZIP/CRC
inventory were added with focused tests. At this checkpoint E45 contains zero downloaded bytes,
zero decoded images and zero scores; the plan/contract checkpoint must be committed before transfer.

After that checkpoint reached GitHub, the official archive downloaded completely to LaCie. It
matches the frozen 3,553,693,205-byte HTTP identity and has SHA-256 `18f1806e...b6e3`; the interrupted
internet caused no partial loss. The 10,000-path structure is exactly balanced, but the mandatory
per-member scan caught one published corruption: `ITW-SM/1_fake/x_618.jpg` cannot decompress.
A fresh range request around that member matched the local compressed bytes exactly, so another
3.55 GB download would reproduce the fault rather than fix it.

The archive is not relabelled as fully clean. E45 will disclose 9,999 usable rows /99.99% official
coverage, excluding this single AI member before pixels or model access. The exclusion is technical
and preregistered by the integrity gate, not selected from a score. Acquisition and inventory
evidence contain zero decoded images and zero model scores; next comes local decode/hash and
protected-role overlap audit.

That audit decoded all 9,999 usable members without another failure. It removed the second record
from 19 exact duplicate REAL pairs and two AI records whose dHashes matched protected prior data.
No exact duplicate crosses labels and no protected exact/dHash overlap remains. The resulting
untouched final is 9,978 rows: 4,981 REAL and 4,997 AI, with Facebook, Instagram, LinkedIn and X
preserved from the publisher filenames. The manifest covers 99.78% of the official 10,000 rows and
has SHA-256 `3e7c1d7e...d7e03`.

One hundred forty-one within-final dHash groups are reported but not automatically removed: exact
dHash equality can describe visually similar small/simple images without proving byte identity.
This rule was fixed before the audit. The tracked manifest evidence still records zero model scores;
the candidate and success gates must be bound to this exact manifest in a separate committed
contract before inference begins.

The second lock then bound all 9,978 record identities to the unchanged E43-S generalist, official
DDA specialist and E44 fusion hashes. Binary threshold `0.3477933653`, selective REAL/AI cuts,
ten pass gates and a 10,000-repeat platform/label-stratified bootstrap are now immutable. Detailed
contract SHA-256 is `4a5d4999...9ac83`. Generalist, specialist and fused score streams still contain
zero rows at this checkpoint; inference may begin only after this code and contract reach GitHub.

After the contract reached GitHub, the E43-S generalist processed all 9,978 frozen rows without a
decode, identity or inference failure. The 1,338,053-byte score stream has SHA-256
`43ecaa3f...fc171` and 100% manifest coverage. Batch results were fsync'd in manifest order so an
interruption could resume safely. This is only one frozen arm; no metric was inspected and both the
official-DDA arm and fused final result were still absent.

The official-DDA/DINOv2-L specialist then scored the same 9,978 rows with zero failure and 100%
manifest coverage. Its 1,338,964-byte stream SHA-256 is `88946986...69bb7`. The resumable design
preserved the first 200 batch-2 results when throughput was increased to the already-permitted
batch 8; model weights, preprocessing, row order and scores were not changed. Both arms are now
complete, but no fused metric had been opened at this checkpoint.

The frozen fusion then created all 9,978 final score rows, but reporting stopped before its first
metric because the shared evaluator requires a `source` key and E45 calls that identical grouping
field `platform`. The existing 2,142,780-byte fused stream is preserved at SHA-256
`b84f8c40...3c67e`; neither model will rerun and the stream will not be rewritten. The correction
only aliases `source=platform` in memory for the metric helper and can resume reporting only after
being committed. At this point no final report or pass/fail decision exists.

Reporting then resumed from that exact fused stream and produced the first independent E45 verdict:
**failed 4 of 10 gates**. Ranking still transfers—AUC is 0.95020, TPR@FPR10 is 0.87352—and the
model catches 95.40% of AI images. But it falsely accuses 34.13% of real social-media images, so
balanced accuracy falls to 0.80634. The false-AI rate is high on every platform: Facebook 39.30%,
Instagram 32.50%, LinkedIn 32.25% and X 32.11%. This confirms the camera/social-real weakness that
smaller local tests had warned about; it is not a single-device anomaly.

The selective E44-D policy also misses its safety claim. It automatically decides 80.54% and sends
19.46% to UNCERTAIN, satisfying both coverage limits, but accuracy among covered rows is only
90.07% rather than 95%. Ten-thousand platform/label-stratified bootstraps put covered accuracy at
89.43–90.71% and REAL false-AI at 32.82–35.45%, ruling out chance as an explanation. The score
stream remains `b84f8c40...3c67e`; no threshold, row, model or label was changed.

E45 is now consumed and permanently prohibited from training, calibration or model selection. The
next candidate must learn social-platform REAL safety from a separate licensed source while
retaining the current AI-recall gates, and it requires another untouched final before any success
claim. This result is disappointing but scientifically valuable: the project now knows exactly
which side fails and no longer relies on optimistic internal data.

The consumed-only E46 arm diagnosis then separated representation from calibration. E43-S alone
reaches only 0.80108 AUC /0.72970 balanced accuracy, with 21.22% REAL false-AI and 67.16% AI recall.
The official DDA expert is much healthier at 0.94010 AUC /0.87255 balanced accuracy, 10.74% REAL
false-AI and 85.25% AI recall. The frozen fusion improves ranking and AI recall but applies its old
operating point far too aggressively on social-media scores.

For proof only, the 90th percentile of consumed E45 REAL scores gives cut `0.7541002115`: balanced
accuracy 0.88667, REAL false-AI 10.02%, AI recall 87.35%, worst-platform REAL false-AI 12.77% and
worst AI recall 72.40%. All six binary gates would pass. That cut is permanently forbidden because
it was learned from the final, but it changes the next engineering decision: do not spend first on
a new backbone. Obtain a separate in-the-wild CAL population to learn transfer-safe calibration,
lock a different untouched FINAL, and keep E45 only as archived diagnosis. Detailed diagnostic
SHA-256 is `2ba9234a...41164`.

E46-A was then frozen before any new image transfer. The recovery no longer jumps directly to a
larger backbone: the official 2,000-row SynthWildX list is assigned only to calibration and
development, while the independent 3.9 GB TrueFake Facebook partition is reserved as untouched
final material. The scientific test is deliberately cross-platform—learn the operating behavior
from X, then transfer it unchanged to Facebook. SynthWildX will be split deterministically before
scoring; TrueFake will receive a score-blind balanced 2,000-row manifest after integrity and
protected-overlap auditing. The first candidate comparison keeps both existing model arms fixed
and tests a global REAL-safe cut against a small quality-conditioned calibration inspired by the
2026 QuAD work. No downloaded image or score exists at this checkpoint, and neither E45 nor the
future TrueFake final may participate in fitting or selection.

The official SynthWildX transfer then recovered 1,723 of 2,000 live X-hosted images, totaling
553,125,164 bytes. The other 277 publisher URLs consistently returned 403/404 and were recorded
rather than silently replaced. The surviving data still contains 418 REAL images and 1,305 AI
images spread across DALL-E 3, Firefly and Midjourney v5; the score-blind role split leaves 1,034
CAL and 689 DEVELOPMENT rows. A path-default mistake initially placed these ignored bytes under
`ml/data`; it was caught before scoring, and all payloads were moved unchanged to the LaCie
research store. Only absolute manifest paths and their receipt hash changed; image-byte changes
were zero. The resulting unscored manifest is 1,224,028 bytes /SHA-256 `fd8008a...a89f3f`.
Two exact duplicate groups are disclosed for the identity-audit stage. No detector has read the
data yet.

The second E46 source then arrived intact on LaCie: TrueFake's Facebook archive is
4,207,525,545 bytes /SHA-256 `413cb7f9...cda0d63`. Its complete gzip and TAR checks pass, and the
publisher structure matches the paper exactly—20,000 REAL images split between FFHQ and FORLAB,
plus 40,000 AI images split evenly across eight GAN/diffusion generators. This resolves the earlier
final-data availability blocker within the 3–4 GB planning envelope. The archive is still sealed
from the model: the next committed code will first freeze a hash-ranked 3,500-candidate reserve,
decode only enough to produce the balanced 2,000-row final, and reject prior-data overlaps before
any score exists.

The first SynthWildX identity-audit command failed closed before producing a manifest: acquisition
had preserved exact hashes and geometry but had not yet calculated perceptual dHash. The audit was
corrected to decode each already-validated file and derive dHash locally, with a regression test.
No image, role or model score changed, and no partial scientific result was retained.

The repaired audit completed and removed 15 SynthWildX identities before model access: two were
duplicate copies within the set and thirteen overlapped protected earlier manifests by exact hash
or dHash. The clean pool is 1,708 rows, split 1,024 CAL /684 DEVELOPMENT, with 415 REAL images and
396–472 samples from each AI generator. Its audited manifest SHA-256 is `953490a9...da8d4`.

TrueFake's selection was then locked while still score-blind. The complete 60,000-file inventory
hash is `b59e78de...8ba28b`; a deterministic 3,500-member reserve contains enough headroom to reject
bad or contaminated samples while retaining exactly 500 FFHQ, 500 FORLAB and 125 from each of the
eight AI generators. Detailed contract SHA-256 is `1e77dfbd...cead3`. This checkpoint contains no
decoded final payload and no model score, so later extraction cannot steer the sample.

The contracted TrueFake reserve was then streamed once. Every one of the 3,500 candidate images
decoded, no candidate matched any of 24 protected manifests, and no internal exact/dHash duplicate
group appeared. The lowest clean ranks therefore filled every quota exactly: 500 FFHQ, 500 FORLAB,
and 125 from each of eight AI generators. The untouched final is now a balanced 2,000-row manifest
at SHA-256 `4572339e...b225b`, still with zero model scores. Only these rows may be used in the
future one-shot evaluation; the additional reserve payloads are not evaluation observations.

E46 development scoring was then bound before either model loaded. Contract SHA-256
`b3fe31a3...5c98c` fixes 1,708 clean SynthWildX rows, the exact E43-S /official-DDA /E44-fusion
artifacts, 1,024 CAL and 684 DEVELOPMENT roles, three allowed calibration candidates and four
development gates. The quality experiment may use only log minimum dimension, log bits-per-pixel
and mean neighboring-pixel difference; it cannot introduce a hidden new model. TrueFake remains
unread and all score streams are still empty.

The E43-S arm then completed all 1,708 SynthWildX rows with full coverage. Its 394,197-byte
resumable score-and-quality stream hashes to `8be0aefd...ce88d`. No DDA, fusion, calibration or
TrueFake result existed at this checkpoint, preserving a clean recovery point before the heavier
specialist run.

Official DDA then completed the matching 1,708-row SynthWildX stream with full coverage. Its
268,844-byte output hashes to `a7fbd7e2...257eda`. Both expensive inference arms are now safely
recoverable and immutable; fusion/calibration statistics remain unopened, and the TrueFake final
still has zero scores.

Before opening either stored score stream, E46 froze a second internal split and selection rule.
Contract SHA-256 `6799231f...c9228c` divides the 1,024 CAL identities within every source into 612
QUALITY_FIT and 412 OPERATING_CAL rows. The quality-aware Gaussian may learn only on the first;
the REAL-10% threshold and method comparison use only the second. Global methods win by default,
and the quality method may replace them only when non-inferior on every safety/recall/AUC measure
and materially better on at least one. DEVELOPMENT and TrueFake remain unread.

The CAL-only comparison selected a much safer operating point without touching E45 or TrueFake.
Official DDA reached 0.9003 balanced accuracy and 90.06% AI recall at 10% REAL false alarms. Global
fusion improved to 0.9179 balanced accuracy and 93.59% AI recall at the same REAL budget, with
threshold `0.6688565013`. The QuAD-inspired quality model pushed AI recall to 96.79% but slightly
lowered AUC; the frozen conservative rule therefore rejected the added complexity and kept global
fusion. Candidate SHA-256 is `9fec91b8...b84a1`; its CAL-only selective band is REAL below
`0.5185430496`, AI at or above `0.6688565013`. DEVELOPMENT and the final remained unread.

The frozen candidate then passed all four SynthWildX DEVELOPMENT gates without refitting. Across
684 unseen rows it achieved 0.9720 AUC, 0.9122 balanced accuracy, 11.38% REAL false alarms and
93.81% AI recall; even the weakest generator, Firefly, retained 84.82% recall. This is the clearest
valid improvement over E45's 34.13% REAL false alarms so far. The selective diagnostic covered
96.49% but reached 94.39% covered accuracy, narrowly missing the future 95% claim; it remains
unchanged. Score stream SHA-256 is `e9443455...24c9b8`, and TrueFake still has zero scores.

The successful DEVELOPMENT candidate was then bound to TrueFake before model load. Final contract
SHA-256 `1cf28d2d...7c4262` fixes all 2,000 rows, the candidate and two model identities, binary cut
`0.6688565013`, selective band, ten pass/fail gates and 10,000 source-stratified bootstrap draws.
No later score may change these values, remove a hard image or trigger a retry. At this checkpoint
the independent final still contains zero model scores.

E43-S then scored all 2,000 contracted TrueFake rows with 100% coverage. The immutable 298,721-byte
generalist stream hashes to `43eb1562...b5f25c`. DDA and fused results remained unopened, so this
checkpoint contains no partial interpretation of the final.

Official DDA subsequently completed all 2,000 matching TrueFake rows. Its 298,587-byte stream
hashes to `13947caf...878d0b`. Both costly arms are now complete and independently recoverable;
the fusion report and every final metric are still unopened pending this commit.

E46 then opened its independent final exactly once, with every previously frozen row and decision
rule intact. The result is a valid failure: 100% coverage, 0.8155 AUC, 0.7345 balanced accuracy,
5.60% REAL false accusations and 52.50% AI recall; 5 of 10 gates pass. Diffusion performance ranges
from 51.2% on SD3 to 100% on SD1.5, while StyleGAN/2/3 collapse to 1.6%/1.6%/3.2% recall. Thus the
new cross-platform calibration fixed the earlier REAL-safety problem but exposed a different,
severe GAN-family blind spot. The selective band does not rescue it (95.85% coverage, 74.86%
covered accuracy). Evidence SHA-256 is `e7e14fdf...d7a7ed`; no threshold repair, sample removal,
refit or retry will be performed on this consumed final. E47, if pursued, must obtain separate
GAN-aware development evidence and a genuinely new final.

E47 was therefore opened as a narrow GAN-blind-spot recovery, not another threshold sweep. Its
first step is deliberately cheap: ask whether the already-trained, hash-pinned GenImage ResNet-18
contains the missing GAN signal on the now-consumed E46 rows. Those labels may diagnose
complementarity but cannot train, calibrate, validate or prove the successor. A strict unlock rule
was written before scoring; failure routes to an official frozen GAN specialist, while success
still requires new source/identity-separated CAL, DEVELOPMENT and final evidence.

The cheap E47-R1 hypothesis failed cleanly. The old GenImage ResNet sees only 4%/8%/8% of
Facebook-transported StyleGAN/2/3 at a diagnostic 10% REAL false-positive cut. OR-combining it with
E46 recovers 60 of 475 misses but raises REAL false accusations to 15.1% and leaves AI recall at
58.5%. Because all three frozen unlock conditions fail, the arm is retired from E47. The next
permitted action is an official frozen GAN specialist, not another threshold adjustment.

E47-R2 chose official UniversalFakeDetect rather than retraining another local CNN. Its repository,
MIT licence, 4 KB ProGAN-trained linear head and 932.8 MB OpenAI CLIP ViT-L/14 backbone were each
hash-pinned; the advertised 72 GB training corpus was deliberately skipped. The old code needed
two non-semantic compatibility shims for removed text-only dependencies, after which a two-image
smoke test returned finite outputs. No performance score was inspected. The unchanged R1 unlock
rule will decide whether this arm earns a new calibration/development population.

UnivFD supplied the first convincing missing signal: 94.4%/74.4%/80.0% recall on
StyleGAN/2/3 and 310 of E46's 475 AI misses recovered. A simple diagnostic OR would raise total AI
recall from 52.5% to 83.5%. It nevertheless failed the pre-registered admission rule by the
narrowest measurable margin—15.5% pooled REAL false-AI against a 15% ceiling. The rule was not
relaxed after seeing the result. UnivFD remains a valuable near-success and reference arm, while
E47 proceeds to the already-authorized UNINA compression-trained GAN specialist comparison.

The comparison arm is now pinned: GRIP-UNINA's StyleGAN2-trained ResNet50-NoDown at repository
commit `543943c...df88` and checkpoint SHA `65467594...d5a08`. Only 282.5 MB of weights were
downloaded; no training/test corpus. The licence is nonprofit research only, an important product
constraint even if accuracy is strong. A two-official-example smoke test verified finite logits and
the documented fake direction without opening E46 metrics.

The first UNINA pass exposed a product blocker before exposing a metric: native-resolution
ResNet50-NoDown dropped below 0.5 image/s on common 960 px inputs. It was stopped at 655/2,000;
the 88,487-byte partial stream was hash-preserved but excluded. Before restarting, E47 froze an
aspect-preserving 512 px long-side cap for every row. This changes only deployment preprocessing,
not labels, identities or the admission rule, and makes latency part of model selection.

The capped UNINA comparison completed and made the next problem precise. It sees
StyleGAN/2/3 at 100%/94.4%/73.6%, recovers 375 of 475 E46 misses, and would lift pooled AI recall
to 90%. But a naive OR still raises REAL false accusations to 15.5%, so the unchanged admission
gate rejects direct serving. UnivFD and UNINA independently prove that the missing GAN
representation now exists; what E47 lacks is a decision layer trained on separate data to know
when the specialist is trustworthy. No E46 score will be used to learn that decision.

E47-R3 was bound around that conclusion before extracting another image. It will use 2,400 new,
hash-ranked TrueFake members outside all 3,500 prior reserve identities: FFHQ/StyleGAN2 plus two
diffusions for CAL, and source-held FORLAB/StyleGAN/StyleGAN3 plus FLUX/SD3 for DEVELOPMENT.
Twenty-percent reserve headroom permits score-blind audit failures. Every backbone stays frozen;
only a regularized decision gate and CAL threshold may learn. The MIT UnivFD route wins ties close
to nonprofit UNINA, and no DEVELOPMENT row may repair the chosen rule.

The new-data contract then froze cleanly: 2,880 score-blind reserve members, balanced 1,440/1,440
between CAL and DEVELOPMENT and disjoint from all 3,500 E46 reserve identities. Contract SHA is
`c031ef92...d0753`; the intended post-audit population remains 2,400. No candidate image had been
decoded and no model had been loaded at this checkpoint.

Extraction then validated every one of the 2,880 candidate payloads. A single SD1.5 reserve image
matched a protected perceptual hash and was removed before model access; the 20% headroom absorbed
it without changing any quota. The frozen clean population is exactly 1,200 CAL and 1,200
DEVELOPMENT, each 600 REAL/600 AI, at manifest SHA `378b83fe...85739`. Scores remain zero.

All four inference identities were then frozen together under contract SHA
`ee2a2958...95798`: E43-S, official DDA, the old fusion, UnivFD and capped UNINA. The contract binds
the exact 2,400 rows and forbids role drift, backbone updates and DEVELOPMENT-informed fitting.
No score existed when it was committed.

E43-S then completed all 2,400 new CAL/DEVELOPMENT rows. Its 399,398-byte stream hashes to
`073110f4...f30c03`. This checkpoint preserves raw inference only; no candidate metric or
DEVELOPMENT result was opened.

Official DDA then completed the matching 2,400 identities. Its 399,187-byte stream hashes to
`8001c60b...d75f5`. Both base inputs for the frozen E44 fusion are now safe on disk; specialist
comparison and CAL fitting remain unopened.

MIT-licensed UnivFD then scored all 2,400 rows. Its 403,436-byte stream hashes to
`67b7b94c...e2829`. No class/source metric was read; capped UNINA and decision-gate fitting remain.

Capped UNINA subsequently completed the same 2,400 identities with full coverage. Its
398,998-byte stream hashes to `7efb36c0...5e16d`. E47-R3 now has all four frozen inference
streams, but still has no opened CAL or DEVELOPMENT metric. The exact CAL-only fitting,
threshold selection and candidate-selection policy must be written and committed before these
scores can influence a decision.

That decision policy is now executable and frozen before any aggregate score was opened. Contract
SHA `a4515caf...875a` compares frozen E46 with CAL-only C=0.1 logistic gates adding UnivFD, UNINA
or both; fixes the REAL-safety threshold rule, all seven success gates, deterministic ranking and
the two-point MIT-licence preference. DEVELOPMENT remains a one-shot transfer test and cannot fit,
select, repair or retry the candidate. Six focused contract tests pass; both CAL and DEVELOPMENT
metric counts are still zero.

CAL then opened exactly once and justified the new decision layer. Frozen E46 still catches only
73.5% of CAL AI and 20.5% of StyleGAN2 at the 10% REAL budget. UnivFD raises pooled AI recall to
85.67% but lands at 59% on the hardest source, one point below the written floor. UNINA passes,
and the all-three gate ranks first: 0.9897 AUC, 0.9367 balanced accuracy, 97.33% pooled AI recall
and 95% worst-source AI recall at 10% REAL false accusations. The selected threshold is
`0.3353660721`; candidate SHA is `f659ee4f...0b0d`. This is strong calibration evidence, not yet
a success claim: DEVELOPMENT remains completely unopened and may only be evaluated once.

The frozen all-three candidate then faced DEVELOPMENT once. It transferred well overall—0.9549
AUC, 0.8842 balanced accuracy, 7.33% REAL false accusations and 84.17% pooled AI recall—and the
original GAN failure is repaired (StyleGAN 100%, StyleGAN3 87.5%). SD3 also reaches 84%. FLUX.1,
however, falls to 46%, below the pre-registered 60% worst-source floor. Six of seven gates pass,
so E47-R3 is archived as a valid near-miss rather than tuned into a success. The score stream SHA
is `97fbe4b7...72cd`. This isolates the next architectural problem: the GAN-aware logistic gate
must not be allowed to veto a diffusion signal; any repair requires new fit/test evidence.

Before another metric is computed, the allowed post-failure diagnostic is limited to locating
that FLUX failure across the already-frozen arms. It may distinguish missing representation from
decision-layer veto, but cannot choose a new threshold or candidate and cannot repair E47. Any
successor must be bound and tested on fresh evidence.

The bounded diagnostic found the missing mechanism. Frozen E46 already detects 95% of the held
FLUX rows, but at that CAL-only cut falsely accuses 30.67% of FORLAB. The all-three gate repairs
FORLAB to 7.33% and rescues 149 StyleGAN plus 141 StyleGAN3 examples, yet vetoes 50 of E46's 95
FLUX hits and falls to 46%. The representations are complementary; forcing low GAN-specialist
scores to count against the diffusion expert is the mistake. E48 must learn safety from more than
one REAL pipeline and combine experts without allowing an irrelevant specialist to veto positive
evidence, using fresh CAL and DEVELOPMENT rather than recycling this diagnosis.

GitHub CI later reported a red build for commit `1212628`, but the code was not the cause: Python
tests and web lint/typecheck/tests all passed, while npm's audit API returned HTTP 503 after seven
minutes. CI now retries that exact blocking audit up to three times. It still fails on a persistent
registry outage or any real critical advisory; only transient service noise gains recovery.

E48 is now planned around the E47 mechanism rather than another backbone. It will calibrate each
frozen expert against fresh authentic camera photos and take the maximum AI-evidence percentile,
so an irrelevant low specialist score cannot veto FLUX. FIT, CAL and DEVELOPMENT are separated;
the decision layer learns from unused VISION/CSAFE identities and must transfer to unused FODB
camera originals plus mostly held AI generators. The 2,400-row design uses existing LaCie data,
downloads nothing, excludes all E46/E47 candidates and keeps the web model unchanged until a new
publisher-separated final passes.

The E48 selector then bound 2,880 score-blind reserve candidates for the 2,400-row target. FIT,
CAL and DEVELOPMENT are class-balanced; camera rows are device-balanced, and all 6,380 prior
E46/E47 TrueFake candidates plus the current model's E32 training identities are excluded before
ranking. Contract SHA `dbb6f4aa...0e6e` fixes every identity and quota with 20% audit headroom.
No new AI payload has been extracted and no model score exists at this checkpoint.

E48's first payload audit stopped safely before producing a manifest: the old R1b role ledger
enumerates the whole 22,688-row C3 planning population, so treating it as consumed evidence masks
every new camera candidate. That is broader than the current model's actual data boundary. E48
therefore keeps its frozen candidate identities but protects the exact E42 training subset and all
actual historical CAL/DEVELOPMENT/final manifests, not the superseded planning ledger. No score or
performance metric was created by the failed audit.

The next audit stop found no payload mutation: every checked camera file reproduced its exact
SHA-256, but the current helper's EXIF-orientation path does not reproduce the older realization
dHash. E48 now treats byte SHA as the payload-integrity authority and retains the original,
already-decoded audit dHash for perceptual-overlap checks. Recomputing and silently replacing that
pinned identity would be less reproducible. Candidate rows, roles and quotas remain unchanged;
model scores are still zero.

The corrected E48 audit then completed every one of its 2,880 candidates with no decode failure.
One VISION and one FODB reserve row overlap protected dHashes and were removed before model access;
the precommitted 20% headroom filled all quotas without replacement by score. The immutable clean
population is 600 FIT, 600 CAL and 1,200 DEVELOPMENT rows, each exactly half REAL and half AI, at
manifest SHA `1404a3ff...5b68`. No model score or DEVELOPMENT metric exists yet.

The project now has an explicit two-module completion order. Module 1 must first pass E48's fresh
DEVELOPMENT and then a separate E49 comprehensive final with multiple real pipelines, at least five
AI families, social recompression and confidence intervals. Only that all-gate result may create
Module-1 v1 and replace the demo. Module 2 model work resumes afterward, scoped to AI local
inpainting rather than every historical manipulation type.

That re-entry preserves the value of the unsuccessful early localisation work. E17 found genuine
CocoGlide signal but retained only 35/120 examples; raw IoU was confounded by mask size; classic
splices asked the AI-texture detector the wrong question; ELA worked on its JPEG positive control
but the compiled PNGs destroyed its input; and AI-filled regions showed a promising noise-energy
drop from 0.0164 to 0.0088. Module 2 will fix those measurement errors, test dense DINO/noise
features against exact masks and keep fully re-rendered edits out of localisation claims. Later
Module 2 discoveries may motivate fresh Module 1 experiments, but can never silently tune the
frozen Module-1 v1 artifact or its evidence.

E48 implementation resumed only after that order was committed. Score contract SHA
`ea7de06c...9516` binds the clean manifest plus exact E43-S, official DDA, E44 fusion, UnivFD and
capped-UNINA identities. The executable scorer can access only the 600 FIT and 600 CAL rows;
DEVELOPMENT's 1,200 rows are rejected until a candidate is frozen. Six focused tests pass and no
model or DEVELOPMENT score exists at this checkpoint.

The frozen E43-S generalist then scored all 1,200 permitted E48 FIT+CAL rows with full coverage.
Its 258,603-byte raw stream hashes to `f2a1be3b...137cf7`. No aggregate metric was opened and no
DEVELOPMENT row was accessed; the checkpoint preserves inference only.

Official DDA then scored the matching 1,200 FIT+CAL identities with full coverage. Its 257,443-
byte stream hashes to `7ebc7831...b4a23b`. UnivFD/UNINA and every aggregate metric remain unopened;
DEVELOPMENT score count is still zero.

MIT-licensed UnivFD then completed the same 1,200 FIT+CAL rows. Its 259,689-byte stream hashes to
`e768d591...81d635`. The UNINA arm and all fitted percentile/threshold metrics remain unopened;
DEVELOPMENT still has zero scores.

Capped UNINA completed the fourth and last permitted E48 inference stream: 1,200/1,200 FIT+CAL
rows, 257,330 bytes, SHA `e3d47527...4b3c01`. All four raw arms are now recoverable, while every
aggregate metric and all 1,200 DEVELOPMENT rows remain unopened.

Before opening those aggregate metrics, E48 froze decision contract `22154ab9...590e`. Only the
300 authentic FIT rows may define each expert's empirical authentic-percentile map; four declared
maximum-evidence combinations then compete on the 600-row CAL split under the already-recorded
authentic-safety and AI-recall gates. FIT AI and DEVELOPMENT remain forbidden. Seven focused tests
pass, and an all-candidates-fail result now leaves a durable failure report but no candidate
artifact—so neither missing evidence nor a relaxed threshold can masquerade as progress.

E48 then failed exactly as that contract required. Its best ranking quality remained strong
(AUC 95.42%) and authentic safety held (2.33% pooled false accusation, 20% worst camera), but E46
caught only 11% of StyleGAN2 and 63.67% of AI overall; adding either specialist raised the hardest
GAN result only as far as 19%. No candidate passed, no deployable artifact was written and all
1,200 DEVELOPMENT rows stayed unopened. The immutable CAL report hashes to `032944b8...75e`.

The allowed diagnosis exposed a simpler and much more useful result: frozen E43-S alone passes
every CAL gate—AUC 98.84%, balanced accuracy 93.83%, 3.33% pooled REAL false accusations, 91% AI
recall and 76% on the hardest source. It catches FLUX.1/SD1.5/SDXL at 98%/98%/100%. E46's signed
DDA fusion cuts StyleGAN2 from 76% to 13%, so E48 could not recover the evidence because its base
had already vetoed it. A new E50 checkpoint will lock that single existing generalist and threshold
before one use of the still-clean DEVELOPMENT split; E49 remains mandatory before Module-1 v1.

E50 is now physically locked before that use. Contract `18ae708f...20f6` fixes the exact E43-S
artifact, CAL-derived threshold `0.07940196245908739`, all seven gates and the identities of 600
FODB camera originals plus 600 held AI images across FLUX.1, SD3, StyleGAN and StyleGAN3. No
specialist, retraining, threshold repair, row replacement or second attempt is permitted. Eight
focused boundary tests pass; DEVELOPMENT scores and metrics remain zero at this commit.

The frozen E43-S candidate then scored all 1,200 E50 DEVELOPMENT rows with full coverage, no
failure and no replacement. Its 271,063-byte raw stream hashes to `07461b09...d5fd`. Aggregate
metrics are still unopened at this checkpoint, preserving a recoverable boundary between inference
and the one-shot verdict.

E50 passed all seven precommitted DEVELOPMENT gates on the first and only evaluation. Across 1,200
new rows, AUC is 97.84%, balanced accuracy 90.17%, pooled authentic false accusation 3.83%, worst
camera false accusation 18.18%, pooled AI recall 84.17% and worst-source recall 68.67%. FLUX.1 and
SD3 transfer at 98.67% and 92.67%; the unseen GAN families reach 68.67% StyleGAN and 76.67%
StyleGAN3. The threshold did not move and there was no retry. This is strong independent evidence,
but Module 1 is deliberately not declared finished: E49's larger publisher/transport-separated
final remains mandatory before Module-1 v1 or any demo replacement.

E49 planning then converted “a larger final” into an exact last proof. The target is 2,000 balanced
parents, not thousands of loosely sourced files: 1,000 new Wikimedia Commons camera-category
original uploads across ten device groups, 800 August-2026 Datapoint outputs across five current
provider/model cells, and 200 StyleGAN2 images from an unused test-only AIGC source. Received bytes
and a deterministic social JPEG child are separate paired columns; the same ten frozen binary and
selective gates must pass in both, with 10,000 parent-level stratified bootstraps.

Two tempting choices were rejected before an image transfer. SCIMD-17 is compact, licensed and
genuinely photographed, but every file was publisher-resized to 224 x 224, so it would recreate the
resolution shortcut instead of proving gallery-real safety. ImageBench covers the newest models
beautifully, but its canonical site currently reserves generated-image reuse without written
permission. Datapoint is both current and CC BY 4.0 at the metadata level, but requires the account
holder to share contact details and accept provider-output terms; the access page was opened for
the user and no form was submitted by the project. Until that gate clears, implementation may bind
metadata and test fail-closed behavior but cannot download or inspect its image payloads.

The first E49 implementation checkpoint now enforces that boundary in code. It validates both Hub
revision/licence pairs, screens Commons metadata for licensed original JPEGs, prevents one uploader
from dominating a camera cell and selects StyleGAN2 by Parquet coordinates without reading the
image column. Four focused tests pass. The live probe reproduced both pinned revisions and returned
the expected Datapoint `GatedRepoError`; no E49 image byte was created. This means the final cannot
quietly run on the convenient two-thirds of its sources while calling itself comprehensive.

The Module 2 archive was also reread before planning its return. Its brightest idea survives—the
absolute AI detector sees genuine signal inside CocoGlide inpainting, and lower residual noise plus
dense DINO patch tokens are promising complementary cues. Its failure is now more precisely stated:
E17 capped each image at 36 non-overlapping tiles, silently discarded masks lacking both majority-
inside and majority-outside tiles, pooled correlated tiles as samples and compared a tile-threshold
IoU with a pixel-fraction random baseline. Only 35 of 120 CocoGlide candidates survived. The old
head was also a pre-DINO handcrafted 128 px model, not E43-S. Module 2 will therefore begin—only
after E49—with a scene-disjoint manifest and corrected dense, image-level evaluator, then compare
absolute crop evidence, noise energy and frozen-DINO dense tokens before a one-shot held-source
final. Classic splices remain specificity controls; they will not dilute the AI-inpainting target.

E49's measurement engine was then implemented while the gated source remained unopened. It binds
the exact parent/source composition across publisher-original and deterministic social-Q75 children,
keeps the E50-derived binary and selective cuts immutable, evaluates ten gates per transport and
uses source/label-stratified parent bootstraps. Inference errors become UNCERTAIN and contribute
pessimistically to class/source rates rather than disappearing. A regression test initially exposed
that an explicit null error score was still being converted to a float; the implementation was
corrected before any final data existed. The resulting 17-test acquisition/evaluation regression
set passes, while E49 score and metric counts remain zero.

The already-local AIGC benchmark was then verified rather than redownloaded. Its cache ref matches
the pinned commit, all 60 Parquet shards are present, and a metadata-only scan counted 125,026 rows
including 1,997 eligible StyleGAN2 examples. The frozen rank can therefore supply the required 200
parents plus 40 reserves; the reserve identity digest is `15e5c131...cc731`. The new probe reads
only `label` and `generator`, never the image column or a detector score. No E49 image was decoded
and no network byte was spent.

The Datapoint contact-sharing form was subsequently submitted by the user. The repository page now
states that the request is awaiting review by its authors; the CLI is authenticated as the same
`efehankeles` account, and a fresh probe still fails closed with `GatedRepoError`. This is an
external approval wait rather than a login or implementation fault. E49 remains unscored and no
Datapoint image byte has been downloaded.

ITW-SM access was rechecked as the other remembered gated request and still returns HTTP 403
`GatedRepoError`. This does not block new work: the same official MediaEval/ITW-SM distribution was
already acquired publicly and consumed by E45, so later Hugging Face approval could not create a
second independent final. A new source audit rejected MLLMGenSet as a final substitute because its
publisher/collection already supplied E30/E31 DEVELOPMENT, and rejected currently licence-unstated
SafeIMG/NTIRE releases from a promotion claim.

The immediate open diagnostic is therefore Dotting Test: CC BY 4.0, ungated, 384 MB total and 8,396
successful images across 40 generators. It contains 210 rows each for the five selected 2026 models.
Before image transfer, E49-D1 freezes an AI-only 960-row reserve and 512 MiB ceiling. Its narrow
Turkish text/sign content is explicitly prevented from masquerading as a balanced final; it can
measure current-model AI recall and recompression loss while Commons/Datapoint E49 remains clean.

E49-D1 acquisition code was committed before transfer and passed 13 focused E49 tests. Its first
metadata-only bind stopped safely because a few successful publisher rows omit width/height; no
contract or image existed. The parser now keeps absent declared geometry as unknown for decode-time
recovery. The completed pre-transfer contract freezes 960 WebP files (192 per modern model) totaling
23,936,830 bytes, with contract SHA-256 `170f70db...ed36` and reserve identity SHA-256
`9637626d...f5a`. This is below five percent of the 512 MiB stop; image and score counts remain zero.

All 960 E49-D1 files then transferred successfully to LaCie, totaling the exact 23,936,830 bytes
bound in advance and 192 rows per model. Every actual WebP matches its expected size and SHA-256.
The first completion check refused to write a receipt because exFAT exposed macOS `._*` AppleDouble
sidecars as extra files. A regression-tested change now ignores only `.cache` and `._*` auxiliary
metadata while still rejecting other extras; the rerun completed. No detector loaded these images,
and decode/decontamination plus the 800-row target freeze remain the next checkpoint.

The E49-D1 target is now frozen before any detector access. All 960 reserves decoded with zero
failure; exact and perceptual comparison to 15 protected-role manifests plus internal duplicate
checks excluded six identities. The reserve still produced precisely 160 clean parents from each
of GPT Image 2, Nano Banana 2, FLUX.2 Pro, Ideogram 4 and Seedream 5.0 Lite. Each of the resulting
800 parents is paired with one deterministic social-Q75 child, yielding 1,600 unscored observations
at manifest SHA-256 `048572a4...ccc9`. This remains AI-only diagnostic evidence, never training data
or a substitute for E49's balanced REAL/AI final.

The diagnostic scorer was then implemented and committed before model access. Its contract
`d567965d...1cf9` locks the frozen E43-S artifact, DINOv2-S weights, both historical thresholds,
all 1,600 observation identities and the six declared coverage/recall checks. It explicitly forbids
training, row or source removal, threshold repair, retry and Module-1 promotion. Eighteen focused
E49 tests pass; no score or aggregate metric has yet been produced.

E43-S then scored all 1,600 frozen E49-D1 observations. The first finalization stopped safely after
inference because the validator incorrectly demanded manifest-only file paths from the deliberately
smaller score schema; it wrote neither completion receipt nor metrics. A regression test now keeps
payload and score validation distinct, and the preserved prefix finalized without rescoring. The
326,693-byte raw stream hashes to `c97b02a4...fa90`, covers every row and still has zero opened
aggregate metrics.

The single permitted E49-D1 metric opening passed all six preregistered checks. Publisher originals
reach 97.38% pooled AI recall with 91.25% on the weakest model; social-Q75 reaches 95.88% pooled and
86.88% worst-model recall. GPT Image 2 is the weakest in both transports, while the other four
modern generators remain at least 96.25% after Q75. Compression costs just 1.50 recall points. No
threshold, row, source or model changed and retry count is zero; report SHA-256 is
`bb62ad92...d77b`. The result materially strengthens the modern-AI side of Module 1 but cannot say
anything new about authentic-photo false alarms, so comprehensive balanced E49 remains open.

The authenticated ITW-SM Hugging Face preflight was also repeated on 2026-09-04. It still returns
HTTP 403 with `awaiting a review from the repo authors`. This creates no new blocker because the
official MediaEval copy was already consumed in E45; a later mirror approval cannot be reused as a
second independent final. The Datapoint E49 source was checked in the same authenticated session
and independently remains HTTP 403 /author-review pending. Neither probe downloaded an image.

Work continued on the open E49 REAL component instead of waiting. A licence-only Commons metadata
scan proved nine planned camera categories can fill an uploader-capped reserve, but Fujifilm X-T5
cannot: its 793 files collapse to only 66 eligible rows after the contributor cap. Because no E49
source contract, REAL image or detector score exists, the category was replaced transparently with
Nikon Z 8 before binding; it supplies 110 rows across 25 uploaders. The target remains ten devices x
100 photos. Commons headroom becomes 10% rather than 20% to preserve the 4 GiB network ceiling, and
the API now requests only licence fields the audit consumes. Eleven focused acquisition/evaluation
tests pass.

An ungated route around the external Datapoint wait is now preregistered without opening a model
score. E49-B pins `ComplexDataLab/OpenFake` revision `3fd1109d...0163b`, CC-BY-NC-4.0 and the exact
91,398-row `core/test` split. It will inspect only deterministic 100-row Dataset Viewer metadata
pages until five exact modern-generator cells each reach 192 eligible rows, then freeze 160 target
+32 reserve identities by namespace hash. Full 5+ GB Parquet shards, prompts and expiring URLs are
excluded. Only after identities are immutable may fresh asset sizes be bound against the unchanged
4 GiB ceiling. This is a separate E49-B candidate, not a rewrite of E49-A; Dotting remains a narrow
diagnostic and no OpenFake image, score or training row exists at this checkpoint.

The OpenFake Viewer metadata scan then cached 51,900 exact ordered rows before repeated HTTP
429/502/503 failures made completion through that public service unreliable. No contract, image or
score was created. Before knowing the complete model populations, the transport rule was amended:
the same pinned Parquets may be read by measured HTTP byte ranges for only `label`, `model`, `type`
and `release_date`. Full shards, image payloads and prompts remain forbidden, while row order, five
cells, 192-row quotas, hash ranking and the first-complete-page stop rule stay frozen.

The complete 91,398-row E49-B qualification then rejected the source plan before selection. GPT
Image 2/Seedream v5.0/FLUX.2 Klein 9B/Midjourney 7 supply 470/372/8,093/3,586 eligible images, but
Nano Banana Pro supplies only 60 against the frozen 192-row reserve. The exact-revision projection
transferred just 2,597,624 metadata bytes in 650 ranges, avoided all 67,649,942,401 Parquet payload
bytes and cross-validated 52,600 Viewer rows. Zero identity, asset, image or model score was created.
E49-B is archived without substitution; any replacement cell requires a separately preregistered
successor.

E49-C is now preregistered as that separate successor before identity selection. It replaces only
Nano Banana Pro with `z-image-turbo`, based on population capacity rather than model performance:
the already cross-validated continuous 52,600-row metadata prefix holds 6,876 eligible Z-Image rows,
while all four retained cells also exceed 192. The new `E49_C_OPENFAKE_V1` namespace will freeze
160+32 per cell at the first complete page. Asset URLs, image bytes and detector scores remain zero;
all later byte, decontamination, pairing and 20-gate final rules remain unchanged.

E49-C then froze successfully without new network access. The first qualifying 100-row boundary is
46,600, with GPT Image 2/Z-Image Turbo/Seedream v5.0/FLUX.2 Klein 9B/Midjourney 7 populations of
236/6,111/192/4,068/1,791. Namespace hashing selected exactly 192 identities per source. Contract
`0abae56a...d702` and reserve identity `f9f7bf74...69ec` bind the result; fresh asset requests,
image bytes and detector scores remain zero pending the separate feasibility gate.

The E49-C no-body feasibility gate then passed all 960 frozen identities. Fresh revision-bound
Viewer resolution plus HEAD metadata binds 241,736,938 OpenFake bytes; with Commons, the exact
expected network total is 2,948,318,716 bytes, leaving 1,346,648,580 bytes under 4 GiB. Aggregate
Viewer failures were handled only by querying already-selected rows individually. Asset contract
`7b71449e...1415` stores no signed URL, and image-body/model-score counts remain zero.

Both previously gated sources then became accessible. Hugging Face ITW-SM resolves at revision
`3060094f...f86`, but it is the same scientific distribution already consumed from the official
MediaEval E45 archive and cannot count twice. Datapoint resolves at the original pinned
`e1d8719a...c928` revision and confirms 30 current models, 14,952 images and 500 prompts. The audit
fetched only 1,737,709 metadata bytes (models, test responses and prompt references), with zero
image-Parquet/image-body/model-score bytes. Because E49-C was already identity/byte-frozen and
OpenFake is a purpose-built OOD detection split, it remains the sole final route; Datapoint stays
clean for possible post-final diagnosis rather than silently replacing the selected source.

E49-C transfer began and preserved 201 validated files before its strict JPEG assumption stopped on
selected Seedream row 8,770. The Viewer path ends in `.jpg` and S3 declares generic binary, but the
7,460,595-byte body is a valid 2,048 x 2,048 RGBA PNG. The row was not removed and no detector was
loaded. The transport contract now preserves received bytes while permitting only decoded JPEG,
PNG or WebP and requires per-source format/geometry disclosure; resume will reuse the 201 files.

While the exact transfer resumed, the next model-blind gate was implemented and committed before
production realization. It hard-binds the identity and download receipts, revalidates every payload
hash/format/geometry, compares original SHA-256+dHash against all protected roles plus the scored
Dotting diagnostic and selected StyleGAN2 final component, and applies only the preregistered rank
order. Fixed 1080-long-side JPEG-Q75 children receive their own protected/internal collision check.
The command refuses anything except 160 clean parent-child pairs per modern family and records
format/geometry disclosure. Twenty-one focused E49 tests pass; no OpenFake detector score or metric
has been created by this checkpoint.

The 2.7 GB Commons transfer path was also implemented before any REAL body was requested. It accepts
only the 1,100 rows of the frozen V2 contract, resumes verified originals, and requires exact byte
length, Wikimedia SHA1, JPEG decode and bound dimensions for every file. It additionally records a
local SHA-256 and any preserved EXIF make/model while retaining category, uploader and licence
provenance. Unexpected files, partial quotas or aggregate byte drift stop the run. Eighteen focused
E49 acquisition/download/evaluation tests pass; this checkpoint downloads and scores nothing.

The first Commons execution showed that eight parallel original-file workers are too aggressive for
Wikimedia: the service returned an explicit 429 after two complete files. The executor stopped and
no partial body survived. Before resuming, transfer policy was reduced to one request stream with
0.75-second pacing, a public-repository research User-Agent and bounded Retry-After/exponential
backoff up to 60 seconds. The two exact files remain reusable; no thumbnail, source identity, byte
contract or model boundary changes.

The paced retry then reached the first bound iPhone 15 Pro original without another 429, but the
decoder exposed a native-container detail: Wikimedia and system inspection identify a valid JPEG,
while Pillow reports `MPO` because the Apple capture retains an MPF multi-picture segment. This is
authentic camera-pipeline evidence, not corruption. The received bytes are preserved and JPEG/MPO
are now admitted as separately reported decoded formats, still requiring exact SHA1, dimensions and
size. No row was substituted and no model score exists.

The next iPhone original exposed the related EXIF geometry convention. Row 148,952,501 physically
stores 4,032 x 3,024 pixels with orientation 6, while Wikimedia correctly binds the displayed image
as 3,024 x 4,032. The validator now records both encoded and display geometry, swaps axes only for
EXIF orientations 5–8, and requires the display result to match the prebound API dimensions. It does
not rotate or rewrite the publisher original. A regression test covers this native-phone case; no
identity or score changed.

The paced Commons transfer then completed all 1,100 bound originals and exactly 2,706,581,778 bytes,
with 110 rows for each of ten devices. Every Wikimedia SHA1, local SHA-256, JPEG-family decode and
display geometry validates. The natural transport contains 861 plain JPEG plus 239 Apple MPO files;
1,074/1,100 preserve both EXIF make and model. Detailed receipt SHA-256 is `2511f0ad...7e04`.
Detector/model-score count remains zero, so this is still a reserve pending the already-committed
device-evidence, decontamination and Q75 freeze.

A score-blind EXIF audit then found one genuine metadata mismatch in the reserve: a file categorized
as Nikon Z 8 reports `NIKON D70`. Before selecting any REAL parent, the realization rule now
normalizes expected vendor/model spellings and valid Apple hardware aliases, requires agreement
when EXIF exists, and records the 26 missing-EXIF rows explicitly as category-only evidence. The D70
row will be excluded by fixed rank rather than silently relabelled. Thirteen focused tests pass;
no realization manifest or detector score exists at this checkpoint.

The first Commons realization invocation stopped before selecting a row or writing a child because
the transfer receipt intentionally contains SHA-256 but not dHash, while the audit read dHash before
deriving it. The missing stage is now explicit: revalidate each receipt SHA-256, reproduce its EXIF-
display geometry, derive dHash from oriented RGB pixels, and only then run protected/internal
comparison. Fourteen focused tests pass. The source identities, rank, quotas and model-score count
remain unchanged and no partial manifest exists.

The corrected Commons realization then passed. All 1,100 files reproduced their receipt hashes and
display geometry. Exactly two identities were excluded before selection: one prior protected dHash
overlap and the already-disclosed Nikon-D70 metadata mismatch. The fixed reserve still filled every
device at 100 parents. Of the selected 1,000, 979 retain matching EXIF make/model and 21 are clearly
marked category-only because EXIF is absent. Together with their 1,000 deterministic Q75 children,
the REAL component freezes at manifest SHA-256 `657be9bb...8e7b`. Scores remain zero.

The complete score-blind final assembly then passed. Exact Commons, OpenFake and StyleGAN2 component
hashes were joined; 200 StyleGAN2 Q75 children were created without protected/internal collision.
All 4,000 observation file hashes and display geometries reproduce, both conditions share exactly
2,000 parents, and every one of sixteen source quotas is exact. Publisher originals comprise 1,585
JPEG, 213 Apple MPO and 202 PNG files, while all 2,000 social-Q75 children are JPEG. The 3,937,874-
byte frozen final manifest hashes to `9744a9d2...5909`. Model scores and aggregate metrics are still
zero; this is the last commit required before locking E43-S to the final identities.

E43-S is now irreversibly bound to that manifest before inference. Contract SHA-256
`fecd724c...61dd` seals the 4,000 observation identities (`3cf565a1...5242`), three component hashes,
E43-S artifact `a3aec445...7390`, DINOv2-S weights `04d27f34...0081`, the 0.07940196245908739 binary
cut, 0.011505939625203613 selective REAL cut and all twenty gates. Training, source/row removal,
threshold change, score replacement and metrics-before-raw-lock are forbidden. At this commit score
and metric counts are both zero.

The one permitted E43-S inference completed every frozen observation with 100% coverage. Its
ordered 4,000-row raw stream is 1,005,967 bytes /SHA-256 `249f005c...10a8` and remains bound to
contract `fecd724c...61dd`. Only identity, label, source, condition and score were written; no AUC,
accuracy, false-positive rate, recall or selective metric has yet been calculated. This raw-stream
evidence is committed before the single allowed metric opening.

The single permitted E49-C metric opening failed 11/20 gates and is now consumed. Publisher
original passed 6/10: 100% coverage, AUC 90.24%, AI recall 94.30%, worst-family recall 91.88%,
automatic coverage 85.95% and uncertainty 14.05% pass; balanced accuracy 77.60%, REAL false-AI
39.10%, worst-device false-AI 71% and covered accuracy 76.27% fail. Social-Q75 passed 5/10: AI
recall rises to 95.50% and worst family remains 91.25%, but AUC falls to 86.89%, REAL false-AI rises
to 49%, worst-device reaches 84%, balanced accuracy is 73.25% and covered accuracy 71.01%.

The scientific diagnosis is unambiguous: E43-S generalizes strongly to all six unseen/current AI
families, including after recompression, but its low fixed cut does not generalize to these new
high-resolution camera pipelines. Canon EOS R5, Sony ILCE-7M4 and Nikon Z 8 are the largest failure
cells; phones also exceed the target except Samsung/iPhone 14 originals at 19%. This final may be
used only for disclosed diagnosis—not threshold selection, training, row deletion or a retry.
Report SHA-256 is `10fc0649...5573`; retry count is zero and Module-1 v1 remains unearned.

E51 is planned before any successor data or fit. The failed final proves a cut-only repair cannot
work: at 10% FPR the frozen ranking recalls just 72.30%/58.80% AI on original/Q75, and even requiring
both paired scores to vote AI leaves 32.80% REAL false alarms. MPO is not the culprit (36.15% FP
versus JPEG 39.90%), while high resolution has only weak overall score correlation despite a bad
>20 MP pocket. The next experiment therefore targets authentic/compression representation.

The preregistered path first audits new licensed, grouped camera sources; then compares only two
practical candidates on new CAL: a source-balanced frozen-DINO head with real transport hard
negatives, and a hybrid adding fixed residual/DCT statistics plus source-bias suppression. A
real-only arm may abstain but never certify REAL. Fresh publisher/device DEVELOPMENT must pass both
transports and the same ten gates before any E52 final. Datapoint's approved image bodies remain
untouched as possible future final AI evidence. E49 and its unused reserves are permanently barred
from successor fitting or selection, and Module 2 stays planning-only until Module 1 earns v1.

The corresponding REAL realization gate was committed before the Commons transfer. It binds the
future exact receipt, checks received originals for protected/internal SHA-256+dHash overlap, and
also protects final independence from the scored Dotting diagnostic plus both final AI components.
Within each immutable device rank it keeps the first 100 parents whose originals and fixed Q75
children remain clean. The command refuses any result other than 1,000 REAL parents and their 1,000
paired children, while retaining uploader, licence and EXIF-availability evidence. Twelve focused
tests pass and no detector import, score or aggregate metric exists at this method checkpoint.

One stale name-level boundary was corrected before a final manifest existed: E49's evaluator still
listed the five never-consumed Datapoint families from E49-A. Its AI source quota now names the five
exact OpenFake families frozen by E49-C plus unchanged StyleGAN2. Parent counts, all ten REAL device
cells, thresholds and twenty gates are identical. A regression test binds the six labels; this is a
pre-score contract alignment, not a source substitution after seeing performance.

The resumed OpenFake transfer reached 521 atomically verified payloads, but sequential resolution of
hundreds of sparse Viewer pages—not image bandwidth—was measured as its dominant delay. The process
was stopped cleanly with no receipt and no partial file admitted. Its restart strategy now groups 24
remaining pages while preserving the previously measured public limit of two concurrent Viewer
requests, followed by the same eight exact-body workers. Frozen identities, expected bytes, decode
rules and model boundary are unchanged; all 521 bodies will be revalidated rather than redownloaded.

The faster resume reached 850 verified bodies and then correctly refused Seedream row 43,863 even
though its HTTP headers exactly reproduced the asset contract. Diagnosis found one pre-score guard
mismatch: its already-bound 6,144 x 11,008 geometry is 67,633,152 pixels, making it the only reserve
row above the inherited 50 MP default. The decoder limit is therefore raised only to that exact
frozen maximum; the identity is neither removed nor replaced. The rejected body was not admitted,
and transfer errors now retain their actual validation cause. No detector or metric was opened.

E49-C OpenFake transfer subsequently completed with exact coverage: 960/960 selected files,
241,736,938/241,736,938 bytes and 192 reserves in each of the five modern generator cells. All files
decode; 958 are JPEG and two are PNG despite Viewer `.jpg` naming. The external detailed receipt
hashes to `4dfb942c...26c2`, while the compact committed evidence contains no signed asset URL.
Detector/model-score count remains zero, so these bytes are still only an unscored reserve pending
the already-committed identity and Q75 realization gate.

That model-blind gate then passed. All 960 OpenFake reserves decode without failure and none overlaps
a protected historical role. It exposed 26 repeated Seedream payload identities, each identical by
both SHA-256 and dHash; these were excluded rather than allowed to inflate the test. The fixed
headroom still leaves 166 clean Seedream candidates and at least 160 in every other cell, so the
prebound rank order freezes exactly 160 parents per modern generator. Their 800 originals and 800
deterministic social-Q75 children form a 1,600-observation component at manifest SHA-256
`38048803...7442`. Selected originals comprise 798 JPEG and two PNG; detector scores remain zero.

The complete E49-C assembly gate was also implemented before component realization. It binds the
future exact Commons and OpenFake manifests plus the already-frozen StyleGAN2 component, creates the
200 missing StyleGAN2 Q75 children without score-dependent replacement, and fails on protected or
internal child collision. It then reproduces every one of 4,000 observation SHA-256 values and
dimensions, enforces exact parent pairing and all sixteen source quotas, and records format/geometry
shortcut evidence. Twelve focused final/component/evaluation tests pass; no final manifest, model
score or metric is produced by this method-only checkpoint.

The final scorer was then split into three committed checkpoints before it could see any final row.
The first seals the complete manifest, component hashes, E43-S artifact, DINOv2-S weights, fixed
thresholds and twenty gates. The second alone performs restart-safe ordered inference and records a
raw-stream hash while exposing zero aggregate metrics. The third refuses to run until that stream is
complete and then opens the preregistered metrics and 10,000 parent-level bootstraps exactly once.
Fourteen focused final lock/score/evaluation tests pass; training, source removal, threshold repair
and a second attempt are explicitly forbidden.

E51's first preregistered checkpoint is now complete. The reproducible diagnosis binds the consumed
E49 manifest, raw score stream and final report by exact SHA-256 before joining all 4,000 rows and
2,000 original/Q75 pairs. It independently reproduces the cut-only impossibility, paired-consensus,
format and resolution findings in report `3e5caa86...bd70f`. The procedure performs no inference,
creates no candidate threshold and cannot select any future source or row. The project can now audit
new REAL sources without silently learning from the failed final identities.

The following metadata-only E51 source audit resisted a tempting shortcut. SCMI30 is genuinely
useful—9,937 high-resolution phone JPEGs across 30 normalized devices, selectable file by file—but
it can safely supply only TRAIN/CAL if whole devices are separated. SCIMD-17's 17,000-image claim
compresses to a 174 MB archive because every frame is already 224x224; it is therefore an auxiliary
resize hard-negative source, not evidence of native-camera generalization. RAISE is valid but
~350 GB and only three cameras, while Dresden, IMAGINE and SOCRatES fail current transport/terms
requirements. With no independent DEVELOPMENT publisher yet, evidence `9ddab57a...c4883` freezes a
zero-download decision instead of manufacturing progress from a scientifically weak split.

Before any E51 payload, the practical role route was preregistered. Existing legitimate TRAIN roles
plus SCIMD resized-real hard negatives may fit the candidates; 1,200 device-balanced SCMI30 native
JPEGs are reserved for CAL only; the IEEE SP Cup's 2,640 hidden-camera test images become independent
REAL DEVELOPMENT with unaltered/postprocessed cells; approved Datapoint becomes current-generator
AI DEVELOPMENT. None may enter E52 final. The IEEE rules gate must be accepted by the user before its
payload, so implementation may bind inventories but cannot silently bypass that agreement.

That route is now machine-bound without downloading a single image. Contract `975e8164...15e4`
freezes 1,200 balanced SCMI30 CAL parents, every one of the IEEE test split's 2,640 REAL camera
images, and 920 Datapoint AI reserves from five current generators. The AI design was corrected
before commit so all generators share exactly the same 23 prompts in each of eight categories;
model comparisons therefore cannot inherit a content mismatch. Only 800 clean AI parents may be
realized, at 20/category/model. The selected seven Datapoint shards total 3.22 GB, IEEE test totals
0.84 GB and SCMI30 CAL totals 4.25 GB. The IEEE payload remains blocked by the user's unaccepted
Kaggle competition rules; no legal gate was bypassed, no model was opened and no score exists.

The two preceding GitHub failures were dependency drift, not scientific or web regressions: the new
E51 metadata audit imported Kaggle's official client, which existed in the workstation environment
but not the declared CI experiment group. `kaggle>=2.2,<2.3` is now explicit in `pyproject.toml`;
the complete local 471-test suite remains green before the replacement CI run.

The user subsequently accepted the archived IEEE/Kaggle rules; the official 81,853-byte submission
metadata probe now downloads instead of returning HTTP 403. Before any image transfer, the E51
IEEE downloader was frozen against route contract `975e8164...15e4`: only the 2,640 selected test
TIFFs can enter, interrupted files resume, unexpected paths fail closed, and each admitted payload
must reproduce its byte count, decode safely and receive a SHA-256 receipt. No detector or metric is
available in this acquisition path.

The first bounded transfer then exposed a useful publisher quirk and failed closed: IEEE names the
test files `.tif`, but all eight concurrently staged bodies were 512x512 RGB PNG payloads. Zero were
admitted to the dataset root and no model opened them. The decode gate is corrected before resume to
require the measured PNG signature and exact 512x512 geometry. Identities, expected bytes, role and
selection remain frozen; only the false filename-suffix assumption changes.

The corrected resume safely admitted 494 rows /about 240 MB before Kaggle enforced HTTP 429. This
is a transport limit, not a dataset/model result; no receipt or score was produced and every admitted
file remains restart-verifiable. The next resume is globally paced to roughly 75 API requests per
minute across two workers and backs off 60–300 seconds on 429, trading speed for deterministic
completion without redownloading the verified 494.

A better official transport was then measured before further payload: Kaggle's single signed ZIP
supports HTTP Range, and its central directory exactly reproduces the frozen 5,391-file /
11,447,649,387-byte expanded inventory. The paced per-file API is superseded. A TLS-verified remote
ZIP reader now fetches only compressed ranges for the selected `test/test/*` members, checks archive
CRC and all existing admission gates, and never downloads the 2,750 excluded training images or
stores the signed URL. The 520 rows admitted by the earlier route remain valid.

That range transfer completed the IEEE component: 2,640/2,640 bodies, exactly 837,665,909 expanded
bytes and 836,795,134 compressed member bytes, with 1,320 unaltered and 1,320 postprocessed REAL
rows. Every publisher body decodes as 512x512 RGB PNG despite its `.tif` path and now has a SHA-256
receipt. Detailed receipt `09188d49...3794` reproduces contract identity `fc3657dd...fb05`; no
training image, detector score or metric was opened.

While that publisher quota cools down, the independent Datapoint acquisition method was prepared
without transferring a shard. It permits only the seven already-contracted Parquets at pinned
revision `e1d8719a...c928`, reproduces each remote byte count and full local SHA-256, and records that
the image columns and all detectors remain unopened. Its exact transfer ceiling is 3,220,281,593
bytes; the 920 paired reserves and later 800-parent target cannot change in this step.

The Datapoint transport then completed all seven pinned shards at exactly 3,220,281,593 bytes.
Receipt `18b8326a...bad1` binds their local full-file hashes and revision while preserving the 920
paired reserves and 800-parent realization target. Image columns, detectors and metrics remain
unopened; this is acquisition evidence, not a model result.

SCMI30's transfer path is likewise frozen before its first payload. The official v2 ZIP central
directory reproduces 9,940 members /35,592,872,377 expanded bytes, but the range reader may fetch
only the 1,200 contracted CAL JPEGs /4,247,339,334 bytes. It reads selected members in archive order,
checks ZIP CRC, decode, geometry, byte count and SHA-256, and refuses any deviation from 40 images per
device with equal Random/Similar totals. No model is importable from this acquisition command.

SCMI30's first execution stopped before payload because Similar paths legitimately include nested
scene folders such as `objects/`; the initial safe-path predicate had assumed exactly four parts.
The corrected pre-transfer gate still binds dataset root, branch and exact contracted device folder,
rejects traversal or suffix changes, and permits only descendants beneath that device. No identity,
quota or byte changed and zero SCMI30 images had been downloaded.

The valid single-reader transfer was then measured at 59 admitted images /231 MB in roughly three
minutes, too slow for a 4.25 GB component. It was interrupted cleanly before a receipt; the 59 exact
files remain restart-safe and the lone inadmissible `.part` was removed before restart. Four independent TLS-verified Range
readers now process four contiguous archive-order partitions with hash-derived staging names. This
changes transport throughput only; every identity, byte, CRC, decode and CAL quota remains fixed.

The four-reader restart completed the SCMI30 calibration component. It reused the 59 verified rows,
downloaded the remaining 1,141 and closed at exactly 1,200 files /4,247,339,334 bytes with no
staging partial. Every one of 30 devices contributes 40 rows, Random and Similar contribute 600
each, and all rows retain make/model EXIF. External receipt `01cc5921...f33e` and ordered identity
digest `c47d411f...1d12` bind the result. No model was loaded, no score was made, and CAL remains
separate from fitting and final evidence.

Before the SCIMD-17 archive byte, its transfer gate was made executable and testable. It requires
Zenodo record 17317613, DOI `10.5281/zenodo.17317613`, version `Version  1.1`, CC-BY-4.0, exactly
174,438,734 bytes and MD5 `37da574c9e8d9c0fd3a7c9bedc5d72a6`. Resuming cannot alter the
payload, unsafe ZIP paths/expansion fail closed, and the receipt explicitly marks this uniformly
224x224 source as REAL hard negatives for TRAIN only with zero decoded image bodies or scores.

After that method was committed, the SCIMD-17 archive completed at the exact 174,438,734 bytes and
publisher MD5. SHA-256 is `ef1fe3e7...0201`; safe inventory reveals 17,620 images /172,781,180
expanded bytes, refining the publisher's approximate 17,000 claim. Receipt `8b38fa82...b230`
still reports zero decoded bodies and model scores. The source remains TRAIN-only and cannot repair
or influence any calibration/development/final metric by role reassignment.

The pre-realization audit then found a protocol gap before it could contaminate a result: the E51
route supplied new REAL calibration images but no AI calibration arm. Moving Datapoint would destroy
its independent DEVELOPMENT role. E51 therefore pre-registers a fit-internal holdout instead: 20
score-blind historical TRAIN rows from each of 18 AI sources (360 total) are removed from fitting
and become paired AI-CAL. Separately, SCIMD-17 binds 120 archive candidates per device so identity
audit can retain exactly 100/device (1,700) as TRAIN-only resize hard negatives. No new body or score
was opened to make either choice.

The subsequent realization implementation keeps that order enforceable. It can open only SCIMD-17,
SCMI30 and the explicitly held-out historical AI rows; Datapoint/IEEE DEVELOPMENT is absent from the
command. All SCIMD reserve bodies are byte/CRC/decode/224x224 checked and protected by exact+dHash
before 100/device selection. TRAIN removes every AI-CAL parent, while CAL verifies 1,560 parents and
creates deterministic original/Q75 pairs. The code and tests are committed before execution.

That first execution failed closed before CAL or evidence on a historical AI-CAL dHash mismatch.
The file SHA-256 (`425c8c99...45e0f`), bytes and 224x224 JPEG geometry reproduced exactly; E42's
left-greater-than-right comparator yielded `46c6...3277`, while the later shared helper yielded its
inverse `b919...cd88`. This is a hash-convention mismatch, not data corruption. The rerun preserves
the pinned E42 hash for intentional AI-CAL reuse and expands protected checks to both complementary
64-bit conventions. No detector, threshold or DEVELOPMENT data informed the correction.

The next restart reached the new SCMI30 overlap gate and stopped before CAL/evidence on
`D17_rnd_150.jpg`: its dHash is the degenerate all-zero value (all-one under the historical
direction), shared by two unrelated protected images. Exact SHA-256 differs, and fixed 64-bit pHash
distances are 31 and 36 versus the predeclared <=4 near-duplicate boundary. E51 therefore uses dHash
as a cheap collision candidate and pHash as confirmation; exact, uncheckable or confirmed-near
matches still fail. This repairs identity precision only and uses no model score or DEVELOPMENT row.

A complete model-blind sweep then confirmed that this was the only true collision among 1,200
SCMI30 CAL parents. Twenty-five other degenerate dHash candidates clear at pHash distance 22–36;
`D04_black.jpg` alone matches at distance 0 and is explicitly publisher-labelled `no_content`.
Because the original route lacked decode headroom, a bounded amendment freezes the first five
previously unselected D04-Similar identities under the original namespace rank. Only these may be
fetched, and the first clean row replaces the black frame without changing device/branch quotas.

The amendment then transferred exactly its five rows /7,710,716 bytes. All passed decode and
protected identity checks, so original-rank head `D04_nat_45.jpg` replaces `D04_black.jpg`; no later
reserve was consulted for performance. Receipt `66e3d063...9722` closes the correction with zero
rejections and zero model scores while preserving all CAL quotas.

The fourth clean realization then closed both preregistered manifests. TRAIN contains 5,978 parents
(4,035 REAL/1,943 AI): all 360 AI-CAL identities are absent, SCIMD contributes exactly 100 from each
of 17 devices, no body failed decode and two conservative dHash collisions used the prebound
headroom. CAL contains 1,200 SCMI30 REAL plus 360 source-balanced historical AI parents, represented
as exactly 1,560 original and 1,560 deterministic Q75 observations. TRAIN hash
`41444640...77ef`, CAL hash `60688291...2356`; detector scores and DEVELOPMENT access remain zero.

### 2026-09-07 — E51 remote pre-fit safety audit (no new model claim)

The external LaCie disk was mounted successfully without a manual workstation action. No image
payload was downloaded and the serving artifact was not changed. Before fitting, the implementation
now checks frozen TRAIN/CAL hashes, all declared payload bytes, paired CAL labels/roles and decoded
RGB identity. A single explicit LANCZOS dHash convention is recomputed; radius-4 candidates are
confirmed using the existing radius-4 pHash63 screen. This is a duplicate-candidate check, not proof
that every semantic scene/prompt is independent. Restart caches never waive input-byte verification.

Two earlier claims need qualification without rewriting their archive entries. Inverse dHash keys
are NOT generally equivalent: equal neighbours and different resampling break complementation.
Also, `e49/dotting/manifest_unscored.json` was a stale path; the real protected manifest lives under
`e49_d1_dotting`. The old reader silently skipped it. The path is repaired and all 20 required
protected manifests must now exist or admission raises an error. Complement keys remain only a
legacy candidate heuristic, not evidence of complete perceptual protection.

The independent metadata inventory verifies 22 pinned manifests/contracts, including the v2 E49
Commons/StyleGAN2 and OpenFake asset reserves (1,100/240/960 identities). It finds zero exact
metadata-key overlaps under the declared historical-reuse policy. It does NOT claim a canonical
pixel comparison against every protected image or coverage of all superseded reserve contracts;
those remain explicit pre-fit work. Evidence: `evidence/e51_protected_inventory.json`.

SCIMD has eight `chatgpt-*` filenames, two in selected TRAIN. The publisher's 2,388,083-byte metadata
table reproduces MD5 `47279dd6c20ba1e9da8bef11623b9da2`; all eight carry INFINIX/X6851 camera fields.
The two selected images were visually inspected and appear to be camera photographs of laptop
screens. They were not relabelled based on filenames. Publisher metadata and appearance support,
but do not prove, authenticity. Evidence: `evidence/e51_scimd_filename_review.json`.

Code verification: **500 Python tests passed**, compilation and diff-whitespace checks passed.
No threshold was selected, detector scores remain zero, and this checkpoint is not a model accuracy
improvement. The first audit invocation exposed a missing `condition` on historical parent-only
rows; it wrote no final report. The reader now explicitly treats those original files as original
observations, with a regression test, and resumes byte-verified fingerprints.

### 2026-09-09 — offline recovery and protected-body resolution

The interrupted pre-fit run had actually completed: all 9,098 TRAIN/CAL observations were verified,
with **zero cross-role matched parent pairs** under the canonical byte/pixel/dHash+pHash screen.
Report SHA-256 `021800783a7d0ed4159856e5fa72c944609dc585d5aea28620d3c0a825d577a2` is now archived
in `evidence/e51_prefit_identity.json`. This passes only that identity check, not a model benchmark.
No training process or download was running when the connection dropped again.

The next offline step resolves protected source bodies without extracting or downloading them.
Historical E33/E36/E39 paths are relative to their manifest directory, not the working directory;
DDA/COCO and ITW-SM members are read from their already-local bound ZIP routes. The locator checks
manifest hashes, file/member sizes, traversal, ZIP member type/encryption and conflicting SHA-256
expectations. It finds **117,898 local body locations** (73,165 files /44,733 ZIP members), zero
unresolved body locations and 50,577,346,337 existing image bytes to verify. These are disk-read
bytes, not network bytes. All merged locations have a prior content SHA-256. Locator manifest hash
`a51cb45736641fa4e93c1d2171f17c06bf939cd4257effd78c8f6498c711eb81`.

The 9,800 metadata-only references are retained explicitly; many are parent/reserve references,
not missing image bodies. Identity joining, superseded-reserve coverage and canonical protected-
pixel comparison remain pending. Do not infer training permission from successful path resolution.
No new detector score, threshold, training artifact or demo change was produced. All **504 Python
tests pass**. No new image data or dependencies were downloaded; after the latest connection loss,
only local work was performed. GitHub publication is deferred while offline.

The user's generated-image idea is recorded but not executed: an image-generation tool can create
a separate synthetic pool, but requires network transfer and must retain prompt/model/date/role
provenance. It is neither necessary to spawn a separate agent nor valid to mix generated training
examples into the final test. Existing data remains the priority under the mobile-data constraint.

### 2026-09-09 — protected-pixel runner and fixed E51-B feature implementation

Implemented and started the actual offline protected-image comparison, not another download.
`ml/experiments/e51_protected_pixels.py` binds locator hash `a51cb457...eb81`, checks current
file/member bytes and SHA-256 (plus ZIP CRC), and computes the same full-resolution canonical
fingerprints. Bounded worker batches and a local SQLite cache avoid repeating fingerprint
computation after interruption; resumes still verify current bytes. Progress is stored under
`e51/audit/protected_pixels_progress.json` on LaCie. It checks new SCIMD/SCMI against all protected
bodies and every E51 TRAIN/CAL observation against E49. No network or detector is used.

The first run reached 64,000 locations, caching 63,996 distinct fingerprints, then stopped safely:
six historically protected RR images exceed the default 100 MP decode cap. This was not an internet
failure. Their prior manifest hashes/geometry bind 102.96–178.56 MP. An explicit six-SHA allowlist
now processes only those images serially, at full resolution, preserving Pillow bomb checks and
the ordinary 100 MP default. No blanket size limit was removed and no training example was admitted
by this exception. The corrected run is in progress; no completed protected-pixel result is claimed.

The predeclared E51-B hypothesis now has tested feature code in `ml/experiments/e51_features.py`:
eight normalized DCT bands plus eight residual/gradient statistics per shared E42 crop, aggregated
across three crops by mean/std into 32 features. E51-A stays at 3,072 DINO features; E51-B would use
3,104. No filenames, EXIF fields or explicit format/resolution scalars enter this branch, but
pixel-level source shortcuts can still exist. Fixed C=0.01 and source/parent/class-balanced fitting
remain planned; this code has not yet extracted real-data features or trained a new head.

Verification: **512 Python tests passed**, compilation and diff-whitespace checks passed. Existing
model, demo, thresholds and benchmark scores are unchanged. Protected/reserve admission remains
open; after closure, fit and select on CAL, use fresh DEVELOPMENT once, and label any old-test rerun
as regression/diagnostic evidence rather than a new independent final. Work is committed locally;
no GitHub push is attempted while the user reports a disconnected/mobile connection.

### 2026-09-09 — E51 reserve closure and fixed offline training method

The reserve audit is complete: all 9,800 metadata-only references joined to known local bodies;
434 supplementary bodies were byte-verified and canonically compared, with zero matched E51
parent pairs and zero reserved-identity hits. Report SHA-256 `2c2cdb07...9ad36`. The 419 superseded
Commons identities never downloaded remain forbidden; their unavailable pixels are explicitly
not claimed as checked. Zero image bytes downloaded. The main 117,898-location protected audit
has verified its body inventory and is completing the 9,098 query observations.

Before model scores, PLAN now fixes conservative whole-parent exclusions (maximum 5% per role,
with CAL device/source minimums), three symmetric TRAIN views, and exactly two C=0.01 logistic
heads: frozen DINO global/texture A, or A plus 32 residual/DCT features B. Train-only weighted
standardization and class/source/parent weighting are used. CAL alone sets the common
original/Q75 REAL-safe threshold and selective band; the unchanged AI and coverage gates still
must pass. Three deterministic lbfgs seed repetitions are not three independent scientific trials.

Added admission, restartable hash-bound extraction, per-candidate fitting checkpoints, and
tests for paired-parent removal, threshold ties, invalid scores, missing classes and prevention
of double JPEG compression. **523 Python tests pass.** No candidate has been fitted at this
checkpoint. Any later E49 rerun is regression-only, cannot select the winner, and cannot become
another independent final. Serving remains unchanged; commits remain local under mobile data.

Admission subsequently completed: protected report `2f070e7d...c1e54` has zero matches in both
comparisons after all 9,098 query observations. `evidence/e51_admission.json` binds admission
`c406edb6...7a39b`, retaining 5,978 TRAIN /1,560 CAL parents with no exclusions. Every CAL device
still has 40 originals, every AI source 20. The fixed 21,054-view feature extraction may now run.

The user-requested old/new comparison is implemented before fitting in `e51_regression.py`:
both fixed A/B candidates, all original 4,000 E49 observations, no threshold refit or winner
selection. It binds artifacts/CAL/manifest/prior-score hashes and reproduces the old E43 scores
(maximum allowed absolute numerical difference 0.00005) before accepting a comparison. Reports
include per-source errors and paired source-stratified parent-bootstrap differences. The 2,000
resamples describe this fixed test population, not unseen-publisher uncertainty. Regression is
allowed even for a CAL-rejected candidate solely to explain failure; fresh DEV stays unopened.

Feature extraction completed offline on MPS: 21,054 views, 3,072 DINO +32 residual dimensions;
8,916 hash-pinned historical backbone views reused and 12,138 newly computed. Feature archive
SHA-256 `55f92db2...82d0a`, admission `c406edb6...7a39b`. No classifier scores existed at the feature
freeze. All 526 Python tests pass, including an actual synthetic-data fit and interrupted-result
recovery that proves completed candidate checkpoints do not retrain. Fixed real-data fitting started.

### 2026-09-09 — E51 A/B trained; A selected on CAL only

All six fixed head fits completed. Both A and B pass the unchanged CAL gate, so A wins by the
predeclared simplicity rule (not by an E49 result). A artifact `60d56c0b...b6b39`, threshold and
REAL cut both 0.3316505551338196. CAL original: AUC 0.997699, BA 97.00%, REAL false-AI 2.67%,
AI recall 96.67%, worst-device FP 20%, worst-AI-source recall 90%. Q75: AUC 0.998319,
BA 97.56%, REAL FP 1.83%, AI recall 96.94%, worst-device FP 15%, worst-AI-source recall 90%.
Automatic coverage is 100%; accuracy 97.18%/97.88%. This is a CAL-selected operating point,
not independent final evidence or a calibrated probability. B's additional residual branch does
not justify extra complexity: original BA 96.92%, Q75 97.44%. Seeds reproduce exactly (max
score difference zero), as expected from deterministic lbfgs. All artifacts and CAL scores are
hash-bound in `evidence/e51_cal_result.json`. Serving is unchanged; fresh DEV and E49 regression
have not yet been scored at this checkpoint.

Fresh DEV realization method is now frozen before Datapoint image-column access. It reads only
the seven already-local pinned Parquets, decodes the exact 920 reserved bodies and pairs these
plus all 2,640 IEEE files with Q75. Canonical screens cover protected history, E51 TRAIN/CAL,
reserves and internal cross-parent matches. Any IEEE match stops admission; any AI match removes
the entire shared five-model prompt group, with the first 20 remaining ranked prompts/category
selected. Quota exhaustion stops without scores. The resulting target is 3,440 parents/6,880
observations. Hidden IEEE device ids and 512px publisher preprocessing are explicit limitations;
paired prompt sets must not be counted as 800 independent AI prompts. No download is requested.

### 2026-09-09 — measured E51 versus consumed E49: authentic safety improves, AI recall drops

All 4,000 E49 observations were rescored with frozen A/B and their CAL thresholds; the old E43
scores reproduced **exactly** (maximum difference 0.0). A original REAL false-AI falls 391→159
of 1,000, but missed AI rises 57→225 of 1,000. Balanced accuracy increases 77.60%→80.80%,
while AUC falls 0.902425→0.884358. Q75: REAL false-AI 490→307, missed AI 45→179,
BA 73.25%→75.70%, AUC 0.868850→0.843796. All original gates still fail. This is not a
universal representation improvement or deployable success. Source-stratified paired 95% bootstrap
BA-delta intervals are +1.50 to +4.85 pp original and +0.85 to +4.10 pp Q75; they do not cover
new-publisher uncertainty. B remains unselected: BA 80.75%/75.85%, FPR 17.2%/31.3%,
AI recall 78.7%/83.0%. No test-informed cut, winner switch or serving promotion occurred.

Fresh DEV scorer is fixed before its first score: A only, same CAL cut and admitted identities,
per-transport metrics, source errors, paired-prompt bootstrap (each shared AI prompt is one
five-generator cluster). IEEE hidden device ids remain an unverified dimension even if observable
group checks pass. The existing benchmark result is archived in `evidence/e51_e49_regression.json`.

DEV admission v1 stopped safely with zero model scores. Canonical audit of 7,120 reserve
observations against 122,782 protected bodies found one AI parent perceptually matching a protected
body and seven internal unordered candidate pairs (five IEEE REAL, two AI). There are no exact
encoded/pixel duplicates and no protected REAL match. Audit `975569d3...258e8` is preserved.
All ten flagged IEEE bodies were visually inspected before scores: four pairs show the same scene
with capture/processing differences, and a smooth grey pair is ambiguous. A score-blind grouping
amendment now retains all 2,640 REAL images but conservatively groups those pairs into 2,635 detected
scene clusters. This repairs statistical independence accounting, not the classifier or threshold.
The original audit match sets must reproduce exactly; external REAL overlap still blocks admission.
AI shared-prompt exclusions/reserves remain unchanged. DEV scoring intervals are amended to retain
REAL scene clusters as well as five-generator AI prompt clusters; all metrics/cuts/rows remain fixed.

DEV admission completed after all original matches reproduced exactly. Manifest
`1b1882f3...ea224` binds 6,880 observations /3,440 image parents: 2,640 IEEE REAL
(2,635 detected scene groups) and 800 AI (160 shared prompts x5 generators). Grouping amendment
`10792cb7...a495a`, selected A `60d56c0b...b6b39`, zero scores at freeze. Source quotas remain
1,320 unaltered +1,320 postprocessed REAL, and 160 each FLUX.2 max, Nano Banana 2, GPT Image 2
high, Ideogram v4.0q, Seedream 5.0 Pro. All 532 Python tests pass. One-shot A scoring may now run.

### 2026-09-09 — E51 fresh DEVELOPMENT completed; observable checks pass, universal proof does not

The frozen A (`60d56c0b...b6b39`) scored all 6,880 observations once at the unchanged CAL cut
0.3316505551338196. Raw score SHA `088718f8...724be`; report `evidence/e51_development_result.json`.
Original: AUC 0.993367, balanced accuracy 92.85%, REAL false-AI 44/2,640=1.67%, AI recall
699/800=87.38%, total/covered accuracy 95.78%. Q75: AUC 0.989087, BA 92.08%, REAL
false-AI 52/2,640=1.97%, AI recall 689/800=86.13%, total/covered accuracy 95.26%. Coverage
100%, no abstentions. Lowest generator recall is Ideogram 78.13% original/79.38% Q75.
Paired-scene/prompt-cluster 95% BA intervals are 91.28–94.38% original and 90.45–93.74% Q75.
All observable predeclared checks pass. Camera ids are hidden, IEEE bodies are publisher 512px,
and further scene dependence may remain; worst-device and native-resolution guarantees are not
established. This is a newly measured DEV success, not IEEE endorsement or an E52 final pass.

Interpretation jointly with E49: authentic safety and balanced accuracy improved on the old
benchmark, but AI recall and AUC fell there. The new-source DEV is strong, yet it cannot erase
known native/transport failures. A stays a research candidate; current serving/artifacts remain
unchanged. PLAN now makes the next work explicit: TRAIN-only source-held-out augmentation research
with an AI-recall guard, then genuinely unused native/modern-source E52 preflight. No download or
test-informed weight/cut changes. Completed tests are not to be resumed merely because their
immutable raw checkpoint filenames retain `.partial.jsonl`; the final reports prove full coverage.
532 tests, compilation and whitespace checks passed. Every current result is recorded and committed
locally; no network download or GitHub push occurred. No model/test process is left running.

### 2026-09-09 — E53 research plan: reduce authentic errors without trading away AI recall

User requested current ML research and an existing-data-only improvement plan, then asked to
resume after losing internet. Reviewed E43/E51 fitting code, E49 paired regression, E51 DEV,
historical failed fusion/adaptation results and dataset eligibility records. Read primary B-Free
(CVPR 2025), Community Forensics (CVPR 2025), Effort (ICML 2025) and NTIRE 2026 publications;
source links and the limits of transferring their methods are recorded in PLAN. No dataset,
weight, code or dependency download, new fitting/scoring, model promotion or push in this update.

Critical distinction: E43 and E51 changed fit populations as well as views/calibration. E43 fitted
8,844 parents/19,648 views versus E51's 5,978 parents/17,934 TRAIN views. The observed E49 AI
recall loss is not explained by a controlled one-factor experiment. New-source E51 DEV success
cannot erase the same-population E49 regression or establish improvement over an unmeasured
comparator on that DEV. No new performance gain is claimed by this planning work.

Rehashed the C3 manifest (`0b6656a2...91eb`), resolved all 18,154 historical TRAIN locators
(9,073 REAL/9,081 AI), statted loose files and read 35 Parquet footers without opening image
columns. Loose-image logical bytes total 42,135,059,473, excluding Parquet bodies. Compact audit:
`evidence/e53_local_inventory_plan.json`. This proves availability only; latest role, licence,
source, body-hash and duplicate checks must precede admission. Old TRAIN is not automatically
eligible now and cannot become an independent final. DATASETS records exact source counts.

PLAN now defines E53 as research, leaving E52 as the independent final: admit existing TRAIN;
run a bounded data-pool x compression/resize-order study with fold-refitted comparators; reject
REAL gains accompanied by AI recall loss; consider restricted DINO adaptation only if the cheap
study fails; freeze before diagnostic regression and a genuinely independent final. Existing
class/source balance, Q75 and intermediate/texture features are acknowledged as already present.
All selection stays inside newly partitioned eligible TRAIN. Previously trained heads cannot be
used as leakage-free out-of-fold baselines or teachers. Statistical uncertainty is not a licence
to call an unproven no-loss result equivalent. If local independent final data is insufficient,
stop at an acquisition specification without downloading. Module 2 protections remain unchanged.

Planning validation passed: inventory JSON class/source totals, byte sum and manifest SHA match;
HISTORY, EXPERIMENTS and DATASETS preserve their entire previous content as append-only prefixes;
`git diff --check` is clean. This is a documentation/evidence-only change, so the model test suite
was not rerun and no new numerical model result is implied.

### 2026-09-09 — office execution authorized, six TRAIN-only arms preregistered

User approved approximately 90 minutes of offline optimization/preparation, with downloads deferred
until their explicit home-network approval. E53 starts with the existing admitted E51 TRAIN pool,
not unadmitted C3 expansion. Six fixed arms cross full/mean-only DINO aggregation with old two-view,
E51 three-view and clean plus compression/resize-order views. C=0.01, three source-component outer
folds, disjoint inner CAL and refitted scalers/heads; all results must be retained. RR topics are
grouped as one publisher, shared-prompt AI batches stay together, and no old fitted head is used
as an honest out-of-fold baseline. Historical explicitly consumed E51 replay keeps that recorded
TRAIN role; this does not make it fresh E36 evidence. E51 CAL/DEV remain protected. PLAN records
this bounded first slice before any E53 model scores; current serving and external tests stay unchanged.

E53 contract `95e9ea22...1d1c` froze 5,978 TRAIN parents, 11 conservative publisher components,
three outer folds, disjoint inner CAL and six arms. All TRAIN bytes were rehashed; canonical
screens against E51 CAL/consumed DEV find zero matches. One internal RR REAL near-duplicate pair
stays within its publisher component. The 11,956 new order/scale views completed, feature SHA
`71e6d1d2...5b6f7`. All 18 fold heads were trained/evaluated without opening a new external test.

The six-arm source-held-out result is archived in `evidence/e53_source_held_out_result.json`.
No arm passes the AI-preservation research guard. Mean-only features improve pooled AI recall
and REAL errors versus full aggregation, but regress on individual AI sources and do not establish
both REAL improvements statistically. These TRAIN-derived, deliberately source-held-out results
are not a fall of the unchanged served model from its old benchmark score. Comparators are newly
fitted fold heads with different training subsets, not the deployed/E43/E51 full fitted model.
Old two-view versus three-view fits also change total mean-one weight mass, a disclosed confound.

Exact duplicate-crop prototype: 60 predetermined TRAIN parents, 120 clean/Q75 views, 360 input
crops versus 316 unique crops. Three warmed timing passes show median 2.0802s→1.8546s
(1.1216x throughput, ~10.84% elapsed-time reduction); maximum feature/score difference 0.0,
zero decision flips across four passes. Small-sample engineering evidence only, not accuracy gain
or production-wide equivalence. Prototype remains separate from serving and frozen E53 extractors.

Native-source audit completed: 20,826 original bodies verified (16,292 candidate TRAIN +4,534
protected C3 CAL), 134,196 canonical protected references. Fifteen protected match observations
plus internal propagation exclude 17 parents, leaving 16,275 candidates; 83 internal pairs are
recorded. Audit does not itself authorize training. Supplementary E51 reserve check covers the
120 unselected AI parents/240 original-Q75 bodies beyond the already-protected 800 selected AI;
zero matches against all 16,275 native candidates. All 920 E51 AI reserve parents are accounted for.

The predeclared native-expansion contract then selects 5,652 parents: 3,000 REAL and 2,652 AI,
excluding every detected internal-pair endpoint, preserving source/group diversity and original
C3 research terms. Full and mean-only three-view heads are the only two extra arms. Added rows
inherit existing publisher-fold assignments and enter FIT only; CAL/validation images stay exactly
unchanged. Total sample-weight mass is fixed to the corresponding E51 FIT mass to control effective
regularization. Original-based features are now extracting; no expansion result is claimed yet.
548 Python tests passed at this checkpoint. No image/weight download, GitHub push or serving change.

Further bounded controls preserve all prior results: full/mean features with C=0.001 and with
row-L2 normalization at C=0.01, unchanged E51 three-view population and folds. All four completed;
none passes the full preservation guard. The highest pooled BA among these controls is mean/C=0.001
(74.82% clean /74.43% Q75, AI recall 63.00%/61.19%, REAL false-AI 13.36%/12.34%). These are
the same difficult TRAIN source-held-out trials, not the current full-model benchmark score.
Evidence: `evidence/e53_source_held_out_controls_result.json`.

New reusable `pixelproof.training_weights.balanced_parent_weights` replaces quadratic repeated
parent masks with counters. On all 17,934 E51 TRAIN view records, output weights are bitwise
identical (SHA `0ebb98a9...06147`), while median construction time falls 0.31790s→0.00915s,
34.75x for this step only. It additionally rejects contradictory parent label/source identities.
The helper is used in the new head controls; historical frozen fit code is not edited.

Metadata inspection of the original three-fold protocol shows E36 REAL and E32 FLUX/Qwen
components never appear in FIT (only CAL/validation). This is a conservative but inefficient
data-use choice, not test contamination. A separately frozen coverage-v2 protocol jointly chooses
inner CAL groups while preserving every original outer validation assignment, both-class minimums
and component separation, requiring every component to train at least once. It repeats the same
twelve already-declared arms, not a new hyperparameter sweep. Original results remain preserved;
v2 reuses consumed TRAIN validation and cannot serve as independent confirmation. Its contract
is frozen before any v2 model score. Native features and full-TRAIN numerical parity are ongoing.

Offline runtime registry check also passes for all six required artifacts (E20, legacy CNN/ResNet,
two statistical heads, CF-ViT), and `pip check` finds no broken requirements. Serving code confirms
canonical E20 plus optional R1b research presentation; E43/E51/E53 research metrics must not be
described as the current web demo's measurements. No runtime/model selection change was made.

### 2026-09-09 — E53 native replay and coverage-v2 completed; no promotion

Native feature extraction completed: 5,652 added TRAIN parents/16,956 three-view observations,
SHA `82ab703a...82c1c5`. Six native-expanded fold fits complete the twelve v1 configurations;
the separately frozen coverage-v2 repeats the same twelve configurations in 36 further fits.
Total E53 classifier fits: 72, not 72 independent tests. All eleven source components appear in
FIT at least once in v2. Fold 2 legitimately adds zero native rows because those publishers are
held out, not because extraction failed. Every old v1 result remains preserved.

No configuration passes the combined AI-preservation/REAL-improvement research guard in either
protocol. In v1, native full features raise AI recall from 50.13%/51.67% to 77.61%/75.76%, but
REAL false-AI rises from 15.64%/15.17% to 23.74%/22.97%. V2 repeats the same qualitative conflict.
Mean-only alternatives help pooled metrics but lose specific AI sources; a higher average does
not meet the user's no-AI-loss requirement. These are difficult source-held-out TRAIN fits,
NOT a decrease of the unchanged served/full E43 model from 94.3% AI recall.

Frozen-prediction diagnostics identify both calibration transfer and representation limitations.
For the v1 RR-publisher held-out fold, native-full clean AUC improves 53.29%→69.32%, yet AI recall
at an evaluation-derived FPR<=10% is only 36.49%. Its CAL-selected cut catches 77.39% AI while
accusing 55.76% REAL. This cannot be repaired by claiming 77.39% recall in isolation. Diagnostic
ROC cuts are not deployed or used for refitting. Source transitions retain both rescues and losses.

TRAIN input audit: 2,314/4,035 REAL (57.35%) versus 353/1,943 AI (18.17%) are exact 224x224
publisher/legacy inputs. Added native parents are all >=512 on the short side; the combined
candidate inventory still has 32.89%/7.68% exact-224 REAL/AI. Median crop-dispersion norms change
from 10.70 REAL/23.56 AI to 27.15/27.13 in the combined pool. These are source/processing
associations, not proof of causality, a trustworthy authenticity heuristic, or per-fold FIT counts.

Full exact-crop verification passed on 5,978 TRAIN parents/11,956 clean-Q75 views: 35,868 input
crops become 30,103 unique crops, feature error=0, score error=0, decision flips=0. The full check
is equivalence, not a timing benchmark or accuracy gain; the separate small timing result remains
the only speed estimate. Serving and frozen research extraction implementations remain untouched.

Last-two-block preparation succeeded on MPS using eight deterministic TRAIN parents/24 crops,
one warmup and three timed AdamW steps. 3,553,537 parameters are trainable; frozen backbone
parameters retain their hash, all gradients are finite, differentiable aggregation error is
2.38e-7 and converted-head score error 6.71e-8. Median step time is 0.20565s; sampled driver
allocation is 1.222 GB, not an allocator peak. No adapted candidate is saved, no CAL/test is
scored, and no quality improvement is inferred from a four-step repeated-batch resource probe.
The next real study must initialize its own FIT-only head, not reuse this E51 probe head.

Evidence: `e53_source_held_out_expanded_controls_result.json`, `e53_coverage_result.json`,
`e53_diagnostics.json`, `e53_shortcut_audit.json`, `e53_crop_dedup_full.json`,
`e53_adaptation_probe.json` under `evidence/`. PLAN now gives the home handoff: existing native
bytes first, fixed restricted adaptation next, no new download until explicit permission and a
source/budget contract, and no relabelling consumed tests as E52. Full native order/scale expansion
and real adaptation training are explicitly deferred. Module 2 protection and serving remain intact.

Final serialization replay reloads all 72 saved fold heads from hash-verified artifacts and reproduces
286,944 already-consumed prediction observations exactly (max score difference=0, decision flips=0).
Receipt: `evidence/e53_artifact_replay.json`. This is reproducibility, not another accuracy test.
Final local verification: 574 Python tests pass, compilation passes, `pip check` finds no broken
requirements. One pre-existing Starlette/httpx deprecation warning remains; no dependency is changed
or downloaded. Existing web/serving code remains unchanged; no new web build is claimed.

User then requested pausing while travelling home. No training or download is left running; all
completed methods/results are saved and the next study is deliberately not started. Local E53
working artifacts/caches occupy about 2.2 GiB on the existing disk, not network download volume.
Resume only after the user returns; any new dataset/weight download still needs explicit permission.

### 2026-09-09 — home continuation authorized; E54 preregistered

User returned, confirmed the external disk, and authorized dataset downloads while asking for
continued development. No paid image-generation/API spend is inferred. Existing bytes come first.
E54 compares continued head-only training with last-two-DINO-block adaptation plus cosine feature
preservation, both warm-started from the appropriate FIT-only E53 native fold head. Same admitted
rows/folds, two complete epochs, fixed seed/LRs/weighting/clipping and final-epoch selection are
recorded in PLAN before execution. No validation-driven early stopping or old-test threshold tuning.
Original E43/E51/served models and Module 2 protected data remain untouched. New public acquisition
requires a separately bounded, licensed, group-role manifest; general permission is not bulk-download
or final-data reuse permission.

E54 data-preparation implementation correction before any E54 fit/score: NPZ member access inside
the parent loop repeatedly decompressed the full teacher array. The CPU-bound preparation was
interrupted before producing a teacher/crop cache. Load each archive member once, preserving
exact array values/order and all roles. Original data contract `b96447f0...bbd99` remains archived;
the corrected code is bound to a separate `data_contract_v2.json` and evidence receipt, not a
silent rewrite of the first contract. No model metric or scientific parameter changed.

### 2026-09-09 — E54 independent current-AI reserve acquired, not scored

Official Microsoft–Northwestern–WITNESS revision `c93abf43e8157558a0e60aab7df4278b2c539253`:
300 selected fully-generated AI images, exactly 50 each Firefly v4, Flux 2 pro, Imagen4,
MAI image2, Midjourney v8, GPTimage2. Frozen LFS image identities/sizes were committed before
image transfer. Completed 229,880,188 verified image bytes (229.88 MB /219.23 MiB), below the
768 MiB cap; conservative charged-request bytes also equal 229,880,188 in this successful run.
Metadata freeze used 416,467 bytes; earlier exploratory listings are additional metadata traffic,
including a discarded/truncated ~14.5 MB recursive Git listing. No dataset weights or paid API.

Model-blind audit compares against 150,483 unique canonical historical/native/development/reserve
bodies, covering all 11,630 current TRAIN originals and all 920 latest AI reserve parents.
Zero cross-protected or internal matches under byte/RGB and the established dHash/pHash heuristic.
All 300 remain unscored reserve candidates. This is not exhaustive semantic/prompt deduplication,
not evidence of model quality, and not a balanced E52 final admission. Whole publisher stays
protected, including unknown prompt relations. MNW terms prohibit training and threshold tuning.
Receipts: `evidence/e54_mnw_reserve_{manifest,download}.json`, `evidence/e54_mnw_overlap*.json`.
Future TRAIN admission must include this new reserve, not only the older frozen inventories.

### 2026-09-09 — E54 cache complete and restricted training started; REAL reserve closed

Completed 11,630-parent /34,890-view crop/teacher preparation from existing local originals.
Teacher SHA `101c539958f72913273a0a7e91f9ee7f7b2c025d73fee956177d7fcf8c9b62d8`; receipt
`evidence/e54_crop_cache.json`. Every crop archive is hash-bound to the corrected data contract.
Head-only controls finish all three folds; last-two-block adaptation now executes the same fixed
two epochs. FIT parents per fold: 8,070/6,992/2,360; fixed steps per arm: 6,054/5,244/1,770.
No adaptive early stopping, serving replacement or independent final opening. Training-input
parity checks pass before updates. End-to-end CPU AdamW checkpoint continuation also reproduces
uninterrupted parameters bit-for-bit in its added unit test; this is not a claim of all-device bitwise
determinism. Full suite before that extra test: 586 passed, one existing Starlette warning.

Official Google HDR+ camera-only reserve: 100/100 full-resolution final JPEGs, 423,955,391 verified
bytes (423.96 MB), GCS object generation/MD5 verified and local SHA256 recorded. Excluded all
20 explicit synthetic folders before selection. Historical Nexus/Pixel HDR finishing, not modern
2026 captures or generative content; preserve CC-BY-SA-4.0 attribution. Camera EXIF includes code
names and product-name aliases: nine strings must not be claimed as nine distinct camera models.
There are 81 capture-session/day proxies, not 81 proven independent scenes. Against the exactly
reproduced 150,483-body reference snapshot plus all 300 MNW images, zero cross/internal matches
under the declared identity heuristic. All 100 remain unscored reserve candidates, not balanced
E52 admission. Receipt `evidence/e54_hdrplus_overlap.json` retains geometry and raw device labels.

Total new image bodies this home slice: 653,835,579 bytes (653.84 MB, 623.55 MiB), 400 parents.
No full dataset archives, RAW inputs, model weights, dependencies or paid API images downloaded.
Metadata traffic is additional. Original bodies live on LaCie; only code/method/evidence receipts
are committed locally. FiveK was investigated but no FiveK image or RAW decoder was acquired.

### 2026-09-09 — E54 completed: measurable paired gains, no accepted replacement

All six prescribed fits complete: three continued-head controls and three last-two-block adapted
heads, each exactly two epochs. Adaptation took 2,363.42/2,028.13/741.14 seconds per fold including
post-fit scoring. Frozen parameters and pre-fit feature/head parity checks passed. Pooled consumed
TRAIN OOF results, clean / Q75 respectively:

- Native logistic baseline: AI 66.39/65.11%, REAL false-AI 19.50/18.76%, BA 73.44/73.17%.
- Continued-head control: AI 66.65/65.11%, REAL false-AI 19.48/18.64%, BA 73.58/73.23%.
- Restricted adaptation: AI 68.76/66.80%, REAL false-AI 17.50/16.85%, BA 75.63/74.98%.

This is useful representation progress, but not accepted: supported AI sources still lose some
examples, adjusted Q75 AI-preservation intervals include loss, and absolute gates fail in every
fold. Against original non-native recipes REAL FPR also remains higher. No reference is quietly
dropped to manufacture a win. These fold scores are not comparable with the full E43 E49 94.3%
headline as a before/after decline. Source transitions and all failures are preserved in
`evidence/e54_result.json` and `evidence/e54_diagnostics.json`; saved-model replay runs separately.
No full-data candidate, serving update, external DEV rescore or independent-final opening occurs.

Exploratory input audit finds 230/4,035 near-monochrome REAL global crops, 229 from RR. Baseline
clean false positives: 169/230 monochrome versus 618/3,805 other crops; adaptation 153/230 versus
553/3,805. This correlation mixes source/content/colour and does not prove a cause. It motivates
E55, preregistered before extraction/fitting: identical frozen-DINO native linear recipe, 80/20
original-versus-grayscale FIT loss mass for both classes, versus an exact-duplicate control at the
same mass. Original CAL/validation and all AI replay stay intact; same no-loss and absolute gates.
New derivatives are not new independent images. The four added transformation/weight tests pass.

User broadened public-download permission while working remotely and asked to avoid manual login
or approval. Accessible sources can be used after role/licence checks, but no paid API spend is
inferred. Qwen-Image-Bench is not new data: its historical adaptation/protected-test roles prevent
blindly adding a fresh mirror to training. Both newly acquired publishers remain protected reserves.

E54 serialization closure completed: all six saved models reproduce all 23,912 archived OOF
prediction observations bit-for-bit (maximum score error 0, decision changes 0), with frozen
backbone hashes unchanged. `evidence/e54_acceptance.json` explicitly rejects both arms despite
successful replay because relative preservation and absolute gates fail. This is repeatability,
not additional accuracy evidence. E55 extraction is running on existing local inputs. Full Python
suite at this checkpoint: 598 passed, one pre-existing Starlette/httpx warning; no dependency update.

### 2026-09-09 21:18 local — remote run paused safely at 3% battery

While investigating slow E55 extraction, `pmset -g batt` reports Battery Power, 3%, eight to nine
minutes remaining. No AC connection is available remotely. Stop only this turn's extractor and
temporary timing diagnostic, both exit 143; preserve all existing demo processes and data.
The timing diagnostic returned finite features but was interrupted before completing comparisons;
it establishes no speedup or proven cause. No E55 fit or quality measurement has started.

Preserved 17 complete hash/shape/binding-verified chunks: 816/34,890 grayscale views, 8,941,247
bytes, no partial files. E55 `extract` can resume those exact chunks under its unchanged contract,
then run `fit` and `report`, after AC power is connected and verified. Receipt
`evidence/e55_pause_checkpoint.json` binds every preserved chunk. E54 is fully complete, archived
and reproducible but quality-rejected; E55 is operationally paused, not quality-rejected. All
400 newly downloaded reserve bodies remain unscored and protected. No paid API spend or GitHub
push occurred. This turn leaves no active ML training/download; user notified about charging.

### 2026-09-10 — overnight continuation authorized; power restored, external disk absent

User asks to work through the morning and permits public downloads. At 00:29 local, AC Power is
confirmed (battery 1%, charging), but `/Volumes/LaCie` is absent, physical disk inventory lists
only the internal drive, and USB inventory has no LaCie/Rugged/Seagate device. User notified;
no mount placeholder, destructive repair, internal bulk mirror or replacement dataset created.
No E55 fit or resumed extraction is claimed. Existing 816-view checkpoint remains the known
external state, not reverified local availability. Later battery check: AC/8% and charging.

Using the official OpenAI Docs workflow, configured thread heartbeat `gece-model-geli-tirme-takibi`
every 30 minutes through 10 September 08:00 Europe/Istanbul. Continue only on actual AC/external
data availability, skip unchanged notifications and duplicate jobs, retain training/test separation,
and keep new public acquisition licensed and role-bound. No manual-gated or paid services inferred.

New operational wrapper `ml/experiments/e55_resume.py` surrounds the unchanged frozen E55 script.
It checks real mounted external storage, charging and 10 GiB free; verifies paused contract/chunk
hashes; takes a single-run lock; rejects concurrent direct E55 jobs and orphaned final-feature
artifacts; sets the exact external/offline environment; runs only pending extraction/fit/report.
Every 30 seconds it checks power/storage/deadline and stops only its own child process group on
failure. SIGTERM triggers child cleanup. Logs stay under ignored `ml/work/e55_resume/`. No serving,
label, image, threshold, model-weight or frozen-experiment-code changes. Eleven operational unit
cases pass, and real preflight returns missing-disk refusal as expected. This is engineering
readiness, not completion of a model experiment. The frozen E55 scientific gates remain unchanged.

Final local verification for this engineering slice: 609 Python tests passed in 11.29 seconds,
one pre-existing Starlette/httpx deprecation warning, and `git diff --check` passed. The real `run`
path returns `not_started` for the absent volume without creating `/Volumes/LaCie` or even its
local work/log directory. No data/model/dependency download, no model quality measurement, and
no new training child is left running. Overnight continuation is scheduled; physical disk access
remains required. Only this turn's verified code/docs are committed locally, without GitHub push.

### 2026-09-10 01:41 local — disk reconnected; original E55 checkpoint verified

After the user reconnects LaCie, USB sees the external Rugged device. macOS independently starts
`fsck_exfat` before mount; this agent waits without interrupting it or issuing repair/forced-mount
commands. After approximately 90 seconds the physical volume mounts and guarded preflight passes.
All 17 preserved chunks verify against their archived hashes (816 views); frozen E55 script and
all bound repository helpers also match. AC remains attached. Resume the unchanged extraction,
then six fixed fits and reporting under the guarded runner with the 08:00 local deadline. This
entry records readiness/resume authorization, not successful fitting or improved model quality.

Guarded runner actually starts at 01:41:42 local; extraction log is
`ml/work/e55_resume/20260910T014142+0300_extract.log`. The first new progress marker reaches
1,008/34,890 views, beyond the preserved 816, confirming resumed feature computation rather
than merely a queued command. No fitting or quality result yet. Single supervised child only;
night heartbeat remains active and the wrapper enforces the 08:00 deadline and power/disk checks.

### 2026-09-10 — continuous background research/test loop requested

User explicitly asks for ongoing background improvement, testing, error analysis and research.
Update the existing heartbeat instead of creating a duplicate: 30-minute recurring follow-up,
now without the original morning expiry. The current E55 child retains its original 08:00 runtime
safety limit; future jobs require their own bounded run/deadline and AC/storage checks. At this
check extraction has passed 25,968/34,890 views and no fit result is claimed. Follow-up proceeds
from measured evidence, not repeated arbitrary tuning; existing AI-preservation, absolute gates,
unopened reserves, source licences and append-only experiment records stay intact. Stop when
requested or the documented objective is genuinely independently established. Only meaningful
updates or needed user action should notify the user. Local computer/app availability remains
required for scheduled local work, per official OpenAI scheduled-task documentation.

### 2026-09-10 — E55 completed and rejected; numerical diagnosis completed

All 34,890 grayscale derivative views and six prescribed fits finish after the guarded resume.
No new image download. Clean/Q75 AI recall: native 66.39/65.11%, grayscale 67.78/66.03%; REAL
false-AI rates: native 19.50/18.76%, grayscale 20.12/19.45%. These are the same consumed source-held-out
TRAIN folds, not a deterioration from the historical full-model 94% benchmark. All acceptance
gates reject both arms; saved-head replay is exact. Duplicate control also fails its frozen parity
requirement (maximum score differences .00118/.00768/.00238; two fold-1 decisions change). Serving
is unchanged and the 400 new reserve images remain unscored. Results archived in evidence/e55_result.json.

Preregistered FIT-only audit separately freezes code/inputs in commit aed70a6 before execution.
Exact native/E55 FIT features, labels, source and parent order match all folds; collapsed duplicate
weights match within 2.23e-16. Float64/tol=1e-8 fixed refits reduce original-versus-duplicate FIT
score differences to 1.36e-6/2.24e-6/6.99e-7. This supports numerical optimization sensitivity;
it does not establish separate effects of precision and tolerance, improve validation quality,
or retroactively pass the old study. No new diagnostic candidate or external score is produced.

Archived colour groups show the tradeoff: grayscale clean monochrome REAL errors 169->152, but
other REAL errors 618->660; Q75 170->159 and 587->626. These are net counts, not counts of paired
rescues, and source/content confounding remains. Global grayscale is not an accepted solution.
Next work targets training-source/content/processing coverage with stable controls, not repeated
colour tuning or opening protected final reserves. Full receipt: evidence/e55_audit.json.

Verification: 611 Python tests pass in 12.32s, one existing Starlette/httpx deprecation warning;
the diagnostic run exits 0 and leaves no training child or saved candidate. Local commits only,
no GitHub push. The existing 30-minute heartbeat remains ACTIVE; chat-turn completion does not
mean a continuously running agent, and follow-up depends on app/machine/storage availability.

### 2026-09-10 — E56 metadata coverage diagnosis and candidate source preparation

Heartbeat finds E55 complete, no duplicate job; AC attached and actual external LaCie has ~386 GiB
free. Metadata-only admitted-TRAIN audit is preregistered and committed c815439 before execution.
Ten of eleven frozen publisher components contain only one class: 9,270/11,630 parents (79.71%).
Fold-0 REAL FIT has 9/5,314 monochrome images versus 229/1,250 held-out RR. Source/class association
is a plausible shortcut risk, not proof of model causation. Topic fields are absent in normalized
rows; scene metadata present for 1,000 REAL, not a complete semantic inventory. All colour joins and
publisher role boundaries validate. No image read or new model score. See evidence/e56_coverage_audit.json.

Primary B-Free documentation supports paired content/processing as a research direction. Its COCO
training release is not admitted because current project protection includes COCO/DDA/Module 2;
background-restored edits also require a distinct label policy. Checked old E40: content-cluster
weighting already existed, so do not present another blind cluster sweep as novel work.

FiveK provides a possible licensed research REAL supplement with annotated subject/light and human
tonal processing, but is old SLR data, not modern phone proof or paired AI. Metadata-only contract
66ad22f caps 12 MiB; five official text resources total 4,725,181 bytes on external storage, zero
image bodies. First parser fails closed because official licence files use stems without `.dng`.
Preserve frozen v1 source/contract. Separate preregistered amendment 17fc506 pins all acquired bytes
and joins exact DNG filename stems offline; all 5,000 identities have one licence, 2,690 Adobe and
2,310 Adobe+MIT. Full metadata manifest SHA ace6999013bf359db1d7d6b8b22361b9cfc2b9bb97aaf55c71e5f5b7ced10487.
No fuzzy match, licence waiver, image substitution or training. Receipts: evidence/e56_fivek_*.

Next registered package is a 12-original-DNG, <=384 MiB decoder/overlap pilot before any larger
TRAIN supplement. Isolate decoder dependencies from frozen ML; no image pilot has started yet.
Keep AI replay/gates, serving, Module 2 and independent reserves unchanged. No accuracy gain claimed.

Verification for this slice: 620 Python tests pass in 12.14s, one existing Starlette/httpx warning;
git diff --check passes. All acquisition/audit commands completed, no training or download child
left running. Own verified changes and receipts committed locally; no GitHub push. The scheduled
follow-up remains responsible for the next bounded pilot, not an unstarted background training job.

### 2026-09-10 — E56 12-parent RAW pilot completed; E57 supplement preregistered

AC/real LaCie/386 GiB free verified; no overlapping jobs. Isolated `ml/work/e56_decoder` installs
only binary rawpy 0.27.1, NumPy 2.5.1 and Pillow 12.3.0, LibRaw runtime 0.22.1. pip check passes;
the ML environment still has no rawpy and unchanged NumPy/Pillow. Installer report hashes and
decoder/helper code are pinned. Official rawpy parameter documentation informed explicit colour,
gamma, WB and brightness choices, not outcome-based adjustments.

Commit c9e484a freezes 12 subjects-balanced identities, HEAD sizes/ETags/dates, 32 MiB/file and
384 MiB total ceiling before image GET. Actual originals total 122,074,000 bytes; fixed local
8-bit RGB PNGs total 166,400,845 bytes. As-shot WB/full-resolution AHD/sRGB primaries/gamma
(2.4,12.92), no auto-bright, bright=1, highlight clip. Preserve all original DNG/licence bytes.
Second offline decode of all twelve exactly reproduces saved RGB arrays and unchanged receipt;
zero extra image transfer. No screenshot/thumbnail substitution, image selection by detector score,
source-variable brightness tuning or change to frozen experiment code.

Protected screening includes 150,883 observations (snapshot +MNW +HDR+): zero raw SHA, decoded
exact/pixel/near-hash cross matches and zero internal pairs. This is heuristic, not complete scene
or edited-variant deduplication. Two visual sanity checks show coherent photographs, but no label
reclassification or exclusion. Two outputs have >20% white clipped pixels, consistent with visible
bright sky in one inspected image; record as a rendition limitation, not a quality claim or filter.
An ad-hoc file listing initially hit macOS `._` AppleDouble metadata; ignoring those sidecars fixes
the inspection. Frozen pipeline uses exact named files and was unaffected.

All 626 Python tests pass (12.50s, one existing Starlette/httpx warning), six new pilot fixtures;
actual download/isolated decode/replay/overlap phases all exit 0. No fit, model score, serving change,
final-reserve inference or TRAIN admission. Evidence: e56_fivek_pilot_contract/download/audit JSONs.

Next E57 is preregistered: up to 201 subject x light-stratified FiveK parents, <=4 GiB original
budget, protected screening, fixed decoder and existing AI replay. Six source-held-out fits compare
stable float64 native control with the new REAL supplement under unchanged gates. Acquisition and
training each require their own frozen contract; neither has started. No success promised.

### 2026-09-10 — E57 missing-WB amendment and guarded continuation

Commit 9ed4541 freezes 201 original identities and 1,999,000,262 HEAD-pinned bytes before GET;
5bc35ec commits the paired six-fit implementation before any new model score. Acquisition v1
stops after 61 decoded parents: `a1854-kme_290.dng`, DCS460D Bayer RGB, reports as-shot WB
[0,1,0,0]. The unchanged E56 decoder correctly rejects missing positive RGB coefficients.
This is not evidence of inverted labels. The waiting continuation refuses missing completion
receipts and starts no fitting. Preserve original code, contract and partials.

Before model scoring, PLAN registers v2 amendment f840136: inspect WB model-blind, exclude
unsupported as-shot metadata without replacement, retain original/hash/reason, otherwise use
the unchanged decoder. No daylight/auto-WB fallback or score-based selection. Other failures
still stop. Separate v2 receipts bind the old selection and verified cached originals/PNGs;
report lost strata and counts. This restricts supported RAW inputs, not a universality claim.

At this checkpoint v2 acquisition is active. A guarded continuation waits for that exact PID,
then audits overlap, freezes the model contract, extracts features, fits six fixed arms and
reports unchanged gates. Combined follow-up <=60 minutes, AC/real-disk/free-space checks,
single-job lock and owned-child cleanup. No duplicate download, final-reserve score, serving
change or accuracy gain claimed. All 633 Python tests pass (12.63s, one existing warning).
Archive completion receipts/results next; an active job is not a completed experiment. No push.

### 2026-09-10 — E57 completes with modest aggregate benefit but fails preservation

Guarded acquisition/audit/freeze/features/six-fit/report sequence completes at 05:08 local; no
duplicate workers or owned training/download processes remain at 05:38 check. Real LaCie and
AC verified. Official originals 1,999,000,262 B, of which 24,508,438 B reuse two pilot bodies;
new original bodies across E57 v1/v2 total 1,974,491,824 B. Local derived PNGs 3,022,249,526 B,
not downloaded images. Two missing-WB parents excluded; 199 admitted FIT-only, zero matches
against 150,883 protected observations or internal pairs under the frozen heuristic. Both lost
cells retain eleven of twelve selected rows (animals/sun and unknown/sun); no replacement.

Model contract 87cf485974e8c53b5e93d15c0afb64767f3dc5961ca291aa67ab917b71aeb222 freezes
before 597-view feature extraction; pretrained parity exactly zero. All six saved heads replay
with zero score error/decision changes; all existing AI view weights preserved. Stable native
versus FiveK supplement on 5,978 source-held-out research parents: clean AI 66.2378->66.4436%,
REAL FPR 19.5291->18.3147%; Q75 AI 64.9511->65.8775%, FPR 18.7361->18.2404%.

This is not success: Seedream/GPT Image 2/Qwen Image 2 and other AI strata lose detections,
worst supported-source loss versus stable native 7.5 percentage points. AI-preservation intervals
include loss; all absolute fold gates and full relative acceptance fail. Candidate is rejected,
not served; no final reserve opened and no comparison to full-model historical 94% headline.
All result/input receipts archived locally in commit 78c959a. Do not enlarge/sweep FiveK.

Next E58 preregistered existing-score diagnosis separates error transitions from optimistic ROC
limits; no new inference, fit or exportable threshold. Code 3c46ada plus separately frozen input
hashes. Primary UnivFD/B-Free research reviewed; E47 already used official UnivFD, while B-Free
paired generation is not reproduced by this unpaired REAL supplement. No public-source licence
or held-out publisher restrictions changed. No GitHub push.

### 2026-09-10 — E58 confirms threshold-only repair is insufficient

Frozen score diagnosis 36c7a25 completes with no new images, inference, fitting or saved cuts.
E57 supplement rescues 105 REAL clean errors but introduces 56 (net49); Q75 rescues76,
introduces56 (net20). AI clean rescues50, loses46 (net4); Q75 rescues62, loses44 (net18).
Aggregate benefit therefore conceals real per-image and modern-generator losses.

Optimistic label-oracle AI recall at FPR10, supplement clean/Q75: fold0 37.30/34.59%, fold1
75.83/80.00%, fold2 66.29/68.27%. AI95/FPR10 is infeasible in every cell even with consumed
validation labels choosing the best possible scalar cut. No cut is exported. This rules out
threshold-only repair on these frozen score streams, not future representation improvements.
Fold1 clean also loses ranking: native 78.96% vs supplement75.83% at FPR10. Full paired
transitions and tie-aware envelopes are archived in evidence/e58_score_diagnosis.json.

E59 next distinct representation hypothesis registered in PLAN: cached CLIP image features,
not the previously tested E47 ProGAN head, trained later on the existing E54 pool alongside
DINO. First resource-only probe uses eight deterministic fold0 FIT parents, no candidate or
quality scores, no download; code 0948bdc, pinned-input contract 93b99f7. Full extraction and
nine fixed source-fold fits require their own later contract. Current full suite638 passed
(11.93s, one existing warning). Existing upstream tracked source is unchanged; only Python
bytecode caches appear untracked in the external checkout. Serving and closed reserves intact.

E59 resource probe completes successfully under its guard: eight FIT parents/24views,72crops
per pass; MPS elapsed7.9457/7.8218s, embedding replay error0. Sampled driver allocation
2,194,358,272B, not peak. No downloaded bytes, detector scores, fit or saved candidate. Guard
exits0 with no owned workers left. Archive evidence/e59_clip_probe.json; this is engineering
feasibility only. Full-scale extraction likely needs hours and resumable <=60-minute guarded
chunks; no full E59 feature run or quality experiment has started. Next scheduled continuation
must freeze full feature/training contracts first; no manual user action is needed for preparation.

### 2026-09-10 — E59 full cached-CLIP extraction begins, no new acquisition

06:18 follow-up verifies clean main, no duplicate workers, AC and real LaCie (~381GiB free).
Preregister fd3120a fixes all 11,630 existing E54 TRAIN parents/34,890 ordered views, no FiveK.
Official frozen CLIP ViT-L/14 normalization on existing RGB224 crops; raw768-D three-crop
mean/population-std concatenation gives1536 features per view, no L2 or fitted preprocessing.
Three planned representations are DINO3072, CLIP1536, concatenated4608 under unchanged folds,
AI views/weights, original FIT mass, CAL and all gates. Training must freeze separately later.

Implementation314593a adds parent-bound atomic chunks, internal SHA/shape/finite/aggregation
checks, exact archive reconstruction for interrupted finalization, and a locked60min guard
checking power/storage every2s. Preserve completed chunks on interruption; never overwrite an
incompatible completed archive. Three new fixtures cover statistics, corruption/identity and
interrupted finalization. Full suite641 passes (12.07s, one existing Starlette/httpx warning).

Commit f533877 freezes feature contract
224382c10f2831a9a522bb668e952c30b3a1be890fa2219044f99e7679605954 before full inference.
06:22 local guarded run starts (observed guard87100, worker87108), log
ml/work/e59_features_20260910T062252.log. No data/weight download or ML dependency change;
all generated feature chunks reside on real external storage under e59. Runtime is expected
to need multiple bounded invocations, not guaranteed by the tiny probe. No completed full cache,
new fitted detector, quality score, final-reserve opening or serving change at this checkpoint.
Next follow-up must inspect active process/log before attempting a resume; never duplicate it.

### 2026-09-10 — E59 downstream comparison implemented while extraction continues

06:54 follow-up verifies active original guard87100/worker87108, AC/LaCie and ~380GiB free;
at least2,300 parents complete, progressing beyond2,700 during this slice. No second extractor,
training process or weight/data download started. Registered control parity6e1a648 before fits:
E59 DINO must reproduce paired E57 native scores/cut within5e-5 and zero decision flips.

Implemented the already registered DINO3072/CLIP1536/combined4608 nine-fit experiment with
fixed FIT-only scaler/float64 head, unchanged CAL/OOF and AI view weights. Completion receipts
are mandatory before model-contract freeze; no model contract or real-data fit exists yet.
Historical reference JSONs must match E57's archived bindings and all saved-head hashes.
Atomic head writes, explicit orphan refusal, exact replay, source-wise preservation checks and
AI weight hashes across all three arms prevent silently changing the comparison. Full report
includes every fixed arm and fold; a control mismatch blocks candidate eligibility.

Prepared a <=60min freeze/fit/report guard using the SAME lock as feature extraction; tests
confirm it cannot overlap a held extraction lock. Full suite646passed (12.88s, one existing
warning), plus separately passed new lock-exclusion fixture in the six-test E59 focused suite.
This is tested implementation, not a claim that the pending nine real-data fits have passed.
No feature protocol edits, final-reserve inference, serving change or GitHub push.

### 2026-09-10 — E59 bounded extraction resumed at07:53

Original06:22 invocation exits1 at its registered60min deadline with TimeoutError; the guard
cleans up its worker. This is an expected bounded interruption, not changed data or an accuracy
failure. No old process remains. Exact filesystem inventory:4,465 parent NPZs,00000..04464,
173,197,992B, no pending `.part` files (AppleDouble ignored). This counts persisted chunks,
not an assertion of completed full-feature validation. Frozen resume rechecks each binding,
content hash/shape/aggregation before reuse. AC and real LaCie/~379GiB free verified.

Same unchanged contract/command restarts07:53 for<=60min, guard10657/worker10663; log
ml/work/e59_features_20260910T075334.log. No duplicate extraction, fit, tests repeated for
activity, new data/weights, dependency changes, final exposure or serving change. Training
remains blocked until the complete11,630-parent feature archive and both receipts exist.

### 2026-09-10 — E59 third bounded feature run and completion handoff

09:23 check: second invocation ends at its registered60min deadline (TimeoutError), no old
worker survives. Exact inventory8,308 completed parent chunks,00000..08307,342,619,493B,
no `.part` files. AC/real LaCie/~378GiB free verified. Unchanged feature command resumes09:24,
guard40155/worker40165, log ml/work/e59_features_20260910T092449.log. It revalidates retained
chunks; early log confirms zero replay error and reuse, not duplicated inference of the pool.

Operational plan80f8bff and code530513e add a separate bounded exact-guard-PID waiter.
Observed waiter41371 starts after verifying the actual feature guard. Once that guard exits,
full archive and both receipts must validate before exec handoff to the existing exclusive
training runner. Waiting plus training fits within the original60min waiter budget. PID reuse,
incomplete receipts or less than one remaining minute stop the handoff. A separate waiter lock
prevents duplicate queued continuations; replacing its own process avoids an orphan supervisor.
No training has started at this checkpoint. If another bounded resume is needed, inspect state
first; never rerun completed science or weaken the budget to force a result.

Full649-test suite passes (13.42s, one existing Starlette/httpx warning); two new fixtures cover
PID identity/exit and remaining-budget accounting. Frozen feature/model science unchanged.
No new source/weight download, ML dependency change, final scoring, serving update or push.

### 2026-09-10 — User redirects to the strong model; pause and offline office plan

User rejects continued weaker-replacement exploration and requests planning now, application
in the office with no downloads. Stop queued followup41371 first (exit143), then feature
guard40155 (exit130 via its SIGTERM cleanup), which stops owned worker40165. Process inventory
confirms no E59 worker/trainer remains. Preserve9,599 completed parent chunks/399,641,982B,
00000..09598, no `.part`; no full feature receipt or E59 trained model exists. No files deleted.
Existing thread heartbeat gece-model-geli-tirme-takibi changed toPAUSED through the app tool;
saved status verified. No automatic resume while awaiting the user's office restart.

Read-only E43-S artifact check matches the original E49-bound SHA exactly:
a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390. The good full research
model was never replaced by these source-fold candidates. E49 recall94.3/95.5% remains
historical alongside REAL FPR39.1/49.0%; not a final pass or necessarily the current web profile.
Initial macOS shasum invocation failed from the environment's C.UTF-8 locale; LC_ALL=C retry
verifies the bytes without modifying anything. No artifact/dependency/serving changes.

PLAN now prioritizes exact strong-model reproduction and lawful local data/split audit, then
one separately registered zero-initialized correction to the frozen E43 head using existing
DINO features, REAL errors and complete eligible AI replay with retention constraints. No
new architecture/weight/data download or fit is authorized in this turn. E43's prior exposure
invalidates naive reuse as a clean OOF comparator on rows it trained on; fresh eligible local
development must be checked before quality claims. Cannot recycle final/Module2/reserves or
guarantee AI retention through a penalty alone. Historical E59 work is parked, not erased.

### 2026-09-10 — Full documentation review and current-state handoff

Read all 19 project-owned Markdown files in full (dependency documentation excluded), then
inspected frontend request/result contracts, serving/model/input code, E43 feature/head code,
E51/E59 implementation paths and saved evaluation evidence. User reaffirms that every development,
experiment and planning change must update the relevant Markdown in the same work step; recorded
this standing rule in PLAN. This review does not authorize resuming the paused experiments.

Read-only checks confirm the E43-S artifact remains 87,916B with SHA
`a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390`.
The existing localhost:8799 health endpoint reports ready/MPS/demo, canonical E20 ready,
CF-ViT as the only loaded verdict arm, R1b research ready and no load errors. This is runtime
availability, not fresh inference or accuracy validation; E43 is not the served model.
No E59 worker/trainer is running; saved heartbeat status is PAUSED. External E59 inventory
contains 9,599 payload NPZ chunks totaling 399,641,982B, plus 9,599 macOS AppleDouble sidecars
totaling 39,317,504B. Sidecars are not additional parents. No partial files, complete feature
archive, feature completion receipt or model contract exists. Inventory is not chunk-content
revalidation. No training, extraction, new image scoring, downloads or test suite was run.

Documentation erratum: the earlier E49-C wording "failed 11/20 gates" reverses the passed
count. `evidence/e49_final_result.json` records 11 passed and 9 failed checks (original 6/10
passed; Q75 5/10 passed). The overall FAIL decision and all underlying scores remain unchanged.
Append the same correction to the scientific log; preserve the original historical wording.
`evidence/e51_e49_regression.json` confirms E51-A's REAL-FP reduction also loses AI recall
on the same consumed E49 parents; it does not establish a champion-preserving improvement.

### 2026-09-10 — User authorizes offline champion-first execution

User requests optimal continuation towards the goals without data downloads. PLAN is active
for the frozen E43 audit and one bounded correction experiment; E59 and the scheduled heartbeat
remain parked. No source, weight or package downloads. Begin E60 with pinned teacher exposure,
local admission and evaluation-role audits before fitting or selecting any correction recipe.

E60 audit implementation has three passing role/exposure fixtures. Initial v1 stopped before
scores or fitting because it required uppercase TRAIN, while 4,278 inherited admitted rows
use lowercase train. Metadata inspection confirms identical TRAIN meaning and integer 0/1
labels; normalize case only, with no membership or role reassignment. Preserve v1 code in
`ml/experiments/archive/e60_audit_v1.py` and its contract/evidence; register audit v2 separately.

Before v2 could freeze, an inherited `e54_data` import reached Kaggle's import-time OAuth
introspection and timed out. No data download or training occurred; stopped the other owned
audit invocation. Removed that import chain in favour of explicit pinned admission JSON/hash
checks. v2 is still pre-freeze. New execution will deny network connects at process startup.

After the user's connectivity interruption, no E60/E59 worker remained. Audit-v2 contract
had completed; resumed analysis without repeating freeze. Audit completes with zero inference
or downloads:8,844 E43 FIT parents/19,648 views;4,278 shared current TRAIN identities and encoded
bodies. Current pool7,035 REAL/4,595 AI. Missing old AI include360 protected CAL; other old
unadmitted rows stay excluded. The inspected unused-native pool has no balanced unseen-recorded-
group DEV. Six audit/correction fixtures pass, including finite-difference gradient, exact
saturated zero-output parity and correction bound. Register one fixed CPU correction and
a later consumed E49 diagnosis in PLAN/EXPERIMENTS; no quality result yet.

E60 fixed200-step CPU fit completes in9.12s on all11,630 parents/34,890 views. Candidate
SHA582d6c4f020ce309c8e44a88383ef9d69c559773d6332694bd203e1132415bc5; E43 SHA unchanged.
Zero-init and saved-artifact replay errors are exactly0. Largest TRAIN logit correction is.17156.
TRAIN clean/Q75 REAL FP falls15.0675->14.8543% /13.8024->13.5039%, while AI recall loses
one/three images. These are fitting diagnostics, not held-out quality. Keep all losses visible.
Twenty focused E60, fixed-evaluation and training-weight tests pass. Proceed only to the
preregistered frozen-candidate E49 consumed regression; no refit or cut selection.

E60 consumed E49 scoring completes all4,000 views in183s with exact E43 score reproduction
(maximum error0 and zero binary/selective decision changes); raw scores frozen before metrics.
Original/Q75 AI recall remains94.3/95.5%, zero new AI misses/rescues across all six generators.
REAL FPR39.1->39.0% /49.0->48.8%: one/two rescued REAL observations, zero new REAL errors.
The four familywise source-cluster delta intervals are REAL[-.004,0]/[-.008,0] and AI[0,0]/[0,0].
REAL improvement is not established; the20 absolute/selective checks still fail overall.
E60 is insufficient, not a promoted successor. Research candidate and complete evidence are
retained; E43 artifact and serving code/registry unchanged. No protected final reserves opened,
no source/weight/package downloads, no automatic E59 or heartbeat restart.20 focused tests pass;
documentation changes pass git diff --check. PLAN records the next evaluation eligibility
requirement and forbids retuning on this consumed regression. User's v1 quality goal is not achieved.

### 2026-09-10 — GitHub synchronization and literature-guided continuation requested

User explicitly requests GitHub pushes, renewed primary-source web research and continued
REAL false-positive reduction without sacrificing AI recall. No new data may be downloaded
now; record suitable future acquisitions in PLAN for home, with evaluation/optimization later.
Revisit image-structure/forensics notes and distinguish historical heuristics from validated
current evidence. Prepare the completed E60 checkpoint for commit/push before the next study.


## 2026-09-10 — GitHub synchronization, primary-source review and E61 registration

User explicitly requested pushing progress to GitHub, researching REAL false accusations,
retaining AI detection and deferring dataset downloads until home. `git push origin main`
succeeded through `2da9fe81ee97df1b7725bcfd1bb659fa26dbe24f`; `git ls-remote` matched local
HEAD and worktree was clean. This also synchronized previously unpushed history. Server
reported two required checks expected and allowed the push; no CI success is inferred.

Revisited the image notes/references and existing experiment history against B-Free,
CNNDetection, SFLD, GEM and Neyman–Pearson primary sources. Corrected current interpretation
in dated addenda without erasing historical notes. WIFD, RawNIND and conditional SIDD are
now a home-download queue in PLAN; RealHD remains unavailable in its official repository.
No image/data/model/package download performed. No protected corpus role changed.

Implemented strict TRAIN AI replay validation in `pixelproof/retention_gate.py`: reject any
new miss among previously caught AI views; report condition/source transitions and distinct
parent losses; reject incomplete/malformed scores, duplicate parents and non-TRAIN roles.
A pass is explicitly not model promotion. E61 runner binds admitted cached features, manifest,
E43, frozen E60 and code hashes before one known-result engineering replay. No fitting,
threshold changes, E49 reads or new quality claim. Documentation/contract written before
execution; unit and replay results follow as a separate entry.


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


### GitHub CI observation for pushed E60 checkpoint

[Run34468501197](https://github.com/EfeHanKeles346/ai-image-detector/actions/runs/34468501197)
completed: Python job (full tests, compilation, dependency consistency and audit) passed;
web lint, typecheck and tests passed. Web npm audit failed on the existing pinned Next.js
critical advisories GHSA-p293-qw3h-jr36 and GHSA-2xp9-vwfh-vxw4. Audit also lists moderate/high
transitive debt. This is not an ML regression; do not disable the critical check or silently
run force upgrades. Added a separate dependency-maintenance item to PLAN. No web package,
lockfile or serving modification was made as part of the no-download ML research checkpoint.


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


## 2026-09-13 — Resume with data-download authorization

The user explicitly authorizes continuing analysis/development and downloading needed data.
This supersedes the previous no-download restriction in historical sections below. Preserve
AI retention, immutable E43/experiment evidence, protected roles and commit/push checkpoints.
First audit source metadata/license/size and scene independence before acquiring pixels.
Use existing external volume; no automatic full-corpus download. WIFD/RawNIND/SIDD remain
candidates, not automatically admitted TRAIN or a balanced independent final. E59 stays parked.

Power preflight reports13% battery, below the existing30% bounded-work floor. Asked the user
to connect power. Metadata research and small code/documentation work can proceed; long
image downloads/extraction/training wait for verified AC power or sufficient battery.



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


E65 engineering verification: all697 Python tests passed in12.96s, including11 new E65
tests. One existing Starlette/httpx deprecation warning. E65 compile check, pip check and
git whitespace check passed. No dependency change. Remote main was synchronized before
checkpoint work (0 ahead/0 behind); previously documented web npm-audit debt is separate.


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


Overnight engineering checkpoint:710 Python tests passed in13.92s (13 new E66/E67 tests),
with the existing Starlette/httpx warning. Compile and whitespace checks pass. Native AI
admission is complete; SIDD download and E67 TRAIN feature extraction are still in progress.
Commit/push this reviewed protocol/code checkpoint without claiming a fitted or improved model.


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


During the continuing download, reviewed primary perturbation-response papers and DEnD's
example inference code. Found batch-context dependence and evaluated-label threshold search
in that example, which do not meet PixelProof's fixed single-image decision protocol. Details
and primary links appended to IMAGE_FORENSICS_REFERENCE.md. E67's fixed recipe is unchanged;
no third-party code/weights executed or downloaded. Four E67 DEV guard tests passed.


Full Python verification after the E67 DEV runner: 714 tests passed in 12.65 seconds,
with one existing Starlette/httpx deprecation warning. No model-quality claim follows
from software tests. SIDD archive final integrity verification is still in progress.


Remote CI for f325b7c (run 34723716630) completed: Python succeeded; web failed at
"Audit production/build dependencies (known high vinext debt is documented)". This is
not an all-green CI claim; the existing dependency-security maintenance remains open.
Local 714-test suite passed. Power check: AC attached, 80%.


### E66 SIDD acquisition complete (2026-09-13)

All 6,615,978,508 archive bytes verified against published MD5 and SHA1. Recorded SHA256:
855c375ae20312386cd961e7fdbbeaeb33efdecf0f6f115c01e09f45ec471fdf. No members decoded or
scored at acquisition completion. Starting the already frozen 160-NOISY-image scene and
protected-overlap audit; acquisition alone does not admit DEV. Raw archive stays external.
E67 feature/DEV-runner checkpoint 61f96f85f6039a6e88ee18e05ac37967e4544af7 is pushed,
with identical local HEAD and remote main verified. No fitted candidate yet.


Reviewed the DFRWS 2026 camera-fingerprint interference paper while E66 audit runs.
Recorded its distinct attribution task and limitations in IMAGE_FORENSICS_REFERENCE.md.
Its public dataset is only a metadata-audit lead, not downloaded/admitted or added to E67.


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


E68 implementation prepared: five new tests pass (analytic risk/objective/epigraph
gradients, worst-group bounds, equal parent mass under duplicate views, non-tradable
AI/REAL decisions, and parent/source identity rejection). Compile and whitespace checks
pass. E67 TRAIN-only source diagnostic is locked; no additional candidate was fitted
during diagnosis. E67 result checkpoint 9ef5deb is pushed and remote main verified.


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


### Additional data lead: Boon_or_Bane (metadata only, 2026-09-13)

The author-linked public Nextcloud share opens without authentication in the browser:
https://cloud.digfor.code.unibw-muenchen.de/s/BEEzsMnZLFCXw9S . Its root shows only two
folders, ChatGPT5 (145 MB) and FireflyImage4 (139 MB), rounded total 284 MB. No licence
file is visible at the root. ChatGPT5 has folders named 1024_1024, 1024_1536 and
1536_1024; the first contains fingerprint/test subfolders. These names are not verified
pixel dimensions or generator identities. They differ from the paper's GPT size table,
so exact published-file provenance needs clarification before use. No image download
was triggered and no image was opened/scored. A legacy read-only DAV request returned
401; the ordinary public browser listing succeeded, without login or access changes.

Keep as an unadmitted acquisition lead pending explicit dataset licence, exact manifest,
prompt/source/processing provenance and full protected overlap checks. Do not infer an
image licence from the paper's CC BY-NC-ND licence. This lead contributes no E69 data and
does not resolve modern REAL scene coverage; it is not a ready TRAIN/DEV/final pool.


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


Added evidence/overnight_2026-09-13.md as a compact human-readable comparison of the
shared TRAIN results and acquisition/DEV boundaries. It distinguishes failed candidates
from engineering progress and records the unchanged serving/reference. Checkpoint
7537a3971101738c86b10241370a46e5dd78362d has identical verified local/remote identity.


GitHub CI for 7537a39 (run 34725498398): Python succeeded; web failed only at the
existing production/build dependency audit step. The local 730-test result remains
valid, but CI is not fully green. E69 extraction continues; no candidate scored yet.


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

### 2026-09-13 — E71 cache contract frozen; completion started

All9,599 historical numeric CLIP chunks passed parent/binding, raw/aggregate digest,
exact mean/std and original crop-body checks. The immutable E71 inventory binds those
read-only E59 inputs; 2,031 remaining TRAIN parents will be written only under E71.
The extraction contract requires re-encoding30 cached source representatives before
completion, <=1e-5 raw-vector error, AC/20GiB reserve and60min stage limit. No E59
fit/handoff/automation resumed. Feature extraction has started; no result claimed yet.

Safe feature adapter tests6/6; full Python suite745passed in12.71s. The new original64+
CLIP64 correction has4 additional focused tests passing: exact original-basis preservation,
zero-init E43, independent branches, serialization and batch replay. An initial1e-14
probability batch tolerance exposed unchanged float32 reference rounding (5.96e-8);
use1e-7 probability tolerance with identical decisions, while correction coordinates still
require1e-14 and same-batch serialization remains exact. Candidate fit and conditional
consumed-DEV runners are prepared but neither frozen nor executed. E70 remains rejected;
E49/serving unchanged. Remote main verified at84ffd1eccf17d41be84b939e7c7963f35e2ff3bc.

E71 historical CLIP replay passed on all30 source representatives with maximum raw-vector
error exactly0.0; missing-parent completion is active. Conditional DEV runner now separately
tests failed-TRAIN denial, exact TRAIN/DEV CLIP mean/std parity and prior reference replay
at both AI/REAL cuts (even sub-tolerance score drift must not change either decision).
The749-test suite passed in14.33s before these3 additional DEV tests.

E71 preregistration/code checkpoint validation:752 Python tests passed in13.11s,
including all13 new E71 feature/model/DEV tests. Compile and diff checks passed.
Feature extraction remains active; model and DEV contracts are not yet frozen.

### 2026-09-13 — E70 margin diagnostic plan during E71 extraction

E70 meets training decision retention but fails consumed E66 AI retention. Before inspecting
per-image margins, register a descriptive diagnostic of the fixed E70 artifact: all previously
caught TRAIN AI by source/condition, logit-margin quantiles0/1/5/25/50/75/95/99/100%, counts
within1e-6/.01/.1/.5 of the fixed AI cut, and the already-locked consumed DEV margin groups
and all existing new AI misses. No candidate fitting, new image inference, E49 or E71 scores.
This asks whether decision-only constraints leave training AI close to the boundary; it cannot
certify a causal explanation or unseen-image retention. New diagnostic code/tests prepared.

Literature refresh: QuAD is already represented in E46 (quality-conditioned Gaussian lost the
conservative CAL selection on AUC), so it is not a new untried fix. Its full method aggregates
retrieved near-duplicates, distinct from single-image inference. ForensiCam and SOCRatES retain
previous access/terms restrictions. New SIDL lead is research-only paired iPhone12Pro ProRAW,
300 scenes,1588 contaminated/clean pairs; full-resolution/RAW links and derivation provenance
still need inspection before admission. No SIDL or other new image has been downloaded.

E70 margin diagnostic completed: previously caught TRAIN AI4508/4491/4475 by clean/
assigned-transport/Q75; zero new TRAIN misses, but24/40/45 views now lie within1e-6 logit
units of the decision cut (minimum about1e-7). Lower1% margins moved0.88384→0.37102,
0.83336→0.05094,0.50470→0.00605; mean shifts-0.78165/-0.95040/-0.83261.
These109 are condition views, not necessarily109 unique parents. Existing consumed DEV
misses remain4/3, with old margins0.0255–1.5608 original and0.1355–1.1134 Q75. Therefore
floating-point equality alone cannot explain the failures; some substantial corrections cross
formerly comfortable margins. This motivates investigating preservation of confidence/margins,
but does not prove it will reduce REAL false positives or protect unseen AI. E71 stays frozen.
No new fit, image inference, E49 or E71 score access occurred;2 focused diagnostic tests passed.

SIDL official Drive folders and metadata preview are accessible. The full training archive is
listed as25.15GB (rounded UI size), with six alternative split pieces. Visible metadata examples
confirm iPhone12Pro/iOS17.5.1,4032x3024 linear ProRAW and applied noise reduction; do not describe
these as untouched Bayer RAW or native camera JPEG. Metadata was inspected, no image downloaded
or scored; exact archive hash/size, RGB rendering provenance and grouped inventory remain pending.
Research tabs closed. Personal/location fields from publisher metadata are not copied to docs.

E70 margin diagnostic verification:754 Python tests passed in14.82s; diff/compile clean.
The diagnostic and locked outputs are ready for commit; active E71 extraction unchanged.

SIDL metadata-only fetch succeeded without credentials:8,874,397 bytes /SHA649e408b…baf2527,
1605 unique DNG records but only253 filename scene groups versus website300. All recorded
cameras iPhone12Pro, Linear Raw, noise reduction≈0.95. This is an unresolved release/coverage
mismatch, not grounds to infer1588 complete pairs or admit a partial TRAIN set. Compact
non-personal receipt added; images/scoring0. E70 diagnostic push0f960bd verified at origin.

Research refresh recorded in IMAGE_FORENSICS_REFERENCE: PiD residuals and MPFT masked CLIP
fine-tuning are possible distinct future mechanisms, neither implemented nor claimed reproduced.
MIDD official share is accessible (20 sensor ZIPs, UI331.3GB); dataset terms and native/patch
scene packaging unresolved, no image bytes downloaded. Public research tab closed.

Documentation navigation checkpoint: README and PLAN opening status now identify active E71
feature completion and rejected E70, mark older checkpoints as previous, and replace outdated
SIDD acquisition/power/candidate-role wording with completed consumed-DEV roles and AC80% status.
Historical experiment entries remain preserved. No scientific code or serving change.

MIDD metadata probe succeeded using exact HTTP ranges: ISOCELL_3P9 archive5,774,265,212B,
842 training originals/partners,79 test originals/partners. Embedded dataset license is
CC BY-NC-SA4.0. Only directory/license read, no images. Sparse original-member acquisition
can support a separate bounded research TRAIN expansion after scene/overlap audit. E71
continues its fixed recipe; do not mix new MIDD images into that running experiment.


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

During E75 extraction, checked primary2026 papers on single-class feature reconstruction,
MAFL adversarial bias suppression (already in PLAN), ACEF paired-artifact experts and DEAR
channel pruning. Record methods/resources and distinguish paper reproduction from possible
local adaptations. No change to the fixed E76 data-effect experiment. Latest inspected CI
34730124772: Python success; web fails only the existing dependency audit, unchanged rules.

DEAR source/model metadata pinned and Git blob bodies verified. Code and weight licences
are distinct; checkpoint terms are non-commercial with stated SD1.5 restrictions and
upstream lineage limitations. No model payload/image or third-party execution. E75 remains
active and E76's prepared code has been pushed at6fb0890 (origin verified).

Consolidated the living overnight report into current TRAIN/DEV tables, data roles and E76
next actions; removed contradictory intermediate running-state prose from that overview.
All historical stages remain in append-only HISTORY/EXPERIMENTS and frozen JSON receipts.


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


### E83 evaluation-time generalization literature review (2026-09-13)

- [Supervised contrastive few-shot detection](https://arxiv.org/html/2511.16541v1) uses
  an embedding learner followed by k-NN with150 support examples per generator. This is
  target-generator adaptation, so its reported gains do not justify admitting E66/E49
  labels into our model. No support-set download or test-derived retrieval classifier.
- [FAIR](https://arxiv.org/html/2607.22087v1) supplies scene-composition features during
  learning, then removes the auxiliary classifier columns at export. It requires a new
  structural-feature extraction/training contract; existing CLS mean/std features cannot
  be called that prior. No implementation or reproduction claim, no training-rank sweep.
- [MAFL](https://arxiv.org/html/2604.12353v1) alternates bias-network and authenticity
  optimization with entropy, alignment and label-reversal losses. The inspected text
  does not establish that simply training a cached-feature adapter reproduces its full
  recipe. Our source tags are partly class-confounded and RR categories are not verified
  generator identities; a naive global source adversary could erase the target label.
  The existing PLAN warning about label-confounded domain losses remains applicable.

These are mechanism leads while the frozen E83 comparison runs, not a selected successor
experiment or proof of the cause of any as-yet unmeasured E83 DEV error. No new model fit,
new data, DEV role change or E49 read in this review.


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


### Future authentic-capture source leads, no pixel acquisition (2026-09-13)

The metadata-only E84B inventory records REAL7,034 JPEG/1 PNG/511 unspecified and
AI786 JPEG/3,809 PNG. This association motivates examining lossless/RAW capture coverage;
it does not establish a shortcut or alter the active E84B/E85/E86 protocol.

- [SID official repository](https://github.com/cchen156/Learning-to-See-in-the-Dark)
  publishes short/long RAW pairs and scene/burst identifiers. Sony/Fuji archives are
  advertised25/52GB; RAWpy-rendered16-bit reference RGB alternatives12/22GB. Multiple
  short exposures share a long reference, so image count is not independent scene count.
  README states MIT; [license](https://raw.githubusercontent.com/cchen156/Learning-to-See-in-the-Dark/master/LICENSE.md)
  uses software/documentation wording. Record that scope rather than inventing separate
  image terms. Original captures or explicitly documented RAWpy references could supply
  REAL; network-enhanced outputs must not. The September2025 storage-cost note asks
  users to retain one local copy and avoid repeated download per experiment. Source tree
  revision0448b58d8c57459b1550ddf68c160d9212046a25 has six split lists (~367KB).
  Next: metadata-only scene/pair inventory; no images acquired or role admission yet.
- [PolyU original release](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
  lists40 scenes/five cameras and100 crops. Original noisy and averaged reference files
  are JPEG, so this is not direct lossless coverage. Its
  [license](https://raw.githubusercontent.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/master/License.txt)
  restricts use/redistribution to noncommercial purposes and also refers to software.
  Prefer original captures; averages/crops are dependent derivatives. Not acquired.
- [RENOIR author page](https://ani.stat.fsu.edu/~abarbu/Renoir.html) is a lead for CanonT3i,
  CanonS90 and XiaomiMi3 capture groups. Search exposes raw/aligned archives but two
  direct reads timed out. Exact current inventory, rights and rendering provenance remain
  unresolved; no download or admission. Do not rely on mirror terms or claim full review.

Whole SIDD remains consumed DEVELOPMENT; WIFD/RawNIND remain diagnostic-only. No source
above has been added to TRAIN, DEV or final. No author contact or external messages.


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


### Compression/size confounds: primary-method review (2026-09-13)

[Fake or JPEG?](https://arxiv.org/html/2403.17608v2), sections3–4, tests GenImage detectors
on compression and size changes, including real FFHQ PNG images. It reduces class-related
compression mismatch by selecting JPEG96 REAL images and encoding AI at96, then controls
size/content distributions. The authors explicitly note that random JPEG augmentation
alone does not equalize compression history: previously compressed REAL receives another
encoding. Their reported cross-generator gains belong to those datasets/detectors and
protocols, not this project. Existing historical experiments already used an unbiased-tiny-
GenImage subset; this is a primary explanation of a known issue, not a newly discovered
training corpus. No new GenImage download or reproduction is planned here.

Our E84B metadata inventory is consistent with a possible confound but cannot prove it.
In fact E83's consumed SIDD REAL FPR rises0% original to13.75% socialQ75, whereas the
paper's particular real-image JPEG experiment improves REAL recognition. Different data,
preprocessing and model matter; do not transplant its error direction or numerical gains.
E84B adds exact social transport coverage without claiming to erase historical bias.
E87/E88 prepares documented RAW-derived REAL coverage independently, while keeping the
current E85/E86 comparison fixed. No threshold, score-based selection or extra fit follows
from this literature read alone. SFLD is already reviewed/tested in bounded E69 form;
DEAR is already included, so neither is presented as a new unused detector.


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


### Consolidated GitHub repair record

Failed run34748481696 passed Python and web functional checks but npm audit found
11 vulnerabilities (1critical,8high,2moderate). Next16.3.2→16.3.5,
vinext0.0.50→1.0.0-beta.9, plugin-rsc0.5.26→0.5.34,
Cloudflare Vite plugin1.53.1→1.54.8, Wrangler4.125.0→4.131.1,
workers-types5.20260823.1→5.20260911.1 and eslint-config-next16.2.6→16.3.5,
plus compatible transitives, reduce npm audit to0. No force/legacy-peer-deps bypass.
Primary advisory: https://github.com/advisories/GHSA-2xp9-vwfh-vxw4.

CI now blocks high as well as critical advisories and cancels superseded runs per ref.
vinext changes hashed assets to /_next/static; the initially failing /assets assertion
was corrected, and the test additionally verifies emitted HTML JS/CSS files exist.
The prerelease remains explicit; build/SSR checks pass, no comprehensive browser
interaction test was claimed. Clean npm ci, build, lint, types,6 web tests and856 Python
tests passed locally (Python17.01s, one existing Starlette/httpx deprecation warning).
GitHub Linux/Node22/Python3.13 also passed pytest, compileall, pip check, npm audit and
serving-lock pip-audit. Code validation run34749560334 passed at bca8a1e; final report
commit988f65acab5a13a711a294ca32729fdd950b3703 passed run34749693377 (web32s,
Python2m18s). Old failed runs remain historical. Actions Node20 deprecation annotations
are warnings, not failed jobs. Machine-readable evidence remains in
evidence/ci_dependency_repair_2026-09-13.json. These are recorded prior verifications,
not rerun network/package checks while on mobile data.


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


## 2026-09-14 — E92 milestone preserved; end-to-end critical review completed

The user asks whether the20 conditions, data freshness, bias controls and performance
justify a student MVP or market entry. The success stays recorded: E92 passes20/20
numeric checks on consumed E66 DEV and improves E86 without losing its correct AI or
REAL decisions. The separate E43-relative original AI retention guard still fails.
This audit does not retroactively modify that result or promote the model.

Added `ml/tools/audit_e92_readiness.py`, an offline read-only audit of existing JSON
scores/manifests and artifact hashes. It loads no image, deserializes no model and
performs no inference/training/download. It reproduces the frozen report and separately
checks confusion counts, balanced accuracy, selective metrics and pairwise-rank AUC by
direct formulas. Evidence: `evidence/e92_readiness_audit_2026-09-14.json` (12,308 bytes).
The10 gate constants and two cuts match the original E49 evaluation commit b173d5b,
2026-09-04; they were not relaxed to make E92 pass. The rules originate in E45/E46
project acceptance budgets, not an external certification or market cost assessment.
Two criteria are complements, several others correlated, and the two conditions are
paired;20/20 is not20 independent tests or100% classification accuracy.

Key findings, with full rationale and primary-source links in ml/EXPERIMENTS.md:
- No observed TRAIN/DEV parent-ID or file-SHA intersections across12,269 TRAIN/320 DEV
  parents. Canonical pixel hashes cover only639 TRAIN rows in the current metadata
  audit. Earlier perceptual screening helps but cannot establish unknown scene/prompt
  or pretrained-corpus independence. No affirmative direct identity leak was found.
- E66 was adaptively reused from E70 through E92. Keeping its rows out of fitting
  prevents one leakage route, but does not remove model-selection/holdout overfitting.
- E92's social REAL errors concentrate: scene0026/15=40%, scene0067/19=36.8421%;13/14
  errors occur in these two scenes. Worst-camera18.75% passes while worst-scene40%
  was not in the original criteria. This is diagnostic, not a new retroactive gate.
- Social covered accuracy300/315=95.2381% would become94.9206% with one additional
  error at fixed coverage. Both conditions contain one AI below the internal REAL cut.
  Neither zero original REAL errors nor20/20 means zero errors or certified authenticity.
- SIDD is a2018,10-scene denoising dataset with RAW-derived non-tone-mapped sRGB from
  five older phones. SID is also2018 low-light research data. Some TRAIN AI labels
  are2026-era, but DEV tests only GPT Image1 and nano-banana-local. Acquisition date
  is not capture date or current-generator coverage. Demographic fairness is unmeasured.
- JPEG/PNG, dimensions, processing and semantic/source shortcuts remain plausible.
  Explicitly supersede the early HISTORY assertion that fixed tiles make biased data
  "perfectly safe". Tiling cannot erase all compression/processing/content traces.
- Conditional algebra, not market evidence: if social TPR/FPR transferred to10% AI
  prevalence, AI-positive precision would be55.79%; at1% AI,10.29%. The balanced DEV
  result cannot establish trustworthiness in a mostly-real gallery.
- E92 combines pretrained DINO/CLIP/DEAR with project-trained adaptation. DEAR weights
  have CC BY-NC plus further restrictions; MIDD and other data have research limits.
  Publicly accessible code/weights do not clear commercial use or all upstream rights.
- The web still presents E32 R1b and serve.py retains E20/older research models.
  E92 cached score timing excludes encoders and image decoding. No native-image E92
  serving parity, load/latency qualification or accurate E92 UI integration is complete.

Historical continuity: early CIFAKE/GenImage and native-tile phases exposed format,
resolution and source-transfer limits; localization metrics also needed correction.
Later E20/E32/E43 research and stronger versioned runtime/test contracts improved
reproducibility without proving generalization. E49 was consumed historically by E43
and failed11/20; its later reuse cannot count as a fresh independent final. E70–E92
are a documented rejection/adaptation chain, not92 independent successful trials.
The856-test suite missed the E92 happy-path manifest bug; the repaired858-test suite
is useful engineering evidence, not ML market validation. Archive claims remain intact
with these explicit dated corrections; no frozen research code was changed by this audit.

Verdict: E92 is a credible student research milestone that can be presented with the
population, limits and remaining guard failure. A live E92 research prototype needs
separate runtime and licence-compliant scope. It is not presently a validated paid or
public general-purpose authenticity detector. Market demand/competitive advantage are
unmeasured. Prioritize a prospectively locked representative evaluation, TRAIN-only
shortcut/group diagnostics, calibrated product risk and lawful/runtime integration
before another DEV-driven fit. Do not target the known missed sample or remove its
label/guard. No deployment or new ML experiment was executed. Only the three English
living Markdown records were updated; small primary text pages were researched with
no image/model/package acquisition.


## Local internship scope and E93 implementation registration (2026-09-14)

The user clarified that this is an internship project for a professor, with no
immediate formal report/presentation. Authorized work: connect E92 to a plain-language
local demo with error protection; revisit the Model 2 local AI-edit plan; research
and apply Model 1 reliability checks using existing data. Mobile-data restrictions
remain. This explicitly supersedes the prior review-only/no-serving pause for local
research use, without waiving E92 acceptance or authorizing public deployment.
Registered a fixed two-view abstention rule and deterministic native parity followed
by the existing E65 REAL-only source diagnostic; see PLAN and EXPERIMENTS. Existing
E20/E32 legacy code and all frozen ML evidence remain separate from the new demo.


### E93 initial startup failure and pre-score repair (2026-09-14)

The explicit local-only DINO loader initially omitted timm's official checkpoint
filter; strict state loading rejected positional-embedding1370/257 shape mismatch.
No image prediction, parity result or E65 score existed. Preserve attempt1 contract
and startup receipt; apply the same installed checkpoint_filter_fn used by the frozen
pretrained path, then re-register the adapter before any image scoring. This is a
loading repair, not model selection. Frozen E92 and all experiment helpers are unchanged.


## E93/E94 — E92 local internship demo, offline transfer check and Model 2 plan (2026-09-14)

Implemented the user's three requests within the local internship scope. No dataset,
weight or package downloads, no new fit, no public hosting and no E49 scoring. Only
PLAN, HISTORY and ml/EXPERIMENTS carry new Markdown progress. Existing artifacts and
frozen experiment sources are unchanged.

### What now runs

The web demo uses E92 through `pixelproof.internship_serve:app` on127.0.0.1:8800.
The preview started at http://localhost:3002/ because3000/3001 were occupied. A local
HTTP GET returned200; preview opening was queued by the app. Legacy8799/E20/E32
service was not terminated. The runtime adapter verifies local E43/E92/CLIP/DEAR
artifacts and its frozen helper manifest; explicit DINO cache loading uses timm's
same positional-embedding filter. All encoders operate offline. This remains a
repository-backed research adapter, not a portable/public weight distribution.

The main interface has no score/probability bar, false-positive jargon or misleading
"REAL" certificate. It says AI traces found, no clear trace found, or unable to decide.
It explains local temporary memory processing, no upload archive/training, and the
absence of a local-edit heatmap. A new response schema checks E92's exact artifact and
presentation-policy identity before displaying anything. Errors and replaced files
clear old results; a100s client timeout cancels the request. Backend safeguards bound
streamed uploads to12MiB, geometry to16MP and one concurrent request; animations,
transparent and unsupported colour inputs are rejected, under224px inputs abstain.
30s upload/90s inference response timeouts have explicit messages. A timed-out GPU
worker retains its slot until it actually finishes, preventing overlapping inference.
Missing/changed assets and invalid predictions fail without silently using an old model.

### The protection tradeoff was measured, not hidden

E93's frozen strict two-view AND rule reduced E66 AI indications from159 to158/160.
That conflicts with the user's original-alert retention requirement, so it is NOT the
active demo decision rule. Preserve its prototype `research_serve.py` and results as
experimental evidence. The separate current `e92-preserve-alerts-v1` policy keeps every
original E92 AI indication visible, attaching a prominent warning when the E93 check
is uncertain. Only negative evidence can be replaced by abstention. It does not reduce
raw false alerts: E65 still has2/83 original false indications, one with a warning.
A low score is never a probability of authenticity. All403 locked parent pairs replay
with zero added/lost original AI alerts. This is presentation/error protection, not
new learning, a revised E92 score or a claim that OOD is solved.

E92 native parity is exact on14 deterministic source-representative views; real HTTP
integration passes six outcome-selected cases (including unstable positive, uncertain
AI, uncertain REAL, no-clear REAL and false positive). Native pair inference1.13–1.89s,
weight startup17.59s; six local HTTP cases1.11–2.56s. These are small MPS engineering
samples, not production p50/p95. Raw scientific inputs and API admission differ:
all160 SIDD original PNGs exceed12MiB; only23/83 E65 originals pass input limits.
No rejected file was silently converted or removed from scientific counts.

### Offline source-shift evidence

E92 on83 previously admitted E65 REAL parents:2 false AI originals,1 false AI Q75;
E43 previously14 originals/18 Q75. WIFD contributes2/67 and1/67; RawNIND0/16 in both.
The entire fixed population was scored; nothing was fit or selected using its labels.
However these publishers were already exposed as diagnostics and E65 has no AI images.
This supports narrower REAL-transfer evidence, not fresh-final accuracy or retention
on unseen generators. E92's20/20 numeric DEV milestone and failed E43 AI guard remain.
The current UI preserves159/160 original E66 AI alerts; one receives an extra warning.
It abstains on89/160 E66 REAL observations, a visible coverage cost, not hidden success.

### Model 2 evidence and concrete next work

Read E17/E18 and the old two-module plan. Existing CocoGlide provides512 edited images,
512 binary/nondegenerate geometry-matched masks and512 linked authentic files. Initial
audit falsely reported missing links because it omitted the compilation's documented
extra `.png`; attempt1 evidence is retained and a precise, ambiguity-rejecting resolver
corrects this without renaming data. Authentic exact hashes are unique; semantic/near
parent duplicates remain unaudited. Mask median area15.974%;93 masks cover only1–5%.

A key limitation in E17 is its per-image percentile threshold using true mask area:
its IoU was an oracle ranking diagnostic, not a deployable localiser. The >=50% tile
coverage filter also dropped85/120 candidate images. Next: group/provenance audit,
complete-coverage64px stride evaluator, CAL-only threshold, pixel AUC/AP and area-
stratified image-level metrics against random/centre/authentic controls. Only then a
frozen DINO dense-token linear head, using eligible grouped TRAIN masks. Do not promise
localisation for classic splicing or fully re-rendered AI edits. No Model2 training or
heatmap release occurred. Detailed prospective ladder is at the top of PLAN.

### Validation and checkpoints

- 887 Python tests passed in16.92s (one existing Starlette/httpx deprecation warning).
- Web lint/typecheck and Sites build passed;10 Node contract/SSR/asset tests passed.
- Native parity14/14, original-alert display replay403/403, real HTTP6/6 passed.
- No browser visual QA was requested or claimed. No package/environment upgrade.
- Evidence: e93_runtime_manifest, e93_native_parity, e93_demo_validation,
  e94_display_audit, e94_http_smoke and model2_local_inventory_2026-09-14 JSON receipts.

Local start (the existing environment and disk are required):

```sh
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets PYTHONPATH=ml:ml/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIXELPROOF_CORS_ORIGINS=http://localhost:3002,http://127.0.0.1:3002 ml/.venv/bin/python -m uvicorn pixelproof.internship_serve:app --host 127.0.0.1 --port 8800
NEXT_PUBLIC_PIXELPROOF_API_URL=http://127.0.0.1:8800 npm run dev -- --port 3002
```

Use the actual Local URL printed by vinext and explicitly allow that origin if its
port is occupied. The commands are separate foreground terminals. Do not launch the
frozen E93 prototype as the current demo. Git synchronization is verified at final handoff;
local check success is not an assertion that the latest remote CI completed.


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


## E95/E96 completed — private gallery analysis and native24MP demo support (2026-09-14)

The user's project gallery is the historically relocated PixelProof Workspace/Samples
owner gallery. Verified the same210 declared-REAL stills/206 unique byte parents
(identity390e3c21…ac09); one protected reserve still and one MOV were excluded. Scored
all206 unique parents twice with frozen E92, retaining210-file counts and duplicates.
No downloads, fitting, threshold selection, source-specific exceptions or E49 access.
Individual image paths/scores and review details stay under ignored ml/work/e95_owner_gallery;
only aggregate evidence/code/these three records are committed.

Results on all210 files:9 original false AI indications (4.29%),18 Q75 (8.57%). The
current original-submission display yields165 no-clear-signal,36 uncertain,9 AI;3 of
those9 AI indications carry an inconsistency warning. These165 are NOT certified REAL,
and the36 abstentions must not be counted as successful classifications. On206 unique
byte parents:9/18 false indications and161/36/9 display counts. Camera bursts/scenes are
still correlated. The old E43 yields15/28 false indications on the same images/views.

Paired comparison exposes the hidden tradeoff: E92 rescues13 old original false
indications but creates7 new ones; social rescues12 and creates2. Hence lower pooled
FPR does not mean every photo improved. The old16MP API-accepted137-file subset even
has7 E92 false indications versus5 E43, despite pooled15→9 improvement. Keep these
failures visible, not removed from the denominator.

Limited direct visual inspection of high-scoring errors found an application-screen
capture, a water-texture scene and a night street photograph. Do not reproduce its
private text, faces, locations or filenames in Git. This is qualitative posthoc review,
not proof that noise reduction/sharpening/texture selection causes the errors. Screenshot
scope and scene/transport differences are distinct hypotheses. The screenshot remains
in the declared gallery counts; no posthoc removal improves the reported result.

Concrete development:
-73 gallery files are5712x4284 (24.47MP). Replaced only the current local demo's16MP
 admission profile with32MP. All210 now pass instead of137. Kept12MiB, one inference
 slot, dimension/aspect bounds, unsupported-format/alpha/animation checks and timeouts.
 No downsampling or recompression is applied in decoding. All210 decoded RGB arrays
 exactly match the native E95 inputs; old accepted images also match the old decoder.
- Updated frontend response geometry validation and visible limits. Added plain input
 scope explaining that documents/application screens are not the intended photo input.
 This copy does not suppress an AI alert or claim automatic screenshot detection.
- Added a paired consumed-development regression checker. Frozen private baseline:
 206 owner REAL+160 E66 AI parents, two conditions=732 views. It rejects missing,
 duplicate/relabelled/rehashed rows, nonfinite scores, any newly wrong REAL or lost AI;
 net rescues cannot cancel a new error. CLI comparison exits2 on failure. Reference
 replay passes; an explicitly synthetic corrupt-score probe fails. E43→E92 posthoc
 replay correctly fails with9 new REAL-error views and1 lost AI view. This diagnoses
 current limitations, not a retroactive waiver or rewrite of original acceptance.

The active API was restarted on127.0.0.1:8800 with the new32MP decoder. Six targeted
HTTP cases pass: three SHA-selected newly eligible24MP originals plus three existing
AI outcomes, including the unstable alert. All match their frozen expected results.
The three gallery results deliberately include no-clear, uncertain and false AI;
admitting the image is not correcting its prediction. Targeted timings1.10–2.49s
are not production percentiles. The preview remains http://localhost:3002/.

898 Python tests pass (19.94s, one existing warning); web lint/typecheck/build and11
Node contract/SSR tests pass. New24MP support is tested for native pixel equality,
unchanged small-image decoding and rejection above32MP. Gallery images remain private.
Further ML work is registered as a TRAIN-only processing/context hypothesis in PLAN;
this task does not claim the nine original model errors have been fixed.


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


E99 source inventory and selection completed:256 original TRAIN JPEGs across four new
MIDD sensors,3,323,883,735B. Contract0f67891068521533355ec39a90e3d4b396c14bb77b42fff9fc29a77745703729.
Two selection tests pass for order invariance, split exclusions, duplicate/path/capacity
rejection and no-refill overflow handling. Transfer now starts; no completion/admission
or detector-quality claim. Full acquisition record is in DATASETS.md as requested.


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


E101 frozen and started:256 parents/1024 TRAIN views, contract SHA2560a4dd9a88cb40810412647df6123aec01827ef6cd0635036577f74dd92020e9d.
Two targeted tests pass for exact E89 transform parity and rejection of CAL/changed-
original chunk identities. The local E92 API is temporarily stopped to avoid concurrent
GPU memory use; frontend remains available and the same API must be restored after the
encoder stage. No E92 weights/cuts/policy changed. Feature completion is not yet claimed.


E101/E102 engineering checkpoint:915 Python tests pass in19.26s (existing warning only).
Additional conditional-DEV test confirms failed TRAIN denies cache access. E102 uses
frozen E92 representation coordinates and zero delta initialization; all original AI
views protected relative to E92, old gate replay required before optimization. The
prepared consumed-DEV runner additionally requires zero lost E43/E92 AI and zero new
E92 REAL errors per source/condition. No current candidate fit/DEV score exists yet.


Remote verification checkpoint: commits7f30378dcd7847588a2790e17d8ad38c3b1d42d3
and ee57215b51ea2de43991d214151c8747c0d79ee5 passed GitHub Actions runs34854170864
and34858887430. Run34860324077 for55afb6a4ce507e88c3c067db4ef45203756a5719
was still in progress when checked; do not infer its result from the earlier runs.
E101 GPU extraction remains active, E102 fit remains prepared. No frozen dependency
or model artifact was altered by documentation/source commits.


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
