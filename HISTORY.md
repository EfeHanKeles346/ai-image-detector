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
