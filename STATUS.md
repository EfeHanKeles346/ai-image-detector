# Status — where we are, what we are doing

*Last updated: 2026-07-27. Read this when you lose the thread. `ROADMAP.md` holds the long-term
plan; you do not need to consult it day to day. Full detail for every result below is in
`ml/EXPERIMENTS.md` (E7–E10).*

---

## In one sentence

We have a working AI-image detector, and today we established that **how an image is fed to the
model matters as much as the model itself** — measured it, built an alternative that attacks the
problem directly, and found it to be a specialist rather than a replacement.

---

## What was measured today

**1. The detector degrades on modern generators (E7).** Tested on Defactify — 16,875 images from
five generators newer than anything in training. AUC fell from 0.888 (archive1) to **0.760**.

**2. The cause is our own preprocessing.** Results ordered themselves almost perfectly by source
resolution, in the direction opposite to a shortcut:

```
DALL-E 3     270px  → 0.896   (barely downscaled)
Midjourney   436px  → 0.821
SD 2.1       768px  → 0.696
SD 3        1024px  → 0.670   (downscaled 4.6x)
```

Everything is resized to 224×224 before the model sees it. Generation artefacts live in fine
texture; downscaling removes exactly that.

**3. A resolution-independent alternative works — as a specialist (E8).** 68 hand-crafted
statistics per image (frequency spectrum, noise residuals, cross-channel demosaicing traces,
compression footprint), computed over every pixel at native resolution, then gradient boosting.
Trained on the *same* GenImage split as the ResNet, so the comparison is controlled.

| | ResNet-18 | features |
|---|---|---|
| GenImage test | **0.982** | 0.919 |
| archive1 | **0.888** | 0.505 |
| Defactify | **0.760** | 0.717 |
| — SD 2.1 (768px) | 0.696 | **0.784** |
| — SD 3 (1024px) | 0.670 | **0.760** |
| — SDXL (1024px) | 0.717 | **0.867** |
| — DALL-E 3 (270px) | **0.896** | 0.377 |

The resolution ordering did not merely vanish — it inverted. The method wins by +0.09 to +0.15
exactly where the CNN is weakest, and collapses on small, heavily-compressed inputs.

**4. Combining them does not help (E9).** Eight blending rules tested; the best beats the ResNet
by +0.002 on average. A fixed blend *relocates* accuracy (Defactify +0.036, archive1 −0.036)
rather than adding it, because the feature model is near-random on archive1. **Decision: the demo
shows both scores side by side and flags disagreement instead of averaging.**

**5. archive1 is a badly confounded benchmark — but our CNNs cannot exploit it (E10).**
Real images are 100% JPEG and rectangular; AI images are 100% PNG and square 512×512. Width and
height alone separate the classes at AUC 1.000. Removing both confounds changed CNN performance
by **+0.008** — because `Resize()` destroys format and dimensions before the model sees anything.
So **E1's 77.1% and E6's 0.888 stand.** The caveat: the feature model *does* read native pixels
and its shortcut probe scores 92.6%, so this immunity does not transfer to it.

---

## The rule we have now learned three times

**Whatever you will feed the model at test time, you must feed it during training.**

| Experiment | Trained on | Given at test | Result |
|---|---|---|---|
| E5 | blurry 32→224 upscales | sharp native photos | called 984/995 images "AI" |
| E6 | native high-resolution | 32×32 CIFAKE | 50% — complete collapse |
| E7 | downscaled crops | native-resolution patches | 96% of real photos called "AI" |

`configs/genimage.yaml` sets `crop_augmentation: true`, which downscales every 1024px training
image by 3.8–4.6×. **The ResNet has never seen a native-resolution pixel.**

---

## Current demo

`serve.py` reports two independent families of evidence, deliberately unblended:

1. **CNN verdict (primary)** — routed by resolution (<128px → SmallCNN, else ResNet-18), with the
   uncertainty band, plus an `enough_evidence` flag below 48px.
2. **Feature analysis (secondary, experimental)** — both variants (`full`, `crop128`), each with
   its own verdict, plus an `agree` / `conflict` flag.

The UI shows the CNN result as the headline and the statistical signals underneath.

Run: `PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799` + `npm run dev`.

---

## Data on disk

| Name | Location | Role |
|---|---|---|
| CIFAKE | `~/Desktop/archive` | 32×32, training data for SmallCNN |
| archive1 | `~/Desktop/archive1` | 995 images. **Confounded** (see E10) — CNN numbers valid, native-resolution methods must control for it |
| GenImage | `~/Desktop/genimage_split` | 9,917 train / 1,742 test, 7 older generators. Perfectly balanced. Training set for both the ResNet and the feature model |
| **Defactify** | `~/Desktop/defactify_test` | **16,875 images, 5 modern generators**, both classes JPEG. Our best validated test set. Never used for training |

Cached feature matrices live in `ml/artifacts/features/` — re-running an analysis costs seconds.

---

## Next candidates (nothing committed yet)

1. **Retrain with native crops** (`RandomCrop` instead of `RandomResizedCrop`) so patch inference
   becomes consistent — the fix E7 points at, and it costs no more compute (measured: 224px input
   runs at 193 img/s either way; only the crop method changes).
2. **A properly controlled modern test set.** Public datasets reproduce archive1's flaw almost
   universally — AI tools emit square PNGs, cameras emit rectangular JPEGs. Building 30–50 images
   ourselves from ChatGPT/Gemini, with both classes pushed through one identical pipeline, is the
   only uncontaminated route to 2025–26 generators.
3. **Compression robustness** — still never tested, and every image on the internet is compressed.
4. **Calibration** — 44% of real photographs are still called "AI" at threshold 0.5.

---

## Known open problems

1. **Calibration** — ranking is decent, the threshold sits in the wrong place.
2. **Compression never tested.**
3. **Single seed** — every experiment so far, against our own ≥3 rule (`ROADMAP.md` §5).
4. **Shortcut leakage in the feature model** — 92.6% width-prediction accuracy; unquantified
   effect on E8's numbers.
5. **Module 2 (manipulation detection)** — `ela.py` written, never evaluated.
