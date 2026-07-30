# Datasets — inventory and assignment

What we hold, what each set is good for, and which module it feeds. Replaces the
former `STATUS.md`, whose content is now covered in more detail by `ROADMAP.md` §2b.

**The rule that governs everything here** (`ROADMAP.md` §1b): a dataset flaw is a
*usage condition*, not a disqualification. A shortcut only exists if the model can
perceive it. Whole-image training can see image dimensions; tile training cannot,
because every tile arrives at 128×128 regardless of where it came from.

---

## Storage

| Location | Contents |
|---|---|
| `~/Desktop/` | The four original sets: `archive` (CIFAKE), `archive1`, `genimage_split`, `defactify_test` |
| `/Volumes/LaCie/pixelproof-datasets/` | 255 GB acquired 2026-07-28/29 |

⚠️ Nothing on the SSD is reachable when it is unmounted. Scripts hard-code the path.

---

## Module 1 — is this image AI-generated?

### Training

| Dataset | Size | Contents | Mode | Why |
|---|---|---|---|---|
| **`OwensLab/CommunityForensics-Small`** | 47 of 260 GB | **228 distinct generator models**, per-image `model_name` / `prompt` / `architecture` / `real_source` | any | The single most valuable training set we hold. Generator diversity is the known lever for cross-generator generalisation, and the metadata makes leave-one-generator-out possible — the only honest way to claim generalisation |
| **`theminji/AI-vs-Real-balanced`** | 12 GB | 19,960 AI / 19,660 real, mixed formats both sides | any | Clean and balanced. Good second source so training is not tied to one collection's quirks |
| `TheKernel01/AIGC-Detection-Benchmark` | 30 GB | 18 generators incl. GANs, balanced | **tiles only** | Shape trap: AI 100% square, real 40%. Adds GAN-era diversity that CommunityForensics (all latent-diffusion) lacks |
| `theminji/ai-vs-real-200k` | 49 GB | 200k balanced | **tiles only** | Resolution trap: AI median 1024px, real 263px. Volume, but the least clean of the four |
| `genimage_split` | ~2 GB | 7 older generators, perfectly balanced | any | Keep as the control: every model to date was trained on it, so it is the fair comparison baseline |

**Recommended starting mix:** CommunityForensics-Small + AI-vs-Real-balanced (~59 GB, both clean).
Add the two tile-only sets once tile training is the default, for GAN coverage and volume.

### Testing — never train on these

| Dataset | Size | Why it is a test set |
|---|---|---|
| **`defactify_test`** | 1.2 GB | Five generators newer than any training data, both classes JPEG. Our established benchmark — every number in `EXPERIMENTS.md` E7–E11 is measured here |
| **`julienlucas/midjourney-dalle-sd-nanobananapro`** | 2.9 GB | Contains **Nano Banana Pro** (2026) with real photos, formats mixed on both sides. The cleanest modern set we have; small enough to be a test set rather than training data |
| `archive1` | 240 MB | **Confounded** (see `ROADMAP.md` §1b). Keep only for continuity with E1–E6; do not use for new claims |
| `archive` (CIFAKE) | 469 MB | 32×32. Only SmallCNN's domain |

### Current generators, AI-only — pair with care

| Dataset | Size | Model | Era |
|---|---|---|---|
| `bitmind/nano-banana` + `Nano-banana-150k` | 24 GB | Gemini 2.5 Flash Image | 2025 |
| `kaupane/nano-banana-pro-gen` + `ash12321/…-1k` | 2.5 GB | Nano Banana Pro | 2026 |
| `ash12321/flux-1-dev-generated-10k` | 3.0 GB | FLUX.1-dev | 2024-25 |
| `a3xrfgb/gpt-image-mega-4k` | 3.3 GB (partial) | GPT Image, 4K | 2025-26 |
| `34data/communityforensics-fake` / `-real` | 3.3 GB | CommunityForensics sample | — |

⚠️ These have no real half. Pairing them with camera photos **recreates the archive1 trap**:
they are PNG squares, photos are JPEG rectangles. Two safe options — push both classes through
one identical encoder, or use them in tile mode only.

Best immediate use: **per-generator recall probes**. Score them with an existing model and read
the recall; that needs no real half and answers "does our detector see FLUX at all?"

---

## Module 2 — where was the image manipulated?

| Dataset | Size | Contents |
|---|---|---|
| **`ductai199x/image-manipulation-dataset-compilation`** | 78 GB | 13 forensic datasets, split `auth` / `manip`, **with pixel-level ground-truth masks** |

Per-image files:

```
<name>.png            the image
<name>.mask.png       binary mask — 0 / 255, same dimensions, marks the tampered pixels
<name>.json           {"manip_label": 1, "auth": "…/Au_ani_00018.jpg"}   ← points at the ORIGINAL
<name>.cls            class label
```

Verified on a CASIA 2.0 sample: 384×256 image, mask covering 37% of pixels in a contiguous band
(y 0–117). The `auth` pointer means we also have **before/after pairs** of the same scene.

### What is inside, and why the split matters

| Sub-dataset | Tars | Manipulation type | Expected difficulty for our tile model |
|---|---|---|---|
| OpenForensics | 139 | face manipulation | unknown |
| CASIA 2.0 | 26 | classic splice / copy-move | **hard** — Photoshop, not AI |
| **CocoGlide** | 2 | **diffusion inpainting** | **easiest** — genuinely AI-filled regions |
| IMD2020, NIST2016, Columbia, Coverage, DSO-1, CMFD, RealisticTampering, VIPP | 12 | classic edits | hard |

**Report per sub-dataset, never as one average.** Our tile model asks *"does this tile look like
AI-generated texture"*, not *"was this tile edited"*. Those coincide for a diffusion-inpainted
region and diverge for a Photoshop splice. A single pooled number would hide exactly the
distinction that matters.

---

## Assignment summary

```
MODULE 1 train   CommunityForensics-Small ┐
                 AI-vs-Real-balanced      ├─ clean, any mode
                 genimage_split           ┘  (control)
                 AIGC-Detection-Benchmark ┐
                 ai-vs-real-200k          ┘  tiles only

MODULE 1 test    defactify_test              established benchmark
                 julienlucas                 Nano Banana Pro, cleanest modern
                 AI-only sets                per-generator recall probes

MODULE 2         image-manipulation-compilation   masks, 13 sub-datasets
                 └─ report per sub-dataset, not pooled
```

---

## Gaps

1. **CommunityForensics is 47 of 260 GB.** Before training on it, check whether the shards we
   hold are representative — if they are ordered by generator, our slice may cover only a few of
   the 228 models. (The auditor already learned this lesson once, on `theminji/ai-vs-real-200k`.)
2. **No 2026-era editing data.** Every manipulation set is classic or 2023-era inpainting. There
   is nothing from ChatGPT or Gemini image editing, and per `IMAGE_FORENSICS_REFERENCE.md` §3.2
   those re-render every pixel — a different problem that localisation may not address at all.
3. **Nothing has been evaluated under compression.** Every image on the internet is recompressed;
   none of our measurements are.
4. **The AI-only sets are unusable as-is.** They need either a controlled real half or tile-mode
   evaluation.
