# Datasets — inventory and assignment

What we hold, what each set is good for, and which module it feeds. Replaces the
former `STATUS.md`, whose content is now covered in more detail by `HISTORY.md` §2b.

**The rule that governs everything here** (`HISTORY.md` §1b): a dataset flaw is a
*usage condition*, not a disqualification. A shortcut only exists if the model can
perceive it. Whole-image training can see image dimensions; tile training cannot,
because every tile arrives at 128×128 regardless of where it came from.

---

## Storage

| Location | Contents |
|---|---|
| `$PIXELPROOF_WORK_ROOT` (default `ml/work/`) | Prepared working sets: `archive`, `archive1`, `genimage_split`, `defactify_test`, probes and temporary experiment encodings |
| `$PIXELPROOF_WORK_ROOT/manipulation_test/` | Module 2 working set — 10 sub-datasets extracted from the 78 GB compilation (6.7 GB, 2,385 manipulated images each with a mask + 2,289 authentic). Rebuild: `python -m pixelproof.prepare_manipulation` |
| `$PIXELPROOF_DATA_ROOT` (default `ml/data/`) | Acquired source datasets; may point at an external volume |

The original machine can retain its existing layout without code edits by exporting, for
example, `PIXELPROOF_WORK_ROOT=/path/to/prepared-work` and
`PIXELPROOF_DATA_ROOT=/path/to/pixelproof-datasets`. Active commands no longer contain a
personal absolute path. Both portable defaults are gitignored.

⚠️ `manipulation_test` lived in `/tmp/m2` until 2026-08-04, where macOS would have
wiped it on the next reboot — and it was the only copy. E17/E18 take `--root`, and
`prepare_manipulation.py` rebuilds the portable work directory from the data root.

Extracted by default: everything except **OpenForensics**, which is 138 of the
191 tars and is face manipulation rather than splicing or inpainting. One tar per
split (~500 images) — more buys nothing until E17's mask-coverage filter is
loosened, since only 35–95 images per set survive it today.

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
| `archive1` | 240 MB | **Confounded** (see `HISTORY.md` §1b). Keep only for continuity with E1–E6; do not use for new claims |
| `archive` (CIFAKE) | 469 MB | 32×32. Only SmallCNN's domain |

### E30 pinned current-science sources — role assignment before download

| Source (pinned revision) | Full source | E30 role and initial slice | Why selected | Boundary |
|---|---:|---|---|---|
| `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (`1498eead…b9de`) | 4,356 rows / 3.32 GB | **DEVELOPMENT TEST**; planned 180 preprocessed JPEGs: 20 per GPT Image 2 / Nano Banana 2 / real × texture / structure / hybrid cell | Matched 2026 generators and real class with three explicit artifact regimes; independent of detector score | Dataset card is research-use restricted. JPEG arm is standardized transport, not native output; raw arm remains separate |
| `Qwen/Qwen-Image-Bench` (`d2493deb…7038`) | 12.7 GB | **LOCKED FINAL TEST candidate**; first sealed scout is 5 each from 8 named 2026 generators (40 PNGs) | Independent collection and broad frontier coverage: GPT Image 2, Nano Banana 2, Seedream 5, Qwen Image 2 Pro, FLUX.2 Max/Pro, GLM-Image, Hunyuan Image 3 | Five per generator is scout-only. No success claim below 40 per reported generator; selected rows stay unscored until candidate/threshold freeze |
| `laionmobile/laion-mobile` (`0c60f598…3465`) | 935,399 metadata rows / 151 MB; evaluation manifest 9,115 rows / 2,639,565 B | **DEVELOPMENT TEST**; planned 8 declared phone/web pipeline groups × 10 local reconstructions | Real-only false-positive stress test with EXIF make/model and upstream content hashes, fetched row-wise rather than mirroring the corpus | Metadata is CC-BY-4.0; image licences remain upstream. Web-reprocessed and mostly older phones, not a native-camera vault |
| New private multi-phone vault | Not yet collected | **LOCKED FINAL TEST**, target 4 pipelines × 40 untouched originals | Only reliable way to match native iPhone/Samsung/Pixel computational-photography pipelines without web laundering | Existing owner gallery is exposed development regression; no personal bytes, names, GPS or per-image identifiers enter Git |

`ml/e30_sources.json` is the machine-readable source registry. The pinned sizes above are upstream
reported totals, not local acquisitions. Exact selected counts, downloaded bytes, hashes and audit
results will replace the planned slice descriptions after E30-A2/A3 realization. E30 test bytes
remain under ignored `ml/data/e30/` and are forbidden from TRAIN/CALIBRATION.

### E30-A2 low-bandwidth realization (2026-08-25)

| Arm | Realized local data | Audit outcome | Scientific use |
|---|---:|---|---|
| MLLMGenSet parents | 180 JPEGs / 4,419,610 B: 120 AI, 60 matched real; exactly 20 per nine frozen generator/class x regime cells | 180 unique SHA-256; metadata-only AUC 0.6238, pass | DEVELOPMENT TEST only; standardized-JPEG GPT Image 2 / Nano Banana 2 diagnostic |
| MLLMGenSet derivatives | 720 JPEGs / 14,029,255 B: q90, q75, q50 and resize256-q90 for every parent | 900/900 hashes unique across parents+children; transport AUCs 0.6096, 0.6191, 0.6362, 0.6127, all pass | Robustness views of the same underlying content; never independent samples or another split |
| LAION-Mobile attempt | Metadata manifest 2,639,565 B; 55/80 URLs eligible under the frozen 375 KB/file rule; **zero images downloaded** | `source_incomplete`: Apple cells 10/10 each; Samsung/Xiaomi cells 9/10, 5/10, 1/10, 0/10. 287/361 rejects exceeded the per-file cap | No benchmark arm exists yet. Do not report the 55-row partial selection or substitute other phone groups |

The complete realized MLLM image battery is **18,448,865 bytes**, below the 30 MB target. Its
parent content-set SHA-256 is `1f3a7333...df2e`; the parents-plus-derivatives content-set SHA-256
is `7634755c...24b8`. The frozen parent selection remains `f71c8d02...035e`. All third-party
bytes and detailed URL diagnostics stay under ignored `ml/data/e30/`; the compact, presentation-safe
aggregate is `evidence/e30_development_realization.json`.

The LAION outcome is not repaired post hoc. Selecting the ten smallest reachable candidates in
each frozen phone cell would require about 45.96 MB for that arm alone (the Redmi cell about
22.81 MB), exceeding both its 30 MB arm contract and the complete low-bandwidth development
budget once MLLM is included. A future full-internet profile may pre-register a larger budget or
replace this source with a native multi-phone vault, but must create a new selection version.

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

**Status: parked, not served.** The current tile overlay is an uncalibrated detector-score map.
E17/E18 found localisation signal on diffusion inpainting (CocoGlide) but not a general result on
classic splicing. Module 2 resumes only after a localisation model is evaluated against the pixel
masks below on the relevant manipulation family.

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
2. **No realized native 2026 editing arm yet.** E30 pins GPT Image 2 / Nano Banana 2 data whose
   paper covers direct generation, reference reconstruction and local editing, but the compact
   local slice has not yet been downloaded/audited and the exposed HF folder hierarchy does not
   carry every protocol field. It cannot close Module 2's localisation gap by assumption.
3. **Compression is measured but remains regime-specific.** E23c evaluated q50 degradation and
   showed thresholds do not transfer safely between compression regimes. E30 therefore keeps
   native/standardized/q90/q75/q50 claims separate instead of treating augmentation as a cure.
4. **The AI-only sets are unusable as-is.** They need either a controlled real half or tile-mode
   evaluation.
