# Image Structure — Working Notes for Feature-Based Detection

> **Purpose.** Answers to the design questions raised on 2026-07-27, when we decided to test a
> resolution-independent, feature-based detector (extract a fixed-length vector of statistics from
> every image, then classify with classical ML). Companion to `IMAGE_FORENSICS_REFERENCE.md`,
> which covers *what traces exist*; this file covers *how an image is physically structured* and
> what that means for building a standard evaluation pipeline.

---

## Q1. What is an image, structurally? Is the structure constant across resolutions?

An image in memory is a **3-dimensional array**:

```
Height  ×  Width  ×  Channels
```

Each cell holds one number, usually 0–255 (8 bits per channel).

**The channel count does NOT depend on resolution.** A 32×32 RGB image and a 4000×3000 RGB
image both have exactly 3 channels. Only Height and Width change.

This is precisely why the feature approach works: any statistic computed *per channel* and
normalized *per pixel* produces the same-length vector regardless of image size. Resolution
changes how many pixels we average over — never how many numbers come out.

---

## Q2. How many layers (channels), and does it change by format?

Yes, channel count varies — but by **color mode**, not by resolution.

| Mode | Channels | Meaning |
|---|---|---|
| L (grayscale) | 1 | brightness only |
| RGB | 3 | red, green, blue |
| RGBA | 4 | + alpha (transparency) |
| CMYK | 4 | print color space |
| YCbCr | 3 | luminance + 2 chrominance (JPEG's internal space) |

Per format:

| Format | Typical channels | Bit depth | Notes |
|---|---|---|---|
| **JPEG** | 3 | 8 | Lossy. Stored internally as **YCbCr**, not RGB. No alpha. |
| **PNG** | 1–4 | 8 or 16 | Lossless. Alpha common. |
| **WEBP** | 3–4 | 8 | Lossy or lossless. Different transform than JPEG. |
| **HEIC** | 3 | 8–10 | Apple default; often 10-bit. |
| **TIFF** | arbitrary | 8/16/32 | Can carry many channels. |

### The JPEG detail that matters most

JPEG does **not** store RGB. It converts to YCbCr (one luminance channel + two color channels)
and then usually **subsamples the color channels to half resolution** (4:2:0 chroma subsampling).

So a "1024×1024 JPEG" actually holds:

```
Y  (brightness)  1024 × 1024
Cb (color)        512 ×  512
Cr (color)        512 ×  512
```

**Consequence:** inside a single JPEG the channels are not even the same size, and fine detail
survives far better in brightness than in color. Any feature that mixes them without accounting
for this is measuring two different things at once.

---

## Q3. Can the channels be evaluated independently? Can one model handle them all at once?

**They are NOT independent — and the dependency is itself one of the strongest forensic signals.**

Why, physically (see `IMAGE_FORENSICS_REFERENCE.md` §1):

- A real camera sensor measures **only one color per photosite** (Bayer color filter array). The
  other two channel values at every pixel are **interpolated from neighboring pixels** during
  demosaicing. This creates a strong, structured, periodic correlation *between* the channels
  across the entire image.
- A latent-diffusion image never passed through a sensor. There is no CFA and no demosaicing, so
  this specific inter-channel correlation is **absent or different**.

Therefore: analyzing R, G and B in isolation **throws away the CFA trace** — one of the few
signals that is about physics rather than about a particular generator, and therefore one of the
few that should survive when a new generator ships.

**Answer to "can one model do it in one shot": yes.** Not by processing channels separately and
voting, but by putting three groups of numbers into the *same* feature vector:

1. **Per-channel statistics** — compute each statistic separately on R, G, B.
2. **Cross-channel statistics** — correlations between channels, channel differences,
   residual-after-demosaicing-prediction. This is where the CFA trace lives.
3. **Alternative color spaces** — YCbCr (matches JPEG's own space, so compression artifacts show
   up cleanly) and HSV (saturation statistics differ measurably between real and generated).

All of it concatenates into one flat vector, and one classifier consumes it. No separate models,
no fusion logic needed.

---

## Q4. Normalization — where does it help, and where does it destroy the evidence?

This is the central risk of the whole approach. The rule:

> **Normalize away what varies for reasons we don't care about (image size).
> Never normalize away what varies for reasons we do care about (texture, noise, color statistics).**

### Operations that DESTROY signal — do not apply

| Operation | What it destroys |
|---|---|
| **Resizing / downscaling** | High-frequency generation artifacts. Measured directly in E7 (2026-07-27): detection quality fell monotonically with the amount of downscaling. |
| **Per-image contrast/brightness standardization** | Global color and saturation statistics, which differ systematically between real and generated images. |
| **Grayscale conversion** | The entire cross-channel / CFA signal (Q3). |
| **Re-encoding (save as JPEG again)** | Overwrites the original compression history — the basis of ELA and all double-JPEG analysis. |
| **Denoising / sharpening** | The sensor-noise fingerprint itself. |

### Operations that are SAFE — and required

| Operation | Why it is safe |
|---|---|
| **Per-pixel averaging (densities, ratios)** | Removes dependence on pixel count without touching the underlying distribution. This is what makes the vector resolution-independent. |
| **Ratios between quantities measured on the same image** | Both terms scale together, so the ratio is scale-free. |
| **Feature-space standardization (z-score across the dataset, after extraction)** | Applied to the *numbers*, not to the *image*. Nothing is lost — it only puts features on a comparable scale for the classifier. |

The distinction that matters: normalize **after** extraction, in feature space — never **before**
extraction, in pixel space.

---

## Q5. What happens when the model meets an image structure it has never seen?

A real risk, and it has bitten this project three separate times already (E5, E6, and the patch
experiment on 2026-07-27 — all the same failure: input the model was not trained on ⇒ biased
toward "AI").

Mitigations, in order of importance:

1. **Canonicalize at load.** Convert every image to a single known mode (RGB, 8-bit) before
   extraction. Anything exotic (CMYK, 16-bit, palette, alpha) is converted once, consistently, by
   the same code path in training and inference. This is cheap insurance and costs almost no
   information for photographic content.

2. **Record the original structure as metadata, but keep it OUT of the training features.**
   Original format, mode, bit depth and resolution should be logged so we can slice results by
   them — but if they go into the feature vector the model can learn "PNG ⇒ AI", which is the
   format-shortcut trap documented in `IMAGE_FORENSICS_REFERENCE.md` §5. Diagnostics, not inputs.

3. **Test explicitly on structures held out of training.** The honest version of "will it
   generalize?" is a matrix: evaluate on formats and compression levels the training set never
   contained. This is the pending compression-robustness experiment.

4. **Prefer physics-based features over generator-specific ones.** Features that encode how a
   camera works (sensor noise, CFA correlation, JPEG history) keep their meaning when a new
   generator ships. Features that encode a particular generator's quirks do not.

5. **Know the floor.** Some inputs genuinely fall outside what any single model can serve — a
   32×32 thumbnail has no fine texture left to measure. The correct response is an explicit
   "insufficient evidence" output, not a confident guess. This makes the routing rule in
   `serve.py` principled ("can we measure this?") instead of an arbitrary pixel threshold.

---

## Practical consequence for the feature vector

Grouping, with the reason each group exists:

| Group | Captures | Resolution-independent by |
|---|---|---|
| Per-channel intensity statistics (mean, std, skew, kurtosis) | global color/tone behavior | statistics, not sums |
| Cross-channel correlations & differences | **CFA / demosaicing trace** — present in real, absent in generated | correlation is scale-free |
| Noise-residual statistics (image − denoised) | sensor noise fingerprint | per-pixel averages |
| Radial FFT power spectrum (binned) | upsampler / VAE decoder periodic traces | normalized by total energy, binned into fixed bands |
| High-frequency energy ratio | diffusion suppresses local high-frequency variance | ratio, not absolute |
| DCT coefficient statistics | JPEG compression history | histogram shape, normalized |
| Local variance distribution | texture consistency across the image | distribution summary |
| Saturation / HSV statistics | generated images sit in a different color regime | statistics |

Every entry is a summary statistic over **all pixels of the image**. Nothing is cropped, nothing
is resized, no pixel is skipped.

---

## Open questions to settle experimentally

- Does the resolution ordering seen in E7 (270px → 1024px monotonic decline) actually disappear
  with these features? That is the direct test of the whole premise.
- Do cross-channel (CFA) features carry more cross-generator generalization than frequency
  features? Ablate by group.
- Does a one-class model trained on real photographs only (Phase 5) hold up better on unseen
  generators than the supervised classifier? Same features, different learning setup.
