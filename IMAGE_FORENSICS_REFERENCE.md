# Image Generation, Editing & Forensics — Technical Reference

> **Purpose.** Context document for AI coding assistants (e.g. Claude Code) working on the
> PixelProof / ai-image-detector project. It explains (1) how real photographs are formed,
> (2) how generative models create images, (3) how AI and human editing actually work,
> (4) which forensic traces each process leaves, and (5) what that implies for detector and
> dataset design. Confidence levels are marked explicitly: **[established]** = textbook /
> peer-reviewed consensus, **[reported]** = vendor docs or community reports about closed
> systems, **[open problem]** = active research, no settled answer.
> All sources are listed at the bottom. Last updated: July 2026.

---

## 1. How a real photograph is formed (the camera pipeline)

A digital photo is not a neutral recording of light. Every stage of the pipeline leaves
statistical traces, and *all* forensic detection ultimately reads these traces.

1. **Sensor.** Light hits a CMOS/CCD sensor. Manufacturing imperfections give every camera a
   unique multiplicative noise pattern called **PRNU (Photo-Response Non-Uniformity)** — a
   stable, per-device fingerprint spread across every pixel of every photo the camera takes.
   On top of that: shot noise and read noise. Net effect: every real photo carries a fine,
   spatially consistent stochastic "grain" everywhere. **[established]**
2. **Color Filter Array + demosaicing.** Each photosite measures only ONE color channel
   (Bayer pattern). The other two channels are *interpolated* from neighbors. So ~2/3 of the
   color values in any real photo are predictions, and this interpolation leaves a periodic
   inter-pixel correlation pattern (CFA trace) across the whole image. **[established]**
3. **In-camera processing.** White balance, gamma, sharpening, denoising — each reshapes the
   noise statistics in characteristic ways. **[established]**
4. **JPEG compression.** The image is split into 8×8 blocks, each transformed with the DCT and
   quantized. This leaves blocking artifacts and characteristic DCT-coefficient statistics.
   Saving again ("double JPEG") leaves detectable inconsistencies between the two
   quantization histories. WEBP/HEIC use different transforms and leave different traces.
   **[established]**

**Key concept — processing history.** Every save, resize, recompress, or screenshot rewrites
or attenuates traces. Social-media "laundering" (resize + recompress) weakens *all* low-level
forensic signals; screenshots strip metadata entirely. Any detector claim must state which
processing history it survives.

---

## 2. How generative models create images

- **GANs (historical context).** Generator vs discriminator. Transposed-convolution
  upsampling leaves periodic "checkerboard" artifacts and strong, model-specific spectral
  fingerprints — the reason early detectors were frequency-based. **[established]**
- **Diffusion models.** Start from pure Gaussian noise; a trained network iteratively
  denoises it toward an image, guided by a text prompt (classifier-free guidance). Backbones:
  U-Net (SD1.5/2/XL) or transformer/DiT (SD3, FLUX). **[established]**
- **Latent diffusion (the dominant family: Stable Diffusion, SDXL, FLUX, most commercial
  tools).** A **VAE encoder** compresses the image ~8× per side (512×512 px → 64×64 latent);
  diffusion happens in that latent space; a **VAE decoder** maps the result back to pixels.
  Consequences that matter forensically: **[established]**
  - The output never passed through a camera: **no PRNU, no CFA correlation, no sensor
    noise.** Its fine texture is whatever the VAE decoder synthesizes.
  - Decoder upsampling leaves spectral traces; the diffusion process tends to **suppress
    local high-frequency variance** relative to optical imaging, creating a measurable
    statistical energy gap (basis of the FLAME localization method). **[established]**
  - The VAE encode→decode roundtrip is **lossy**: even pixels the model was told not to touch
    come back slightly changed. This single fact drives the spliced-vs-FR distinction in §3.1.
- **Autoregressive / native multimodal generators (gpt-image-1, GPT Image 1.5, Gemini's
  native image stack).** The model generates image tokens directly; internals are not
  published. Treat all mechanistic claims about them as **[reported]**.
- **Provenance signals.** Google adds a visible mark plus an invisible **SynthID** watermark
  to all images generated or edited in the Gemini app **[reported — Google's own blog]**.
  C2PA metadata exists in some tools but is stripped by screenshots and re-encoding.
  Watermark/provenance checks are complementary to, not a replacement for, pixel forensics.

---

## 3. How AI image *editing* works — the three paradigms

This is the most important section. "AI edited a photo" means very different things
depending on the tool, and the detector consequences differ radically.

### 3.1 Masked latent inpainting (SD/SDXL inpaint, FLUX.1-Fill, Adobe Firefly Generative Fill, DALL-E 2 edit)

Pipeline **[established]**:
1. VAE-encode the source image to latent space.
2. Downscale the user mask to latent resolution (1/8), usually blur it.
3. Run denoising **only inside the masked latent region**; at each step the unmasked latent
   is clamped/blended back to preserve context. Inpainting-specialized U-Nets take a
   9-channel input (noisy latent + masked-image latent + mask).
4. Decode the **whole** latent back to pixels.

Two possible outputs — the **spliced vs fully-regenerated (FR)** distinction:
- **Spliced:** the pipeline pastes the original pixels back outside the mask ("paste-back" /
  compositing). Outside-mask pixels are **bit-identical to the original**; only the inpainted
  region is synthetic. Side effects: possible boundary color/tone mismatch at the mask seam
  (caused by VAE imperfections; SDXL inpainting is known to discolor the full image, which is
  very visible when splicing). **[established]**
- **Fully regenerated (FR):** the pipeline just decodes the full latent. **Every pixel is
  rewritten** (because of the lossy VAE roundtrip), even though only the masked region changed
  *semantically*. **[established]**

The TGIF dataset was built specifically to provide *both* variants of the same edits
(SD2, SDXL, Firefly; TGIF2 adds FLUX.1) — see §5.

### 3.2 Full-regeneration editors (ChatGPT / gpt-image-1 / GPT Image 1.5)

- Multiple independent developer reports (2025) show that **gpt-image-1 re-creates the entire
  image even when an API mask is supplied**; the mask acts as *soft guidance*, unlike DALL-E 2,
  which performed true pixel-level masked replacement. **[reported]**
- OpenAI's own description of GPT Image 1.5 masking says the mask is *prompt-guided
  directional guidance* rather than a hard pixel boundary; a "high input fidelity" mode
  improves preservation of faces/logos. **[reported]**
- Some 2026 reports describe better mask adherence in newer versions. Until verified by a
  pixel-diff experiment, the **safe operational assumption is: any ChatGPT-family edit is a
  fully re-rendered image** — at the pixel level it is a *generated* image, regardless of how
  small the requested edit was. **[reported / verify empirically]**

### 3.3 Native targeted editing (Gemini 2.5 Flash Image "Nano Banana", Nano Banana Pro)

- Google markets "pixel-perfect editing": the model identifies the region of interest from
  the prompt and claims to re-synthesize only that region while preserving the rest.
  **[reported — vendor + third-party articles]**
- The internal mechanism is unpublished. "Preserved" may mean *visually indistinguishable*
  rather than *bit-identical*. Treat preservation claims as unverified until a pixel-diff test
  is run on actual outputs. All Gemini outputs carry SynthID (§2). **[reported]**

### 3.4 Instruction-based diffusion editing (InstructPix2Pix-style)

The whole latent is regenerated conditioned on (source image + instruction). FR-like by
construction: no pixel of the output is the original pixel. **[established]**

### 3.5 Classic human editing (Photoshop-style)

Operations and their traces **[established]**:
- **Splicing** (paste a region from another photo): donor region carries a *different* camera
  fingerprint, noise level, and JPEG history than the host → local inconsistency.
- **Copy-move** (clone within the same image): the cloned region is anomalously
  self-similar to another region of the same image.
- **Removal / content-aware fill, retouching, resampling** (scale/rotate before pasting):
  double-JPEG traces, edge halos, resampling periodicity, CFA disruption.
- **Boundary blur warning:** modern Photoshop's fill tools *are* generative AI (Firefly).
  "Human vs AI manipulation" increasingly means "which trace family is present", not who
  clicked the button. Hybrid cases (AI output manually retouched) exist.

---

## 4. Forensic traces and what detectors actually do

### 4.1 Two trace families, one core question

- **Camera traces** (present everywhere in a pristine photo): PRNU/sensor noise, CFA
  correlations, JPEG history.
- **Generator traces**: VAE/upsampler spectral patterns, suppressed high-frequency variance,
  model-specific fingerprints.

Detection is **single-image analysis** — at inference there is no original to diff against.
The core question is therefore: *is the trace texture spatially consistent, and which family
is it?*
- Consistent camera traces everywhere → **real**.
- Consistent generator traces everywhere → **fully AI-generated**.
- Localized inconsistency (camera traces outside, generator/foreign traces inside a region)
  → **local manipulation**, and the inconsistent region *is* the localization mask.

### 4.2 The spliced-vs-FR consequence (single most important finding)

- **IFL** (image forgery localization) methods rely on local trace inconsistency → they work
  on spliced images and classic edits, and **fail on FR images**, where global latent
  reconstruction destroys the local contrast they depend on. **[established — TGIF benchmark]**
- **SID** (synthetic image detection, whole-image) methods can flag FR images as fake but
  **cannot localize** the edited region. **[established — TGIF benchmark]**
- **Localization inside FR images is an open research problem.** Early attempts:
  DiffusionPrint (contrastive patch-level generative fingerprints), FLAME (high-frequency
  energy-gap map + SAM adapter), X-Edit. TGIF2 reports that fine-tuning on FR data improves
  FR localization somewhat. **[open problem]**
- **Product implication:** a ChatGPT-edited photo is, at the pixel level, a fully generated
  image. The semantic label "manipulated" cannot be recovered from pixels alone in the FR
  case; the honest verdict is "AI-regenerated / fully synthetic", optionally with an
  explanation that it may be a re-render of a real photo.

### 4.3 ELA (Error Level Analysis) — mechanism and exact scope

- Mechanism: re-save the image as JPEG at a fixed quality; map how much each region changes.
  Regions whose **compression history differs** from the rest respond differently.
- **Works:** JPEG splices where donor and host had different qualities/histories.
- **Fails by design (uniform/flat map is the *expected correct output*):** fully generated
  images (uniform history), FR edits (whole image re-rendered → uniform new history), any
  image uniformly re-encoded after editing, PNG/screenshot pipelines (no JPEG history).
- A flat ELA map on AI-generated or ChatGPT-edited images is **not a bug or a failed
  experiment** — it is the method behaving correctly outside its scope. Always pair a
  negative result with a **positive control** (a hand-made JPEG splice ELA *can* catch).

### 4.4 Learned detector families (canonical examples)

**SID (whole-image real-vs-synthetic):**
- **CNNSpot** — plain ResNet-50 + heavy augmentation (JPEG, blur); showed augmentation is the
  key to cross-generator generalization. **[established]**
- **Frequency methods** (FreDect, F3-Net) — detect spectral anomalies. **[established]**
- **NPR** — reinterprets upsampling artifacts as neighboring-pixel relations; efficient and
  generalizable. **[established]**
- **DIRE** — diffusion reconstruction error: real images reconstruct *worse* through a
  pretrained diffusion model than diffusion-generated ones. Weaknesses: slow (~40 model
  passes) and degrades on text-to-image models. **[established]**
- **UnivFD / CLIP-feature detectors** — frozen CLIP-ViT features + linear probe (or SVM);
  currently among the best out-of-distribution generalizers, even trained on little data.
  **[established]**
- Known failure modes: detectors trained on GANs collapse on diffusion images unless
  retrained; typical pattern is >95% in-distribution accuracy vs <75% out-of-distribution.
  **[established]**

**IFL (localization):**
- Lineage: ManTra-Net → SPAN → MVSS-Net (noise + edge supervision) → PSCC-Net → CAT-Net
  (explicit DCT/JPEG stream) → ObjectFormer.
- **TruFor** (current strong open baseline): fuses RGB with **Noiseprint++**, a noise-sensitive
  fingerprint learned self-supervised on real data only; transformer fusion; outputs
  **(a) pixel anomaly map, (b) whole-image integrity score, (c) confidence map** to suppress
  false alarms; ~69M params; code public. **[established]**
- **HiFi-Net / HiFi-Net++**: hierarchical formulation — Level 1: fully-synthesized vs
  partially-manipulated; Level 2: editing vs CNN/diffusion-based manipulation; deeper levels
  down to the specific generator; plus a localization branch. Demonstrates that
  generated-vs-manipulated and human-vs-AI distinctions are learnable as a hierarchy.
  **[established]**
- **2025–26 era:** SAM-adapter approaches (FLAME), contrastive generative fingerprints
  (DiffusionPrint), text-guided-edit localization (X-Edit). **[open problem area]**

**Common architecture pattern across modern IFL:** RGB branch + low-level branch (noise
residual / frequency / DCT), multi-scale encoder, segmentation decoder, multi-task loss
(pixel mask + image-level score), sometimes edge supervision.

### 4.5 Robustness realities

- JPEG/WEBP compression sharply degrades both SID and IFL; modern codecs (WEBP) are worse
  for detectors than classic JPEG. **[established — TGIF benchmark]**
- Train-time compression/blur augmentation is the standard generalization lever (CNNSpot);
  the trade-off against preserving subtle artifacts should be **measured, not assumed**
  (controlled experiment: augmented vs not, evaluated on clean *and* compressed sets).
- Resize/screenshot laundering strips metadata and attenuates traces.
- **Calibration does not transfer across domains even when ranking (AUC) does** — thresholds
  and temperature must be set per deployment domain. (Independently observed in this
  project's E6 experiment.)

---

## 5. Dataset map

| Purpose | Datasets | Notes |
|---|---|---|
| SID (real vs generated) | CIFAKE (32×32, CIFAR-10 vs SD — single generator); GenImage (7+ generators, native resolution; the "unbiased-tiny" subset avoids the JPEG-vs-PNG format shortcut); DiffusionForensics | Beware format shortcuts: if real=JPEG and fake=PNG, the model learns the codec, not the content. |
| Classic IFL (human edits, with masks) | CASIA v2, Columbia, NIST16, Coverage, IMD2020 | Ground-truth masks available. |
| AI-inpainting IFL | **TGIF** (~75k fakes; SD2/SDXL/Adobe Firefly; **both spliced and FR variants** of the same edits; masks; compression variants); **TGIF2** (+FLUX.1 edits, ~196k new fakes, random non-semantic masks, super-resolution impact analysis); CocoGlide; AutoSplice; OpenSDI; HiFi-IFDL (hierarchy labels); SID-Set | TGIF/TGIF2 are the closest match to this project's Module 2. TGIF sources authentic images from MS-COCO val2017. |

**Recommended evaluation protocol for this project:** a 3-column matrix —
classic edits / AI-spliced / AI-FR — each under clean, JPEG-75, and WEBP conditions;
image-level Acc/AUC/F1 + pixel-level F1/IoU for localization; leave-one-generator-out for
any generalization claim.

---

## 6. Mapping to this project (PixelProof)

- **Module 1 = SID**, **Module 2 = IFL.** Combined verdict logic = read two signals together:
  image-level synthetic score + pixel localization map (+ confidence).
  Empty map & low score → *real*. Localized map → *tampered* (map = "where").
  High score without local contrast → *fully AI* — **including FR edits** (taxonomy decision:
  report these honestly as "AI-regenerated", do not promise localization there).
- **ELA** = weak baseline valid only for the classic-JPEG-splice column (with a positive
  control). **Pretrained TruFor** = strong reference baseline (inference only, no training).
  The project's own learned model is positioned between these two.
- Known project findings that mirror the literature: CIFAKE→OOD drop (E1), preprocessing
  domain shift collapse (E5), resolution-domain blindness & catastrophic forgetting → dual
  routing (E6), calibration non-transfer (E6).

---

## References (accessed July 2026)

**Forgery localization & unified detection**
- TruFor — paper: https://arxiv.org/abs/2212.10957 · project/code: https://grip-unina.github.io/TruFor/
- HiFi-Net (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.pdf
- HiFi-Net++ (language-guided extension): https://arxiv.org/pdf/2410.23556
- FLAME (energy anomalies + SAM adapter): https://arxiv.org/pdf/2606.02178
- DiffusionPrint (FR-inpainting localization): https://arxiv.org/html/2604.12443
- X-Edit (detect & localize text-guided diffusion edits): https://arxiv.org/pdf/2505.11753
- Noise Doesn't Lie (universal deep-inpainting detection): https://arxiv.org/pdf/2106.01532

**Inpainting datasets & benchmarks**
- TGIF paper: https://arxiv.org/abs/2407.11566 · dataset/code/blog: https://github.com/IDLabMedia/tgif-dataset
- TGIF2: https://arxiv.org/html/2603.28613v1 · journal version: https://link.springer.com/article/10.1186/s13635-026-00235-9

**Synthetic-image detection (SID) methods & surveys**
- Survey — Methods and Trends in Detecting AI-Generated Images: https://arxiv.org/pdf/2502.15176
- Survey — Unmasking AI-created visual content: https://link.springer.com/article/10.1007/s44443-025-00154-8
- AIGI-Holmes (related-work overview of CNNSpot, FreDect, UnivFD, DIRE): https://arxiv.org/pdf/2507.02664
- CLIP-based lightweight detection & DIRE cost analysis (backbone study): https://arxiv.org/pdf/2605.14799
- Post-hoc distribution alignment (DIRE/ZeroFake limits discussion): https://arxiv.org/pdf/2502.10803
- Cross-domain OOD accuracy-drop statistics (ForensicFormer intro): https://arxiv.org/pdf/2601.08873

**How editors actually behave (closed models — treat as [reported])**
- gpt-image-1 mask ignored / full recreation (OpenAI dev forum):
  https://community.openai.com/t/image-editing-inpainting-with-a-mask-for-gpt-image-1-replaces-the-entire-image/1244275
  · https://community.openai.com/t/gpt-image-1-problems-with-mask-edits/1240639
  · https://community.openai.com/t/dall-e-gpt-image-1-edits-entire-image-instead-of-only-masked-area/1271182
- GPT Image 1.5 feature description (prompt-guided masks, input fidelity): https://www.imagine.art/blogs/gpt-image-1-5-features
- Gemini Nano Banana editing + SynthID (Google blog): https://blog.google/products-and-platforms/products/gemini/updated-image-editing-model/
  · tips post ("pixel-perfect editing"): https://blog.google/products-and-platforms/products/gemini/nano-banana-tips/
  · third-party overview: https://www.digitalocean.com/resources/articles/nano-banana

**Inpainting pipeline mechanics**
- SD inpainting internals (latent blending, mask at 1/8 scale): https://deepwiki.com/stable-diffusion-windows/stable-diffusion-windows/2.3-inpainting-and-outpainting
- Mask-conditioned inpainting overview (9-channel U-Net, clamping): https://www.emergentmind.com/topics/mask-conditioned-inpainting
- Lossy VAE roundtrip & pixel-exact paste-back: https://arxiv.org/pdf/2606.31603
- Boundary/color inconsistency causes (VAE + blending): https://arxiv.org/html/2506.12530v1 · ASUKA: https://arxiv.org/html/2601.15368v1

---

## Note to the assistant using this document

Bu dokümandaki bilgiler bir görev için yeterli gelmezse, daha fazla bilgi için **web
araştırması yapmayı unutma** — bu alan aylık hızla ilerliyor; 2024 ve sonrası kaynaklara
öncelik ver. (If the information here is insufficient for a task, do additional web research
before proceeding; prefer 2024+ sources.)

Useful starting search terms: `image forgery localization`, `text-guided inpainting
detection`, `fully regenerated inpainting localization`, `synthetic image detection
generalization`, `noiseprint`, `TGIF dataset benchmark`, `diffusion fingerprint detection`,
plus the specific tool name (e.g. `gpt-image-1 mask behavior`, `FLUX.1-Fill inpainting`).

Also remember the project-specific ground rules: single-image inference (no original to diff
against), evaluate under compression, never train on the OOD evaluation sets, and keep the
spliced-vs-FR distinction explicit in every dataset and every metric table.
