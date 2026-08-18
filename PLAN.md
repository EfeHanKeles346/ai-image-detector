# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(frozen) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

## Where the project stands (2026-08-18, after E20-v2 / E21 protocol work)

- **Best detector:** ResNet-18 fine-tuned on 128px native tiles — Defactify AUC 0.770,
  61.4% AI recall on the untouched evaluation half. Best numbers the project has produced.
- **Why it is not deployable:** a threshold fitted for 10% false positives reaches 19% on
  Defactify's own held-out half and up to **96% on the worst unseen camera source**. The
  bottleneck is no longer data (Phase 1) or representation (E20) — it is the operating
  point under source/pipeline shift.
- **Module 2** (where was it edited?) is measured and parked: tile localisation carries
  signal only on diffusion inpainting (CocoGlide); the classic-splice line is closed.

## Literature survey, 2026-08-18 — what the field says about our two blockers

A focused review (sources at the bottom of this section) mapped 2025–26 work onto the two
problems E20-v2 left open. Two findings matter more than the rest:

**1. Our E14 result is independently confirmed at scale.** A benchmark of 23 open-sourced
detectors run out-of-the-box ([arXiv 2602.07814](https://arxiv.org/html/2602.07814v1))
finds 20–60 point swings between identical architectures trained on different data, and
concludes that *training-data alignment outweighs architecture* — the same conclusion E14
and E20 reached here, measured independently. Two practical facts fall out of it:
the **Community-Forensics ViT-S** is the strongest single out-of-the-box detector (first
on 8 of 12 datasets, 75% mean accuracy, checkpoint on HuggingFace:
[`buildborderless/CommunityForensics-DeepfakeDet-ViT`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)),
and even the best frozen detectors collapse to 18–30% on 2026-era generators (Flux Dev,
Firefly v4, Midjourney v7) — so our DALL-E 3 failure has company, and our test sets need a
2026-era column.

**2. The narrow-real-class disease has a published cure to test.**
[Stay-Positive](https://arxiv.org/html/2502.07778v1) diagnoses exactly what E14 measured —
detectors learn spurious features of their *real* class — and constrains the final layer
to **non-negative weights** so the model can only accumulate evidence *for* generation
artefacts, never "unlike my training reals". Frozen backbone, minutes of retraining,
directly applicable to our tile ResNet-18, and our E20-v2 evaluator measures precisely the
number it claims to fix (worst-source FP, currently 96%).

Also relevant: [B-Free](https://arxiv.org/pdf/2412.17671) (CVPR 2025) generates fakes as
self-conditioned SD reconstructions of real images — content-aligned training pairs, the
training-data version of Defactify's content control, and claims better *calibration*
across 27 generators; [conformal abstention](https://arxiv.org/pdf/2502.07255) gives the
planned "insufficient evidence" band a statistical footing instead of a hand-tuned floor;
[TGIF/TGIF2](https://arxiv.org/abs/2407.11566) remains the right Module 2 target (FLUX.1
inpainting, and the only set separating spliced from fully-regenerated edits).

## Next, in order

1. **External baselines through our protocol** *(no training)* — every arm through the
   same E20-v2 evaluator: disjoint calibration/evaluation halves, threshold transferred to
   ten forensic real sources, macro + worst-source FP as headline columns.
   - [x] **Community-Forensics ViT-S** *(run 2026-08-19, see E21)* — beats our tile
     ResNet on every column (AUC 0.876, recall 70.8%, macro FP 29.9%) and **still fails
     the gate: 81.6% worst-source FP.** Representation-shopping alone does not solve
     cross-source specificity; CF-ViT becomes the strongest baseline going forward.
   - [x] **B-Free** *(run 2026-08-19, see E21b)* — best on nearly every column (AUC
     0.926, recall 81.2%, macro FP 23.6%, and it rescues DALL-E 3: 68% recall) yet
     **worst on the gate: 96.8% FP on NIST2016.** Content-aligned training did not close
     the source gap either. Three training philosophies, one shared failure.
   - ~~CLIP linear probe~~ — dropped; a third frozen model cannot answer a question two
     have already answered. Both external score JSONLs are cached, so all further
     calibration experiments on them cost seconds.
2. ~~Stay-Positive constraint on our tile ResNet-18~~ — **mooted by E22**: under any
   source-robust threshold our model keeps 1.2% recall; its scores are not
   source-invariant, and a last-layer constraint cannot repair that. Recorded, not run.
3. **Source-robust decision rule** — ✅ **measured 2026-08-19, see E22.** Two deployable
   operating points now exist: CF ViT-S + worst-source calibration passes the gate on
   *unseen* pipelines (worst held-out FP 6.6%, 28.4% recall); B-Free's abstention band
   reaches **65% recall at ≤8% FP on all eleven pipelines** with 21% abstention, when each
   pipeline family contributes ~100 calibration images (threshold-only, no retraining).
   Remaining sub-items: grow the real-pipeline calibration library (Phase 4.1 personal
   photos as a fresh unseen-pipeline test), and a midjourney diagnostic (40% of it is
   actively called "real" by the band).
4. **Data work** — compression augmentation (JPEG q30–q95; the E12 debt), a compressed
   copy of every test set (q50 + 75% resize, the literature's social-media standard), and
   optionally a B-Free-style content-aligned pool built from our own real photographs.
   E22's H1 adds urgency: the calibration domain sits at 0.16 B/px, the transfer domain at
   1.1–1.9 — the compression axis is now measurably entangled with the decision layer.
5. **Report + demo polish** — the report's arc is now complete (data → representation →
   decision, each measured); the demo could honestly ship the B-Free band as its verdict
   layer, licence permitting, or stay research-only.

Survey sources: [out-of-the-box benchmark](https://arxiv.org/html/2602.07814v1) ·
[Stay-Positive](https://arxiv.org/html/2502.07778v1) ·
[B-Free](https://grip-unina.github.io/B-Free/) ·
[Community Forensics (CVPR 2025)](https://arxiv.org/abs/2411.04125) ·
[conformal abstention](https://arxiv.org/pdf/2502.07255) ·
[TGIF2](https://www.emergentmind.com/papers/2603.28613) ·
[NTIRE 2026 challenge](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gushchin_NTIRE_2026_Challenge_on_Robust_AI-Generated_Image_Detection_in_the_CVPRW_2026_paper.pdf)

## Standing rules (unchanged)

- Headline metric = AI recall at a fixed FP budget on **unseen real sources**; AUC is
  reported alongside, never alone.
- ≥3 seeds on anything reported. Audit every dataset before use (`ml/tools/audit_datasets.py`).
- Thresholds are chosen on calibration halves and measured on untouched halves — always.

## Repo conventions after the 2026-08-18 tidy-up

- `ml/experiments/` holds only runnable protocol scripts (e20, e21); finished evidence
  scripts are frozen in `ml/experiments/archive/`.
- `ml/src/pixelproof/archive/` holds retired modules (E2–E4 analysis, ELA, DINOv2
  extraction). Nothing in the live path imports them.
- `ml/artifacts/archive/` (not committed) holds superseded artifacts, including the
  poisoned `*.BOZUK_etiket.bak` evidence files. Nothing is deleted.
