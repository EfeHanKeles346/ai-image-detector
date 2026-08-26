# PixelProof E20 project model — model card

**Version:** E20-v2, deployed training seed 2024

**Status:** runnable research model; not an authenticity authority

**Last verified:** 2026-08-24

## Successor candidate decisions (2026-08-26)

### E32 runnable candidate decision (2026-08-26)

E32 produced a technically runnable frozen-DINOv2-S candidate after rebuilding the pool around
22,688 balanced modern-AI/authentic parents. Its 4,534-row group-held-out, source-stratified
CALIBRATION result passed the preregistered screen (AUC 0.9964, AI recall 99.07%, REAL recall
90.14%). However, the untouched threshold then mislabeled 159/210 already-consumed owner-gallery
authentic stills as AI: only 24.29% REAL recall. This proves the internal split still rewards
source/pipeline shortcuts.

E32 is therefore available only through the research CLI `pixelproof-predict-e32`; it does not
replace the canonical E20 API/web contract and is not an authenticity authority. Its artifact SHA
is `7f170340...a85e`; evidence is `evidence/e32_r0_dinov2s.json` plus
`evidence/e32_owner_gallery_smoke.json`. The owner gallery did not select a new threshold or refit
the head.

### E31 candidate decision

E31 did **not** replace this served E20 contract. Its single-DINOv2 research candidate is locally
runnable and hash-verified (`99901219...4d860`), but it failed the pre-registered independent E30
DEVELOPMENT gate: AUC 0.385, 80.67% current-AI macro recall, **83.63% macro authentic false
positives** and 100% worst authentic group FP. Three of 900 transport views were tile-ineligible.

Earlier E31 CALIBRATION performance—0.966 AUC, 90.72% current-generator macro recall, 4.67% macro
and 6.70% worst real FP—therefore does not transfer to independent MLLM-matched real content. A
diagnostic threshold meeting the real budgets reduces AI macro recall to 0.33%, so recalibration
cannot rescue the candidate. Qwen LOCKED FINAL was not opened. `pixelproof.e31_candidate` and
`ml/experiments/e31_score_folder.py` exist only for labelled research/error analysis; the FastAPI,
CLI and web model remain E20. Evidence: `evidence/e31_b5_development.json`.

## What this model is

The canonical project-owned model answers a narrow question: *does this image contain a signal
that resembles the AI-generated class learned by E20?* It is a ResNet-18 with ImageNet-pretrained
features and a one-logit binary head, trained by this project on native 128×128 image tiles.
Label `1` means AI-generated and label `0` means authentic.

The model is usable through `pixelproof-predict`, `pixelproof-evaluate-project`, FastAPI and the
model-first Turkish web UI. All four paths use the same verified loader and scorer. A positive
result is an experimental AI-oriented signal. A negative result is **uncertain**, not “real”.

## Artifact identity

| Field | Frozen value |
|---|---|
| Registry id | `e20-tile-resnet18-seed2024` |
| File | `ml/artifacts/tile_resnet18_seed2024.pt` |
| SHA-256 | `b9f39eda10ba3de54b706d6448b67d93ce8e4c7bae97a685f3c1b57ebfd65adf` |
| Architecture | ResNet-18, ImageNet initialization, one binary logit |
| Training seed | 2024 |
| Best epoch | 6 of an 8-epoch ceiling |
| Stored validation AUC | 0.9096272753 |
| Size | 44,789,451 bytes |

The runtime verifies the registry hash before deserialization and then rejects incompatible
architecture, state dictionary, preprocessing, aggregation, threshold or split metadata.
Weights are owner-supplied and intentionally not committed.

## Training data

E20 used one seeded, texture-qualified native tile per pool image. The cached training tensor held
48,037 tiles: 24,011 authentic and 24,026 AI-generated. These counts and source names are stored
inside the checkpoint:

| Source | Tiles |
|---|---:|
| `communityforensics` | 5,558 |
| `ai_vs_real_balanced` | 15,515 |
| `genimage` | 4,034 |
| `aigc_benchmark` | 14,768 |
| `ai_vs_real_200k` | 8,162 |
| **Total** | **48,037** |

The pool includes sources that are safe specifically for fixed-size tile training; that does not
make their whole-image metadata distributions clean. Dataset purposes, limitations and acquisition
boundaries are in `DATASETS.md`. Source licences do not become a licence for this repository or
for redistribution of the checkpoint; see `LICENSE.md` and `ml/ARTIFACTS.md`.

## Inference contract

1. Decode only JPG, PNG or WEBP under the bounded serving policy; apply EXIF orientation and
   composite transparency onto white.
2. Select native 128 px tiles with texture floor 0.04, preserving image scale. Pad a smaller image
   rather than silently changing the model contract.
3. Evaluate at most 256 tiles, once each, with ImageNet normalization.
4. Average the three highest tile scores (`top3`).
5. Compare with the seed-2024 calibration-only threshold 0.9894907077.

The tile map is a map of detector scores. It is not a validated manipulation-localisation mask.

## Measured performance

The headline numbers are population mean ± population standard deviation over seeds 42, 1337 and
2024 under E20 protocol v2. Aggregation and threshold selection use calibration halves; evaluation
halves do not choose the rule or threshold.

| Evaluation metric | Three-seed result |
|---|---:|
| Defactify evaluation ROC-AUC | 0.751 ± 0.033 |
| AI recall at stored per-seed threshold | 49.9% ± 6.1% |
| Defactify authentic false-positive rate | 8.7% ± 2.2% |
| Ten-forensic-source macro false-positive rate | 42.7% ± 1.0% |
| **Worst-source authentic false-positive rate** | **86.2% ± 3.1%** |

Each run evaluated 150 Defactify authentic images, 750 Defactify AI images and 1,776 authentic
images across ten forensic sources. The deployed seed-2024 run itself measured AUC 0.720, recall
48.1%, Defactify FP 11.3%, forensic macro FP 43.3% and worst-source FP 83.2%.

The distinction matters: the model can rank some generator families while failing catastrophically
on an unseen authentic pipeline. The four-image M4 operational smoke repeated this behavior: both
authentic upstream B-Free demo examples crossed the stored threshold.

## Appropriate uses

- Demonstrating a complete project-owned training-to-inference pipeline.
- Reproducible research comparisons on labelled `real/` and `ai/` folders.
- Inspecting raw model scores and studying source/pipeline shift.
- Teaching why ranking, calibration and cross-source specificity are different claims.

## Prohibited or unsupported uses

- Certifying that an image is authentic or using a negative score as proof of reality.
- Automated moderation, disciplinary, legal, hiring, insurance or other high-impact decisions.
- Claiming universal detection of unseen generators, cameras, editors or compression pipelines.
- Treating tile scores as proof of where an image was edited.
- Retraining or redistributing data/weights without checking each source's terms.

## Known limitations

- **Authentic source shift:** worst-source false positives remain the dominant failure.
- **Small/compressed generators:** DALL-E 3 was consistently weak in E20; the tile route can lose
  the texture evidence it needs.
- **Generator drift:** the model predates future generator families and does not establish
  out-of-collection performance on them.
- **Threshold transfer:** a threshold calibrated on one population does not automatically transfer
  to another. User-folder evaluation reports the stored threshold; it never refits on evaluation.
- **No authenticity class guarantee:** “below threshold” means only “this detector did not trigger”.

## Reproduction and traceability

```bash
# Verify and demonstrate the exact checkpoint
./tools/pixelproof-demo check
./tools/pixelproof-demo start

# Evaluate labelled folders without fitting a new threshold
cd ml
.venv/bin/pixelproof-evaluate-project \
  --real /path/to/real --ai /path/to/ai --output artifacts/my-evaluation
```

Scientific provenance: `ml/EXPERIMENTS.md` E20 and its three-seed addendum; raw local result
`ml/artifacts/e20/results_3seed.json`. Engineering provenance: `PLAN.md` M0–M6, append-only
`HISTORY.md`, artifact manifest, and `PRESENTATION_EVIDENCE.md`.
