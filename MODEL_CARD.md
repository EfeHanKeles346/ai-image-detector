# PixelProof E20 project model — model card

**Version:** E20-v2, deployed training seed 2024

**Status:** runnable research model; not an authenticity authority

**Last verified:** 2026-09-09 (runtime artifact integrity; no new E20 accuracy evaluation)

## Successor candidate decisions (2026-08-26)

### E60 offline champion-preserving correction (2026-09-10)

E43-S remains the frozen research reference, separate from served E20. One zero-initialized
bounded logit correction was fitted on all11,630 admitted local TRAIN parents without downloads.
On the same consumed E49 population, AI recall stays94.3%/95.5% (original/Q75), while REAL
false-AI only falls39.1->39.0% /49.0->48.8%. No new E49 AI errors, but the REAL improvement
intervals include zero and absolute/selective acceptance gates still fail. E60 is insufficient
and is not promoted. Small TRAIN AI losses and the absence of a fresh balanced group-disjoint
development population prevent a universal retention or final-quality claim. Reference and
saved-head replay checks are exact; these are engineering checks, not accuracy certification.
See `evidence/e60_regression.json`, `ml/EXPERIMENTS.md` E60 and current `PLAN.md`.

### E51/E53 offline successor decisions (2026-09-09)

Neither replaces the served contract. E51-A improves authentic-photo safety on consumed E49 but
loses AI recall: original/Q75 AI 77.50%/82.10%, REAL false-AI 15.90%/30.70%. On a different,
newly evaluated IEEE/Datapoint DEV population, BA is 92.85%/92.08%, REAL false-AI 1.67%/1.97%,
AI recall 87.38%/86.13%. That DEV is now consumed; hidden device ids and publisher resizing limit
native-camera claims. Results from these different populations must not be combined into one score.

E53 compares twelve fixed configurations in two protocols on the same 5,978 TRAIN-derived
source-held-out parents (72 fold fits), including 5,652 audited original-based replay additions.
None satisfies both REAL improvement and no-AI-loss guards. Native replay improves AI detection
but increases REAL errors in a difficult held-out publisher; simpler features lose particular AI
sources. These small refitted fold models are not new measurements of the full E43/E51/served
model. No E52 independent-final pass, universal-detector claim or serving promotion follows.

Engineering checks establish exact score parity for crop dedup on 11,956 TRAIN views and exact
fast-training-weight equivalence. A last-two-block resource probe confirms local MPS feasibility
but saves no adapted model and supplies no accuracy evidence. See `evidence/e53_coverage_result.json`,
`evidence/e53_source_held_out_expanded_controls_result.json` and `PLAN.md` for the next gated study.

### E43-S comprehensive-final decision (2026-09-04)

E43-S is a frozen DINOv2-S representation plus project-fitted binary head. It passed earlier
development and a narrow modern-generator diagnostic, but it did **not** earn Module-1-v1 status in
the one-shot E49-C comprehensive final. The final contained 2,000 independent parents: 1,000
Wikimedia Commons camera originals across ten devices and 1,000 AI images across five current
OpenFake families plus StyleGAN2; every parent also had a fixed social-Q75 child.

E43-S scored all 4,000 observations once and passed 11/20 preregistered checks. AI recall transferred
well (94.30% originals, 95.50% Q75; worst family above 91%), but authentic-camera safety failed:
REAL false-AI was 39.10%/49.00%, worst-device false-AI 71%/84%, and balanced accuracy 77.60%/73.25%.
Original/Q75 AUC was 90.24%/86.89%. This shows the candidate detects current AI strongly but its
threshold and representation do not generalize safely to new camera pipelines, especially after
recompression. E49-C is consumed and cannot tune this model. E43-S remains research-only and does
not replace the canonical served contract. Evidence: `evidence/e49_final_result.json`.

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

A second frozen-head screen, R1a, replaced DINOv2-S embeddings with the pinned forensic CF-ViT CLS
representation and passed internal CALIBRATION even more strongly (AUC 0.9982, AI recall 99.91%).
It still mislabeled 154/210 owner stills: 26.67% REAL recall. R1a is likewise research-only via
`pixelproof-predict-e32-cf` and rejected from serving. The result rules out an encoder-only fix;
the next correction must add licensed authentic camera-source coverage and source-held-out gates.

R1b then added 3,994 audited CSAFE iPhone 14 natural photos while preserving every earlier role,
input transform and AI row. Its selected CF-ViT head passed internal CALIBRATION at AUC 0.9981 and
99.82% current-AI macro recall, but failed the frozen external authentic gate: 249/960 IPN-NFID
images were false positives (40.0% worst-device FP) and 144/210 owner-gallery stills were false
positives (31.43% REAL recall). R1b is excluded from every official decision path; the local demo
may show its frozen score only as a non-voting `research_only` card. Neither external population
changed its threshold or weights; no LOCKED AI set was opened. Evidence:
`evidence/e32_r1b_external_development.json`.

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


### E61 engineering safeguard (2026-09-10)

The strict TRAIN replay gate now rejects any newly missed AI view that frozen E43
previously caught. It rejects E60 on5 known TRAIN parent losses; E49 pooled recall
remaining unchanged does not override that diagnosis. No new model was trained or
promoted. Passing the software gate is necessary for future candidate development,
not independent evidence of REAL false-positive reduction or unseen-AI preservation.
See `evidence/e61_replay_gate.json` and the dated image-reference literature review.


### E62–E64 constrained corrections (2026-09-10): rejected, no promotion

E62 PCA64 and E63 RBF64 retained all previously caught admitted TRAIN AI views. On the
same consumed E49 benchmark, both lower REAL FPR39.1->35.8%; Q7549.0->46.8/46.7%.
However AI recall94.3->93.8% and95.5->95.1/95.3%, including supported-generator losses.
Both retain only11/20 absolute/selective gates and fail the extra E43-retention guards.
Finite TRAIN preservation therefore did not establish external AI preservation.

E64 permits correct REAL confidence to move inside its original decision margin. TRAIN
REAL FPR11.8977/11.3291%, zero newly missed AI, but all3 TRAIN conditions fail its<=10%
pre-test requirement; no E49 scoring. No candidate is accepted. Original E43 hash/cuts
and serving are unchanged; no independent final or new model-quality claim.49 focused
tests pass. Full preregistration, hashes, paired counts and intervals are in EXPERIMENTS.


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


E71 rejected after consumed DEV: REAL FPR34.375% original /38.125%Q75, only12/20 gates.
Aggregate AI recall unchanged but2/1 newly missed AI views; no E49/serving promotion.
Full frozen scores SHA`971b6e130b51f8ef741e4a194782e4b9c02e6ba18b68582db82f8eea07be8db0`; compact `evidence/e71_development.json`.


E73 full-confidence candidate rejected at TRAIN: REAL FPR14.57%/15.84%/13.35%,
zero new AI/REAL errors, no DEV/E49 allowed. Artifact SHA`147551945c7fdb9843cd93c4262a8b07381f15f738b897412fb9cfd2354f88b9`.


E74 rejected at TRAIN: REAL FPR13.29%/14.36%/12.14%, no new AI/REAL errors.
No DEV/E49 scoring or promotion; candidate SHA`cadf88f4fc3e3030495bc9cbdc0c74a6d2ffb205091fcebaac063dbf731eb05a`.


E76 camera-expanded candidate rejected at TRAIN. Old REAL FPR13.28%/14.37%/12.25%;
expanded12.64%/13.65%/11.62%, new-MIDD3.91%/3.72%/2.94%. No new AI/REAL errors.
No DEV/E49 or promotion. Artifact SHA`dd6fd99607f674cafa3a4f9982002ce2c688d564ebd4b94bcd57e4ac3106a117`.


E77 is rejected at TRAIN: old-REAL FPR10.96%/12.38%/10.39%, all above10%. No AI
confidence loss beyond solver tolerance, no newly missed AI or newly wrong REAL.
No DEV/E49 or promotion. Receipt `evidence/e77_fit.json`; failed candidate retained externally.
