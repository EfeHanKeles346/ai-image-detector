# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(append-only project archive) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only scientific
log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

## Documentation and acquisition rule — home-network update (2026-09-14)

The user is now at home and explicitly permits dataset downloads. This supersedes the
mobile-data acquisition prohibition below. Use downloads that fill identified coverage
or evaluation gaps; verify provenance, licence, identities and disk reserve before
admission. Keep experimental fitting offline and model/environment versions pinned.

Ongoing English records: PLAN.md for next actions, HISTORY.md for engineering decisions,
ml/EXPERIMENTS.md for protocols/results, and, at the user's explicit request, DATASETS.md
for acquisitions. Record source/revision, reason, intended role, date, licence, requested
versus completed files/bytes, integrity, local storage, failures and next admission step.
Historical entries remain historical. No raw data, private gallery information or model
weights in Git; small code/evidence commits and pushes remain authorized, never forced.

## Research and execution programme for both models

This programme supersedes the historical rule that Model 2 work must wait for an E49
pass. The current instruction authorizes research, automatic acquisition and implementation
for both models. It does not waive data-role, retention or evidence controls. The deliverable
remains a reproducible student research system; no commercial or universal certification is
claimed. Keep all four English records current. No paid API generation, account creation,
external messaging or interactive login is needed for the steps selected here.

### Scope and present evidence

Model 1 identifies evidence of full-image AI generation. Model 2 localizes AI-assisted local
edits. A fully regenerated image can carry synthetic traces outside a nominal editing mask;
a conventional paste from another camera is not necessarily AI generation. These cases need
separate labels and tests. Neither a low Model 1 score nor an empty Model 2 heatmap certifies
that a photograph is authentic. Combining the modules must not silently change either model's
fixed decision rule. A percentage is currently a raw model score, not a calibrated probability.

E92 is the running local demo. E102 reduced consumed social-Q75 REAL errors from 14/160 to
12/160 while retaining 159/160 AI, but both E102 and E103 miss an original AI caught by E43.
E103's near-perfect TRAIN results did not improve binary consumed-DEV results. The original
160 REAL observations represent just 10 dependent SIDD scenes, and the AI sample covers two
seen families. The owner gallery is also consumed development: it cannot be used to train a
fix and then to prove that fix independently. The 20 historical numeric checks are 10 metrics
in two conditions, not 20 independent datasets or complete acceptance.

E104 established a representation property: mean/std pooling erases which of three crop
vectors came from center-context versus texture. The named global crop is itself a resized
center crop. This motivates a complementary feature experiment but does not prove why any
particular image is misclassified. Existing CAL populations are historically consumed, with
incomplete upstream independence in some sources. Exact hashes and source names alone do
not establish scene, prompt, photographer or generator-family independence.

Model 2 has only a historical diagnostic signal: E17 used at most 36 tiles, dropped masks
without a tile at least half edited, and selected a threshold from the true mask fraction.
That last operation is an oracle. Its reported IoU cannot serve as a deployable accuracy
claim. The inventory contains 512 CocoGlide image/mask pairs, with the first 120 potentially
exposed. Authentic decoding, cross-role near duplicates and ancestry require further audit.

### What current research supports

| Work and status | Mechanism and reported difficulty | Adaptation and boundary here |
| --- | --- | --- |
| [UniversalFakeDetect, CVPR2023, official code](https://github.com/WisconsinAIVision/UniversalFakeDetect) | A pretrained CLIP representation provides a useful transferable detector baseline. | Already available locally; retain it as a frozen comparator. Its name does not establish success on all future generators or local edits. |
| [Community Forensics, CVPR2025](https://jespark.net/projects/2024/community_forensics/) | Large generator diversity is a central training resource. | Local Small data has 300 generator identities, not the full release's breadth. Retain model-family labels and match REAL content/encoding; the local native-resolution shortcut was previously measured. |
| [B-Free, CVPR2025](https://github.com/grip-unina/B-Free) | Semantically paired real/reconstructed images reduce content bias; large native crops and processing augmentation support robustness. | Apply identical processing to both classes and consider context plus native-detail features. Its COCO training release overlaps protected project ancestry, so it is not an automatic new TRAIN source. |
| [DEAR, ICML2026](https://github.com/dahyedahye/dear) | Inpainted masks diagnose channel responses; bilateral feature pruning/refinement targets fragile processing cues. | DEAR-r is already a frozen feature expert here. Do not claim our pooled crops reproduce its native inference or that imported weights eliminate dataset bias. Mask-based diagnostics are possible only on eligible TRAIN. |
| [GAPL, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Qin_Scaling_Up_AI-Generated_Image_Detection_with_Generator-Aware_Prototypes_CVPR_2026_paper.pdf) | More generator data can eventually create conflicting embeddings; prototypes and two-stage LoRA address representation limitations. | Preserve generator grouping; prototype or low-rank adaptation is a later isolated experiment if the cheaper crop-role extension fails. Do not add data volume without checking the bottleneck. |
| [NTIRE2026 robust detection report](https://arxiv.org/html/2604.11487v1) | The challenge combines 42 generators and 36 distortions; strong systems use matched corruptions and complementary high-resolution/robustness experts. Some reported systems require multi-GPU or very large memory resources. | Reproduce the evaluation ideas and bounded augmentation design on this 18GB Mac. Do not promise equivalent training scale or compare our consumed accuracy with their ranking metric. |
| [Out-of-box benchmark, February2026 preprint](https://arxiv.org/abs/2602.07814) | Detector rankings vary substantially across datasets; its evaluation finds no universally winning detector. | Use a common locked benchmark and exact pretrained artifacts for comparisons. This study is evidence of evaluation dependence, not proof that every detector must always fail. |
| [TruFor, CVPR2023, official code](https://github.com/grip-unina/TruFor) | RGB semantics and Noiseprint++ traces are fused for localization and reliability estimation. | Start with a small dense/token or residual baseline plus authentic controls. A visually plausible heatmap is insufficient; residual maps must beat simple spatial baselines. |
| [TGIF2, Journal on Information Security2026](https://arxiv.org/html/2603.28613v1) | Spliced and fully regenerated edits behave differently. Random-mask tests expose object-boundary bias; fine-tuning gains can remain generator-specific. | Separate SP/FR and semantic/random masks, include authentic controls and hold out source parents and generator families. Super-resolution requires its own stress column. |
| [DiffSeg30k, November2025 preprint](https://arxiv.org/abs/2511.19111) | Eight editing models, sequential edits and pixel labels support localization; processing robustness remains difficult. | Acquire a bounded author-hosted TRAIN image/mask subset for provenance/schema audit. Multi-class values 1–8 are edited pixels; a binary >127 mask conversion would be wrong. COCO ancestry prevents an automatic independent-final claim. |

These are primary research references, not a ranking of commercial APIs. Marketing claims
without a public paired benchmark, versioned artifact and protocol are not evidence of
universal performance. The practical path is to reuse verified encoders, preserve data
lineage, test a single mechanism at a time, and spend fresh evaluation only after selection.

### How broader reliability will be demonstrated

The following are project research targets, not an industry standard and not replacements
for historical contracts. Acceptance needs both non-regression and new independent evidence.

1. Freeze the candidate, preprocessing, input admission, decision/abstention rules and all
   evaluated conditions before opening a new test. Publish artifact and manifest hashes.
   TRAIN fits weights and feature transforms; grouped CAL selects thresholds/calibration;
   consumed DEV remains diagnostic; an unexposed benchmark is used once. If its result
   motivates a change it becomes consumed and cannot prove the successor independently.
2. Define a coverage matrix before sampling: at least five genuinely held-out AI families,
   including modern diffusion/flow and another architecture where available; multiple
   independent REAL acquisition sources; camera/scene/photographer groups, low-light/HDR,
   portrait/landscape, screenshots/documents as explicit scope controls. Versions within one
   family are not independent families. Unresolved upstream membership must be disclosed.
3. Evaluate native originals, the frozen social-Q75 transform, and additional fixed JPEG,
   resize, blur/noise and compound-processing stresses applied identically to both classes.
   Keep processing descendants in their original parent group. Hold out some processing
   combinations for final stress rather than tuning all transforms on final failures.
4. Preserve all E43/E92 and accepted-predecessor AI detections and previously correct REAL
   decisions on the existing paired regression. No improved pooled count may offset a new
   lost AI. Report source transitions, failure/abstention coverage and both unconditional
   and accepted-case errors. Do not lower the guard to make the known DEV miss disappear.
5. Broader target: one-sided 95% upper REAL false-alert rate <=1% on declared clean groups,
   <=5% on declared stress groups; lower AI recall >=95% pooled and >=90% per held-out
   family. Count meaningful independent groups and adjust for multiple predeclared claims.
   These ambitious targets may fail and cannot be inferred from current point estimates.
   A scene-level 'any false alert' bound is a different, conservative endpoint from a
   per-photo rate; label it accordingly. Unknown group independence denies certification.
6. Use [exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)
   only for justified independent Bernoulli units. With zero failures, 299 independent
   trials are needed for a single 95% upper bound <=1%; ten scenes cannot establish it.
   Use parent/scene-cluster bootstrap for paired continuous/model comparisons, never tiles
   or repeated encodings as independent samples. Predeclare multiplicity and sample size.
7. Calibration is a separate fitted component: reliability bins, Brier/log loss, ECE and
   prevalence sensitivity on grouped CAL and a distinct evaluation. A monotone calibration
   cannot repair ranking errors. [Research on risk control under covariate shift](https://proceedings.mlr.press/v266/almeida25a.html)
   needs additional assumptions/weighting; ordinary conformal or calibration guarantees
   do not automatically survive unknown shifts. Keep raw-score UI until validated.
8. Include reproducibility and operational evidence: hashes, dependency lock, deterministic
   transform tests, CPU/MPS batch parity, native input limits, corrupt-file behavior,
   latency/memory and end-to-end coverage. A research model may pass statistical tests but
   still fail licensing, latency or product-scope requirements. Record those separately.

For Model 2, require full spatial coverage and image-macro pixel AUC/AP, fixed-threshold
F1/IoU by mask area, authentic false-localized area and image detection. Pixel labels are
not independent sample units. Compare center, constant and deterministic random spatial
baselines under the same fixed area/rule; no truth-area-selected prediction threshold.
Prospective MVP targets: macro pixel AUC >=.75 and AP above prevalence in each supported
source/generator stratum; fixed-threshold macro IoU improvement >=.10 over the strongest
predeclared simple baseline, supported by parent bootstrap; authentic flagged-area mean
<=1% and no more than5% of authentic parents exceeding5% flagged area. These are proposed
project gates requiring a separately frozen protocol, not achieved results. Report tiny
masks and FR cases even when they fail; never silently exclude them from the headline.

### Implementation sequence and exit conditions

| Stage | Action | Completion evidence and fail behavior |
| --- | --- | --- |
| P0 | Implement finite-sample proof checks and audit today's evidence. | Test exact bounds, multiplicity and consumed/group-unknown rejection. Report current evidence as insufficient; do not open protected images. |
| P1 / E105 | Acquire 512 hash-selected DiffSeg30k TRAIN images and corresponding masks from a pinned author revision. | Size/CRC/SHA and immutable receipts, native mask schema/geometry audit, explicit quarantine. No full 21GB archive or validation pixels required. |
| P2 / E106 | Audit complete old raw CLIP TRAIN inventory for the context extension. | Verify every parent/binding/aggregate and source; save complete availability report. No partial-population fit. Register missing-view cost probe separately. |
| P3 / E107 | Repair Model 2 evaluator and replay the already exposed first120 CocoGlide parents plus authentic counterparts. | Complete half-stride128/64 edge coverage, all masks retained, no oracle threshold, fixed baselines and image-level resampling. Diagnostic only; no TRAIN/CAL role change. |
| P4 | Decode and fingerprint all Model2 candidate parents and bind masks/derivatives to whole ancestry groups. | Check protected Model1/Model2 exact/RGB/perceptual references; retain unresolved COCO parent lineage in quarantine. New publisher alone is not independence. |
| P5 | Complete context raw features and register one complementary Model1 representation/head. | Exact old-feature replay, both labels/four conditions, fixed optimizer and runtime gates, then staged consumed comparison. No new DEV score until preceding gates pass. |
| P6 | Train one small dense Model2 head only on admitted parent-grouped TRAIN, with CAL-frozen threshold. | DINO/noise baseline first; registered augmentation, fixed budget, no architecture sweep on final masks. Keep module outputs separate. |
| P7 | Admit a genuinely unexposed source/generator benchmark and evaluate a locked qualified candidate once. | Full coverage matrix, paired baselines, uncertainty bounds and failure disclosures. If any prerequisite fails, record incomplete proof instead of 'universal'. |

Run network transfers with bounded workers, immutable selections, resumable receipts, timeouts
and disk reserve. Never require a browser login or a phone confirmation while unattended.
Use only public authorized endpoints; stop access-denied sources and record alternatives.
Keep fitting offline and avoid concurrent GPU encoder jobs. No always-on automation is created.
Completed bounded stages are reported honestly; a long-term research programme is not marked
complete merely because a plan, a download or a synthetic test passes.

### Acquisition decisions

TGIF2 is highly relevant but the official public-share DAV endpoints returned401 during this
session, including the documented public-share request form. No image bytes were acquired and
no access controls were bypassed. Keep it as a future source pending a working public download.
DiffSeg30k is an author-linked, publicly downloadable alternative with masks and multiple edit
generators. Its Apache-2.0 dataset card does not erase underlying COCO or model-specific terms;
keep research-only quarantine until ancestry/rights/role checks finish. NTIRE TRAIN is public,
but its inspected metadata lacks an explicit dataset licence and its labels do not expose all
needed parent/generator provenance; research its terms rather than treating it as fresh final.
DATASETS.md records exact revisions, selected counts, transferred bytes and all later outcomes.


## Current E98–E102 checkpoint — data admitted, candidate not accepted (2026-09-14)

E98 role/lineage audit completed without TRAIN/CAL overlap in the mapped identities.
E99 downloaded 256 originals / 3,323,883,735 bytes from four additional MIDD sensors.
E100 admitted all 256 research TRAIN parents after 152,124-reference checks, including
all 206 unique owner-gallery identities; no detected overlaps, failures or removals.
E101 completed all 1,024 frozen training views with exact prior parity. DATASETS.md
records source, rationale, licence, exact totals, storage and admission receipts.

E102 completed one registered correction fit on 12,525 TRAIN parents / 50,100 views,
passing its TRAIN and runtime gates. In consumed E66 DEV it passes all 20 numeric gates:
original REAL false alerts stay 0/160, social-Q75 false alerts improve 14/160 to 12/160,
and AI recall stays 159/160 in both conditions, with no lost E92-caught AI. However,
one E43-caught original AI remains missed, so the full DEV screen FAILS. No gallery,
E49 or independent-final scoring follows this failure. No candidate promotion.
The local E92 API has been restored and remains the demo model.

Both E92 and E102 already classify all 256 new MIDD TRAIN photos below the AI threshold
in all four conditions. This acquisition expands sensor coverage but does not itself
establish an improvement on the owner's gallery or an unseen publisher/generator.

## E103/E104 completed — next: complete crop-role features (2026-09-14)

E103 passes TRAIN/runtime: zero missed AI in all four TRAIN conditions; REAL errors
2/0/0/0 (clean/assigned/Q75/social). On consumed DEV the binary result is unchanged
from E102: REAL errors 0/160 and 12/160; AI caught 159/160 in each condition. All 20
numeric gates pass but the E43 retention deficit persists. No gallery/E49 scoring,
no promotion. E92 remains served. Do not claim TRAIN perfection as external accuracy.

E104 audited 30 hash-selected TRAIN parents / 90 existing views: exact saved-feature
replay, zero pooled change on center/local swap, nonzero separate role-coordinate
change in every view. The helper preserves resized-center context and its signed
contrast to local texture; it is not fitted or used by the demo. Historical "global"
is a center crop, not a complete-frame input. No causal error claim follows.

Next inventory all eligible raw CLIP chunks before extraction. The old E71 lineage
records 11,630 parents x 3 conditions = 34,890 raw views; E104 checked only 30 parents.
If the full old inventory verifies, complete the remaining 15,210 views to retain the
same 12,525-parent/four-condition population: 11,630 old social views plus all four
views of 511 first-MIDD, 128 SID and 256 new-MIDD parents. Do not drop sources, labels
or conditions simply because pooled caches are easier to access. Register exact crop/
encoder parity, source hashes, resumable chunks and a bounded cost probe before new
extraction. Only complete audited features permit one new TRAIN representation fit.
This crop-role feature is separate from any future uncropped-frame transform, which
would need its own protocol. Keep all AI/REAL retention and consumed-DEV/final rules.
E104 was audit-only; no extraction or new fit is currently running. No downloads needed
for this next cache extension if original bodies and existing raw chunks verify.

## Next controlled model cycle after the E102 result

Preserve E102 as a failed full-acceptance candidate and keep its locked evidence. Do not
relax the E43 retention condition, tune on the missed DEV item, adjust raw percentages,
or open later evaluation stages to find a favourable result. The two social-Q75 rescues
are useful consumed-DEV evidence, not an independent estimate of generalization.

The next design question is whether a complementary global-context/processing-stability
reviewer can distinguish acquisition/transport artifacts from generation evidence. First
inventory eligible TRAIN features and source/scene/parent groups; register one hypothesis,
matched processing on both labels, objective and AI-retention constraints before fitting.
Keep gallery, consumed DEV and protected reserves out of TRAIN and calibration. Check
TRAIN processing failures before acquiring more of a sensor source already handled by
E92. More MIDD volume alone is not established as a solution to the observed errors.

E98 mapped all 6,619 E32-derived TRAIN rows to original TRAIN and found no mapped role
overlap. Existing R1b/C3 CAL populations remain historically consumed and overlapping;
the audit does not make them fresh validation. Any probability calibration needs a
separately justified, grouped population and explicit evaluation protocol. Until then
retain raw-score language and guarded uncertain results. New source acquisitions need
provenance/licence and role registration, protected-overlap admission, and DATASETS.md
receipts. No subsequent fit or new acquisition is registered by this planning note.

## E97 — honest score display completed; learned reviewer remains planned (2026-09-14)

User requests percentages and a complementary check for uncertain gallery photos.
Expose the unchanged E92 original and social-Q75 raw scores multiplied by100, explicitly
not calibrated AI probabilities or correctness confidence. Show the existing7.940196%
AI-alert threshold; do not remap it to50%, average views, add arbitrary±5 points, or
force abstentions into binary claims. Small/rejected/unavailable inputs get no score.
Retain all original alerts, guard warnings, weights and cuts. Keep frozen E93/E95
helpers intact; add a presentation wrapper and versioned API/frontend validation.

Completed read-only diagnosis:36 uncertain gallery files partition into12 social AI
crossings,17 reference-only vetoes and7 borderline E92 pairs. These are guard branches,
not proven causal image features. All530 paired displays (210 gallery+320 E66 parents)
retain exact outcomes and raw scores. Seven actual scored HTTP cases have exact score
parity; an eighth tiny input returns null. All8 payloads pass the frontend contract.
909 Python tests,13 Node tests, lint/typecheck and build pass. No fitting, threshold
sweep, gallery CAL/TRAIN use, new download or final-set access. Existing9 original
false AI gallery indications and36 uncertain decisions remain unresolved.

The next research step is a role/overlap audit of existing TRAIN and CAL caches before
registering one complementary reviewer. Candidate evidence is full-image context plus
agreement/processing stability, with the hypothesis fixed before evaluating errors.
Group camera/scene/parent/generator so derivatives never cross roles; do not assume an
existing cache is eligible. E95, E66 and consumed E65 remain regression/diagnostics only.
If no uncontaminated CAL population is available, retain raw-score language and defer
probability fitting. Evaluate fewer abstentions at a fixed error budget, accepted-case
errors, REAL false alerts and AI retention per source; reducing abstentions alone is
not success. Measure probability calibration (reliability bins, Brier score/log loss)
separately and report its source/prevalence scope. A subsequent learned reviewer
requires audited TRAIN-only complementary features and separately grouped CAL; calibration and fewer abstentions must be evaluated separately
from classification accuracy. Preserve the732-view regression guard, source groups,
original/Q75 views and a genuinely unexposed evaluation before broader claims. Ordinary
score calibration cannot guarantee OOD correctness. Research: Guo et al. (ICML2017,
https://proceedings.mlr.press/v70/guo17a.html) and ReSIDe (May2026 preprint,
https://arxiv.org/abs/2605.08574). No reviewer has yet been trained or validated here.

## Current gallery checkpoint — E95/E96 completed (2026-09-14)

The owner gallery remains consumed DEVELOPMENT, excluded from TRAIN/CAL. E92 measured
9 original and18 Q75 false AI indications among210 files (206 unique parents); display
165 no-clear/36 uncertain/9 AI,3 positive warnings. E43 previously15/28, but E92 creates
7 new original/2 new social errors while rescuing13/12. Do not call pooled improvement
universal progress or claim the model is now error-free. A screenshot, water scene and
night street example were reviewed qualitatively; processing causes remain hypotheses.
No private image/text/filename is published. Detailed results stay in ignored work.

**Current local API admission is32MP, superseding the earlier16MP profile below.** All
210 gallery stills now fit; previously73 24.47MP images were blocked. Keep12MiB, one
request slot and all other bounds. Native RGB equality210/210 and HTTP6/6 pass. E92
weights/cuts and original-alert display policy stay unchanged; this fixes admission,
not the nine wrong classifications. The existing preview remains localhost:3002.

### Next model improvement, based on these findings

1. Freeze E95 errors and keep the gallery as consumed regression only. Do not change a
   cut, add filename/camera/screenshot exceptions, or train on these nine failures.
2. Register a distinct TRAIN-only context/processing experiment: use eligible existing
   real and AI parents with matched transforms to test whether highest-texture crops
   over-weight repetitive backgrounds compared with image context. Compare a declared
   uniform-coverage/global-context representation only after crop/source/scene roles,
   input identities and resource budget are frozen. This hypothesis is motivated by
   visual review, not yet proven. No new model fit started during the gallery audit.
3. Fit every representation/scaler/head inside TRAIN and select only on separately
   grouped CAL; do not choose crop counts/weights/thresholds on gallery outcomes. Keep
   strict E92/E43 AI retention, previous real-source guards, fixed cuts and original/Q75
   reporting. Screenshots/document images need an explicit separate specificity set,
   not posthoc deletion from the existing gallery's denominator.
4. Use the newly implemented732-view paired regression guard before accepting a future
   change:206 owner REAL+160 E66 AI, two conditions. Any newly wrong previously correct
   REAL or lost AI must reject it. Baseline self-check and synthetic failure probes
   passed. This is necessary consumed-DEV protection, not independent validation;
   candidate artifact and input contracts must be verified before comparing scores.
5. Require a separately audited unexposed source/generator evaluation for broader trust.
   Current gallery shows camera/scene concentration and no AI positives; volume does
   not prove OOD generalization. Keep data acquisition disabled on mobile connection.

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
## Historical overnight execution (2026-09-13; acquisition superseded on2026-09-14)

User explicitly requests continuous active-session work, data acquisition when needed,
experiment -> diagnosis -> justified development -> experiment, with MD and git checkpoints.
Do not create30-minute polling/heartbeat automation. No promise of execution after an app or
session interruption. E66 SIDD Small6.62GB acquisition and grouped REAL+AI admission are complete;
E70 consumed that limited DEV and failed. E71 passed TRAIN but failed consumed DEV; E72 admitted511 MIDD TRAIN rows; E75 features are complete; E76 failed TRAIN; E77 constrained head failed TRAIN; E78 DEAR-r weights verified; both synthetic numeric probes passed but the2h resource guard failed; E79 explicitly budgets2.5h for TRAIN features.
Whole E65 WIFD/RawNIND publishers stay diagnostic-only; old E59 fits stay paused. Keep existing AI
retention/absolute gates. Do not use later test errors to choose thresholds or recipe sweeps.

## Current authorization — local internship demo and reliability work (2026-09-14)

The user now authorizes connecting E92 to the local student demo, simplifying its
language, adding error protection, investigating/applying offline OOD reliability
checks, and revisiting the Model 2 plan. This supersedes the previous review-only
pause and the historical prohibition on changing the demo before E49 passes.
Research-demo integration is NOT Module-1 v1 promotion: E92's full acceptance still
fails its AI-retention guard. No public hosting, new training, downloads or changed
thresholds are part of this implementation. Existing frozen evidence stays immutable.

### Completed implementation — E93 evidence and alert-preserving demo

- Local web now uses E92 via `pixelproof.internship_serve:app` on127.0.0.1:8800.
  Current preview is http://localhost:3002/ (3000/3001 were already occupied).
  Set NEXT_PUBLIC_PIXELPROOF_API_URL=http://127.0.0.1:8800 and explicitly allow the
  preview origin in PIXELPROOF_CORS_ORIGINS. No public hosting was performed.
- E93 native parity passed14/14 source-representative views with exact score equality
  and zero cut changes. Its166 E65 scores are frozen. Raw E92 false alerts:2/83
  original and1/83 Q75; old E43:14/83 and18/83. This is exposed REAL-only source-shift
  evidence, not fresh final validation or a measurement of unseen-generator recall.
- **Do not use E93's strict positive suppression as the default UI policy.** The
  two-view AND rule preserves only158/160 E66 AI indications versus159/160 original
  E92. Its frozen results remain available; no E93 threshold or score was changed.
- Current `e92-preserve-alerts-v1` displays every original E92 AI indication and adds
  a visible review warning if its fixed E93 check disagrees. Negative outputs can
  abstain; no output certifies REAL. Pure replay on403 locked parents confirms zero
  added/lost original alerts. This preserves false alerts too: it is an honest
  presentation/error-handling improvement, not a learned OOD solution or lower FPR.
- Original E66 presentation:159 AI indications/1 uncertain among160 AI;71 no-clear
  and89 uncertain among160 REAL. One AI indication has an extra warning. The wrapper
  does not pass the old80% non-abstention target (230/320=71.875%); E92's raw20/20
  and its failed E43 AI-retention guard remain the correct historical claim.
- The12MiB/16MP API input policy is intentionally separate from native ML tests:
  all160 original SIDD PNGs exceed12MiB; E65 accepts23/83 originals, rejecting24 for
  size and36 for other geometry limits. Do not advertise320/83 scientific populations
  as end-to-end API coverage. Do not silently shrink test files to improve coverage.
- Six outcome-selected real HTTP cases pass, including unstable AI warning and
  uncertainty. No performance claim is based on this targeted integration selection.
  No uploads archived or used for training; offline encoders and loopback-only API.

### Model 2 next steps — existing data, AI local edits only

1. **Partial M2-0 complete:** audited512 local CocoGlide edited images,512 masks and
   512 authentic pointers. All512 nondegenerate binary masks match image geometry;
   exact authentic hashes are unique. The compilation appends another `.png` to
   original filenames; the new audit resolves only that documented mapping and
   rejects missing/ambiguous/out-of-root pointers. No extraction/download/inference.
   Remaining M2-0: decode authentic parents, audit pixel/perceptual near duplicates
   against all protected Module1/Module2 roles, verify semantic-scene/prompt ancestry
   and provenance/rights. Do not assign TRAIN/CAL until that grouped audit is frozen.
2. Preserve E17/E18's first120 possible exposures as historical diagnostic rows.
   The other392 files are not an unseen-source final: all are the same old CocoGlide
   collection. Reserve whole parent/scene groups prospectively; require a different
   audited publisher/generator before claiming broad localisation generalization.
   TGIF/TGIF2 remains a future acquisition lead, not downloaded on mobile data.
3. Repair evaluation first: include all eligible masks; disclose every rejection;
   half128px-tile stride=64px with edge coverage and overlap-averaged maps. No36-tile
   cap or truth-dependent >=50% mask-coverage survival filter. No per-image threshold
   chosen from the true mask area. E17's percentile(mask_fraction) IoU is an oracle
   diagnostic, not deployable localisation performance. Use CAL-only fixed thresholds.
4. Run fixed, zero-training baselines first (existing128px detector, local DINO/noise
   features) with pixel AUC/AP, image-macro metrics, area-stratified F1/IoU and authentic
   false-localised area. Compare random-area and centre baselines, resampling whole
   parent/scene groups. Never report pooled tiles as independent observations.
5. Only if the repaired baseline establishes signal, register one frozen DINOv2-S
   dense-token/linear-head baseline using existing eligible TRAIN masks; fit any
   normalization inside TRAIN, select on grouped CAL, then evaluate once. A shallow
   decoder/noise fusion is a later distinct hypothesis, not an architecture sweep on
   exposed examples. Keep E92 weights/cuts untouched. No training was started here.
6. Product scope: answer where an AI-assisted local edit may occur. Classic splices
   are specificity controls; fully re-rendered images have no reliable local-mask
   promise. Do not display E92's texture crops as an edited-region heatmap. No Model2
   demo integration until measured localisation exceeds its declared baselines and
   authentic-image controls at a CAL-frozen threshold.

### Model 1 next steps after this checkpoint

The original goals remain unmet in the broad sense: no clean independent final, two
seen E66 AI families, scene dependence and one E43-caught AI miss. E65 transfer is
encouraging but cannot prove unseen-generator retention. The next meaningful model
work is a registered source/scene/generator-held evaluation and matched-processing
TRAIN-only controls, not a cut adjustment to the new WIFD errors. Audit available disk
manifests for unexposed groups first; inventory volume alone does not grant TRAIN/final
eligibility. Every scaler/PCA/supervised map must be fit inside each TRAIN fold. Record
per-group recall/FPR, coverage and compression-chain failures. Current API bounds,
multi-user memory/latency and already-compressed submissions need separate prospective
coverage/stress tests before wider use. No new weights are justified by this small
REAL-only diagnostic alone.

### Executed E93 registration — preserve as the original protocol

1. Add a separate offline E92 runtime/API with verified local weights, exact frozen
   preprocessing, no old-model fallback, bounded uploads/inference concurrency, clear
   failures and no REAL certification. Keep the legacy endpoint as historical code.
2. Freeze guard `e92-stability-v1` before new diagnostic scoring: compare the submitted
   image with exact E49-style long-side1080/JPEG75 processing. Emit AI indication only
   if BOTH E92 scores reach the old AI cut; emit no-clear-signal only if BOTH are below
   the old REAL cut AND neither E43 reference reaches its AI cut. All other cases
   abstain. Reference disagreement can only abstain, never create an AI accusation.
   Reject unsupported animated/high-bit-depth inputs; under224px inputs abstain before
   inference. No fitted confidence, calibrated probability or OOD guarantee is claimed.
3. Replay this guard on all320 consumed E66 parent pairs, reporting AI indications,
   abstentions, no-clear outputs, source/scene false indications and raw E92 rates.
   Register deterministic native-image parity representatives before inference (first
   SHA-sorted parent per source, both views; tolerance1e-5, no cut changes). This is an
   engineering replay, not new validation. Keep raw and guarded metrics separate.
4. Once parity passes, evaluate all83 previously admitted E65 WIFD/RawNIND diagnostic
   parents and both transports using frozen E92. Bind existing manifest/file hashes
   and new runtime/guard code before scoring; lock scores before reporting. These
   publishers remain DIAGNOSTIC_DEV_ONLY, never TRAIN/fresh final. This checks REAL
   source shift only; E66's paired AI replay is limited to its two seen families.
   No follow-up threshold/recipe sweep after scores. No E49 access. Max1h on AC power.
5. Update the ordinary-user demo to consume only the new versioned E92 response; show
   plain indications/uncertainty, no percentage bar, no fake localisation heatmap.
   Explain upload destination and temporary processing accurately. Verify real API
   inference plus unit/API contracts and the existing web build checks.
6. Audit existing Model 2 image/mask pointers and preserve its inpainting-only scope.
   Plan mask coverage repair and scene-grouped evaluation before any training; record
   exposure/holdout limitations. E92 integration does not certify a localisation model.

## Current checkpoint — E92 numeric20/20; AI retention still fails (2026-09-14)

E92 completes the offline SID coverage experiment. On consumed E66 DEV, original
REAL FPR0% (0/160), social-Q75 REAL FPR8.75% (14/160); AI recall99.375% (159/160)
in both. All20 numeric gates pass, up from E86's17/20. Relative to E86,1 original and3
social REAL errors are rescued, with no new AI or REAL decision regressions.

Full acceptance is still **failed**: the same one E43-caught GPT original is missed.
At this frozen checkpoint: no E49 access, serving change or promotion. Original accuracy99.6875%, social95.3125%
are descriptive results on consumed DEV, not independent-final estimates. The20/20
count is acceptance-criterion coverage, not100% image accuracy. Preserve all old
scores/contracts and the fixed cuts. Detailed results live in HISTORY/EXPERIMENTS.

E91 completed100 epochs on49,076 existing TRAIN views in77.851s. E92 fit/runtime
217.573s, all120 TRAIN metric checks and AI/REAL/population guards pass. Cached DEV
scoring6.674s. No downloads, new encoder inference or training on DEV. A pre-freeze
DEV parent/view pairing bug was corrected and tested before any new scoring; the full
local suite passes858 tests (16.46s, one existing warning). E91/E92 candidate recipes
and predecessor artifacts stayed unchanged through the evaluation repair.

## Audit conclusion — numeric milestone, not market validation (2026-09-14)

The read-only audit independently verifies the counts, selective metrics and AUC, and
finds no observed parent-ID/file-SHA TRAIN/DEV overlap. It does not prove unknown
prompt/scene/pretraining separation. All gate constants match2026-09-04;20/20 includes
correlated/complementary criteria on paired conditions, not20 independent tests.
Social worst-scene FPR is40% (6/15) and36.84% (7/19) in another scene, while the
registered worst-camera gate passes. Covered accuracy is one extra error away from
failing95%. Detailed critique and evidence are in HISTORY and ml/EXPERIMENTS.

SIDD's10 scenes/older RAW-rendered phones and two seen AI families limit freshness and
representativeness. Repeated E66 use creates adaptive-selection risk even without
training on DEV. E49 is historically consumed and cannot be relabeled fresh. DEAR
weight/data restrictions block assuming commercial clearance. At the time of this audit, E92 was not the current E20/E32 web runtime and
lacked native-image end-to-end serving validation; E93/E94 now cover the local adapter. A student
research presentation is supportable with explicit limitations; public/paid general
purpose deployment is not yet validated or cleared. The review authorized no deployment; the later local-only internship request above supersedes that pause.

## Earlier audit priorities — retained context for the follow-up above

1. Freeze E92 as the reviewed checkpoint. Keep20/20 numerical success and the failed
   E43 AI guard together. No sample-specific GPTIMG_431 fix, post-hoc threshold change
   or retrospective waiver. Any revised product criterion must be prospective and
   cannot rewrite E92's acceptance result.
2. Define the intended user/input population and binary versus mixed-edit scope,
   false-accusation cost, prevalence and abstention policy. Do not call raw scores
   calibrated p(AI), certify REAL, or attribute E92 results to the E32 interface.
3. Inventory existing disk data for truly unexposed publisher/scene/prompt groups and
   legal use. Mark unknown provenance honestly. Reserve and hash a representative
   holdout before fitting/model selection; if none qualifies, record the gap and
   postpone independent-final claims. Unused files from consumed sources are not
   automatically a fresh benchmark. No new downloads on mobile data.
4. Register TRAIN-only source/scene/transport stability and matched-processing controls.
   Fit every learned scaler/projection inside its training fold; audit class-correlated
   format/size/noise/content cues. Keep held-out groups out of representation fitting.
   Use equal-budget comparisons before attributing gains solely to SID data coverage.
5. Plan prospective evaluations with realistic phone processing, compression chains,
   unseen generators and explicit edited-image cases. Report per-source AND scene
   failures, cluster-aware uncertainty, specificity/precision at realistic prevalence,
   calibration and coverage. Treat10 current REAL scenes as a limited diagnostic,
   not enough evidence for broad deployment confidence. Retain previous gates as history.
6. Compare E92 with simpler eligible baselines on the same locked evaluation and assess
   accuracy/latency/rights tradeoffs. Obtain suitable component/data permissions or
   replace restricted components and re-evaluate; free access is not automatic clearance.
7. Only after evidence/rights scope supports a research prototype, prepare a separate
   portable E92 inference bundle with native-image parity, correct model identity,
   p50/p95 latency/memory/concurrency checks, bounded input handling and a concrete
   image retention/privacy policy. Review before any user-facing release.

The critique registered no fit or serving change; see the newer local-demo authorization above. Update only these
three English Markdown records after each plan/development/experiment; keep old results
append-only. Reproducible audit: ml/tools/audit_e92_readiness.py and
evidence/e92_readiness_audit_2026-09-14.json. The separate Turkish report stays removed.

## Previous checkpoint — E71 TRAIN passed before failed DEV (2026-09-13)

All9,599 eligible E59 numeric chunks and their crop bodies passed validation; historical CLIP
replay on30 source representatives is exactly equal. All2,031 remaining TRAIN parents are now
complete under E71, with a verified11,630 x3 x1536 archive. The fixed next model uses E67 original64 plus TRAIN CLIP64 coordinates with unchanged
objective and AI/REAL decision constraints. Code is tested; the feature receipt and fit contract are frozen.
The single fit passed: REAL FPR6.99%/7.76%/6.82%, no new AI misses/REAL errors.
Freeze and run one separate consumed-E66 comparison; all20 numeric and retention gates remain.

E70 margin diagnosis is complete:109 TRAIN condition views sit near the protection boundary;
existing DEV losses include substantial reference margins. This is a possible next-mechanism
clue, not a change to E71 or proof of external retention.754 Python tests pass. E49/serving stay
unchanged. New SIDL/MIDD metadata leads are not admitted images. No new heartbeat was created.

## Previous checkpoint — E70 failed separate DEV (2026-09-13)

E70 passed TRAIN but failed the first E66 DEV screen: original REAL FPR 46.88% versus
42.50% reference, Q75 30.63% versus42.50%; AI recall fell in both conditions, including
four newly missed original and three newly missed Q75 AI views. Only12/20 numeric gates.
No E49 regression or promotion. E66 is now consumed DEVELOPMENT, excluded from TRAIN/final.
Preserve all failed candidates and scores. Assess a complementary pretrained representation
and the role/resource conditions of eligible cached work; do not sweep the failed maps.
739 tests passed. E43/serving unchanged; target remains unmet.

## Previous checkpoint — E65 diagnostic complete (2026-09-13)

83 source files acquired (1.01GB), all decoded,0 matches under the fixed151,082-record screen;
166 frozen E43 scores. WIFD false AI16.42% original/19.40% Q75; RawNIND18.75%/31.25%.
No candidate produced, no AI-retention measurement; target still unmet. Preserve E43.
See `evidence/e65_diagnostic.md` for descriptive counts, paired examples and confounders.

Next: freeze a licensed, scene/source-grouped balanced DEV population before a new fit.
Audit SIDD's small paired scene structure and the previously identified unused local AI
records for role/provenance/overlap first; their availability is not admission. Never reuse
whole WIFD/RawNIND publishers as TRAIN/fresh final after this diagnostic. The representation
hypothesis is processing robustness using label-matched transforms, not camera/ISO lookup.
Retain per-image AI replay constraints plus separate source-aware AI evaluation. No E49
threshold/rank sweep, relaxed E64 guard, or automatic E59 restart.

## Previous checkpoint — E62–E64 complete; goal remains unmet (2026-09-10)

Three preregistered recipes completed. E62/E63 reduce consumed E49 REAL FPR to35.8% original
and46.8/46.7% Q75, but AI recall falls to93.8% original and95.1/95.3% Q75. Both fail AI retention
and still pass only11/20 benchmark gates. Reject both; preserve E43 and live serving.
E64 preserves all caught TRAIN AI and rescues223/174 clean/Q75 REAL observations, but TRAIN
FPR11.90/11.33% exceeds its preregistered10% ceiling. No E64 test scoring is permitted.

- Do not repeat E62/E63 evaluations, relax E64's failed pre-test ceiling, overwrite artifacts,
  or conduct a rank/L2/threshold sweep on E49. These three recipes are finished; none is v1.
- Next meaningful improvement requires a distinct representation/processing hypothesis plus
  valid development data. E60's local audit has not established such a balanced fresh pool.
  The previous no-download condition is superseded; bounded E65 pilot is now acquired.
- Future optimization should preserve correct REAL decisions rather than unnecessarily freeze
  their confidence. Keep explicit AI constraints and the strict replay gate, but also require
  independent AI retention: TRAIN success alone failed to predict external preservation here.
- E43 remains immutable at its existing cuts. The unchanged official20 gates and stronger
  E43 AI-retention requirement remain the goal. E59/CLIP and heartbeat stay paused.
- 49 focused tests passed. Update GitHub with this completed checkpoint; existing separate
  Next.js audit debt is not evidence of an ML failure and must not be hidden.

## Executed E62 registration (2026-09-10)

User explicitly requests continuing toward the fixed test targets after E61. Proceed with one
new TRAIN-only mechanism; this supersedes waiting for home before any further engineering fit.
It does not authorize test-label training, score-based threshold selection, new downloads or
calling a consumed benchmark an independent final. Keep AI recall at the E43 reference level.

E62 replaces the soft-retention bounded correction with a convex constrained correction:
TRAIN-only PCA64 on E43-scaled cached features (seed62, randomized3 power iterations),
variance-normalized coordinates plus intercept; additive linear logit. Freeze code/input hashes
before fit. Class/source/parent-balanced BCE at fixed AI cut, hard REAL views2x within class,
L2.01. Every E43-caught admitted TRAIN AI must remain above the same cut; correct REAL
corrections cannot be positive. Zero start, SLSQP200 iterations, final solution only, CPU2
threads,20min budget,30% battery floor, no sweep. E43 artifacts/cuts remain immutable.

Require solver success, numerical feasibility and E61 zero newly missed AI views, zero new
REAL false alarms and lower REAL FPR on all3 TRAIN conditions. This is TRAIN feasibility,
not a quality claim. Only if it passes, freeze one consumed E49 paired comparison separately.
That report may establish whether this fixed candidate passes the existing benchmark gates;
it cannot establish a fresh final or choose another rank/loss/cut after scores are observed.
If TRAIN fails, stop this candidate before opening E49. E59 and home-data acquisition stay parked.

## Current follow-up — literature, strict replay guard and home data queue (2026-09-10)

- GitHub synchronization completed through `2da9fe8`; remote `main` verified. Continue
  pushing reviewed checkpoints. The server accepted the push while noting that two required
  status checks were still expected; push success is not a CI-pass claim.
- E61 is a **TRAIN-only engineering guard**, not another fit: reject an experimental candidate
  if any AI view caught by frozen E43 crosses below the unchanged AI cut. Another generator's
  rescue cannot offset that miss. Require complete bound parent-by-condition replay. Passing
  this necessary condition never permits promotion or proves unseen-generator retention.
- E61 completed: input/code hashes frozen in `evidence/e61_replay_contract.json` before
  replay. Unchanged E43 passes; frozen E60 is rejected on its known TRAIN losses
  (1/1/3 clean/transport/Q75, five distinct parents). No fitting, threshold selection,
  E49 reads, protected-data use or downloads. Do not rerun this write-once verification.
- GitHub CI for `2da9fe8`: Python job and web lint/typecheck/tests passed; web dependency
  audit failed on critical Next.js advisories. This is a separate dependency-maintenance
  item: verify supported patched version, update exact pin/lock together and run web checks
  before any deployment. Do not weaken the audit threshold or claim all CI passed.
- Independent promotion still waits for a valid development/final protocol/population;
  the newly authorized E62 TRAIN feasibility step is specified above. Investigate matched
  processing across classes and replay-constrained optimization, using TRAIN only; preregister
  one recipe before fitting. Do not tune E60 again against consumed E49 or treat a penalty
  coefficient as an AI-retention guarantee. E59 remains parked.

### Eve geçince indirilecek — metadata researched, NO image download now

| Priority / source | Intended question and role | Availability and admission conditions |
| --- | --- | --- |
| 1 — [WIFD](https://github.com/CSCRC-SCREED/WIFD) | REAL camera/exposure/ISO diagnostic; reserve the entire publisher for diagnostic DEV, no TRAIN until roles are deliberately revised before scoring | Official README: >6,000 images, 14 physical cameras, MIT for data/code. Includes non-Bayer Sigma Foveon. Burst/ISO/exposure copies are scene-dependent; group by underlying scene across cameras. Exclude flat reference frames from general-photo metrics. Byte budget and usable independent scenes unverified. |
| 2 — [RawNIND](https://github.com/trougnouf/rawnind_jddc), [official data DOI](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM) | REAL clean/noisy RAW pairs to diagnose denoising/development sensitivity; reserve publisher for diagnostics | Linked by authors. Bayer/X-Trans paths and paired clean/noisy workflow documented. Verify dataset/per-image license, byte budget, cameras, original splits and cross-corpus duplicates before download/admission; code/weights licenses do not establish data rights. |
| 3 — [SIDD](https://abdokamel.github.io/sidd/) | Conditional fallback for smartphone noise/ISO diagnostics; evaluate need after priorities 1–2 | Official site states MIT for dataset/code. Small has160 pairs, Medium320 pairs across160 scene instances of only10 underlying scenes. Processed ground truth is not a pristine camera JPEG. Older phones, no modern-phone coverage claim. Group underlying scenes across devices/settings, verify space/overlap; prefer Small if sufficient, not Full. |
| Watch only — [RealHD](https://github.com/Hanzhe-yu/RealHD) | Later mixed-generation/edit benchmark if actually released | Official repository still says “Comming soon!” on 2026-09-10. No verified downloadable corpus/license; not acquisition-ready. |

These are candidates **to download at home**, not accepted TRAIN or fresh final data. Reserve
publisher/scene/parent roles before scoring, retain RAW/JPEG derivation lineage, audit exact and
near duplicates against all protected pools, and cap downloads after checking metadata size.
REAL-only diagnostic evidence cannot establish balanced detection quality or AI non-regression.
Need a separately eligible AI population and supported independent source counts before a new
balanced DEV/final; this gap remains explicit. MNW/HDR+/E52/COCO/ITWSM/Module2 stay protected;
do not expand FiveK or download B-Free training data that collide with protected COCO.

## Earlier authorized office plan — E60 completed (2026-09-10)

The user now explicitly requests optimal continuation towards the existing goals with NO data
downloads. Execute the champion-first audit and one bounded correction study below. E59 feature
extraction and queued training remain parked; scheduled heartbeat remains PAUSED. This restart
does not resume the superseded E59 route or authorize new source, weight or package downloads.
Keep completed research/cache/history, do not delete or promote it. The lower source-fold
scores are not a degraded overwrite of the good full model; nonetheless, experiment expansion
displaced the user's priority and must stop being the default development route.

### Offline office plan — improve the actual strong reference, not a weaker replacement

1. **Pin the reference and the evaluation question.** E43-S is the historical high-recall
   research model: `/Volumes/LaCie/pixelproof-datasets/e43/e43_small_predev.joblib`, verified
   SHA `a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390` matches E49's
   frozen score contract. Preserve head/scaler/backbone/crops/cuts; do not overwrite. E49
   original/Q75 recall94.3/95.5%, REAL FPR39.1/49.0% are paired historical diagnostics, NOT
   final success. Separately identify the web serving profile (canonical E20/optional E26/R1b
   in current code); do not confuse a research champion with what the UI actually serves.
2. **Audit existing local material before any fitting.** Recover E43's exact FIT parent/source
   membership and preprocessing, compare the prior E51/E54 data reductions and loss mass, and
   identify eligible REAL hard negatives and complete AI replay already on disk. Check label
   direction, RGB/EXIF/resize/compression and inference parity. This is not another acquisition
   or broad architecture search. Do not assume all 8,844 old E43 parents are reusable: preserve
   modern protection/overlap rules and explicitly report conflicts. No protected/test data can
   be promoted to TRAIN to reconstruct the old training recipe.
3. **Audit a valid development comparison for a warm-started champion.** E43 already saw some
   E54 fold validation parents; its frozen head must NOT be treated as an unseen-fold comparator
   or teacher in a purported clean OOF experiment on those rows. Identify a genuinely held-out,
   local development population outside the teacher's FIT and every prohibited role. Existing
   consumed E49 scores can explain failures and later supply one preregistered regression check,
   not choose weights/cuts or certify a fresh final. E51 consumed DEV, MNW/HDR+, E52 and Module2
   boundaries remain intact. If no permitted fresh development population exists offline, report
   that limitation: engineering/TRAIN checks may proceed, but no independent quality claim.
4. **One bounded champion-preserving candidate, after steps1–3.** Keep E43 backbone, existing
   scaler and original head frozen. Proposed first mechanism is a small zero-initialized additive
   logit correction using existing DINO features, rather than a randomly rebuilt detector. At
   initialization it must reproduce E43 scores/decisions exactly. Fit ONLY admitted TRAIN hard
   REAL examples together with all eligible AI replay; use true labels, plus an explicitly frozen
   penalty against changing the reference's correct AI margins. Teacher outputs are a retention
   aid, not ground truth. Do not merely raise a threshold, apply a blanket real veto, or assume
   distillation guarantees unseen-AI preservation. Freeze exact loss coefficients, correction
   bound, optimizer/steps, runtime and split/input hashes BEFORE training, after the eligibility
   audit. No hyperparameter/architecture sweep. This is a proposed mechanism, not a fitted winner.
5. **Compare champion versus candidate on identical lawful held-out rows.** Fixed operating
   rules, original andQ75, paired rescued/new errors by REAL source and AI generator. Preserve
   pooled and supported-generator AI recall with the existing uncertainty/no-loss guards while
   reducing REAL false accusations; report inconclusive/regression as such. No blanket %66 vs
   %94 cross-protocol comparison. One frozen candidate, not endless retuning on observed tests.
   If it fails, retain E43 and explain the failure before another experiment. Independent final
   and serving promotion require all existing prior gates and an eligible new final population.

Execution order in the office: reproduce/reference audit -> lawful data/role audit -> register
the single correction experiment -> bounded fit -> paired report. Reuse cached DINO features;
zero source/weight/package/image-API downloads. E59/CLIP is parked as a later optional route,
not an automatic dependency or the default starting point. The user's restart authorizes this
plan, not the old queued E59 continuation. Keep planned versus measured results separate.

### E60 completed checkpoint — correction insufficient; retain E43 (2026-09-10)

The one registered fit and4,000-view consumed E49 comparison are COMPLETE. Do not repeat
freeze/fit/score/report. Baseline reproduction error0; no binary/selective decision drift.
E60 preserves observed E49 AI recall94.3/95.5% with zero new AI misses, but REAL FPR only
changes39.1->39.0% /49.0->48.8% (one/two rescued REAL observations). Source-cluster improvement
intervals include0; absolute/selective gates still fail. This is not an accepted improvement,
fresh final or serving promotion. Keep E43 unchanged and archive the correction as research.

Before any subsequent fit, define one new TRAIN-only hypothesis and a valid local evaluation
population. The audited cached/native pool lacks balanced unseen recorded groups; do not
recycle E49 to select a stronger correction, change the bound/L2/retention after its scores,
or open MNW/HDR+/E52/Module2 as convenient DEV. Other already-local populations require a
score-blind provenance/role/group audit before eligibility can be claimed. If none qualify,
state the independent-evaluation limitation; no new downloads, automatic E59 restart or
claim of v1 completion. All E60 processes are finished; scheduled heartbeat remains PAUSED.

The following is the executed E60 recipe, retained here only as checkpoint context; full
protocol/outcomes are in HISTORY and EXPERIMENTS.

Audit v2 reproduces E43's8,844 parents/19,648 views. Current E54 TRAIN has7,035 REAL and
4,595 AI; use every admitted AI parent.4,278 parents/encoded bodies match E43, while360 old
AI parents are now protected E51 CAL and4,206 other old parents lack current admission.
Do not restore those rows. E54 validation folds contain2,360/480/1,438 E43-seen parents.
The10,508 unused audited native parents supply no unseen recorded REAL group; only2,385
AI remain outside current recorded groups. No fresh balanced group-disjoint cached DEV in
this audited pool. Keep E51 consumed DEV and E52/MNW/HDR+/Module2 protected.

Freeze e60_run.py/e60_correction.py, then one CPU-float64 fit on all34,890 existing TRAIN
views. Frozen E43 backbone/scaler/head; correction2*tanh(X*w/2), no intercept, w=0; X is
E43-scaled features divided by sqrt(3072). Class/source/parent-balanced operating-cut BCE;
false-positive REAL views2x, renormalized within class. Correct-AI negative-margin penalty10;
L2 .01. Full-batch Adam200 steps, lr.02, betas.9/.999, eps1e-8, final step, two CPU threads,
<=20min. Fixed AI/REAL cuts.07940196245908739/.011505939625203613. Exact zero and saved replay.

After candidate hash freeze, one consumed E49 regression:4,000 views, batch16, <=15min
inference, network connects denied, battery floor30%. E43 must reproduce old scores within
5e-5 with zero binary/selective decision changes. Lock raw scores before metrics; report paired
source-wise rescued/new errors,20 absolute/selective gates and AI no-loss guards with20,000
source-cluster paired resamples (seed60, four familywise intervals). This is engineering and
consumed regression evidence, not fresh final or serving promotion. If it fails, retain E43.

Candidate SHA582d6c4f...415bc5; zero-init and saved-vector replay error0, unchanged E43.
TRAIN clean/Q75 already had one/three new AI misses, despite no such misses on consumed E49;
this distinction further prevents claiming universal AI retention from a penalty or one test.

### Historical E57–E59 execution checkpoints (not active instructions)

**Current active invocation09:24:** persisted8,308 parent chunks (342,619,493B, no partials)
after the second planned deadline. Original processes exited; unchanged extractor resumed
with guard40155/worker40165 and log `ml/work/e59_features_20260910T092449.log`.
Bounded waiter41371 watches ONLY guard40155 (`experiments.e59_followup`, <=60min combined
wait/training). Do not start another extractor/waiter/trainer while these are active. It will
handoff to existing training guard only with all completion receipts and remaining time;
otherwise stop, leaving the next scheduled check to inspect/resume. No model score yet.

**09:23 operational continuation:** second feature invocation reached its registered60min
deadline, no old worker remains. Resume the unchanged extractor after AC/real-disk checks.
To avoid an idle gap if this invocation completes, a separate bounded operational waiter may
watch the exact observed feature GUARD PID (not duplicate or interfere with it). After that
guard exits, require full archive and both completion receipts, then hand off to the existing
exclusive freeze/fit/report runner using only the remaining time within a60min total budget.
Reject PID reuse or incomplete receipts; no retry loop or scientific-code change. A separate
waiter lock prevents duplicate queued continuations. Training remains forbidden while the
feature lock is held. This extends operational sequencing only; all experiment gates unchanged.

**07:53 resume checkpoint:** first E59 extraction stopped at its registered60min deadline
(TimeoutError, not a quality failure); no old worker remains. Inventory contains4,465 complete
parent chunks,173,197,992B, no `.part` file. AC/actual LaCie/~379GiB free verified. Start the
same frozen guarded command for another<=60min; new log
`ml/work/e59_features_20260910T075334.log`, observed guard10657/worker10663. It revalidates
every retained chunk before reusing it, then continues from the next parent. This inventory
count is not a completed feature receipt or fitted-model result. No new tests/code/weights/
source downloads needed for unchanged science. Do not duplicate the active resume.

**06:54 implementation checkpoint, before any E59 fit:** extraction still active (at least
2,300 completed parents); do not spawn another extractor or fit alongside it. Prepare the
registered nine-fit implementation and fixtures while the worker continues. Freeze its actual
feature/teacher/reference/code hashes only once both full feature receipts exist. Retain all
source/fold losses. Additionally verify within-E59 DINO control versus E57 native64 on exactly
paired validation rows (max score/cut difference<=5e-5, zero verdict changes); a mismatch blocks
interpretation/promotion, not permission to adjust tolerance after seeing results. Fit outputs
are atomic and resumed only with matching contract/artifact hashes; an orphan completed head
without its receipt stops for explicit audit. A <=60-minute locked guard handles freeze/fit/report
after complete features, never opens final reserves or modifies serving. This prepares the
next stage; it does not mean training has already started.

The downstream implementation is now prepared in `experiments.e59_model` and
`experiments.e59_train_run`. Do not start it while extraction is active/incomplete. Once full
feature archive and both receipts exist and feature worker has exited, run
`PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets PYTHONPATH=ml:ml/src
ml/.venv/bin/python -m experiments.e59_train_run --minutes 60`.
It freezes actual completed-feature/code/reference hashes before fitting, then fits all nine
fixed heads and reports under the shared exclusive lock. Archive contract/results in git at
next checkpoint; source/pre-plan are committed before execution. Do not rerun completed stages
or declare a winner without inspecting every gate and independent-evaluation prerequisites.

**Active work 06:22 local:** E59 full CLIP feature extraction started under60min guard;
feature contract224382c10f2831a9a522bb668e952c30b3a1be890fa2219044f99e7679605954,
local commit f533877. Log ml/work/e59_features_20260910T062252.log; inspect actual process
before any next invocation. Start command: `PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets
PYTHONPATH=ml:ml/src ml/.venv/bin/python -m experiments.e59_features run --minutes 60`.
Do not run it while its guard/worker is active. After bounded stop inspect log/power/mount;
the same command validates and reuses completed chunks. After full completion, do NOT repeat
extraction: implement/freeze the separately registered nine-fit training comparison. No new
source or weight download, fitted detector, full feature completion or quality gain claimed yet.

E57 completes 199 admitted FiveK parents/597 views and six fixed fits; two unsupported WB
originals excluded without replacement. Stable native -> supplement clean REAL FPR
19.5291 ->18.3147%, AI recall 66.2378 ->66.4436%; Q75 FPR 18.7361 ->18.2404%, AI
64.9511 ->65.8775%. Paired REAL improvement intervals exclude zero, but modern AI sources
lose up to 7.5 percentage points; AI preservation intervals include loss. All absolute gates
fail; exact saved-head replay passes. No winner, serving change or new final-reserve inference.
Do not enlarge FiveK or sweep its weights following this failed hypothesis.

### E58 next bounded diagnosis — existing scores only, before any new fit

First freeze E57 results/artifact hashes, then inspect native64 versus native64_fivek using
their existing source-held-out observations. Report rescued/new errors by source, fold and
clean/Q75. Separate observed CAL-selected operating-point loss from ranking limitations:
compute an explicitly optimistic, validation-label oracle ROC envelope for each arm/fold/
transport, with maximum recall at FPR<=10% and minimum FPR at recall>=80% and >=95%.
These are descriptive impossibility checks on already-consumed development folds, NOT
new deployable thresholds. Do not export threshold values, save a candidate, adjust CAL,
infer any new image, pool incomparable fold scores, or open independent reserves. Ties
must be indivisible; compare original observations exactly and retain all subgroup losses.
Register code/input hashes before executing the diagnostic, with fixtures for ties and
perfect/reversed rankings. Report all six fold/transport cells, not only favourable ones.

Primary UnivFD research motivates examining pretrained representations; E47 already tested
its official CLIP head, so do not claim this is a new untried detector. B-Free motivates paired
content/processing coverage, not repeating our unpaired FiveK addition or admitting protected
COCO training data. After the diagnosis, choose one separately preregistered representation/
training-data hypothesis based on the limiting cells; no automatic parameter sweep.

**E58 completed in the 05:38 follow-up:** the supplement's optimistic AI recall at REAL FPR10 is clean/Q75
37.30/34.59% (fold0), 75.83/80.00% (fold1), 66.29/68.27% (fold2). Even label-oracle
thresholds cannot reach AI95/FPR10 in any of six cells. Fold1's clean ranking also worsens
against native (78.96 ->75.83% at FPR10); this is not solely a changed CAL cut. No cuts saved.

### E59 next distinct hypothesis: complementary pretrained representation, no new data

Do not repeat E47's frozen ProGAN detector or retune FiveK. Assess the already licensed/cached
CLIP ViT-L/14 image encoder as additional features learned from the existing E54 TRAIN pool,
with no E57 supplement. E47 demonstrated different error patterns but did not fit this encoder
on these source-held-out folds. This is a new representation hypothesis, not guaranteed recovery.

First preregister a resource/parity-only probe: four REAL and four AI parents from fold-0 FIT,
deterministic SHA256 rank `E59_RESOURCE|`+parent ID per class. Use their existing three views
and three RGB224 crops; CLIP's official normalization on unchanged cached pixels. Verify pinned
CLIP backbone, upstream checkout/code/licence and crop hashes. Frozen eval/no gradients, batch
three crops, one warmup then all 24 views twice, finite 768-D embeddings and replay tolerance
1e-5. Report runtime and sampled memory (not peak), no head scores, fit, accuracy or saved
candidate; <=15 minutes, AC/external disk checks and no download. Raw 768-D crop features are
used for the parity probe; any normalization/aggregation for a later classifier needs its own
contract. Do not infer future throughput from this tiny sample as a guarantee.

Only after this engineering check, freeze complete feature and training plans: existing 11,630
parents/34,890 views, unchanged folds/AI replay/CAL/gates, fixed DINO-only baseline versus
CLIP-only and DINO+CLIP features, three folds each. No expansion, grid search, test-selected
mixture weights, learning from protected publishers or final inference. Define exact aggregation,
feature scaling/loss mass and code/input hashes before full extraction or any fit; benchmark
determines resource feasibility, not accuracy-based hyperparameters. E59 is not yet a fitted study.

**Resource check completed:** MPS, 24 views/72 crops per pass; 7.9457 and7.8218 seconds,
exact embedding replay error0; sampled driver allocation2,194,358,272B (not peak).
No downloads, detector scores, saved head or fit. Evidence/e59_clip_probe.json; full suite638
passes. No worker remains active after this bounded probe. Next scheduled continuation should
start with E59 full feature/training contract preparation, NOT repeat completed E55-E58/probe.

Full extraction will plausibly take hours (small-sample extrapolation only); implement resumable,
parent-bound external chunks and <=60-minute guarded batches before starting. Preserve completed
chunks on timeout; power/disk loss stops only owned workers. Freeze exact CLIP normalization,
three-crop aggregation, offline weights, whole parent/fold joins and all helper hashes before any
full feature run. The later nine fits must be a separately frozen contract with no score-guided
feature selection, no FiveK enlargement, no weakening of existing source-wise AI preservation.

**E59 full feature registration, 06:18 follow-up (before extraction):** keep all existing
11,630 E54 parents/34,890 ordered views. Offline pinned CLIP ViT-L/14, official image
normalization on exactly the cached three RGB224 crops per view, frozen float32 eval, batch
three crops. Save raw 768-D crop embeddings and fixed concatenated three-crop mean and
population standard deviation (1,536-D/view); no per-vector L2 normalization, feature selection,
colour/resize change or fitted preprocessing. This mirrors the DINO crop-statistics structure,
not an assertion that it is optimal. Hash-bound parent chunks resume without resampling;
verify content digests, finiteness, shape and aggregation equality. Retain completed chunks on
deadline or power loss. Full cache on actual LaCie under e59 only; no source/weight downloads.
Feature validation uses no classifier or performance labels. Exact same-batch replay required
within1e-5 at each worker start on the first bound parent. Parent chunks can commit only after
power/storage/deadline checks. Guard each invocation<=60min, one shared lock, 2s monitoring,
stop owned process group. No endless unmonitored process. Immutable completion receipt binds
all chunks, final feature archive, parent order, upstream inputs and frozen code. Interrupted
finalization may reuse a complete archive only after exact reconstruction matches its binding,
parents and arrays; never overwrite incompatible completed evidence.

Training recipe locked before future scores: DINO3072-only, CLIP1536-only and concatenated
DINO+CLIP4608, each on the unchanged three source-held-out folds. Fit weighted StandardScaler
only on FIT, float64 logistic regression C=.01/tol1e-8/max_iter1000/seed53/lbfgs/twoCPUthreads;
convergence warnings fail. Retain every AI view and unchanged class/source/parent weights with
original base FIT total loss mass (not native-expanded count). Existing inner CAL clean/Q75
selects the cut; validation clean/Q75 unchanged. No FiveK. Compare all nine fits against existing
required historical references, stable E57 native control and within-E59 DINO control, with
all unchanged absolute, exact-replay and paired AI-preservation/REAL-improvement gates. CLIP
alone and combined are separately reported; neither may replace serving without further gates.
The actual training implementation/artifact and input hashes must freeze separately after
feature completion but before fitting. Full feature acquisition does not authorize a final test.

### Prior E56/E57 registration and execution record (historical checkpoints)

E56's 12 DNGs complete: 122,074,000 original bytes +166,400,845 derived PNG bytes. All fixed
decodes repeat with exact RGB equality; external reference screen (150,883 observations including
MNW/HDR+) finds zero byte/canonical/near-hash matches and zero internal pairs. This is a passed
engineering pilot, not detector accuracy or exhaustive deduplication; no model fitted or served.
Current historical E55 rejection remains intact. Do not repeat completed E55/E56 jobs.

### E57 next bounded acquisition and training hypothesis (before new selection/scores)

Hypothesis: a small, independent REAL source with explicit content/light coverage may reduce
held-out REAL errors without losing modern AI recall. This is a falsifiable supplement trial,
not paired-AI alignment or proof that all previous errors came from data.

1. From the pinned 5,000-row FiveK metadata, take min(12, cell size) from each subject x lighting
   cell, hash-ranked using `FIVEK_E57_TRAIN_V1|` + exact filename. Expected 201 parents over all
   18 cells; two cells have only four/five parents. Do not upsample those as independent examples.
   Freeze selected identities and HEAD validators before image GET. <=48 MiB/parent, <=4 GiB total
   original payload; real external disk, AC, >=20 GiB free, <=60-minute acquisition runs. If selected
   resource violates cap/identity, report it without replacing a difficult or large row silently.
   Reuse any exact pilot originals/PNGs only with their verified recipe and hashes; never redownload
   completed verified bytes. Copy both research licences/attribution with the source.
2. Preserve isolated E56 decoder/recipe and all frozen ML dependencies. Canonical and near-duplicate
   screen against the pinned complete protected reference plus 400 reserve bodies; exclude both
   endpoints of any internal pair and every cross-protected match, without replacement. Group all
   renditions under one parent and entire FiveK publisher as FIT-only research, never CAL/validation
   or final. Archive exclusions and actual count before feature extraction/training admission.
3. Freeze DINOv2-S/3072-dimensional clean, assigned-transport and Q75 feature extraction exactly as
   E54's admitted input recipe. One original parent stays one parent. No random augment/grey sweep.
4. Register a distinct E57 training contract: three old source-held-out folds, two arms each —
   stable float64 native baseline and identical baseline plus admitted FiveK FIT parents. Fixed
   C=.01, lbfgs tol=1e-8/max_iter=1000, seed=53, two CPU threads; convergence warnings fail closed.
   Keep all existing AI replay rows/views and class/source/parent balancing, 50/50 class mass and
   original base FIT view-count total mass. No validation-guided fit/threshold sweep. Existing
   inner CAL alone selects each cut, outer validation is unchanged; freeze code/inputs first.
5. Require saved-artifact replay and all unchanged absolute/AI-preservation/REAL-improvement gates,
   against both stable baseline and historical required references. Report all six fits, all source
   losses and paired intervals, not just pooled gains. No external reserve or serving change until
   prior gates pass. A failed supplement ends this hypothesis; no automatic enlargement or retuning.

No E57 image selection, acquisition or training has started at this registration checkpoint.

**04:32 operational continuation:** acquisition is now active for the frozen 201-parent/1,999,000,262
byte selection. Scientific model implementation is committed before extraction/fitting. A separate
operational guard may wait for the exact existing acquisition PID to exit successfully with both
receipts, then run audit -> model-contract freeze -> feature extraction -> six fits -> report.
No duplicate download; refuse PID reuse or missing receipt, stop owned children on power/storage
loss, and cap the combined wait/follow-up invocation at 60 minutes. Each stage logs under ignored
ml/work. This automation does not relax admission, frozen gates or authorize serving. Freeze-stage
input hashes are written before any new feature/fitting step; archive the resulting receipts at
the next checkpoint. If the bounded run ends unfinished, resume only pending stages after checks.

**E57 v1 acquisition stopped before model scores:** 61 parents decoded; selected
`a1854-kme_290.dng` (DCS460D) reports camera WB [0,1,0,0], with a valid Bayer RGB sensor. This
is missing as-shot metadata, not a flipped label or proof of a monochrome image. The queued
follow-up correctly refuses missing completion receipts. Preserve v1 code/contract/partial data.

Preregister v2 engineering admission amendment before continuing: retain exactly the same 201
identities, byte caps, fixed decoder, licences and folds. Before decoding each original, inspect
as-shot WB model-blind in the isolated environment. Require four finite coefficients with the
first RGB three positive, as the unchanged E56 decoder requires. Every failure of this one
predicate is quarantined/excluded without replacement; archive its RAW hash and metadata. Do not
invent daylight/auto-WB, silently change pixels, remove by detector score or reinterpret labels.
All other download/decode failures still stop. Reuse verified v1 originals and completed RGBs,
and keep v2 receipts/manifest/results separate. Report lost subject/light/camera coverage and
remaining N before any fit. This restricts the experiment to supported as-shot RAWs; it does not
solve missing-metadata photographs or establish universality. New model contract must bind v2.

**03:48 pilot implementation registration:** use seed string `FIVEK_E56_PILOT_V1|` plus exact
filename SHA-256 ranking, two per all six declared subjects. HEAD-pin size/ETag/Last-Modified
before image GET and fail on oversize/identity change. At most 30 minutes per acquisition run;
check AC/mount/free space before files and periodically while streaming. Isolated decoder under
ignored `ml/work/e56_decoder`, rawpy=0.27.1, numpy=2.5.1, Pillow=12.3.0, binary wheels only; archive
installer report hashes and runtime LibRaw version. Fixed AHD, full resolution, camera WB, auto-WB
off, sRGB primaries, gamma=(2.4,12.92), auto-bright off, brightness=1, highlight clip, 8-bit RGB,
no added denoise/sharpening. Keep original orientation handling and record dimensions. No expert
TIFF, screenshot, automatic brightness tuning or image replacement. Standalone decoder must not
import or alter ML training dependencies. All raw/decoded bodies stay quarantined on LaCie.

**03:08 follow-up, before execution:** add a descriptive metadata-only coverage audit of the
11,630 admitted E54 parents. Bind existing data/colour receipts and immutable code hash; count
class support per publisher, geometry/colour strata per fold/role and available scene/device
metadata. Mark absent content annotations unknown, never infer topics from filenames or invent
matched REAL/AI pairs. Verify exact parent joins and no publisher crosses FIT/CAL/VALIDATION.
No image reads, feature inference, model fit, new test scores or threshold tuning. Compare these
coverage facts with primary B-Free documentation; its COCO origin overlaps a protected publisher
in this project, so its training release is not automatically admissible. Evaluate FiveK as a
research-only REAL candidate, but require fixed colour decoding, byte cap and overlap checks
before any image transfer. No existing frozen helpers are edited.

Coverage result: 10/11 frozen publisher groups have only one class (9,270/11,630 parents,
79.71%). Fold 0 has 9/5,314 monochrome REAL in FIT versus 229/1,250 in held-out RR. Missing
topic annotations prevent claiming content matching; old E40 already tried embedding-cluster
weights, so do not repeat that as a new method. B-Free's paired COCO training is scientifically
relevant but not admitted under current whole-publisher protection. FiveK offers explicit subject,
lighting and human tonal-edit metadata, not modern phones or paired AI. Next bounded acquisition:
archive only the official FiveK index, both research licences and both filename licence lists,
at most 12 MiB total /8 MiB per response on LaCie; zero image bodies. Parse candidate metadata,
validate all 5,000 identities and unique licence assignments before selecting any pilot. Missing
or ambiguous metadata blocks image acquisition, not a reason to guess. Future image pilot must
be separately frozen (byte cap, deterministic strata, decoder, overlap, TRAIN-only designation).

FiveK metadata v1 downloads all five text resources but correctly stops before admission: official
licence lists name extensionless stems, while the index names `.dng` originals. Preserve v1 code
and contract; add a separate offline v2 reconciliation using exact stem equality for validated DNG
URLs only. Pin all five downloaded byte hashes, retain the failed-v1 reason, reject duplicate or
cross-licence stems and require a complete 5,000-row union. No fuzzy/numeric-id match, image
download, metadata refetch or model score. This is a parser schema correction, not a licence waiver.

**Metadata review complete:** 4,725,181 archived bytes, zero image bytes; all 5,000 parents have
one licence (2,690 Adobe /2,310 Adobe+MIT). Six subject categories and three lighting categories
are available, including explicit unknowns. V1 schema failure is preserved; v2 reconciles exact
filename stems offline. No fitted candidate or accuracy improvement is claimed.

Next bounded work package: freeze 12 original-DNG pilot parents, two per declared subject category,
deterministically hash-ranked from the pinned manifest (including `unknown`, no score selection).
Maximum 32 MiB per image /384 MiB image payload, on LaCie only; reject oversize/changed resources
without silent substitution. Pin observed HTTP identity/size and receipt hashes; upstream lacks a
per-file cryptographic digest, so distinguish local integrity from publisher authentication.
Use a separate, version-pinned RAW decoder environment, never alter the frozen ML environment.
Record LibRaw/rawpy versions and a fixed camera-WB, sRGB, explicit gamma/brightness/bit-depth recipe;
verify dimensions/finite decoding and preserve DNGs, licences and lossless derived RGB hashes.
Do not treat expert TIFF16 ProPhoto bytes as sRGB. Check byte/canonical/near-duplicate overlap against
the entire protected reference including MNW/HDR+ before TRAIN admission; pilot images remain
quarantined until that passes. No held-out scoring. If the decoder, licence or source check fails,
archive the failure and stop this pilot. Only then preregister a bounded content/light-stratified
REAL supplement against the same baseline with unchanged AI replay, stable float64 controls and
all original gates. New REAL alone is not paired-data alignment or a guaranteed fix.

Numerical audit now completes: exact FIT features, labels, sources and parent order match in all
three folds; collapsed weights differ by at most 2.23e-16. Fixed float64/tol=1e-8 duplicate fits
agree on FIT scores within 2.24e-6. This supports numerical sensitivity of the old float32/default
tolerance comparison, not a label reversal or a quality improvement. Precision and tolerance were
changed jointly; no old contract/result is replaced. No diagnostic candidate was saved.

Grayscale reduces monochrome REAL errors (clean 169->152, Q75 170->159) but increases other REAL
errors (618->660, 587->626). Reject the global augmentation; do not sweep grayscale percentages.
Next: audit admitted TRAIN coverage by publisher, content and processing style, and identify an
independent licensed REAL training source before acquisition. Existing MNW/HDR+ reserves and
consumed CAL/DEV/final publishers remain protected. Use source/content-matched training research
to motivate one bounded successor, with numerically stable paired controls, fixed AI replay and
the unchanged no-loss gates. Freeze its own contract before fitting; no successor fit is started
by this diagnostic. New downloads require a size, licence, split and overlap plan first.

E55 completes 34,890 derivative views and all six fits. Grayscale augmentation has clean/Q75 AI
recall 67.78/66.03%, but REAL FPR 20.12/19.45%, worse than native baseline 19.50/18.76%; all
acceptance gates reject it. Exact saved-head replay passes. Duplicate-control score parity fails
(max errors .00118/.00768/.00238 by fold; two decisions differ in fold 1). Do not relax tolerance
or rewrite this completed contract; grayscale is not an accepted fix and the controlled causal
comparison is qualified by this numerical discrepancy. No independent reserve opened.

Completed preregistered diagnostic, before any successor fit: reconstruct the native baseline FIT arrays and
compare with the E55 teacher/order/labels/sources/parent weights exactly. Inspect saved scaler,
coefficient, iteration and objective/gradient differences. Test one fixed numerical hypothesis:
unduplicated versus 80/20 duplicated mathematically identical FIT loss, with float64 inputs and
tol=1e-8, max_iter=1000, C=.01, two CPU threads, for each existing fold. Compare predictions on FIT
only; no CAL/validation retuning or accuracy-based parameter choice. Do not save these diagnostic
heads as candidates or change old artifacts. Freeze diagnostic code/input hashes before executing.
Separately join already recorded E55 predictions with colour descriptors to report errors rescued
and introduced by source/colour; zero new external model scores. Use this evidence to choose the
next coverage/representation hypothesis rather than repeat grayscale variations.

**Continuous follow-up requested, 01:53 local:** user explicitly asks for ongoing background
research/development/testing. Update the existing thread heartbeat (no duplicate automation) to
30-minute recurring continuation without the former morning expiry. Each cycle uses measured
results -> source-level errors -> primary-source research -> preregistered bounded experiment ->
verification; no repetitive busywork or unsupported success claims. The already-running E55
process keeps its original 08:00 safety deadline; do not modify or duplicate it. Later individual
jobs need their own explicit bounded runtime and power/storage checks. Stop the follow-up when
the user requests it or the documented objective has genuinely been independently established.
Keep notifications limited to meaningful changes/results or necessary user actions.

**01:41 local:** user reconnects LaCie. USB identifies the external Rugged disk; macOS runs its
own `fsck_exfat` check before mounting. Let that OS-initiated check finish without interruption,
forced mount or repair commands. The volume then mounts normally and guarded preflight passes
all power/storage checks. All 17 archived E55 chunk hashes (816 views) verify, as do the frozen
experiment and bound repository-helper hashes. Resume via the existing guarded command below;
do not refreeze the scientific contract or count previous cached rows as new independent data.
The older missing-disk notes describe the earlier pause, not the current state.

**00:29 local continuation check:** user authorizes overnight work and public downloads. AC Power
is confirmed, battery 1% and charging. `/Volumes/LaCie` is absent; `diskutil list physical` finds
only the internal physical disk, and USB inventory finds no LaCie/Rugged/Seagate device. Do not
invent a replacement mount directory or mirror hundreds of GB onto the internal drive. User has
been notified that the external disk must reconnect. No E55 extraction/fit is resumed yet.

Thread heartbeat `gece-model-geli-tirme-takibi` now checks every 30 minutes without the former
08:00 expiry, following the later continuous-work request. Avoid repeated unchanged notifications, duplicate jobs and manual-gated/paid
services. Before any heavy work verify AC, the actual mounted external disk and available space.
An unchanged-code E55 resume wrapper is the next engineering step: validate the archived pause
chunk hashes, obtain a single-run lock, select only unfinished stages, and stop its own child if
power/disk safety fails. It must not alter the frozen colour experiment or scientific gates. Test
these checks with synthetic fixtures without requiring or fabricating external data. Resume the
existing contract on disk return, then assess all gates before planning the next experiment.

Resume command (after AC and disk verification):
`PYTHONPATH=ml:ml/src ml/.venv/bin/python -m experiments.e55_resume run --until 2026-09-10T08:00:00+03:00`.
The wrapper fixes the external/offline/thread environment for its children; logs go to ignored
`ml/work/e55_resume/`. Eleven operational unit cases pass: charging/fail-closed power parsing,
unfinished-stage selection, orphaned feature-finalization rejection, preserved chunk hashes/path
safety, and stopping only its owned live child. A real preflight correctly refuses the absent disk.
This verifies engineering behaviour with fixtures, not actual E55 resume or model improvement.

The operational notes below are retained as the checkpoint context, not a claim of active training.

**Operational pause, 21:18 local:** `pmset -g batt` reports Battery Power, 3%, approximately eight
minutes remaining. Stop only this turn's E55 extractor and temporary throughput diagnostic;
both exit 143. No training/download is left running by this turn, and the existing web demo is
untouched. Seventeen verified derivative chunks preserve 816/34,890 views (8,941,247 bytes),
no partial files and no E55 fit yet. Full inventory: `evidence/e55_pause_checkpoint.json`.
User must connect AC power; remote access cannot do that. Do not automatically resume heavy work
on the remaining battery. Once charging is verified, resume `experiments.e55_color extract`,
then `fit`, then `report` with the same frozen contract; do not re-freeze or discard completed
chunks. Keep `PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets`, `PYTHONPATH=ml:ml/src`,
offline HF/Transformers flags and OMP/OPENBLAS threads=2. This is not a scientific rejection of E55.

E54 finishes all six fixed fits. Against the identical native fold baseline, last-two-block
adaptation changes clean AI recall 66.39% -> 68.76%, REAL false-AI 19.50% -> 17.50%, balanced
accuracy 73.44% -> 75.63%; Q75: AI 65.11% -> 66.80%, REAL false-AI 18.76% -> 16.85%, BA
73.17% -> 74.98%. These are the same consumed TRAIN source-held-out parents, not a replacement
measurement of full E43's 94.3% and not E52. Relative acceptance fails: several supported AI
sources regress, and adjusted Q75 AI-preservation uncertainty includes loss. Absolute fold gates
also fail. Saved-artifact replay passes all 23,912 observations exactly, with zero changed decisions;
the separate acceptance receipt rejects both arms. No deployment or final-reserve opening is authorized.

### E55 — class-symmetric grayscale coverage, fixed exploratory ablation

E54 exploratory audit: 230/4,035 held-out REAL global crops are near-monochrome, with 229 in RR.
The native baseline wrongly flags 169/230 clean versus 618/3,805 other REAL crops; restricted
adaptation still flags 153/230 versus 553/3,805. Colour and source are confounded: this is a
coverage hypothesis, not proof that grayscale causes errors. Never use colour as an authenticity
rule, filter hard rows, or change labels.

Preregister before extraction/fitting: same 11,630 admitted TRAIN parents, same E54 source-fold
roles, all three existing FIT transports retained. Frozen pretrained DINOv2-S only, identical
3-crop/4-block/3,072-feature layout. Convert each exact uint8 crop with Pillow RGB -> L -> RGB;
extract a derivative feature for every existing view. This is class-identical processing, not new
independent data. No CAL/VALIDATION derivative is used for fitting; their original clean/Q75
features and the existing CAL threshold algorithm are unchanged.

Two fixed arms, three folds each: (1) duplicate-view control, and (2) grayscale-view augmentation.
Allocate 80% of each existing parent/source/class-balanced loss mass to original three views and
20% to either exact duplicate views or grayscale counterparts. Fix total loss mass to the same
base-FIT three-view count used by E53 native expansion, including in the weighted StandardScaler.
Use C=.01, lbfgs, max_iter=1000, seed=53 and convergence warnings as errors, exactly the native
linear recipe; no hyperparameter sweep or post-score tuning. Verify duplicate-control prediction
parity against the archived native baseline (<=5e-5, zero decisions changed), and reconstruct all
six saved heads before interpreting results. Preserve all AI rows and report per-source losses.

Compare with both original reference recipes and native baseline using the existing paired
publisher-bootstrap/AI-preservation guard and absolute gates; include duplicate-control comparison.
Passing is only eligibility for separately frozen full-data/CAL plus new DEV, never direct serving
or a final certificate. Failure preserves the current model. This exploratory follow-up is informed
by consumed TRAIN validation and cannot be called an independent replication. Grayscale augmentation
is an established transform ([official Torchvision documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomGrayscale.html));
that documentation provides no evidence it improves this detector. The deterministic 80/20 mixture
is our preregistered ablation, not a claim to reproduce a paper or Torchvision's random implementation.

Remote continuation: public data downloads are authorized, but skip manual sign-in/approval and
paid services. Do not download a familiar benchmark as a supposedly new training publisher:
Qwen-Image-Bench is already split among consumed adaptation and protected test roles in DATASETS.
MNW and HDR+ remain entirely unscored and protected. New TRAIN acquisition still requires a
licence/provenance/overlap-and-role contract; more bytes alone do not demonstrate improvement.

**Home execution authorized (2026-09-09):** the user returned, confirmed the disk is attached and
permitted dataset downloads. Resume with E54 below; public downloads still require a concrete
source/role/byte-cap manifest, and paid APIs are not authorized. Previous office verification:
72 saved heads reproduce 286,944 predictions exactly; 574 Python tests passed. Serving is unchanged.

### E54 — bounded representation adaptation (preregistered, execution underway)

Use the admitted E53 base+native population and coverage-v2 folds unchanged. Cache the identical
three clean/assigned-transport/Q75 crop views and their pinned frozen-DINO teacher features on
the external disk; no new final rows. Native rows enter matching-source FIT only. Compare two
fixed arms: continued **head-only** AdamW control versus **last-two-blocks + head** with a fixed
0.1 cosine feature-preservation loss to the pretrained DINO features. Both initialize from that
fold's hash-pinned E53 full/native head (never the full-data E51 probe head). This controls for
continued head optimization; it is not a claim that AdamW duplicates the original logistic penalty.

Fixed seed 54, two complete epochs, deterministic shuffled visitation of all FIT views, batch 8
views/24 crops, per-parent/source/class weights, no sampler dropping old AI. AdamW: backbone
LR=1e-6, head LR=1e-5, weight decay=.01, global gradient-norm clip=1.0; no schedule or early stopping
selected from validation. Backbone remains in evaluation mode to disable stochastic dropout while
retaining gradients for the last two blocks. Final epoch only; periodically save resumable model,
optimizer and exact next-batch index. First verify frozen-head parity and finite gradients. Any
data-integrity/numerical failure halts that run; it does not license unrecorded parameter changes.

Choose each final cut on the existing disjoint inner CAL only. Evaluate all folds/arms on the
same consumed TRAIN outer validation, original/Q75 separately. Report all failures and source
losses; require the existing REAL-improvement/AI-preservation guard against both original reference
recipes and the native baseline before advancing. Report the head-only comparator additionally.
No winner means preserve serving and diagnose, not silently relax the AI requirement. Only a
passing candidate may proceed to separately frozen full-data/CAL fitting and diagnostic regression,
then a genuinely independent E52 population. A successful training process is not a final pass.

Execution checkpoint: all 11,630 original-based parents/34,890 crop views are cached and bound;
all three head-only controls complete, restricted-backbone folds running serially. MNW 300 AI
and HDR+ 100 REAL bodies total 653,835,579 verified bytes; both blind overlap screens complete,
zero detector scores. No further bulk acquisition is needed while this hypothesis is tested.

Post-fit integrity/acceptance: separately verify saved checkpoint hashes and exact epoch/offset,
reconstruct the model and replay every archived outer-validation prediction. This is serialization
verification on already consumed rows, not another test. Require max score error <=5e-5 and zero
decision changes; fail closed otherwise. The report's relative `research_guard_passed` field is
not sufficient for advancement: all existing absolute fold gates must also pass. Freeze a separate
acceptance receipt that combines these checks before any full-data fit, new DEV or E52 opening.

Exploratory error audit (not a new model): four identity-ordered false positives in the first
RR fold include monochrome and visibly heavily processed photographs. This selected error sample
does not establish prevalence or causality. Quantify near-monochrome global model-input crops
(mean RGB channel range <=2/255), overall/source/class and FIT-versus-validation exposure, across
the already-bound E54 cache. Join completed fixed predictions only after all folds finish. No
image relabelling, row removal, threshold change, classifier colour shortcut or final-reserve access.
If the measured coverage supports it, a future separately frozen class-symmetric colour-processing
ablation can be considered; do not automatically start one from four illustrative errors.

Public test-reserve acquisition in parallel: Microsoft–Northwestern–WITNESS (MNW), pinned Git
revision `c93abf43e8157558a0e60aab7df4278b2c539253`, six text-to-image folders (GPTimage2,
Midjourney v8, Imagen4, Firefly v4, MAI image2, Flux 2 pro), 50 hash-ranked images each. Fetch
complete per-folder metadata, not the truncated recursive repository listing; verify Git LFS
pointer/blob hashes and freeze exact image SHA/size before transfer. Maximum 768 MiB image
transfer and 12 MiB per selected image; fail rather than silently replace an oversized selection.
Non-commercial evaluation only: no TRAIN, threshold calibration, edits/inpainting, or Module 2
mixing. The whole publisher is reserved; unknown shared prompts are not claimed independent.
After byte/decode/protected-overlap checks, keep it unscored until a candidate and evaluation
protocol are frozen. This AI-only reserve is not a balanced E52 final or an official pass certificate.

MNW model-blind admission screen: freeze the completed download report and both canonical
fingerprint databases, current E54 TRAIN contract, previous reserve closure and latest 920-parent
AI reserve identities. Verify that all current TRAIN originals are covered. Compare byte/RGB
identity plus the established radius-4 dHash/pHash heuristic; withhold every cross-protected match
and both endpoints of internal pairs without detector scores or replacement sampling. Unknown
prompt relations remain a whole-publisher limitation. Record the new reserve manifest/report in
every future training admission inventory; this screen does not certify semantic non-overlap.

Second bounded reserve: official Google HDR+ `20171106/results_20171023/` full-resolution final
JPEGs, not gallery thumbnails or RAW bursts. CC-BY-SA 4.0, camera-derived computational photographs
from older Nexus/Pixel devices, not generative images or a modern-phone universal sample. Exclude
all `synthetic_*` folders explicitly (20 present in the official listing); validate dated capture
names. Hash-rank 100 distinct burst parents from the complete public object listing before image
access; freeze GCS generation, MD5, bytes and source terms. Cap 768 MiB cumulative requested image
bytes/12 MiB per image; no full 765 GiB archive or RAW decoder dependency. Whole publisher stays
evaluation-only by project policy, never TRAIN/CAL or score-driven filtering. Session/day ids are
grouping proxies, not proven independent scenes. Verify decoding, device metadata availability and
all protected-overlap before any final consideration. A small one-publisher REAL reserve plus MNW
is still not the complete multi-source E52 final. Keep future TRAIN admission aware of both reserves.

**Authoritative decision:** no replacement model is accepted. Twelve fixed configurations were
evaluated in each of two declared TRAIN source-held-out protocols: 72 separately fitted fold heads.
Both protocols reject every candidate under the combined REAL-improvement/AI-preservation guard.
They reuse the same 5,978 outer-validation parents; they are not two independent finals and their
scores must not be compared with the old full-model 94.3% AI result as a before/after decline.
The served E20/optional R1b and full E43/E51 research artifacts remain unchanged. E52 is not passed.

Completed locally: 20,826 original-body audit, full reserve closure, 5,652 additional original-based
TRAIN parents /16,956 cached views, all 72 fold heads, source-transition/ROC/geometry diagnostics,
11,956-view exact-crop parity, fast-weight equivalence, and an eight-parent last-two-block training
resource probe. Zero new image/weight/dependency downloads; no API spending or GitHub push.

The strongest useful direction is **native input coverage plus restricted representation adaptation**,
not an arbitrary threshold shift or larger ensemble. Native replay improves AI recall but raises
REAL errors in the RR held-out publisher. Mean-only features reduce some REAL errors but lose
specific AI families. RR separation remains weak even at an evaluation-derived common FPR,
so calibration alone cannot fix the tested ranking. None of these observations certifies a winner.

Engineering gains are safe but limited: fast weights are bitwise-identical and ~34.75x faster for
that tiny preparation step; crop dedup has zero score/decision differences on all current TRAIN
clean/Q75 views, with ~10.84% less elapsed time in the separate small benchmark. Dedup remains
an opt-in research prototype, not a silent serving change. Restricted DINO adaptation is feasible
on MPS, but its four-step resource probe saved no candidate and establishes no accuracy gain.

### Next execution order — home handoff, no automatic download

1. **First use existing bytes.** Preserve the current artifacts and all failed results. Freeze a
   separate small adaptation study before training: local DINOv2-S, same 3-crop/4-block input and
   full 3,072-feature layout, last two blocks plus binary head only. Reinitialize from pinned
   pretrained weights and a head fitted only on that fold's FIT rows; do not reuse the resource
   probe's E51 head or weights as a cross-fold teacher. Compare with the frozen-backbone recipe
   on exactly the same source-held-out rows. The probe suggests batch 8 parents/24 crops is feasible;
   it does not settle full-epoch throughput or memory peak. Freeze LR groups, two-epoch ceiling,
   weighting, sampler, seed and fixed final-epoch rule in the next contract, before fitting.
2. **Remove processing shortcuts without losing AI diversity.** E51 TRAIN is 57.35% exact-224 REAL
   versus 18.17% AI; added native data reduces but does not erase this imbalance. Preserve all
   eligible AI replay. Use class-identical TRAIN processing, preserve originals, and group every
   source/scene/prompt derivative. Test matched processing/feature-preservation as a separately
   declared ablation, not several changes sold as a causal result. Never classify from resolution,
   file size or crop dispersion. Do not simply remove all texture features after pooled gains.
3. **Apply the same AI-preservation gate.** Check both transports and individual supported AI
   sources against both refitted references. Preserve the original no-loss requirement; do not
   substitute an 80% floor or retune on E49/IEEE/Datapoint. Only a passing TRAIN-selection candidate
   can be frozen for diagnostic regression; that regression still cannot tune it or certify E52.
4. **Download only after explicit home-network permission and a concrete coverage gap.** No bulk
   repetition of the already-audited 16,275 local candidates. Specify missing licensed native REAL
   publishers/devices and independently documented modern-AI prompt/generator families. Assign
   TRAIN/CAL/DEV/FINAL roles by independent groups before bytes, verify licence/provenance and
   existing-body overlap, then use a small byte-capped resumable pilot before any larger transfer.
   The transfer cap and exact source manifest must be frozen before the first request. API generation
   requires a separate spending budget; it is not included in permission to download public data.
5. **Keep E52 genuinely independent.** Existing consumed E49, IEEE/Datapoint DEV and these E53
   validation rows cannot be relabelled final. Unused rows from the same publisher are not unseen-
   source proof. If local independent coverage is insufficient, remain at the acquisition boundary
   until permission. Module 2 stays protected/planned until the Module 1 final gate is satisfied.

Office scope closure: native expansion used the existing three-view recipe, not a completed
native-expansion x order/scale 2x2. That extra native extraction and a real adaptation training
study are deferred, not silently marked done. No claim to have exhausted all ML methods or found
the global optimum. The completed bounded studies eliminate several unsupported shortcuts.

Earlier E51 evidence remains unchanged: IEEE/Datapoint DEV BA 92.85%/92.08%, REAL FPR
1.67%/1.97%, AI recall 87.38%/86.13%; consumed E49 BA 80.80%/75.70% with material AI loss.
Hidden IEEE device ids and 512px publisher processing still limit native/worst-device proof.

### E53 — original research plan and execution preregistrations (retained method context)

#### Authorized office execution slice — 2026-09-09, before new E53 scores

The user approved an approximately 90-minute offline preparation/optimization session. Preserve
serving and do not download. Start with a bounded six-arm diagnostic on the **already admitted E51
TRAIN population**, not expanded C3 before admission: full 3072 versus mean-only 1536 features,
crossed with old two views, E51 three views, and the preregistered clean plus two order/scale views.
This separates crop-dispersion features and transport effects with cached features where possible.
Use three whole-publisher/source-component outer folds and disjoint inner CAL; refit every head
and scaler. No historical fitted classifier is an out-of-fold comparator. Freeze population,
folds, C=0.01, view recipes, all six arms and code before scores. Report all arms, including failures.
The mean-only ablation tests whether crop dispersion is a brittle processing cue; it is not a
claim that texture features are bad. No test-driven feature selection or model promotion.

Historical reuse clarification: E51 already admitted explicitly consumed historical replay, e.g.
`e36_cal_consumed`. This diagnostic inherits that documented TRAIN role; it is not fresh evidence
on the original E36 CAL. E51 CAL, all later DEV/final/reserves and every other currently protected
role remain excluded. Recheck canonical overlap with E51 CAL and its now-consumed DEV, bind the
previous full admission audits, and group internal near-duplicates. Unknown shared prompts are
held together by source; RR topics are one publisher, not seven independent AI generators.
Further C3 expansion still requires original-image/protected-role audit before fitting. Pending
work or a failed/inconclusive research guard must be reported as such at the office handoff.

Additional safe engineering experiment: benchmark exact duplicate-crop elimination on two fixed,
hash-ranked TRAIN parents/source, clean/Q75, one warmup and three alternating timed passes. Restore
the original crop order before feature aggregation. Require max feature/score difference <=0.00005
and zero decision flips on that sample; report the limited scope. Keep this optimization opt-in
and out of serving/frozen E53 extraction until broader equivalence is established. No accuracy
improvement is inferred from faster inference.

Conditional native-replay expansion (freeze before its first feature/model score): once the
original-body inventory completes, take at most 1,000 eligible parents per REAL source and 500
per AI source from C3 TRAIN, preserving source/group diversity and excluding all detected internal
duplicate pairs as well as protected matches. This adds at most 6,000 parents to E51's 5,978.
Use existing per-source research licences; CF remains non-commercial/per-model-terms restricted.
Extract full original-based clean/old-transport/Q75 features, not the old 224px export. Fit exactly
two additional arms (full and mean-only, C=0.01), retaining all eligible E51 AI replay. Each new
source inherits the matching publisher's already-frozen fold role; **only new FIT rows** are added.
Existing inner CAL and outer validation parents, pixels, features and cuts-selection method remain
unchanged, so this isolates adding native training data. Refuse unknown/cross-component source
assignments or failed native audit; do not change folds to make expansion possible. Compare on the
same TRAIN-derived outer validation rows with both fixed reference recipes. No promotion or E52
claim follows from this study. If the office time budget ends, leave a resumable preparation checkpoint.

Regularization control: expansion sample weights must sum to the original E51 three-view FIT
observation count per fold, keeping effective C comparable while adding parents. The initial
old-two-view versus three-view ablation inherits historical mean-one weights, so its total loss
mass changes with view count; report this confound instead of attributing every difference solely
to augmentation. It does not affect the honesty of its held-out predictions, but limits causality.

Engineering preparation: replace repeated whole-array parent masks with counter-based weight
construction in a new reusable helper. Benchmark all E51 TRAIN metadata with unequal source/parent
counts; require bitwise-identical weights and reject conflicting parent labels/sources. Keep
frozen historical/E53 fit implementations unchanged. This optimizes preparation cost, not accuracy.

Extend crop-dedup numerical verification to all 5,978 admitted TRAIN parents in clean/Q75
(11,956 views), against hash-pinned E51 cached features and the same research head/cut. Fixed
5e-5 feature/score tolerances and zero allowed decision flips; preserve failures and resumable
chunks. This is numerical regression, not a new accuracy test or permission to change serving.

Last bounded head controls, before their scores: four additional arms on unchanged E51 three-view
TRAIN, the same folds/CAL/validation and sample-weight mass. Cross full/mean features with either
stronger regularization C=0.001 or row-L2 normalization at C=0.01; no extra C sweep. Use the
bitwise-verified fast weight helper. Report all four even on failure, with the same AI-preservation
guard; these are TRAIN-selection hypotheses, not independent confirmation of a selected winner.

Metadata-only split coverage refinement, preserved as a separate v2 protocol: initial folds leave
E36 REAL and E32 shared FLUX/Qwen components in CAL/validation only, never FIT. Keep every outer
validation assignment unchanged, but jointly choose inner CAL components so each source component
appears in FIT at least once across the three folds. Minimize the same class-wise 20% CAL target
subject to both-class minimums and whole-component separation. Freeze this rule before v2 scores;
repeat the same twelve declared arms, with no new hyperparameter search. Preserve v1 results and
compare arms within v2, not v2 scores against v1 as an independent performance gain. This is a
data-use efficiency study on consumed TRAIN validation, not a replacement final or relaxed gate.

Final office preparation: after all twelve arms in both protocols, audit every frozen prediction
for source-wise gains/losses and per-fold ROC operating points without choosing new cuts. A bounded
resource-only probe may use eight hash-ranked existing TRAIN parents (four/class): copy the local
DINOv2-S and E51 head into memory, verify the differentiable mean/std implementation, unfreeze only
the last two blocks plus head, and run one warmup plus three timed AdamW steps at fixed LR=1e-5.
Check finite gradients, frozen-parameter identity and MPS memory. Do not save adapted weights,
report training accuracy, open CAL/test or infer quality from this feasibility probe. A real
adaptation study still needs its own fit/CAL contract, budget and independent acceptance decision.

**Original planning-only scope, superseded by the authorized office slice above:** inspect existing
assets, research primary publications and plan first. No new training,
model scoring, serving change, image/weight/dependency download or GitHub push in that planning
update. E52 remains the independent final gate; E53 is the successor research experiment, not a
renaming of consumed tests. Success is better authentic-photo recognition without buying it by
missing more AI. This is a measurable acceptance condition, not a promised outcome.

#### What the evidence actually says

- E49 paired original results: old E43 catches 94.3% AI but falsely accuses 39.1% REAL; E51-A
  catches 77.5% AI and falsely accuses 15.9% REAL. Its higher balanced accuracy alone is inadequate.
  AUC also falls, so moving the cutoff cannot repair the entire ranking problem.
- E51's 92.85%/92.08% BA on IEEE/Datapoint is encouraging evidence on another population, not
  an improvement measured against E43 on those same images. That DEV is now consumed.
- E43 fitted 8,844 parents/19,648 views; E51 fitted 5,978 parents/17,934 TRAIN views. Both use
  weighted standardization and C=0.01 logistic regression over frozen DINO features, but populations,
  transports and calibration differ. We have not isolated which change caused the AI regression.
- Class/source/parent balancing, intermediate DINO features, texture crops and Q75 already exist.
  Do not advertise them as new optimizations. Residual B did not justify its complexity. Earlier
  DDA/fusion experiments expose source-specific failures; blindly adding experts is not the default.

#### Primary research and the limited lesson we can transfer

- [B-Free, CVPR 2025](https://grip-unina.github.io/B-Free/): align REAL/AI content and processing
  so a classifier cannot win through dataset shortcuts. Locally, audit geometry/codec/content
  imbalance and use class-identical transforms. JPEG augmentation alone is **not** B-Free's
  semantically matched diffusion reconstruction method. Do not recycle protected DDA test pairs
  or start paid/local image generation to claim a reproduction.
- [Community Forensics, CVPR 2025](https://jespark.net/projects/2024/community_forensics/):
  generator diversity helps generalization. Preserve broad eligible AI families while adding REAL
  cameras; count independent generator/prompt groups, not just files. Its full 4,803-model release
  is not our locally available Small subset, and its FFHQ REAL half is not broad phone coverage.
- [NTIRE 2026 report](https://arxiv.org/html/2604.11487v1): realistic degradation and unseen
  generators matter. Test a small compression/resize-order variant on both labels and retain clean
  images. The leading MICV recipe uses multiple DINOv3 models and 32 A100 GPUs; copying its scale
  is not an offline laptop plan. Challenge AUC is not a guarantee about false accusations at our cut.
- [Effort, ICML 2025](https://proceedings.mlr.press/v267/yan25b.html): preserve pretrained
  information while adapting the detector. Consider restricted adaptation only after the cheap
  head/data experiments. Ordinary LoRA is not Effort's orthogonal SVD decomposition, and published
  success on other benchmarks does not establish a gain on our data.

These are evidence-backed research directions, not a claim that one method is universally best.

#### Phase 1 — establish what existing bytes are actually eligible

- [x] Recheck local C3 manifest SHA and metadata availability. All 18,154 historical TRAIN locators
  resolve: 9,073 REAL and 9,081 AI, nine sources. Loose files total 42,135,059,473 bytes, excluding
  image cells in 35 checked Parquet shards. This is availability, **not** fresh pixel validation or
  permission to reuse every row. See `evidence/e53_local_inventory_plan.json` and `DATASETS.md`.
- [ ] Join C3 TRAIN, E51 TRAIN and later role/admission/reserve histories by original identity,
  canonical pixels and duplicate groups. Later CAL/DEV/test/reserve protection overrides old TRAIN
  eligibility; exclude the whole component. Preserve the original manifests. Verify source terms,
  label convention (project 0=REAL, 1=AI), file hash, decoder and device/prompt provenance.
- [ ] Prioritize native CSAFE/FODB/VISION plus already-admitted authentic training sources, alongside
  eligible GPT/FLUX/Qwen/Nano Banana and diverse CF generators. The exact accepted counts must be
  reported after the audit, not inferred from historical counts. Keep unlicensed/unattributed
  `theminji` and `34data` repacks out; Module 2 manipulation data is not automatically fully-AI data.
- [ ] Bound the first experiment to at most 16,000 admitted parents (at most 8,000 per class), using
  deterministic group/source quotas, not model-score selection. Retain eligible old AI diversity
  as well as native REAL diversity. Keep all original resolutions; create views only from TRAIN.
  Hash the population, exclusions, source counts and limitations before feature extraction.

#### Phase 2 — separate data effects from augmentation effects

- [ ] Freeze common group-held-out TRAIN validation folds before fitting. Use three source-aware
  outer folds with group-disjoint inner calibration where feasible. Keep all transforms, scenes,
  shared prompts and duplicate components together; unknown camera/prompt provenance requires
  holding out the whole publisher. FODB's crossed cameras/scenes cannot prove unseen-camera
  transfer through scene splitting. If the group graph cannot support valid folds with both labels,
  stop and revise the split before scores, rather than falling back to random image splitting.
- [ ] Compare a bounded 2x2 design: E51-eligible baseline pool versus expanded eligible pool,
  each with existing three views versus a fixed three-view transport variant. Use the same
  frozen DINOv2-S feature architecture, weighted scaler/head, C=0.01 and outer validation rows.
  Baseline means the currently eligible subset of that recipe, not unrestricted historical reuse.
- [ ] Variant: retain clean input; derive two additional views with resize-then-JPEG and
  JPEG-then-resize. Assign scale from {0.5, 0.75, 1.0} and JPEG quality from {75, 90} by a fixed
  parent hash and class-identical probabilities, before unchanged model crop/cap logic. Preserve
  parent total weight across views. Freeze implementation details, hash seed and minimum-size
  policy before extraction. No aggressive erasing, blur, MixUp, or extra inference ensemble yet.
- [ ] Refit every scaler/head inside each training fold from scratch. Frozen E43/E51 classifiers
  have seen historical TRAIN rows and therefore cannot be honest out-of-fold comparators or
  teachers on those rows. Use fold-refitted baseline recipes; no teacher-generated labels.
  Cached pure frozen-backbone embeddings are reusable only with identical pinned input semantics.
- [ ] Record pooled, macro and worst-source/device recall/FPR, AUC, TPR@FPR10, coverage, latency
  and memory. Evaluate original/Q75 separately. New transforms receive separate stress columns,
  not a silently changed final distribution. Deterministic head seed repeats are not independent
  evidence; the held-out source groups provide the useful variation.

#### Phase 3 — reject apparent improvements that lose AI

- [ ] Choose each cutoff using inner CAL only under the existing REAL error budgets, then freeze
  it. Compare fixed-cut predictions on identical held-out parents. Also report recall at a common
  FPR=10% as a ranking diagnostic; do not transplant a held-out ROC-derived cut into the product.
- [ ] To advance, require lower REAL false-AI and no negative AI recall point delta versus the
  fold-refitted baseline in either original/Q75, pooled and each declared AI source with adequate
  independent support. No-source-loss is additional to the existing absolute gates, not replaced
  by the much weaker rule “AI recall is still above 80%”. Define support before scores; missing
  generator identity or too-small groups remain unverified, not assumed passing.
- [ ] For a statistical preservation claim on the tested population, require paired cluster-bootstrap
  lower 95% bound for pooled AI recall delta >=0 in both transports, plus upper 95% bound for
  REAL FPR delta <0 for an improvement claim. Resample scene/prompt components, preserving paired
  predictions; use simultaneous coverage across the four primary comparisons (Bonferroni-adjusted
  intervals). Per-group point guards are not per-group statistical guarantees. Wide intervals mean
  inconclusive, not “no significant drop, therefore equivalent”. No hidden loss tolerance.
- [ ] Select at most one eligible research candidate; ties favor fewer features/lower latency.
  No passing candidate means retain current serving and archive the failure. Selection-fold
  intervals are development evidence, not final confirmation after multiple-candidate selection.

#### Phase 4 — conditional representation improvement, not model churn

- [ ] Only if Phase 2 cannot improve the guarded frontier, preregister one small adaptation study
  using already-local DINOv2-S weights: restrict updates to the last blocks/low-rank adapters,
  preserve clean/AI replay, and compare against the frozen-backbone reference. First measure
  MPS memory/throughput on TRAIN-only data. Freeze adapter layout, learning rate, regularization,
  seeds, training budget and stopping rule before fitting; do not launch an open hyperparameter sweep.
  Feature-preservation/transport consistency are hypotheses to isolate, not automatic improvements.
- [ ] Keep high-capacity DINOv3 downloads, large ensembles, API-generated images and full B-Free
  reproduction deferred. Do not rescore consumed tests to choose among successive adapters.

#### Phase 5 — diagnostic regression, then genuinely independent E52

- [ ] After candidate/CAL freeze, compare it with both frozen E43 and E51-A on existing test
  populations for regression reporting. Preserve hashes, cuts and all rows. Known E49 AI loss
  must not disappear into an aggregate average. These reruns are consumed diagnostics; they
  cannot choose a new winner or tune a threshold. A failure blocks promotion.
- [ ] Inventory truly unused local native-camera and modern-generator sources before any final
  scoring. Protect shared scenes/prompts and later-role overlaps; hold out publishers where needed.
  Unused rows from a familiar source are not new-source proof. Bind candidate, comparator artifacts,
  independent-unit quotas, original/Q75, all existing absolute gates and the AI-preservation test
  before scores. Compare actual frozen E43/E51-A and the candidate on the same final population;
  require preservation against both references, not unrelated percentages from old datasets.
- [ ] If sufficient independent licensed local data is absent, stop at a documented acquisition
  specification, with zero downloads. Do not rename IEEE/Datapoint DEV or old CAL as E52.
- [ ] Module 2 stays protected/planned until Module 1 satisfies the agreed final. Future Module 2
  findings may motivate new TRAIN-only hypotheses, never leakage of its held-out masks/images.
  Append methods, failed attempts and results to HISTORY/EXPERIMENTS; serving changes require the
  full gate, not a successful training run or higher headline accuracy.

### Completed E51 execution checkpoints (retained method context)

The older E41–E50 sections below are historical execution context, not a request to repeat their
consumed finals. The completed order was: canonical protected/reserve checks, admissible E51
TRAIN/CAL, features, fixed A/B fitting, CAL selection, then evaluation without further tuning.

Admission complete: protected `2f070e7d...c1e54` and reserve `2c2cdb07...9ad36` checks have
zero matched pairs/identity hits. Frozen admission `c406edb6...7a39b` retains all 5,978 TRAIN
and 1,560 CAL parents, with zero exclusions. The 21,054-view feature and fixed A/B stages completed.

TRAIN/CAL complete: features `55f92db2...82d0a`; both fixed candidates pass CAL, so the
predeclared simpler **A** wins, artifact `60d56c0b...b6b39`, AI/REAL cut 0.3316505551338196.
CAL original BA 97.00%, REAL FPR 2.67%, AI recall 96.67%; Q75 97.56% /1.83% /96.94%.
This is CAL evidence only. Frozen old-test regression, score-blind fresh DEV realization and
exactly one evaluation of A on admitted IEEE/Datapoint DEVELOPMENT subsequently completed.

- Before scores, conservatively exclude a whole E51 parent (including every paired child) when the
  fixed protected-identity/perceptual checks flag it. No label changes, no score-dependent replacement,
  no downloads. Preserve the original frozen manifests. Abort rather than continue if exclusions
  exceed 5% of either role, any CAL device falls below 30 originals, or any AI CAL source below 15.
- Freeze three equal-weight TRAIN views per parent: clean, its existing E42 deterministic transport,
  and full-image Q75 before the unchanged E42 cap/crops. CAL uses its already-bound original/Q75
  files. Reuse only hash-pinned frozen E42 DINO features whose exact parent/condition/label matches;
  no previous classifier weights are reused. Cache new features in restartable, input-bound chunks.
- Fit only fixed A/B heads (C=0.01), train-only weighted standardization, class/source/parent balance,
  seeds 42/43/44. Report that deterministic lbfgs seed repetitions are not independent data trials.
  Choose the smallest threshold satisfying both transports' 10% pooled/20% worst-device REAL FPR
  limits. Select a REAL/uncertain band on CAL only, subject to 95% covered accuracy; unchanged AI
  recall/AUC/coverage gates still decide whether either candidate is eligible. Prefer A if both pass.
- Freeze artifacts before any old-test rerun. An E49 paired comparison is diagnostic/regression,
  never a fresh final or a source of new thresholds. Keep the serving model unchanged; fresh
  DEVELOPMENT/E52 policy remains as specified below. No image or model-weight download is allowed.
- Regression implementation is fixed before fitting: evaluate both predeclared A/B artifacts,
  regardless of CAL eligibility, at their frozen CAL cuts on all 4,000 consumed E49 observations.
  Reproduce the old E43 raw scores within max absolute error 0.00005 as a preprocessing guard.
  Report per-source errors and paired 2,000-replicate, source-stratified parent-bootstrap deltas;
  never use these diagnostics to select or refit either candidate. A CAL failure keeps fresh DEV closed.
- Fresh DEV realization is fixed before opening Datapoint image columns: verify all seven local
  shard hashes, decode exactly the 920 reserved AI bodies, preserve all 2,640 IEEE files and pair
  original/Q75. Compare both transports against canonical protected bodies, E51 TRAIN/CAL and local
  reserves, plus cross-parent internal duplicate screens. Any IEEE match aborts admission; an AI
  match removes its entire shared five-model prompt group. Take the first 20 valid ranked prompts
  per category (eight categories) for 800 AI; depleted quotas abort, with no score-driven replacement.
  IEEE camera ids are hidden and files are publisher 512px images: two transport groups cannot
  establish worst-device performance or native-resolution transfer. Keep this limitation explicit.
- DEV scoring: lock the selected A artifact, CAL cut and admitted 6,880-row manifest before its
  first score. Report each original/Q75 column with exact counts, per-source rates and 2,000
  bootstrap intervals grouping all five AI images sharing a prompt. No second candidate or refit.
  Even a pass of observable transport-group checks is not proof of the hidden worst-device check.

Old-test regression completed (consumed E49, not new final): A original BA 77.60%→80.80%,
REAL FPR 39.10%→15.90%, AI recall 94.30%→77.50%; Q75 BA 73.25%→75.70%, REAL FPR
49.00%→30.70%, AI recall 95.50%→82.10%. AUC fell 0.902425→0.884358 and
0.868850→0.843796. This is a real-safety/AI-recall tradeoff, not stronger universal separation.
Keep A as the CAL-selected research artifact and current serving unchanged; do not promote or
choose B using this regression. Fresh DEV subsequently completed; no E49 threshold sweep was run.

DEV realization v1 stopped before scores on five internal IEEE near-duplicate pairs (no IEEE
cross-role/protected match). Score-blind review inspected all ten bodies: four visibly same-scene
variant pairs, one ambiguous smooth-grey pair. Preserve the failed audit `975569d3...258e8` and
amend grouping, not model/labels/rows: retain all 2,640 IEEE images, conservatively use 2,635 detected
scene clusters for uncertainty, reverify identical old match sets, retain the AI shared-prompt
reserve policy and all 6,880 planned observations. Protected overlap still cannot be waived.
No DEV score existed when this amendment was decided; candidate A and every cut/gate stay fixed.

## Current execution slice — E41 external proof, then E42 only if needed (2026-08-28)

The goal is **success**, defined as a detector that survives independent, source-aware tests while
keeping both authentic-photo false accusations and modern-AI misses within the frozen budgets. A
high internal AUC, a visually convincing demo or a threshold selected on an owner gallery is not
success. E41 is already a runnable frozen candidate; therefore the shortest honest route is to test
it before paying for another architecture or training run.

### F0 — freeze what current science changes and what it does not

- [x] Re-audit E1–E41 and the 2025–2026 primary literature before new bytes. The strongest repeated
      finding is data/pipeline alignment, not a magic backbone: ITW-SM reaches 0.9823 AUC with a
      DINOv2-L RINE variant only after in-the-wild training, texture-aware crops and realistic
      augmentations; SPAI reaches 0.9810 with spectral any-resolution processing; NTIRE 2026's top
      robust AUC is 0.9723 using very large DINOv3 ensembles, millions of images and hierarchical
      degradation. A 2026 out-of-box study finds no universal winner and only 0.780 mean accuracy
      even for its leading released ensemble.
- [x] Keep the internship-success gate unchanged: AUC >=0.90, TPR@FPR10 >=0.80, EER <=0.15,
      balanced accuracy >=0.85, REAL macro/worst FP <=10%/20%, AI macro/worst recall >=80%/60%
      and complete declared coverage. NTIRE's 0.97 robust AUC is top-challenge territory, not a
      defensible minimum for a local prototype. NIST defines blind metrics but no universal
      certification mark.
- [x] Reject immediate ensemble/model churn. E41 already has strong current-family separation and
      a broad-real threshold; testing it is cheaper and more informative than choosing a new model
      from the same consumed scores.

### F1 — two open external gates before any E42 training

- [x] Acquire the official B-Free viral-image URL registry as `E41_WILD_STRESS`, preserving its
      34 source events (17 REAL, 17 AI) and every surviving web version as a child of that event.
      Verify the published MD5 for each download, count every dead/changed URL as coverage failure,
      and never let heavily reposted events dominate metrics. This is a difficult web-propagation
      stress test, not a modern-generator final: its dated rows end in 2024 and its effective
      independent N is 34.
      Verified 811/1,111 rows / 162,894,149 B while retaining all 34 events. The decontaminated
      unscored manifest keeps all 811 rows: zero earlier-role overlap and zero cross-event duplicate
      group against 14 protected prior manifests.
- [ ] Reassign the still-unopened CC BY 4.0 `RRDataset_test.tar.gz` only as
      `E41_EXTERNAL_ROBUSTNESS`, bound to the frozen E41 artifact/threshold before bytes. Resume the
      exact 20,117,869,400-byte archive, reproduce MD5 `13c3ff3d61986170cc0c8cf76a35cd4b`,
      inventory/extract safely and score once. Prior RR validation exposure weakens collection-level
      independence, so report this as robustness transfer, never as the sole final claim.
      **Deferred without transfer:** B-Free already rejected E41, so spending 20.12 GB to seek a
      second E41 verdict cannot promote it. Preserve RR test unopened for the eventual E42 winner.
- [ ] Freeze both unscored manifests and protected-role overlap audits before model access. E41 may
      be called an externally validated prototype only if the original global gates pass on the RR
      clean-parent population, the declared robust/transmission columns remain above working AUC
      0.85 and balanced accuracy 0.80, and the parent-weighted B-Free stress result has balanced
      accuracy >=0.80 with no class below 0.75 recall. Report confidence intervals and limitations;
      no retry or threshold change is allowed.
      The B-Free half is now frozen before score: E41 artifact/threshold, the 811-row manifest hash,
      one score per surviving web version, equal weighting of the 34 source events, 10,000-event
      bootstrap confidence intervals and the 0.80/0.75/0.75 gate are executable and covered by
      focused tests. Repost volume is diagnostic only and cannot dominate the verdict.
      **B-Free result: failed.** AI parent recall is 100%, but REAL parent recall is only 18.41%
      and parent-weighted balanced accuracy is 59.20%; the candidate's broad-real threshold did not
      transfer to viral/web-propagated authentic images. No threshold retry is allowed.

### F2 — success branch and controlled failure branch

- [ ] **If E41 passes:** package the same artifact for the research API/web path, keep the current
      uncertainty wording, run all Python/web/registry checks and retain ITW-SM/NIST as stronger
      later external confirmation. Do not claim NIST approval or universal detection.
- [x] **E41 failed:** preserve the failed external scores and do not use RR test or B-Free viral
      rows for E42 training or threshold selection. Open exactly one E42 line based on the research:
      DINOv2 global semantics plus deterministic texture-rich multi-crop aggregation, symmetric
      JPEG/WebP/resize/blur augmentation for both labels, and source-held-out calibration. Compare
      the smallest adequate DINOv2-S implementation with one DINOv2-L intermediate-block candidate;
      pick by consumed DEVELOPMENT only, not by the external tests.
- [ ] E42 may train on the existing licensed TRAIN pools, RR's official **train** split and all
      explicitly consumed adaptation populations. It must retain source caps, parent grouping,
      label `0=REAL, 1=AI`, equal transform probabilities and exact/perceptual decontamination.
      The untouched ITW-SM 10,000-row social-media benchmark becomes E42's preferred independent
      final after the user authenticates and accepts its non-commercial terms; no local Hugging
      Face identity or approval exists today, so its 3.57 GB cannot be fetched silently.

### F2.1 — executable E42 recovery contract (frozen before extraction/features)

- [x] Bind exactly 4,638 base-training parents: the fixed 1,067-row E32 TRAIN replay, all 1,071
      consumed E36 CAL rows and RRDataset's 2,500 official train rows. Bind 2,250 consumed
      source-held-out DEVELOPMENT parents: 640 E36 former-final rows, 440 E39 rows, 960 IPN phone
      originals and 206 unique parents in the declared 210-file owner gallery. B-Free viral and RR test are prohibited
      from fitting, threshold selection and model choice.
- [x] Safely extract only RR `train/{real,ai}` from the already MD5-verified 2.16 GB archive. Decode,
      hash and count all 2,500 rows; preserve seven AI topic/scenario groups and one explicitly
      pooled REAL source. Freeze the combined parent manifest and exact/dHash overlap audit before
      any E42 feature extraction.
      Complete: RR train is 1,860,689,134 decoded image bytes; the combined frozen manifest has
      6,884 unique parents across 63 sources, 4,638 TRAIN +2,246 DEVELOPMENT, with zero cross-role
      exact SHA-256 or exact dHash group. Manifest SHA-256 is `15124d93...3e238`.
- [x] Implement one fixed RINE-inspired representation ladder: normalized CLS tokens from four
      intermediate DINOv2 blocks, aggregated over one global crop plus two deterministic highest-
      texture native crops. Compare DINOv2-S and DINOv2-L only; DINO-L reuses the hash-pinned pure
      backbone tensors already present inside the official Apache-2.0 DDA checkpoint, avoiding a
      redundant network download.
      DINOv2-S feature extraction is complete for all 20,506 planned views /61,518 crops, shape
      20,506x3,072, 235,605,776 bytes and SHA-256 `452fec98...69ac5a`. Evaluate S first; because
      the fixed rule prefers the smallest full pass, DINOv2-L can change selection only if S fails.
- [x] Give every training parent one clean view plus one deterministic, class-symmetric transport
      view chosen from JPEG, WebP, resize+JPEG and mild blur. Evaluate every DEVELOPMENT parent as
      clean plus all four transports. Fit one source-balanced logistic head per backbone at fixed
      C=0.01; use source-held-out clean OOF scores for the threshold and require the unchanged cut
      to pass the frozen clean success gate plus robust AUC >=0.85 / balanced accuracy >=0.80.
      E42-S passes every check at threshold 0.660046: clean AUC 0.99287, balanced 0.95477, REAL
      macro/worst FP 1.23%/20%, AI macro/worst recall 92.69%/75%; robust combined AUC 0.99338,
      balanced 0.93923, REAL macro/worst FP 0.84%/13.5%, AI macro/worst recall 88.99%/68.13%.
- [x] Select DINOv2-S immediately if it fully passes; otherwise evaluate the only remaining fixed
      DINOv2-L candidate and select it only on a full pass. If
      neither passes, stop without another backbone/threshold sweep. If one passes, refit once on
      all consumed training+development parents, freeze a research candidate, then and only then
      transfer/inventory the unopened RR test for a one-shot external result.
      E42-S is the full pass and therefore wins without running L. The 87,977-byte candidate SHA-256
      is `6768466a...9062e7`; state remains research-only awaiting RR external evidence.

### F2.2 — one-shot RR external gate

- [x] Bind E42-S artifact `6768466a...9062e7`, threshold 0.660046 and the exact 20,117,869,400-byte
      CC BY 4.0 RR test archive /MD5 `13c3ff3d...cd4b` before transfer. The machine-readable contract
      is `evidence/e42_rr_final_contract.json`; B-Free and RR labels cannot alter the candidate.
- [x] Resume the archive to LaCie with >=100 GiB reserve, verify exact size+MD5, inventory every tar
      member and extract safely. Then freeze decoded counts, parent/condition mapping and protected-
      role exact/dHash audit without loading E42.
      The fail-closed implementation is now committed before archive completion: test extraction
      accepts only `original|transfer|redigital` and explicit `real|ai` paths; manifesting decodes
      and hashes every row, normalizes derivative filenames to parent IDs, audits E42/B-Free
      overlap and writes a zero-score receipt. Transfer, MD5, inventory and safe extraction are now
      complete: 50,999 official rows and 20,354,797,721 expanded image bytes. The public archive
      has 8,500 rows per condition/class except redigital REAL=8,499, rather than the paper's
      described 10,000+10,000 population. The first full decode audit correctly stopped on 35
      same-label exact duplicate components, 13 prior exact REAL overlaps and one prior-dHash AI
      parent. Before any score, exclude every protected parent across all conditions and keep one
      lexical parent per clean exact component; freeze/report official coverage rather than hiding
      the removals.
      Complete: 47 parents /141 rows excluded before score, leaving 50,858 images from 16,953
      parents and 99.7235% official-row coverage. Manifest SHA-256 is `b2d815af...30c98`; the second
      immutable score contract binds that manifest, candidate and threshold for exactly one run.
- [x] Score each declared test file once. The original/clean population must pass all internship
      success gates; every sufficiently populated transmission/redigitization condition must retain
      AUC >=0.85 and balanced accuracy >=0.80, with 100% coverage. A pass promotes E42-S to the
      research API/web; a miss is final for this candidate and cannot trigger threshold repair.
      Scoring code is also frozen ahead of the manifest: the final manifest hash and candidate hash
      must be rebound in a second machine contract and committed before the DINO model can load.
      **Completed once; failed without retry.** All 50,858 rows scored with 100% coverage. Original
      AUC is 0.94448, TPR@FPR10 0.85139 and AI recall 93.54%, but balanced accuracy is 0.84634 and
      REAL FP is 24.27%. Transfer passes its AUC/balanced gate at 0.92582/0.83993. Redigital AUC
      passes at 0.85629 but balanced accuracy is 0.78756, below 0.80. E42-S therefore remains a
      rejected research candidate and cannot replace the served model.

### F2.3 — E43 success route after the honest E42 miss

- [x] Diagnose the completed RR score stream without changing E42. This is not a threshold-only
      defect: the best condition-specific redigital threshold reaches only 0.78943 balanced
      accuracy, and no single threshold satisfies the frozen original/transfer/redigital gates.
      The next candidate must change representation and realistic transport coverage.
- [ ] Formally consume RR test only as `E43_DIAGNOSTIC_DEVELOPMENT`; it can never be independent
      FINAL again. Freeze E43 before fitting: compare the existing DINOv2-S path with exactly one
      DINOv2-L intermediate-feature arm, add class-symmetric screen/recapture and stronger social-
      transport views, retain parent/source grouping and optimize no threshold on the future final.
- [ ] Secure a genuinely untouched final before spending the E43 training run. Preferred route is
      ITW-SM after the user authenticates to Hugging Face and accepts its manual non-commercial
      terms; NIST Image-D remains the stronger registered blind route. Until one is available, an
      E43 development improvement may be measured but cannot honestly be called project success.
      The student accepted the terms and authenticated locally on 2026-09-02, but the first content
      request returned `awaiting manual author review`; repository metadata visibility is not file
      access. Before image transfer, bind gated
      repository revision `3060094fb576669927134193de3f517d7e64af86`: exactly 10,004 files /
      3,573,691,324 bytes, including 5,000 `0_real` and 5,000 `1_fake` images. Download only to
      LaCie with a 100 GiB reserve and resumable Hugging Face local-dir state; a receipt is forbidden
      until every pinned file and byte is present. Acquisition creates zero scores.
      **2026-09-03 recheck:** authenticated content preflight still returns HTTP 403 `awaiting a
      review from the repo authors`; zero payload file and no receipt exist, so this item remains
      blocked on the dataset authors rather than local authentication.
      **Plan B initiated on 2026-09-02:** the official NIST GenAI Image portal still exposes
      participant sign-in/registration through Login.gov, but the published Image-D round-3
      schedule released D-Testset-3 on 2026-02-23 and closed outputs on 2026-04-03. The next step is
      user-controlled Login.gov authentication, followed by a read-only check for late/new-round
      registration and the required data agreement. Do not claim availability, download data or
      submit a system until the portal confirms an active Image-D participation path.
      **Authenticated portal finding:** Login.gov succeeded, but NIST permits individuals to
      participate only on behalf of a legally registered/incorporated organization; foreign
      organizations may apply and can require IAAO approval. The account currently has no NIST
      `site`, so track registration, licence and downloads remain locked. Complete the truthful
      profile only after the user supplies the exact official university/organization name and
      confirms authority to register under it. Keep every NIST download <=4 GB total unless the
      user later changes that cap.
- [ ] Run the chosen E43 candidate exactly once on the newly bound final population. Only a pass of
      the same class-balanced, source-aware gates permits API/web promotion; otherwise preserve the
      miss and stop rather than retuning on final labels.

### F2.4 — E43 local development while ITW-SM approval is pending

- [x] Re-read the prior fusion record before proposing a new architecture. E8/E9 and E31 already
      showed that fixed or stacked DINO/68-feature fusion recovers too few AI misses for the added
      authentic-photo false positives. Do not repeat that branch. E42 RR also proves threshold-only
      repair is impossible; change the learned boundary using transport-aligned data first.
- [x] Freeze a score-blind, parent-linked RR adaptation population from the now-consumed external
      set: only parents with original+transfer+redigital versions; 1,960 REAL from the pooled source
      and 280 AI from each of seven scenario sources (1,960 AI). Use independent SHA-256 selection
      and role keys to assign exactly 1,960 TRAIN, 980 CAL and 980 DEVELOPMENT parents, preserving
      all three conditions (11,760 rows). No E42 score may influence selection or role.
      Complete without reading a score: 3,920 parents /11,760 rows, roles exactly 1,960/980/980,
      every condition 3,920 and every role class-balanced. Detailed manifest SHA-256 is
      `29dd9b56...4b16`; compact tracked evidence records zero scores and zero image copies.
- [x] Reuse the already frozen E42-S representation and cached earlier E42 features. Extract its
      exact global+two-texture-crop /four-intermediate-block features only for the 11,760 RR rows.
      Fit one source- and parent-balanced logistic head at fixed C=0.01 on earlier consumed E42 fit
      views plus RR TRAIN triplets. CAL alone chooses a REAL-safe threshold; DEVELOPMENT stays
      unopened until the head and threshold are frozen.
      Complete before DEVELOPMENT: all 11,760 RR rows produced a 11,760x3,072 archive,
      134,777,581 bytes /SHA-256 `fdc5d4c8...a4aa4`. The fixed head then fitted 19,648 views
      (13,768 consumed E42 +5,880 RR TRAIN) with source/parent-balanced weights. Original-only CAL
      selected threshold `0.8712875247`: AUC 0.97369, balanced accuracy 0.92551, REAL FP 10.0%
      and AI recall 95.10%. Frozen candidate SHA-256 is `a3aec445...47390`; RR DEVELOPMENT and
      ITW-SM still have zero scores.
- [x] Require the unchanged full clean gate on RR DEVELOPMENT original, AUC >=0.85 and balanced
      accuracy >=0.80 on both transfer and redigital, plus no material regression on the earlier
      E42 multi-source DEVELOPMENT checks. Freeze “no material regression” before score as clean
      AUC within 0.02 and balanced accuracy within 0.05 of E42-S, robust AUC within 0.02 and
      balanced accuracy within 0.05, plus clean REAL macro/worst FP <=10%/20%, AI macro/worst
      recall >=85%/60% and 100% coverage. If E43-S passes, package it and wait for ITW-SM. Only if
      S fails may the already-local DINOv2-L intermediate arm run on the identical frozen roles.
      Neither local result is final evidence and neither may open ITW-SM before its own score
      contract is committed.
      **E43-S passed on the first frozen run:** original AUC/balanced 0.98194/0.93265 with 7.96%
      REAL FP and 94.49% AI recall; transfer 0.97826/0.92755; redigital 0.95186/0.88673. Historical
      regression checks also pass, although they are explicitly consumed/replayed diagnostics.
      Candidate `a3aec445...47390` is now packaged in `evidence/e43_candidate_contract.json` and
      waits for ITW-SM author approval; DINOv2-L remains locked because S did not fail.

### F2.5 — immediate open Plan C: E43 on untouched DDA-COCO

- [x] Select a licence-clear, ungated and still-untouched external source without searching for an
      easier result after seeing E43 scores. Reuse the already pinned NeurIPS 2025 DDA-COCO
      evaluation benchmark at revision `8c9330a3...68fb`, Apache-2.0, 4,301,452,066 bytes and
      SHA-256 `8cd60077...9c24`. It pairs MS-COCO validation reals with five semantically/frequency-
      aligned VAE reconstruction variants. It tests non-causal shortcut reliance; it cannot replace
      ITW-SM's social-media/camera-pipeline claim.
- [ ] Assemble the five existing multipart files only after binding E43-S candidate
      `a3aec445...47390` and threshold `0.8712875247`. The parts now sum exactly 4,301,452,066 bytes,
      so expected new network transfer is zero; if final SHA verification fails, stop rather than
      silently redownloading beyond the user's 4 GB cap. Run safe ZIP/CRC inventory before decoding
      or extracting a member.
      **Assembly passed with zero network bytes:** the official 4,301,452,066-byte SHA-256 matches,
      ZIP/CRC safety passes and the archive contains 29,969 images /4,298,688,287 expanded bytes.
      Observed structure corrects the card-level assumption: six synthetic variants are present,
      while original COCO reals are not bundled.
- [x] Build a score-blind paired manifest. Keep every real parent and all its reconstruction
      variants indivisible; audit exact SHA-256 and dHash against every E42/E43 protected role and
      exclude an entire pair group for any prior overlap. Freeze counts, bytes, source variants,
      archive/manifest hashes and zero scores before model loading.
      First acquire the official 5,000-image COCO `val2017.zip` companion from the COCO S3 bucket:
      815,585,330 bytes, immutable observed ETag `d366be60d3dc737327160d62453e3973-98` and no more than
      this single 815 MB transfer. Validate its size, ZIP/CRC and exact `val2017/<12-digit>.jpg`
      schema; bind the newly computed SHA-256 before decoding. Retain only IDs present in all six
      DDA variants (`sd-vae-ft-ema`, `sd-vae-ft-mse`, `sdxl-vae`, `stable-diffusion-2-1`,
      `stable-diffusion-3.5-large`, `FLUX.1`).
      **Source and structure checks passed:** the 815,585,330-byte transfer completed once with
      SHA-256 `4f7e2ccb...82f05`; all 5,000 JPEGs passed schema and ZIP/CRC validation. Exactly 4,969
      parents have REAL plus all six synthetic views (34,783 rows). Decode, decontamination and
      model scoring are still pending and must preserve seven-view parent groups.
      Before decode, freeze the duplicate rule: any exact SHA-256 or exact dHash hit against a
      protected prior role removes the whole seven-view parent; a cross-label exact duplicate
      removes every touched parent; a same-label cross-parent exact component retains only its
      lexical first parent. Within-DDA cross-parent dHash matches remain a diagnostic because an
      exact 64-bit perceptual collision alone is not identity evidence.
      **Completed:** all 34,783 candidate rows decoded successfully. Nineteen protected dHash hits
      touched four parents, so those four complete seven-view groups /28 rows were excluded. The
      frozen unscored manifest contains 4,965 parents /34,755 rows, zero exact duplicate groups,
      zero cross-label exact groups and zero within-pool cross-parent dHash diagnostics. Detailed
      manifest SHA-256 is `e663d679...a3db`; model scores remain zero.
- [x] Score the frozen manifest exactly once with the unchanged E43-S candidate/threshold. Require
      100% coverage, ROC-AUC >=0.90, TPR@FPR10 >=0.80, EER <=0.15, balanced accuracy >=0.85,
      REAL FP <=10%, AI macro/worst reconstruction recall >=80%/60%. A miss is preserved without
      threshold repair; a pass is strong independent aligned-benchmark evidence but not a NIST
      certification or a substitute for the pending ITW-SM in-the-wild final.
      Bind manifest/candidate/threshold/counts and these gates in a tracked score contract before
      model loading. Score original archive bytes with the existing E43-S clean three-view DINOv2-S
      feature path; add no test-only resize/compression and report every synthetic variant both
      separately and in the pooled gate.
      **Score contract frozen:** SHA-256 `a414e500...b69da` binds 4,965 parents /34,755 rows, the
      unchanged candidate, threshold and all eight gates with zero model scores.
      **Failed once, no retry:** coverage is 100%, but pooled AUC **0.54178**, TPR@FPR10
      **0.11712**, EER **0.47051**, balanced accuracy **0.51114**, REAL FP **14.44%**, AI
      macro/worst recall **16.67%/12.77%**. Only coverage passes. Post-hoc, even the best pooled
      threshold reaches balanced accuracy just **0.53159** with 43.26% REAL FP, proving this is a
      representation/generalization failure rather than an operating-threshold repair.

### F2b — response to the consumed DDA-COCO failure

- [x] Reclassify DDA-COCO as consumed DEVELOPMENT for every future candidate; preserve its first
      score and never present a later run as independent final evidence.
- [ ] Design E44 around content-matched REAL/AI training pairs and generator-held-out validation.
      A lightweight DINOv2 adapter or selectively unfrozen late blocks must learn causal synthesis
      traces; a new linear threshold over the unchanged representation is not justified by the
      post-hoc ceiling.
- [ ] Acquire or generate a separate paired TRAIN/CAL population (not these 4,965 test parents),
      including reconstruction-style and modern diffusion/flow families. Keep complete prompt or
      image parents in one role and reserve at least one generator family from fitting.
- [ ] Retain the current real-camera/RR regression gates so improving DDA recall cannot silently
      restore the old “real photos become AI” failure. ITW-SM or a future NIST round remains the
      untouched final; no universal-success claim until that independent gate passes.

### F2c — E44 capability isolation before new training bytes

- [x] Freeze a score-blind 700-parent /4,900-row hash sample from the now-consumed DDA manifest.
      Contract SHA-256 is `df256498...5ce9`; selected-parent-list SHA-256 is
      `b1ac6bb2...1990`. No model score existed when this contract was sealed.
- [x] Score the already-pinned official DDA DINOv2-L/14 rank-8 LoRA checkpoint at its published
      threshold 0.5. This is a comparative DEVELOPMENT screen, never a second external-final claim;
      E43 scores cannot select parents or alter the cut. It passed all seven gates: AUC 0.99006,
      balanced accuracy 0.93917, REAL FP 0.86%, all-six macro recall 88.69% and worst recall 64.57%.
- [x] Treat the official DDA representation as a useful E44 specialist only if the fixed screen has
      100% coverage, pooled AUC >=0.85, balanced accuracy >=0.80, REAL FP <=20%, four core
      reconstruction variants macro recall >=80%, all-six macro recall >=70% and worst recall
      >=40%. All gates passed, so retain this representation and do not construct replacement pairs.
- [ ] If the specialist passes, adapt one conservative head on existing source/parent-separated
      RR/E36/E39 training roles while retaining an aligned-data replay/anchor; then require both the
      consumed DDA diagnostic and every real-camera/RR regression gate. If it fails, generate a
      compact separate DDA population using VAE reconstruction, matched JPEG quality and fixed
      pixel-mixup (`Ppixel=0.2`, `Rpixel=0.8`) rather than VAE reconstruction alone.

### F2d — E44-B conservative two-specialist fusion

- [x] Bind the existing E44 DDA specialist stream, the immutable full E43 DDA stream, the frozen
      E35 RR/IPN/owner DDA stream and unchanged E43-S artifact before producing any missing joint
      score. Download zero new image bytes. Assign DDA parents, RR parents and whole IPN devices to
      fit/calibration/development solely by namespaced SHA-256; keep the 210 owner-gallery images
      development-only. Detailed contract SHA-256 is `25681b62...3fb4`; model scores created: zero.
- [x] Produce the missing E43-S score for the frozen 1,670-row E35 population, preserving every
      original identity and path hash. Join exactly two scalar inputs per row: E43-S generalist
      score and official-DDA specialist score. No image label, filename, source or device may be an
      inference-time feature. Coverage is 1,670/1,670; score-stream SHA-256 is
      `35d9d2c2...ad5af`.
- [x] Fit only `StandardScaler + LogisticRegression` on clamped logits of the two scores, with
      source/label-balanced weights. Select one threshold on CAL under REAL macro/worst-FP and AI
      macro/worst-recall constraints; freeze the artifact before reading DEVELOPMENT metrics.
      Candidate SHA-256 is `19fd7bbc...b100`; threshold is `0.3423850493` and DEVELOPMENT scores
      created remain zero.
- [x] Evaluate the frozen fusion once: DEVELOPMENT coverage is 100%, pooled AUC 0.97165 and
      balanced accuracy 0.91099; DDA macro/worst recall is 91.33%/74.67%, RR AI macro/worst is
      99.29%/95.00% and IPN worst-device FP is 1.25%. It nevertheless fails the preregistered
      acceptance gate because RR REAL FP is 12.00% (>10%) and owner-gallery FP is 20.48% (>20%),
      each by one image. Preserve this result; do not repair its threshold post hoc.
- [x] Apply the frozen acceptance rule: DEVELOPMENT coverage must be 100%, pooled AUC >=0.90 and
      balanced
      accuracy >=0.85; DDA all-six macro/worst recall >=75%/50%; RR AI macro/worst recall
      >=80%/60% with REAL FP <=10%; IPN worst-device FP <=20%; and owner-gallery FP <=20%.
      The candidate fails 2/10 checks, so retain separate expert outputs instead of serving a
      falsely universal scalar.

### F2e — E44-C successor without test-set threshold laundering

- [x] Treat all E44-B rows and its threshold miss as consumed DEVELOPMENT. Diagnose disagreement
      and margin patterns read-only, but never rename a post-hoc E44-B threshold as validated.
      The smallest diagnostic cut satisfying both missed budgets is `0.3477933653`; at that cut the
      old population would pass 10/10, but this is hypothesis generation only.
- [x] Freeze `0.3477933653` as the E44-C successor cut using the consumed E44-B evidence, then bind
      5,100 new comparative DEVELOPMENT views: all 2,940 source/parent-separated E43 RR development
      views plus clean and one hash-assigned robust view for 1,080 E42 E36/E39 parents (2,160 views).
      Exclude E42 IPN/owner rows already scored by E35 and reject every E35 exact-byte overlap.
      Detailed contract SHA-256 is `b3c399e9...e1152`; population SHA-256 is `ac79ea36...89aa3`.
- [x] Create a new
      comparative DEVELOPMENT population from already-local, parent/source-separated E43 RR and
      E42 real-camera/modern-AI roles that have never received an official-DDA score. Download zero
      new images and bind identities/roles before DDA inference. All 4,020 unique paths passed their
      frozen hashes; `dda_scores_created` is zero.
- [x] Score the official DDA expert on the new bound population, combine it with the already-frozen
      E43-S stream and evaluate the E44-C candidate once. Require real-camera macro/worst FP,
      modern-AI macro/worst recall, transport robustness and coverage gates simultaneously. Keep
      ITW-SM or a future NIST round as the only independent final. E44-C passed 20/22 checks but
      failed RR-original REAL FP (16.33% >10%) and E42 clean worst-device FP (31% >20%); preserve
      the failed result and do not tune on it.
- [x] Complete the official-DDA arm on 5,100/5,100 bound views before fusion aggregation. The
      1,793,353-byte score stream SHA-256 is `3618b158...d3108`; coverage is 100%.

### F2f — stop binary threshold chasing; add a selective decision layer

- [x] Preserve E44-B/C as consumed failures. Measure score-arm disagreement and risk-versus-
      coverage curves without changing either record. The remaining error is concentrated in
      camera-specific DDA false positives, while ranking and modern-AI recall are already strong.
      The consumed 6,706-row diagnostic supports 87.40% automatic coverage at 96.47% covered
      accuracy with 12.60% `UNCERTAIN`.
- [x] Design a three-outcome policy (`AI`, `REAL`, `UNCERTAIN`) using only the two frozen scores.
      Require high-confidence AI and REAL decisions to meet their class/device budgets; send model
      disagreement and unsafe margins to `UNCERTAIN` instead of forcing a wrong binary claim.
      Hypothesis cuts are REAL below `0.2545712170`, AI at/above `0.6938513176`, otherwise
      `UNCERTAIN`; these are not validated deployment thresholds.
- [ ] Freeze the selective policy on consumed DEVELOPMENT, then validate it only on ITW-SM/NIST or
      another genuinely new source-separated population. Report both automatic coverage and error
      among covered rows; never quote selective accuracy without its abstention rate.

### F2g — E45 official MediaEval/ITW-SM independent final (2026-09-03)

- [x] Bind the official MediaEval public validation distribution before transfer: repository
      `mever-team/mediaeval2026-sid`, direct archive `itw-sm-sid-val.zip`, HTTP identity
      3,553,693,205 bytes /ETag `"68555a02-d3d10e15"` /Last-Modified 2025-06-20. The publisher
      declares 10,000 in-the-wild images, exactly 5,000 REAL and 5,000 synthetic. Preserve the
      already accepted ITW-SM research-only/no-redistribution terms; public reachability is not a
      licence expansion.
- [x] Freeze E44-D and the success contract before bytes. Binary gates remain AUC >=0.90, balanced
      accuracy >=0.85, pooled REAL false-AI <=10%, pooled AI recall >=80% and complete score
      coverage. Source/platform worst REAL false-AI must be <=20% and worst AI recall >=60%.
      Selective reporting additionally requires automatic coverage >=80%, covered accuracy >=95%
      and uncertainty <=20%. The fixed policy is REAL `<0.2545712170`, AI
      `>=0.6938513176`, otherwise `UNCERTAIN`.
- [x] Download only to LaCie with curl resume and >=100 GiB reserve. Require the bound HTTP size,
      ETag and Last-Modified, compute SHA-256 after completion, then perform safe ZIP schema and full
      CRC inventory without extraction or model access. Compare the archive identity/layout with
      the still-gated Hugging Face ITW-SM inventory; never report them as independent benchmarks if
      they are the same distribution.
      Transfer completed at exactly 3,553,693,205 bytes /SHA-256 `18f1806e...b6e3`; all 10,000
      declared paths are structurally present. Per-member CRC found exactly one publisher-side
      corrupt entry, `ITW-SM/1_fake/x_618.jpg`. Its local compressed range is byte-identical to a
      fresh HTTP range response, so redownloading cannot repair it. Preserve 9,999 usable rows and
      disclose 99.99% official-archive coverage; the bad member is excluded before any model load.
- [x] Decode/hash every member, reconcile any bundled metadata, audit exact and dHash overlap
      against all protected prior roles and remove an overlapping parent before scores only. Freeze
      the complete zero-score manifest and its exclusions. If source/platform metadata is absent,
      report pooled groups honestly rather than inventing platform labels.
      Complete: 9,999 CRC-usable members decoded; 19 same-label exact duplicate copies and two
      protected-dHash AI rows were excluded before inference. The immutable final contains 9,978
      rows (4,981 REAL /4,997 AI) across publisher-derived Facebook, Instagram, LinkedIn and X
      groups, with 99.78% official-row coverage. Manifest SHA-256 is `3e7c1d7e...d7e03`; score rows
      remain zero.
- [x] Bind the unchanged E44 fusion artifact and E44-D cuts to that manifest in a second score
      contract. Score every retained member once, report binary and selective gates with 10,000-
      sample bootstrap intervals, and preserve pass or failure without threshold repair. The final
      dataset never enters training; a failure may define a future hypothesis only after E45 is
      marked consumed.
      Score contract is now frozen before model load: detailed SHA-256 `4a5d4999...9ac83` binds
      all 9,978 rows, E43-S `a3aec445...47390`, official DDA `b27a31d3...e3e`, fusion
      `19fd7bbc...b100`, binary cut `0.3477933653`, the E44-D selective cuts and all ten gates.
      The generalist arm has since completed 9,978/9,978 rows with SHA-256
      `43ecaa3f...fc171`; the official-DDA specialist also completed 9,978/9,978 with SHA-256
      `88946986...69bb7`.
      **Completed once; failed 4/10 gates.** AUC is 0.95020 and AI recall 95.40%, but balanced
      accuracy is 0.80634 and REAL false-AI is 34.13%; every platform exceeds the 20% REAL safety
      cap, worst on Facebook at 39.30%. Selective coverage is 80.54% and uncertainty 19.46%, but
      covered accuracy is only 90.07%. Preserve the failure; E45 is consumed and may not tune E44.

### F2h — E46 recovery after the E45 social-real failure

- [x] Diagnose the already-consumed E45 arm scores without changing the result: compare E43-S,
      official DDA and fused score distributions by platform/label; quantify disagreement and the
      REAL error clusters. This is hypothesis generation only—no threshold sweep may become a
      repaired E45 claim.
      Complete: generalist AUC/BA/REAL-FP/AI-recall is 0.8011/0.7297/21.22%/67.16%; official DDA
      is 0.9401/0.8726/10.74%/85.25%; frozen fusion is 0.9502/0.8063/34.13%/95.40%. A post-hoc
      REAL-10% cut at `0.7541002115` would pass all six binary gates with BA 0.8867 and AI recall
      87.35%, proving a calibration-transfer defect—but that cut is selected on consumed E45 and
      is permanently forbidden for deployment.
- [ ] Keep all 9,978 E45 rows prohibited from TRAIN/CAL/model selection. Build the next candidate
      using separate licensed, camera/social-transmission REAL data already local or newly sourced,
      plus existing modern-AI replay. Prefer a learned real-safety gate or source-invariant
      adaptation over another global threshold; retain AI recall regression gates.
- [ ] Before any E46 fitting, reserve a new untouched source-separated final distinct from
      MediaEval/ITW-SM, RR, DDA-COCO and every earlier role. If no such final is legally available,
      E46 may be reported only as development progress, never as project success.

#### E46-A — cross-platform calibration recovery (frozen before transfer, 2026-09-03)

- [x] Select two role-separated primary sources before downloading any image. Use the official
      GRIP-UNINA **SynthWildX** list (2,000 X-hosted images: 500 REAL, and 500 each DALL-E 3,
      Midjourney v5 and Firefly) only as `E46_CAL_DEV`. Reserve the official UNITN **TrueFake
      Facebook** distribution (advertised 3.9 GB; one Facebook-processed copy of the paper's
      60,000-image shared subset) as `E46_UNTOUCHED_FINAL`. Never exchange these roles.
- [x] Acquire SynthWildX from the publisher's immutable `list.csv` first. Preserve URL/filename/
      label metadata, per-file hashes and failures; do not redistribute social-media bytes. Split
      identities deterministically and label-stratified into 60% CAL / 40% DEVELOPMENT before
      model scoring. Do not silently replace dead URLs with lookalike images.
      Completed score-blind: 1,723/2,000 publisher URLs yielded valid images (553,125,164 bytes),
      while 277 current X CDN URLs returned persistent 403/404 failures and were preserved as
      failures. Successful CAL/DEVELOPMENT counts are 1,034/689; REAL remains 418 and each AI
      generator retains 396–474 images. The external manifest SHA-256 is `fd8008a...a89f3f`.
- [x] Acquire the TrueFake Facebook archive with resume into the external LaCie store, record the
      exact byte count and SHA-256, then inventory it without extraction. Before any score, freeze
      a deterministic, class-balanced 2,000-row final manifest: 1,000 REAL balanced across the
      available real origins and 1,000 AI balanced as closely as possible across generator
      families. Exclude corrupt files and exact/perceptual overlaps with protected earlier roles;
      disclose every exclusion. No label or score may influence selection.
      Transfer complete and still unscored: 4,207,525,545 bytes /SHA-256 `413cb7f9...cda0d63`.
      Independent `gzip -t` and TAR listing pass; the archive contains exactly 60,000 JPG files,
      including 10,000 FFHQ, 10,000 FORLAB and 5,000 from each of eight declared AI generators.
      Binding now complete before extraction: all member facts hash to `b59e78de...8ba28b`; a
      3,500-row reserve and exact per-source quotas are fixed in contract SHA-256
      `1e77dfbd...cead3`. The contract still contains zero decoded final images and zero scores.
      Extraction/audit also complete: all 3,500 reserve candidates decode, none overlaps 24
      protected manifests, and the frozen 2,000-row final is exactly balanced. Manifest SHA-256 is
      `4572339e...b225b`; model-score count remains zero.
- [x] On SynthWildX CAL only, compare the unchanged official DDA arm, frozen E44 fusion, a global
      REAL-safe cut, and a small QuAD-inspired quality-conditioned calibration. Method choice may
      inspect only CAL. DEVELOPMENT must retain REAL FP <= 20%, worst available REAL group FP <=
      25%, AI recall >= 80%, and worst AI-generator recall >= 60%; otherwise preserve the failure.
      Keep a simpler global cut when the quality model does not materially improve both safety and
      AI retention. Do not retrain a backbone in this stage.
      Score contract now frozen before model load at SHA-256 `b3fe31a3...5c98c`, binding all 1,708
      clean rows, both model weights, the old fusion, three allowed methods and all development
      gates. CAL is 1,024 rows; DEVELOPMENT is 684; score count is zero.
      E43-S scoring is now complete at 1,708/1,708 rows; its resumable stream SHA-256 is
      `8be0aefd...ce88d`. Official DDA also completed 1,708/1,708 at SHA-256
      `a7fbd7e2...257eda`. All fused/calibrated results remain unopened.
      Calibration-method contract SHA-256 `6799231f...c9228c` now splits CAL score-blind into 612
      QUALITY_FIT and 412 OPERATING_CAL rows within every source. It freezes the REAL-10% threshold
      rule, method eligibility, simple-model preference and selective-band rule before reading any
      score. DEVELOPMENT and TrueFake reads remain zero.
      CAL-only fitting selected the simpler `fusion_global` at threshold `0.6688565013` and
      candidate SHA-256 `9fec91b8...b84a1`. On OPERATING_CAL it has AUC 0.97362, BA 0.91795,
      REAL FP 10.0% and AI recall 93.59% (worst generator 84.07%). The quality Gaussian improved
      recall but had lower AUC and was correctly rejected by the frozen non-inferiority rule.
      DEVELOPMENT passed 4/4 gates unchanged: AUC 0.97203, BA 0.91217, REAL FP 11.38%, AI recall
      93.81%, worst-generator recall 84.82%. The optional selective diagnostic has 96.49% coverage
      but 94.39% covered accuracy, just below the future final 95% gate; do not repair it from
      DEVELOPMENT.
- [x] Freeze the chosen artifact, threshold(s), selective band and ten E45-style gates before
      opening TrueFake labels/scores. Score the 2,000-row Facebook final exactly once and report
      source/generator metrics plus bootstrap intervals. E45 remains archived and prohibited;
      passing TrueFake does not rewrite the E45 failure, and both are required in the final claim.
      Final score contract is now frozen before model load at SHA-256 `1cf28d2d...7c4262`. It binds
      the 2,000 rows, candidate and both model hashes, threshold `0.6688565013`, selective cuts,
      all ten gates and 10,000 source-stratified bootstrap samples. Final score count is zero.
      Final generalist arm is complete at 2,000/2,000 rows; stream SHA-256 is
      `43eb1562...b5f25c`. Official DDA also completed 2,000/2,000 at SHA-256
      `13947caf...878d0b`. The fused result then opened exactly once. E46 failed 5/10 gates:
      AUC 0.81548, balanced accuracy 0.73450, REAL false-AI 5.60%, AI recall 52.50%, worst REAL
      source false-AI 10.20%, and worst AI-source recall 1.60%. StyleGAN/2/3 recall is
      1.60%/1.60%/3.20%, which isolates the principal blind spot. Coverage is 100%; the selective
      policy covers 95.85% but is only 74.86% correct. This final is consumed forever: no threshold
      repair, row removal, refit or retry is allowed.

### F3 — recording and stop rules

- [x] Record every source/byte/label fact in `DATASETS.md`, every measurement or failed hypothesis
      in `ml/EXPERIMENTS.md`, and every decision/result in append-only `HISTORY.md`. Commit the
      source contract before transfer and each completed scientific gate afterward; push only
      verified checkpoints with green CI.
- [x] If B-Free URLs are too incomplete, the RR server cannot resume, or ITW-SM remains gated,
      stop with the exact external blocker. Do not replace a difficult test with an easier dataset
      after seeing scores and do not manufacture a “pass” by lowering the gate.
      E46 reached an independent final rather than an availability blocker; its failed result is
      preserved. Any E47 improvement must use a new development source and a new untouched final.

## E47 — GAN-blind-spot recovery without sacrificing real-camera safety (started 2026-09-03)

E46 is consumed and failed because the frozen detector retained diffusion signals but missed
Facebook-transported StyleGAN/2/3. E47 treats that result as a diagnosis, never as reusable final
evidence. Its objective is not to lower the threshold: add a complementary GAN-sensitive arm while
keeping pooled REAL false-AI <=10%, worst REAL source <=20%, pooled AI recall >=80%, and worst AI
source >=60% on a genuinely new final.

- [x] **R0 — preserve E46 before repair.** Commit the immutable arm streams, one-shot report,
      bootstrap intervals and failed 5/10 gate. Forbid row removal, refit, threshold repair and
      repeat evaluation on TrueFake Facebook.
- [x] **R1 — cheapest complementary-arm diagnostic.** Score the already-trained, hash-pinned
      GenImage ResNet-18 on the consumed E46 identities using its original deterministic 224 px
      preprocessing. Report only diagnostic AUC/TPR at a pooled 10% REAL-FP cut, per-generator
      recall, and how many frozen-fusion misses it recovers. This is architecture triage, not a new
      E46 result. Unlock it only if mean StyleGAN-family recall is >=50%, every StyleGAN generation
      is >=30%, and diagnostic OR-fusion REAL-FP is <=15%.
      The arm failed every unlock condition: StyleGAN/2/3 recall is 4%/8%/8%, pooled AI recall
      23.6%, and OR-fusion reaches only 58.5% AI recall while REAL false-AI rises to 15.1%.
      Reject the legacy arm; do not spend new training time on this representation.
- [x] **R2 — external specialist only if R1 fails.** Acquire a hash-pinned official GAN detector
      (prefer UnivFD's frozen CLIP linear head or UNINA's compression-trained GAN detector) under
      its licence; do not download the 72 GB training corpus. First run the same consumed-data
      diagnostic and reject any arm that merely increases REAL accusations.
      UnivFD repository is pinned at `030495a...c619`; its 4,083-byte head hashes to
      `47710074...c7847`. The official 932,768,134-byte CLIP ViT-L/14 backbone hashes to the
      publisher-declared `b8cca3fd...03836`. A two-row hidden-score smoke test passes; no metric
      has been opened. Apply the R1 unlock rule unchanged before admitting this arm to R3.
      UnivFD then showed the needed complementarity—StyleGAN/2/3 recall 94.4%/74.4%/80.0%,
      recovering 310/475 E46 misses and lifting diagnostic OR recall to 83.5%. However, pooled REAL
      false-AI is 15.5%, missing the frozen <=15% unlock by 0.5 points. Preserve this near-success
      but do not relax the rule; compare the permitted UNINA compression-trained specialist next.
      The official UNINA checkout is pinned at `543943c...df88`; its StyleGAN2-trained
      ResNet50-NoDown is 282,549,121 bytes /SHA-256 `65467594...d5a08`. Licence is nonprofit-only,
      so it can inform research and an opt-in arm but cannot become an unrestricted default.
      Official-example direction/load smoke passes; performance metrics remain unopened.
      Native-resolution inference was stopped before any metric at 655/2,000 rows because
      throughput fell below 0.5 image/s on ordinary 960 px inputs—unfit for the web objective.
      Preserve the partial stream (`87417d5f...733a4`), then restart from zero with a score-blind,
      aspect-preserving 512 px long-side cap. The cap is frozen for all rows before result access.
      Capped UNINA catches StyleGAN/2/3 at 100%/94.4%/73.6%, recovers 375/475 E46 misses, and
      raises diagnostic OR AI recall to 90.0%. It still reaches 15.5% REAL false-AI, so the direct
      frozen-arm unlock fails exactly like UnivFD. Conclusion: the representation gap is solved,
      but a new-data decision gate is mandatory; neither specialist can be OR-ed into serving.
- [ ] **R3 — new CAL/DEVELOPMENT, identity-separated.** Build a compact source-balanced pool from
      data that was never in E46 final: at least two independent camera-real sources, StyleGAN
      generations, and the modern diffusion generators already protected by E46. Split by source,
      parent identity and transport before scores. Fit only a small calibrated fusion/gate; keep
      all backbones frozen. Require the E46-style REAL and AI gates on untouched DEVELOPMENT.
      Frozen pool design: exclude all 3,500 E46 reserve members, then hash-rank new archive members
      under namespace `E47_TRUEFAKE_CALDEV_V1`. CAL contains 600 FFHQ REAL and 200 each from
      StyleGAN2, SD1.5 and SDXL. DEVELOPMENT contains 600 FORLAB REAL, 200 each StyleGAN/StyleGAN3,
      and 100 each FLUX.1/SD3. Extract 20% score-blind reserve headroom, decode and exclude exact/
      dHash overlaps before filling the fixed 1,200/1,200 roles. Compare frozen E46 alone,
      E46+UnivFD, E46+UNINA and three-arm regularized logistic gates using CAL only. Prefer the MIT
      UnivFD path when it passes and is within two recall points of the nonprofit UNINA path.
      Score-blind binding completed: 2,880 reserve candidates, 1,440 per role, outside all 3,500
      E46 reserve identities. Contract is 919,423 bytes /SHA-256 `c031ef92...d0753`; target remains
      2,400 and model-score count remains zero.
      Extraction/audit completed: all 2,880 candidates decode; one SD1.5 reserve row is excluded
      for protected dHash overlap. The clean manifest fills all nine quotas at 1,200 CAL and 1,200
      DEVELOPMENT, each 600 REAL/600 AI. Manifest SHA-256 `378b83fe...85739`; scores remain zero.
      Four-arm score contract SHA-256 `ee2a2958...95798` now binds the manifest, E43-S, DDA,
      E44 fusion, UnivFD backbone/head and capped UNINA weights before model load. Score count zero.
      E43-S completed 2,400/2,400 with stream SHA-256 `073110f4...f30c03`; no metric opened.
      DDA completed 2,400/2,400 with stream SHA-256 `8001c60b...d75f5`; no metric opened.
      UnivFD completed 2,400/2,400 with stream SHA-256 `67b7b94c...e2829`; no metric opened.
      Capped UNINA completed 2,400/2,400 with stream SHA-256 `7efb36c0...5e16d`; no metric
      opened. All frozen score arms are complete. Next freeze the exact CAL-only fitting,
      threshold and candidate-selection rule before interpreting any score.
      Decision rule to freeze: reconstruct frozen E46 from E43-S+DDA, compare it with
      C=0.1 standardized logistic gates adding UnivFD, UNINA, or both. Fit with equal CAL
      label/source mass. For every candidate choose the lowest CAL threshold holding pooled
      REAL FP <=10% and worst-source FP <=20%; require AUC >=0.90, BA >=0.85, pooled AI
      recall >=80% and worst AI-source recall >=60%. Rank eligible candidates by worst AI
      recall, pooled AI recall, AUC and BA. If an UNINA-bearing winner is within two points
      of eligible MIT-only E46+UnivFD on both AI-recall measures, select E46+UnivFD.
      Freeze the selected head and threshold before opening DEVELOPMENT once.
      Decision contract frozen at SHA-256 `a4515caf...875a`; CAL and DEVELOPMENT metrics
      both remain unopened. Six focused contract tests pass.
      CAL opened once: frozen E46 fails (AUC 0.8735, BA 0.8175, AI recall 73.5%,
      StyleGAN2 20.5%). E46+UnivFD narrowly fails the 60% worst-AI floor at 59.0%.
      E46+UNINA passes, while E46+both ranks first and is selected at threshold
      `0.3353660721`: AUC 0.9897, BA 0.9367, pooled AI recall 97.33%, worst AI recall
      95.0%, REAL FP 10.0%. Candidate SHA-256 `f659ee4f...0b0d`; DEVELOPMENT unopened.
      One-shot DEVELOPMENT is a valid near-miss: AUC 0.95491, BA 0.88417, REAL FP
      7.33%, pooled AI recall 84.17%, StyleGAN/StyleGAN3/SD3 recall 100%/87.5%/84%,
      but FLUX.1 recall is 46%. Six of seven gates pass; the frozen 60% worst-source
      floor fails. Stream SHA-256 `97fbe4b7...72cd`. Archive E47-R3 without threshold
      repair or retry. Next work must diagnose arm-level FLUX complementarity on this now-
      consumed DEVELOPMENT only, then pre-register a non-veto successor on new data.
      Post-failure diagnostic boundary: on consumed DEVELOPMENT report frozen E46 and each
      specialist's source behavior plus score correlation with the selected gate. Answer only
      whether FLUX evidence existed and was vetoed. Do not select a threshold, candidate or
      serving rule from these rows. If confirmed, E48 must pre-register a non-veto conditional-
      OR/mixture rule on fresh CAL and prove it on fresh DEVELOPMENT; otherwise acquire broader
      FLUX-like CAL evidence before changing architecture.
      Diagnostic result: E46 sees 95/100 FLUX rows at its CAL-only cut, while the selected
      gate sees 46/100 and vetoes 50 E46 hits. Conversely, the selected gate reduces FORLAB
      false-AI from 30.67% to 7.33% and rescues 149 StyleGAN plus 141 StyleGAN3 rows over
      E46. Representation complementarity is proven; a single logistic compromise is the
      failure. E48 must use fresh, diverse REAL plus diffusion CAL and a non-veto expert
      router/conditional union, then face a fresh source/transport-held DEVELOPMENT.
- [ ] **R4 — new final, then serving.** Bind a new publisher-separated final before model load,
      including GAN, diffusion and two REAL pipelines plus social-media degradation. Score once
      with 10,000 source-stratified bootstraps. Only a full gate pass may replace the served E32
      model; otherwise keep the current demo and archive the failure.

### E48 — monotone non-veto successor (planned 2026-09-04)

E47 proved that the required signals already exist but a signed logistic compromise suppresses
FLUX when GAN specialists are quiet. E48 changes only the decision geometry and evidence split;
all four backbones/scores remain frozen.

- [x] **Fresh zero-download population:** 2,400 identities outside every E46 reserve and E47
      candidate, and outside current-candidate training identities. FIT =300 REAL +300 AI;
      CAL =300 REAL +300 AI; DEVELOPMENT =600 REAL +600 AI. FIT/CAL REAL each use 150 unused
      VISION camera originals plus 150 unused CSAFE S21 originals, device-balanced. DEVELOPMENT
      REAL uses 600 unused FODB originals, device-balanced and capped at five cameras per shared
      scene. FIT and CAL AI each use TrueFake FLUX.1 100, StyleGAN2 100, SD1.5 50 and SDXL 50.
      DEVELOPMENT AI uses fresh FLUX.1/SD3/StyleGAN/StyleGAN3 at 150 each. All rows must decode,
      pass exact/dHash protection and be selected by namespace hash before model access.
      **Selection bound:** 2,880 score-blind candidates (720 FIT, 720 CAL, 1,440
      DEVELOPMENT) for the 2,400-row target; every role is class-balanced. All 6,380 prior
      E46/E47 TrueFake candidates and current-candidate E32 training identities are excluded.
      Contract SHA-256 `dbb6f4aa...0e6e`; model scores zero. Next: payload verification,
      extraction and protected exact/dHash audit only.
      Audit amendment before model access: the first extraction correctly hard-stopped because
      legacy `r1b_role_manifest` enumerates the complete 22,688-row candidate plan, not the
      current model's consumed training identities, and therefore masked every new camera row.
      Exclude that planning ledger from E48's role-hash set while retaining the exact E42 current-
      training exclusion plus every actual CAL/DEVELOPMENT/final manifest. Scores remain zero.
      Second pre-score audit amendment: camera bytes reproduce their pinned SHA-256 exactly, but
      the current helper's EXIF handling does not reproduce the older realization audit's dHash.
      Do not replace/recompute the historical dHash. Verify byte SHA and reuse the already-decoded,
      pinned audit dHash for overlap checks. Candidate identities/quotas and score count stay fixed.
      **Manifest complete:** 2,880/2,880 candidates verified/decoded, zero decode failures and
      two protected-dHash exclusions (one VISION, one FODB). The 20% headroom fills every quota:
      600 FIT, 600 CAL and 1,200 DEVELOPMENT, each exactly class-balanced. Manifest SHA-256
      `1404a3ff...5b68`; model scores remain zero. Next bind all frozen arm identities before
      inference, then score FIT+CAL first and keep DEVELOPMENT unopened through selection.
      **FIT+CAL score lock:** contract SHA-256 `ea7de06c...9516` binds the 2,400-row
      manifest and exact E43-S, DDA, E44 fusion, UnivFD and capped-UNINA identities. Only
      600 FIT +600 CAL may be scored; 1,200 DEVELOPMENT rows are explicitly forbidden.
      Six focused tests pass; model/development score counts remain zero/zero.
      **E43-S FIT+CAL arm:** 1,200/1,200 rows, 100% coverage; 258,603-byte stream
      SHA-256 `f2a1be3b...137cf7`. DEVELOPMENT rows scored: zero; no aggregate metric opened.
      **DDA FIT+CAL arm:** 1,200/1,200 rows, 100% coverage; 257,443-byte stream
      SHA-256 `7ebc7831...b4a23b`. DEVELOPMENT rows scored: zero; no aggregate metric opened.
      **UnivFD FIT+CAL arm:** 1,200/1,200 rows, 100% coverage; 259,689-byte stream
      SHA-256 `e768d591...81d635`. DEVELOPMENT rows scored: zero; no aggregate metric opened.
      **Capped-UNINA FIT+CAL arm:** 1,200/1,200 rows, 100% coverage; 257,330-byte stream
      SHA-256 `e3d47527...4b3c01`. All four permitted arms are complete; DEVELOPMENT rows
      and aggregate metrics remain zero.
      **Decision lock:** contract SHA-256 `22154ab9...590e` fixes the empirical-percentile
      formula, four nested monotone candidates, authentic-safety threshold rule, seven CAL/DEV
      gates, ranking and MIT-licence preference before any aggregate score is interpreted.
      FIT AI and all DEVELOPMENT access remain forbidden. Seven focused score/decision tests pass;
      a no-candidate outcome is persisted as a clean failure instead of producing an artifact.
- [x] **Monotone evidence fit:** use FIT REAL only to map each frozen arm score to its empirical
      authentic-image percentile. Compare E46, E46+UnivFD, E46+UNINA and all-three using the
      maximum expert percentile. Because `max` is monotone, a low irrelevant-specialist score can
      never veto another arm's high AI evidence. No backbone or signed multivariate head is fit.
      Completed exactly as bound with 300 FIT REAL rows; FIT AI usage remained zero.
- [x] **CAL selection — failed cleanly:** select one threshold and candidate only on CAL. Require coverage 100%,
      AUC >=0.90, BA >=0.85, pooled REAL FP <=10%, worst camera/device FP <=20%, pooled AI recall
      >=80% and worst AI-source recall >=60%. Rank eligible candidates by worst AI recall, pooled
      recall, AUC and BA; retain the two-point MIT preference when it does not weaken either recall
      measure by more than two points. No candidate passed: best AUC was 95.42%, but E46's
      StyleGAN2 recall was 11%; the best specialist combination reached only 19%. CAL report
      SHA-256 `032944b8...75e`; no candidate artifact was created.
- [x] **One-shot DEVELOPMENT — cancelled unopened:** no candidate qualified, so the 1,200-row
      FODB/held-AI split was not scored. It remains clean evidence for a separately pre-registered
      successor rather than being consumed to diagnose or repair E48.

### E50 — frozen generalist transfer, then final (planned 2026-09-04)

E48's permitted post-failure CAL diagnosis found that the frozen E43-S generalist alone passes
every declared CAL gate at threshold `0.07940196245908739`: AUC 98.84%, BA 93.83%, pooled REAL FP
3.33%, worst camera FP 20%, pooled AI recall 91%, and worst-source recall 76% (StyleGAN2). E46's
signed DDA fusion reduces that StyleGAN2 recall to 13%; the later percentile layer cannot restore
evidence already vetoed inside E46. Treat this as model selection on CAL, not as an E48 repair.

- [x] **Bind E50 before DEVELOPMENT:** single candidate = exact frozen E43-S artifact and exact
      1,200-row E48 generalist FIT+CAL score stream; threshold and all seven gates remain fixed.
      Bind the untouched 1,200-row DEVELOPMENT identities and forbid every other expert, training,
      threshold adjustment and second attempt. Contract SHA-256 `18ae708f...20f6` binds exact
      candidate, threshold, 1,200 identities and seven gates with score/metric counts at zero.
      Eight focused E48/E50 decision-boundary tests pass.
- [x] **One-shot E50 DEVELOPMENT — PASS:** score only E43-S on the untouched 600 FODB REAL +600 held-AI
      rows (FLUX.1, SD3, StyleGAN and StyleGAN3, 150 each). Apply the same seven gates once, with
      FODB camera pipeline as the worst-REAL unit. Archive pass or failure before any next step.
      **Inference checkpoint:** 1,200/1,200 rows scored with 100% coverage and no replacement;
      271,063-byte stream SHA-256 `07461b09...d5fd`. DEVELOPMENT metrics remain unopened.
      The frozen threshold passed all seven gates on first use: AUC 97.84%, BA 90.17%, pooled
      REAL FP 3.83%, worst camera FP 18.18%, pooled AI recall 84.17% and worst-source recall
      68.67%. FLUX.1/SD3/StyleGAN/StyleGAN3 recall =98.67/92.67/68.67/76.67%. No retry or repair.
- [ ] **E49 comprehensive independent final:** only an E50 DEVELOPMENT pass may bind the >=2,000
      row publisher-separated native + social/recompressed final already required below. A full
      ten-gate pass creates Module-1 v1 and may update the web demo; otherwise Module 1 stays open.
      **Completed one-shot result — FAIL, Module 1 remains open:** 11/20 checks pass (original 6/10,
      Q75 5/10). AI recall is strong at 94.30%/95.50%, but REAL false-AI is 39.10%/49.00%; worst
      device is 71%/84%. Original AUC is 90.24%, Q75 AUC 86.89%. No retry or repair is allowed.

#### E49-A — comprehensive-final source and decision contract (frozen before image transfer)

E50 passed, so the last Module-1 proof may now be built. This is one independent, one-shot test;
it is not another development set and it cannot be used to repair E43-S.

**Post-approval routing checkpoint:** authenticated access to both Datapoint revision
`e1d8719a...c928` and Hugging Face ITW-SM is now open. ITW-SM is the same distribution already
consumed through the official MediaEval E45 archive and cannot become independent again. Datapoint
is a strong 2026 preference benchmark, but OpenFake `core/test` is purpose-built OOD detection data
and E49-C identities/bytes were already frozen without scores. E49-C therefore remains the sole
balanced-final route; Datapoint stays unscored and image-free for possible post-final diagnosis.

- [x] **Reject attractive but invalid shortcuts before bytes.** SCIMD-17 is Apache-2.0 and compact,
      but every camera image was publisher-resized to 224 x 224; it cannot prove gallery-like
      authentic-photo safety and would create a resolution shortcut. ImageBench publishes useful
      2026 outputs, but its current site licence reserves redistribution/republication and therefore
      is not admitted. Qwen Image Bench, TrueFake, ITW-SM, FODB, VISION, CSAFE, IPN, owner-gallery
      and every earlier TRAIN/FIT/CAL/DEVELOPMENT source remain excluded or consumed.
- [ ] **Freeze exactly 2,000 balanced parent identities.** REAL = 1,000 Wikimedia Commons original
      uploads, 100 each from ten declared modern phone/camera categories, JPEG only, with matching
      camera EXIF/category evidence and a per-uploader cap. AI = 800 Datapoint 2026 benchmark
      outputs, 160 each from GPT Image 2, Nano Banana 2, Seedream 5 Pro, FLUX 2 and Ideogram 4,
      plus 200 StyleGAN2 rows from the already-local Apache-2.0 AIGC Detection Benchmark. Datapoint
      stays metadata-only until its authors approve the user's submitted contact-sharing request;
      the request is currently awaiting review. No substitute source may be chosen after a detector
      score is seen.
- [ ] **Bind before download.** Pin repository revisions, Commons page/revision ids, prompt ids,
      generator/provider names, expected byte lengths, licences/terms, a deterministic hash rank,
      10% Commons reserve plus 20% AI reserve where available, and a 4 GiB network stop. Store payloads only under
      `/Volumes/LaCie/pixelproof-datasets/e49/`; never commit third-party images. Abort rather than
      silently replacing a source, device or generator after scoring starts.
      **Implementation checkpoint:** `e49_acquisition.py` now pins both Hugging Face revisions,
      validates licence/revision drift, filters Commons to licensed original JPEG metadata, caps
      repeated uploaders and selects AIGC StyleGAN2 by Parquet row coordinates without reading image
      payloads. Four focused tests pass. Live probe reports the exact Datapoint revision but correctly
      refuses its gated payload (`GatedRepoError`); downloaded E49 image bytes remain zero.
      **Local GAN checkpoint:** all 60 pinned AIGC shards reproduce 125,026 metadata rows and 1,997
      eligible StyleGAN2 rows. A deterministic 240-row reserve is now reproducible from only the
      `label`/`generator` columns; its identity digest is `15e5c131...cc731`. No image was decoded.
      **Commons feasibility amendment before binding:** nine original device categories fill their
      uploader-capped reserve, but Fujifilm X-T5 yields only 66/120 and cannot support a 100-row
      diverse target. Replace only that unbound category with Nikon Z 8, which yields 110/110 from
      25 uploaders. Reduce Commons headroom to 10% so the same 1,000-parent target respects the
      frozen 4 GiB stop. No E49 REAL image or detector score exists.
      **Open-component V1 stop:** the complete 1,100-REAL +240-StyleGAN2 identity bind succeeded,
      but its Commons reserve alone is 4,140,590,955 bytes, leaving no honest room for Datapoint
      inside the total 4 GiB ceiling. Archive contract `c6f2cfb0...f794` as rejected before transfer;
      do not download it. Pre-register a size-aware successor from the same cached metadata.
      **Open-component V2 lock:** a predeclared 4 MiB original-file cap fills every device reserve
      and reduces 1,100 Commons rows to 2,706,581,778 bytes, leaving 1,588,385,518 bytes of the
      global ceiling for Datapoint. The 240 local StyleGAN2 coordinates are also frozen. Contract
      SHA-256 `1d4e184c...82aa`, reserve identity `31c0e420...e171`; images/scores remain zero.
      **Commons transfer implementation:** the restart-safe downloader is committed before body
      access. It binds the V2 contract, exact byte length and Wikimedia SHA1, requires decoded JPEG
      and frozen geometry, records SHA-256 plus available EXIF make/model, and refuses unexpected
      files or partial completion. Eighteen focused acquisition/download/final tests pass.
      **Wikimedia pacing amendment:** the first eight-worker attempt received an explicit 429 after
      two complete files. Preserve those files, reduce to one request stream with 0.75 s pacing,
      identify the public research repository in User-Agent and honor bounded Retry-After/backoff.
      No thumbnail substitution or identity change is allowed.
      **Native iPhone container amendment:** Wikimedia declares row 139,916,479 as JPEG and `file`
      confirms JPEG, but its Apple MPF segment makes Pillow report `MPO`. Admit JPEG/MPO while
      preserving exact original bytes and record the distinction; dimensions/SHA1 remain mandatory.
      **EXIF geometry amendment:** Wikimedia reports display-oriented dimensions, while encoded
      iPhone pixels may be transposed under EXIF orientations 5–8. Record both geometries and require
      the EXIF-display dimensions to equal the frozen API contract; never rotate original bytes.
      **Commons transfer complete:** 1,100/1,100 originals and all 2,706,581,778 contracted bytes
      validate, exactly 110 per device. Format audit reports 861 JPEG +239 Apple MPO; 1,074 files
      retain EXIF make/model. Receipt SHA-256 `2511f0ad...7e04`; detector scores remain zero.
      **Local GAN realization:** all 240 frozen coordinates decode with zero failure and zero
      protected/internal overlap; the first 200 clean rows are fixed at manifest SHA-256
      `150ed354...ec99`. All are publisher 256x256 PNG, so the format/geometry audit must flag the
      shortcut risk and E49 may credit this source only alongside the other five AI families.
      Selected bytes total 20,111,615; detector scores remain zero.
      **Datapoint approval audit:** access now succeeds at the same revision/licence. Only models,
      test responses and prompt reference metadata were fetched (1,737,709 bytes total); all 40
      image Parquets /image-body bytes remain untouched. This access does not reopen or replace the
      already-selected E49-C final route.
- [x] **Realize and decontaminate without a model.** Decode every candidate; verify label/source,
      image MIME, dimensions and EXIF where promised; reject exact SHA-256 and dHash overlap against
      every protected role; cap repeated Commons uploader/prompt groups; then freeze the first clean
      quota by the predeclared hash order. Record all exclusions and 100% retained-manifest coverage.
      **REAL implementation checkpoint:** the Commons realization is receipt-bound and requires 100
      clean parents per frozen device after protected/internal SHA-256+dHash checks. It protects
      against the final AI components and scored Dotting diagnostic, creates fixed Q75 children,
      rejects child collisions and refuses anything except 1,000 parents/2,000 observations.
      Twelve focused transfer/realization/evaluation tests pass before production execution.
      **Device-evidence lock:** the completed receipt exposes one Nikon Z 8 category row whose EXIF
      says Nikon D70. Before realization, require normalized make/model aliases where EXIF exists,
      accept the 26 explicitly category-only rows as such, and exclude any mismatch by fixed rank.
      Thirteen focused tests pass; no camera identity or score has yet been selected.
      **Realization wiring correction:** the transfer receipt intentionally has SHA-256 but no
      dHash. The first production call stopped before output when the audit read dHash too early.
      Reproduce each file SHA/EXIF-display geometry and derive dHash inside realization before
      overlap checks. Fourteen focused tests pass; identities and score count remain unchanged.
      **REAL freeze PASS:** all 1,100 candidates realize. One protected dHash overlap and the one
      Nikon-D70 mismatch are excluded; every device still fills exactly 100. The 1,000 parents and
      1,000 Q75 children freeze at SHA-256 `657be9bb...8e7b`; 979 selected parents have matching EXIF
      device evidence and 21 are explicitly category-only. Model-score count remains zero.
- [x] **Create two paired conditions per parent.** `publisher_original` preserves received bytes.
      `social_q75` applies EXIF transpose, RGB conversion, long-side cap 1080, JPEG quality 75,
      4:2:0 subsampling and metadata removal. The derived child inherits its parent id, label and
      source; 4,000 observations still count as N=2,000 parents. Bootstrap and split only by parent.
      **Complete freeze:** all three component hashes bind and all 4,000 files reproduce SHA-256 and
      display geometry. Exactly 2,000 parents share both conditions and all sixteen source quotas.
      Original formats are 1,585 JPEG/213 MPO/202 PNG; every Q75 child is JPEG. Manifest SHA-256
      `9744a9d2...5909`; model-score/metric counts remain zero.
- [x] **Freeze E43-S and the decision layer before scoring.** Candidate artifact SHA remains
      `a3aec445...7390`; binary AI cut remains `0.07940196245908739`. Selective decisions are REAL
      below `0.011505939625203613`, AI at/above the binary cut and UNCERTAIN between them. These cuts
      were selected only from consumed E50 CAL/DEVELOPMENT and may not move in E49.
      **Execution implementation checkpoint:** separate `bind-score`, `score` and `open-metrics`
      commands enforce this order. The contract binds manifest/component/model/weight hashes and
      fixed gates; resumable inference writes only identity+raw score rows; metric opening requires
      the completed stream hash. Fourteen focused lock/score/evaluation tests pass before use.
      **Lock complete:** contract SHA-256 `fecd724c...61dd` binds final manifest
      `9744a9d2...5909`, observation identities `3cf565a1...5242`, E43-S artifact, DINO weights,
      thresholds and all twenty checks. Score/metric counts are zero at this commit.
- [ ] **Require the full ten gates independently on both conditions.** Coverage =100%; AUC >=0.90;
      balanced accuracy >=0.85; pooled REAL false-AI <=10%; worst REAL device/source false-AI <=20%;
      pooled AI recall >=80%; worst AI generator recall >=60%; selective automatic coverage >=80%;
      covered accuracy >=95%; uncertainty <=20%. Report every source and 10,000 parent-level,
      label/source-stratified bootstrap intervals. Also report a model-blind format/geometry audit;
      a shortcut warning cannot be hidden by a passing pooled number.
      **Implementation checkpoint:** `e49_evaluation.py` now fixes both thresholds and all 20 checks,
      requires exact 2,000-parent pairing and all 16 source quotas in both transports, treats score
      failures pessimistically, and implements deterministic source/label-stratified parent
      bootstrap intervals. It has not opened a final metric because the frozen manifest does not yet
      exist. Seventeen acquisition/evaluation/shared-metric regression tests pass together.
- [x] **Open metrics once.** Raw score streams are committed before evaluation. Any failure, missing
      source, coverage miss or gate miss keeps Module 1 open; no threshold repair, row removal,
      source removal or second E49 attempt is allowed. Only a 20/20 pass (ten gates x two conditions)
      freezes Module-1 v1 and authorizes a separately reviewed demo update.
      **Raw-score lock:** E43-S scored all 4,000 frozen observations with 100% coverage. The
      1,005,967-byte stream SHA-256 is `249f005c...10a8`; aggregate metrics remain unopened and
      this evidence is committed before the one allowed evaluation.
      **One-shot result — FAIL 11/20:** publisher original passes 6/10 and social-Q75 5/10. Coverage
      is 100%; AI recall 94.30%/95.50% and worst-family recall 91.88%/91.25% pass. REAL false-AI
      39.10%/49.00%, worst-device false-AI 71%/84%, balanced accuracy 77.60%/73.25% and covered
      accuracy 76.27%/71.01% fail. AUC is 90.24%/86.89%. Report SHA `10fc0649...5573`; retry zero.

#### E49-D1 — ungated current-generator diagnostic while Datapoint review is pending

This is an AI-only stress test, not a replacement E49 final. Its Turkish text/sign content cannot
measure authentic-photo false positives, AUC or balanced accuracy and must never be combined with
unmatched random REAL photos to manufacture an easy binary result.

- [x] Pin ungated CC-BY-4.0 `fge-auto/dotting-test` revision
      `0bcc6877c7d23f4e615b5470f06b1c00e7db7311`. Bind 160 target +32 reserve images for each of
      GPT Image 2, Nano Banana 2, FLUX.2 Pro, Ideogram 4 and Seedream 5.0 Lite by deterministic
      request-id hash before image transfer. Preserve attribution and provider-output caveats.
      **Bound result:** 960 exact files /23,936,830 bytes; contract SHA-256 `170f70db...ed36` and
      reserve-identity SHA-256 `9637626d...f5a`. Image and model-score counts remain zero.
- [x] Download only the 960 bound WebP files to LaCie with per-file byte/format checks, resume and a
      512 MiB stop. Decode, exact/dHash-audit protected roles and freeze the first 160 clean parents
      per model without detector access. Derive paired `social_q75` children exactly as E49.
      **Transfer checkpoint:** 960/960 files and all 23,936,830 expected bytes are present with exact
      per-file SHA-256. **Freeze result:** all 960 decode, six identities are excluded by the frozen
      overlap rules, and the reserve still supplies exactly 160 clean parents per model. Manifest
      SHA-256 `048572a4...ccc9` binds 800 publisher originals plus their 800 deterministic Q75
      children. Detector/model-score count remains zero.
- [x] Score only frozen E43-S with the existing E50 binary/selective cuts. Commit both raw streams
      before opening metrics; report pooled and per-model AI recall, automatic AI-decision rate and
      original-to-Q75 recall loss. Diagnostic gates are coverage=100%, pooled recall>=80%, worst
      model recall>=60% and each condition's worst-model result disclosed.
      **Pre-score lock:** contract SHA-256 `d567965d...1cf9` binds the exact 1,600-row manifest,
      E43-S artifact `a3aec445...7390`, DINOv2-S weights `04d27f34...0081`, binary threshold
      `0.07940196245908739` and selective REAL cut `0.011505939625203613`. Eighteen focused E49
      tests pass. **Raw-score lock:** all 1,600 observations scored with complete coverage; the
      unopened 326,693-byte stream hashes to `c97b02a4...fa90`. Aggregate metrics remain zero.
- [x] Archive pass or failure without retuning. A pass adds current-generator evidence only; a fail
      may pre-register a successor experiment, but neither outcome promotes Module 1 or consumes the
      publisher-separated E49 final. Dotting images remain forbidden from training in this branch.
      **One-shot result — PASS 6/6:** original/Q75 pooled AI recall is 97.38%/95.88%; the weakest
      model is GPT Image 2 at 91.25%/86.88%. Q75 costs only 1.50 recall points. All other generators
      remain at least 96.25% after Q75. Threshold/retry counts are unchanged/zero; report SHA-256
      `bb62ad92...d77b`. This strongly supports modern-generator transfer but leaves balanced E49 open.

#### E49-B — ungated OpenFake fallback qualification, before detector access

E49-A remains frozen, but manual Datapoint approval must not be the only route to completion. This
fallback is chosen from licence, source separation, date/family coverage and transfer feasibility—
never E43-S scores. It becomes a final candidate only if every pre-score qualification below passes.

- [x] Pin ungated CC-BY-NC-4.0 `ComplexDataLab/OpenFake` revision
      `3fd1109dc3258874243fa31c5bda9ee24260163b`, `core/test` and its exact 91,398 rows. Use only
      official Hugging Face Dataset Viewer `/rows`, whose asset path must embed that exact revision;
      never download a 5+ GB Parquet shard. **Pre-result reliability amendment:** Viewer cached
      51,900 ordered rows but repeated 429/502/503 responses prevented a dependable full count.
      Exact-revision HTTP byte ranges may therefore project only `label`, `model`, `type` and
      `release_date` from the 13 source Parquets. Range bytes must be counted; image/prompt columns,
      row order, model cells, quotas, rank and stop rule may not change.
- [x] Scan metadata in deterministic 100-row pages, cached without prompts or expiring asset URLs,
      until every exact model cell has 192 rows: GPT Image 2, Nano Banana Pro, Seedream v5.0,
      FLUX.2 Klein 9B and Midjourney 7. Require label `fake` and non-video type; freeze 160 target
      +32 reserve per model by namespace hash. No image or detector access during selection.
      **Qualification result — FAIL before selection:** complete eligible populations are GPT Image 2
      470, Nano Banana Pro 60, Seedream v5.0 372, FLUX.2 Klein 9B 8,093 and Midjourney 7 3,586.
      Nano cannot supply the preregistered 192; selected rows remain zero.
- [x] Stop before resolving any asset URL, byte total or image because the identity qualification
      failed. Exact range projection transferred 2,597,624 metadata bytes in 650 requests and cross-
      validated 52,600 Viewer rows while avoiding 67,649,942,401 source-Parquet bytes.
- [x] Archive E49-B without substitution, model access or metric. No E49-B final contract exists;
      image bytes and detector scores remain zero. A successor may replace the underfilled cell only
      under a new preregistration. E49-A is unchanged and Dotting remains diagnostic-only.

#### E49-C — capacity-repaired OpenFake successor, before identity selection

E49-B failed only because Nano Banana Pro has 60 eligible images, not because of a detector score.
E49-C is a new experiment and hash namespace. It replaces only that underfilled cell with
`z-image-turbo`, which already has 6,876 eligible non-video rows in the independently cached first
52,600 rows. Every other source, quota, filter and final gate remains unchanged.

- [x] Pin the same OpenFake revision/config/split/licence and reuse only the already-validated
      continuous Viewer metadata prefix. Freeze 160 target +32 reserve rows each for GPT Image 2,
      Z-Image Turbo, Seedream v5.0, FLUX.2 Klein 9B and Midjourney 7 under namespace
      `E49_C_OPENFAKE_V1`, stopping at the first complete 100-row page where all cells reach 192.
      **Identity freeze:** the stop is row 46,600. Cell populations there are GPT 236, Z-Image
      6,111, Seedream 192, FLUX 4,068 and Midjourney 1,791; exactly 192 each are frozen. Contract
      SHA-256 `0abae56a...d702`, reserve identity `f9f7bf74...69ec`; zero new network/image/score bytes.
- [x] Commit the 960 exact identities before resolving any asset. Then fetch fresh revision-bound
      Viewer URLs only for those identities, bind response type/dimensions/exact byte lengths and
      require OpenFake plus 2,706,581,778 Commons bytes to remain within 4 GiB.
      **Feasibility PASS:** 960/960 HEADs bind 241,736,938 OpenFake bytes. Combined expectation is
      2,948,318,716 bytes, leaving 1,346,648,580 bytes below the 4 GiB stop. Contract SHA-256
      `7b71449e...1415`; signed URLs stored zero, image-body/model-score bytes zero.
- [x] Download only the bound OpenFake assets to LaCie with resume and exact receipt, decode and
      exact/dHash-audit every reserve against protected roles, then freeze the first 160 clean rows
      per family. Received Viewer bytes are the declared publisher transport; derive paired Q75 only
      after the parent manifest freezes. **Decode amendment from first transfer, before scoring:**
      row 8,770 has a `.jpg` Viewer path/generic MIME but valid PNG bytes. Preserve the body and
      accept only decoded JPEG/PNG/WebP; report per-source format/geometry so this shortcut cannot
      be hidden. **Pre-realization implementation checkpoint:** the receipt-bound realization now
      validates all 960 payload hashes/formats/geometries, compares SHA-256+dHash against protected
      roles plus Dotting and StyleGAN2 component manifests, applies the frozen rank order and creates
      deterministic 1080-long-side JPEG-Q75 children. It requires exactly 160 clean parents per
      family and refuses partial quotas; 21 focused E49 tests pass before production execution.
      **Resume-performance amendment:** after 521 verified payloads, the single-page resolver was
      measured as the bottleneck and stopped cleanly. Resume keeps every identity/byte rule and the
      already-established maximum of two Viewer requests, but resolves them in 24-page batches
      before the same eight body workers. No completed payload is fetched twice.
      **Frozen-geometry correction:** row 43,863 is the sole reserve image above the inherited
      50 MP safety default (6,144 x 11,008 =67,633,152 pixels), a dimension already bound before
      body access. Raise the decoder ceiling only to that exact frozen maximum; do not remove or
      replace the row. The HTTP 200 body was rejected before admission and no score exists.
      **Transfer complete:** 960/960 exact files and all 241,736,938 contracted bytes validate;
      every family supplies 192 reserves. Decoded formats are 958 JPEG and two PNG. Receipt
      SHA-256 `4dfb942c...26c2`; signed URLs stored zero and detector-score count remains zero.
      **Clean paired freeze complete:** all 960 decode with zero failure. Twenty-six repeated
      Seedream payload identities are excluded by both exact and dHash duplicate checks; the reserve
      still yields exactly 160 parents per family. The 800 parents plus 800 deterministic Q75
      children freeze at manifest SHA-256 `38048803...7442`; protected overlap and scores are zero.
- [x] Assemble the complete balanced E49-C manifest: 1,000 native-camera REAL, 800 modern OpenFake
      AI and 200 local StyleGAN2 AI parents, each paired original/Q75. Commit one-shot E43-S score
      streams before opening metrics and require all existing 20 gates without threshold/source repair.
      **Evaluator source lock:** replace the obsolete unconsumed Datapoint source labels in the
      pre-score quota validator with the five exact E49-C OpenFake labels; all ten device quotas,
      StyleGAN2 quota, thresholds and 20 gates remain unchanged. This occurs before the final
      manifest and before any E49-C detector score.
      **Assembly implementation checkpoint:** the final builder binds the three component manifests,
      creates and collision-checks the 200 StyleGAN2 Q75 children, revalidates every observation's
      SHA-256 and geometry, and invokes the exact 2,000-parent/4,000-observation source validator.
      It archives per-condition formats and per-source original geometry before model access; twelve
      focused final/evaluation/component tests pass.

#### E51 — authentic-safety successor after consumed E49-C (planned before new data/model work)

E49-C is not repairable, but it narrows the problem. A threshold-only change is mathematically
insufficient: at 10% FPR, the frozen ranking reaches only 72.30% AI TPR on originals and 58.80% on
Q75, below the 80% gate. Requiring both existing paired scores to vote AI would still leave 32.80%
REAL false-AI while recalling 93.50% AI. JPEG originals fail slightly more than MPO (39.90% versus
36.15%), and log-resolution correlation is only 0.056 despite a 70.27% FP pocket above 20 MP.
Therefore the successor needs a better authentic/compression representation, not a leaked E49 cut,
format rule or resolution heuristic. These are diagnosis-only observations, never model selection.

- [x] **Freeze E49 protection and reproduce the failure diagnosis.** Bind final manifest/raw-score/
      report hashes; report per-device, format, resolution-bin and paired-consensus behavior. Emit no
      candidate threshold and forbid every E49 identity—including unused reserves—from later TRAIN,
      CAL, DEVELOPMENT or successor-final roles.
      **Completed:** the immutable manifest/raw-score/final-report hashes reproduce, all 4,000 rows
      and 2,000 parent pairs join exactly, and the machine-readable diagnosis is frozen at
      `3e5caa86...bd70f`. It creates zero candidate thresholds and zero new model scores. The measured
      10%-FPR TPR remains 72.30%/58.80%; paired AND remains 32.80% REAL false-AI at 93.50% AI recall;
      JPEG/MPO and resolution findings reproduce. This closes diagnosis only, not model selection.
- [x] **Audit new REAL sources before downloading images.** Compare official RAISE, Dresden/IMAGINE
      camera collections and a current-phone source for licence, native-vs-publisher transport,
      device/scene grouping, resolution and download size. Choose disjoint TRAIN/CAL and a separate
      publisher/device DEVELOPMENT source. Require >=4,000 TRAIN, >=1,000 CAL and >=2,000 DEVELOPMENT
      REAL parents where feasible, with fixed original/Q75 pairs. No E49 metric may rank individual
      source rows. Relevant directions: [real-only one-class detection](https://arxiv.org/abs/2311.00962),
      [B-Free content alignment](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html),
      [AIDE hybrid visual/noise experts](https://openreview.net/pdf/67e6139d293501496907c5dc7468eb9a370685dd.pdf) and
      [MAFL source/content-bias suppression](https://arxiv.org/abs/2604.12353).
      **Metadata-only checkpoint:** the complete public SCMI30 v2 inventory reproduces 9,937 native
      JPEGs /30 device ids /35,592,810,773 image bytes and supports individual-file selection; it is
      the leading device-disjoint TRAIN/CAL candidate under CC-BY-NC-ND research terms. Open
      CC-BY-4.0 SCIMD-17 is only a 224x224 resize corpus and may enter auxiliary TRAIN hard negatives,
      never native CAL/DEVELOPMENT. RAISE is valid native RAW but ~350 GB/three cameras; Dresden,
      IMAGINE and SOCRatES remain blocked by unavailable official transport, unverifiable TLS/explicit
      terms, or signed agreement. Evidence `9ddab57a...c4883`; image downloads remain forbidden until
      an independent, explicitly licensed DEVELOPMENT publisher is bound. The audit is completed by
      the following IEEE/Datapoint route bind; this metadata-only checkpoint remains unchanged.
- [x] **Bind the practical E51 data route before payload access.** Reuse only historical TRAIN-role
      parents for the base fit and add SCIMD-17 solely as resized-real hard negatives. Freeze native
      SCMI30 v2 to CAL at 40 parents/device—20 Random plus 20 Similar for each of 30 normalized device
      ids, 1,200 parents total—without using it in TRAIN or DEVELOPMENT. Use all 2,640 hidden-label
      IEEE SP Cup 2018 test camera images as independent REAL DEVELOPMENT: report its 1,320 unaltered
      and 1,320 postprocessed cells separately because camera ids are hidden. Use approved Datapoint
      as the independent current-generator AI DEVELOPMENT component, never TRAIN/CAL/E52 final.
      IEEE payload access requires the user to accept the archived competition rules; until that
      one-click gate is complete, bind metadata/contracts only and download zero image bytes.
      **Bound, still zero-payload:** exact contract `975e8164...15e4` freezes 1,200 SCMI30 CAL
      parents (30 devices x20 Random+x20 Similar; 4,247,339,334 expected bytes), every one of the
      IEEE test split's 2,640 REAL DEVELOPMENT TIFFs (1,320 unaltered +1,320 postprocessed;
      837,665,909 bytes), and a 920-row Datapoint reserve for five current generators. The AI
      reserve uses the same 23 score-blind prompts in each of eight categories for every model;
      realization must retain 20/category/model =800 parents. Seven exact Parquet shards total
      3,220,281,593 transfer bytes while the selected payloads total 654,005,247 bytes. The IEEE
      archive remains HTTP 403 until the user accepts Kaggle's competition rules. No image or model
      score has been opened.
      **Access + transfer-method checkpoint:** the user's rule acceptance now passes an official
      81,853-byte metadata probe. Before the first image, a restart-safe six-worker transfer binds
      contract `975e8164...15e4`, rejects every non-test path, verifies all expected bytes, TIFF
      decode/geometry and SHA-256, preserves unaltered/postprocessed cells and emits zero scores.
      The method is committed before payload execution.
      **Publisher-container correction before admission:** the first bounded transfer stopped on the
      first row because the `.tif` path actually contains a 512x512 RGB PNG body. Eight concurrent
      bodies reproduced the same signature and geometry; none entered the payload root or a model.
      The gate now requires this exact publisher reality—PNG +512x512—instead of trusting the suffix,
      while retaining the same identities, bytes and roles.
      **Rate-limit correction:** the first valid resume admitted 494 exact rows, then Kaggle returned
      HTTP 429 before any receipt could be sealed. Those files remain restart-safe and unscored.
      The transfer now uses two workers behind one global 0.8-second request gate (~75 requests/min)
      and honors `Retry-After` or a bounded 60–300 second backoff before continuing.
      **Transport optimization supersedes the paced endpoint:** Kaggle's official 11,333,585,079-
      byte ZIP supports HTTP Range and its central directory reproduces 5,391 members /
      11,447,649,387 expanded bytes. The downloader now reads only the bound `test/test/*` member
      ranges, validates ZIP CRC plus the existing byte/decode/SHA gates and never transfers the
      2,750 excluded training images. The 520 already admitted rows remain valid and restart-safe.
      **IEEE transfer complete:** 2,640/2,640 selected REAL DEVELOPMENT bodies reproduce exactly
      837,665,909 bytes and identity hash `fc3657dd...fb05`, split 1,320 unaltered +1,320
      postprocessed. The range reader transferred 836,795,134 compressed member bytes from the
      official ZIP; receipt `09188d49...3794` records every decoded SHA-256. Model scores remain zero.
      **Datapoint transfer method:** while the Kaggle quota cools down, the next acquisition gate is
      frozen without payload access. It verifies the manual-gated repository at exact revision,
      requires all seven contracted shard byte counts (3,220,281,593 total), downloads only those
      paths restart-safely, hashes the complete Parquets and explicitly leaves image columns unopened.
      **Datapoint transfer complete:** all 7/7 pinned shards reproduce 3,220,281,593 bytes and full
      local SHA-256 values. Receipt `18b8326a...bad1` retains 920 paired reserves /800-parent target;
      image columns and model scores remain unopened/zero.
      **SCMI30 CAL transfer method:** exact v2 bulk-ZIP metadata reproduces 9,940 members /
      35,592,872,377 expanded bytes. A score-blind range reader is bound before payload to fetch only
      the 1,200 selected CAL members (4,247,339,334 bytes), in archive order, then require byte count,
      ZIP CRC, safe JPEG decode, geometry, SHA-256 and the 30-device/branch quotas.
      **Pre-payload path correction:** the first run stopped before transfer because Similar images
      legitimately add scene subfolders below the bound device (`objects/`, etc.). The safe-path gate
      now fixes root +branch +exact contracted device while allowing nested descendants; traversal,
      suffix and device substitution remain rejected. Identities and byte quotas do not change.
      **Measured transport parallelism:** one Range reader admitted 59 exact rows /231 MB but would
      take roughly an hour. It was stopped cleanly; the lone inadmissible partial was removed before
      restart. The same frozen archive route now uses four independent readers over four contiguous
      archive-order partitions, with collision-proof staging names and unchanged per-file
      CRC/decode/hash gates.
      **SCMI30 transfer complete:** 1,200/1,200 rows reproduce 4,247,339,334 bytes, 30 devices x40,
      Random/Similar 600/600 and make/model EXIF 1,200/1,200. Receipt `01cc5921...f33e` and ordered
      identity digest `c47d411f...1d12` bind the clean result; no partial or model score remains.
      **SCIMD-17 method before payload:** the auxiliary TRAIN-only downloader now pins Zenodo record
      17317613, DOI/version/CC-BY-4.0, exact 174,438,734 bytes and MD5. It supports restart-safe
      transfer, validates archive paths/expansion and records that image bodies and model scores
      remain unopened. Commit this gate before the first archive byte.
      **SCIMD-17 transfer complete:** the pinned archive reproduces its MD5 and SHA-256
      `ef1fe3e7...0201`; safe central-directory audit finds 17,620 image files /172,781,180 expanded
      bytes. Receipt `8b38fa82...b230` records zero decoded bodies and scores. Realization may now
      select only score-blind TRAIN hard negatives; the full archive may never become CAL/DEV/final.
- [ ] **Bind only two practical successor families before fitting.** E51-A reuses frozen DINOv2-S
      features but refits a source-balanced head with new camera originals plus Q75/JPEG/resize
      hard negatives. E51-B adds a compact fixed residual/DCT statistics branch to the same features,
      inspired by AIDE, and adversarially/source-balances the head rather than fine-tuning a huge
      backbone. A real-only distance arm may only turn a positive into UNCERTAIN, never certify REAL.
      **Pre-realization CAL repair:** the original route had independent REAL CAL but no AI CAL,
      making balanced threshold selection impossible. Before any new body decode or score, reserve
      20 rows from each of 18 sufficiently populated historical AI TRAIN sources (360 total), remove
      them from candidate fitting and pair them with Q75. Datapoint remains DEVELOPMENT-only.
      SCIMD-17 freezes 120 score-blind candidates/device for a 100-clean/device TRAIN target.
      **TRAIN/CAL realization method:** audit all 2,040 SCIMD reserves against protected exact/dHash
      identities and retain 100/device; rebuild TRAIN from historical TRAIN rows minus the 360
      AI-CAL holdouts. Verify all 1,200 SCMI30 and 360 AI-CAL parent bytes, then make deterministic
      original/Q75 pairs. DEVELOPMENT remains unopened until a CAL winner exists.
      **Pre-score dHash compatibility correction:** the first realization wrote no CAL/evidence and
      stopped on the first historical AI parent: SHA-256, bytes and geometry matched, but E42 used
      the inverse left-to-right comparator from the current shared helper. Preserve the pinned E42
      value for reused AI-CAL and protect both 64-bit conventions for new-source overlap checks.
      **Perceptual-collision confirmation:** the corrected run then stopped on an SCMI30 row whose
      dHash is all-zero/all-one and collides with two unrelated protected images. Exact SHA differs;
      pHash distances are 31 and 36, far above the <=4 near-duplicate threshold. Treat exact dHash
      as a candidate filter and require pHash confirmation; uncheckable or <=4 matches still fail.
      **Identity-audit amendment:** the full 1,200-row scan confirms exactly one near duplicate, the
      publisher's `D04_black.jpg`; 25 other coarse dHash candidates clear at pHash distance 22–36.
      Bind five unselected D04-Similar rows by the original namespace rank, download only these, and
      let the first clean identity replace the black frame while preserving 30 x40 and 600/600.
      **Amendment complete:** all five bound reserves /7,710,716 bytes passed acquisition and identity
      audit; the first ranked clean row `D04_nat_45.jpg` replaces the black frame. Receipt
      `66e3d063...9722`; rejected reserves and model scores are zero.
      **TRAIN/CAL realization complete:** TRAIN freezes at 5,978 parents (4,035 REAL/1,943 AI),
      including 1,700 SCIMD hard negatives with zero decode failures and two conservative dHash
      exclusions. CAL freezes 1,560 parents (1,200 REAL/360 AI) and exactly 3,120 original/Q75
      observations. Manifest hashes are `41444640...77ef` and `60688291...2356`; scores remain zero.
      **2026-09-07 remote pre-fit safety checkpoint (before implementation):** verify every local
      TRAIN/CAL byte hash and recompute one canonical perceptual convention from decoded pixels.
      Check parent, encoded-byte and decoded-pixel overlap across roles; confirm near-dHash
      candidates with the fixed pHash rule, without detector scores. Do not infer hash compatibility
      by bitwise inversion: equal adjacent pixels and different resampling invalidate that shortcut.
      Audit SCIMD filename/provenance anomalies against publisher metadata rather than relabelling
      by filename. Preserve the frozen manifests, emit a separate fail-closed readiness receipt,
      and keep training disabled on unresolved overlaps. Explicitly account for E49 unused reserves;
      passing only a TRAIN/CAL check is not permission to skip protected-role admission.
      **Reserve closure policy (before scores):** join metadata-only parent references to already
      protected bodies, include full local Commons/OpenFake/Dotting/StyleGAN2 reserves, and include
      superseded E49-v1 identities. An identity never downloaded is still forbidden, but cannot be
      pixel-compared offline; report that limitation, rather than pretending its pixels were
      checked or demanding a new download. All available reserve bodies must be checked. No
      E51 input may share any reserved identity/known encoded hash; no unexplained body-bearing
      reference may disappear from coverage. This is admission accounting, not a relaxed model gate.
      **2026-09-09 offline continuation:** the full 9,098-observation pre-fit check completed,
      with zero cross-role matched parent pairs (receipt `02180078...77a2`). Archive it without
      repeating or weakening the check. Next, resolve the protected populations from existing
      local files and ZIP members: historical relative paths are relative to their manifest
      directories, not the process working directory. Freeze a body-locator inventory, retain
      missing/ambiguous coverage explicitly, and never download a replacement on mobile data.
      Only after protected-pixel/reserve admission closes may E51-A/B fit and CAL selection run.
      Model generation is deferred: any future generated-image collection must record provenance,
      prompts and generator version, with TRAIN/test separation fixed before images are scored.
      **Offline locator checkpoint complete:** 117,898 distinct local body locations resolve
      (73,165 files +44,733 ZIP members), totaling 50,577,346,337 existing image bytes to verify;
      zero unresolved body locations and zero downloads. Every resolved location has a prior
      SHA-256 after merging identical-location references. The 9,800 metadata-only rows are
      retained separately (including duplicate parent references/reserves), not silently counted
      as pixel-audited or assumed to be 9,800 missing images. Locator hash `a51cb457...eb81`.
      Next: join metadata-only identities, review superseded reserves, then canonical protected-
      body comparison. Training and the served model remain unchanged until admission closes.
      **Protected-pixel execution (2026-09-09, before code/run):** read the frozen 117,898 local
      locations, verify expected size/SHA-256 (and ZIP CRC), and compute canonical RGB/dHash/pHash63.
      Use bounded worker batches and a local SQLite fingerprint cache so interruptions do not
      require recomputing every fingerprint; cached fingerprints never waive current byte checks.
      Compare new SCIMD/SCMI TRAIN/CAL against all protected bodies, and every E51 TRAIN/CAL row
      against E49 bodies. Preserve overlap candidates and missing coverage without relabelling or
      threshold selection. Metadata-only/superseded reserves remain a separate admission gate.
      **Protected-only size exception, before restart:** the run stopped after 64,000 locations
      because six hash-pinned historical RR test images exceed 100 MP (maximum 178,562,880 pixels).
      Permit only those six exact SHA-256/geometry pairs, process them serially without resizing,
      and keep the default 100 MP limit for every other image and all E51 TRAIN/CAL inputs.
      Preserve Pillow's independent bomb checks and the complete decoded RGB fingerprint.
      **Fixed E51-B implementation specification (before feature code):** use the same three
      224x224 global/texture crops as E51-A; add eight normalized DCT energy bands and eight
      residual/gradient statistics per crop, aggregated by crop mean/std (32 values). No filename,
      EXIF field or format/resolution scalar is a classifier input. Compare A versus A+32 only at
      fixed C=0.01 with parent/source/class-balanced training weights, train-fitted standardization
      and unchanged CAL gates. Source/pipeline shortcuts may still survive pixel preprocessing;
      this feature branch is a hypothesis, not an assumed improvement. Unit tests may use generated
      numerical arrays; no real-data feature extraction/fit until protected admission closes.
- [ ] **Select on new grouped CAL, then open fresh DEVELOPMENT once.** Group by device/scene/parent;
      require both original and Q75 to meet AUC >=0.90, BA >=0.85, pooled REAL FP <=10%, worst-device
      FP <=20%, AI recall >=80%, worst-generator recall >=60%, automatic coverage >=80%, covered
      accuracy >=95% and uncertainty <=20%. Archive failure; no DEVELOPMENT-informed retuning.
- [ ] **Only a DEVELOPMENT pass may bind E52.** Datapoint is now irrevocably assigned to E51
      DEVELOPMENT; select entirely new REAL and current-AI publishers for E52 final.
      E52 repeats the paired >=2,000-parent, 20-gate protocol once. Module 2 remains planning-only
      until a successor earns Module-1 v1; its old masks/results stay protected but documented.

## Two-module completion contract — Module 1 proof before Module 2 (2026-09-04)

### Stage A — finish and prove Module 1

1. Complete E48 in strict order: bind frozen score identities; score only FIT+CAL; fit authentic-
   percentile maps on FIT REAL; select one monotone expert set and threshold on CAL; commit the
   candidate; only then score/open DEVELOPMENT once. A failure is archived and cannot be repaired
   on those rows.
2. A passing E48 does **not** finish Module 1. Bind E49 as the comprehensive final before model
   access: publisher/collection-separated from every TRAIN/FIT/CAL/DEVELOPMENT source, >=2,000
   balanced rows, at least two REAL pipelines/transports and at least five AI source families
   spanning diffusion and GAN. Exact/dHash decontamination, generator/source reporting, native and
   fixed social-recompression columns, 10,000 stratified bootstraps and the full E46 ten-gate
   contract are mandatory. Score once; no retry or source removal.
3. Only an E49 all-gate pass creates **Module-1 v1** and permits replacing the web-demo model.
   Freeze its model hashes, preprocessing, threshold, uncertainty policy and benchmark report.
   Until then the existing served result remains unchanged.

### Stage B — resume Module 2 after Module-1 v1

Module 2 v1 is explicitly **AI-assisted local editing/inpainting**, not universal Photoshop/splice
detection. Preserve the useful E17/E18 lessons instead of repeating the failed branch:

- E17's absolute tile signal is real but small: CocoGlide tile AUC 0.648, image AUC 0.721 and IoU
  margin +0.155 over random. The old filter retained only 35/120 images; half-tile stride is the
  first zero-download repair and must disclose every skipped mask.
- Raw IoU is invalid as a headline because it rewards large masks. Report pixel ROC-AUC and AP,
  mask-size-stratified F1/IoU, IoU-minus-random, image AUC and pristine false-localisation.
- The eight classic-splice/copy-move sets are specificity controls, not positive training data.
  ELA's controlled JPEG splice reached 0.719 but the PNG compilation erased its compression input;
  close that branch for Module 2 v1 rather than calling the method generally broken.

#### Module 2 execution ladder — planning only until E49 passes

- [ ] **M2-0 evidence audit and role split.** Inventory every CocoGlide image/mask/auth pointer,
      disclose missing/invalid masks instead of silently skipping them, deduplicate by parent scene
      and freeze TRAIN/CAL/FINAL by complete scene. Keep classic-splice sets and fully generated
      images as named negative/specificity controls. No image may cross roles through its authentic
      parent, mask derivative or alternate encoding.
- [ ] **M2-1 repair the evaluator before fitting.** Replace E17's 36-tile cap and `coverage >=.5`
      survival filter with deterministic half-tile stride and overlap-averaged pixel maps. Report
      per-image then image-macro pixel ROC-AUC/AP; choose localisation threshold on CAL only; report
      mask-size-stratified F1/IoU, IoU-minus-matched-random, image AUC and authentic false-localised
      area. Bootstrap complete images/scenes, never correlated tiles.
- [ ] **M2-2 zero-training baselines.** Re-run the old 128 px absolute detector, E43-S-compatible
      local crop evidence and the measured residual/noise-energy signal through the repaired
      evaluator. ELA remains a JPEG-only diagnostic control, not a universal branch. This establishes
      what the learned model must beat without spending a validation set on architecture choice.
- [ ] **M2-3 learned AI-edit localiser.** Freeze DINOv2-S and expose dense intermediate patch tokens
      from the same representation family that made E43-S succeed. Compare only predeclared small
      heads: linear/1x1 dense head, noise-energy fusion and a shallow upsampling decoder. Train on
      mask-derived soft targets, source/scene-balanced sampling and scale/JPEG augmentation; select
      on CAL and score source-held FINAL once.
- [ ] **M2-4 join modules without corrupting either proof.** Module 1 answers fully generated vs
      authentic/insufficient; Module 2 runs as a separate local-edit analysis and returns a heatmap
      only when spatial evidence passes its own threshold. Module-2 findings may pre-register a new
      Module-1 successor, but the frozen Module-1-v1 artifact/cuts and E49 rows never change. Require
      both modules to retain independent model cards, hashes, gates and failure disclosures.
- Test the measured noise-energy clue (AI-filled region 0.0164→0.0088) and dense DINO patch tokens
  beside the absolute tile score. Fully re-rendered ChatGPT-family edits must be labelled
  AI-regenerated, never promised a local mask.
- Train a small dense/localisation head with exact masks; do not alter Module-1 v1 weights. FIT/CAL/
  DEVELOPMENT split by source image and generator; TGIF/TGIF2 or another publisher-separated,
  mask-preserving set is the preferred untouched final.

### Cross-module feedback without regressions

After Module-1 v1, effort shifts roughly 70% to Module 2 and 30% to Module 1 maintenance. A Module 2
finding may open a new Module 1 experiment only when recorded as a data, preprocessing,
representation or decision-layer hypothesis and tested on fresh evidence. Module-1 v1 remains the
served control until a successor repeats DEVELOPMENT plus an independent final; no Module 2 mask,
threshold or failure may silently tune it. Shared code may include decoding, transport simulation,
DINO feature extraction, audit/provenance and UI components, while artifacts, thresholds and claims
remain separate.


## Current execution slice — R1c threshold repair and external benchmark (2026-08-27)

The immediate product defect is no longer ambiguous: E32/R1b ranks the existing modern-AI pool
well but its internally fitted `0.125935` threshold does not transfer to independent camera
pipelines. The next candidate therefore changes **only the threshold**, using genuinely new data;
it does not add an ensemble, retrain the CF-ViT head or use the already-consumed owner gallery/IPN
scores. In parallel, the project's loose Desktop assets are consolidated without touching any
unrelated personal, academic or EOE material.

### D0 — consolidate only proven PixelProof Desktop assets

- [x] Keep the active Git checkout at `~/Desktop/ai-image-detector`; moving the live workspace adds
      no model value and would invalidate the current app/tool path. Create
      `~/Desktop/PixelProof Workspace/{Documents,Legacy Datasets,Samples}` and move only items whose
      content or recorded history proves PixelProof ownership. Never use a broad glob.
- [x] Move the known legacy dataset directories (`archive`, `archive1`, `defactify`,
      `defactify_test`, `e23b_nist_capped`, `e23c_degraded`, `e24_iphone_capped`,
      `e25_modern_probe`, `e27_pool`, `genimage`, `genimage_split`, `manipulation_test`) and the
      original `archive.zip` into `Legacy Datasets`; move the owner gallery, empty `ai gen foto`
      staging folder and the verified ChatGPT sample into `Samples`. Move only the closed report,
      plan and presentation copies into `Documents`.
- [x] Do not move the currently open `PixelProof_Sunum.pptx` or its PowerPoint lock file; defer both
      until PowerPoint is closed. Do not touch screenshots/forms containing personal information,
      `Improvements.md`, or any non-PixelProof folder. Update the eight tracked legacy path defaults
      that would otherwise break, then verify exact source/destination counts and Git references.

### D1 — freeze the benchmark hierarchy and honest meaning of “pass”

- [x] Treat NIST GenAI Image-D as the highest-authority future **external blind evaluation**. It
      requires participant registration/data terms, forbids inspecting or tuning on the test set,
      and reports ROC-AUC, EER, TPR at a fixed FPR and target/non-target Brier scores. NIST defines
      metrics, not a universal certification score, and explicitly does not endorse participants;
      the project must never claim “NIST approved/passed.”
- [x] Use NTIRE 2026 only as a published competitive reference until its missing dataset licence is
      clarified. Its 10k clean validation ZIP (3,185,123,401 B), 10k hard/transformed ZIP
      (804,902,498 B) and labels are public, but public access is not a reusable licence. Do not
      download those image bytes under the project's fail-closed licence policy.
- [x] Select ICCV 2025 RRDataset as the immediate open robustness benchmark: official Zenodo record
      `14963880`, CC BY 4.0, a 2,163,176,547-byte original train/validation archive and a
      20,117,869,400-byte test archive spanning original, multi-platform transmission and physical
      re-digitization. Its authors report a best detector overall accuracy of 89.59%; that is a
      research reference, not a vendor-independent certification threshold.
- [x] Pre-register project-owned gates rather than inventing an industry standard. **Working
      candidate:** all files counted, ROC-AUC >=0.85 and balanced accuracy >=0.80. **Internship
      success:** ROC-AUC >=0.90, TPR@FPR=10% >=0.80, EER <=0.15, balanced accuracy >=0.85,
      authentic macro FPR <=10%, worst sufficiently sized authentic pipeline FPR <=20%, AI macro
      recall >=80% and weakest sufficiently sized AI family recall >=60%. Report calibration
      (Brier target/non-target) but do not gate it until the candidate emits calibrated
      probabilities. NTIRE robust AUC around 0.93 is a competitive reference and about 0.97 is
      top-challenge territory, not the minimum for this internship prototype.

### D2 — acquire with receipts; never tune on the locked test

- [x] Before image bytes, freeze source URL, revision/record id, CC licence, filenames, exact
      published sizes, MD5 and destination under
      `/Volumes/LaCie/pixelproof-datasets/e33_rrdataset/`. Require >=100 GiB free, resumable
      `.partial` transfers, exact final checksum and archive safety inventory. Git stores only
      compact receipts/aggregate evidence.
- [ ] Download and audit `RRDataset_original_train_val.tar.gz` first. Decode and label-audit every
      member, infer no label from an unexplained number, preserve original/transmission/redigital
      parent groups, and decontaminate against protected PixelProof roles. Only its declared
      train/validation portion may form `R1C_CAL`; no RR test row may select a threshold, transform
      or retry.
      The passed pre-score inventory contains 1,250 REAL + 1,250 AI train and 250 REAL + 250 AI
      validation images. R1c-T uses only the official 500-row validation split. Filenames expose
      seven AI scenario groups (22–93 rows) but collapse REAL to one undisclosed pool, so the
      frozen minimum reportable group size is 20 and the REAL gate is aggregate—not a multi-camera
      transfer claim. IPN per-device and owner-gallery DEVELOPMENT must still verify transfer.
- [ ] Download `RRDataset_test.tar.gz` only after the R1c-T artifact/threshold contract is frozen.
      Inventory and extract safely, then open/score the official labels once. If transfer time is
      interrupted, preserve the partial and stop honestly; do not substitute an easier set after
      seeing any score.

### D3 — implement and evaluate R1c-T before another model change

- [x] Add one reusable E32 benchmark adapter that accepts a manifest with explicit
      `0=REAL, 1=AI`, parent/source/condition columns, counts decode/inference failures as failed
      rows, and reports ROC-AUC, EER, TPR@FPR=10%, balanced accuracy, confusion, per-source/per-
      condition rates and uncalibrated-score Brier diagnostics. Tests use synthetic scores/files.
- [x] Keep R1b backbone, CF head, preprocessing and score direction byte-identical. Select one
      R1c-T threshold exclusively on eligible `R1C_CAL` authentic rows under macro FPR <=10% and
      worst-pipeline FPR <=20%; use CAL AI rows only to reject a threshold below the pre-registered
      recall gates. Freeze the threshold, source receipt, score hashes and code revision.
      **CAL result: rejected.** All 500 rows scored, but ROC-AUC was 0.80728. The frozen R1b cut
      produced 82.8% REAL FP; the first aggregate-REAL-safe cut was 0.998400 at 10.0% REAL FP but
      only 60.52% AI scenario-macro / 26.88% worst-scenario recall. It fails both the working AUC
      tier and the 80%/60% AI gates. This is a rejection receipt, not a deployable threshold.
- [ ] Reopen IPN and the owner gallery only as consumed DEVELOPMENT regression. Pass requires
      IPN worst-device and owner FPR <=20% while the frozen internal modern-AI macro recall remains
      >=80% and weakest family >=60%. They cannot move the threshold.
      Not opened: R1c-T failed CAL, so DEVELOPMENT cannot rescue or retune it.
- [ ] A DEVELOPMENT pass permits exactly one RRDataset locked test. Meeting the internship-success
      gate promotes R1c-T to the API/web path; a miss remains a documented working/rejected
      candidate according to the frozen tiers. Only a clean threshold-transfer failure opens the
      already-planned paired semantic+frequency R1c-P training path; ensembles and new encoders
      remain later hypotheses.
      No 20.12 GB locked-test byte was downloaded because the prerequisite candidate does not
      exist.

### D3.5 — evaluate official DDA before paying the 113 GB training-data cost

- [x] Freeze official NeurIPS 2025 `Junwei-Xi/DDA-COCO` at Hugging Face revision
      `8c9330a3...68fb`: Apache-2.0, one 4,301,452,066-byte ZIP, Xet SHA-256
      `8cd60077...9c24`. It contains MS-COCO validation reals and corresponding VAE reconstructions
      across five alignment variants. Official project documentation identifies this as an
      **evaluation benchmark**, not the training release; keep it locked and never fit on it.
- [ ] Download resumably to `/Volumes/LaCie/pixelproof-datasets/e34_dda_coco/`, verify size/SHA-256
      and run ZIP safety inventory, but do not extract/open members before a DDA candidate contract.
      The official DDA training set is Apache-2.0 but consists of ten 10,737,418,240-byte parts plus
      a 5,591,345,987-byte final ZIP (~112.97 GB); this violates the current minimum-data objective
      and is deferred to full home internet.
      **Paused safely:** 4,252,382,809/4,301,452,066 B (98.86%) exist as one prefix plus four range
      parts. No member has been opened. Do not fetch the missing 49,069,257 B until E36 CAL passes.
- [x] Freeze the official `Junwei-Xi/Dual-Data-Alignment` checkpoint at revision
      `4390d902...16c`, Apache-2.0, `DDA_ckpt.pth` 1,255,621,296 B / SHA-256
      `b27a31d3...e3e`. Vendor the minimal Apache inference modules with attribution, pin the
      offline DINOv2-L architecture (the checkpoint supplies every base tensor), reproduce
      center-crop 336 + published normalization and verify strict state/score direction with
      synthetic contract tests before production scoring.
- [x] Score official DDA first on consumed RR validation, IPN and owner gallery as DEVELOPMENT—no
      threshold fit on them. Use the checkpoint's published 0.5 decision cut for the first gate and
      report raw AUC/frontiers only as diagnostics. If authentic FP and AI coverage pass the frozen
      gates, open DDA-COCO once as its aligned benchmark and only then consider API/web promotion.
      If it fails, the honest next cost is official DDA training data or self-generated aligned
      pairs; do not train on DDA-COCO or hide the cost by calling it a training subset.
      **Measured result:** RR is strong (AUC 0.978192, EER 0.08, TPR@FPR10 0.92, balanced accuracy
      92.4%, REAL FP 6.4%, AI recall 91.2%), but the published cut fails transfer: IPN worst-device
      FP 36.25% and owner-gallery FP 34.76%. The candidate therefore fails DEVELOPMENT and
      DDA-COCO remains locked. A post-hoc curve finds a conservative region near 0.90, but every
      displayed value is contaminated by consumed DEVELOPMENT and is permanently ineligible.

### D3.6 — E36 clean DDA calibration and a genuinely unseen final gate

This is the shortest scientifically defensible route to a usable detector. E35 proves that the
representation ranks RR images well and that the dominant defect is operating-point transfer on
native camera pipelines. Do not discard DDA, deploy the post-hoc 0.90 value, mix old models, or pay
113 GB for training until this threshold-transfer hypothesis receives one clean replication.

- [x] **Freeze source registry before bytes.** CAL and FINAL must have disjoint parent images,
      capture sessions/devices, prompts/seeds and generator accounts. Perceptual-hash and exact-hash
      deduplication must also cover every earlier TRAIN/CAL/DEVELOPMENT manifest. Licences, URLs/API
      model versions, timestamps, expected counts and failure policy enter `DATASETS.md` first.
      Frozen selection SHA-256 `01eec03e...2dcc`; 600 CAL AI / 240 family-disjoint FINAL AI and
      five CAL / four FINAL REAL archives. Metadata freeze downloaded zero image bytes.
- [x] **Build a compact new CAL (target 1,200 parents, not 20,000 arbitrary files).** Acquire 600
      native authentic stills from at least six previously unseen phone/camera pipelines (target
      100/device, scene-balanced, original bytes/EXIF retained) and 600 clean AI images from at least
      six current production generator families (target 100/family, matched prompt/content strata,
      provider/model/version/seed receipt where available). VISION/Dresden-style academic native
      camera sources are candidates only after licence and overlap audit; web-resaved stock photos
      cannot stand in for native camera output. Current commercial AI families should be generated
      through pinned APIs at collection time rather than scraped from uncertain web labels.
      Realized as 471 native REAL +600 current AI =1,071 unscored parents. Exact/dHash overlap with
      prior E32 realizations is zero; every AI family has 100 rows and FINAL downloaded bytes are
      zero. Detailed manifest SHA-256 `4ed1b734...2e03`.
- [x] **Calibrate one threshold only.** Keep the verified DDA checkpoint, 336 crop, normalization
      and score direction fixed. Select the lowest threshold satisfying CAL real macro FP <=10% and
      worst-device FP <=20%, then require AI family macro recall >=80% and worst-family >=60%.
      Bootstrap 95% confidence intervals by parent; count decode/inference failures as errors.
      CAL must also retain ROC-AUC >=0.90, TPR@FPR10 >=0.80, EER <=0.15 and balanced accuracy
      >=0.85. No architecture, ensemble or second threshold may be selected from the same CAL.
      **Result: rejected.** At the first real-safe threshold `0.756332`, REAL device-macro/worst
      FP are 9.36%/20.0%, but AI family-macro/worst recall collapse to 27.67%/1.0%. ROC-AUC is
      0.58753, TPR@FPR10 0.285, EER 0.4267 and balanced accuracy 0.5895. The published 0.5 cut also
      fails both sides (REAL macro/worst FP 16.61%/35.0%; AI macro/worst recall 38.0%/6.0%). No
      E36 threshold is eligible and no FINAL byte may be downloaded for this candidate.
- [ ] **Freeze a new LOCKED FINAL set before scoring.** Minimum 160 native reals from four unseen
      device/session pipelines (40 each) plus 240 clean modern AI parents from six held-out
      model/version cells (40 each). Add deterministic JPEG, resize, screenshot/social-transmission
      derivatives, but split and bootstrap by parent so copies never inflate N. The old IPN,
      RR-validation and owner gallery remain diagnostic only and cannot be called final again.
- [ ] **One-shot promotion gate.** Require ROC-AUC >=0.90, TPR@FPR10 >=0.80, EER <=0.15,
      balanced accuracy >=0.85, real macro FP <=10%, worst real pipeline FP <=20%, AI macro recall
      >=80%, worst AI family >=60%, with 100% declared coverage. Report Wilson/bootstrap intervals
      and every subgroup; these are PixelProof preregistered gates, not NIST certification.
- [ ] **Only after a CAL pass:** complete the last 49,069,257 B of DDA-COCO, verify the full
      4,301,452,066-byte SHA-256, inventory safely and score it once as an aligned benchmark. A CAL
      or FINAL miss keeps the current web verdict unchanged and opens exactly one training path:
      content-matched real/reconstruction pairs using the official 112.97 GB DDA training release
      at home internet, or a smaller self-generated paired equivalent. NIST GenAI Image-D remains
      the later registered blind external evaluation; there is no universal public pass score.

#### E36-A source decision — frozen before image bytes (2026-08-27)

Primary-source inspection changes the generic 600/600 target into a more independent, lower-byte
design without weakening subgroup gates. Zenodo SCIMD-17 is rejected for this purpose because its
17 phone folders were pre-resized to 224×224; it is not native gallery-like evidence. CSAFE's
remaining 18–29 GB model archives are deferred because their scenes/collection overlap the S21 and
iPhone14 training source. The selected REAL source is Zenodo record `18136670`, version 1.0.0,
CC BY 4.0, published 2026-02-03. It preserves device-separated archives and explicit
normal/QQ/Weibo views.

- [x] **E36 CAL REAL:** download only devices 001, 002, 003, 005 and 009 (five previously unseen
      phone pipelines; 2,052,606,020 B declared archive bytes). Inventory
      safely, bind derivatives by parent and select at most 100 `view_000` originals per device.
      Model-free inventory found 138/139/168/100/71 normal originals; a pre-score amendment accepts
      >=70/device rather than silently dropping device 009. Five independent phone groups replace the generic six-device target
      because the source's remaining four named devices are reserved intact for FINAL; subgroup
      confidence and worst-device gates remain unchanged.
- [ ] **E36 FINAL REAL:** keep devices 004, 006, 007 and 008 fully locked until CAL freeze. These
      are the source authors' held-out Honor/Samsung/Motorola phone groups plus Sony NEX-7 camera;
      use up to 100 normal originals/group. Derived QQ/Weibo views are robustness children, never
      independent N, and may be scored only after the native-parent result is sealed.
- [x] **E36 CAL AI:** pin Apache-2.0 `Qwen/Qwen-Image-Bench` revision
      `d2493deb...7038`; select prompt indices 101–200 from exactly six families: GPT Image 2,
      Nano Banana 2, Seedream 5, Qwen Image 2 Pro, FLUX.2 Max and GLM-Image (600 clean parents).
- [ ] **E36 FINAL AI:** reserve prompt indices 1–40 from six family-disjoint cells: GPT Image 1.5,
      Nano Banana Pro, Imagen 4 Ultra, Hunyuan Image 3, FLUX.2 Pro and Seedream 4.5 (240 parents).
      The old unscored 40-row Qwen scout is superseded by a pre-score role amendment: overlapping
      rows remain sealed unless they belong to these new FINAL cells; none may enter CAL.
- [ ] **Balanced selection:** choose one threshold from CAL under equal per-device/per-family macro
      weights, not raw class counts. Require REAL macro FP <=10% and worst phone FP <=20% together
      with AI macro recall >=80% and worst family >=60%; also report pooled balanced accuracy, AUC,
      EER and TPR@FPR10. This explicitly prevents fixing real false alarms by simply calling every
      image REAL.
      Source/role metadata is now frozen; no CAL image or FINAL byte had been downloaded at this
      checkpoint. Compact evidence: `evidence/e36_acquisition.json` and the unscored old-scout role
      amendment `evidence/e36_qwen_role_amendment.json`.
      CAL transfer and CRC inventory later completed with FINAL still at zero bytes; the model-free
      71-row device-009 count amendment is `evidence/e36_real_count_amendment.json`.

### D3.7 — E37 source-held-out adaptation before FINAL

E36 falsifies threshold-only repair but creates a useful, now-consumed DEVELOPMENT adaptation
pool. E37 may fit a new head from these rows only if every E36 score used for model/threshold
selection is out-of-fold by source. FINAL devices/families remain inaccessible. The goal is
balanced transfer: reducing REAL accusations cannot be accepted unless modern-AI coverage passes
at the same frozen operating point.

- [x] **Amend the role before fitting.** Record E36 CAL as consumed `E37_ADAPTATION`; it can no
      longer provide an independent DDA calibration claim. Preserve all 1,071 rows and labels—no
      score-based removal, hard-example cherry-picking or family/device reweighting after results.
      Frozen before feature extraction/fitting in `evidence/e37_role_amendment.json`, including
      the five exact source-held-out folds and fixed head contract.
- [x] **Reuse the smallest adequate frozen representation.** Reuse the existing E32 DINOv2-S/14
      feature archive and preprocessing for the old TRAIN rows, extract the same 384-dimensional
      embedding only for the 1,071 E36 parents, and fit only a standardized class-weighted logistic
      head. Do not sweep encoders, crops or ensembles on E36. This tests whether source-balanced
      adaptation is sufficient without another dataset download or full-backbone fine-tune.
- [x] **Generate honest E36 out-of-fold predictions.** Use five fixed source-disjoint folds. Each
      fold holds out one complete REAL device and one or two complete AI families while always
      retaining the original E32 TRAIN base. Every E36 parent is scored exactly once by a head that
      saw neither its device nor its generator family. Fit `StandardScaler + LogisticRegression`
      with fixed `C=0.1`, `class_weight=balanced`, seed 42; no hyperparameter sweep.
- [x] **Select exactly one OOF threshold and gate both classes.** Choose the lowest OOF threshold
      with REAL device-macro FP <=10% and worst-device FP <=20%; require AI family-macro recall
      >=80%, worst-family >=60%, ROC-AUC >=0.90, TPR@FPR10 >=0.80, EER <=0.15, balanced accuracy
      >=0.85 and 100% coverage. Bootstrap by parent. A REAL-safe but AI-blind head fails; an
      AI-sensitive but camera-unsafe head also fails.
      **Result: rejected, but representation recovered.** ROC-AUC 0.94811, TPR@FPR10 0.82 and EER
      0.12976 pass. At the first source-safe threshold, REAL macro/worst FP are 4.14%/19.72%, but
      AI macro/worst recall are only 57.5%/42.0% and balanced accuracy is 0.7716. No E37 artifact
      was created and FINAL remains absent.
- [ ] **Only after the OOF gate passes, freeze the candidate.** Refit the identical fixed head on
      old E32 TRAIN plus all E36 adaptation rows, store feature/input/role/code hashes and retain
      the OOF-selected threshold unchanged. Then acquire the already-preregistered FINAL bytes,
      audit/decontaminate them without model access and score exactly once. Failure leaves FINAL
      sealed and opens paired DDA-style training—not another post-hoc threshold or ensemble.

### D3.8 — E38 fixed adaptation emphasis, then the one untouched FINAL

E37 proved DINOv2-S ranks the new domain well but the 21,349-row historical base overwhelms only
1,071 current adaptation rows. A consumed-DEVELOPMENT diagnostic varied regularization and a
single uniform adaptation weight; it did not write an artifact or access FINAL. It found that
uniformly emphasizing every E36 row—not selecting examples or sources—can move the joint frontier
past all gates. Because those outcomes were inspected, E38 is a DEVELOPMENT-selected candidate,
not fresh validation. Its only honest confirmation is the already locked FINAL.

- [x] **Freeze E38 before fitting.** Keep the same backbone, preprocessing, five source folds and
      complete row set. Fix `C=0.0003`, `class_weight=balanced`, seed 42 and a uniform sample weight
      of 100 for every E36 adaptation row versus 1 for every historical E32 TRAIN row. Do not use
      DDA scores, an ensemble, per-source weights or another grid.
      Frozen in `evidence/e38_fixed_contract.json` before the formal OOF reproduction/artifact fit.
- [x] **Reproduce the fixed DEVELOPMENT frontier and freeze one artifact.** Generate one OOF score
      per E36 row, select the same REAL-budget threshold and require the unchanged eight quality
      gates plus full coverage. Record explicitly that the hyperparameters were selected on this
      consumed population. If it fails, no artifact/FINAL access; if it passes, refit the identical
      head on all E32 TRAIN + E36 rows and bind its artifact/hash/threshold.
      **Passed:** AUC 0.98062, TPR@FPR10 0.975, EER 0.06162, balanced accuracy 0.8955;
      REAL macro/worst FP 4.34%/19.72%; AI macro/worst recall 82.5%/77.0%, coverage 100%.
      Candidate SHA-256 `fddbe475...4067`, threshold `0.896190`. This is DEVELOPMENT-selected and
      authorizes one untouched FINAL only; it is not itself final evidence.
- [x] **Acquire/audit the preregistered FINAL without model access.** Download only REAL devices
      004/006/007/008 and the six family-disjoint AI cells already frozen in E36-A. Verify exact
      archive/blob checksums, safe extraction, decode, label/source counts and exact/dHash overlap.
      No source, prompt, row, transform or threshold may change after any FINAL score.
      Verified 2,038,841,380 REAL archive bytes +311,236,195 AI bytes. The frozen unscored manifest
      contains 400 REAL (4x100) +240 AI (6x40), zero prior exact/dHash overlap, SHA-256
      `cad71ff5...66e6`; candidate and threshold remain unchanged.
- [x] **Score FINAL exactly once.** Require the unchanged internship gates on native/clean parents
      first. Only after sealing that result may parent-linked QQ/Weibo or deterministic degradation
      children be reported as robustness columns. A miss is the final result for this candidate;
      it cannot trigger a retry on the same FINAL.
      **Result: failed the strict joint gate.** AUC 0.98185, TPR@FPR10 0.95, EER 0.075 and all
      400/400 REAL correct at the frozen threshold, but AI macro/worst recall are 67.5%/50.0% and
      balanced accuracy is 0.8375. Coverage is 640/640. The result is final for E38; no threshold,
      model or subgroup was retried.

### D3.9 — E39 calibration-transfer correction requires a new FINAL

E38 is a strong ranker and conservative working prototype, but its OOF threshold did not retain
the same score scale after the final head was refit on all adaptation rows. A post-hoc FINAL curve
finds a jointly feasible region near `0.270069` (REAL macro/worst FP 10%/17%; AI macro/worst recall
95%/90%), proving the failure is operating-point transfer rather than missing separation. That
value is permanently contaminated and cannot be served or used to relabel E38 as passed.

#### E39-A — correct the decision layer without retraining

- [x] Reclassify all 640 E38 FINAL parents as consumed `E39_CALIBRATION`; they can select E39 but
      can never again provide final evidence.
- [x] Keep the DINOv2-S representation and fitted head byte-identical (`fddbe475...4067`). Freeze
      exactly one E39 threshold from the consumed calibration scores under the same REAL and AI
      subgroup budgets. Do not change examples, weights, architecture, crop or score direction.
- [x] Package the threshold as a new research candidate with explicit E38/E39 provenance. The
      currently observed `0.270069` value is development-selected; it is eligible only for a new
      independent test and must not alter the recorded E38 result.

#### E39-B — freeze a genuinely new compact FINAL before bytes

- [x] Research licensed sources and write the source/role decision to `DATASETS.md` before any
      image transfer. REAL must contain at least four native camera devices/sessions absent from
      every earlier role. AI must contain at least six unused modern generator/model-version cells.
- [x] Target 40 native REAL parents/device and 40 clean AI parents/family: frozen allocation is four
      REAL devices plus seven AI families, 160 REAL +280 AI =440 parents. Prefer diversity and
      provenance over another 20,000-image download. Cap every
      source equally so no large group dominates the result.
- [x] Freeze URLs/API versions, licences, exact counts/checksums, prompts/seeds where available,
      allocation and failure policy. Reject social copies of consumed parents, extra prompts from
      consumed families, owner-gallery rows and old TRAIN sources as substitutes for independence.

#### E39-C — acquire and audit without model access

- [x] Download resumably to a new role-separated directory, verify every published checksum and
      keep a 100 GiB disk floor. Do not open the E39 model while acquisition/audit runs.
- [x] Decode every parent; verify explicit labels, source counts, native dimensions and licence
      receipts. Run exact and perceptual overlap checks against all TRAIN/CAL/DEVELOPMENT/FINAL
      manifests. Derived resize/JPEG/social copies remain grouped children and never increase N.
- [x] Commit the unscored manifest and compact evidence before the first prediction. Once frozen,
      no source, row, threshold or model setting may change.

#### E39-D — one-shot decision and product promotion

- [x] Score the frozen 440-parent FINAL exactly once. Require 100% coverage, AUC >=0.90,
      TPR@FPR10 >=0.80, EER <=0.15, balanced accuracy >=0.85, REAL macro/worst FP <=10%/20% and
      AI macro/worst recall >=80%/60%. Report every source and confidence interval.
      **Result: failed.** Coverage 440/440, AUC 0.90033, TPR@FPR10 0.7714, EER 0.1933,
      balanced accuracy 0.7004, REAL macro/worst FP 53.13%/60.0%, AI macro/worst recall
      93.21%/90.0%. The new AI side is strong; new native REAL transfer is unsafe.
- [x] Apply the promotion rule. E39 failed multiple gates, so it is explicitly not promoted to the
      API/web verdict; the currently served model remains unchanged.
- [x] If E39 misses only the thresholded gates while AUC remains strong, do not tune on the new
      FINAL; consume it as the next calibration source and obtain another independent final. If AUC
      itself falls below 0.90, stop threshold work and open paired/content-aligned backbone training.
      E39's AUC is only marginally above 0.90 and TPR@FPR10/EER also fail. A post-hoc REAL-safe
      threshold still misses AI macro, balanced accuracy, TPR and EER; threshold-only work is closed.
- [ ] After each completed phase, append facts to `HISTORY.md`, measurements to
      `ml/EXPERIMENTS.md`, data roles to `DATASETS.md`, update this checklist, run the full test
      suite, commit, push and require green CI.

### D3.10 — E40 content-balanced source-held-out adaptation

E39 proves the candidate recognizes seven unseen current generators, but its REAL score distribution
shifts sharply on coordinated outdoor phone photographs. Because no threshold passes the consumed
E39 population, E40 must improve source/content generalization without hiding the failed result.

#### E40-A — consume E39 correctly and build model-free features

- [x] Reclassify all 440 E39 FINAL parents as consumed `E40_ADAPTATION_DEVELOPMENT`; they can train
      and select E40 but can never be final evidence again. Bind E39 manifest/result/score hashes.
- [x] Cache one unchanged DINOv2-S embedding per E39 parent without filtering rows. Create seven
      source-held-out folds so every AI family and every REAL device receives predictions from a
      head that did not see that source. Content clusters may weight training rows but must never
      select rows or define folds, because each FloreView device shares the same scene catalog.
- [x] Cluster frozen embeddings only to balance content, not to label/select examples. Use inverse
      class x source x content-cluster weighting so repeated FloreView scenes and generator prompt
      styles cannot dominate the decision boundary.

#### E40-B — a small preregistered head ladder, not another sweep

- [x] Compare exactly three fixed linear heads on the same source-held-out predictions: uniform
      modern replay, source-balanced replay and source+content-balanced replay. Reuse the unchanged
      DINO backbone and include a fixed 5% stratified replay buffer from historical E32 TRAIN to
      reduce forgetting; do not add DDA/CF-ViT score features or an ad-hoc ensemble.
      **Frozen implementation:** 1,067 deterministic E32 replay rows plus all E36/E39 development;
      seven source folds; C=0.01; 16 training-fold-only KMeans cells; primary seed 42; fixed
      simplest-first order uniform -> source -> source+content. Exact contract is
      `evidence/e40_fixed_contract.json` and must be committed before feature/scoring commands.
- [x] Select only by the complete frozen gate: coverage 100%, AUC >=0.90, TPR@FPR10 >=0.80,
      EER <=0.15, balanced accuracy >=0.85, REAL macro/worst FP <=10%/20%, AI macro/worst recall
      >=80%/60%. If none passes, stop E40 before refit; do not soften thresholds.
- [x] For a passing head, repeat three fixed seeds and require every seed to preserve the REAL and
      AI subgroup gates. Then refit one artifact on all consumed adaptation rows and freeze one OOF
      threshold; no E40 FINAL byte may exist yet.

#### E40-C — robustness checks with already consumed/local data

- [x] Run grouped JPEG/resize derivatives as parent-linked stress tests and use the owner gallery
      only as a disclosed DEVELOPMENT smoke. Require no collapse toward either class; derivatives
      never inflate N and cannot promote the model.
      **Frozen implementation:** native, JPEG-q50 and 75%-resize+q50 views share the same 440 E39
      parents and unchanged threshold; derivative AUC/TPR/balanced/REAL/AI floors plus >=80%
      per-class decision agreement are fixed in `evidence/e40_robustness_contract.json`. The
      hash-bound 210-photo owner gallery must remain <=20% FP; no row may tune E40.
- [ ] Package the research candidate only if E40-A/B/C all pass. Record artifact, feature cache,
      replay selection, threshold and seed hashes in HISTORY/EXPERIMENTS/DATASETS.
      **Measured stop:** transports pass strongly, but owner-gallery FP is 69.52% at the frozen
      threshold, so E40-C fails and no `e40_candidate.joblib` is created. No retry is allowed.

#### E40-D — stop at the next-data boundary

- [ ] Before downloading, preregister another independent FINAL with at least four new native
      devices/sessions and six new generator/model-version cells from sources outside E39. Keep
      40 parents/group, source balance, prompt/scene provenance and exact/perceptual decontamination.
- [ ] Score that new FINAL exactly once and promote only if every original joint gate passes. E39
      rows, unused members from its same archive and extra FloreView rows are not substitutes for
      this final independence.

### D3.11 — E41 broad-real threshold transfer, then new FINAL

E40 repaired representation/head ranking and transport robustness, but its OOF threshold did not
transfer to the owner's casual-gallery score scale. A sealed post-hoc diagnostic on consumed native
scores finds a complete-gate frontier at 0.619554. This is not E40 evidence; E41 may package it only
as a contaminated calibration candidate for a genuinely new FINAL.

- [x] Reclassify the 440 E39 native rows plus 210 owner-gallery rows as consumed
      `E41_BROAD_REAL_CALIBRATION`; bind E40 draft, robustness report and score hashes. Derivatives
      remain parent-linked stress evidence and never enter threshold selection.
- [x] Package the byte-identical uniform E40 head with the single diagnostic threshold
      `0.6195540428161622`. Forbid retraining, another threshold, row filtering or web/API promotion.
- [ ] Before transfer, freeze a disjoint E41 FINAL source contract: at least four new native
      devices/sessions and six new generator/model-version cells, 40 parents/group, with licences,
      scene/prompt provenance and exact/perceptual decontamination. E39 sources and the owner gallery
      are forbidden.
- [x] Stop before downloading; E41 FINAL remains zero bytes/zero rows until exact sources and
      licences are frozen and new transfer is authorized/available.
- [ ] Then acquire without model access, freeze the unscored manifest, score once, and promote only
      if every original joint gate passes.

### D4 — close the slice reproducibly

- [x] Append acquisition facts to `DATASETS.md`, measured results to `ml/EXPERIMENTS.md`, and every
      move/decision/result to append-only `HISTORY.md`. Update this checklist after each gate,
      verify focused tests plus the full Python/web suite, commit in reviewable checkpoints, push
      through protected `main`, and require green CI.
      **Latest closeout:** 251 Python tests, compileall, `pip check`, six-artifact registry, web
      production build + six tests, TypeScript and ESLint all pass. One upstream Starlette/httpx
      deprecation warning remains; it does not affect inference or the result.

## Current execution slice — repository rewiring and R1c pre-acquisition (2026-08-27)

This slice makes the project easier to understand without changing a model, threshold, API
decision or measured claim. It also turns the already-selected R1c direction into an explicit
stop/go path up to—but not including—the next image transfer.

**Hard boundary for this slice:** download no dataset, model, API image or third-party binary.
Do not delete `HISTORY.md`, `ml/EXPERIMENTS.md`, evidence, experiment scripts, local datasets or
model artifacts. Generated caches may be ignored/removed, but scientific bytes and append-only
records are not “cleanup.” The Sites/Vinext chain (`.openai/`, `vite.config.ts`, `build/`,
`worker/`, PostCSS and the lockfile) remains because it is the verified web build path.

### S0 — map the live circuit before moving wires

- [x] Re-audit every Markdown surface, the tracked tree, package entry points, Python imports,
      browser/API boundaries and ignored disk usage. The 5.1 GB `ml/` directory is dominated by
      ignored local artifacts/data, not tracked source bloat; it must not be erased as a code tidy.
- [x] Freeze the three ownership zones: active product (`app/`, `pixelproof.serve`, project model,
      verdict and demo launcher), reproducible research (`ml/experiments`, E31/E32 research CLIs
      and compact evidence), and frozen history (`archive/`, HISTORY/EXPERIMENTS/report material).
      Cleanup may cross none of these boundaries silently.

### S1 — remove only proven residue

- [x] Delete the unused Claude-specific launcher that automatically opts into B-Free's restricted
      licence, the empty no-op Next configuration and the three unreferenced starter SVG assets.
      Keep the project favicon/social card and every file required by the Sites/Vinext build.
- [x] Add explicit ignore coverage for pytest caches so local verification noise cannot re-enter
      the project view. Remove only empty/generated cache directories after validation; preserve
      environments, installed packages, model artifacts and datasets.

### S2 — separate web orchestration from result presentation

- [x] Move the four result-only React components out of `app/page.tsx` into one focused module.
      Keep upload/request lifecycle in the page and response validation in
      `app/analysis-contract.ts`; do not change endpoint, payload, labels, thresholds or copy.
- [x] Remove CSS selectors belonging to the retired method picker, tile overlay, legacy result,
      old R1b card and probability meter. Prove every removed selector has no live markup owner;
      preserve responsive, keyboard, touch and reduced-motion behavior.

### S3 — make the remaining structure self-explanatory

- [x] Replace the flat repository map in `README.md` with active product, research/archive and
      generated-local boundaries. Update the runnable experiment index through E32 so a reader can
      tell which code serves users, which reproduces rejected candidates and which must stay frozen.
- [x] Run the complete 207-test Python suite without caller `PYTHONPATH`, compileall, dependency
      and artifact checks, plus web lint, typecheck, production build and all browser-contract tests.
      Record exact results in `HISTORY.md`, commit and push, then require green GitHub CI.

### S4 — R1c work allowed before the next data transfer

- [ ] Consolidate the existing C4-R1c requirements into one metadata-only source receipt for three
      mutually disjoint roles: `R1C_CAL`, `R1C_LOCKED_REAL` and `R1C_LOCKED_AI`. Each proposed
      source must declare revision, licence, label direction, parent/group identity, pipeline or
      generator version, expected count/bytes/checksum where published and protected-role overlap
      policy before any image is selected.
- [ ] Prefer unused, licensed local holdings and a compact new multi-device capture. Admit no source
      merely because its folder says REAL/AI; require at least five unused authentic pipelines in
      CAL, five other authentic pipelines in LOCKED_REAL, and five current AI families with at
      least 100 native parents each in LOCKED_AI. IPN, the owner gallery, E30 and named older tests
      remain consumed DEVELOPMENT and cannot fill these roles.
- [ ] Implement only the metadata/schema validator, deterministic parent-level allocator,
      protected-hash interface, free-space estimate and resumable acquisition-receipt generator.
      Unit tests use synthetic metadata/temporary files; no network image byte is permitted.
- [ ] **Stop boundary:** present the frozen source allocation, estimated transfer size, licence
      decisions, exact destination and acceptance tests to the user. Actual download begins only
      in a later authorized slice with suitable internet. After bytes arrive, the existing order
      remains audit -> R1c-T threshold selection on CAL only -> consumed DEVELOPMENT gate -> one
      locked final; paired training R1c-P starts only if threshold transfer fails.

## Active goal — E32/R1c conservative generalization recovery (2026-08-27)

The product goal remains a genuinely testable binary detector, not a high score on a familiar
dataset. E32/R1b changed the diagnosis materially: its frozen CF-ViT representation recalls modern
AI extremely well internally, but the 0.125935 CALIBRATION threshold transfers badly to independent
authentic pipelines. It produces 25.94% IPN macro / 40.0% worst-device false positives and 68.57%
owner-gallery false positives. This is unacceptable, but it is not the inverted ranking seen in
E31: a read-only post-hoc frontier on the already-consumed DEVELOPMENT scores found that threshold
0.863312 would reduce owner FP to 20.0% and IPN worst-device FP to 15.0% while retaining 91.00%
six-source / 90.01% current-family internal AI macro recall (80.0% weakest family). At 0.95 the
same diagnostic is 9.52% owner FP, 7.5% IPN worst-device FP and 85.13% six-source / 83.28%
current-family internal AI macro recall (65.0% weakest family).

Those thresholds are **evidence of feasibility, not candidates**: IPN and the owner gallery were
already consumed and may never select a deployable threshold. The next development is therefore a
threshold-first R1c recovery on genuinely new CALIBRATION sources. Only if that clean replication
fails will the project spend on paired-content training or a new representation. More arbitrary
volume, another encoder-only swap and an ensemble are explicitly lower priority.

The user's gallery is excluded from TRAIN and CALIBRATION. Existing gallery scores are historical
DEVELOPMENT evidence; newly contributed, never-scored gallery content may enter a separately
sealed owner-real final arm. The attached `/Volumes/LaCie` disk has about 651 GiB free and is the
only target for third-party image bytes, caches and derived E32 image archives. Git receives only
small manifests, aggregate evidence, code and documentation.

### Phase C0 — freeze scope, roles and stop/go order before acquisition

- [x] Keep the project label invariant explicit at every boundary: `0 = REAL`, `1 = AI`. Every
      source declares its raw label names and `raw -> project` mapping; ambiguous numeric labels,
      changed upstream class names or an undeclared source are hard failures. Never auto-flip a
      model merely because an external AUC is below 0.5.
- [x] Preserve all earlier protected roles. E30 MLLM DEVELOPMENT, the scored owner gallery,
      Julien/Defactify named test sets and Qwen LOCKED FINAL cannot fit rows, weights, crop rules,
      augmentation, thresholds, model selection or ensemble coefficients. ITW-SM and newly
      generated API/gallery final rows remain unopened until their exact gate permits scoring.
- [x] Treat parent content as the indivisible unit. Crops, JPEG/WebP versions, resizes and social
      derivatives inherit their parent's label, role and group; no derivative may cross a split.
- [x] Fix the order: source/licence audit -> resumable acquisition -> byte/label/shortcut audit ->
      frozen TRAIN/CALIBRATION manifest -> low-cost representation screen -> controlled training ->
      DEVELOPMENT gate -> one locked Champions League final. No model score may select download
      rows or repair the test after results are known.
- **Acceptance:** this E32 section is committed before a new dataset byte, E32 manifest, embedding,
  checkpoint, API-generated image or candidate score exists.
- **C0 recorded:** the SSD was inspected read-only at 651 GiB free; the existing AI holdings and
  public source metadata were inventoried without downloading an E32 image. SSAFE/PE-Core,
  DINOv2/RINE and Hive/EfficientNet-B4 are frozen as comparable representation hypotheses rather
  than assumed winners. C1 may start only after this plan/history checkpoint is committed.

### Phase C1 — acquire a compact, diverse authentic-photo pool on the SSD

- [x] Target **10,000–20,000 eligible REAL parents**, nominally about 15,000, across native camera,
      modern computational-photography and web-photo pipelines. Cap devices/scenes so a repeated
      burst, camera or source cannot dominate. Prefer the following audited candidates, not a blind
      union:
      - VISION native parents: 35 portable devices / 11 brands; social variants remain derived
        transport evidence rather than additional independent photographs.
      - Forchheim FODB original parents: 3,851 photos / 143 scenes / 27 devices / 25 models / nine
        brands. Group all cameras' versions of one scene together; its Facebook, Instagram,
        Telegram, Twitter and WhatsApp copies remain derivatives.
      - CSAFE Multi-camera Smartphone Image Database: roughly 50,000 JPEGs / 60 modern phones,
        CC BY 4.0. It is 123.62 GB in six 17–29 GB model archives, so inspect archive inventories
        and fetch only the minimum device/model subset needed for modern Apple/Samsung coverage.
      - A 2,000–3,000 row web/professional-photo complement from a source with per-image provenance
        and compatible research terms; do not scrape a site whose terms do not permit it.
- [x] Keep SOCRatES (9,700 images / 103 phones / 15 makes) conditional because it requires a
      signed licence agreement, and keep ForensiCam-215K conditional because its only public
      download is Baidu and its repository exposes no clear dataset licence. Neither may silently
      become a dependency.
- [x] Freeze a resumable acquisition receipt before transfer: pinned URL/revision, declared
      expected bytes/hash where published, `.partial` state, retry/resume, 100 GiB free-space
      floor and an explicit target below `/Volumes/LaCie/pixelproof-datasets/e32/`. The C1a receipt
      selects 3,500 VISION native parents, all three FODB archives and only CSAFE `s21.zip`; its
      detailed SHA-256 is `200a7aeb...ca4d`. Freeze downloaded zero image bytes.
- [x] Complete the frozen transfers and final content hashes without overwriting existing E31
      holdings or modifying an upstream archive in place.
- [x] Replace only the stalled CSAFE single stream with a tested four-range resume path. Preserve
      the existing contiguous prefix, download disjoint exact byte ranges to separate partials,
      verify every `Content-Range`/length, assemble to a new temporary file, verify the published
      full MD5, and only then atomically promote. Never overwrite the source prefix on failure.
      Nineteen focused acquisition/archive tests pass.
- [x] Run the committed four-range recovery on the preserved CSAFE prefix, require published MD5,
      then freeze the ZIP inventory before selecting or extracting internal rows.
- [x] Complete CSAFE four-range recovery: preserve 4,723,834,880 prefix bytes, fetch four exact
      ranges, assemble 17,588,803,163 bytes and reproduce published MD5 `5c5f...91d8` before
      promotion. Temporary range files were removed only after verification.
- [x] Pass CSAFE ZIP inventory: 7,996 JPEG under ten physical S21 devices; 4,000 `blank` flat-field
      images and 3,996 `natural` images across front/telephoto/ultra/wide pipelines. Preserve all
      rows as unselected and precommit natural-only selection before extraction.
- [x] Freeze all 3,996 CSAFE `natural` members from inventory metadata before reading member bytes;
      bind device and lens pipeline, exclude all 4,000 `blank` rows by contract, then extract only
      the frozen natural set atomically with per-file SHA. Implement the realization gate before
      production extraction and keep all outputs role-free.
- [x] Implement/test CSAFE natural selection, atomic extraction and receipt-bound realization before
      production use. Unknown device/content/lens/suffix paths fail closed; 23 focused tests pass.
- [x] Freeze the exact CSAFE natural selection before member bytes: all 3,996 natural JPEGs, ten
      devices (398–400 each) and four lenses (998–1,000 each); exclude all 4,000 blank fields.
- [x] Extract the frozen 3,996 CSAFE natural parents atomically: 13,219,178,988 B with per-file SHA,
      device/lens metadata and exact selection binding. No blank member was extracted; keep rows
      role-free until full realization.
- [x] Pass CSAFE full realization: 3,996/3,996 RGB JPEG with EXIF, unique SHA/pHash, zero confirmed
      duplicate and zero protected/passed-peer overlap. One equal-dHash pair remains a visible
      nonduplicate candidate; all rows stay role-free pending global overlay.
- [x] Bind CSAFE's exact natural-extraction receipt and schema-v2 audit into the global overlay,
      recompute across 15,000 AI + 11,347 REAL selected rows, and freeze >=10,000 eligible REAL
      only if no unresolved cross-label component survives. Preserve the existing AI subset absent
      a newly discovered collision. Result: no new component or cross-label ambiguity; AI remains
      14,786 and REAL reaches 11,344 (VISION 3,497 + FODB 3,851 + CSAFE 3,996).
- [x] Implement/test CSAFE overlay binding before production rerun; require exact extraction state,
      row equality and audit SHA. Fourteen focused overlay/realization tests pass.
- [x] Before extracting FODB/CSAFE, commit a ZIP safety and inventory gate: reject absolute or
      traversal paths, symlinks, encryption, duplicate member names, undeclared archive sizes and
      implausible expansion; summarize member hierarchy/suffixes/bytes. For FODB, verify exactly
      3,851 `orig` JPEG parents across 27 device roots and parent-link each social transport by
      scene index; extract only `orig` candidates atomically. CSAFE rows remain unopened and
      unselected until its verified `s21.zip` inventory is frozen. Fifteen focused tests pass.
- [x] After the frozen transfers finish, run the committed inventory gates, preserve their receipts,
      extract only FODB `orig` members, and audit every extracted parent before role assignment.
- [x] Extract all 3,851 FODB `orig` parents atomically from the passed inventory: 15,416,129,383 B,
      each with extraction SHA and device/scene binding. No social or `inspection` member was
      extracted; all rows remain role-free pending realization.
- [x] Pass FODB full realization: 3,851/3,851 RGB JPEG with EXIF, 3,851 unique SHA, zero confirmed
      perceptual duplicate and zero protected/passed-peer overlap. Preserve seven equal-dHash
      candidate pairs as nonduplicates under pHash; retain all rows as role-free candidates.
- [x] Extend the committed global eligibility overlay with the extraction-bound 3,851 FODB rows,
      then recompute exact/perceptual components across 15,000 AI + 7,351 REAL selected rows.
      Preserve scene parent metadata, exclude both sides of any REAL/AI ambiguity, and do not alter
      the already-frozen AI source-cap selection unless a new global duplicate requires exclusion.
      Result: no new component; AI remains 14,786 and REAL becomes 7,348 (3,497 VISION + 3,851 FODB).
- [x] Implement and test the FODB overlay input binding before production rerun. Require the exact
      extraction-receipt SHA/state and exact equality with the FODB schema-v2 audit; 13 focused
      overlay/realization tests pass.
- [x] Implement the FODB role-free realization command before extraction. It binds the extraction
      receipt, rechecks byte count and SHA, decodes every original, records camera/device/scene and
      native state, and applies the shared protected/duplicate gate. Sixteen focused tests pass.
- [x] Preserve the first production FODB inventory stop: all device members matched, but part03
      also contains 4,004 JPEGs / 2,834,597,196 bytes under `inspection/` (3,861 device-check and
      143 scene-comparison helpers). They are derived inspection material, not new parents; no
      inventory receipt or extraction was accepted.
- [x] Precommit an explicit `inspection/` exclusion while continuing to fail every other unknown
      root/member, record excluded counts/bytes in evidence, then rerun all CRC/SHA checks.
      Seventeen focused archive/realization tests pass before the production rerun.
- [x] Pass the corrected production FODB inventory: all three archives pass CRC/SHA and declared
      size; 3,851 parents / 27 pipelines / 143 scene groups each have `orig` plus five transports.
      Preserve 4,004 `inspection` derivatives / 2,834,597,196 B as explicit nonparents.
- [x] Implement the role-free realization gate before any transfer completes. It binds every audit
      to the frozen selection SHA, ignores exFAT AppleDouble sidecars, requires all selected bytes
      to decode, derives format from payload bytes, records geometry/EXIF/compression summaries,
      and rejects exact/dHash repeats against protected E30 roles and already-passed E32 sources.
      A pass means only `candidate`; the gate cannot assign TRAIN/CALIBRATION itself.
- [x] Decode and inventory every selected parent; record camera/device/model, scene/event group,
      native/social state, format, dimensions, orientation, EXIF availability, bytes/pixel and
      licence/provenance. Remove exact and perceptual duplicates against every protected role.
- **Acceptance:** at least 10,000 eligible REAL parents, at least three independent collections,
  broad device/brand support, groupable provenance, zero protected overlap and no one source,
  device or repeated scene capable of defining the REAL class. `DATASETS.md` records exact realized
  counts, bytes, revisions, terms, selection reason and limitations.

### Phase C2 — build a modern, source-capped AI pool without redownloading blindly

- [x] Complete a physical metadata/licence/provenance inventory before selecting rows. C2a finds
      only three currently admissible modern families: GPT Image 1 (1,060 local images), Nano
      Banana (9,457 rows) and Nano Banana Pro (200 licensed loose images). FLUX.1-dev (10,000), the
      1,250-row second NBP source and the 127,835-member Nano editing archive remain conditional;
      missing dataset licences or contradictory counts are not inferred away. At least two
      additional licensed, explicitly generated families are required.
- [x] Target **10,000–20,000 eligible AI parents**, nominally about 15,000, with at least five
      verified modern generator families and no family above 20% of the selected pool. Audit the
      current SSD holdings first: FLUX.1-dev (10,000), Nano Banana (9,457), Nano Banana Pro
      (registered 1,250 plus a separate bounded holding), GPT Image 1 (1,060 PNG plus 1,061 text
      sidecars—not 2,122 images) and the Julien modern mixture. A folder name is not generator
      provenance.
- [x] Pre-register the nominal **15,000-parent source allocation** before remaining byte selection:
      Qwen Image 2512 3,000; FLUX.2 Klein 9B 3,000; Nano Banana 3,000; GPT Image 1 3,000; licensed
      Nano Banana Pro 200; CommunityForensics AI diversity anchor 2,800. The first four sources
      each equal—not exceed—the 20% ceiling; CommunityForensics is 18.67% and does not count as a
      sixth current family. The five-family gate is Qwen, FLUX.2, Nano Banana, GPT Image 1 and Nano
      Banana Pro. If pinned GPT cannot supply every pair in the later exact selection, stop and
      document a replacement source; never inflate another source or reuse a protected final to
      hide the gap.
- [x] Implement and run the metadata-only exact selector after the GPT gate passed.
      Nano uses stable-hash id selection; CommunityForensics uses model-identity round-robin; NBP
      uses all 200 licensed images; Qwen/FLUX inherit their frozen prompt groups; GPT selects from
      all pinned upstream pairs independently of local availability. GPT revision, CC-BY-4.0 tag
      and exact 4,000-pair listing reproduced; the 4,752,567-byte detailed 15K receipt is frozen at
      SHA-256 `3230f026...80b7`, with content-selection SHA `2a31e792...0ef7`. It selected 795
      already-local GPT pairs and 2,205 download-required pairs; freeze downloaded zero image bytes.
- [x] Precommit a GPT-only acquisition gate tied to the 15K record-selection SHA. It reuses exact
      local pairs, writes missing pairs only below external `e32/ai/gpt-image-1`, preserves
      `.partial` resume and requires one selected missing image/prompt pair to pass decode and
      UTF-8 prompt smoke before the 2,205-pair bulk transfer.
- [x] Pass the exact GPT smoke: selected `GPTIMG_852.png` is a 3,486,339-byte RGB PNG at
      1024x1536 with non-empty 1,341-byte UTF-8 prompt. Evidence is bound to selection SHA
      `2a31e792...0ef7`; bulk may start without changing the selected rows.
- [x] Research and freeze only the measured two-family gap, without downloading OpenFake's full
      3.44 TB or reassigning protected tests. C2b pins Qwen Image 2512 (CC BY-SA 4.0) and FLUX.2
      Klein 9B Base (CC BY 4.0), selecting 750 complete prompt groups / 3,000 JPEG XL outputs from
      each by category round-robin. Selected native image bytes are 7,108,445,821 and
      4,400,537,141 respectively; FLUX editing references are excluded. Detailed selection SHA is
      `e9c3d3da...af7a` after adding expected dimensions to the unchanged row selection.
- [x] Pass one-image decoder smokes for both sources and bind the bulk gate to the current selection
      SHA. Both `.jxl` paths contain PNG payloads: Qwen decodes RGB 1328x1328 and FLUX RGB
      1024x1024 directly through Pillow. Record the upstream extension/byte-format mismatch; do not
      add a JPEG XL dependency or let extension become a label feature.
- [x] Acquire the frozen 6,000 gap images after the passed decoder gate. OpenFake
      `core/test`/Reddit splits and frontier held-out models remain test candidates, never TRAIN.
- [x] Pin the AI realization contract before bulk completion: every four-output prompt group needs
      four decodable images, four non-empty matching UTF-8 prompt sidecars, declared byte counts
      and expected dimensions. Missing/partial/mislabeled rows produce a rejected audit and never
      a silently smaller training pool.
- [x] Extend the same role-free realization gate to the exact 15K local pool before opening its
      production rows. Nano and Community read only the frozen Parquet row locators and actual
      embedded image bytes; NBP resolves exact-size loose files; GPT resolves each frozen pair from
      the original checkout or E32 acquisition root. All paths share full decode, SHA-256, dHash,
      protected/peer overlap and duplicate rejection; passing still assigns no model role.
- [x] Correct the perceptual-duplicate gate after Nano exposed a real dHash collision. Exact
      SHA-256 remains definitive; exact dHash now creates candidates and a separately computed
      64-bit DCT pHash must also be within Hamming distance <=5 to confirm a within-source/modern
      E32 peer duplicate. The five collided Nano images are visibly unrelated and 24–32 pHash bits
      apart. Legacy protected E30 dHash hits remain conservative hard exclusions because their
      original audit contract did not persist pHash. Commit this schema-v2 rule before rerun.
- [ ] Verify generator version, generation date, prompt/content group, native output status,
      licence/usage boundary and label direction for every admitted collection. Unknown generator
      identity may contribute only to a capped `unknown` group and cannot satisfy the five-family
      requirement.
- [x] Realize the complete licensed Nano Banana Pro arm: 200/200 PNG decode, 200 unique SHA-256,
      200 unique dHash, zero within-source duplicate and zero exact/dHash overlap with all four E30
      protected manifests. It passes only as a role-free candidate; 136 RGB and 64 RGBA modes must
      receive the same later input normalization as every other source.
- [x] Realize the frozen Nano Banana arm under schema v2: 3,000/3,000 RGB PNG decode, 3,000 unique
      SHA-256 and pHash, zero exact/confirmed-perceptual duplicate and zero protected/passed-peer
      overlap. Preserve the five-row dHash candidate collision in evidence; it is not a confirmed
      duplicate because pairwise pHash distances are 24–32.
- [x] Preserve the first full Qwen realization as rejected: 3,000/3,000 decode and zero
      protected/peer overlap, but eight exact duplicate pairs link two composition groups to two
      architecture groups, and one style group contains a confirmed near-duplicate pair. Do not
      rewrite the receipt or cherry-pick individual variants.
- [x] Precommit a pool eligibility overlay that removes the three affected Qwen prompt groups as
      indivisible units and applies deterministic source-cap trimming to the other 3,000-row arms.
      The overlay must remain bound to the immutable 15K selection and every realization receipt;
      it may exclude audited failures but cannot add an unselected replacement after byte access.
- [x] Preserve the first full FLUX.2 realization as rejected: 3,000/3,000 decode and zero
      protected/peer overlap, but 28 exact and 41 confirmed perceptual duplicate groups leave 2,964
      unique SHA / 2,932 unique pHash. Duplicate members touch 32 prompt groups, especially
      `diffusiondb_orig` and editing. Defer exact canonical-group exclusions until every 15K arm is
      audited; do not rewrite FLUX selection or fill from unseen rows.
- [x] Preserve the first GPT and VISION full-audit results before repair. GPT realizes 2,893/3,000
      images: 107 prompt sidecars fail the UTF-8-only gate, five perceptual duplicate pairs remain,
      and protected/peer overlap is zero. VISION realizes all 3,500 balanced camera parents with
      3,500 unique SHA values and zero protected/peer overlap, but three within-source perceptual
      pairs reject the intact source.
- [x] Add and test a byte-preserving GPT prompt decoder that accepts UTF-8 first and Windows-1252
      only as an explicit fallback. Preserve both original-byte and normalized-text hashes and
      report encoding counts; 20 focused selection/acquisition/realization tests pass.
- [x] Rerun the same immutable GPT selection after the decoder commit: all 3,000 RGB PNGs and
      prompts realize; 2,893 prompts are UTF-8 and 107 are Windows-1252. Preserve the intact-source
      rejection because six perceptual pairs remain, and send only stable loser exclusions to the
      later eligibility overlay rather than replacing rows.
- [x] Realize the 2,800-row CommunityForensics diversity anchor under schema v2: all 300 model
      identities remain represented, every SHA/dHash/pHash is unique, and protected/passed-peer
      overlap is zero. Retain it only as a role-free candidate.
- [x] Reissue Nano Banana Pro's 200-row receipt under schema v2 so every AI arm uses the same
      SHA+dHash+pHash rule. All 200 hashes remain unique and overlap-free; state stays
      `candidate_only`.
- [x] Extend the immutable-selection eligibility overlay to VISION parent rows as well as AI prompt
      groups. Resolve each duplicate component by a stable content-independent canonical key,
      exclude losers only, bind the overlay to every detailed receipt SHA, and assign no role yet.
- [x] Implement and test global decontamination over all 15,000 AI plus 3,500 VISION audit records,
      not only previously passed peers. Exact SHA and frozen dHash+pHash components preserve parent
      units; REAL/AI components lose both sides; source-cap trimming respects four-variant groups.
      The method checkpoint passes 32 focused E32 tests and opens no production receipt.
- [x] Run the committed overlay on the production receipts, verify every <=20% share and receipt
      binding, then freeze only the role-free eligible subset; never add a replacement row.
      Result: 14,786/15,000 AI and 3,497/3,500 VISION rows remain; maximum AI source share is
      19.998647%, no REAL/AI duplicate component exists, and all seven audit SHAs are bound.
- [ ] Match semantic topics across classes before representation training. Measure topic/source,
      format, geometry, compression and bytes/pixel shortcuts on native and every proposed model
      input. Apply transport augmentation with the same probability/range to REAL and AI; never
      make PNG/JPEG, resize or screenshot history a label proxy.
- **Acceptance:** 10,000–20,000 decontaminated AI parents, five or more verified current families,
  source caps, topic coverage and zero overlap with E30/Qwen/ITW/API final roles. Native risks and
  safe model-input conditions are frozen in `DATASETS.md` before training.

### Phase C3 — freeze TRAIN/CALIBRATION and the Champions League test battery

- [x] Precommit the first role-free-to-role transition before reading image bytes. Balance the
      parent pool at 11,344 REAL + 11,344 AI. Retain all eligible REAL; deterministically select AI
      as Qwen 2,232, FLUX.2 2,232, Nano 2,227, GPT 2,227, NBP 200 and Community 2,226. Preserve
      Qwen/FLUX prompt groups as indivisible. Assign about 20% CALIBRATION within every source by
      stable parent-group hashing; keep VISION and CSAFE physical devices disjoint, FODB scenes
      disjoint, Qwen/FLUX prompts disjoint and Community generator identities disjoint. FODB's
      crossed 27-device x 143-scene design makes simultaneous device- and scene-disjoint roles
      impossible without placing the entire collection in one role; prioritize scene identity and
      report this limitation explicitly. The detailed manifest stays on the SSD; Git receives its
      hash, role/source/group counts and leakage checks only. No image byte, feature or score may
      influence selection or role assignment.
- [x] Implement and test the metadata-only role freezer before production use. It binds the final
      overlay and every audit SHA, uses deterministic exact group-preserving selection plus
      nearest-target subset assignment, rejects duplicate IDs/role-group leakage/empty role cells
      and writes only the detailed manifest to the SSD. Eight focused manifest/overlay tests pass.
- [x] Build a balanced parent manifest with source/device/generator/scene-disjoint folds. TRAIN may
      fit representations and heads; CALIBRATION may select aggregation, abstention and thresholds;
      neither may receive a row or derivative from DEVELOPMENT or LOCKED FINAL. Result: 22,688
      parents, exactly 11,344/class; TRAIN 18,154 and CALIBRATION 4,534, with zero protected-group
      overlap and every source represented in both roles.
- [ ] Keep tests as separate arms and report source-macro metrics; never pool them into one large
      accuracy number that lets the largest arm hide a failure:
      1. **E30 DEVELOPMENT:** already consumed, diagnostic comparison only.
      2. **Owner-real stress:** the already-scored gallery remains DEVELOPMENT; only new unscored
         content can form a locked owner-real arm, grouped by event/burst/device.
      3. **API-current LOCKED:** about 1,000 newly generated parents across at least five current
         commercial families, using a frozen topic/prompt matrix and balanced provider counts.
      4. **Unseen-camera/web LOCKED:** authentic devices/sources absent from TRAIN and CALIBRATION.
      5. **ITW-SM LOCKED EXTERNAL:** untouched 10,000-row social-media benchmark after overlap
         screening and access approval.
      6. **Qwen LOCKED FINAL:** retain the existing conditional one-shot scout and its small-cell
         claim limit.
- [ ] Produce transport columns from each parent (`native`, standardized JPEG, q90/q75/q50,
      resize/screenshot-like) but count uncertainty and confidence intervals by parent, not by
      correlated derivative.
- **Acceptance:** immutable manifests/hashes and a role-access test prove that candidate code cannot
  read a locked arm early. Threshold-independent AUC/AP and thresholded AI recall, REAL recall,
  balanced accuracy, F1, macro/worst-source FP/FN and parent-group bootstrap intervals are fixed.

### Phase C4 — screen representations before paying for full fine-tuning

- [x] Precommit the first runnable R0 contract before materializing model inputs. Decode every C3
      parent with EXIF orientation, convert to RGB, resize the short side to 256, center-crop 224
      and re-encode all classes identically as JPEG quality 90 / 4:4:4 under the external E32
      model-input root. This removes container, mode, geometry and filename from the model API,
      though pre-existing compression/content bias remains a limitation. Bind every derived byte
      and the complete input receipt to C3's detailed-manifest SHA; never read DEVELOPMENT/LOCKED.
      Extract the cached `vit_small_patch14_dinov2.lvd142m` frozen final embedding, fit only a
      standardized class-weighted logistic head on TRAIN over C in {0.01, 0.1, 1.0, 10.0}, select
      C by CALIBRATION AUC with smaller-C tie break, and select the lowest threshold satisfying
      CALIBRATION authentic macro FP <=10% and worst-source FP <=20%. Report AUC/AP, recall,
      balanced accuracy, F1 and per-source errors. Save a locally runnable, hash-bound artifact.
      CALIBRATION is source-stratified but group-held-out; genuinely unseen-source evidence remains
      reserved for DEVELOPMENT/LOCKED and no final-generalization claim is permitted here.
- [x] Implement and test the resumable standardized-input realizer and receipt-bound R0 trainer
      before production bytes. The realizer rechecks each original SHA and refuses changed derived
      bytes; the trainer rechecks all 22,688 input hashes, caches record-aligned features and saves
      a hash-bound artifact. Thirteen focused input/train/role tests pass.
- [x] Realize every C3 parent under the fixed R0 input contract: 22,688/22,688 standardized JPEGs,
      487,845,683 logical bytes, exactly 11,344/class and unchanged TRAIN/CALIBRATION counts. Every
      original and derived SHA passes; no protected role is read. Freeze receipt SHA
      `2255b123...5199` before DINO feature extraction.
- [x] Complete the R0 frozen DINOv2-S screen and save a runnable candidate. Selected C=0.1;
      CALIBRATION AUC 0.9964, AP 0.9968, AI recall 99.07%, REAL recall 90.14%, balanced accuracy
      94.60%, macro REAL FP 9.97% and worst-source FP 13.84%. Every preregistered screen check
      passes. Artifact SHA is `7f170340...a85e`; this remains group-held-out/source-stratified
      evidence, not unseen-source final validation.
- [x] Precommit the serving smoke boundary: add a hash-verifying `pixelproof-predict-e32` CLI that
      reproduces the exact EXIF/RGB/resize/crop/JPEG contract in memory, verifies the fitted
      artifact and cached DINO weights, and emits score/threshold/verdict JSON per image. Test the
      implementation on synthetic images first, then score the previously consumed 210-image
      owner gallery as DEVELOPMENT only; never relabel it locked or use its results to refit.
- [x] Implement the hash-verified batch/single-image E32 CLI and aggregate-only owner-gallery smoke
      runner before opening gallery pixels. The CLI reproduces the JPEG round-trip exactly, emits
      JSON and rejects unsupported paths; ten focused candidate/input/trainer tests pass.
- [x] Run the frozen E32 R0 artifact once on the previously consumed owner-real DEVELOPMENT
      gallery without refit: 210 stills scored, one MOV excluded, 159 false positives and only
      24.29% REAL recall at threshold 0.141444. Preserve the internal CALIBRATION pass and this
      external-pipeline failure together. R0 remains runnable research software but cannot advance
      to serving or LOCKED FINAL; do not tune its threshold on this gallery.
- [x] Precommit a cheap leave-one-collection-out (LOCO) diagnosis over the frozen R0 features. For
      each of nine sources, remove every row from head fitting and threshold selection; fit C=0.1
      on remaining TRAIN, choose the same FP-budget threshold on remaining CALIBRATION and score
      all held-out rows. Report REAL FP or AI recall per held-out source. This diagnostic cannot
      mutate the accepted artifact or promote a candidate; it decides whether data/source coverage
      must precede richer representations.
- [x] Implement and test the receipt/hash-bound nine-arm LOCO runner before results. It uses the
      frozen feature matrix only, rejects a missing class after exclusion, applies the original FP
      budget and emits source-specific FP/recall without mutating the artifact. Five focused tests
      pass.
- [x] Complete nine LOCO diagnostics. Held-out AI transfer remains strong: macro recall 98.34%,
      worst source 95.78%. Held-out REAL transfer fails the budget: macro FP 23.47%, worst FODB
      34.85% (CSAFE 15.74%, VISION 19.82%). Combined with owner-gallery FP 75.71%, prioritize a
      fourth format/content-matched REAL collection and source-held-out real gating before PE-Core,
      intermediate-block or full-fine-tune expense.
- [x] Reject the tempting already-local REAL shortcuts before enrollment. CommunityForensics-Small
      exposes 32,912 REAL rows but every one is FFHQ/face-only; `34data` exposes 8,000 diverse JPEGs
      through an unofficial repack with no dataset card/licence/provenance; the `theminji` real
      parquets likewise lack upstream provenance/licence. None may silently become E32 correction
      data merely because it is local.
- [x] Precommit an R1a forensic-representation screen before feature extraction. Reuse the exact
      22,688 R0 standardized inputs and C3 roles, but extract the frozen CLS embedding from the
      pinned/hash-verified MIT Community-Forensics ViT-S via its official processor. Fit the same
      class-weighted C grid and authentic FP-budget threshold on TRAIN/CALIBRATION. This is not an
      ensemble and does not open the owner gallery. If the internal screen passes, freeze the
      artifact first; only a separately precommitted refit-free gallery stress may then decide
      whether the forensic trunk improves authentic-pipeline transfer.
- [x] Implement and test the R1a receipt/weight/record-bound feature cache and fixed head screen
      before production extraction. It verifies every standardized input SHA, freezes CLS features,
      evaluates the preregistered C grid and writes a separate artifact/evidence receipt. Four
      focused CF-head/threshold tests pass.
- [x] Complete and freeze the R1a internal screen. Selected C=0.01; CALIBRATION AUC 0.99822, AP
      0.99835, AI recall 99.91%, REAL recall 90.05%, balanced accuracy 94.98%, macro REAL FP 9.97%
      and worst-source FP 12.77%. All five gates pass; artifact SHA `6288acba...d670`. Preserve it
      before any owner-gallery access.
- [x] Precommit R1a's external stress after artifact freeze. Add a hash-verified CF-ViT/head scorer
      that first reproduces R0's standardized-array JPEG round-trip, then uses the pinned official
      processor and frozen CLS head. Score the same 210 owner stills once at threshold 0.118110;
      exclude MOV, forbid refit/threshold change and compare REAL recall directly with R0's 24.29%
      and the original frozen CF decision's historical 99.51%.
- [x] Implement and unit-test the R1a scorer and aggregate-only DEVELOPMENT runner before reopening
      owner pixels. The scorer hard-verifies artifact/revision/weight identities, applies the exact
      R0 JPEG round-trip followed by the official CF processor and exposes single/batch JSON through
      `pixelproof-predict-e32-cf`; four focused candidate/input/trainer tests pass.
- [x] Run the frozen R1a artifact once on the 210 supported owner stills with no refit or threshold
      change. It fails: 154 false positives, 26.67% REAL recall (R0 24.29%) and median AI score
      0.4892. Freeze evidence SHA `2e242ef5...b3a`; reject R1a from serving/LOCKED advancement.
      The near-identical failure across DINOv2-S and CF-ViT proves that the present E32 real pool
      and source-stratified head objective—not merely the encoder—are the limiting components.
- [ ] Enroll a fourth, licensed and provenance-complete REAL camera collection only after recording
      its source/device/scene groups and decontaminating it against every protected role. Re-run a
      REAL-source-held-out gate before any R2/R3 expense; do not use the owner gallery for training
      or threshold selection. Until this data gate exists, retain E26 as the working demo and both
      E32 artifacts as reproducible rejected controls.
- [x] Freeze the corrective acquisition before new bytes. Use CSAFE MCSIDB `iPhone14.zip`
      (20,428,338,922 B, MD5 `dfc01c89...946c`, CC BY 4.0) only as a role-free TRAIN/CALIBRATION
      candidate after natural-only inventory/audit. Independently freeze IPN-NFID v3's twelve
      linked device articles: exactly 960 `natural` JPEGs / 3,889,897,594 B, CC BY 4.0, as a
      source-held-out DEVELOPMENT set that may never fit data, representation, threshold or policy.
      API article/version/licence/file-size/MD5 drift and <100 GiB free space are hard stops.
- [x] Implement and test resumable, MD5-bound acquisition for both frozen correction sources.
      `e32_r1b_acquisition.py` separates metadata freeze from transfer, preserves `.partial` bytes,
      enforces the 100 GiB floor and refuses article/version/licence/size/checksum drift. Four
      focused selection/contract tests pass; commit the method before selected-byte transfer.
- [x] Download and verify IPN-NFID natural bytes and CSAFE iPhone 14 archive, preserving partials on
      interruption. Inventory iPhone 14 before selecting natural members; decode/hash/decontaminate
      both sources and record exact realized counts/bytes/limitations in DATASETS and evidence.
      IPN transfer is complete at 960/960 files, 12 devices and 3,889,897,594 MD5-verified bytes.
      Before decoding it, implement a receipt-bound DEVELOPMENT audit that preserves shared scene
      groups, fails decode/exact/protected-peer overlap and records dHash+pHash candidates without
      allowing any model score. Twenty focused audit/acquisition/realization tests pass; commit the
      audit method before opening IPN pixels. Production audit passes: 960 RGB JPEGs, all with EXIF,
      960 unique SHA, 80 shared scene groups and zero protected/peer/cross-scene collision. Preserve
      detailed SHA `f5827dce...243b`; do not score until R1b artifact freeze.
      iPhone 14 transfer also passes exact size+MD5. Central-directory-only inspection finds the
      expected CSAFE shape: 7,996 JPEGs, ten physical devices, 4,000 blank and 3,996 natural rows
      across front/telephoto/ultra/wide. Precommit a receipt-bound CRC/path/symlink/encryption/
      expansion inventory, natural-only freezer and atomic extractor before reading member pixels.
      Implementation passes 24 combined archive/acquisition tests; commit before production CRC.
      Production inventory passes all CRC/path/ratio checks: archive SHA `22f04a95...8cbb9`, exact
      content/device/lens counts and no unknown member. Freeze inventory SHA `8931a535...912e`
      before running the already-implemented natural-only metadata selector. Selection freezes all
      3,996 natural rows (398-400/device; 998-1,000/lens), excludes 4,000 blank and binds detailed
      SHA `88dc326e...7b74`; commit before extraction. Atomic extraction passes 3,996/3,996 at
      12,914,703,500 B with receipt SHA `46b36e56...09de`. Before pixel decode, precommit a
      receipt-bound realization that checks format/EXIF/SHA+dHash+pHash, protected E30/passed peers,
      stored IPN hashes and owner-gallery exact bytes only (identity must remain `390e3c21...ac09`);
      it must not score IPN/owner or assign a TRAIN role. Implementation passes 18 focused
      iPhone/realization/protected-identity tests; commit before production pixel decode.
      Production realization decodes all 3,996 RGB+EXIF parents and finds zero protected/IPN/owner
      overlap, but correctly stops on one two-image near-identical burst (`IMG_1290/1291.JPG`).
      It also records 3,945 MPO containers and 51 JPEG. Preserve rejected audit SHA
      `8325aaf4...05fd`; precommit a deterministic overlay that excludes the entire two-row
      perceptual component (never choose a preferred side) and freezes 3,994 role-free candidates.
      Overlay implementation binds the stopped audit and exact component; two focused tests pass.
      Commit before production eligibility freeze. Production overlay passes at 3,994 eligible,
      two excluded, detailed SHA `a71c4a06...57bf`; raw files remain intact.
- [x] Preserve the first iPhone 14 single-stream stop at 92,274,688 bytes and precommit four-range
      recovery before changing transfer code. Split only the exact remaining interval, require HTTP
      206 plus exact `Content-Range`/length per part, assemble prefix+ranges into a new temporary
      file, verify the published whole-file MD5 and atomically promote; retain every partial on any
      failure. The concurrently independent IPN transfer may continue. Implementation and combined
      acquisition tests pass (18/18); commit before launching production ranges.
- [x] Freeze R1b roles by adding only audited iPhone 14 natural parents to the training-side REAL
      pool with device/scene grouping and a balanced source-capped AI selection. Refit R0/R1a heads
      under the unchanged CALIBRATION budgets; no owner/IPN pixel may be opened before artifacts
      freeze. Advance only the stronger internal candidate.
      Controlled correction rule: preserve all 22,688 C3 roles byte-for-byte and append the 3,994
      eligible iPhone parents only; allocate eight complete physical devices to TRAIN and two to
      CALIBRATION by stable hash. Do not add AI rows or rebalance—class-weighted heads isolate the
      causal effect of authentic Apple coverage. Standardize every appended parent through the
      identical JPEG q90/4:4:4 route before either frozen encoder. Manifest extension implementation
      preserves the C3 prefix and device groups; six focused role tests pass. Production manifest
      freezes 26,682 rows: AI 11,344 / REAL 15,338; TRAIN 21,349 / CALIBRATION 5,333. iPhone is
      TRAIN 3,195 on eight devices and CALIBRATION 799 on devices 4/8, with zero group leakage.
      Detailed SHA `16deb276...750f`; commit before derived input work.
      Input extension implementation hard-binds the R1b manifest and old R0 receipt, reuses every
      old derived byte and appends only iPhone through the exact EXIF/RGB/256/224/JPEG-q90 contract.
      Seven focused input tests pass. Production realizes 26,682/26,682 rows / 568,959,891 logical
      bytes, detailed receipt SHA `400a990d...6af8`; commit before feature work.
      Trainer implementation reuses each frozen 22,688-row feature archive, extracts only 3,994
      iPhone embeddings, merges by record id and refits the unchanged class-weighted C grid/FP
      budgets. Six focused head/merge tests pass. Run DINO then CF separately. If both pass, select
      higher CALIBRATION AUC; exact AUC tie -> smaller selected C -> DINO lexical tie. Freeze that
      choice before opening IPN/owner model scores; if neither passes, stop.
      Both pass. DINO AUC 0.996860 / current-AI macro 99.18% / macro-worst REAL FP 9.97/15.91%;
      CF AUC 0.998079 / current-AI macro 99.82% / macro-worst REAL FP 9.97/12.64%. Frozen rule selects
      CF, C=0.01, threshold 0.125935, artifact SHA `68a54aa2...701c`. Selection receipt is committed
      before any IPN/owner model score; DINO cannot be promoted by external outcomes.
- [x] Score the frozen R1b candidate once on all IPN-NFID devices and the consumed owner gallery.
      Require <=20% worst-device IPN FP and <=20% owner-gallery FP while preserving >=90% modern-AI
      recall before any locked AI arm or serving replacement. Failure returns to data/objective
      redesign; it cannot be threshold-repaired on either test.
      External method is precommitted after CF selection: hard-bind artifact `68a54aa2...701c`, CF
      weights and threshold 0.125935; verify IPN realization `f5827dce...243b` and owner identity
      `390e3c21...ac09`; score CF only, once, with exact standardized JPEG round-trip. Report every
      IPN device plus aggregate and owner aggregate/high scores. No DINO fallback, refit or policy.
      Hash-verified CLI and aggregate-only runner implemented; three focused tests pass. Commit
      method before loading either DEVELOPMENT population into the selected model.
      **Result: failed without repair.** IPN produced 249/960 false positives (25.94% macro-device,
      40.0% worst-device FP; 74.06% REAL recall). The owner gallery produced 144/210 false
      positives (31.43% REAL recall). Internal current-AI macro recall remained 99.82%, so both
      authentic gates failed while the AI gate passed. Evidence:
      `evidence/e32_r1b_external_development.json`. R1b is rejected from serving and no LOCKED AI
      set was opened.

### Phase C4-R1c — threshold-first recovery before another training run

- [x] **Record a no-write feasibility diagnostic, not a new result.** Re-score the unchanged R1b
      head on the already-consumed 960 IPN and 210 owner images and sweep thresholds only to decide
      whether clean replication is worth attempting. Do not write an artifact or change serving.
      Result: the first post-hoc threshold satisfying owner FP <=20% and every IPN device FP <=20%
      is 0.863312; owner FP 20.0%, IPN macro/worst FP 5.42%/15.0%, internal current-AI macro/worst
      recall 90.01%/80.0% (six-source macro 91.00%). This rescues the *hypothesis*, not R1b: the
      number is test-derived and permanently forbidden from candidate selection.
- [ ] **Freeze new roles before acquiring or scoring pixels.** Create three disjoint parent-level
      populations with exact hashes, provenance, licences and camera/generator groups:
      1. `R1C_CAL`: at least 1,000 authentic parents from >=5 previously unused pipelines, nominally
         >=200 per pipeline, spanning native modern phones/cameras plus one web/repost pipeline.
      2. `R1C_LOCKED_REAL`: at least 1,000 untouched parents from >=5 other pipelines, never used
         for threshold, model, crop, quality rule or stop/go selection.
      3. `R1C_LOCKED_AI`: >=100 native outputs from each of >=5 current families, with generation
         version/date/prompt group and transport parentage; keep Qwen Image Bench sealed until the
         preceding gates pass.
      Prefer unused licensed holdings and a small new multi-phone capture over another 100 GB blind
      download. IPN, owner gallery, E30 MLLM and every prior named test remain DEVELOPMENT only.
- [ ] **Build R1c-T, a threshold-only candidate.** Keep the exact R1b CF backbone, head, weights,
      input transform and score direction. Fit no model parameter. Select one threshold solely from
      `R1C_CAL` plus the existing frozen internal AI CAL rows using source-wise finite-sample
      quantiles: target authentic macro FP <=5%, every-source FP <=10%, current-AI macro recall
      >=80% and every sufficiently sized AI family >=60%. If no threshold meets all four, reject
      R1c-T; never loosen a gate after seeing DEVELOPMENT.
- [ ] **Separate calibration quality from discrimination.** Report ROC/PR curves, source-wise score
      histograms, Brier score and expected calibration error, but do not mistake temperature,
      isotonic or beta calibration for improved ranking. Add quality/transport-conditioned
      calibration only if pre-registered CAL ablations show a stable score shift across native,
      JPEG q90/q75/q50, resize and blur views. Every derivative inherits its parent role; the same
      transform policy is applied to both labels.
- [ ] **Freeze R1c-T, then reopen consumed DEVELOPMENT only as a gate.** Score IPN, owner gallery and
      E30's five transport views without any refit. Require owner FP <=20%, IPN macro FP <=10%,
      IPN worst-device FP <=20%, E30 current-AI macro recall >=60%, every AI family >=40% and no
      transport recall loss above 15 points. Bootstrap by parent/device and report 95% intervals.
      Failure moves to R1c-P; success freezes hashes and permits exactly one locked run.
- [ ] **Run the locked final once.** Require authentic macro FP <=10%, worst-pipeline FP <=20%,
      modern-AI macro recall >=80%, weakest family recall >=60%, ROC AUC >=0.90 and balanced
      accuracy >=0.85, with every input/failure counted. Only this result may promote R1c-T into
      the canonical scorer and the web/API decision path. Below threshold remains “insufficient
      evidence”, never a certificate that the image is real.

### Phase C4-R1c-P — paired-content repair only if threshold transfer fails

- [ ] Build a compact 2,000–4,000-pair ablation from licensed TRAIN real parents and semantically
      matched reconstructions, following B-Free's content alignment and DDA's additional frequency
      alignment. Preserve exact real/synthetic parent links; estimate/match JPEG quality and apply
      identical transport augmentations to both labels. Existing unrelated modern-AI parents remain
      a diversity regularizer, not the source of pair labels.
- [ ] Reuse the pinned CF-ViT representation first. Compare only: frozen linear head, source-balanced
      worst-group head, then a small LoRA adapter if the frozen head fails. Use leave-one-real-source
      and leave-one-generator-family-out outer folds; select by worst-group FP/recall, not pooled
      accuracy. Do not combine source-adversarial losses with label-confounded groups unless every
      domain contains both labels.
- [ ] Add one degradation-stability ablation inspired by NTIRE 2026 and GlobalForge: contrast clean
      and compound JPEG/resize/blur views of the same parent while retaining a global image token.
      A spectral/phase or real-envelope branch (SPAI/REM direction) is a later alternative only if
      the paired CF adapter fails; do not launch several architectures at once.
- [ ] Generator-aware prototypes/auxiliary source labels (GAPL/Hive-inspired) are permitted only if
      modern-AI family recall, rather than authentic FP, becomes the limiting gate. Any ensemble is
      deferred until two independently passing arms show >=5-point complementary recall at unchanged
      real-FP budgets.

**Why this order is current-science aligned:** Community Forensics supports generator breadth, which
R1b already inherits; B-Free and DDA show that semantic and frequency alignment target dataset
shortcuts; SPAI and GlobalForge motivate real-centric/global degradation-stable cues; GAPL warns
that blindly adding generators can eventually create representation conflict; NTIRE 2026 measures
36 realistic transformations. Sources: [Community Forensics (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) ·
[B-Free (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html) ·
[DDA](https://arxiv.org/abs/2505.14359) · [SPAI (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html) ·
[GAPL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_Scaling_Up_AI-Generated_Image_Detection_with_Generator-Aware_Prototypes_CVPR_2026_paper.html) ·
[NTIRE 2026](https://arxiv.org/abs/2604.11487) · [GlobalForge](https://arxiv.org/abs/2607.14684).

### Phase C5 — controlled training, complementarity and final decision

- [ ] Execute in strict cost order: deterministic R1c-T threshold transfer first; only a failed
      clean gate permits R1c-P paired training; only a failed paired CF adapter permits a new
      spectral/global representation. R1c-T has no training seed. Every trained R1c-P candidate
      must pass seed 2024 before identical seeds 42/2026 and report source/group intervals.
- [ ] Freeze candidate artifact, preprocessing, threshold, aggregation, abstention, role receipts
      and hashes before DEVELOPMENT. A successful DEVELOPMENT result permits one locked run; it
      does not permit another parameter, threshold or policy choice.
- [ ] Compare model families individually first. An ensemble may be fitted only from out-of-fold
      TRAIN/CALIBRATION rows when two independently passing arms make complementary errors. Require
      at least +5 percentage points macro current-AI recall without worsening either authentic FP
      budget; otherwise retain the best single arm, as E9/E31 required.

### Demo hardening — R1b research visibility without promotion (queued 2026-08-27)

- [x] Add the frozen R1b CF head to the local demo-profile API only as an optional, hash-verified
      `research_signal`; never add it to E26's OR verdict, `project_model`, readiness or the
      canonical artifact registry. Reuse the already-loaded CF-ViT backbone when available so the
      demo does not pay for a duplicate model in memory. Absence/failure must degrade only this
      card and remain explicit in `/health`.
- [x] Extend the typed browser contract and show R1b as a visually subordinate “experimental second
      opinion” with its score, frozen threshold, artifact identity and measured 40.0% IPN
      worst-device / 68.57% owner-gallery FP warning. Below threshold must say “insufficient
      evidence”, never REAL; R1b may not influence the page's official E26 result.
- [x] Simplify the one-page flow and interaction polish without hiding uncertainty: clearer upload
      state, stable result hierarchy, restrained motion with reduced-motion support, keyboard/touch
      focus, responsive layout and no fabricated confidence percentage.
- [x] Verify API schema rejection, optional-load behavior, shared-backbone scoring, web parsing,
      production build, accessibility shell and an end-to-end local image run. Then append measured
      results to HISTORY/EXPERIMENTS/SERVING and commit; do not publish a model endpoint from a
      workstation or expose the external disk.
      Completed with the real LaCie artifact and one owner still: E26 returned `insufficient`, R1b
      returned 0.3132/0.1259 (`ai_signal`) and E20 returned 0.9988/0.9895. The disagreement is
      visible by design and R1b has `affects_decision=false`. Full verification passes: 207 Python
      tests, web 6/6 with production build, typecheck, ESLint, dependency graph and artifact
      registry. Local processes were stopped; no disk or model endpoint was published.

### Demo hierarchy correction — answer the user's first question first (queued 2026-08-27)

- [x] Make E32 R1b the first and only primary result card after an upload. State its direct answer
      in plain Turkish (`AI yönünde sinyal` / `yeterli AI sinyali yok`) and show its 0–100 scaled
      signal bar plus frozen threshold. Explicitly say this is a model score, not calibrated
      probability or authenticity proof.
- [x] Move E26, E20, artifact identity, thresholds and external FP measurements below one collapsed
      `Teknik detaylar` control. Remove E20 routing language and tile overlay from the default view;
      these implementation details must not compete with the requested R1b answer.
- [x] Preserve the scientific boundary: presentation priority does not promote R1b into E26's
      decision rule. Verify keyboard/touch behavior, response parsing, production build and a real
      local R1b request; append results to HISTORY/README/SERVING and commit.
      Completed locally: the primary card now explains the threshold crossing in plain language;
      E26/E20 and measured limitations are collapsed under `Teknik detaylar`. The real R1b E2E
      returned 31.3% against the frozen 12.6% threshold. Presentation changed, voting did not.

### Phase C6 — evidence, history and serving boundary

- [ ] After every completed/rejected phase, update `PLAN.md`, append `HISTORY.md`, append the exact
      hypothesis/config/result to `ml/EXPERIMENTS.md`, update `DATASETS.md` for any data change and
      make one scoped commit. Preserve failures and superseded labels rather than rewriting them.
- [ ] Keep third-party/API/personal images, prompts that reveal private content, credentials,
      embeddings and large score tables out of Git. Commit only aggregate evidence and hashes.
- [ ] Replace the served E20 contract only if C5 passes its locked gate and all dependency,
      licence, artifact-integrity, latency and end-to-end tests pass. Otherwise preserve the best
      new candidate behind a clearly research-only scorer with its measured warning.

## Completed goal — E31 SSD audit, representation ladder and evidence-gated ensemble (2026-08-25)

The immediate product goal is a genuinely runnable detector with useful signal on current
generators—not another attractive in-distribution score. The attached LaCie disk makes a broader
training pool possible, but volume alone is not the remedy. E20 was already trained after the
E19 label correction on 48,037 balanced 128 px tiles and its three seeds agreed; repeating that
same recipe is therefore not an experiment. E28 also showed that changing only the last head does
not repair the representation.

The proposed multi-model direction is scientifically reasonable only after its component models
show complementary, transferable signal. E9 already rejected eight fixed ResNet/feature blends
(best AUC gain only +0.002). On E30 DEVELOPMENT, the frozen E20 and CF-ViT decisions have zero
positive overlap, yet their OR still detects only 52/600 correlated AI views while falsely
triggering on 28/300 real views. Connecting weak arms cannot manufacture evidence. E31 therefore
integrates the user's dataset/retraining/ensemble proposal in this order: audit data, build a new
source-aware TRAIN contract, screen genuinely different representations, calibrate fusion without
test leakage, and only then ask E30 whether the frozen system advances.

### Phase B0 — freeze the decision tree before implementation

- [x] Preserve the E30 five-role contract. The 900-row MLLM battery and owner gallery remain
      **DEVELOPMENT TEST**; Qwen remains **LOCKED FINAL TEST** and unscored. None may fit weights,
      gates, ensemble coefficients or thresholds.
- [x] Record the no-op boundary: do not rerun E20 unchanged and do not average every available
      checkpoint. Retraining is authorized only after the data composition and/or representation
      differs materially and is identified by a versioned contract.
- [x] Use the external disk read-only during audit. Ignore exFAT AppleDouble `._*` files, never
      modify third-party datasets in place, and keep image bytes out of Git.
- [x] Fix the advancement order below before producing an E31 model score.
- **Acceptance:** this section is committed before an E31 audit artifact, TRAIN v2 manifest,
  embedding cache, new checkpoint or ensemble fit exists.

### Phase B1 — inventory and scientifically audit the SSD holdings

- [x] Add a deterministic metadata-first audit command that resolves dataset roots explicitly and
      reports physical files/bytes, Parquet rows/schema, label direction, source/generator
      coverage, formats/geometries, decode failures and duplicate hashes without writing to the
      external disk.
- [x] Re-run shortcut checks at each proposed model input: native whole image and fixed native
      tile/encoder view. Reject any mode where format, resolution, aspect ratio or compression can
      separate labels beyond the existing frozen shortcut ceiling.
- [ ] Hash-check proposed TRAIN content against E22/E24 calibration, owner gallery, all E30
      DEVELOPMENT and LOCKED manifests, and named test-only datasets. Exact/content overlap is a
      hard failure; unresolved provenance is recorded rather than guessed.
- [x] Commit a compact aggregate evidence file and update `DATASETS.md`, `ml/EXPERIMENTS.md` and
      append-only `HISTORY.md`. No copied images or per-image private identifiers enter Git.
- **Current pre-plan observation:** the disk contains about 255 GB of candidate data.
  CommunityForensics-Small has 44,884 rows (32,912 real, 11,972 AI) and 300 distinct AI
  `model_name` values, but its Real/LatDiff architecture and 1024/512 geometry separation makes
  whole-image use unsafe. AI-vs-Real-balanced has 143,070 balanced rows; AIGC has 125,026 balanced
  rows over 18 generator codes; ai-vs-real-200k has 241,609 rows. These counts establish
  availability, not eligibility.
- **Acceptance:** every included/excluded source and safe input mode has a machine-readable reason;
  the audit can be rerun from a user-supplied root and fails clearly if the disk is absent.
- **B1 metadata/probe checkpoint:** registered sources occupy 173.58 GB and inventory-only sources
  97.34 GB. Complete metadata covers 603,991 Parquet rows. The bounded first/middle/last-shard
  probe decoded 3,000/3,000 images, found zero sampled exact overlaps against 980 E30 protected
  parent/derived hashes, and confirmed native shortcut AUCs of 1.000 / 0.967 / 0.841 for
  CommunityForensics / AIGC / ai-vs-real-200k. Their fixed 128 probes pass at 0.636 / 0.540 /
  0.552. B1 remains open only for the hard leakage condition: B2 must select exact TRAIN-v2 rows
  before every selected row can be hashed against every protected role.

### Phase B2 — freeze a source-aware TRAIN v2 and CALIBRATION contract

- [x] Build manifests by source/generator/pipeline rather than maximizing row count. Balance
      labels, cap dominant sources, preserve rare current generators and assign group-disjoint
      folds so one generator/pipeline cannot appear in both a fit fold and its validation fold.
- [x] Start with audited CommunityForensics-Small plus AI-vs-Real-balanced as controls; admit AIGC,
      ai-vs-real-200k and AI-only Flux/Nano-Banana/GPT holdings only through a representation where
      their known geometry/encoding shortcuts are neutralized and only after leakage checks.
      Test-only Defactify, Julien Lucas modern, CIFAKE, the owner gallery and all E30 sources stay
      excluded.
- [x] Create CALIBRATION from held-out source groups or out-of-fold predictions only. Do not reuse
      a training row, E30 score or final row to select a threshold or fusion rule.
- [x] Freeze exact row ids, label map, source caps, folds, transforms, acquisition cutoff and
      manifest SHA before feature extraction or training.
- **Acceptance:** shortcut probes pass, exact/content leakage is zero, each fold has class and
  source support, and re-running selection reproduces the same manifest hash.
- **Frozen pre-byte selection:** 11,300 parents / 5,650 per label across 383 indivisible groups and
  303 named AI identities. TRAIN has 8,561 rows; CALIBRATION has 2,739. Every one of the five
  source collections has both roles, while no group crosses them. CommunityForensics contributes
  8 rows per each of 300 AI generators plus 2,400 real; balanced contributes 2,000 AI / 3,250 real;
  current AI adds 500 Flux, 500 Nano Banana and 250 Nano Banana Pro. AIGC/200k remain deferred
  controls, not silently admitted. Selection SHA is `5907c14b...bfb`; its exact 11,300 row ids are
  committed before realization. B2 acceptance remains open until every frozen row is decoded,
  exact/dHash checked against all protected content, and the deterministic tile archive passes.
- **First realization stop:** all selected bytes decoded, but 3,534/11,300 rows could not satisfy
  the frozen native 128 px / texture-floor input contract. No tile archive was written. Before a
  v2 selection, scan the full balanced source for mechanical eligibility only (decode, dimensions,
  texture; never a model score), freeze the eligible-key set, then reproduce the same source caps
  from eligible rows. The rejected `5907c14b...bfb` selection remains historical evidence.
- **Eligibility + selection v2:** the balanced scan found 24,301/71,535 AI and 21,532/71,535 real
  rows eligible; 47,233 AI + 50,000 real are below 128 px and four additional rows are too flat.
  Eligible-set SHA is `91089e22...eb2`. Selection v2 retains every count/role/group rule, keeps
  7,767 rows and replaces 3,533 mechanically ineligible rows; new selection SHA is
  `5355e430...9b2`. It is committed before the second realization.
- **Selection-v2 rejection:** 11,299/11,300 rows produced tiles, but one Nano Banana Pro row was
  still too flat; 74 rows exactly matched protected tests and nine more matched only by dHash.
  No tile archive was written. Before selection v3, exhaustively screen balanced + Flux + Nano
  Banana + Nano Banana Pro against the same protected exact/dHash library and input floor, then
  select only from that safe set. CommunityForensics stays fixed because its v2 rows had zero
  failure/overlap. Screening is committed before it reads candidate bytes.
- **Protected screen + selection v3:** the complete 163,777-row candidate scan leaves 65,650
  mechanically eligible/decontaminated rows after rejecting 97,982 exact protected matches, 137
  dHash-only matches and six texture-floor failures. The 11,300-row v3 contract preserves every
  class/source/role/group count, keeps 11,216 v2 rows and replaces exactly the 84 previously
  rejected rows from their own sources. Selection SHA is `1a3a5c98...df2e`; exact ids and compact
  aggregate evidence are committed before v3 realization. B2 remains open until realization
  independently reproduces zero overlap/failure and writes the deterministic tile archive.
- **B2 accepted:** independent v3 realization produced all 11,300/11,300 native 128 px tiles with
  zero decode/input failure, zero exact/dHash protected overlap and 11,300 unique tile hashes.
  TRAIN remains 8,561 and CALIBRATION 2,739. The ignored 395,082,960-byte archive SHA is
  `508330c2...9f2b`; compact committed evidence freezes its identity. B3 may now read this archive.

### Phase B3 — screen a small heterogeneous representation ladder

- [x] Run one-seed, low-cost probes before full fine-tuning: (R0) unchanged E20 as the control,
      (R1) a frozen modern ViT/CLIP-family intermediate representation with a regularized linear
      head, and (R2) a low-level residual/frequency specialist. Reuse pinned local artifacts where
      scientifically compatible; record weights, licence, revision and preprocessing exactly.
- [x] Evaluate on source-held-out TRAIN v2 folds and untouched CALIBRATION. Headline gates are
      recall at the fixed real-FP budget, macro/worst-source FP, worst-generator recall and
      compression/resize stability; pooled accuracy cannot advance an arm.
- [x] Stop an arm after one seed if it cannot beat the E20 control materially or adds no
      complementary true positives within the FP budget. Run three seeds only for survivors.
- [x] Fine-tune a backbone or adapter only if the frozen probe has transferable signal but misses
      the gate; otherwise change data/representation instead of spending compute on the same
      failure.
- **Acceptance:** at least one new arm independently meets its pre-registered CALIBRATION gate and
  its three-seed interval is recorded before ensemble work. If none passes, B4 is blocked and the
  negative result becomes the next data/representation decision.
- **Frozen B3 screen implementation (before scores):** R0 uses the unchanged canonical E20 tile
  checkpoint; R1 uses cached `vit_small_patch14_dinov2.lvd142m` frozen at a fixed 224 px encoder
  view; R2 uses the 68 native-tile forensic statistics. R1/R2 fit the same balanced logistic head.
  Fold 1–4 TRAIN predictions are out-of-fold; their real rows choose the lowest threshold meeting
  <=5% source-macro and <=10% worst-source FP. Final heads fit all TRAIN and read CALIBRATION once.
  An arm passes only if CALIBRATION also holds both FP budgets, current Flux/Nano/Nano-Pro macro
  recall is >=50%, and its weakest current source recall is >=30%. E30 remains unopened.
- **B3 result:** E20 control reaches 0.960 AUC / 84.49% current-AI macro recall; frozen DINOv2
  reaches 0.966 / **90.72%** with 4.67% macro and 6.70% worst real FP; the 68-feature arm reaches
  0.849 / 56.24% with 4.24% / 5.51% FP. DINOv2 and the feature arm pass the absolute gate, but
  only DINOv2 materially beats E20. Seeds 42/2024/2026 reproduce identical convex-head metrics.
  No fine-tuning is needed. B4 must first prove whether R2 adds row-level complementarity; it may
  not enter fusion merely because it passed its standalone floor.

### Phase B4 — fit an ensemble only from out-of-fold calibration evidence

- [x] Cache score rows under immutable model/data contracts and measure error correlation,
      disagreement, oracle-union recall and incremental false positives. Only arms with measurable
      complementarity enter fusion.
- [x] Pre-register and compare a deliberately small rule set: OR/max as transparent baselines,
      calibrated logistic stacking, and at most one quality-aware gate whose inputs are label-blind
      image-quality/transport features. No per-test-source manual weight is allowed.
- [x] Select coefficients, abstention band and threshold on out-of-fold CALIBRATION only under the
      <=5% macro / <=10% worst-source FP budget. Report single-arm and fusion ablations so an
      apparent gain cannot hide one useless component.
- [x] Package the winner behind one verdict interface. It may behave like one product model, but
      its response must retain component provenance, uncertainty and `AI detected` /
      `insufficient evidence` asymmetry.
- **Acceptance:** the fused system improves macro current-AI recall by at least 5 percentage points
  over its best component without breaking either FP budget, and the gain survives group-wise
  bootstrap intervals. Otherwise serve the best single arm and record ensemble rejection.
- **Frozen implementation before B4 scores:** CALIBRATION groups receive five deterministic,
  source-stratified meta-folds. Compare DINO alone against DINO+E20 max/stack and DINO+R2
  max/stack only. Each held fold receives coefficients and a real-FP threshold fitted on the other
  four folds. A fused rule advances only at >=5-point current-source macro-recall gain, <=5% macro
  / <=10% worst real FP and paired source/group-bootstrap 95% lower gain >0. Otherwise the frozen
  DINO single arm wins. No quality gate or all-arm search is admitted, and E30 remains unopened.
- **B4 result — fusion rejected:** DINO+E20 max has the largest paired gain, +3.05 points
  (group-bootstrap 95% +1.87 to +4.20), but misses the +5-point gate and reaches 5.34% macro FP.
  DINO+R2 max gains only +1.86 points at 5.08% macro FP; both stacking rules are weaker. E20 catches
  12/24 DINO current-AI misses but adds 50 real false positives; R2 catches 8/24 and adds 42.
  Winner is therefore packaged **single DINOv2**, final CALIBRATION threshold `0.7090073824`,
  candidate SHA `99901219...4d860`. Its head, encoder contract and weight SHA are embedded; E30 is
  still unopened. B5 may test this single candidate once.

### Phase B5 — frozen DEVELOPMENT gate, then one LOCKED FINAL scout

- [x] Commit candidate checkpoint hashes, preprocessing, ensemble rule, thresholds and stop/go
      gates before reading a new E30 score.
- [x] Run once on E30 MLLM DEVELOPMENT. Require the existing working-v1 point gates: macro real FP
      <=5%, worst real-source FP <=10%, current-AI macro recall >=50%, every sufficiently sized
      generator/protocol >=30%, and q75/resize recall loss <=15 points. Treat 20-item cells and
      correlated transport views honestly.
- [x] Only a DEVELOPMENT-passing frozen candidate may consume the sealed Qwen LOCKED scout. Its
      five-per-generator cells are diagnostic and cannot substantiate a production claim; no
      threshold, coefficient or retry changes after seeing them.
- [ ] Keep A5's untouched multi-phone native vault mandatory before claiming general real-photo
      safety.
- **Frozen B5 scorer before DEVELOPMENT:** candidate SHA `99901219...4d860`, single DINOv2 head,
  cached encoder-weight SHA `04d27f34...0081`, one content-id-seeded native 128 px texture-qualified
  tile, fixed 224 px encoder view and threshold `0.7090073824`. Parent/derivative views share the
  same content key. DEVELOPMENT is exactly the existing 900-row content set
  `7634755c...24b8`; no threshold, crop, retry or gate may change after its scores. Qwen code is
  committed but refuses to open its 40+40 LOCKED rows unless a committed evidence file states
  `development_passed` for this exact candidate. The final scout remains diagnostic only.
- **B5 result — DEVELOPMENT failed, Qwen remains sealed:** 897/900 rows scored; three resize views
  were tile-ineligible. AI macro recall is 80.67% and both transport-loss gates pass, but real macro
  FP is **83.63%**, worst real group FP is **100%**, and AUC is 0.385. Standardized-only real FP is
  already 81.67%, so resize is not the root cause. A diagnostic (never adopted) threshold meeting
  the real budgets leaves only 0.33% AI macro recall / 0% worst group, proving recalibration cannot
  repair the inverted ranking. Candidate is rejected; the conditional Qwen step was correctly not
  consumed and B6 serving integration is blocked.

### Phase B6 — integrate, document and preserve presentation evidence

- [x] Replace the served verdict only after B5 passes; keep the last verified contract available
      for rollback. Add readiness, artifact-integrity, deterministic-inference and end-to-end tests.
- [x] Update `MODEL_CARD.md`, `README.md`, `ml/SERVING.md`, `PRESENTATION_EVIDENCE.md`,
      `DATASETS.md`, `ml/EXPERIMENTS.md` and append-only `HISTORY.md` with exact claims and limits.
- [x] Follow the project procedure for every phase: pre-register in this plan, implement/verify,
      close the phase in the roadmap and archive, then make a scoped commit. Never rewrite a failed
      result into success and never commit third-party/personal image bytes.
- **B6 outcome:** the “replace served verdict” condition evaluated false because B5 failed, so the
  verified E20/API/UI contract remains untouched and rollback was unnecessary. E31 is exposed only
  through a research folder scorer whose JSON embeds the 83.63% DEVELOPMENT real-FP warning and
  never says “real.” Documentation/presentation evidence records the clean-data/DINO gain and the
  independent-real falsification together. No third-party/personal bytes or Qwen scores enter Git.

## Ongoing benchmark contract — E30 current-science data and OOD test system (2026-08-25)

The immediate goal is not another unconstrained training run. It is a reproducible data contract
that can tell us whether a candidate is genuinely usable on modern camera photographs and 2026
generation families. E10/E19 showed that a nominal real-vs-AI dataset can be solved through
format, geometry or compression shortcuts; E27 showed that even a promising detector can be
invalidated by evaluation leakage. E30 therefore separates data by scientific role before any new
image is downloaded or score is read.

### The five-role contract

| role | what it may do | what it must never do |
|---|---|---|
| **TRAIN** | fit model weights and training-time preprocessing | set a decision threshold or contribute to a reported test metric |
| **CALIBRATION** | select aggregation/thresholds after weights are frozen | update model weights or select examples from evaluation results |
| **DEVELOPMENT TEST** | reject weak ideas and expose known failure modes repeatedly | support a final/generalization claim after it has guided development |
| **LOCKED FINAL TEST** | run once per pre-registered frozen candidate and decide its gate | influence training, hyperparameters, threshold, row selection or retries |
| **FUTURE TEST** | evaluate generator families released after the candidate's acquisition cutoff | be populated retrospectively with already-seen sources and called future OOD |

Every acquired row must carry one role, source collection, source revision, class, generator or
camera pipeline, content/protocol group, underlying-content id when available, native/derived
status and SHA-256. Derived encodes inherit the role and split of their parent. A command must
refuse ambiguous labels, cross-role hashes and training access to non-TRAIN rows.

The owner's 206-unique-image iPhone gallery is already exposed to several models and remains a
named **development regression** only. It cannot calibrate a threshold, enter training or become a
pristine final test. A new native-camera final vault requires untouched transfers from independent
modern phones; until it exists, E30 may produce a rigorous current-generator result but may not
certify universal real-photo safety.

### Phase A0 — freeze sources, roles and stop/go gates before implementation

- [x] Assign the 2026 `zr-zhang/MLLM-Generated-Image-Detection-Dataset` to a compact, matched
      **DEVELOPMENT TEST**: GPT Image 2, Nano Banana 2 and its real class, stratified independently
      over texture/structure/hybrid. Use only deterministic source paths; never visual quality or
      detector score. Start with the uniformly preprocessed JPEG branch to control transport cost
      and preserve the raw branch as a separate later fidelity regime.
- [x] Assign a deterministic multi-generator slice from the Apache-2.0
      `Qwen/Qwen-Image-Bench` to the first **LOCKED FINAL TEST** candidate. Its source collection is
      independent of MLLMGenSet and covers 2026 families such as GPT Image 2, Nano Banana 2,
      Seedream 5, Qwen Image 2 Pro and FLUX.2. No score may be read until candidate hashes,
      thresholds and the exact selected rows are committed.
- [x] Assign a capped `laionmobile/laion-mobile` URL reconstruction to a real-only
      **DEVELOPMENT TEST** for web-laundered smartphone photographs. It is not a native-camera
      substitute: the source itself says its heavy-ISP tier ends around 2020 and individual image
      licences remain upstream.
- [x] Preserve existing audited E20 training data as **TRAIN** and the E22/E24 source library as
      **CALIBRATION**. E30 downloads do not silently enter either role. The first **FUTURE TEST**
      remains empty and is defined by a release date later than the eventual candidate's frozen
      acquisition cutoff.
- **Working-model v1 gate:** real macro FP <=5%; worst real-source point FP <=10%; current-AI macro
  recall >=50%; every named generator/protocol recall >=30%; JPEG-q75/resize recall loss <=15
  percentage points. Report per-source exact 95% intervals, abstention/coverage and secondary AUC;
  never use pooled accuracy alone. A 40-item source cell is the minimum gate cell, while 5–10 items
  are explicitly scout-only.
- **Acceptance:** this plan is committed before an E30 downloader, local E30 image, model score or
  role manifest exists.

### Phase A1 — implement an enforceable, interruption-safe data contract

- [x] Add a source registry and versioned manifest schema for the five roles. Pin repository
      revisions, dataset/card licences, exact paths/row ids, class direction and acquisition
      cutoff; store third-party bytes only under ignored local data.
- [x] Add deterministic stratified selection, per-file and total-byte preflight, bounded retry,
      resumable atomic writes, content-type/decoder/geometry validation and SHA-256/dHash
      deduplication against all available train/calibration/development/final manifests.
- [x] Add merged-pool shortcut probes for file format, width, height, aspect, squareness and
      bytes-per-pixel. Audit native and standardized encodes separately; never erase the raw/native
      regime to manufacture a clean result.
- [x] Enforce role boundaries in code: TRAIN loaders reject non-TRAIN rows; final scoring requires
      a committed candidate/threshold contract and writes an immutable run receipt; derived copies
      cannot cross their parent's role.
- **Acceptance:** focused tests cover selection, label direction, byte ceilings, resume, role
  violations, duplicate leakage and shortcut-audit failure; full Python tests, compileall and
  dependency checks pass before downloading images.
- **Implemented 2026-08-25 before download:** `e30_sources.json` pins MLLMGenSet
  `1498eead...b9de`, Qwen Image Bench `d2493deb...7038` and LAION-Mobile
  `0c60f598...3465`. `pixelproof.data_contract` validates the five roles, explicit real/AI label
  direction, revisions, safe paths, parent inheritance, exact/underlying-content leakage,
  training-role access, byte gates, metadata-only shortcut AUC and immutable final-run receipts.
  `e30_data_system.py` freezes numeric source paths, resumes Range-capable partial downloads,
  validates hashes/decode/geometry and writes atomic ignored manifests. Twelve focused tests and
  the complete **65/65** suite passed; compileall and `pip check` passed. No E30 image existed when
  these checks completed.
- **Network compatibility correction before first byte:** the first A2 invocation reached the
  frozen 180-row selection but the installed Hugging Face `httpx` session rejected the
  requests-style `stream=True` argument before opening a response or local partial file. The
  adapter now uses the client's streamed `build_request`/`send` path, consumes either httpx
  `iter_bytes` or requests-compatible `iter_content`, closes streamed responses and preserves the
  same Range resume contract. The selection SHA and rows are unchanged.

### Phase A2 — realize the low-bandwidth development battery

- [x] Download a deterministic MLLMGenSet preprocessed slice with 20 examples per
      class x artifact regime: 2 AI generators x 3 regimes x 20 = 120 AI plus
      3 regimes x 20 = 60 matched real, 180 images total. Require all nine cells and disclose this
      as a standardized-JPEG development test, not native-output performance.
- [x] Attempt the pre-registered capped LAION-Mobile real-only slice across eight declared
      phone/pipeline groups. The unchanged 375 KB/file and 30 MB arm limits yielded only 55/80
      eligible URLs (10/10 in four Apple groups; 9/10, 5/10, 1/10 and 0/10 in the remaining
      groups), so this source is explicitly `source_incomplete` and no partial arm is downloaded
      or silently rebalanced. The ignored diagnostic retains upstream URL/hash and failure cause.
- [x] Materialize matched deterministic q90/q75/q50 and resize variants locally without network
      cost, after parent-role assignment. Keep every derivative beside its parent id so no variant
      can leak across roles.
- **Low-bandwidth ceiling:** target <=30 MB for development bytes and stop at 40 MB. A source that
  cannot meet its frozen cell counts within the ceiling is reported incomplete rather than
  silently rebalanced.
- **Acceptance:** complete. The realized MLLM arm is 180 parents / 4,419,610 B plus 720 local
  derivatives / 14,029,255 B (18,448,865 image bytes total). All five transport-specific metadata
  probes pass the frozen AUC <=0.65 rule (0.610–0.636). Exact hashes and the LAION incomplete-source
  audit are archived in `evidence/e30_development_realization.json` before detector scoring.

### Phase A3 — freeze and realize the compact current-generator final candidate

- [x] From Qwen Image Bench, freeze 2026 generator directories and deterministic image paths before
      download. The sealed selection contains 5 per generator / 40 total, 37,907,745 declared
      bytes and selection SHA `50e3fec1...eeb`; exact paths are committed in
      `evidence/e30_qwen_sealed_selection.json`. This is scout-only and cannot produce a pass/fail
      claim until a frozen candidate is tested on at least 40 items per reported generator cell.
- [x] Keep original source encodings (the pre-download tree audit corrected the initial all-PNG
      assumption to mixed PNG/JPEG) and separately create role-inherited standardized JPEG
      variants. All 40 parents downloaded and verified at 37,907,745 B; 40 deterministic q90
      derivatives add 9,449,715 B without network use. Both stay below the 70 MB download ceiling.
- [x] Commit only provenance, hashes, aggregate audits and the sealed row list; keep image bytes and
      any per-image visual material ignored. A3 completed with `detector_scored=false`; no locked
      row was inspected or scored. Aggregate realization is in
      `evidence/e30_qwen_realization.json`.
- **Acceptance:** selected rows and content hashes are sealed, cross-collection overlap checks pass,
  and the working tree is clean before any final score is read.

### Phase A4 — establish candidate and threshold contracts, then score once

- [x] Implement the DEVELOPMENT-only, resumable benchmark before reading a score. It refuses a
      changed 900-row content set or model artifact, keeps E20/CF-ViT decision semantics distinct,
      and reports exact 95% binomial intervals plus per-transport/group metrics. Raw rows remain
      ignored; Qwen LOCKED FINAL paths are not accepted by this command.

- [x] Benchmark existing E20/E26 arms on DEVELOPMENT TEST only. E20 and the available E26 CF-ViT
      arm were scored under committed artifact/preprocessing/threshold contracts. Both are rejected
      at development screening; neither is admitted to a LOCKED FINAL TEST run.
- [ ] Score the sealed final candidate exactly once, accounting for every row and failure. The run
      may reject a candidate but may not trigger threshold/row replacement on the same final set.
- [ ] Report macro and worst-group FP/recall, per-generator/protocol results, exact 95% intervals,
      compression/resize deltas, abstention coverage and shortcut-probe context. Keep native and
      standardized transport claims separate.
- **Acceptance:** every metric is reproducible from committed aggregate evidence and ignored local
  manifests; a failed gate sends the next model back to TRAIN/DEVELOPMENT, not into final-set
  retuning.
- **Current disposition:** return to TRAIN/DEVELOPMENT. Qwen remains unscored. E20 descriptive
  all-transport FP/recall is 9.33%/7.67%; CF-ViT is 0%/1.00%. Frozen cells contain 20 underlying
  items, below the 40-item formal gate minimum, but both candidates already miss aggregate point
  targets by large margins. Full aggregate evidence is `evidence/e30_development_benchmark.json`.

### Phase A5 — build the first genuinely unseen native-camera vault

- [ ] Collect at least 40 untouched stills from each of four independent modern phone pipelines
      (target: current iPhone, Samsung, Pixel and one additional device), transferred by USB/AirDrop
      rather than messaging/social media. Balance indoor/outdoor, day/night, portrait, food,
      landscape and close-up scenes; retain native HEIC/JPEG/MPO provenance privately.
- [ ] Audit and seal the vault without committing personal images, filenames, GPS or per-image
      identifiers. Separate native originals from explicitly derived transport variants.
- [ ] Run only a pre-registered frozen candidate. Until this phase passes, label E30 outcomes
      “current-generator benchmark” rather than “universal deployable detector”.

### Phase A6 — full-internet expansion without contaminating the benchmark

- [ ] At the high-bandwidth location, expand by breadth before volume: at least 100 final examples
      per locked generator/pipeline, raw MLLMGenSet fidelity arms, and additional independent 2026
      collections. Download full 3.32 GB MLLMGenSet or 12.7 GB Qwen assets only when the manifest
      shows which rows add a missing generator/protocol/transport cell; do not mirror data merely
      because bandwidth is available.
- [ ] Keep all E30 development/final sources out of TRAIN. Build a separate representation-curated
      TRAIN pool from the existing 255 GB holdings plus independent current-generator sources,
      targeting source diversity and embedding coverage rather than raw image count.
- [ ] Evaluate a frozen modern multimodal encoder plus lightweight linear/adapter head before
      another end-to-end backbone fine-tune. Select representative generator families in embedding
      space, then require the A4/A5 gates and three training seeds before serving.
- [ ] Populate FUTURE TEST only with generator families released after the candidate's frozen
      acquisition cutoff; version each chronological test rather than rewriting an old one.

### Documentation and commit contract

Each phase is pre-registered before code or measurement, then closed with a separate scoped commit.
`DATASETS.md` records where every source came from, exact selected/full bytes and counts, licence,
role, selection reason, limitations and intended use. `ml/EXPERIMENTS.md` records hypotheses,
protocols and measurements. Append-only `HISTORY.md` records the narrative, rejected attempts and
phase-to-commit ledger needed for the internship report. Third-party and personal images never
enter Git.

## Completed diagnostic goal — compact 2025-generator CF-ViT probe (2026-08-25)

The owner requested an internet-sourced, at-most-100 MB AI-only set from popular 2025-or-newer
generators and a one-pass evaluation with the current strongest gallery arm, CF-ViT. Research
selected the MIT-licensed `saneval-ann/saneval-sample` at revision
`e9e188f6018b3d491708f29e7a387f5043dc8841`: its API-generated images cover GPT Image 1, Imagen
4, Imagen 4 Ultra, Nano Banana and Seedream 3. Imagen 3 is excluded because it predates the stated
2025 boundary.

### Phase Q0 — freeze source, size and decision rules before downloading

- [x] Select 100 rows without reading model scores: 20 per generator, balanced as two fixed rows
      from each of five prompt types x two difficulty splits. Preserve source row ids and revision.
- [x] Use the Hugging Face dataset-server cached JPEG representation and disclose that the source
      card describes raw PNG originals; this is therefore a web-recompression diagnostic, not a
      native-file benchmark.
- [x] Freeze a strict 100,000,000-byte downloaded-image ceiling and CF-ViT's already-adopted
      `0.6617392` AI threshold. Abort before scoring if count, model balance, revision or size fails.
- **Acceptance:** PLAN, `ml/EXPERIMENTS.md` and `HISTORY.md` contain this protocol before an image
  is downloaded or a score is read.

### Phase Q1 — build and verify the compact local subset

- [x] Add a reproducible downloader/probe command with retry, schema/revision checks, deterministic
      row selection, byte cap, SHA-256 manifest and resume-safe writes.
- [x] Download into ignored `ml/data/e29_saneval_2025/`; verify exactly 100 unique decodable JPEGs,
      20 per model, and report the exact on-disk bytes below 100 MB.
- **Acceptance:** unit tests cover row selection and the hard byte ceiling; no third-party image is
  committed and the committed evidence contains only provenance, hashes/aggregates and results.
- **Implementation checkpoint 2026-08-25:** `e29_saneval_2025_probe.py` now fetches the pinned row
  schema with bounded retry and resumable revision/expiry-checked 100-row metadata chunks, verifies
  the revision header, selects the pre-registered 100 rows,
  preflights every cache asset, enforces the decimal 100 MB ceiling, validates JPEG/decode/geometry
  and uniqueness, and writes ignored atomic local files plus a SHA-256 manifest. It reuses the
  existing local CF-ViT adapter and frozen threshold for Q2. Selection/size tests passed 2/2; the
  initial full Python suite passed 52/52, compileall and `pip check` passed; the interruption fix's
  focused tests passed 3/3. No image has been downloaded at this checkpoint.
- **Measured 2026-08-25:** exactly 100/100 selected cells downloaded and decoded as unique
  1024x1024 JPEGs. Image bytes total 11,546,660 and the complete local folder, including resumable
  row metadata, manifest and scores, totals 12,092,513 bytes. Content-set SHA-256 is
  `0e5a2452c2eac44846fb3bc0118fc6bb262db814f693f2183d489b0835c1b9be`; all local files are ignored.

### Phase Q2 — run frozen CF-ViT once and report recall

- [x] Score every downloaded image with the existing hash-verified CF-ViT and frozen E24/E26
      threshold. Do not train, calibrate, select rows or change a threshold from these results.
- [x] Report overall and per-generator recall, prompt-type/difficulty diagnostics, failures, score
      distribution, detector hash and dataset limitations. This AI-only set cannot measure false
      positives, specificity, accuracy or AUC.
- **Acceptance:** all 100 inputs are accounted for, compact evidence and append-only history are
  committed, full relevant tests pass and the working tree is clean.
- **Measured 2026-08-25:** hash-verified CF-ViT `275ba982...1692` scored all 100 on MPS with zero
  failures and detected only 19 (**19% recall**). Per model: GPT Image 1 2/20 (10%), Imagen 4 4/20
  (20%), Imagen 4 Ultra 4/20 (20%), Nano Banana 4/20 (20%) and Seedream 3 5/20 (25%). Hard prompts
  were 7/50 (14%) versus simple 12/50 (24%). The frozen threshold did not change. This result
  confirms that strong authentic-photo specificity does not make CF-ViT a strong current-generator
  detector; it remains an external, asymmetric comparison arm rather than a complete solution.

## Completed goal — real iPhone gallery compatibility and measurement (2026-08-24)

A local owner-supplied gallery exposed a serving blocker before model comparison: 187 of 210
JPEG-family images are iPhone two-frame MPO files. Pillow identifies their container as `MPO`, so
the shared decoder rejects them even though the primary frame is a valid JPEG photograph. The
first 23 decodable images are not a representative result and must not be used to tune a model.

### Phase P0 — record the correction and evaluation boundary before code

- [x] Freeze the input-only scope: accept MPO as a JPEG-family container, decode only its primary
      frame, and preserve the existing byte, pixel, dimension, aspect, EXIF and transparency rules.
- [x] Freeze the gallery protocol: score every supported still image once after the fix, retain
      decode/inference failures in the count, and report authentic false positives without fitting
      any threshold or training on the gallery.
- **Acceptance:** the plan and experiment log exist before decoder code changes.

### Phase P1 — support bounded iPhone MPO input

- [x] Add `MPO` to the JPEG-family decoder contract while explicitly seeking only frame zero before
      load/orientation/flattening. MOV remains unsupported.
- [x] Add automated coverage for a multi-frame MPO-like Pillow input and prove JPEG/PNG/WEBP,
      malformed input and resource limits remain unchanged.
- **Acceptance:** full Python tests, compileall and dependency checks pass; real local smoke decodes
  previously rejected iPhone MPO files without changing model artifacts or scores.
- **Measured 2026-08-24:** the shared decoder now treats `MPO` as a JPEG-family container and seeks
  frame zero explicitly. Focused tests passed 12/12 and the full Python suite passed 50/50;
  compileall and `pip check` passed. Real-gallery decode acceptance rose from 23/210 to 137/210.
  The remaining 73 are correctly distinguished as 5712x4284 images above the unchanged 16,000,000
  pixel product ceiling, not format failures. Model artifacts and inference code are untouched.

### Phase P2 — measure every authentic gallery image across all current arms

- [x] Record the default product decoder result for all 210 stills, then run project E20, legacy
      CNN, full-image statistics, legacy tile statistics and the available E26 external arm once
      per unique file under an explicit 26,000,000-pixel local evaluation ceiling. This second
      ceiling is measurement-only and does not alter API policy. Record duplicates separately and
      skip the MOV.
- [x] Run the rejected E28 Stay-Positive checkpoint as a clearly separated diagnostic with its
      already-frozen N2 `top3` threshold. This cannot reverse its rejection or alter serving.
- [x] Report score distributions, authentic FP/abstention counts, cross-model agreement and the
      practical product conclusion. Do not store personal images, GPS, filenames or per-image
      hashes in the repository.
- **Acceptance:** all supported stills are accounted for and aggregate evidence is appended to
  `ml/EXPERIMENTS.md` and `HISTORY.md`; no gallery image enters training or calibration.
- **Measured 2026-08-24:** the folder contained 210 still-image instances, representing 206 unique
  byte-identical inputs plus four duplicate instances, and one unsupported MOV. The unchanged
  product decoder accepted 137/210 instances; all 73 rejections were 24.47 MP files above its
  16 MP ceiling. Under the declared 26 MP measurement-only ceiling, all 206 unique stills decoded
  and every arm completed with zero failures. Authentic false alarms were E20 178/206 (86.4%),
  rejected E28 170/206 (82.5%), legacy CNN 100/206 (plus 18 uncertain), full statistics 134/206
  (plus 40 uncertain), and legacy tiles/auto 206/206. External CF-ViT triggered once and returned
  `insufficient` for 205/206; it never certifies an image as real. E28's rejection is confirmed,
  E20 remains runnable but not trustworthy on this camera pipeline, and the gallery was not used
  for fitting or threshold selection.

## Next research goal — representation feasibility before another training run (2026-08-24)

E28 showed that changing only E20's final-head constraint leaves its source failure intact, and P2
confirmed the same failure on the owner's real camera pipeline. The
next candidate class must therefore expose a materially different representation. The first
feasibility target is RINE, the ECCV 2024 intermediate-CLIP-block detector: its official paper/code
uses trainable importance over multiple encoder-block CLS representations instead of only the last
feature vector. Sources: [paper](https://arxiv.org/abs/2402.19091) and
[official Apache-2.0 repository](https://github.com/mever-team/rine).

This is not an integration decision. The repository licence does not by itself prove every
checkpoint and transitive base weight may be redistributed, and the paper's reported datasets are
not this project's source-wise evaluation. PixelProof will first audit those boundaries, then run
the candidate through its own frozen protocol. E20 remains the runnable project model throughout.

### Phase O0 — record the representation-first direction before implementation

- [x] Convert E28's measured failure into a new representation-level candidate rather than
      evaluation-driven retuning of its head or aggregation.
- [x] Order the work as licence/provenance audit, isolated adapter smoke, frozen-protocol benchmark,
      and only then a project-trained head experiment.
- **Acceptance:** no external checkout, checkpoint or serving change precedes this recorded plan.
- **Recorded 2026-08-24:** the RINE/CLIP feasibility line below was selected from its published
  representation design and official Apache-2.0 repository; no dependency, code or weight has yet
  been added.

### Phase O1 — audit feasibility and ownership boundaries

- [x] Pin an official repository revision; identify the code, RINE checkpoint, CLIP implementation,
      base-weight and training-data licences separately. Record what may be used locally, committed
      and redistributed.
- [x] Map model input, normalization, score direction, checkpoint schema, memory and dependency
      requirements without changing the locked serving environment.
- **Acceptance:** a written PASS/FAIL matrix exists. Any unclear checkpoint or base-weight licence
  stops redistribution and limits the work to a local research comparison.
- **Measured 2026-08-24 — conditional GO:** `ml/RINE_FEASIBILITY.md` pins RINE revision
  `9b7fd585...620`, its Apache-2.0 source and 25,298,182-byte 4-class trainable checkpoint, plus
  OpenAI CLIP revision `d05afc4...35f6`, MIT code and the official ViT-L/14 SHA-256/932,768,134-byte
  base. RINE excludes CLIP parameters from its saved head, but no separate CLIP base-weight licence
  was found; base weights therefore stay local and unredistributed. The unpinned git dependency,
  Python 3.9/Torch 2.1/CUDA assumptions and dynamic `exec` checkpoint loader are rejected. O2 may
  proceed only with a pinned, hash-checked, strict independent adapter outside serving.

### Phase O2 — benchmark the representation in isolation

- [ ] Build the smallest project-owned adapter needed to score images through a pinned, verified
      local RINE candidate; keep its environment/artifacts optional and separate from E20 serving.
- [ ] Prove score direction, deterministic preprocessing, bounded input and a CPU/MPS smoke before
      the real evaluation.
- [ ] Run the unchanged E20 calibration/evaluation split once. Advance only if AUC >=0.850, recall
      >=35%, Defactify FP <=15%, forensic macro FP <=15% and worst-source FP <=30%.
- **Acceptance:** exact per-source evidence is archived; failure ends the candidate without
  integration or evaluation-driven threshold changes.

### Phase O3 — train a project head only after representation feasibility

- [ ] If O2 passes and licensing permits, freeze the pinned CLIP backbone and train only an
      independently implemented intermediate-block importance/projection head on the existing
      audited project training pool. Pre-register its recipe and gates before training.
- [ ] Require seed 2024 advancement followed by seeds 42/1337 before registering any artifact.
- **Acceptance:** only a three-seed, source-gated project-trained head may replace E20; otherwise
  E20 stays runnable and O3 becomes another reportable negative result.

## Completed experiment goal — source-robust project model v2 (2026-08-24)

**Milestone status: completed with a pre-registered rejection on 2026-08-24.** E28 failed N2, so
N3 was correctly cancelled and serving stayed unchanged. The next plan must target representation
or data composition; it must not retune this rejected candidate against its evaluation results.

The runnable-model milestone proved the complete E20 path, but it also froze the central failure:
seed 2024 reaches Defactify AUC 0.720 and recall 48.1% while misclassifying 83.2% of the worst
authentic source. The next goal is therefore narrowly defined: reduce source-specific authentic
false positives without losing the already modest AI signal. This is research advancement, not a
production or universal-authenticity claim.

The candidate is an independent implementation of the Stay-Positive last-layer constraint from
*Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection* (ICML 2025):
freeze E20's feature extractor, reset its linear head to zero, train only that head over
non-negative features, and clamp its feature weights to be non-negative after each optimizer
step. Sources: [paper](https://arxiv.org/abs/2502.07778) and
[official research repository](https://github.com/AniSundar18/AlignedForensics). The repository
page reviewed on 2026-08-24 did not expose an explicit software licence, so no upstream source,
weights or assets will be copied; the experiment will implement the published algorithm from its
mathematical description using this project's existing code and data.

### Phase N0 — pre-register the experiment before implementation

- [x] Record the method, baseline, stop/go gates, evaluation boundary and licence boundary before
      changing model code.
- [x] Preserve E20 seed 2024 as the exact comparator and keep evaluation data out of head training,
      validation, threshold choice and hyperparameter selection.
- **Acceptance:** `PLAN.md`, `ml/EXPERIMENTS.md` and append-only `HISTORY.md` agree on what may be
  tried and what result permits the next phase.
- **Recorded 2026-08-24:** the N1-N4 order and the single-seed/final gates below were frozen before
  implementation. No external code or weight was imported.

### Phase N1 — implement and mechanically verify the constrained head

- [x] Add one reusable, independently written training primitive that freezes E20's backbone,
      exposes non-negative embeddings, zero-initializes the final linear head and prevents negative
      feature weights after every update. The bias remains unconstrained as described in the paper.
- [x] Add a separate experiment command; do not overwrite E20 or change the served artifact.
- [x] Test zero initialization, frozen backbone, non-negative weights, deterministic splitting and
      compatible checkpoint metadata on tiny synthetic data.
- **Acceptance:** focused tests and the complete Python suite pass; a CPU smoke run produces a
  loadable candidate whose feature-weight minimum is at least zero.
- **Measured 2026-08-24:** `pixelproof-train-stay-positive` now loads the verified-shape E20
  ResNet18, freezes its feature extractor, applies exact ImageNet normalization, extracts explicit
  ReLU embeddings and trains only a zero-initialized linear head with BCE/AdamW while projecting
  feature weights to `>= 0` after each update. The bias is unconstrained. Five focused tests cover
  the invariant, frozen backbone, deterministic source+label split, balanced smoke sampling,
  invalid negative features and model compatibility. A real CPU smoke used the canonical E20
  checkpoint and a balanced 120-tile subset, produced a loadable isolated checkpoint at validation
  AUC 0.900 with zero negative weights, and did not modify serving. The full suite passed 48/48;
  compileall and `pip check` passed. Full-data scientific measurement remains N2.

### Phase N2 — run one full seed and apply the frozen advancement gate

- [x] Train seed 2024 against the existing 48,037-tile E20 training corpus. Training/validation
      choices may see only that corpus; the frozen E20 calibration/evaluation split is used once
      after the candidate is fixed.
- [x] Compare with the exact E20 seed-2024 baseline: AUC 0.7197, recall 48.1%, Defactify FP 11.3%,
      forensic macro FP 43.3%, worst-source FP 83.2%.
- **Advance only if all single-seed conditions pass:** AUC >= 0.710, recall >= 42%, Defactify FP
  <= 15%, forensic macro FP <= 35%, and worst-source FP <= 70%. These tolerances prioritize the
  named failure while refusing a trivial always-real classifier.
- **Stop condition:** a failed gate is appended as a negative result and is not integrated; no
  threshold or hyperparameter is retuned against evaluation.
- **Measured 2026-08-24 — rejected:** all 48,037 tiles produced frozen 512-dimensional features;
  the validation-only choice kept epoch 1/15 at AUC 0.8947 with zero negative feature weights.
  E20 protocol v2 then selected `top3` on calibration macro recall and measured untouched
  evaluation AUC 0.7290, recall 48.9%, Defactify FP 12.7%, forensic macro FP 44.6% and worst-source
  FP 85.0% (`RealisticTampering`). The first three gates passed; macro <=35% and worst <=70%
  failed. The candidate is rejected and cannot advance.

### Phase N3 — require three-seed evidence before integration

- [x] Apply the N2 prerequisite before spending two more training/evaluation runs.
- [ ] Only after N2 passes, train seeds 42 and 1337 with the identical frozen recipe and report
      population mean +/- standard deviation for all gate metrics.
- **Final gate:** mean AUC >= 0.740, recall >= 45%, Defactify FP <= 15%, forensic macro FP <= 35%
  and worst-source FP <= 65%; all seeds must remain below 75% worst-source FP.
- **Acceptance:** the stored three-seed result either passes every condition or records an explicit
  rejection. A single favourable seed never becomes the served model.
- **Status 2026-08-24:** cancelled by the pre-registered N2 stop condition. Seeds 42/1337 were not
  run; this is deliberate protocol compliance, not unfinished work.

### Phase N4 — integrate only a passing model and freeze new evidence

- [ ] If N3 passes, register a hash-verified v2 artifact, update the shared scorer/API/UI/model card
      and add a before/after presentation table. Keep E20 addressable for reproducibility.
- [x] If the candidate fails N2 or N3, leave serving unchanged, archive the result, and choose the
      next method from the measured failure rather than adding unverified product features.
- **Acceptance:** serving changes only after a passing three-seed artifact; otherwise the current
  runnable E20 system remains intact and the negative experiment is fully reportable.
- **Measured 2026-08-24:** `evidence/e28_seed2024_rejection.json` preserves the exact baseline,
  candidate, all diagnostic aggregations, individual gate decisions and hashes. The candidate is
  absent from the runtime manifest; E20 remains the runnable project model. The failure says the
  frozen E20 representation—not merely the sign of its final weights—must change next.

## Completed goal — a runnable project-owned model (2026-08-24)

**Milestone status: completed through M6 on 2026-08-24.** The deferred items below are a new
product/research horizon, not missing requirements for the runnable-model milestone.

The immediate goal is not a production-perfect universal detector. It is a reproducible,
project-owned model that the author can start, give an image to, inspect in the web demo and
evaluate on labelled folders. The canonical model is the E20 ResNet-18 trained on native 128 px
tiles, seed 2024. Its existing checkpoint is 44.8 MB and records ImageNet normalization,
`top3` aggregation, a 0.04 texture floor and calibration-only threshold 0.9894907.

This milestone does not turn that model into an authenticity authority. E20's three-seed result
was Defactify evaluation AUC 0.751 +/- 0.033 and recall 49.9% +/- 6.1, while worst-source real
false positives remained 86.2% +/- 3.1. The model is nevertheless a valid, working project
result when its limitations are shown beside it. The external CF-ViT/B-Free decision layer stays
available as a measured comparison, not as a substitute for presenting the project-owned model.

### Phase M0 — record the model-first milestone before implementation

- [x] Name one canonical project-owned checkpoint and freeze its current inference contract.
- [x] Record the ordered M1-M6 implementation plan before changing model-serving code.
- [x] Keep the completed H0-H6 hardening work addressable through the commit ledger below.
- **Acceptance:** the next implementation phase, its evidence and its reporting boundary are
  unambiguous before code changes begin.

### Phase M1 — make the E20 checkpoint a verified runtime artifact

- [x] Add `tile_resnet18_seed2024.pt` to the artifact manifest with SHA-256, training provenance,
      label direction and exact inference schema.
- [x] Implement one reusable E20 loader/scorer that reads the checkpoint contract instead of
      duplicating preprocessing and aggregation constants in serving code.
- [x] Reject missing, tampered or schema-incompatible checkpoints with an actionable status.
- **Acceptance:** offline tests cover valid, missing, tampered and incompatible checkpoints;
  one real local load reproduces the stored seed/model/inference metadata.
- **Measured 2026-08-24:** the new `project_model` artifact group verified
  `tile_resnet18_seed2024.pt` at SHA-256
  `b9f39eda10ba3de54b706d6448b67d93ce8e4c7bae97a685f3c1b57ebfd65adf` before
  deserialization. The real 44,789,451-byte checkpoint loaded on CPU and reproduced arm
  `resnet18`, seed 2024, 128 px tiles, ImageNet normalization, texture floor 0.04, `top3`,
  threshold 0.9894907, split seed 2026 and validation AUC 0.909627. A three-tile smoke score
  completed through the reusable scorer/aggregator. Valid, missing, tampered and incompatible
  cases passed; the full Python suite passed 29/29, compileall and `pip check` passed, and all
  six default registry entries verified offline. API/UI integration remains correctly scoped
  to M2/M3.

### Phase M2 — expose one canonical project-model inference path

- [x] Add a bounded `project_model` inference path using native 128 px tiles, the stored texture
      floor and stored `top3` aggregation; every tile is scored once and the existing 256-tile
      resource ceiling remains in force.
- [x] Return the raw score, stored experimental threshold, triggered/not-triggered result,
      checkpoint hash and explicit `research_only` limitation in the API response.
- [x] Decouple project-model readiness from the optional external verdict arms and from retired
      CNN/statistics artifacts, so one missing comparison model cannot disable the main demo.
- **Acceptance:** API tests cover small/large images, bounded tiles, unavailable artifact and a
  real checkpoint prediction; the same image produces the same aggregate through CLI and API.
- **Measured 2026-08-24:** `project_model` is now the API and CLI default. API responses carry
  score, experimental threshold, trigger state, `research_only`, limitation, revision, seed,
  aggregation, tile count and the verified artifact SHA-256. Unit tests cover a padded 64 px
  image, a 2304 px image capped at exactly 256 tiles, missing project artifact, and project-ready
  operation with both legacy core and external verdict unavailable. The full Python suite passed
  33/33; web lint/type/build and 6/6 web tests stayed clean. On the real MPS runtime,
  `generators.png` used 51 texture-qualified tiles and returned 0.2409 through both the shared
  scorer and HTTP API; the root-invoked CLI returned the same rounded 0.241 against threshold
  0.990. Health reported project/core/decision ready independently and the result included the
  canonical `b9f39e...65adf` hash. This is a functioning research-model path; M3 still owns the
  model-first web presentation.

### Phase M3 — make the web demo model-first

- [x] Replace the four-method-first interaction with one primary action: run the project model.
- [x] Show the project model's experimental result, score, threshold, model revision and honest
      worst-source limitation together; never label a negative result as proof that an image is real.
- [x] Keep the external decision layer in a clearly separated comparison panel when available,
      and move the older CNN/statistics/tile-feature methods behind an optional research-details view.
- **Acceptance:** a user can upload one JPG/PNG/WEBP and understand which result belongs to the
  project-owned model; rendered, contract, accessibility and stale-request tests pass.
- **Measured 2026-08-24:** the Turkish UI now selects `project_model` by default and gives the E20
  result the primary card. It renders raw score, stored threshold, trigger state, revision,
  aggregation, tile count and verified artifact hash prefix together with the measured 86.2% +/-
  3.1 worst-source false-positive limitation. A below-threshold score explicitly says that the
  image has not been proven real. E26 appears only in a separate external-comparison card; the
  retired `auto`, `cnn`, `stats` and `tiles` paths are inside an optional research disclosure.
  The response parser now validates the full project-model payload, including the 64-character
  SHA-256 and positive integer tile contract. `git diff --check`, ESLint, TypeScript, production
  build and all 6/6 web tests passed; rendered-product assertions cover the Turkish model-first
  shell and its limitation, while the existing request-gate test still proves stale cancellation.

### Phase M4 — add a repeatable folder evaluation command

- [x] Provide a command that accepts user-supplied `real/` and `ai/` folders, runs the canonical
      checkpoint once per image and writes machine-readable JSON/CSV results.
- [x] Report image counts, decode failures, ROC-AUC, recall at the stored threshold, FP rate,
      confusion counts and per-folder/source results without silently pooling away failures.
- [x] Include checkpoint hash, configuration, environment and command provenance in each run.
- **Acceptance:** a tiny fixture proves the output schema and error paths; a held-out local subset
  completes end to end and its exact measured result is appended to `ml/EXPERIMENTS.md`.
- **Measured 2026-08-24:** `pixelproof-evaluate-project` recursively discovers supported files in
  separate `real/` and `ai/` roots, applies the same bounded decoder and verified shared E20
  scorer, and writes non-overwriting `results.json` plus `predictions.csv`. Every failure retains
  its row and stage; a partial run writes its evidence then exits non-zero. Fixture coverage proves
  once-only scoring, the complete output schema, perfect known metrics, per-folder grouping,
  malformed-image retention, invalid nested roots and output-overwrite refusal. Python passed
  36/36. The installed command then ran the real `b9f39e...65adf` checkpoint on MPS against the
  four labelled upstream B-Free demo images: 4/4 decoded, AUC 0.500, recall 1.000, FP rate 1.000,
  confusion TP=2/FN=0/FP=2/TN=0. Exact scores and the operational interpretation are appended to
  `ml/EXPERIMENTS.md`; this tiny smoke set verifies execution and illustrates source failure, not
  generalisation performance.

### Phase M5 — provide a one-command local demonstration

- [x] Add a documented bootstrap/check command that verifies dependencies and the canonical
      checkpoint before starting the API and web client.
- [x] Add a smoke command that checks `/health`, submits one image and validates the response.
- [x] Make startup errors identify the missing dependency/artifact/port instead of ending with an
      unexplained traceback.
- **Acceptance:** from a fresh shell on the supported machine, the documented flow reaches a web
  prediction and CLI/folder evaluation without source edits.
- **Measured 2026-08-24:** executable `./tools/pixelproof-demo` now provides `check`, `smoke` and
  `start`. `start` includes the complete preflight, starts loopback API/web process groups, waits
  for canonical-model readiness, validates a real multipart response and shuts both children down
  on one `Ctrl+C`. A new `project` runtime profile skips all retired/external loaders for this
  primary demo while the normal server default remains `full`. The real preflight verified Python
  3.13.5, the locked import/dependency graph, canonical E20 artifact, both installed CLIs, Node
  v25.2.1, npm graph and ports 8799/3000. The clean live run reached health `ready`, predicted the
  tracked `generators.png` on MPS at score 0.2409 versus threshold 0.9895 using 51 tiles and hash
  `b9f39e...65adf`, served the E20 web shell with HTTP 200, then exited both processes cleanly.
  Python passed 41/41. M4's installed folder evaluator remained available in the same preflight.

### Phase M6 — freeze presentation and report evidence

- [x] Add a concise model card covering training data, architecture, inference contract, measured
      strengths, worst-source failure, intended use and prohibited authenticity claims.
- [x] Record every M1-M5 commit, command, test count and measured model result in this plan and the
      append-only experiment log; generate report-ready tables/figures only from stored results.
- [x] Capture one reproducible demo scenario for the internship presentation: input, project-model
      output, comparison output and the explanation of why they may disagree.
- **Acceptance:** a reader can trace every presentation claim to a result file, experiment entry,
  artifact hash and commit without relying on an undocumented manual run.
- **Measured 2026-08-24:** `MODEL_CARD.md` freezes the exact E20 artifact, 48,037-tile source
  inventory, architecture, inference contract, three-seed and deployed-seed metrics, intended use,
  prohibited claims and limitations. `PRESENTATION_EVIDENCE.md` provides the report-ready metric
  table, M0-M5 commit/test ledger, live-demo order and source map. Machine-readable
  `evidence/demo_disagreement.json` binds an upstream-authentic B-Free demo input to SHA-256
  `c7351a...a79360e`, pinned revision `c6a9f89...`, runtime commit `95fe2b2`, canonical model hash
  and exact project/external outputs. Real full-profile HTTP reproduced the disagreement on MPS:
  E20 score 1.0000/0.9895 and 69 tiles versus CF-ViT -2.4631/0.6617 (`insufficient`). The example
  is an E20 false positive and is documented as a limitation, not a success. Tests bind the model
  card and evidence to the manifest, optional input hash and every M0-M5 commit; Python passed
  43/43. The historical `rapor/` boundary now points readers to this current evidence package.

### Deferred until the runnable-model milestone passes

- Public deployment, authentication, rate limiting, autoscaling and latency SLOs.
- A stronger commercially usable arm (for example a pre-registered Stay-Positive experiment).
- C2PA/Content Credentials fusion and Module 2 localisation.
- Any claim that the system is a general-purpose or production-grade authenticity detector.

## Completed hardening commit ledger

| Phase | Commit | Recorded outcome |
|---|---|---|
| H0 | `ef9edaa` | Ordered hardening roadmap written before implementation |
| H1 | `6509ebf` | Web lint, typecheck, build and product-test baseline restored |
| H2 | `18ab632` | Browser/API contract, response validation and request races hardened |
| H3 | `364d9f0` | Image limits, normalization, bounded inference and truthful health added |
| H4 | `d0d856d` | Evaluation leakage fixed; E27 rerun failed G1 and was removed from serving |
| H5 | `dbafd05` | Locked dependencies and hash-verified model artifact registry added |
| H6 | `9830d31` | Documentation aligned; CI, dependency audit and final E2E gates added |

Every completed phase below contains its acceptance checks and measured result. Git history is the
immutable implementation record; `HISTORY.md` receives dated completion summaries and
`ml/EXPERIMENTS.md` remains append-only for scientific results.

## Completed hardening roadmap (2026-08-24)

The E20-E27 research line produced a defensible asymmetric decision layer, but a full
repository audit found that the product, serving, reproducibility and verification layers
have not yet caught up with the experiment discipline. The work below is ordered by risk.
Each phase follows the same rule: implement, verify against its acceptance checks, record
the measured result here, then commit. No unmeasured product claim is introduced.

### Phase H0 — record the plan

- [x] Turn the audit findings into this ordered roadmap before changing product code.
- [x] Keep every later phase in a separate commit and update its checkbox only after its
      acceptance checks pass.

### Phase H1 — restore a trustworthy web verification baseline

- [x] Replace the deleted starter-skeleton tests with tests for the actual PixelProof page.
- [x] Add the Cloudflare Worker types required by TypeScript and make `tsc --noEmit` pass.
- [x] Keep ESLint out of the Python virtualenv, artifacts, external checkouts and generated
      output; make the repository lint command pass on owned source.
- [x] Update vulnerable web dependencies within compatible release lines, then record the
      remaining `npm audit` result instead of claiming that every advisory is exploitable.
- **Acceptance:** `npm test`, `npm run lint` and an explicit TypeScript check all exit zero.
- **Measured 2026-08-24:** `npm test` rebuilt the deployment and passed 2/2 product/hosting
  tests; `npm run lint` and `npm run typecheck` both exited zero. Compatible dependency
  updates reduced `npm audit` from 21 findings to two high-severity entries, both the same
  `vinext@0.0.50 -> image-size@2.0.2` denial-of-service chain. npm's available remediation
  is the breaking `vinext@1.0.0-beta.8` line, so that migration is not hidden inside this
  baseline phase and remains explicit dependency debt.

### Phase H2 — make the web/API contract honest and race-safe

- [x] Replace the hard-coded browser localhost assumption with one documented API-origin
      contract: same-origin or an explicitly configured URL, with local development as a
      deliberate fallback rather than a deploy-time accident.
- [x] Distinguish unavailable service, rejected input and inference failure in the UI.
- [x] Validate the response shape before rendering, cancel stale requests, revoke preview
      URLs on unmount and prevent a cleared/changed image receiving an old result.
- [x] Fix the upload control's nested interactive element, live status announcements,
      keyboard/focus behavior and four-method layout.
- **Acceptance:** unit tests cover API URL selection, response validation and stale-request
  behavior; the deployment build succeeds.
- **Measured 2026-08-24:** the production default now posts to same-origin `/predict`, while
  `NEXT_PUBLIC_PIXELPROOF_API_URL` selects a separately hosted API and development alone
  falls back to `127.0.0.1:8799`. A configured-origin production build embedded the supplied
  HTTPS origin. `npm test` rebuilt successfully and passed 6/6 tests, including four pure
  contract/race/error tests; `npm run lint` and `npm run typecheck` also exited zero.

### Phase H3 — harden inference inputs and execution

- [x] Enforce upload byte, pixel, dimension and supported-format limits with explicit 4xx
      responses; malformed files must not become generic 500 errors.
- [x] Apply EXIF orientation and a documented transparency background before every arm sees
      the image, so preview geometry and model geometry agree.
- [x] Make evidence sufficiency depend on both dimensions and prevent an official verdict
      below the supported floor.
- [x] Bound expensive tile work, move blocking inference off the async event loop, use CUDA
      when available and expose truthful readiness/degraded health.
- [x] Restrict CORS to configured origins and document rate-limit/auth expectations for any
      non-local deployment.
- **Acceptance:** API tests cover invalid bytes, oversize input, tiny/extreme aspect ratios,
  EXIF orientation, transparency, unavailable verdict arms and one valid prediction.
- **Measured 2026-08-24:** `pytest -q` passed 18/18 tests, including six API-policy tests.
  The real local runtime then loaded both CNNs, both feature models and the then-current
  CF-ViT + E27 arms on `mps`, reported `status=ready` with no load errors, and completed
  one 64x64 CNN prediction. Limits are 12 MiB / 16 MP / 16,384 px / 20:1, tile extraction
  is capped at 256, inference runs off the async event loop through one bounded runtime
  slot, wildcard CORS is rejected, and the external auth/rate/queue boundary is recorded
  in `ml/SERVING.md`.
  H4 subsequently rejected and removed E27 under the corrected calibration-only gate.

### Phase H4 — repair the scientific/product contract

- [x] Replace the false UI statement that no source exceeds 10% FP with the exact measured
      operating point and its uncertainty/limited-population wording.
- [x] Ensure the E27 union gate can never tune a threshold by reading evaluation halves;
      evaluation data is measured once after the threshold is frozen.
- [x] Label research outputs as scores, not calibrated probabilities, and describe the tile
      map as a detector-score map rather than proof of manipulation location.
- [x] Emit the megapixel caveat only when an enabled arm actually receives the capped input;
      describe bytes-per-pixel as a heuristic, not a compression classifier.
- [x] Bring the CLI into the same asymmetric `ai` / `insufficient` verdict contract.
- **Acceptance:** pure tests pin the decision/caveat rules and a synthetic protocol test
  proves that changing evaluation scores cannot change a fitted threshold.
- **Measured 2026-08-24:** the corrected calibration-only E27 rerun froze the candidate
  threshold at 21.71, then measured evaluation exactly once: worst-source FP 10.7%
  (iPhone 11/103; Wilson 95% CI 6.1–18.1%) and macro FP 2.95%. The resulting GPT-probe
  recall was only 14.5% (q75 9.0%), below E27's pre-registered >=40% G1, so the E27 arm
  was rejected and removed from serving; the append-only experiment log records the
  correction. Python tests passed 22/22 and web contract/product tests passed 6/6, with
  lint, typecheck and build all clean. Pure tests pin asymmetric CLI/caveat behavior and
  prove that replacing every evaluation arm score cannot move the union threshold.

### Phase H5 — make a clean clone reproducible

- [x] Declare the live service and experiment dependency groups completely and add a locked
      Python environment artifact suitable for the supported Python version.
- [x] Add a runtime artifact manifest containing source, licence, revision, SHA-256, expected
      path and model/feature schema; loading must reject mismatched artifacts.
- [x] Provide an explicit artifact preparation/check command. A missing model must yield an
      actionable readiness result, not an import-time traceback.
- [x] Replace personal absolute dataset paths in active commands with CLI/config/environment
      inputs while preserving the current machine as an optional local configuration.
- [x] Record the project's own licence posture and keep B-Free opt-in/non-commercial use
      separate from the default servable configuration.
- **Acceptance:** documented clean-environment setup reaches a truthful health response;
  artifact verification and missing-artifact paths are tested without network access.
- **Measured 2026-08-24:** the Python 3.13/macOS-arm64 serving lock resolved successfully
  with installed packages ignored, `pip check` found no broken requirements, and editable
  package metadata resolved with both `test` and `experiments` groups. The offline registry
  verified all five default artifacts plus the optional pinned B-Free checkout; the real
  runtime reported `ready` with CF-ViT only by default. Python tests passed 25/25, including
  offline good/tampered/missing/optional artifact cases, and a missing core produced
  `status=unavailable` with an actionable manifest error. Active scripts contain no personal
  absolute path; `PIXELPROOF_DATA_ROOT` / `PIXELPROOF_WORK_ROOT` preserve arbitrary local
  layouts. `LICENSE.md` records no granted project licence and isolates upstream terms.

### Phase H6 — align documentation and automate the gates

- [x] Update README, this plan, DATASETS, the active experiment index and the report boundary
      so each distinguishes the E26 served system, rejected E27 arm, research-only signals
      and Module 2's parked state.
- [x] Add CI for web lint/type/test/build, Python tests and dependency auditing with generated
      directories excluded.
- [x] Remove or regenerate stale local deployment output; a production artifact must never
      contain an old UI or an absolute developer filesystem path.
- [x] Run the final local end-to-end verification and record exact commands/results here.
- **Acceptance:** all CI-equivalent checks pass from owned source, the working tree contains
  no accidental generated files, and the remaining known limitations are stated in README.
- **Measured 2026-08-24:** `npm ci`, `npm run lint`, `npm run typecheck` and `npm test`
  all exited zero; the production build passed 6/6 web tests. The command
  `npm audit --audit-level=critical` exited zero while still printing the two documented
  high-severity `vinext -> image-size` advisories. `pytest -q` passed 25/25; `compileall`, `pip check`,
  `pip-audit -r ml/requirements-serving.lock` and the five-entry artifact check all passed
  (no known Python vulnerabilities). CI and Dependabot YAML parsed successfully. A real
  uvicorn runtime reported `ready` on `mps` with CF-ViT as the only default verdict arm;
  `POST /predict` accepted a 1280x800 PNG and returned HTTP 200 (`research p_ai=0.7923`,
  official decision `insufficient`), while the production web server rendered PixelProof
  and the exact 11/103 limitation. The final regenerated `dist/` contains neither a stale
  starter screen nor `/Users/` / `file://` paths. All H1-H6 phases were roadmap-updated and
  committed separately before this final phase commit.

### Non-negotiable project rules for H0-H6

- Labels remain `1 = AI-generated`, `0 = real`.
- User-facing decisions remain asymmetric: `AI detected` or `insufficient evidence`; never
  an authenticity certificate.
- Threshold selection sees calibration data only. Evaluation halves never influence a
  threshold, model choice or post-hoc gate.
- Headline quality remains AI recall at a fixed false-positive budget with macro and
  worst-source FP; AUC is supporting evidence, not the deployment decision.
- External model licences and revisions are enforced in the runtime path, not left only in
  prose. B-Free stays explicit opt-in.

### CI portability correction — queued 2026-08-27

- [x] Reproduce the first GitHub `main` run failure from run `33070433088`: the web job passes,
      while Python collection fails because the workflow omits the `experiments` import root and
      installs only serving dependencies although the full suite imports `pyarrow`.
- [x] Make pytest's repository-owned configuration expose both `src` and the `ml` project root;
      install the declared `experiments` and `test` optional groups in CI instead of maintaining a
      second incomplete hand-written test environment.
- [x] Run all 207 Python tests without a caller-supplied `PYTHONPATH`, plus compileall, `pip check`,
      web build/tests/typecheck/lint and artifact verification. Append the result to HISTORY, commit,
      push `main` and wait for the replacement GitHub CI run to finish green.
- [x] After CI is green, protect `main` against force-push/deletion and require the `web` and
      `python` checks before future merges; do not enable a rule that blocks the current owner from
      administering the repository.
- Module 2 remains parked until a localisation model is measured against pixel masks on the
  relevant manipulation family.

## Historical checkpoint (2026-08-18, after E20-v2 / E21 protocol work)

This section is retained as research history and is superseded by the 2026-08-24 active
hardening roadmap and current contract above. It does not describe today's served system.

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

## Historical research queue (completed or superseded)

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

## Execution checklist — decision-layer hardening (queued 2026-08-19, start 08-20)

Every item was pre-flight-checked on 2026-08-19: cached score files present (e20 raw ×3,
e21 ×2, e22 results), NIST2016/auth holds 250 originals, B-Free checkout + MD5-verified
weights under `ml/external/`, CF-ViT checkpoint in the HF cache, `e20 --seeds 3 --arms
resnet18` confirmed in the CLI, 48 GB free disk. No item should hit a missing dependency.

- [x] **E23a — Midjourney diagnostic** *(done 2026-08-19, see E23a)*. Not a subgroup —
      Midjourney's whole distribution sits near the reals in B-Free's space (its training
      is SD-family reconstructions). And the "real" verdict was never consistent: NIST2016
      gets 0% "real" coverage at every miss budget. **Decision: asymmetric band** — verdicts
      are "AI" / "insufficient evidence" only; wrongly-real drops from 13.6% to 0% at zero
      cost to recall or FP.
- [x] **E23b — megapixel policy** *(done 2026-08-19, see E23b)*. The cap rescues the
      last failing pipeline: NIST2016 under a truly-unseen LOSO threshold drops 35.2% →
      **8.8% FP — under budget.** Policy adopted for the B-Free arm (no-op for CF, whose
      preprocessing already shrinks). **The B-Free band now passes the gate on all eleven
      pipelines at ~65% recall — the project's best deployable configuration.**
- [x] **E23c — the compression column** *(done 2026-08-19, see E23c)*. Compression is a threshold domain: CF fails safe under q50 degradation, B-Free fails dangerous on megapixel reals (41% FP frozen) and refit restores the budget at 42.8% recall. Serving contract gains compression-regime routing.
      q50 + 75%-resize copies of the 3,056 scored images, rescore both external arms,
      repeat E22's LOSO + band on the degraded column. The question: does the band
      survive internet conditions? (The E12 debt, now entangled with the decision layer.)
- [x] **E22 bootstrap CIs + E20 three-seed run** *(done 2026-08-19/20, see E22b + the E20
      addendum)*. Intervals attached to every headline number; three seeds confirm our
      model within noise (AUC 0.751 ± 0.033, worst-source FP 86.2% ± 3.1 — the
      cross-source failure is not a seed artifact).
- [x] **E24 — the library promise** *(done 2026-08-20, see E24)*. 207 iPhone camera
      originals as the 12th pipeline: CF passes frozen at 1.0% FP; uncapped B-Free would
      accuse 38.2% (E23b reproduced on real user data); cap + ~100-photo threshold-only
      refit lands the untouched half at **9.7% — budget met** at 62.2% recall. The
      deployment recipe (audit → cap → calibrate → refit) is now measured twice.
- [x] **Demo integration** *(E26 shipped 2026-08-20; E27 was temporarily integrated and
      then removed by the 2026-08-24 protocol correction after its calibration-only recall
      fell to 14.5%, failing G1)*. `pixelproof/verdict.py` serves the
      asymmetric band with every measured policy: 2048px cap (E23b), "AI / insufficient
      evidence" verdicts only (E23a), compression-regime caveat (E23c), E24's
      12-pipeline thresholds with experiment provenance in every response. CF-ViT (MIT)
      always on; B-Free loads only behind `PIXELPROOF_BFREE=1` (nonprofit licence).
      Dead stats2/3 options removed from API and UI; verified end-to-end in the browser.
      **E26 contract:** verdict rule is OR over verified arms (a blind primary cannot veto a
      seeing one — corrected 12-source evaluation worst 10.7%, FLUX 64.5%, the missed
      ChatGPT upload caught);
      the UI shows exactly one verdict, with the research signal demoted and labelled.

## Standing rules (unchanged)

- Headline metric = AI recall at a fixed FP budget on **unseen real sources**; AUC is
  reported alongside, never alone.
- ≥3 seeds on anything reported. Audit every dataset before use (`ml/tools/audit_datasets.py`).
- Thresholds are chosen on calibration halves and measured on untouched halves — always.
- CI reliability repair (2026-09-04): keep `npm audit --audit-level=critical` blocking,
  but retry its registry endpoint up to three times so a transient npm 5xx does not masquerade
  as a project/test failure. A repeated failure or real critical advisory still fails CI.
- E51 acquisition scripts import the official Kaggle client; keep `kaggle>=2.2,<2.3` in the
  declared experiment dependency group so clean CI hosts collect the same tests as the workstation.

## Repo conventions after the 2026-08-18 tidy-up

- `ml/experiments/` holds the runnable E20–E27 protocol scripts; finished earlier evidence
  scripts are frozen in `ml/experiments/archive/`.
- `ml/src/pixelproof/archive/` holds retired modules (E2–E4 analysis, ELA, DINOv2
  extraction). Nothing in the live path imports them.
- `ml/artifacts/archive/` (not committed) holds superseded artifacts, including the
  poisoned `*.BOZUK_etiket.bak` evidence files. Nothing is deleted.


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


### E65 current acquisition

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


### E65 completed action — frozen diagnostic (2026-09-13)

Download, score-blind audit and166-view frozen diagnostic are complete; see current checkpoint
at the top and `evidence/e65_diagnostic.md`. All697 Python tests passed. Commit/push code,
compact evidence and MD updates; raw data/derived images/features stay outside git.


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

E71 execution update: inventory/feature contract frozen after all9,599 chunks and their
source crop bodies verified. Active step is30-source historical encoder parity, then
completion of2,031 missing TRAIN parents. Fit code is prepared; freeze/run only after
complete verified archive. Conditional DEV code binds previous E70 exposure and requires
historical reference pixel/score/decision parity; no automatic DEV after failed TRAIN.

E71 source parity passed30/30 with exact raw equality. Continue missing-parent extraction;
feature completion receipt still required before fitting. The full suite reached749passed;
conditional DEV parity/denial tests added next. No heartbeat or scheduled checking task.

E71 prepared-code verification complete:752 Python tests passed, compile/diff clean.
Commit/push this preregistration checkpoint while the verified feature job continues.

During E71 feature completion, run one separately hashed E70 post-fit margin diagnostic:
fixed TRAIN margins plus existing consumed DEV scores, no new candidate or image inference.
Use preset quantiles and boundary distances; no tuning from the result. QuAD is already an
E46 prior, not a new candidate. Inspect SIDL as a research-only iPhone12Pro optical-degradation
lead; no download/admission until full-res provenance, access and group split are verified.

E70 fixed-margin diagnosis completed:109 condition views at the near-active1e-7 protection
margin; existing DEV losses include substantial-margin AI. Confidence/margin preservation
is a justified future mechanism to examine after E71's fixed result, not a threshold fix or
permission to relax REAL gates. Record any new objective/constraint as a separate experiment.

Margin diagnostic checkpoint:754 tests passed; commit/push completed evidence and continue
E71 extraction. SIDL next read is metadata-only for exact grouping/provenance inventory.

SIDL metadata inspection complete, image acquisition remains pending release reconciliation:
1605 RAW records/253 scene ids differ from the website300 scenes/1588 pairs. Do not infer
full coverage or roles; identify full archive/split mapping first. No image downloaded.

Research queue update: PiD pixelwise quantization residuals and MPFT texture-masked CLIP
fine-tuning are distinct future representations, not implemented E71 variants. QuAD is an
E46 prior. MIDD's official share is accessible with20 ZIPs/331.3GB; inspect dataset-specific
terms, original/patch packaging and scene identities before choosing a bounded sensor subset.
Do not download the whole release by default. Active E71 recipe and gates remain unchanged.

Next data preparation, separate from E71: inspect four sensor packages across vendors
(Hynix_SL846, ISOCELL_3P9, Sony_IMX258, OmniVision_OV32A), chosen for vendor coverage and
smallest visible package per vendor, not detector scores. Verify per-package license, original
TRAIN inventory and exact archive identity. Then preregister a bounded hash-ranked reserve
for possible research-only TRAIN expansion; whole MIDD publisher excluded from fresh final.
No denoised partners or official test members for training. No image acquisition until this
metadata check and a separate bounded acquisition contract; E71 population stays unchanged.


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

Research follow-up after the fixed E76 result: a REAL-only feature reconstruction branch
could learn a nonlinear manifold from existing CLIP features; exact architecture/training
must be separately registered. DEAR is a native-resolution forensic-expert alternative;
inspect its released weight terms and resource requirements before any acquisition/scoring.
MAFL/ACEF are heavier training approaches, not automatic next runs. No new fit selected yet.


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


### E80 prepared, waits for complete E79 (2026-09-13)

One385-coefficient E77+DEAR64 correction, zero start; same objective/AI constraints.
Add all10 fixed numeric metric guards on old and expanded TRAIN in every condition
to existing E76 gates. E77 diagnostic showed a pooled Q75 pass can hide40%+ RR error
and sub95% covered accuracy. No DEV until all TRAIN guards; no E49 until separate
consumed DEV passes. E79 extraction remains active; no partial-feature fit.


E80 runtime addition before fit freeze: provisional TRAIN pass must replay all views in
batch8 with zero changes at both cuts, max score error<=1e-6 and all-AI shift>=-1e-8.
E77 TRAIN visual diagnostic found a non-AI view only7.31e-9 below the AI cut; this is
a numerical reproducibility guard, not a threshold adjustment or new label decision.


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


Future lead: LoRC patch-token geometry (August2026) needs DINOv3/LoRA and a clarified
release recipe; inspect code/weight terms and local cost before any download/execution.
Current CLS caches are insufficient. Post-hoc calibration paper requires target-domain
assumptions and offers no unchanged-AI guarantee; no test calibration planned. Primary
links and exact limitations are in IMAGE_FORENSICS_REFERENCE.md. E79/E80 unchanged.


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


### 2026-09-13 — Remote CI success verified

Commit bca8a1e0283c41e562228fa83be9b32547ab7f14 passes GitHub Actions CI34749560334:
https://github.com/EfeHanKeles346/ai-image-detector/actions/runs/34749560334
Web28s: clean npm ci, lint, typecheck, build/tests and npm audit all pass.
Python2m54s: installs, pytest, compileall, pip check and serving-lock pip-audit all pass.
Machine-readable run/job/step evidence is recorded in
evidence/ci_dependency_repair_2026-09-13.json; Turkish report and current README/PLAN
now distinguish verified CI from the unmet ML target. This final checkpoint changes
only documentation/evidence; no frozen recipe, candidate, threshold or dataset changed.


### E99 MIDD coverage acquisition registration — 2026-09-14, before pixels

Preselect four previously unused sensor packages from the official MIDD listing:
ISOCELL_GN1, ISOCELL_HM3, OmniVision_OV64B and Sony_IMX766. These extend camera-pipeline
coverage beyond the four E72 sensors; they do not make a new independent publisher.
Inspect central directories and source terms first. Select64 publisher TRAIN/original
JPEGs per sensor by SHA256(E99|filename), no score/content-based choice or refill.
Exclude denoised partners and all upstream test rows. Cap selected bodies at4GiB,
32MiB per file,64MiB aggregate directory/header allowance,30GiB free disk reserve,
AC power and two workers. Require exact206/ETag ranges, ZIP length/CRC and stored SHA256;
no whole-archive fallback. Preserve every completed file/receipt for verified resume.
Initial role is QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN, never automatic TRAIN/CAL/final.
Whole MIDD collection retains prior research-TRAIN-only designation. Full native decode,
protected reference/gallery/DEV overlap and connected scene-group checks are a separate
registered admission step; no model may score selected images before that check.


### E100 admission protocol — registered while E99 transfers, before image audit

Require complete256-file E99 receipt and source-code/hash validation before decoding.
Reuse fixed E65 canonical RGB/dHash/pHash fingerprint convention, native image decode,
100MP cap and224px floor. Preserve original JPEG bytes. Compare against the exact E88
protected reference closure plus128 admitted SID originals. All206 unique owner-gallery
body hashes must be present in that closure; all existing reserves/consumed E66 remain
protected. No protected pixels or classifier scores are read. Reject protected byte/RGB
matches or dHash<=4 AND pHash63<=4 matches; internal perceptual and same-sensor/capture-
second links form transitive groups. Reuse frozen E72 resolve rule/seed, propagate rejection
and keep one deterministic representative per surviving component, no refill. Mark scene
independence unverified. Pinned Pillow/NumPy runtime, one CPU decode worker, AC and30GiB
reserve. Downloaded256 is never treated as256 independent scenes or admitted examples.
Output separate E100 audit and TRAIN manifest only if checks complete; no E92 training,
CAL reassignment, AI-retention claim or demo promotion in the admission operation.


### E102 correction protocol — registered before new TRAIN scores or fit

One data-coverage correction on the frozen E92 representation, not a new confidence
calibrator. Append the complete256-parent E100 cohort after its E101 cache; retain all
12,269 old TRAIN parents and4 conditions (50,100 views;18,380 AI views). No representation
refit or source/score filtering. Initialize450 delta weights to zero; baseline logits are
E92 logits. Reuse E81 worst-REAL-source epigraph loss and all-AI nonnegative logit-shift
constraints, preserving every reference-correct REAL binary decision. SameL2=.01,
SLSQP200 iterations/ftol1e-9, no hyperparameter sweep; CPU2 threads,1-hour fit budget.

Replay all frozen E92 TRAIN numeric reports before optimization. Preserve every previous
source/population and numeric gate; additionally require new-sensor pooled REAL FPR<=10%,
worst new sensor<=20%, combined REAL<=10% in each condition, all10 numeric gates on the
expanded population, solver success/constraint error<=1e-8, no lost E92-caught AI/new REAL
errors and full batch8 runtime replay<=1e-6/zero cut crossings. If training fails, lock
the failed result and do not score DEV/gallery. If it passes, separately register one
consumed E66/gallery paired comparison with E92 and E43 retention; no automatic promotion.
No existing CAL, DEV, gallery, E49 or final data in fitting. No calibrated probabilities.

### Execution checkpoint — E106/E107/E108 (2026-09-14)

- P0 implemented: exact finite-sample planner and consumed/dependent-evidence rejection
  (`e108_evidence_limits.py`). Current observations do not support universal proof.
- P2 complete: all34,890 legacy raw CLIP views verified;15,210 further raw views remain
  before full-population context/texture representation fitting. Do not drop these cohorts.
- P3 complete and failed as a model: all120 historically exposed CocoGlide pairs evaluated
  without oracle filtering/thresholds. Crop128 pixel AUC0.572 versus center0.722; authentic
  flagged area27.4%. Keep it as a failed baseline, not a deployable Model2 mask head.
- Next Model2 steps: finish E105 acquisition, audit native mask IDs including empty masks
  (E109), establish authentic-parent/generator ancestry, then register spatial-feature
  training and grouped calibration. A 0-mask downloaded edit is not automatically REAL.
- No changes to serving E92, existing cutoffs, protected final sets or gallery roles.

### Acquisition and representation checkpoint — E109/E110B

P1 completed512 DiffSeg30k image/mask pairs; E109 decoded all and found201 partial,
224 full-positive and87 empty masks. Retain quarantine and task strata. P4 now includes
E111 full Model2 body/perceptual comparison to existing role fingerprints before any
training roles are assigned.

P5 feasibility passed: E110B reproduced all70 selected missing-view CLIP aggregates
exactly across40 source/cohort representatives. The future full15210-view extraction
is approximately2.18hours by this small pilot, with observed2.04GiB MPS allocation.
Use verified E84 source materialization for legacy source_key records. Full extraction
must retain every TRAIN cohort/condition and durable bound chunks; then a separately
registered paired context-feature fit is required. No new classifier was fitted here,
no model improvement inferred from feature parity, and E92 remains served.

### E111 outcome and next gating work

P4 perceptual audit completed1,536 images against152,380 reference records:45 CocoGlide
parent groups have conservative near-match flags (14 exposed120,31 remaining392; no
exact body/RGB matches). Quarantine whole flagged groups. No matching DiffSeg derivative
is not proof of independent original ancestry. Original COCO mapping and mask-scope
interpretation remain unresolved, so Model2 training/CAL assignment remains pending.

Next execution order: (1) register/run the full15,210 missing Model1 raw-feature views
using the passing E110B operations and durable chunks; (2) paired context-feature fit
with all old AI/REAL retention gates; (3) recover author original-parent mappings for
Model2 and resolve near-match groups before supervised localization training; (4) only
qualified candidates can spend new source/generator-held-out evidence. None of these
remaining stages is marked completed by the audits in this session.

## E112 implementation — full Model1 context features (2026-09-14)

The user now explicitly requests execution. Register the full12,525-parent population:
all15,210 missing raw CLIP views, preserving the34,890 old raw views, yielding50,100
ordered TRAIN views. Use the passing E110B source resolution, transforms, batch3 encoder
and1e-5 aggregate-parity ceiling. Verify every parent on creation and resume; durable
per-parent chunks prevent loss of completed work. Fixed4-hour execution limit,6GiB MPS
ceiling, AC power and20GiB disk reserve. No data download, classifier scores or protected
image reads. Stop only the known E92 API while GPU extraction runs, then restore it.

After complete cache admission, fit a paired128-coordinate PCA extension: ordered
center/local-difference versus the same-dimensional pooled mean/std control, both on
all50,100 TRAIN views. Keep the frozen E103 base, identical optimizer/data/constraints,
and every AI-logit and correct-REAL retention guard. No subset fit or DEV-driven sweep.
Register that fit separately before execution; a failed TRAIN gate denies DEV scoring.

### E113 paired-fit implementation and unattended handoff

Implemented the fixed two-branch comparison after E112 completion:128 whitened PCA
coordinates from either pooled mean/std or ordered center/local-difference vectors,
appended to the frozen E103450-dimensional map. Both use the full50,100 TRAIN views,
seed113/power3, identical578 zero delta weights, joint worst-group BCE and the E103
all-AI/correct-REAL constraints. Preserve every previous numerical/population gate,
new-sensor gates, saved artifact parity and complete batch8 inference parity. A passing
TRAIN branch only permits separately registered consumed DEV; no automatic promotion.

A single local `run_e112_e113.py` process executes E112, registers E113 only after the
complete verified cache, and fits both branches. It writes English stage/results entries
to HISTORY.md and ml/EXPERIMENTS.md; detailed logs/chunks remain outside Git. It stops
on any failed command, retains evidence and restores the same E92 API in its finalizer.
`caffeinate -is` holds the machine awake during this one run. This is not a recurring
monitor or30-minute automation. Do not start a second encoder/fit on the same machine.
Model2 ancestry quarantine and protected final/gallery roles remain unchanged.

## E114 implementation while E112 finishes — 2026-09-14

Prepare one consumed-DEV comparison for every E113 branch whose complete TRAIN/runtime
guards pass. Registration must reject both-failed or incomplete paired reports before
reading DEV metadata. Re-encode only the640 ordered CLIP crop vectors after the GPU is
free, verify all reconstructed aggregates against E83, and reuse its frozen DINO/DEAR
features. Apply frozen E113 PCA/heads, lock all branch scores before metrics, and require
all20 numerical gates plus individual E43/E92/E103 AI retention and E92/E103 correct-REAL
retention. Equal pooled counts cannot hide swapped mistakes. No final/gallery access or
training on DEV. The implementation is ready; registration/execution awaits E113 results.

Model2 provenance follow-up: inspect the author-linked CocoGlide ZIP's directory and
small text metadata for original COCO identity mappings (E115). No image members are
needed for this inventory. Keep current overlap/ancestry quarantine regardless of a
successful metadata download until actual parent mappings are established.

### Model2 revised next step after E116/E117

Original lineage recovered for all512 CocoGlide triples. **506 share protected DDA-COCO
parents**, including all120 exposed examples. Exclude those full ancestry groups from
new training/calibration; the6 nonmatches are insufficient and not broadly certified.
Do not proceed with a convenient random CocoGlide split. E111 perceptual disjointness
alone was insufficient; make explicit original/prompt identities an admission requirement.

Next acquisition/design options must expose original identities: an author corpus with
eligible non-overlapping original parents, or a separately registered local inpainting
pilot using already admitted TRAIN-only originals, preserving their roles and all mask/
generator/transport descendants in one group. Any local generation requires its own
pinned generator, model licence, compute probe, deterministic masks and paired controls;
no paid generation or test-parent reuse is implied. Source-held-out evaluation must come
from distinct, unconsumed parents/generator families. DiffSeg stays quarantined until
its original/base-image mapping is established.

E114 one-shot dependent process is now waiting for the running E112/E113 process to
finish. It registers DEV only after complete passing TRAIN guards; both-failed reports
produce an explicit skip. After registration, it temporarily stops only the verified
local E92 API, runs the paired640-view consumed screen, records all outcomes and restores
E92. No recurring scheduler, new candidate promotion or independent-final access.

## Model2 traceable inpainting pilot — E118/E119 design

After E117 excludes506 protected CocoGlide ancestries, use existing admitted MIDD TRAIN originals for a small local engineering pilot. E118 acquires the pinned public SD1.5-inpainting mirror (revision8a4288a76071f7280aedbdb3253bdb9e9d5d84bb), four fp16 safetensors and required configuration/tokenizer files, retaining its safety checker. It does not modify the active Model1 environment or use the GPU. Register E119 separately before generation:16 score-blind native originals, two per eight admitted sensors; exact parent/scene/source/licence lineage; deterministic off-center masks; authentic and traditional-edit controls; matched geometry/encoding; record raw generator outputs and context-preserving composites separately. Never relabel globally VAE-changed background as untouched. All descendants remain TRAIN research material, not CAL/DEV/final. Freeze parameters before pixels and keep failed generations without score-selected replacements. Run the GPU probe only after the active Model1 pipeline finishes.

The generator is an old community mirror, not an official current Runway release or evidence of modern-editor coverage. The [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) documents512px masked inpainting and lossy autoencoding. [Diffusers MPS guidance](https://huggingface.co/docs/diffusers/optimization/mps) motivates batch1 and attention slicing on this Mac. Pin software separately; do not upgrade the running detector environment. A successful preparation/probe requires a later spatial-head experiment and separately sourced, parent/generator-held-out validation.

### E118/E119 execution checkpoint

E118 assets completed with exact publisher identities. E119 contract `80721e3928d94f618ce4525e50f7f3b51de04bdd15ec31f0c36217120ea85e80` and16-parent inputs are frozen. Register fixed30-step/batch1/MPS-fp16 generation, original scheduler, guidance7.5, strength1, fixed generic prompt and per-parent seeds119000–119015. Preserve safety checker, raw output, composited output and mask/control provenance. Bound one run to2400seconds and8GiB MPS. No repeated generation after failures without review.

The single dependent `run_e119_after_model1.py` process waits for both current Model1 workflows to finish and restore E92. A scientific both-TRAIN-failed skip permits the independent engineering pilot; an execution failure or API-restoration failure stops it. It temporarily stops only a verified E92 process and restores it in the finalizer. This is a finite dependent job, not a recurring scheduler. A passing engineering probe still needs registered spatial learning and independent source/editor evaluation; it does not make Model2 deployable.

### E113 automatic execution outcome

Both fixed TRAIN branches completed. {"ordered_context": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}, "pooled_control": {"AI_caught_by_condition": {"assigned_transport": 4595, "clean": 4595, "q75": 4595, "social_q75": 4595}, "dev_scoring_permitted": true, "false_REAL_alerts_by_condition": {"assigned_transport": 0, "clean": 0, "q75": 0, "social_q75": 0}, "solver_success": true}}

Next: separately register consumed DEV only for branches whose TRAIN guards passed. If neither passed, diagnose TRAIN failure before another preregistered experiment. No model promotion or independent evidence has been obtained. Detailed results: evidence/e113_context_fit.json.

### E114 automatic consumed-DEV result

{"ordered_context": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}, "pooled_control": {"by_condition": {"publisher_original": {"AI_caught": 159, "REAL_false_alerts": 0, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": false, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}, "social_q75": {"AI_caught": 159, "REAL_false_alerts": 12, "checks": {"twenty_numeric_gates_condition_passed": true, "zero_lost_E103_AI": true, "zero_lost_E43_AI": true, "zero_lost_E92_AI": true, "zero_new_E103_REAL_errors": true, "zero_new_E92_REAL_errors": true}}}, "passes_consumed_DEV_screen": false}}

Full report: evidence/e114_context_development.json. This is consumed development; no independent proof or serving change. Only passing branches may proceed to a separately registered gallery regression. Failed branches require a new hypothesis, not relaxed gates.

### E119 automatic engineering pilot result

{"contract_sha256": "80721e3928d94f618ce4525e50f7f3b51de04bdd15ec31f0c36217120ea85e80", "detector_scores": 0, "limits": "16 already-used TRAIN parents, one old editor, correlated sensor scenes and fixed prompt. This is an engineering pilot. No calibration, independent evaluation, universal claim or automatic training/serving admission.", "mean_raw_background_changed_fraction": 0.9998254416167689, "parents": 16, "passed": false, "passed_parents": 1, "peak_mps_bytes": 4143857664, "promotion_allowed": false, "result_sha256": "bba363a9c710dd094c09e9717a648b4a5b3024acd7d4c3002a482e227ef61abf", "seconds": 273.0622565409867, "state": "E119_inpainting_engineering_pilot_complete", "training_admission": false}

This measures generation feasibility only. Spatial-head training requires a new protocol; no independent detector evidence or serving change.

## Completed E112/E113/E114 and E119 decisions — 2026-09-14

E112 completed12525 parents /50100 TRAIN views with exact old aggregate parity. Both E113 heads pass complete TRAIN/runtime guards:0 REAL false alerts and4595/4595 AI caught in all four conditions. E114 rejects BOTH on consumed DEV: each has0/160 REAL false alerts and159/160 AI caught in originals;12/160 REAL false alerts and159/160 AI caught after social Q75. All20 numeric gates pass, but the inherited newly missed E43-caught original AI remains. Added context does not fix this failure. No gallery screen, independent final access or serving promotion. Retain E92.

E119 produced only1/16 eligible pairs;15 outputs are black and logs show nonfinite-to-uint8 warnings before safety decisions. Do not admit these to training or interpret the safety output as evidence of image content. E120 registers first original pass/failure parents0/1 under three fixed numerical branches: fp16+sliced reproduction, fp16+SDPA, float32+sliced. Preserve source weights, masks, prompts, seeds and safety checker; instrument VAE/text/UNet/latents before image conversion. Prefer fp16+SDPA for a later complete16-parent replay only if both fixed cases pass; otherwise require float32+sliced to pass. No reroll, threshold relaxation or final-data use.

The [upstream SDXL/MPS issue](https://github.com/huggingface/diffusers/issues/14438) reports slicing/offload NaNs, while [official MPS guidance](https://huggingface.co/docs/diffusers/optimization/mps) recommends slicing for memory. Our SD1.5 run has no offload, so that report is a diagnostic lead rather than an established cause. Probe actual numerical boundaries before deciding the correction.

Model1 next: diagnose the paired full-TRAIN versus consumed-DEV generalization gap from frozen outputs, then register a representation/data hypothesis; never select an exception or cutoff for the known single missed image. Existing center-context vectors still do not represent an uncropped full frame.

E120 completed the fixed precision comparison: fp16+sliced reproduces a UNet nonfinite failure on parent1 before rendering/safety; fp16+SDPA and float32+sliced both pass parents0/1. Follow the prespecified preference: E122 registers complete16-parent fp16+SDPA replay, unchanged weights/images/masks/seeds/prompt and all acceptance checks. Keep finite boundary guards and safety checker; no downloads or detector fitting. E121 initially stopped on4278 lowercase legacy train roles; preserved frozen code/contract and ran E121B with the existing upstream case-insensitive TRAIN rule. E121B reproduces all DEV guards: neither branch changes any of640 AI-cut decisions; each changes two REAL-cut decisions. All12525 TRAIN dimensions are present. The center window spans~28.04–76.56% of source area; this describes only the center branch, not union-of-crops coverage or proof of the missed example cause.

## E123 — full available-frame feature feasibility

Register one hash-selected TRAIN parent per40 class/source groups and all four transport conditions (160views). Compare the unchanged historical center vector to a separately encoded uncropped available frame, warped to224x224 and JPEG90/4:4:4 encoded identically to the center branch. Preserve old center pixel/vector parity and duplicate full-frame vector parity before retaining features; no classifier fit. Warping avoids explicit padding bars but may create aspect-ratio distortion and source bias. The old center window covers28.04–76.56% of source area in the complete TRAIN metadata; texture crops can cover additional regions, so this does not establish total missing-information coverage or the cause of the specific failed DEV image.

Only a passing cost/parity probe can authorize a new full-population feature extraction contract, followed by equal-capacity full-frame versus center-only PCA/head controls with complete TRAIN retention and separately registered consumed DEV. No new threshold, manual score adjustment, gallery fitting or new data download.
