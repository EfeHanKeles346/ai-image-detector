# 13 September overnight research checkpoint

The target remains unmet. E43 and serving are unchanged. E70/E71 passed TRAIN but failed
separate consumed E66 development screens. E73/E74/E76/E77 failed TRAIN. No overnight
candidate has opened E49 or qualified for promotion. E79 completed frozen DEAR-r
features for all12,141 admitted TRAIN parents; E80 is registered for its single fit.

[E77 sonrası sonuç grafiği](overnight_progress_2026-09-13.png) · [Vektör sürümü](overnight_progress_2026-09-13.svg)

## Comparable TRAIN results

These rows share7,035 REAL/4,595 AI parents, three conditions and the frozen E43 AI cut.
They are resubstitution results, not unseen-image accuracy. All candidates must retain
previously caught AI, introduce no errors on previously correct REAL, and achieve REAL
FPR<=10% in every TRAIN condition before a separate DEV registration.

| Candidate | Clean REAL false AI | Assigned transport | TRAIN Q75 | New AI misses | Outcome |
|---|---:|---:|---:|---:|---|
| E43 reference | 15.07% | 16.13% | 13.80% | — | Retained reference |
| E64 original PCA correction | 11.90% | 13.43% | 11.33% | 0 | TRAIN failed |
| E67 original + blur response | 10.25% | 11.33% | 9.48% | 0 | TRAIN failed |
| E68 source/condition minimax | 10.82% | 11.61% | 10.22% | 0 | TRAIN failed |
| E69 fixed patch shuffle | 10.52% | 11.77% | 10.14% | 0 | TRAIN failed |
| E70 original/blur bilinear map | 8.07% | 8.74% | 7.29% | 0 | TRAIN passed; DEV failed |
| E71 original/CLIP linear map | 6.99% | 7.76% | 6.82% | 0 | TRAIN passed; DEV failed |
| E73 E71 map + full AI confidence | 14.57% | 15.84% | 13.35% | 0 | TRAIN failed |
| E74 original/CLIP bilinear + confidence | 13.29% | 14.36% | 12.14% | 0 | TRAIN failed |

All candidates introduced zero errors on previously correct TRAIN REAL. E73/E74 additionally
preserved all13,785 AI-view reference logits, including initially missed AI, within1e-8
numerical tolerance. This substantially restricted REAL improvement in these fixed maps.
It does not prove global infeasibility or external AI retention.

TRAIN Q75 encodes JPEG75 at source resolution before the2048-pixel cap. DEV/E49 social
Q75 caps at1080 before encoding. These are distinct conditions.

## Separate consumed DEVELOPMENT results

Each condition contains160 REAL and160 AI observations. SIDD REAL comprises only ten
dependent scenes from five phones; AI comes from two previously seen generator families,
with unknown prompt dependencies. E70 consumed this set first; E71 is a subsequent consumed
comparison. Neither provides fresh or independent final evidence.

| Candidate/condition | REAL false AI | AI recall | Newly missed AI vs E43 | Numeric gates |
|---|---:|---:|---:|---:|
| E43 original | 42.50% | 97.50% | — | 6/10 |
| E70 original | 46.88% | 95.63% | 4 | 6/10 |
| E71 original | 34.38% | 97.50% | 2 | 6/10 |
| E43 social Q75 | 42.50% | 96.25% | — | 6/10 |
| E70 social Q75 | 30.63% | 95.00% | 3 | 6/10 |
| E71 social Q75 | 38.13% | 96.25% | 1 | 6/10 |

E71's equal aggregate recall conceals lost detections: two original GPT images were lost
while two others were rescued; Q75 lost one GPT while rescuing one Nano Banana. The
per-image/source retention guard correctly rejects this trade. Both candidates also fail
the absolute REAL targets. [E70 evidence](e70_development.json),
[E71 evidence](e71_development.json). E73/E74 have no DEV scores.

E70's [fixed margin diagnosis](e70_margin_diagnostic.json) found109 TRAIN condition views
near the decision-protection boundary. Several existing DEV losses had substantial old
margins, so numerical rounding alone cannot explain them. E73/E74 test stronger confidence
protection without changing thresholds; their failures are preserved.

## Expanded TRAIN experiments

E76/E77 train on the same12,141 parents (7,546 REAL/4,595 AI), including511 audited
MIDD camera observations. All13,785 AI-view reference logits are protected within1e-8;
no previously correct REAL may become wrong. Old/new/expanded REAL slices remain separate.

| Candidate and REAL population | Clean FPR | Assigned transport | TRAIN Q75 | Result |
|---|---:|---:|---:|---|
| E76 old REAL | 13.28% | 14.37% | 12.25% | Failed all10% guards |
| E76 expanded REAL | 12.64% | 13.65% | 11.62% | Failed all10% guards |
| E76 new MIDD | 3.91% | 3.72% | 2.94% | Passed slice/sensor guards |
| E77 old REAL | 10.96% | 12.38% | 10.39% | Failed all10% guards |
| E77 expanded REAL | 10.30% | 11.65% | 9.75% | Only pooled Q75 passed |
| E77 new MIDD | 1.17% | 1.57% | 0.98% | Passed slice/sensor guards |

Both introduced zero new AI misses/REAL errors. E76 isolates data addition to the exact
E74 map; new cameras were easier and did not solve old hard REAL. E77 adds REAL-only
CLIP reconstruction residuals: fixed100-epoch AE, L1 .79261→.54066, serialized features
exact. Reconstruction loss is not detector quality. No DEV/E49 after either failed fit.
[E76 result](e76_fit.json), [E77 result](e77_fit.json).

E77's [full TRAIN operating-point audit](e77_train_operating_point.json) finds expanded
covered accuracy92.05–93.30% below95%, and RR REAL FPR43.84%/48.16%/40.64% above20%.
Coverage92.3–93.2% and uncertainty6.8–7.7% pass. A pooled Q75 pass thus hides source and
covered-accuracy failures. This reused the generic metric function on TRAIN only; no E49
scores, images, bootstrap or final validator were read/run.

## Data and representation work

- **E65 diagnostic only:**83 WIFD/RawNIND files,1.01GB,166 E43 views. WIFD REAL FPR16.42%/
  19.40%; RawNIND18.75%/31.25%. Dependent REAL-only diagnostics, no AI-retention result.
  Entire publishers stay out of TRAIN/fresh final. [Diagnostic](e65_diagnostic.md).
- **E66 consumed DEV:**SIDD Small6,615,978,508B matched publisher MD5/SHA1.160 NOISY images
  and160 admitted local AI formed the limited screen. Its zero-score admission record is
  historical; all2,385 unused eligible AI candidates remain reserved from TRAIN.
- **E71 CLIP features:**11,630 x3 x1536;9,599 validated old chunks plus2,031 new chunks,
  1538.79s. Thirty historical raw-vector replays exact. Old E59 fits/outputs untouched.
- **E72 research TRAIN:**512 score-blind MIDD publisher-original JPEGs from four sensor
  packages,3,821,581,226B. Pinned range/ETag/CRC/SHA acquisition resumed after a timeout.
  All decoded;151,485 protected references yielded zero cross matches; one internal
  duplicate removed,511 admitted. No EXIF camera/time fields; scene independence unknown.
  Whole MIDD publisher is TRAIN only, including unselected/test/denoised members.
  CC BY-NC-SA4.0 research terms. [Admission](e72_audit.json), [download](e72_download.json).
- **E75 MIDD features:**all511 x3 DINO3072/CLIP1536,778.72s. Old30-source E43/CLIP replay
  exact. New MIDD classifier scores were unopened before the E76 fit, then consumed by
  E76/E77 TRAIN fits; do not mistake initial admission receipts for present freshness.
- **E77 REAL-only representation:**22,638 REAL views, zero AI in AE normalization/loss;
  fixed weighted L1 architecture1536→256→64→256→1536. All-TRAIN residual PCA64, one
 321-coefficient fit. [Representation](e77_representation.json).
- **E78 DEAR-r:**one94,372,114-byte pinned research checkpoint, verified SHA256. Strict
  weights-only complete state load; reviewed author CPU features exact, logits within
 5.96e-8. CPU/MPS differences<1.4e-6; batch3/batch9 MPS outputs exact. No local images
  were scored in these synthetic probes. Weight terms are CC BY-NC4.0 plus NOTICE
  restrictions; upstream AlignedForensics lacks explicit licensing. No weight redistribution
  or serving. [Acquisition](e78_acquisition.json), [batch probe](e78_batch_probe.json).
- **E79 complete:**frozen DEAR-r820 active channels on all9 E54/E75 crops per TRAIN parent;
  mean/std1640 per condition. No classifier head execution, no image download. Immutable
  chunks bind parent/source/contract/arrays and support exact-population resume. Both E78
  probes failed the initial2h throughput budget (~2h7m); E79 explicitly allocates2.5h under
  overnight authorization before image features, retaining numeric and6GiB memory limits.
  This is a resource revision, not a changed quality gate or retroactive E78 pass.
  First extraction attempt later hit9000s at12,061 completed parents;80 remained. The
  same-contract resume validated old chunks and created exactly80 missing parents in412.623s.
  Shape12141x3x1640;34 source replays exact. Total loops exceed9412.623s plus setup,
  so full extraction did not complete within2.5h. [Complete receipt](e79_features.json).
- **Leads only:**SIDL metadata has1605 RAW records/253 scene IDs versus advertised300
  scenes; release coverage unresolved, no images admitted. Other primary sources and
  limitations: [forensic references](../IMAGE_FORENSICS_REFERENCE.md).

The frozen E77 slack audit also finds359/435/335 remaining REAL false-AI views whose
logits increased, while60/73/84 AI constraints are within1e-7 of their boundary. RR median
remaining error margins exceed2 logits. This is descriptive, not proof of constraint
causality or infeasibility; no candidate, label, threshold or DEV/final change followed.
See [slack receipt](e77_slack_diagnostic.json).

The RR source paper names mixed upstream news/COCO/CC3M/Unsplash sources, while our
legacy REAL names have no per-file site mapping. Exact-body audits cannot establish upstream
derivative independence; no specific new overlapping file was found. This limitation is
recorded in E80 before fit. No new COCO admission or unseen-COCO/fresh-final claim follows.

## Registered E80 model and verification

E80 adds all-TRAIN DEAR PCA64 (StandardScaler, seed80, randomized power3, whiten) to the
exact E77 original/CLIP/bilinear/REAL-residual320 coordinates, plus one standardized fixed
official DEAR-r head response on the mean crop features;386 zero-start coefficients. This
pre-fit code-review revision preserves information PCA can discard. The earlier385-coefficient
preparation was never frozen/fitted; no DEAR image classifier scores informed the change.
The mean-crop linear response is a local adaptation, not official native full-image inference.
Keep the E64 objective, E73 full-AI-confidence constraints and all E76 population guards.
Additionally require all10 numeric metric checks on old and expanded TRAIN in each of
three conditions. The strengthened screen addresses the E77 audit above. No parameter,
seed, rank or cut sweep; no fit from partial E79 features. A provisional TRAIN pass must
also replay every view at batch8 with zero changes at both cuts and unchanged AI confidence.
The six-image TRAIN visual diagnostic found a non-AI example only7.31e-9 below the AI cut;
this adds a reproducibility guard, with no label or threshold adjustment.

E81's conditional optimizer helper is prepared from the E77 slack diagnosis: minimize
worst REAL source/condition BCE while all AI logits remain non-decreasing and correct-REAL
decisions remain protected. Same REAL coefficient .5, L2.01 and SLSQP settings; no AI BCE
reward. Synthetic gradient and conflicting-class tests pass. No TRAIN fit/contract yet;
consider only after E80 results, retaining every E80 acceptance gate and frozen map.

The optional E80 consumed-E66 implementation is prepared but not frozen/executed. It
refuses failed TRAIN, preserves prior E43 pixels/scores/two-cut decisions, and requires
all20 numeric DEV gates plus zero newly missed AI per source/condition. E66 remains
consumed and dependent, never fresh final. E49 remains closed until that separate pass.

808 Python tests pass (engineering evidence, not detection quality). The existing
Starlette/httpx deprecation remains. Completed CI34732736214 passed Python, web lint,
type checks and tests; the existing web dependency audit failed. Raw images/features/
weights stay external; code, compact evidence and MD checkpoints are committed/pushed.
No CI rule or30-minute monitor/heartbeat automation was created or changed.

[Current plan](../PLAN.md), [append-only experiments](../ml/EXPERIMENTS.md),
[model card](../MODEL_CARD.md). All existing AI-preservation and final targets remain.
