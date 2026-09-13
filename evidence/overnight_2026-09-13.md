# 13 September overnight research checkpoint

The target remains unmet. E43 and serving are unchanged. E70/E71 passed TRAIN but failed
separate consumed E66 development screens. E73/E74/E76/E77 failed TRAIN. No overnight
candidate has opened E49 or qualified for promotion. E79 completed frozen DEAR-r
features for all12,141 admitted TRAIN parents; E80 failed source/selective TRAIN guards; E81 also failed source/selective TRAIN guards. E83 passed TRAIN/runtime but failed consumed E66 DEV: REAL FPR0%/13.75%, AI98.75%/99.375%, one new original AI miss. E84B passed parity/resource checks; full TRAIN transport feature extraction is active.

[E83 sonuç grafiği](e83_progress_2026-09-13.png) · [E83 vektör sürümü](e83_progress_2026-09-13.svg)

[Önceki E77 sonuç grafiği](overnight_progress_2026-09-13.png) · [Vektör sürümü](overnight_progress_2026-09-13.svg)

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
| E80 old REAL | 8.80% | 9.79% | 8.61% | Pooled pass; full TRAIN failed |
| E80 expanded REAL | 8.24% | 9.21% | 8.07% | Pooled pass; source/selective failed |
| E80 new MIDD | 0.59% | 1.17% | 0.59% | Passed slice/sensor guards |
| E81 old REAL | 8.54% | 8.54% | 8.33% | Pooled pass; full TRAIN failed |
| E81 expanded REAL | 8.00% | 8.07% | 7.82% | Pooled pass; source/selective failed |
| E81 new MIDD | 0.59% | 1.57% | 0.78% | Passed slice/sensor guards |

All four introduced zero new E43 AI misses/REAL errors. E76 isolates data addition to the exact
E74 map; new cameras were easier and did not solve old hard REAL. E77 adds REAL-only
CLIP reconstruction residuals: fixed100-epoch AE, L1 .79261→.54066, serialized features
exact. Reconstruction loss is not detector quality. No DEV/E49 after any failed fit.
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

## E80 result and next objective

E80 completed52 iterations/248.664s with zero new AI misses/REAL errors and exact saved
replay. Old-REAL FPR8.7989%/9.7939%/8.6141%; expanded8.2428%/9.2102%/8.0705%.
All population budgets pass. RR35.44%/40.48%/34.96% and expanded covered accuracy
94.4376%/93.7688%/94.5167% fail the source20% and accuracy95% targets.8/10 numeric
checks pass in each old/expanded condition; no runtime or DEV/E49 under the frozen gate.
[Full TRAIN result](e80_fit.json). This justifies the prepared E81 objective below.

## Frozen E80 model and verification

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
reward. Synthetic gradient and conflicting-class tests pass. E80 now confirms the source-failure trigger. The single TRAIN fit is now separately registered,
retaining every E80 acceptance gate and frozen map.

The optional E80 consumed-E66 implementation is prepared but not frozen/executed. It
refuses failed TRAIN, preserves prior E43 pixels/scores/two-cut decisions, and requires
all20 numeric DEV gates plus zero newly missed AI per source/condition. E66 remains
consumed and dependent, never fresh final. E49 remains closed until that separate pass.

815 Python tests pass (engineering evidence, not detection quality). The existing
Starlette/httpx deprecation remains. Completed CI34732736214 passed Python, web lint,
type checks and tests; the existing web dependency audit failed. Raw images/features/
weights stay external; code, compact evidence and MD checkpoints are committed/pushed.
No CI rule or30-minute monitor/heartbeat automation was created or changed.

[Current plan](../PLAN.md), [append-only experiments](../ml/EXPERIMENTS.md),
[model card](../MODEL_CARD.md). All existing AI-preservation and final targets remain.


## E81 measured follow-up

E81 reuses the exact E80 map and changes the REAL-risk objective. Solver succeeds in50
iterations/330.585s, zero new E43 AI misses/REAL errors. Old-REAL FPR8.5430%/8.5430%/8.3298%;
RR32.24%/33.52%/31.84%; expanded covered accuracy94.5844%/94.5498%/94.6857%. The same
two gates fail in all6 TRAIN cells. E80's assigned AI recall99.0424% becomes98.9336%;
E43 reference retention still passes. No runtime/DEV/E49. [E81 receipt](e81_fit.json).
Next hypothesis: supervised nonlinear features on the existing frozen multimodal map;
separate registration required, no old-candidate or threshold retuning.


## Registered E82 representation

One385→256→64→1 ReLU MLP learns binary TRAIN features from exact E80 coordinates.
Class/source/parent balance, baseline hard REAL2x, class mass.5; AdamW2e-4/decay1e-4,
100 epochs/batch256/seed82. No dropout or early stopping. Freeze64 latent features with
CPUfloat64 inference and all-TRAIN scaling; require saved/batch parity and complete roles.
815 Python tests pass. This is representation learning only; E83 would separately fit450
constrained coefficients with the unchanged E81 objective and all acceptance/runtime guards.
No DEV/final pixels or scores, data download, source-role change or quality claim.


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


### E84B metadata-only transport coverage (2026-09-13)

The hash-bound original TRAIN manifests contain12,141 parents, all with dimensions.
REAL:3,091 at most1080px long side,426 at1081–2048,4,029 above2048.
AI:3,549 /1,045 /1 respectively. The social1080 cap potentially affects59.04% of REAL
and22.76% of AI parents. Identical policies do not imply equal transformation exposure.
Declared REAL formats:7,034 JPEG,1 PNG,511 unknown (MIDD metadata); AI:786 JPEG,3,809 PNG.
Do not substitute filename extensions for unknown manifest fields. This class/format
association is a descriptive coverage limitation, not a causal explanation or score result.
For6,640 inputs at most1080, the new view is expected to duplicate old source-JPEG75
under unchanged helpers; no complete pixel-equality test was performed or claimed.

Tool `ml/tools/train_transport_coverage.py` executed once; receipt
`evidence/e84b_transport_coverage.json` pins both manifests and tool SHA. Zero image reads,
model scores, parameter choices or E49 reads. No change to active frozen E84B extraction;
E85/E86 remain prepared, not frozen or fitted. Latest GitHub242dfd7 Python checks pass;
web lint/typecheck/tests pass, existing vinext dependency audit remains failed.


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

E87 implementation verification: full Python suite837 passed in18.00s; one existing
Starlette/httpx deprecation warning. No experiment runtime dependency was changed.
