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

Full verification after E88 preparation:839 Python tests pass in18.18s; only the
existing Starlette/httpx deprecation warning. E88 is still not frozen or executed.


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

Full verification after conditional E86 regression preparation:844 Python tests pass
in20.28s; one existing Starlette/httpx warning. No E86 regression stage executed.


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

Full verification after E89 preparation:847 Python tests pass in18.45s; one existing
Starlette/httpx warning. E89 remains unfrozen/unexecuted. External reserve347GiB;
macOS reports no recorded thermal/performance warning. Only E84B extraction is active.


### E90 diagnostic registered before scalar-head reads (2026-09-13)

Contract SHA256`dd075be24a426cbc0c782f70a51a7d9503138763395e9e2dd11f0c2213cf3d37` binds all640 consumed DEV views, exact E83
replay and frozen existing E82 scalar-head sign0. No diagnostic run yet. Initial focused
and full suites exposed a test fixture exporting float64 network weights, whereas the
frozen inference validator correctly requires saved float32 weights. Fixed only the test:
export original float32 state before converting the independent reference network to
float64. Diagnostic implementation/contract unchanged; two focused tests now pass.
The registration command had already completed while the first test result was pending;
no scalar-head data was read and no diagnostic execution started before the repair.
A complete post-repair suite must pass before execution. Existing E84B/E85/E86 unchanged.

E90 post-repair full verification:849 Python tests pass in17.80s with the existing
Starlette/httpx warning. Diagnostic may now run under its unchanged frozen contract.


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


### Read-only source/visual review of E83's sole newly missed AI (2026-09-13)

The complete set of newly missed E43 AI in E83 contains one original view, GPTIMG_431.png.
Source/E66 body SHA matches`32e1ccf2c9cb523c119f01afd3b7461b3db241a27d837be729b8d22e70a07d93`.
Existing E43 score0.08128665 is just above fixed AI cut0.07940196; E83 lowers it to
0.00507187. The source image is a close portrait with skin texture, sheen and haze/grain-
like detail. Appearance does not prove authenticity, edit type or a labeling mistake.
No relabeling, exclusion, TRAIN admission, model fit or new score was performed.

The [publisher card](https://huggingface.co/datasets/a3xrfgb/gpt-image-mega-4k) describes
curated GPT-IMAGE1 images captioned by Qwen3-VL. The neighboring.txt is descriptive
caption text; it must not be treated as a verified original generation prompt. No edit
mask, input photograph or generation log appears in the reviewed records. Full-generation
versus editing provenance remains unresolved. Keep the existing publisher-provided AI
label and unknown prompt-dependence limitation. Do not infer population-wide bias from
this single case. Receipt `evidence/e83_known_ai_miss_review.json` records both image and
caption hashes; no image or full caption is copied into git. Current E84B/E85/E86 unchanged.


Existing paired E83 scores for that same GPTIMG_431 parent show transport sensitivity:
original E43/E83=0.08128665/0.00507187, socialQ75=0.31710792/0.81932999. The Q75
version is already detected by both; the loss is on the original. These are previously
locked scores, not a new transform or inference. This supports examining transport
stability but does not prove that E84B augmentation will fix the original error.


### Read-only consistency-mechanism review during E84B extraction (2026-09-13)

Added a primary-source and mathematical review to `IMAGE_FORENSICS_REFERENCE.md` under
"Paired prediction consistency". A synthetic fixed-feature gradient check returned
exactly0 parameter-gradient difference; no real images or candidate were used. The
review separates an unverified implementation ambiguity from a possible future local
objective. E84B continues unchanged; E85/E86 remain the next fixed experiment. No new
loss, threshold, feature rank, seed or dataset mixture is selected from this review.
Any later consistency experiment must have its own complete preregistration and keep
all existing quality/retention/runtime guards. This is a research note, not a result.


### E86 aggregate visualization prepared (2026-09-13)

Prepared `ml/tools/plot_e86_checkpoint.py` to compare immutable E43/E83/E86 consumed
DEVELOPMENT aggregate rates and newly missed E43 AI counts. It requires complete
320-view condition reports and writes PNG/SVG plus hashes without image/model reads.
No figure has been rendered and no E86 result exists yet. Python compilation passes;
visual verification is deferred until the real E86 comparison is complete. Existing
E83 historical graphics are preserved. This reporting utility changes no experiment.


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
