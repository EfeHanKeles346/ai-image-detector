# 13 September overnight research checkpoint

The target remains unmet. E43 and serving are unchanged. E70 and E71 passed TRAIN but
failed separate consumed E66 development comparisons. E73/E74 stopped at failed TRAIN
guards. No overnight candidate has opened an E49 regression or qualified for promotion.
E72 admitted511 new camera TRAIN observations; E75 completed their features, and E76
failed its one registered data-expansion fit.

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

## Data and feature work

- **E65 diagnostic only:**83 WIFD/RawNIND files,1.01GB,166 E43 views. WIFD REAL FPR16.42%/
  19.40%; RawNIND18.75%/31.25%. Dependent REAL-only diagnostics, no AI-retention result.
  Entire publishers stay out of TRAIN/fresh final. [Diagnostic](e65_diagnostic.md).
- **E66 consumed DEV:**SIDD Small6,615,978,508B matched published MD5/SHA1.160 NOISY images
  and160 admitted local AI formed the limited screen above. Its initial zero-score admission
  record is historical; all2,385 unused eligible AI candidates remain reserved from TRAIN.
- **E71 features complete:**11,630 x3 x1536 CLIP features,9,599 validated old chunks plus
  2,031 new chunks,1538.79s. Thirty historical raw-vector replays matched exactly. Old E59
  outputs/fits untouched; new code uses a separately frozen read-only cache contract.
- **E72 research TRAIN:**512 score-blind MIDD publisher-original JPEGs,128 each from four
  sensor packages,3,821,581,226B. Exact-range acquisition pinned archive ETags/member CRCs
  and SHA receipts; a connection timeout was recovered without changing the selection.
  All decoded;151,485 protected references yielded zero cross matches. One internal pair
  was deduplicated, admitting511 observations. EXIF identity/timestamps absent: sensor
  provenance is publisher metadata, scene independence unverified. Whole MIDD publisher
  is TRAIN only, including unselected/test/denoised members. CC BY-NC-SA4.0 research terms.
  [Admission](e72_audit.json), [download](e72_download.json).
- **E75 features complete (778.72s):**all511 admitted MIDD observations, frozen DINO3072/CLIP1536 in the
  same three TRAIN conditions. Old30-source replay passed with exactly zero E43-score and
  CLIP-feature error; no decision changes. New MIDD classifier scores remain unopened.
- **Leads only:**SIDL metadata has1605 RAW records/253 scene ids versus advertised300 scenes;
  release coverage remains unresolved, no images admitted. DEAR source/model metadata and
  distinct research weight terms were pinned, no weights/images scored. Other primary
  literature and limitations are in [forensic references](../IMAGE_FORENSICS_REFERENCE.md).

## Next fixed experiment and verification

E76 adds all511 audited MIDD observations to unchanged E54 TRAIN:12,141 parents,
7,546 REAL/4,595 AI. It keeps E74's exact bases, interactions and old-TRAIN scaling,
resets257 correction weights, and retains E73 full-AI-confidence constraints. No parameter
sweep. In each condition, old/expanded/new-MIDD REAL FPR must each be<=10%, with worst
MIDD sensor<=20%; solver, complete AI replay and serialization guards also apply.
E75 features are complete; E76 failed its single fit. No DEV after a failed TRAIN gate.

779 Python tests pass. These verify engineering behavior, not detection quality. The
existing Starlette/httpx deprecation remains. Code, compact evidence and relevant MD
checkpoints are committed/pushed; raw images/features/weights stay external. Latest checked
completed CI34730124772 passed Python and failed the existing web dependency audit.
No CI rules were changed; no30-minute monitor or heartbeat automation was created.

[Current plan](../PLAN.md), [append-only experiment history](../ml/EXPERIMENTS.md),
[model card](../MODEL_CARD.md). E49's target and all AI-preservation requirements remain.


E76 result: old REAL FPR13.28%/14.37%/12.25%, expanded12.64%/13.65%/11.62% fail;
new-MIDD3.91%/3.72%/2.94% passes with all sensor gates. No new AI/REAL errors;
11/28/19 additional TRAIN AI detections. Data addition alone did not solve old hard REAL.
No DEV/E49 allowed. Next E77 prepares a separately registered REAL-only CLIP feature
reconstruction residual branch, preserving all E76 population and AI-confidence guards.


E77 representation completed100 epochs in40.96s, weighted REAL L1 .79261→.54066.
No AI in AE loss; exact serialized feature replay. This is not detector quality.
Next: one321-coefficient full-AI-confidence fit with all E76 population gates.


E77 head FAILED: old REAL10.96%/12.38%/10.39%, expanded10.30%/11.65%/9.75%,
new MIDD1.17%/1.57%/.98%. Zero new AI/REAL errors, but all old-REAL10% gates fail.
No DEV/E49. Next E78: one pinned DEAR-r94.37MB research checkpoint acquisition;
synthetic parity/resource checks are separate from image scoring.
