# 13 September overnight research checkpoint

The target is still unmet. E43 and serving are unchanged. E70 passed TRAIN but failed its first E66 DEV comparison. No E49 regression was opened.

## Comparable TRAIN results

All rows below use the same admitted E54 population: 7,035 REAL and 4,595 AI parents,
three conditions, with the unchanged E43 AI cut. These are resubstitution diagnostics,
not unseen-data accuracy. Each candidate must pass all three REAL ceilings of 10%,
retain every previously caught AI and introduce no REAL error before DEV access.

| Candidate | Clean REAL false AI | Assigned transport REAL false AI | TRAIN q75 REAL false AI | Newly missed AI views | Decision |
|---|---:|---:|---:|---:|---|
| E43 reference | 15.07% | 16.13% | 13.80% | — | Retained reference |
| E64 original PCA correction | 11.90% | 13.43% | 11.33% | 0 | Failed all three ceilings |
| E67 original + blur response | 10.25% | 11.33% | 9.48% | 0 | Failed two ceilings |
| E68 source/condition minimax | 10.82% | 11.61% | 10.22% | 0 | Failed all three ceilings |
| E69 fixed patch shuffle | 10.52% | 11.77% | 10.14% | 0 | Failed all three ceilings |
| E70 bilinear original/blur map | 8.07% | 8.74% | 7.29% | 0 | TRAIN passed; DEV failed |

The five correction candidates introduced zero errors on previously correct TRAIN REAL
observations. Fixed-cut AI preservation on TRAIN does not establish external AI preservation.
TRAIN q75 applies JPEG75 at source resolution before a 2048-pixel cap; it is different from
the intended DEV/E49 1080-pixel/JPEG75 transport.

E67's remaining errors concentrate in the RR REAL pool: 67–71% of remaining errors come
from 1,250 of 7,035 REAL parents. E68 reduced worst-group BCE and rescued more RR images,
but rescued fewer images elsewhere; its pooled error increased versus E67. The minimax
objective change was therefore rejected. No source was removed or relabelled.

E69 tests one fixed 28-pixel patch permutation of the existing crops, intended to expose
complementary texture information. It retains 129 correction coefficients, the E67 objective,
all decision guards and the same acceptance ceilings. It is not a reproduction of SFLD's
CLIP ensemble and cannot inherit that paper's performance.

## Acquisition and development separation

- E65: 83 WIFD/RawNIND source files, 1.01 GB, audited and diagnostically scored in 166 views.
  WIFD false AI 16.42% original / 19.40% Q75; RawNIND 18.75% / 31.25%. These are dependent,
  small REAL-only diagnostics; there is no AI-retention result. Whole publishers stay reserved.
- E66: SIDD Small sRGB archive, 6,615,978,508 bytes, published MD5/SHA1 matched. All 160 NOISY
  observations decoded and audited against 151,165 references, with no protected/cross-scene/
  cross-label match. All 223 internal matches are within the known scene groups.
- E66 limited DEV: 160 SIDD observations from ten scenes/five older cameras plus 160 local
  GPT Image 1/Nano Banana observations, frozen before any model score. This is a rejection
  screen with scene/prompt limitations, not an independent final.
- Boon_or_Bane: public metadata lead only, approximately 284 MB. Image licence and exact
  provenance remain unresolved; no image download or model score and no role admission.

## Verification and synchronization

739 Python tests pass, including new transform identity, objective gradient, hard-decision
retention, serialization and failed-TRAIN-to-DEV access checks. These are engineering results,
not detector-quality improvements. The existing Starlette/httpx warning remains.

Completed code, compact evidence and MD checkpoints were committed and pushed. Latest
prepared-code checkpoint: `7537a39`. Raw image archives, feature arrays and candidate weights
stay on the external volume. GitHub's existing web dependency audit is still failing; do not
claim an all-green CI run. No 30-minute monitor or heartbeat automation was created.

Detailed immutable results: [E67 fit](e67_fit.json), [E67 TRAIN diagnosis](e67_training_diagnostic.json),
[E68 fit](e68_fit.json), [E66 admission](e66_development.json). The living next actions remain in
[PLAN](../PLAN.md), with append-only history in [EXPERIMENTS](../ml/EXPERIMENTS.md).


## First separate DEV result: E70 rejected

| E66 condition | E43 REAL false AI | E70 REAL false AI | E43 AI recall | E70 AI recall | Newly missed AI |
|---|---:|---:|---:|---:|---:|
| Publisher original | 42.50% | 46.88% | 97.50% | 95.63% | 4 |
| Social Q75 | 42.50% | 30.63% | 96.25% | 95.00% | 3 |

Each row has 160 REAL and 160 AI observations. Both models pass12/20 numeric gates;
E70 fails the separate AI-retention requirement. No E49 regression is allowed. E66 is
now consumed DEVELOPMENT; its initial unscored admission is a historical record.

E70 follow-up diagnosis:24/40/45 TRAIN AI views have logit margin<=1e-6 after correction,
with no TRAIN AI losses. Several existing DEV misses had appreciable reference margins,
so numeric roundoff does not explain all failures. Full fixed diagnostic:
`e70_margin_diagnostic.json`. E71 is actively completing cached CLIP TRAIN features;
all9,599 old chunks verified and30-source historical encoder replay exactly equal.
E71 feature/model/conditional DEV code pushed at1977647; no E71 fit result yet.


E71 CLIP TRAIN extraction completed:11,630 parents x3 conditions x1536 features;
9,599 verified legacy chunks plus2,031 new E71 chunks,1538.79s. Thirty source-selected
raw encoder replays matched exactly (max error0). Feature SHA
`dc0c4ab80d3be56495a6c5f19ee2c5a4c63a7f0e9eacf97856d7f529fb5a0603`;
contract SHA`9ffd471967c6b85cdb618ed6714e21844420d4c4b30bd093e295e5ff7ccf26d5`.
No DEV/final feature rows read, classifier scores0, downloads0, E59 outputs unchanged.
Fit contract is now frozen and the single preregistered fit is running. No quality result
or DEV permission is implied by feature completion. Full arrays remain external under e71.


### E71 single frozen TRAIN fit passed (2026-09-13)

28.22s, solver success, zero newly missed AI views and zero new REAL errors across all3
TRAIN conditions. REAL false AI: clean492/7035=6.9936%, assigned transport546/7035=7.7612%,
Q75 480/7035=6.8230%, versus E43 15.0675%/16.1336%/13.8024%. All3 <=10% gates passed.
Candidate SHA`f3e15446623b533c7519def877f855cc13d5e8a0175f43eb6c2daa36e855ef5b`, contract SHA`72c381457d619fd9331023d5cd91bc9f9fcc696aae9d30662cff72aa9954d088`.
This is constrained TRAIN feasibility, not external retention. Proceed to the separately
registered one640-view consumed E66 comparison, with all20 numeric gates and zero newly
missed AI per condition/source. E70 already consumed this set; never call it fresh/final.
No E49 access or promotion from this TRAIN result. Fixed recipe, no parameter sweep.


### E71 rejected on consumed E66 DEVELOPMENT (2026-09-13)

640 scores locked in412.36s; historical E43 pixel/score/decision replay is exact (max score
error0). Original REAL false AI68/160=42.50% ->55/160=34.375%; social Q75 68/160 ->61/160=38.125%.
Original AI156/160=97.50% and Q75 154/160=96.25% remain equal in aggregate, but original loses
2 previously caught GPT images while rescuing2; Q75 loses1 GPT while rescuing1 Nano Banana.
Thus aggregate recall masks per-image regression. Both conditions fail zero-new-AI-miss and
absolute gates (12/20 total). No E49 regression, no serving promotion. Candidate rejected.
Scores SHA`971b6e130b51f8ef741e4a194782e4b9c02e6ba18b68582db82f8eea07be8db0`. E66 remains consumed DEV; target unmet.

Next isolated mechanism E73: retain the exact frozen E71 original64/CLIP64 coordinates,
discard E71 weights, and replace caught-AI decision-only protection with non-decreasing
reference logits for ALL13,785 AI TRAIN views, including currently missed AI. Keep E64
correct-REAL decision constraints, objective/class-source-parent weights, L2.01, zero start,
SLSQP200/ftol1e-9, fixed cuts and all3 REAL FPR<=10% guards. No fractional margin/penalty/rank
sweep. One fit on unchanged E54 TRAIN; MIDD is a separate acquisition. Only complete TRAIN
pass permits a separately registered consumed DEV comparison. No external AI-retention claim
from TRAIN constraints; E70/E71 failures motivate testing confidence erosion explicitly.


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
