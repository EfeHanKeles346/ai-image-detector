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
