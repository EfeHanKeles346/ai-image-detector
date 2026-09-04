# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(append-only project archive) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only scientific
log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

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
- [ ] **Bind only two practical successor families before fitting.** E51-A reuses frozen DINOv2-S
      features but refits a source-balanced head with new camera originals plus Q75/JPEG/resize
      hard negatives. E51-B adds a compact fixed residual/DCT statistics branch to the same features,
      inspired by AIDE, and adversarially/source-balances the head rather than fine-tuning a huge
      backbone. A real-only distance arm may only turn a positive into UNCERTAIN, never certify REAL.
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
