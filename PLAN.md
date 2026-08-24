# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(append-only project archive) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only scientific
log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

## Active goal — a runnable project-owned model (2026-08-24)

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

- [ ] Add a concise model card covering training data, architecture, inference contract, measured
      strengths, worst-source failure, intended use and prohibited authenticity claims.
- [ ] Record every M1-M5 commit, command, test count and measured model result in this plan and the
      append-only experiment log; generate report-ready tables/figures only from stored results.
- [ ] Capture one reproducible demo scenario for the internship presentation: input, project-model
      output, comparison output and the explanation of why they may disagree.
- **Acceptance:** a reader can trace every presentation claim to a result file, experiment entry,
  artifact hash and commit without relying on an undocumented manual run.

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

## Repo conventions after the 2026-08-18 tidy-up

- `ml/experiments/` holds the runnable E20–E27 protocol scripts; finished earlier evidence
  scripts are frozen in `ml/experiments/archive/`.
- `ml/src/pixelproof/archive/` holds retired modules (E2–E4 analysis, ELA, DINOv2
  extraction). Nothing in the live path imports them.
- `ml/artifacts/archive/` (not committed) holds superseded artifacts, including the
  poisoned `*.BOZUK_etiket.bak` evidence files. Nothing is deleted.
