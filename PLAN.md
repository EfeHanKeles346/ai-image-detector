# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(append-only project archive) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only scientific
log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

## Active goal — E32 in-the-wild data rebuild and Champions League evaluation (2026-08-26)

The product goal remains a genuinely testable binary detector, not a high score on a familiar
dataset. E31 proved that the label-clean, source-capped DINOv2 candidate can recall current AI
generators but rank independent authentic photographs backwards: E30 DEVELOPMENT AUC 0.385,
80.67% current-AI macro recall and 83.63% authentic macro false positives. Recalibration reduced
AI recall to 0.33%, so the next experiment must change the authentic data distribution and the
image representation together. It must not reinterpret that failure as a threshold problem.

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

- [ ] Target **10,000–20,000 eligible REAL parents**, nominally about 15,000, across native camera,
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
- [ ] Complete the frozen transfers and final content hashes without overwriting existing E31
      holdings or modifying an upstream archive in place.
- [x] Implement the role-free realization gate before any transfer completes. It binds every audit
      to the frozen selection SHA, ignores exFAT AppleDouble sidecars, requires all selected bytes
      to decode, derives format from payload bytes, records geometry/EXIF/compression summaries,
      and rejects exact/dHash repeats against protected E30 roles and already-passed E32 sources.
      A pass means only `candidate`; the gate cannot assign TRAIN/CALIBRATION itself.
- [ ] Decode and inventory every selected parent; record camera/device/model, scene/event group,
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
- [ ] Target **10,000–20,000 eligible AI parents**, nominally about 15,000, with at least five
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
      Banana Pro. If pinned GPT cannot supply its missing 1,940 local pairs, stop and document a
      replacement source; never inflate another source or reuse a protected final to hide the gap.
- [x] Implement the metadata-only exact selector without writing a production receipt prematurely.
      Nano uses stable-hash id selection; CommunityForensics uses model-identity round-robin; NBP
      uses all 200 licensed images; Qwen/FLUX inherit their frozen prompt groups; GPT selects from
      all pinned upstream pairs independently of local availability. The full freeze hard-stops
      until GPT revision, licence tag and exact 4,000-pair listing are reachable and verified.
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
- [ ] Acquire the frozen 6,000 gap images after the passed decoder gate. OpenFake
      `core/test`/Reddit splits and frontier held-out models remain test candidates, never TRAIN.
- [x] Pin the AI realization contract before bulk completion: every four-output prompt group needs
      four decodable images, four non-empty matching UTF-8 prompt sidecars, declared byte counts
      and expected dimensions. Missing/partial/mislabeled rows produce a rejected audit and never
      a silently smaller training pool.
- [ ] Verify generator version, generation date, prompt/content group, native output status,
      licence/usage boundary and label direction for every admitted collection. Unknown generator
      identity may contribute only to a capped `unknown` group and cannot satisfy the five-family
      requirement.
- [ ] Match semantic topics across classes before representation training. Measure topic/source,
      format, geometry, compression and bytes/pixel shortcuts on native and every proposed model
      input. Apply transport augmentation with the same probability/range to REAL and AI; never
      make PNG/JPEG, resize or screenshot history a label proxy.
- **Acceptance:** 10,000–20,000 decontaminated AI parents, five or more verified current families,
  source caps, topic coverage and zero overlap with E30/Qwen/ITW/API final roles. Native risks and
  safe model-input conditions are frozen in `DATASETS.md` before training.

### Phase C3 — freeze TRAIN/CALIBRATION and the Champions League test battery

- [ ] Build a balanced parent manifest with source/device/generator/scene-disjoint folds. TRAIN may
      fit representations and heads; CALIBRATION may select aggregation, abstention and thresholds;
      neither may receive a row or derivative from DEVELOPMENT or LOCKED FINAL.
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

- [ ] Use the same selected parents, folds and input views for a small representation ladder:
      - **R0 control:** frozen E31 DINOv2-S final embedding / one-tile contract.
      - **R1 current-science:** a feasible pinned PE-Core frozen encoder (begin B/16; advance to
        L/14 only after memory/latency/licence smoke) plus regularized linear head, inspired by the
        2026 SSAFE result that curated 10K data can outperform much larger pools.
      - **R2 low+high-level ViT:** DINOv2-L intermediate-block/RINE-style head over a global view
        plus texture-aware native crops, testing the ITW-SM finding rather than repeating E31's
        single upscaled 128 px tile.
      - **R3 Hive control:** ImageNet-pretrained EfficientNet-B4 under the exact winning crop,
        augmentation and data contract. Hive's public architecture motivates this ablation; its
        unknown private recipe and 0.9 threshold are not copied.
- [ ] For multi-view candidates, compare a fixed small set only: global-only, texture-only and
      global+texture; mean/top-k aggregation is selected on CALIBRATION. Do not change inference
      crops after DEVELOPMENT.
- [ ] Train frozen linear heads first. Permit an adapter or last-block fine-tune only when the
      frozen representation shows transferable signal; permit full fine-tuning only after a
      controlled data-sufficiency/overfit gate. A 10–20K pool does not automatically justify
      end-to-end tuning of a large encoder.
- [ ] Consider Hive-like auxiliary generator supervision only after the binary candidate passes
      independently. It is an ablation regularizer, not a product requirement and not an ensemble.
- **Screen gate:** on untouched source-held-out CALIBRATION, require AUC >=0.85, current-AI macro
  recall >=60%, weakest sufficiently sized AI family >=40%, authentic macro FP <=10%, worst-source
  FP <=20% and no transport recall loss above 15 points. Otherwise stop or change the data/input
  representation; do not tune against DEVELOPMENT.

### Phase C5 — controlled training, complementarity and final decision

- [ ] Advance only the cheapest representation that passes C4 materially over R0. Run at least
      seeds 42/2024/2026, report source/group intervals and freeze the candidate artifact,
      preprocessing, threshold, aggregation, abstention and hashes before DEVELOPMENT.
- [ ] Compare model families individually first. An ensemble may be fitted only from out-of-fold
      TRAIN/CALIBRATION rows when two independently passing arms make complementary errors. Require
      at least +5 percentage points macro current-AI recall without worsening either authentic FP
      budget; otherwise retain the best single arm, as E9/E31 required.
- [ ] Run the frozen candidate on the DEVELOPMENT arms once. Advancement requires AUC >=0.85,
      authentic macro FP <=10%, worst authentic source <=20%, AI macro recall >=60%, every sized
      AI family >=40% and bounded transport loss. A failing candidate cannot consume locked finals.
- [ ] Run the surviving candidate and frozen baselines (E20, E31 and locally permitted external
      controls) over the Champions League final under identical bytes. Hive API may be an optional
      paid external reference on a pre-budgeted subset; its scores never become labels or training
      targets.
- **Successful prototype:** on untouched ITW-SM after decontamination, AUC >=0.90, balanced
  accuracy >=0.85, AI recall >=0.80 and REAL recall >=0.80, with per-platform/source results. A
  lower result remains a runnable research detector, not a universal authenticity authority.

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
