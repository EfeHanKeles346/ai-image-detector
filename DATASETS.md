# Datasets — inventory and assignment

What we hold, what each set is good for, and which module it feeds. Replaces the
former `STATUS.md`, whose content is now covered in more detail by `HISTORY.md` §2b.

**The rule that governs everything here** (`HISTORY.md` §1b): a dataset flaw is a
*usage condition*, not a disqualification. A shortcut only exists if the model can
perceive it. Whole-image training can see image dimensions; tile training cannot,
because every tile arrives at 128×128 regardless of where it came from.

---

## Storage

| Location | Contents |
|---|---|
| `$PIXELPROOF_WORK_ROOT` (default `ml/work/`) | Prepared working sets: `archive`, `archive1`, `genimage_split`, `defactify_test`, probes and temporary experiment encodings |
| `$PIXELPROOF_WORK_ROOT/manipulation_test/` | Module 2 working set — 10 sub-datasets extracted from the 78 GB compilation (6.7 GB, 2,385 manipulated images each with a mask + 2,289 authentic). Rebuild: `python -m pixelproof.prepare_manipulation` |
| `$PIXELPROOF_DATA_ROOT` (default `ml/data/`) | Acquired source datasets; may point at an external volume |

The original machine can retain its existing layout without code edits by exporting, for
example, `PIXELPROOF_WORK_ROOT=/path/to/prepared-work` and
`PIXELPROOF_DATA_ROOT=/path/to/pixelproof-datasets`. Active commands no longer contain a
personal absolute path. Both portable defaults are gitignored.

⚠️ `manipulation_test` lived in `/tmp/m2` until 2026-08-04, where macOS would have
wiped it on the next reboot — and it was the only copy. E17/E18 take `--root`, and
`prepare_manipulation.py` rebuilds the portable work directory from the data root.

Extracted by default: everything except **OpenForensics**, which is 138 of the
191 tars and is face manipulation rather than splicing or inpainting. One tar per
split (~500 images) — more buys nothing until E17's mask-coverage filter is
loosened, since only 35–95 images per set survive it today.

---

## Module 1 — is this image AI-generated?

### E32/C1 authentic-photo acquisition — frozen, not yet realized

The exact acquisition was frozen on 2026-08-26 before an image download. Detailed URLs stay on
the external data root at `e32/real_acquisition_selection.json` (1,166,007 bytes; SHA-256
`200a7aeb23d9c303d880dff76a08b21e38efe666531a6552ffe4bdd5841eca4d`); the compact receipt is
`evidence/e32_real_acquisition_selection.json`. Candidate counts below are not yet usable rows.

All frozen C1 transfers are now physically complete. CSAFE `s21.zip` is exactly 17,588,803,163 B
and reproduces published MD5 `5c5f79e3e508a5cbf7a19e75846091d8`; a stalled 4,723,834,880-byte
prefix was completed through four exact HTTP ranges and promoted only after whole-file MD5. Its
internal rows are still unselected pending ZIP inventory.

CSAFE inventory passes: ZIP SHA `54a7193c...25df`, 7,996 JPEG under ten S21 devices. Exactly 4,000
are `blank` flat-field captures and 3,996 are `natural`; each class spans front/telephoto/ultra/wide
camera paths, with 798–800 total rows per physical device. Blank fields are ineligible for the REAL
training pool. Detailed inventory is 1,306,218 B / SHA `77a88649...fd8d`; compact evidence is
`evidence/e32_csafe_archive_inventory.json`. Natural rows remain unselected/unextracted here.

CSAFE natural selection is now frozen before member bytes: 3,996 JPEG, ten devices at 398–400 rows,
front/telephoto 998 each and ultra/wide 1,000 each. All 4,000 blank fields are excluded. Detailed
selection is 1,193,310 B / SHA `3a24bd50...ad1c`; compact evidence is
`evidence/e32_csafe_natural_selection.json`. Rows remain unextracted and role-free.

CSAFE extraction produced the exact 3,996 natural JPEG parents / 13,219,178,988 B with per-file
SHA and device/lens metadata; no blank member was extracted. Detailed receipt is 1,775,854 B / SHA
`32acdfb3...d7e4`; compact evidence is `evidence/e32_csafe_natural_extraction.json`. These rows are
role-free until the full decode/decontamination audit passes.

CSAFE realization passes: 3,996/3,996 RGB JPEG with EXIF, unique SHA/pHash, ten devices and four
lenses; zero protected/passed-peer overlap. One equal-dHash pair is not pHash-confirmed. Detailed
audit is 2,521,737 B / SHA `3ea951ec...b701`; compact evidence is
`evidence/e32_csafe-mcsidb-s21_realization.json`. Candidate-only volume is sufficient for the REAL
floor once the global overlay confirms no cross-source/cross-label collision.

| frozen source | selected transfer | intended value | licence / role boundary |
|---|---:|---|---|
| VISION base native | 3,500 JPEG parents / 35 devices | Device-balanced native camera data; excludes flat fields, video and all social derivatives | CC BY-SA 4.0; TRAIN/CALIBRATION candidate only after audit |
| Forchheim FODB | 3 archives / 22,940,347,533 declared bytes; 3,851 expected `orig` parents | 27 devices, 143 scene groups and five parent-linked social transports | Research use; no modification, commercial use or redistribution without author agreement |
| CSAFE MCSIDB Galaxy S21 | `s21.zip` / 17,588,803,163 declared bytes / MD5 `5c5f...91d8` | Modern Samsung computational-photography complement; one archive instead of the 132.7 GB collection | CC BY 4.0; internal devices/scenes remain unselected until archive inventory |

The two declared archive collections total 40,529,150,696 bytes; VISION consists of individual
files whose realized bytes will be measured during transfer. SOCRatES remains unavailable without
a signed agreement, ForensiCam-215K remains excluded for unclear dataset licensing/Baidu-only
transfer, and the other five CSAFE archives remain deferred. The owner gallery is absent.

`ml/experiments/e32_data_system.py` verifies the upstream VISION list, source licences, Figshare
article version, filenames, sizes and published CSAFE MD5. Downloads use TLS-verifying system
`curl`, `.partial` resume, retry and a 100 GiB free-space floor. FODB's TLS chain is not accepted by
the Python CA bundle on this machine; verification moved to system `curl`, never disabled.

Archive handling is now code-gated before extraction. `e32_archive_inventory.py` checks declared
archive size, CRC, traversal/absolute/backslash paths, symlinks, encryption, duplicate names,
member limits and expansion ratio. FODB additionally requires one original plus all five linked
social transports per device/scene parent and extracts only originals; CSAFE repeats its published
MD5 and inventories without selecting rows. Fifteen focused tests pass; production inventory waits
for all frozen archive bytes.

FODB's production archive inventory now passes. The three archives occupy 22,940,347,533 declared
bytes and reproduce SHAs `c719cac3...517c`, `271e07da...e5f1`, `a3c2d69f...2a6d`. It proves 3,851
parents across 27 camera pipelines / 143 scene groups, each with one `orig` and five named social
derivatives. Exactly 4,004 `inspection` helpers / 2,834,597,196 B are excluded as nonparents.
Detailed inventory is 5,356,810 B / SHA `d378573f...9631`; compact evidence is
`evidence/e32_fodb_archive_inventory.json`. Originals remain unextracted and role-free here.

The subsequent original-only extraction produced exactly 3,851 FODB JPEG parents /
15,416,129,383 B below the isolated E32 root. Per-parent SHA/device/scene metadata is bound to the
inventory; no social or `inspection` helper was extracted. Detailed receipt is 1,311,414 B / SHA
`a1626b0b...8b05`; compact evidence is `evidence/e32_fodb_orig_extraction.json`. These are still
role-free candidates until the decode/decontamination audit passes.

FODB realization passes: 3,851/3,851 RGB JPEG with EXIF, 3,851 unique SHA, 143 scene groups and 27
camera pipelines; protected/passed-peer overlap is zero. Seven equal-dHash cross-camera scene pairs
are not confirmed by pHash, so no perceptual duplicate is removed. Detailed audit is 2,588,737 B /
SHA `dcbf8b55...fd11`; compact evidence is `evidence/e32_forchheim-fodb_realization.json`. Role is
still candidate-only and later folds must group the 143 shared scenes.

The updated global overlay compares 15,000 AI plus 7,351 REAL selected rows. FODB adds no new
duplicate component or REAL/AI ambiguity, so AI remains exactly 14,786 and REAL becomes 7,348:
3,497 VISION + 3,851 FODB. Detailed overlay is 1,179,329 B / SHA `510e94eb...fc3b`; compact evidence
is `evidence/e32_eligibility_overlay.json`. The >=10,000 REAL floor still requires the selected
CSAFE complement.

### Training

| Dataset | Size | Contents | Mode | Why |
|---|---|---|---|---|
| **`OwensLab/CommunityForensics-Small`** | 49.77 GB / 44,884 rows | **300 distinct AI `model_name` values**; 11,972 AI / 32,912 real; prompt / architecture / real-source metadata | **fixed tile/encoder only** | Highest measured generator breadth, but native metadata AUC is 1.000 because reals are 1024 px and AI images 512 px. E31's fixed 128 RGB/JPEG probe falls to 0.636, below the frozen 0.65 ceiling |
| **`theminji/AI-vs-Real-balanced`** | 12.96 GB / 143,070 rows | 71,535 AI / 71,535 real; upstream order is `0=AI, 1=real` | **fixed tile/encoder preferred** | Balanced and the cleanest large paired source, but E31 still found different native format sets. Native AUC 0.549; fixed 128 probe 0.586 and passes |
| `TheKernel01/AIGC-Detection-Benchmark` | 32.03 GB / 125,026 rows | 62,513 per class; 17 AI generator codes plus the real code | **fixed tile/encoder only** | Native geometry shortcut is severe (AUC 0.967); fixed 128 probe falls to 0.540. Adds GAN-era breadth only after identical input normalization |
| `theminji/ai-vs-real-200k` | 51.93 GB / 241,609 rows | 124,209 AI / 117,400 real; upstream order is `0=AI, 1=real` | **fixed tile/encoder only** | Native resolution/format shortcut remains (AUC 0.841); fixed 128 probe is 0.552. Useful volume only under the audited representation |
| `genimage_split` | ~2 GB | 7 older generators, perfectly balanced | any | Keep as the control: every model to date was trained on it, so it is the fair comparison baseline |

**E31 starting policy:** do not call any large source unconditionally clean. Start TRAIN v2 from
source-capped CommunityForensics-Small + AI-vs-Real-balanced under one fixed tile/encoder input;
admit AIGC and ai-vs-real-200k only after selected-row leakage hashes and the same representation
contract pass. Generator/source-disjoint folds matter more than consuming all 138+ GB.

### Testing — never train on these

| Dataset | Size | Why it is a test set |
|---|---|---|
| **`defactify_test`** | 1.2 GB | Five generators newer than any training data, both classes JPEG. Our established benchmark — every number in `EXPERIMENTS.md` E7–E11 is measured here |
| **`julienlucas/midjourney-dalle-sd-nanobananapro`** | 3.12 GB / 12,695 rows | Contains **Nano Banana Pro** (2026) with 6,195 AI / 6,500 real. E31's first/middle/last-shard sample found a severe native metadata shortcut (AUC 0.974; differing format and square distributions), while the fixed 128 probe was 0.560. Keep it test-only, but never call native pooled accuracy a clean generalisation result; report standardized and source/generator slices |
| `archive1` | 240 MB | **Confounded** (see `HISTORY.md` §1b). Keep only for continuity with E1–E6; do not use for new claims |
| `archive` (CIFAKE) | 469 MB | 32×32. Only SmallCNN's domain |

### E31-B1 attached-disk audit (2026-08-25)

The LaCie source root was audited read-only with `ml/experiments/e31_ssd_audit.py`. Registered
sources occupy **173.58 GB** and inventory-only sources another **97.34 GB** (270.91 decimal GB
total). Complete Parquet metadata covers 603,991 registered rows. A deterministic
first/middle/last-shard image probe decoded **3,000/3,000** samples without failure and compared
their exact byte hashes with **980** E30 parent/derived protected hashes; no sampled exact overlap
was found.

| paired source | native metadata AUC | fixed 128 RGB/JPEG probe | decision |
|---|---:|---:|---|
| CommunityForensics-Small | **1.000** | 0.636 | native reject; fixed representation may enter TRAIN-v2 selection |
| AI-vs-Real-balanced | 0.549 plus differing format sets | 0.586 | fixed representation preferred |
| AIGC benchmark | **0.967** | 0.540 | native reject; conditional fixed-representation candidate |
| ai-vs-real-200k | **0.841** | 0.552 | native reject; conditional fixed-representation candidate |
| Julien Lucas modern | **0.974** | 0.560 | stays test-only; native pooled claims unsafe |

The fixed probe neutralizes geometry/format metadata; it does **not** prove that decoded pixels are
free of compression or collection artefacts. The overlap result is also explicitly sampled, not a
full-pool guarantee. E31-B2 must first freeze the exact TRAIN v2 rows, then hash every selected row
against calibration, owner-gallery, DEVELOPMENT, LOCKED FINAL and named test-only content before
any feature extraction or training. Aggregate evidence SHA-256 is
`2f7399bed965a8a428b4180aab059405fbcc4d4aa4d3754a5295ee4e97021f29` in
`evidence/e31_ssd_audit.json`; no source image or private identifier is committed.

### E31-B2 TRAIN-v2 frozen selection (before image-byte access)

The first training contract deliberately selects breadth rather than all available volume. Exact
Parquet shard/row ids were frozen from metadata only; no image byte, protected test hash or model
score influenced selection.

| selected source | AI | real | grouping/role boundary |
|---|---:|---:|---|
| CommunityForensics-Small | 2,400 | 2,400 | exactly 8 per each of 300 AI generators; real groups are whole shards |
| AI-vs-Real-balanced | 2,000 | 3,250 | whole Parquet shards assign TRAIN or CALIBRATION |
| FLUX.1-dev | 500 | 0 | whole shards; fixed native-tile representation only |
| Nano Banana | 500 | 0 | whole shards; fixed native-tile representation only |
| Nano Banana Pro | 250 | 0 | whole shards; fixed native-tile representation only |
| **total** | **5,650** | **5,650** | 383 groups; 8,561 TRAIN / 2,739 CALIBRATION |

Every source has both roles; no group crosses roles. AIGC and ai-vs-real-200k stay deferred because
the first candidate already supplies 11,300 balanced parents and 303 AI identities without their
severe native shortcuts. Test-only Defactify, Julien Lucas, CommunityForensics probes, owner gallery
and every E30 row are absent. Selection SHA-256 is
`5907c14ba3e173c125c024a30658fb8e7e56788a469614808ad4ef5519a5fbfb`; the complete row contract
is `evidence/e31_train_v2_selection.json`. B2 is not realized until every selected row passes full
exact/dHash protected-content checks and deterministic tile extraction.

The first realization rejected that selection: 3,534 rows could not produce the frozen native
128 px tile. A complete mechanical scan then measured AI-vs-Real-balanced's hidden size problem:
47,233/71,535 AI and 50,000/71,535 real are below 128 px; 24,301 AI and 21,532 real pass, with only
four additional texture-floor rejects. Eligibility-set SHA is `91089e22...eb2` and the full key
set is `evidence/e31_balanced_eligibility.json`. Selection v2 preserves the table above exactly,
keeps 7,767 old rows and replaces 3,533 with eligible alternatives. Its SHA is
`5355e4307eb72053a01fcfc3c13e2a431feed7a313a316317fed4303bd2679b2` and exact rows are frozen in
`evidence/e31_train_v2_selection_v2.json` before another realization.

Selection v2 was then rejected because 84 unique rows overlapped protected content or failed the
tile floor. The pre-registered full protected screen inspected **163,777** candidates from the
four replaceable sources against 176,961 exact and 172,087 dHash protected fingerprints. It left
65,650 eligible rows and excluded 97,982 exact matches, 137 additional dHash-only matches and six
flat rows. Large exact counts in the balanced source are expected because the protected historical
`archive` contains 120,000 earlier project images; those rows are deliberately unavailable for
fresh E31 training.

Selection v3 keeps the same 5,650/5,650 balance, 383 groups, 303 AI identities and TRAIN/CALIBRATION
roles. It retains 11,216 v2 rows and replaces exactly 84 rows source-for-source; all 4,800 selected
CommunityForensics rows remain unchanged after their zero-overlap v2 result. Selection SHA is
`1a3a5c98c4b0614a0af4bd1bc65ca4fbb8ea33404dbb6a2db53b2da17b79df2e`; exact ids are in
`evidence/e31_train_v2_selection_v3.json`. Compact screen evidence is
`evidence/e31_protected_screen.json`; the full eligible-key receipt stays ignored under
`ml/data/e31/` with SHA `16ff5f14...bad10`. This is still a frozen candidate contract, not a
realized training set, until the independent v3 byte pass succeeds.

The independent v3 realization subsequently accepted **11,300/11,300** rows: 5,650 AI / 5,650
real, 8,561 TRAIN / 2,739 CALIBRATION, with zero decode failure, zero input-floor loss, zero exact
protected overlap and zero dHash protected overlap. All 11,300 produced tiles have unique hashes;
the 37 repeated raw-byte hashes occur only across selected parents and do not collapse to duplicate
deterministic tiles. The ignored local archive is 395,082,960 bytes with SHA
`508330c2d8318bcd4c8a92c86a86a627ff98ee1bdc97a67772540a68c8569f2b`. Its compact identity and
full source/role counts are committed in `evidence/e31_train_v2_realization_v3.json`. E31-B2 is now
accepted for representation training; no E30 test image entered the archive.

### E30 pinned current-science sources — role assignment before download

| Source (pinned revision) | Full source | E30 role and initial slice | Why selected | Boundary |
|---|---:|---|---|---|
| `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (`1498eead…b9de`) | 4,356 rows / 3.32 GB | **DEVELOPMENT TEST**; planned 180 preprocessed JPEGs: 20 per GPT Image 2 / Nano Banana 2 / real × texture / structure / hybrid cell | Matched 2026 generators and real class with three explicit artifact regimes; independent of detector score | Dataset card is research-use restricted. JPEG arm is standardized transport, not native output; raw arm remains separate |
| `Qwen/Qwen-Image-Bench` (`d2493deb…7038`) | 12.7 GB | **LOCKED FINAL TEST candidate**; first sealed scout is 5 each from 8 named 2026 generators (40 original mixed PNG/JPEG files) | Independent collection and broad frontier coverage: GPT Image 2, Nano Banana 2, Seedream 5, Qwen Image 2 Pro, FLUX.2 Max/Pro, GLM-Image, Hunyuan Image 3 | Five per generator is scout-only. No success claim below 40 per reported generator; selected rows stay unscored until candidate/threshold freeze |
| `laionmobile/laion-mobile` (`0c60f598…3465`) | 935,399 metadata rows / 151 MB; evaluation manifest 9,115 rows / 2,639,565 B | **DEVELOPMENT TEST**; planned 8 declared phone/web pipeline groups × 10 local reconstructions | Real-only false-positive stress test with EXIF make/model and upstream content hashes, fetched row-wise rather than mirroring the corpus | Metadata is CC-BY-4.0; image licences remain upstream. Web-reprocessed and mostly older phones, not a native-camera vault |
| New private multi-phone vault | Not yet collected | **LOCKED FINAL TEST**, target 4 pipelines × 40 untouched originals | Only reliable way to match native iPhone/Samsung/Pixel computational-photography pipelines without web laundering | Existing owner gallery is exposed development regression; no personal bytes, names, GPS or per-image identifiers enter Git |

`ml/e30_sources.json` is the machine-readable source registry. The pinned sizes above are upstream
reported totals, not local acquisitions. Exact selected counts, downloaded bytes, hashes and audit
results will replace the planned slice descriptions after E30-A2/A3 realization. E30 test bytes
remain under ignored `ml/data/e30/` and are forbidden from TRAIN/CALIBRATION.

### E30-A2 low-bandwidth realization (2026-08-25)

| Arm | Realized local data | Audit outcome | Scientific use |
|---|---:|---|---|
| MLLMGenSet parents | 180 JPEGs / 4,419,610 B: 120 AI, 60 matched real; exactly 20 per nine frozen generator/class x regime cells | 180 unique SHA-256; metadata-only AUC 0.6238, pass | DEVELOPMENT TEST only; standardized-JPEG GPT Image 2 / Nano Banana 2 diagnostic |
| MLLMGenSet derivatives | 720 JPEGs / 14,029,255 B: q90, q75, q50 and resize256-q90 for every parent | 900/900 hashes unique across parents+children; transport AUCs 0.6096, 0.6191, 0.6362, 0.6127, all pass | Robustness views of the same underlying content; never independent samples or another split |
| LAION-Mobile attempt | Metadata manifest 2,639,565 B; 55/80 URLs eligible under the frozen 375 KB/file rule; **zero images downloaded** | `source_incomplete`: Apple cells 10/10 each; Samsung/Xiaomi cells 9/10, 5/10, 1/10, 0/10. 287/361 rejects exceeded the per-file cap | No benchmark arm exists yet. Do not report the 55-row partial selection or substitute other phone groups |

The complete realized MLLM image battery is **18,448,865 bytes**, below the 30 MB target. Its
parent content-set SHA-256 is `1f3a7333...df2e`; the parents-plus-derivatives content-set SHA-256
is `7634755c...24b8`. The frozen parent selection remains `f71c8d02...035e`. All third-party
bytes and detailed URL diagnostics stay under ignored `ml/data/e30/`; the compact, presentation-safe
aggregate is `evidence/e30_development_realization.json`.

The LAION outcome is not repaired post hoc. Selecting the ten smallest reachable candidates in
each frozen phone cell would require about 45.96 MB for that arm alone (the Redmi cell about
22.81 MB), exceeding both its 30 MB arm contract and the complete low-bandwidth development
budget once MLLM is included. A future full-internet profile may pre-register a larger budget or
replace this source with a native multi-phone vault, but must create a new selection version.

### E30-A3 sealed Qwen scout (before image download)

The exact 40-row LOCKED FINAL TEST scout is frozen at five score-blind, numerically first source
paths per each of eight generators. Its declared size is **37,907,745 bytes**, selection SHA-256
is `50e3fec1...eeb`, and the source tree contains 21 PNG plus 19 JPEG files. The initial all-PNG
assumption was corrected from repository metadata before any image or model score was read; native
encodings will be preserved. The complete sealed list is
`evidence/e30_qwen_sealed_selection.json`. Five examples per generator support only a pipeline
scout, never a success/failure claim or threshold choice.

The sealed scout subsequently downloaded **40/40** unique, decodable originals with no byte or
format mismatch. Native image bytes exactly equal the declaration at **37,907,745** and content-set
SHA-256 is `0f25bfe7...8a1`. One deterministic q90 JPEG child was created for every parent
(40 files / 9,449,715 B); all 80 parent/child byte hashes are unique and the combined content-set
SHA-256 is `93dcbc01...49c`. Children inherit LOCKED FINAL role, label, generator, prompt content id
and parent id. Since this arm is AI-only, no real-vs-AI metadata shortcut AUC can be computed.
No detector has read these images. Aggregate evidence is `evidence/e30_qwen_realization.json`.

### E31 role outcome (2026-08-26)

The frozen E31 DINO candidate consumed only the 900-row MLLM DEVELOPMENT battery and failed its
real-photo gate (83.63% source/regime macro FP, 100% worst group). Consequently the conditional
Qwen step was not authorized: all 40 native + 40 standardized Qwen rows remain unscored and retain
LOCKED FINAL ownership. `evidence/e31_b5_development.json` records the failure and the sealed runner
rejects that state before reading the Qwen manifest or loading a detector. No E30 row has moved into
TRAIN/CALIBRATION and no new dataset was selected after seeing the result.

### Current generators, AI-only — pair with care

#### E32/C2a physical and licence inventory (2026-08-26)

`evidence/e32_ai_inventory.json` counts the complete local storage metadata without decoding an
image or changing a role. A folder/repository name is not accepted as generator provenance, and a
model/code licence is not substituted for a missing dataset licence.

| holding | realized local content | provenance/licence verdict | C2 use |
|---|---:|---|---|
| FLUX.1-dev | 10,000 Parquet rows / 3,231,877,594 B | Prompt+seed exist, but card has no dataset licence or narrative generation statement | Conditional; does not count toward five families |
| Nano Banana | 9,457 Parquet rows / 14,853,199,670 B | Card explicitly names Gemini 2.5 Flash Image Preview; MIT | Eligible candidate after byte/decontamination audit |
| Nano Banana Pro (`kaupane`) | 1,250 Parquet rows / 2,205,250,281 B | Repository name and prompts only; no dataset licence or narrative statement | Conditional; does not count |
| Nano Banana Pro (`ash12321`) | 200 PNG / 346,757,202 B | Card explicitly declares AI-generated NBP; MIT | Eligible candidate, same family as above |
| GPT Image 1 | **partial** 1,060 PNG + 1,061 text files / 2,260,502,012 B | 1,060 image/prompt pairs; card declares 4,000 GPT Image 1 images; CC BY 4.0 | Eligible candidate; missing upstream rows are not claimed |
| Nano Banana 150k | one ZIP / 127,835 image members / 10,642,043,397 B | Card claims >150k identity-consistent editing samples; no dataset licence or row manifest | Conditional for licence, count and task identity |
| CommunityForensics-Small | 44,884 rows / 49,764,880,239 B; 300 local AI identities | CC BY-NC-SA 4.0 plus per-model terms; already E31-audited | Diversity anchor only; not a fifth current family |

The verified admissible modern-family count is **3/5**: GPT Image 1, Nano Banana and Nano Banana
Pro. Therefore C2 cannot freeze its 10–20K parent manifest yet. At least two additional licensed,
explicitly generated modern families must be researched; protected AIGC, Julien, MLLM and Qwen
test sources cannot be reassigned to fill the training gap.

#### E32/C2b licensed gap-source selection — frozen, decoder smoke pending

The two-family gap was filled by source research, not by relaxing C2a's rules. Exact selections
stay at external `e32/ai_gap_selection.json` (2,349,078 B; SHA-256
`e9c3d3dad2ceb245b157f6e851e142128573726fe8b963d1811dfdaac4ceaf7a`); compact evidence is
`evidence/e32_ai_gap_selection.json`. Freeze downloaded zero image bytes.

| pinned source | available upstream | frozen selection | why / boundary |
|---|---:|---:|---|
| `stablellama/Qwen-Image-2512_samples@46849cd…` | 3,936 JPEG XL / 984 four-output prompt groups | 750 groups / 3,000 images / 7,108,445,821 B | CC BY-SA 4.0; explicit bf16 Qwen Image 2512 recipe; separate TRAIN/CALIBRATION candidate, not the Qwen Image 2 Pro locked final |
| `stablellama/FLUX.2-klein-base-9B_samples@c07dd3c…` | 4,072 generated JPEG XL / 1,018 groups | 750 groups / 3,000 images / 4,400,537,141 B | CC BY 4.0; explicit FLUX.2 Klein 9B recipe; 160 non-generated editing references excluded |

Category round-robin prevents alphabetic selection from filling the cap with one subject. All four
unfiltered variations of a selected prompt remain one future split group. The cards/path suffixes
declare JPEG XL, but the two exact smoke files contain **PNG payloads**: Qwen decoded RGB
1328x1328 / 2,579,073 B and FLUX decoded RGB 1024x1024 / 1,215,314 B directly through Pillow.
`evidence/e32_ai_gap_decoder_smoke.json` binds these hashes to selection `e9c3...af7a`; bulk is
mechanically forbidden without that receipt. Actual format comes from decoded bytes, and REAL/AI
must receive the same model-input normalization so PNG versus JPEG cannot become the classifier.

Bulk completion is intentionally separate from eligibility. The precommitted C1/C2 realization
gate requires every frozen image/prompt byte, full decode, payload-derived format/dimensions,
SHA-256+dHash decontamination against protected E30 and passed E32 sources, and complete four-output
AI prompt groups. Its detailed row receipts remain under external `e32/audits/`; only compact
hash-bound evidence enters Git. Until those receipts pass, the table above remains a frozen
acquisition selection—not a TRAIN/CALIBRATION dataset count.

The nominal post-audit AI composition is preregistered at exactly 15,000 parents: Qwen 3,000,
FLUX.2 3,000, Nano Banana 3,000, GPT Image 1 3,000, licensed NBP 200 and CommunityForensics AI
2,800. The first four are each 20%; CommunityForensics is 18.67% and NBP 1.33%. The five verified
current families exclude CommunityForensics, which is only the broad model-identity anchor. Local
metadata supports 9,457 unique Nano ids and 11,972 Community AI rows across 300 model identities;
The exact receipt later selected 795 already-local GPT pairs and 2,205 download-required pairs;
the arithmetic 1,940 local-volume shortfall was not used as a row-selection rule. These remain
target counts, not realized counts, until every source passes the byte/decontamination gate.

The metadata-only selector reproduced every allocation without opening image bytes:
Nano 3,000/9,457 unique ids, NBP 200/200, Community AI 2,800/11,972 across all 300 model identities
with at most ten selected per model, plus the existing 3,000+3,000 Qwen/FLUX receipts. The pinned
GPT SHA, CC-BY-4.0 tag and 4,000 complete PNG/TXT pairs reproduced. Detailed selection is external
`e32/ai_pool_selection.json` (4,752,567 B; SHA-256 `3230f0267ca1b9a252ec61d7f94c90bdb820cf8aeec5fce2174ebc5a7ed980b7`),
with record-selection SHA `2a31e7921054ac4915533735f3649cbb2d6b204439e93a6277a2164cd26a0ef7`.
A partial local folder remains an availability cache, not a selection rule: only 795/3,000 exact
GPT pairs are already local and 2,205 require download.

The selected missing GPT smoke pair passed before bulk: `GPTIMG_852.png`, RGB PNG 1024x1536,
3,486,339 B, SHA-256 `8f30398f...6e96`, plus a non-empty 1,341-byte UTF-8 prompt. Compact evidence
`evidence/e32_gpt_decoder_smoke.json` binds it to record-selection SHA `2a31e792...0ef7`. This is a
decoder/acquisition result only, not source eligibility.

The realization implementation now covers every storage form in the frozen pool: embedded
Parquet `image`/`image_data`, licensed loose images, local GPT pairs and isolated E32 downloads.
It revalidates source fingerprints and exact row locators before decode, then applies one shared
hash/duplicate/protected-overlap gate. A successful source receipt still means `candidate_only`;
TRAIN/CALIBRATION assignment remains a later frozen-manifest operation.

Realization schema v2 separates **candidate collision** from **confirmed perceptual duplicate**.
Exact SHA-256 still proves byte identity. Equal 64-bit dHash only creates a candidate pair; the pair
must additionally have DCT-pHash Hamming distance <=5 to be rejected as a modern E32 duplicate.
This was required after five visibly unrelated Nano images shared one dHash but were 24–32 pHash
bits apart. Protected E30 dHash hits remain conservative exclusions because those legacy manifests
did not store pHash. No row selection or role changed while correcting this audit method.

The complete licensed NBP source is the first 15K arm to pass realization: 200/200 PNG, 200 unique
SHA-256, 200 unique dHash, zero duplicate or protected-E30 overlap. Modes are 136 RGB / 64 RGBA and
EXIF is absent, so alpha/mode cannot be exposed as a class shortcut. Detailed external receipt is
91,762 B / SHA `bfc217f0...d17b`; compact evidence is
`evidence/e32_nano-banana-pro-ash-local_realization.json`. Its role remains `candidate_only`.

The frozen Nano Banana source also passes schema-v2 realization: 3,000/3,000 RGB PNG, 3,000 unique
SHA-256 and pHash, zero exact/confirmed-perceptual duplicate, and zero protected/passed-peer
overlap. One five-image dHash collision bucket is retained in evidence but all pairwise pHash
distances are 24–32, so it is not a near-duplicate group. Detailed receipt is 1,767,170 B / SHA
`8cb04e52...fe2f`; compact evidence is `evidence/e32_nano-banana-local_realization.json`.

Qwen's frozen 3,000-output source does **not** pass intact. It decodes fully with zero protected or
peer overlap, but two composition prompt groups duplicate two architecture groups across all four
variants (eight exact duplicate pairs), and one style prompt group contains a confirmed
near-duplicate pair. The source has 2,992 unique SHA and 2,990 unique pHash. Detailed rejected
receipt is 2,020,166 B / SHA `fbdc34d4...ad57`; compact evidence is
`evidence/e32_qwen-image-2512_realization.json`. A later immutable-selection eligibility overlay
must drop all three affected prompt groups as units before any TRAIN/CAL role.

FLUX.2 likewise fails intact-source hygiene: 3,000/3,000 decode and zero protected/peer overlap,
but 28 exact plus 41 confirmed perceptual duplicate groups leave 2,964 unique SHA and 2,932 unique
pHash. The conflict set covers 98 image keys / 32 prompt groups, mainly `diffusiondb_orig` repeats
and editing variations. Detailed rejected receipt is 2,045,961 B / SHA `53c0793b...1451`; compact
evidence is `evidence/e32_flux2-klein-9b_realization.json`. Group-safe canonical pruning is pending
the combined eligibility overlay; no replacement row is selected post-decode.

GPT transfer is physically complete at 3,000 selected image/prompt pairs. Its first strict audit is
rejected: 107 sidecars are Windows-1252 rather than UTF-8, so 2,893 images were realized; those have
zero protected/peer overlap but five confirmed perceptual duplicate pairs. All 107 sidecars decode
under the one explicit Windows-1252 fallback and contain only audited punctuation/accent characters.
Detailed rejected receipt is 1,792,420 B / SHA `9ce487a2...5184`; compact evidence is
`evidence/e32_gpt-image-1_realization.json`. A decoder-method commit and unchanged-selection rerun
are required; duplicate losers remain an eligibility-overlay concern.

VISION transfer is physically complete at 3,500 native JPEG parents / 10,289,109,711 bytes. All
images decode with EXIF, 100 per each of 35 pipelines, and all SHA values are unique. Three
confirmed perceptual pairs reject the intact source despite zero protected/peer overlap. Detailed
receipt is 1,939,155 B / SHA `3312c774...e6b1`; compact evidence is
`evidence/e32_vision-base-native_realization.json`. Stable loser exclusion is pending; no role is
assigned and no after-the-fact replacement is allowed.

CommunityForensics' selected diversity anchor passes schema-v2 realization: 2,800/2,800 RGB PNG,
2,800 unique SHA/dHash/pHash, 300 represented model identities and zero protected/peer overlap.
Detailed receipt is 1,980,274 B / SHA `cb4bffe2...76b2`; compact evidence is
`evidence/e32_communityforensics-ai-local_realization.json`. Nano Banana Pro's 200-row receipt was
also refreshed under schema v2 and remains clean; its new detail is 98,924 B / SHA
`55ec23ec...eb8e`. Both remain role-free candidates.

GPT's unchanged-selection rerun realizes all 3,000 RGB PNGs and prompts: 2,893 UTF-8 plus 107
Windows-1252, each with original-byte and normalized-text hashes. All image SHA values are unique
and protected/peer overlap is zero. Six confirmed perceptual pairs still reject the intact source;
the deterministic overlay must exclude one loser per pair without replacement. Detailed receipt is
2,239,691 B / SHA `48945f7f...73d5`; compact evidence remains
`evidence/e32_gpt-image-1_realization.json`.

The receipt-bound global eligibility overlay now retains 14,786/15,000 AI parents and 3,497/3,500
VISION parents. AI counts are Qwen 2,956; FLUX.2 2,916; Nano Banana 2,957; GPT Image 1 2,957; Nano
Banana Pro 200; CommunityForensics 2,800. Maximum source share is 19.998647%; Qwen/FLUX parent groups
remain indivisible. Across 18,500 records, 59 duplicate components produced 13 internal-parent and
20 noncanonical-unit exclusions, with zero REAL/AI component. Detailed overlay is 913,980 B / SHA
`b6c2101f...32e4`; compact evidence is `evidence/e32_eligibility_overlay.json`. Rows remain
role-free until real-source acquisition and fold design finish.

| Dataset | Size | Model | Era |
|---|---|---|---|
| `bitmind/nano-banana` + `Nano-banana-150k` | 24 GB | Gemini 2.5 Flash Image | 2025 |
| `kaupane/nano-banana-pro-gen` + `ash12321/…-1k` | 2.5 GB | Nano Banana Pro | 2026 |
| `ash12321/flux-1-dev-generated-10k` | 3.0 GB | FLUX.1-dev | 2024-25 |
| `a3xrfgb/gpt-image-mega-4k` | 3.3 GB (partial) | GPT Image, 4K | 2025-26 |
| `34data/communityforensics-fake` / `-real` | 3.3 GB | CommunityForensics sample | — |

⚠️ These have no real half. Pairing them with camera photos **recreates the archive1 trap**:
they are PNG squares, photos are JPEG rectangles. Two safe options — push both classes through
one identical encoder, or use them in tile mode only.

Best immediate use: **per-generator recall probes**. Score them with an existing model and read
the recall; that needs no real half and answers "does our detector see FLUX at all?"

---

## Module 2 — where was the image manipulated?

**Status: parked, not served.** The current tile overlay is an uncalibrated detector-score map.
E17/E18 found localisation signal on diffusion inpainting (CocoGlide) but not a general result on
classic splicing. Module 2 resumes only after a localisation model is evaluated against the pixel
masks below on the relevant manipulation family.

| Dataset | Size | Contents |
|---|---|---|
| **`ductai199x/image-manipulation-dataset-compilation`** | 78 GB | 13 forensic datasets, split `auth` / `manip`, **with pixel-level ground-truth masks** |

Per-image files:

```
<name>.png            the image
<name>.mask.png       binary mask — 0 / 255, same dimensions, marks the tampered pixels
<name>.json           {"manip_label": 1, "auth": "…/Au_ani_00018.jpg"}   ← points at the ORIGINAL
<name>.cls            class label
```

Verified on a CASIA 2.0 sample: 384×256 image, mask covering 37% of pixels in a contiguous band
(y 0–117). The `auth` pointer means we also have **before/after pairs** of the same scene.

### What is inside, and why the split matters

| Sub-dataset | Tars | Manipulation type | Expected difficulty for our tile model |
|---|---|---|---|
| OpenForensics | 139 | face manipulation | unknown |
| CASIA 2.0 | 26 | classic splice / copy-move | **hard** — Photoshop, not AI |
| **CocoGlide** | 2 | **diffusion inpainting** | **easiest** — genuinely AI-filled regions |
| IMD2020, NIST2016, Columbia, Coverage, DSO-1, CMFD, RealisticTampering, VIPP | 12 | classic edits | hard |

**Report per sub-dataset, never as one average.** Our tile model asks *"does this tile look like
AI-generated texture"*, not *"was this tile edited"*. Those coincide for a diffusion-inpainted
region and diverge for a Photoshop splice. A single pooled number would hide exactly the
distinction that matters.

---

## Assignment summary

```
MODULE 1 train   CommunityForensics-Small ┐
                 AI-vs-Real-balanced      ├─ fixed tile/encoder TRAIN-v2 candidates
                 genimage_split           ┘  (older control)
                 AIGC-Detection-Benchmark ┐
                 ai-vs-real-200k          ┘  tiles only

MODULE 1 test    defactify_test              established benchmark
                 julienlucas                 Nano Banana Pro; fixed-view test only
                 AI-only sets                per-generator recall probes

MODULE 2         image-manipulation-compilation   masks, 13 sub-datasets
                 └─ report per sub-dataset, not pooled
```

---

## Gaps

1. **Generator inventory is known; selected-row coverage is not yet frozen.** E31 measured all
   44,884 local CommunityForensics rows and 300 distinct AI `model_name` values, overturning the
   old 228-model estimate. B2 must still source-cap these highly uneven groups, create
   generator-disjoint folds and hash every selected TRAIN-v2 row before training.
2. **No realized native 2026 editing arm yet.** E30 pins GPT Image 2 / Nano Banana 2 data whose
   paper covers direct generation, reference reconstruction and local editing, but the compact
   local slice has not yet been downloaded/audited and the exposed HF folder hierarchy does not
   carry every protocol field. It cannot close Module 2's localisation gap by assumption.
3. **Compression is measured but remains regime-specific.** E23c evaluated q50 degradation and
   showed thresholds do not transfer safely between compression regimes. E30 therefore keeps
   native/standardized/q90/q75/q50 claims separate instead of treating augmentation as a cure.
4. **The AI-only sets are unusable as-is.** They need either a controlled real half or tile-mode
   evaluation.

### E32 final role-free candidate pool (2026-08-26)

The receipt-bound global overlay is complete. It compared 26,347 frozen selected parents and
retained 26,130 eligible, still role-free parents:

| class/source | selected | eligible | role/group boundary |
|---|---:|---:|---|
| AI / Qwen Image 2512 | 3,000 | 2,956 | four outputs per prompt group |
| AI / FLUX.2 Klein 9B | 3,000 | 2,916 | four outputs per prompt group |
| AI / Nano Banana | 3,000 | 2,957 | generated parent |
| AI / GPT Image 1 | 3,000 | 2,957 | image/prompt pair |
| AI / Nano Banana Pro | 200 | 200 | generated parent |
| AI / CommunityForensics | 2,800 | 2,800 | generator-model identity retained |
| REAL / VISION native | 3,500 | 3,497 | device/native parent |
| REAL / Forchheim FODB original | 3,851 | 3,851 | device plus 143 shared scenes |
| REAL / CSAFE S21 natural | 3,996 | 3,996 | ten physical devices/four lenses |
| **AI total** | **15,000** | **14,786** | max source share 19.998647% |
| **REAL total** | **11,347** | **11,344** | three independent collections |

No newly added CSAFE parent creates a cross-source or cross-label collision. The global overlay
retains 59 known duplicate components and excludes 20 same-label noncanonical units plus 13
within-parent rows. Detailed evidence is 1,431,190 B / SHA `45830283...78b6`; compact evidence is
`evidence/e32_eligibility_overlay.json`. The state is `eligibility_frozen_role_free`: these rows
may now feed a precommitted group-aware TRAIN/CALIBRATION split, but none is a locked test sample.

The C3 balanced role manifest selects 22,688 of those eligible parents: all 11,344 REAL and an
exact source-capped 11,344 AI subset. TRAIN has 18,154 rows and CALIBRATION 4,534. Its protected
group intersection is zero for the declared device/scene/prompt/generator boundaries. Record-list
SHA is `568e8e26...d887`; compact receipt is `evidence/e32_c3_role_manifest.json`. FODB remains
scene-disjoint rather than camera-disjoint because its complete crossed design connects all 27
cameras through the same 143 scenes; it cannot support an unseen-camera claim.

R0 derives one class-identical model input from every C3 parent: EXIF transpose, RGB, short-side
256, center-crop 224 and JPEG q90/4:4:4. All 22,688 outputs passed original/derived SHA checks and
occupy 487,845,683 logical bytes (larger allocated size on exFAT due to small-file allocation).
Receipt SHA is `2255b123...5199`; no DEVELOPMENT or LOCKED image was materialized.

Post-R0 REAL-complement audit rejected three already-local shortcuts. The REAL half of
`OwensLab__CommunityForensics-Small` is 32,912/32,912 FFHQ faces, so it cannot supply broad current
camera/content coverage. `34data__communityforensics-real` contains 8,000 JPEGs but revision
`fc9fe1b...81ce` has no dataset card, declared licence or parent-source field. The `theminji`
balanced/200K cards expose only labels/counts and no source/licence. These bytes remain outside
E32 roles. Official Community Forensics documentation says the Small release includes separately
licensed real datasets and is limited to non-commercial research; that does not establish the
unofficial repack's provenance.

### E32 post-R1a corrective authentic sources — frozen before bytes (2026-08-27)

R0 and R1a both collapsed on the same owner-camera DEVELOPMENT population, so the correction
targets authentic pipelines rather than adding more AI volume or shopping another encoder.

| source | frozen transfer | role boundary | selection reason / limitation |
|---|---:|---|---|
| CSAFE MCSIDB iPhone 14 | `iPhone14.zip`, 20,428,338,922 B, MD5 `dfc01c89b14356141f53d253b72e946c` | role-free TRAIN/CALIBRATION candidate only after archive inventory, natural-only selection, decode and decontamination | CC BY 4.0; directly adds a current Apple computational-photo pipeline, but is the same collection as S21 and cannot by itself prove source transfer |
| IPN-NFID v3 linked device articles | 960 `natural` JPEGs, 3,889,897,594 B, twelve device instances | source-held-out DEVELOPMENT only; never fit/select threshold, crop, representation or policy | CC BY 4.0; compact independent smartphone-camera stress. Only 80 natural images/device and structured landscape/portrait captures, so it is a gate rather than a broad training corpus |

The IPN umbrella article is Figshare `25201319`, version 3 (2025-03-28), linking twelve immutable
device articles. Each exposes 80 filenames containing `natural`; all selected API rows include
published byte size, download URL and MD5. Devices are iPhone SE 2020 (two physical instances),
iPhone XR, Motorola G4 Plus/G Play/G20, Samsung Galaxy A01/Note 9, Sony Xperia M4, Huawei P20 Lite/
Y9 2019 and LG L65. The CSAFE source is Figshare article `26932084`, version 1. API id, version,
licence, filename, size and checksum must match before a transfer. Both downloads retain `.partial`
state and preserve a 100 GiB disk floor; no selected image byte existed at this checkpoint.

Production metadata freeze reproduced every declared contract: 960 IPN natural JPEGs across twelve
devices / 3,889,897,594 B and the 20,428,338,922-byte CSAFE iPhone 14 archive. The detailed external
selection is 385,191 B / SHA `c807d140...1c7f`; compact Git evidence is
`evidence/e32_r1b_acquisition_selection.json`. State remains
`selection_frozen_no_selected_bytes_claimed`.

IPN selected-byte transfer and realization now pass. All 960 natural JPEGs / 3,889,897,594 B were
individually MD5-verified, decode as RGB JPEG and retain EXIF; all 960 SHA-256 values are unique.
The names bind 80 scene groups shared across all twelve devices (50 landscape + 30 portrait, twelve
captures each). Protected E30 overlap, passed E32 peer overlap and cross-scene perceptual collision
are all zero. Detailed realization is 642,208 B / SHA `f5827dce...243b`; compact evidence is
`evidence/e32_r1b_ipn_realization.json`. Role remains unscored DEVELOPMENT—not a new training pool.

CSAFE iPhone 14 transfer and safe inventory now pass. The promoted ZIP is exactly 20,428,338,922 B,
reproduces published MD5 `dfc01c89...946c` and has SHA-256 `22f04a95...8cbb9`. All 7,996 JPEG
members pass CRC/path/symlink/encryption/expansion checks: 4,000 blank + 3,996 natural across ten
physical devices and front/telephoto/ultra/wide. Detailed inventory is 1,295,576 B / SHA
`8931a535...912e`; compact evidence is `evidence/e32_r1b_csafe_iphone14_inventory.json`. No member
was selected or extracted by inventory.

The subsequent metadata-only selection freezes all 3,996 natural members and excludes all 4,000
blank images before payload extraction. Device counts are 398-400; lens counts are front 998,
telephoto 1,000, ultra 998 and wide 1,000. Detailed selection is 1,425,474 B / SHA
`88dc326e...7b74`; compact evidence is `evidence/e32_r1b_csafe_iphone14_natural_selection.json`.
Rows remain role-free and unextracted at this checkpoint.

Atomic natural extraction then completed 3,996/3,996 parents / 12,914,703,500 B. Each selected ZIP
member was rechecked against frozen size+CRC, written through a partial and SHA-256 recorded; no
blank member was extracted. Detailed receipt is 1,884,013 B / SHA `46b36e56...09de`; compact
evidence is `evidence/e32_r1b_csafe_iphone14_natural_extraction.json`. These files remain role-free
pending decode/decontamination.

The decode/decontamination audit deliberately stops role assignment on one confirmed two-image
burst: `iPhone14_5/telephoto/IMG_1290.JPG` and `IMG_1291.JPG` are byte-distinct but have equal
dHash+pHash and visibly the same composition. All 3,996 decode RGB with EXIF; unique SHA is 3,996;
protected E30/passed-peer/IPN/owner exact overlap is zero. Payload format is 3,945 MPO + 51 JPEG
despite `.JPG` names—an important container fact that the standardized JPEG model-input route must
neutralize. Detailed rejected audit is 2,638,999 B / SHA `8325aaf4...05fd`; compact evidence is
`evidence/e32_csafe-mcsidb-iphone14_realization.json`. The correction excludes both burst rows,
not an outcome-selected preferred member.

The deterministic overlay excludes both burst members and freezes 3,994 eligible, role-free iPhone
parents; source bytes remain intact. Eligible payloads are 3,943 MPO + 51 JPEG. Detailed overlay is
2,364,384 B / SHA `a71c4a06...57bf`; compact evidence is
`evidence/e32_r1b_csafe_iphone14_eligibility.json`. R1b will neutralize this container imbalance by
the same derived JPEG contract used for every earlier parent.

R1b's controlled role manifest preserves all 22,688 C3 rows/roles in their original order and
appends only 3,994 eligible iPhone parents. Total is 26,682: 11,344 AI / 15,338 REAL; TRAIN 21,349
(AI 9,081 / REAL 12,268) and CALIBRATION 5,333 (AI 2,263 / REAL 3,070). iPhone TRAIN uses eight
complete devices / 3,195 rows; CALIBRATION uses iPhone14_4 + iPhone14_8 / 799 rows. Detailed
manifest is 15,909,170 B / SHA `16deb276...750f`; records SHA `263af46b...5611`; compact evidence
is `evidence/e32_r1b_role_manifest.json`. IPN/owner remain absent.

All 3,994 appended iPhone parents were standardized through the identical R0 derived-input route;
the 22,688 old derived bytes were reused. The complete 26,682-row receipt represents 568,959,891
logical JPEG bytes; detailed receipt is 10,631,702 B / SHA `400a990d...6af8`, records SHA
`3e51f87a...1395`; compact evidence is `evidence/e32_r1b_input_receipt.json`. This removes the
observed MPO/JPEG container difference before representation extraction.

The controlled R1b experiment confirms that these valid bytes were useful for training coverage
but were not sufficient to establish authentic-source transfer. The selected CF head passed its
internal source-stratified screen, then mislabeled 249/960 IPN images (25.94% macro-device and
40.0% worst-device FP) and 144/210 owner-gallery stills. IPN and the owner gallery are therefore
consumed DEVELOPMENT populations from this point forward: they may document regressions but may
not select an R1c representation, loss, threshold, cascade or policy. The iPhone rows remain valid
TRAIN/CALIBRATION data; a new provenance-complete multi-camera REAL source is required for the next
unseen final gate. Evidence is `evidence/e32_r1b_external_development.json`.

### E33/R1c licensed calibration and robustness benchmark — frozen before bytes (2026-08-27)

NIST GenAI Image-D is retained as the future external blind authority, not a locally downloadable
test: access requires registration/data terms and its hidden trials may not be inspected or used
for tuning. NTIRE 2026 is retained only as a 2026 competitive reference because its public
validation repository currently declares no dataset licence. Public downloadability alone does not
satisfy PixelProof's provenance/licence gate.

The immediate reproducible source is **RRDataset**, the ICCV 2025 Real-World Robustness benchmark,
from the official Zenodo record `14963880` under CC BY 4.0. It contains high-impact/everyday scenes
and evaluates original files, repeated internet/social-platform transmission and physical
re-digitization. This is intentionally a robustness source rather than a claim that its two
generator families represent every 2026 commercial model.

| frozen file | exact bytes | published MD5 | PixelProof role |
|---|---:|---|---|
| `RRDataset_original_train_val.tar.gz` | 2,163,176,547 | `2f4498c3690d8f4c7a30d2e41dd34500` | role-free `R1C_CAL` candidate; may select only the threshold after archive/label/decontamination audit |
| `RRDataset_test.tar.gz` | 20,117,869,400 | `13c3ff3d61986170cc0c8cf76a35cd4b` | locked final robustness test; forbidden until the R1c-T contract is frozen |

Destination is `/Volumes/LaCie/pixelproof-datasets/e33_rrdataset/`. The detailed 1,166-byte
selection receipt has SHA-256 `ad6fc31f...3519`; compact evidence is
`evidence/e33_rrdataset_acquisition.json`. Metadata freeze downloaded zero archive bytes. Transfer
must remain resumable, preserve a 100 GiB disk floor and promote a `.partial` only after exact byte
count and MD5 verification.

The calibration archive subsequently completed at its exact 2,163,176,547 B and reproduced MD5
`2f4498c3...34500`. Safe inventory passed 3,000 images / 2,185,661,793 expanded bytes with no
other file, unsafe path or unsupported member: train is 1,250 REAL + 1,250 AI and validation is
250 REAL + 250 AI. Only the 500 validation images were atomically extracted (324,972,659 logical
bytes); exFAT AppleDouble sidecars are ignored and never become manifest rows.

The unscored `R1C_CAL` manifest contains exactly 250 REAL + 250 AI. RRDataset's validation
filenames expose seven AI scenario groups (22–93 rows) but label every REAL file only as `real_*`;
they do not expose the real upstream site, scene or camera. PixelProof therefore records REAL as
one `rrdataset_real_pool`, uses an aggregate calibration false-positive budget and makes no
multi-camera calibration claim. The detailed 124,960-byte manifest SHA-256 is
`5d575a08...b521`; compact evidence is `evidence/e33_r1c_cal_manifest.json`. No model score existed
when this allocation was frozen.

All 500 validation rows later decoded/scored successfully, so the RR validation role is now
consumed CAL/diagnostic and cannot serve a new candidate. R1c-T failed before DEVELOPMENT: AUC
0.80728, EER 0.276 and TPR@FPR=10% 0.52. At the original R1b threshold REAL FP was 82.8%; the first
REAL-safe threshold was 0.998400 with 10.0% REAL FP but only 52.0% pooled AI recall, 60.52% AI
scenario-macro recall and 26.88% worst-scenario recall. The 20.12 GB RR locked test archive remains
undownloaded and unopened.

### E34 official DDA aligned benchmark — frozen before bytes (2026-08-27)

The next data is not another unrelated AI pile. The selected candidate is the official NeurIPS
2025 **DDA-COCO** release (`Junwei-Xi/DDA-COCO`) at revision
`8c9330a3b374bcac46a8045a0e3c09ebcf7868fb`, Apache-2.0. Its single
`DDA-COCO.zip` is 4,301,452,066 B with Xet SHA-256
`8cd600779aaecef21605b07bff9ab3963a7fb9b9614a3d9a0588cd4a5e099c24`. The source provides
MS-COCO validation reals and semantically corresponding synthetic VAE reconstructions across five
alignment variants, directly targeting the semantic/frequency shortcut revealed by R1b/E33.

The first metadata plan treated the compact aligned archive as a possible pair source. Primary
official code documentation corrected that before any member was opened: DDA-COCO is an evaluation
benchmark, while `DDA-Training-Set` is the fitting source. The latter is roughly 112.97 GB in a
mandatory split ZIP (ten 10 GiB parts plus a 5.59 GB final part), so it is deferred rather than
quietly training on test data. DDA-COCO remains locked for the official pretrained DDA candidate.
Detailed metadata selection is 916 B / SHA-256
`f0bc21a7...5184`; compact evidence is `evidence/e34_dda_acquisition.json`. No image byte existed
at this preregistration.

The corresponding official model candidate is `Junwei-Xi/Dual-Data-Alignment`, revision
`4390d9023899196b437480bb6a441915ef5d816c`, Apache-2.0. Its `DDA_ckpt.pth` is 1,255,621,296 B with
Xet SHA-256 `b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e` and uses a
DINOv2-L/14 LoRA detector. It is the next compact candidate; it must be pinned and tested before
DDA-COCO is opened.
Its metadata-only E35 selection is 962 B / SHA-256 `7bdbe886...3fd9`; compact evidence is
`evidence/e35_dda_model_acquisition.json`. Freeze downloaded zero checkpoint bytes. The subsequent
resumable transfer completed at exactly 1,255,621,296 B and reproduced SHA-256
`b27a31d3...e3e`; the model can be constructed offline because this checkpoint contains all 537
base, LoRA and classifier tensors.

### E35 DDA DEVELOPMENT realization and E36 data boundary (2026-08-27)

The verified DDA checkpoint scored 1,670 previously declared DEVELOPMENT rows: 250 RRDataset REAL
+ 250 RRDataset AI, 960 IPN native-phone reals across 12 devices and the exact frozen 210-still
owner gallery. The newly added owner image `WhatsApp Image 2026-08-25 at 17.14.51.jpeg` remains
untouched as an explicit reserve and is not in these counts. Local detailed scores live outside Git
at `/Volumes/LaCie/pixelproof-datasets/e35_dda_model/development_scores.jsonl` (348,372 B,
SHA-256 `ae352ffe...83a`); compact aggregate evidence is
`evidence/e35_dda_development.json`.

At the official 0.5 cut, RR REAL FP is 6.4% and AI recall 91.2%, but IPN worst-device FP is 36.25%
and owner FP is 34.76%. All three sources are consumed DEVELOPMENT and may never calibrate a
deployable threshold. A post-hoc diagnostic shows the first all-real-safe observed boundary at
0.901156 (RR AI recall 82.4%, RR REAL FP 0.4%, IPN worst-device FP 20.0%, owner FP 13.33%); that
number and every other inspected E35 cut are permanently ineligible. Evidence is
`evidence/e35_dda_threshold_diagnostic.json`.

DDA-COCO remains an unopened benchmark. Its preserved transfer consists of an exact 212,860,928 B
prefix plus four non-overlapping range files totaling 4,039,521,881 B: 4,252,382,809 of
4,301,452,066 B (98.86%). The missing 49,069,257 B is intentionally not fetched until a new E36 CAL
passes; no ZIP member has been listed, extracted, decoded or trained on.

E36 requires new, role-separated bytes rather than more volume from old sources: target 600 native
authentic CAL parents from >=6 unseen capture pipelines and 600 clean modern-generator CAL parents
from >=6 pinned production families. A separate locked FINAL minimum is 160 authentic parents from
four unseen device/session pipelines plus 240 AI parents from six held-out model/version cells.
Derived degradation variants stay grouped by parent. Exact source/licence/API receipts and overlap
checks must be appended here before the first acquisition byte; no named source is approved merely
because it is downloadable.

### E36-A selected CAL/FINAL sources — metadata decision before image bytes (2026-08-27)

REAL source: Zenodo `18136670`, `sns-homogenization-forensics-dataset` v1.0.0, publication
2026-02-03, record-level CC BY 4.0. The source declares nine device ZIPs and three parent-linked
conditions: `view_000` normal/unprocessed, `view_001` QQ and `view_002` Sina Weibo. PixelProof CAL
uses archives 001/002/003/005/009 only; FINAL reserves 004/006/007/008 before any score. CAL selects
at most 100 normal originals per device and requires at least 80; FINAL uses at most 100 per device.
Social copies remain children of the same parent and never inflate sample size.

AI source: Hugging Face `Qwen/Qwen-Image-Bench` revision
`d2493deb153b020cf169c7e3f57d15e4dd697038`, dataset-card Apache-2.0. The pinned repository exposes
18 generator directories with 1,000 prompt-aligned outputs each. CAL is six families × prompt ids
101–200: `gpt-image-2`, `nano-banana-2.0`, `Seedream-5.0`, `Qwen-Image-2.0-pro`, `FLUX.2_max` and
`GLM-Image`. FINAL is six different families × ids 1–40: `GPT-Image-1.5`, `nano-banana-pro`,
`Imagen-4.0-Ultra`, `HunyuanImage-3.0`, `FLUX.2-pro` and `Seedream-4.5`. This makes FINAL generator-
cell-held-out, not merely new prompts from families that calibrated the threshold.

Two tempting alternatives are explicitly rejected. SCIMD-17/Zenodo `17317613` is CC BY 4.0 and
small, but the publisher states every image was preprocessed to 224×224, so it cannot represent
native phone output or owner-gallery behaviour. Remaining CSAFE archives are native and licensed,
but share the source collection/scenes already used through S21 and iPhone14; spending another
18–29 GB would add weaker independence than the selected 2026 source. No image byte from either
rejected source is authorized for E36.
