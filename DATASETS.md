# Datasets — inventory and assignment

What we hold, what each set is good for, and which module it feeds. Replaces the
former `STATUS.md`, whose content is now covered in more detail by `HISTORY.md` §2b.

**The rule that governs everything here** (`HISTORY.md` §1b): a dataset flaw is a
*usage condition*, not a disqualification. A shortcut only exists if the model can
perceive it. Whole-image training can see image dimensions; tile training cannot,
because every tile arrives at 128×128 regardless of where it came from.

---

## Current acquisition policy — 2026-09-14 home-network update

Dataset downloads are authorized again. This record is explicitly reactivated by the
user alongside PLAN/HISTORY/EXPERIMENTS. Every acquisition must identify what and why,
source/revision/licence, intended role, planned/completed counts and bytes, integrity,
local location, failures and next step. Quarantined downloads are not admitted TRAIN,
CAL or independent tests. Never publish personal gallery pixels or individual metadata.
Older sections retain their original dates/status; later receipts supersede them.

### E98 source research and role audit — in progress

Purpose: prepare a complementary reviewer without reusing consumed calibration as new
evidence. Current E92 TRAIN contains12,269 parents (7,674 REAL/4,595 AI), including
previously reassigned E36/RR lineage. Inspect old manifests and overlap/role history
before selecting another fit or a new CAL pool. No new model score is part of this audit.

Sources investigated before pixel acquisition:

| Source | Why considered | Current decision / limitation |
|---|---|---|
| [SIDL, AAAI2025](https://sidl-benchmark.github.io/) | iPhone12 Pro clean/dirty-lens capture pairs across lighting; potential REAL processing robustness | Official research/education-only terms; no redistribution. Prior metadata receipt already records1605 RAW entries/253 scenes versus the webpage300 scenes; release/split reconciliation remains unresolved. No new pixels downloaded. A single camera cannot establish broad generalization. |
| [CSAFE MCSIDB](https://iastate.figshare.com/articles/dataset/CSAFE_Multi-camera_Smartphone_Image_Database/26932084) | Additional camera pipelines with explicit capture provenance | Existing S21/iPhone14 are already used; New archive metadata fetched (6,622B); iPhone12 download HEAD returned403 and institutional endpoint a WAF challenge. No image transfer. Same collection cannot establish independent-source transfer. |
| [LAION-Mobile](https://huggingface.co/datasets/laionmobile/laion-mobile) | Large smartphone metadata collection | Metadata-only card; CC BY4 applies to metadata, underlying image rights differ. EXIF alone cannot certify REAL. No bulk pixel acquisition selected. |
| [GenSyn10](https://arxiv.org/abs/2607.16283) | Recent FLUX.2/Hunyuan/Qwen families |32x32 images are below current224-pixel input floor; not selected for this native-photo detector. |

No new dataset body is claimed downloaded at this registration point. Research pages
and manifests are metadata, not image counts. The raw-score UI remains uncalibrated.

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

### E49 comprehensive-final registry — frozen intent before payload transfer (2026-09-04)

E49 is the last independent promotion test for Module 1. Its target is **2,000 unique parents**,
exactly 1,000 REAL and 1,000 AI. No third-party image is committed to Git; payloads belong under
`/Volumes/LaCie/pixelproof-datasets/e49/`. The 4 GiB network ceiling excludes the already-local
AIGC archive and includes every newly transferred image byte.

| Source | Pinned identity / licence | Frozen E49 role and quota | Why it is here | Boundary |
|---|---|---:|---|---|
| Wikimedia Commons camera-category originals | Commons API page/revision ids and per-file licence/attribution will be frozen by the binder | FINAL REAL, 1,000: 100 each across ten declared current phone/camera categories | Independent public publisher; original-upload transport, multiple devices and uploaders; directly stresses the false-AI failure on real photography | JPEG only; camera category and EXIF must agree; uploader capped; “original upload” is not claimed to be untouched sensor-native data |
| `datapointai/text-2-image-human-preferences-2m` | revision `e1d8719a2d521eac6c62ee84f329afc2c03ec928`; metadata CC BY 4.0; image-output rights remain provider-specific | FINAL AI, 800: 160 each GPT Image 2 / Nano Banana 2 / Seedream 5 Pro / FLUX 2 / Ideogram 4 | August-2026 independent collection, fixed seed, 500 prompts in ten categories, current provider breadth and full-resolution output table | User submitted the contact-sharing request on 2026-09-04; authors' review is pending; evaluation/research only; no post-score source substitution |
| `TheKernel01/AIGC-Detection-Benchmark` | revision `c91d9024a5a77ef06e2ec681b53f9caf08675663`; Apache-2.0; local 60-shard test release | FINAL AI, 200 StyleGAN2 (generator code 14) | Adds a GAN family from a source not used for E43-S training or E48/E50 selection; prevents “current diffusion only” success | Test-only; select bytes without reading scores; source-native format preserved; exact/dHash protected-role audit required |

The local AIGC copy now reproduces that pinned Hub revision and all 60 shards. A metadata-only scan
read exactly the `label` and `generator` columns: 125,026 total rows, of which 1,997 are label=AI and
generator code 14 (StyleGAN2). The deterministic 240-row reserve identity digest is
`15e5c1315d1411c1c106dc457166b2b097dd3753bcb357e35178f279a95cc731`. No image column or model
score was read, and no network byte was required for this source.

The first Commons feasibility scan occurred before any source contract or image transfer. Nine
declared categories can fill an uploader-capped reserve, but Fujifilm X-T5 yields only 66 eligible
rows after the contributor cap and cannot honestly supply its 100-row target. It is therefore
replaced pre-binding by Nikon Z 8, which supplies 110 selected rows from 25 uploaders. Commons
headroom is reduced from 20% to 10% to keep the unchanged 1,000-REAL target within the 4 GiB network
stop. API responses now request only the two licence fields actually used. No REAL image was
downloaded, decoded or scored during this metadata audit.

The first complete open-component bind froze 1,100 Commons identities plus 240 local StyleGAN2
coordinates, but correctly stopped before transfer: the Commons reserve alone totals 4,140,590,955
bytes and would leave essentially no room for the still-missing 800 Datapoint images under the
total 4 GiB ceiling. Contract SHA-256 `c6f2cfb0...f794` is retained as a rejected, zero-image,
zero-score feasibility record. The cached eligible metadata shows that a predeclared 4 MiB maximum
can still fill 110/110 rows in every device while reducing the Commons reserve to about 2.52 GiB;
that rule must be frozen in a successor contract before any transfer.

The V2 successor is now frozen without overwriting V1. Its <=4 MiB rule binds all 1,100 Commons
reserves at 2,706,581,778 bytes and leaves 1,588,385,518 bytes of the global ceiling for the 800
Datapoint targets. It simultaneously binds the 240 already-local StyleGAN2 coordinates. Contract
SHA-256 is `1d4e184c27cb87cf832045a23b6966f382673c3bcd8342a900c07130bd9182aa` and combined reserve-
identity SHA-256 is `31c0e420b694b1774382c1ab299d6115008656ac7fc1e3c5b0de7deececde171`.
No image transfer or detector score occurred; Commons download must still wait until Datapoint's
exact selected byte total proves the complete E49 remains inside the global stop.

After E49-C supplied an exact 241,736,938-byte AI component, the combined network expectation fit
the unchanged ceiling and Commons transfer proceeded. All 1,100/1,100 bound originals and
2,706,581,778/2,706,581,778 bytes now validate, exactly 110 per device. The received camera
transport contains 861 JPEG and 239 Apple MPO files; 1,074 preserve EXIF make/model and orientation
is retained without rewriting originals. External receipt SHA-256 is
`2511f0ad0ad3f22e72ab5bf04da69fed0e9efac2bb4768d5365cf734fb5a7e04`. No detector score exists;
these rows remain reserves until device-evidence and protected-overlap checks select 100 per device.

The pre-realization EXIF audit found one category mismatch: one Nikon Z 8 reserve reports Nikon D70.
It is excluded rather than trusted or relabelled. Known make/model spellings and Apple hardware codes
are normalized; 1,073 rows have matching EXIF device evidence and 26 rows with no make/model remain
explicitly category-only candidates. Selection still follows the prebound rank and must fill 100
clean rows per device before the component becomes final.

The realization passed all 1,100 receipt rows. It excluded one protected dHash overlap and the one
Nikon-D70 mismatch, then froze the first 100 clean rank-ordered parents in every device category.
The selected 1,000 REAL parents contain 979 matching EXIF device records and 21 explicit category-
only/no-make-model records; none is silently described as sensor-native. Their originals plus 1,000
deterministic social-Q75 children form a 2,000-observation component. External manifest SHA-256 is
`657be9bb6e2632f5b00f2f5a1f37b36640d186b53044cdb2cdf0697cb9208e7b`; model scores are zero.

The three unscored component manifests now assemble into the complete E49-C final: 1,000 Commons
REAL, 800 OpenFake modern-AI and 200 local StyleGAN2 parents. Every parent has its received original
and deterministic social-Q75 child, so the test contains 4,000 observations but an effective
N=2,000 parent units. All SHA-256 values, display geometries, paired identities and sixteen source
quotas validate. Original formats are 1,585 JPEG, 213 MPO and 202 PNG; all Q75 children are JPEG.
External manifest SHA-256 is `9744a9d2385ef2f105b7a132bfee76a7099d280ac9d075d81220f572429c5909`.
The test remains unscored and immutable; it is never a training or threshold-selection source.

E49-C has now been scored once and is a **consumed failed FINAL**. All 4,000 observations completed,
but the frozen candidate passed only 11/20 gates because new-camera REAL false-AI was 39.10% on
publisher originals and 49.00% after Q75. AI recall remained 94.30%/95.50%. These 2,000 parents,
their children and all 100 downloaded Commons reserve identities are prohibited from future
TRAIN/CAL/threshold selection and cannot be repackaged as a second independent final. Detailed
result SHA-256 is `10fc0649a68d31c57f815949e3c8d52f2d2bdfaeaadc9422c0ae83c32d525573`.

The already-local StyleGAN2 reserve was then realized without network or detector access. All
240/240 frozen Parquet coordinates reproduce label=AI/generator=14 and decode successfully; exact
and dHash comparison with 15 protected-role manifests finds zero overlap. The first 200 clean rows
are frozen in a 152,210-byte manifest at SHA-256
`150ed35481bc0ae7653ff34eae879a227c1e359e4ab9ca772c681bbdef22ec99`; selected payloads total
20,111,615 bytes. Every selected image is publisher-native 256x256 PNG. That homogeneous geometry
is an explicit shortcut warning: StyleGAN2 contributes per-source GAN recall, never independent
proof of balanced detection, and must remain accompanied by the five modern AI families.

Every parent receives two paired conditions: `publisher_original` and deterministic `social_q75`
(EXIF transpose, RGB, long side <=1080, JPEG q75, 4:2:0, metadata stripped). The conditions are
reported separately and never counted as 4,000 independent samples. A source/label-stratified,
parent-level 10,000-bootstrap contract protects confidence intervals from derived-copy inflation.

Rejected alternatives are part of the registry, not forgotten research. SCIMD-17 is only 177 MB,
Apache-2.0 and genuinely camera-captured, but the publisher resized all 17,000 images to 224 x 224;
it does not answer the native/gallery-real question and would leak class through scale. ImageBench
has excellent 2026 coverage, but its current canonical licence reserves reuse of generated images
without written permission. Qwen Image Bench and every TrueFake/E30/E36/E45/E48/E50 source are
already protected, consumed or publisher-overlapping and cannot be relabelled as fresh E49 proof.

#### E49-D1 open-access diagnostic source — frozen intent before image transfer

`fge-auto/dotting-test` is ungated and CC BY 4.0 at revision
`0bcc6877c7d23f4e615b5470f06b1c00e7db7311`. Its 384 MB repository contains 8,396 successful
generated outputs across 40 models; each of the five selected 2026 families has 210 available rows:
GPT Image 2, Nano Banana 2, FLUX.2 Pro, Ideogram 4 and Seedream 5.0 Lite. E49-D1 will bind 192 per
model (160 target +32 reserve), at most 960 WebP files /512 MiB.

This dataset is useful because it provides immediate, attribution-clear coverage of the same modern
families sought from Datapoint. It is deliberately narrow: all images test Turkish glyphs in signs
or text-centric scenes. Therefore it is an AI-recall/transport diagnostic only, never a balanced
final and never evidence of REAL-photo safety. The full repository, VLM labels and unrelated model
rows are not required. Attribution must credit Fırat Gelbal and Dotting Test; downstream users must
also review the applicable upstream provider-output terms.

#### E49-B ungated fallback candidate — metadata qualification only

`ComplexDataLab/OpenFake` is an ungated CC-BY-NC-4.0 dataset. E49-B pins revision
`3fd1109dc3258874243fa31c5bda9ee24260163b`, configuration `core`, split `test` and its 91,398 rows.
The complete repository is about 3.44 TB and the 13 test Parquets are roughly 5 GB each, so neither
is an acceptable acquisition unit. The official Hugging Face Dataset Viewer `/rows` API can expose
individual revision-bound examples without transferring a whole shard.

The Viewer successfully cached the first 51,900 ordered rows, then repeated 429/502/503 responses
made it unsuitable for a dependable full population audit. Before any detector access or complete
population result, the transport rule was amended to permit exact-revision HTTP range projection of
only `label`, `model`, `type` and `release_date` from the 13 source Parquets. This is not a shard
download: every returned metadata byte is counted and neither the nested image payload nor prompt
column may be requested. The original row order, five cells, quotas, hash namespace and first-
complete-page stop rule remain unchanged.

The candidate cells are the exact publisher model strings `gpt-image-2`, `nano-banana-pro`,
`seedream-v5.0`, `flux.2-klein-9b` and `midjourney-7`. A deterministic metadata-prefix scan will
freeze 160 target +32 reserve identities per model before image or detector access. Prompts and
expiring Viewer URLs are deliberately excluded from the cache and contract. The dataset card notes
a prompt-mapping defect for five older generators; E49-B does not consume prompts, and the defect
does not affect image labels or model names according to the publisher.

OpenFake is valuable here because it supplies a permissively accessible, publisher-separated pool
of recent generator families while Datapoint remains under manual review. It is not yet downloaded,
approved as final evidence or training data. Viewer-delivered assets must be described honestly as
received JPEG transport rather than guaranteed original Parquet bytes, and the final byte sum must
still fit the existing 4 GiB global network ceiling beside the frozen Commons reserve.

E49-B source qualification is now closed as **rejected before selection**. Across all 91,398
`core/test` rows, eligible non-video fake populations are: GPT Image 2 470, Nano Banana Pro 60,
Seedream v5.0 372, FLUX.2 Klein 9B 8,093 and Midjourney 7 3,586. Nano cannot supply the frozen
192-row reserve, so no identity, asset, image or score was selected. The measured projection used
2,597,624 bytes across 650 exact ranges and cross-validated 52,600 cached Viewer rows, rather than
downloading 67,649,942,401 bytes of source Parquets. The Hub card metadata says CC-BY-NC-4.0 while
README prose also mentions CC-BY-SA-4.0 plus non-commercial terms for proprietary subsets; this
project keeps the stricter non-commercial boundary and must preserve attribution.

#### E49-C capacity-repaired OpenFake successor — pre-identity-selection

E49-C keeps the identical pinned OpenFake source and replaces only the underfilled Nano Banana Pro
cell with exact model string `z-image-turbo`. This is a metadata-capacity repair, not score-driven
source shopping: no OpenFake image or detector score exists. The already-validated continuous
52,600-row Viewer prefix contains 270 GPT Image 2, 6,876 Z-Image Turbo, 222 Seedream v5.0, 4,624
FLUX.2 Klein 9B and 2,045 Midjourney 7 eligible non-video fake rows, so all five preregistered
192-row reserves are feasible without another metadata transfer.

Selection will use the distinct `E49_C_OPENFAKE_V1` namespace and stop at the first complete
100-row page where every cell has 192 rows. Only after all 960 identities are committed may fresh
revision-bound Viewer asset URLs, byte sizes and geometry be accessed. This component remains
non-training final evidence and must be paired with the frozen Commons REAL and local StyleGAN2
components; it cannot independently establish balanced detection.

The E49-C identity contract is now frozen at the first qualifying page boundary, row 46,600. At
that point the eligible populations are GPT Image 2 236, Z-Image Turbo 6,111, Seedream v5.0 192,
FLUX.2 Klein 9B 4,068 and Midjourney 7 1,791. Hash ranking selected exactly 192 identities per cell.
The 415,653-byte external contract hashes to
`0abae56af862c9b402ef5ef594a21181cbbb7f72ba7495a491b0389bfdfcd702`; reserve identity SHA-256 is
`f9f7bf74958ced020fcc2d38729640fc3bb7cb14dc64c691be0b827f592069ec`. It reused cached metadata,
so new metadata/image/model-score bytes are all zero.

The separate no-body asset bind also passed for all 960 identities. Official Viewer pages and
selected-row fallback resolved fresh revision-bound paths; S3 reports either JPEG or generic binary
transport, so actual image format remains a mandatory decode check after transfer. HEAD metadata
binds 241,736,938 OpenFake bytes. Added to the exact 2,706,581,778-byte Commons reserve, the final
network expectation is 2,948,318,716 bytes with 1,346,648,580 bytes of 4 GiB headroom. Contract
SHA-256 is `7b71449e0e7d9ea22973f021af2d4ec49cc395e3fffe6816ffc123274a571415`;
no signed URL, image body or model score was stored.

The first body-transfer pass then proved that a Viewer `.jpg` path is not a format guarantee:
Seedream row 8,770 is a valid 2,048 x 2,048 RGBA PNG behind generic binary MIME. It is not removed or
converted. Received bytes remain canonical; only decoded JPEG, PNG or WebP are admitted, and actual
format/geometry distribution must be reported by source as a potential shortcut. The 201 previously
validated payloads remain resumable and no detector was loaded.

The exact transfer is now complete under
`/Volumes/LaCie/pixelproof-datasets/e49/openfake/payloads`: 960/960 contracted payloads and
241,736,938/241,736,938 network bytes, with 192 files per generator. Decode coverage is 100%; 958
files are JPEG and two are PNG. The external 411,483-byte receipt SHA-256 is
`4dfb942c215bbe15e15aea81e9a6c6873670a0bb5019c1c64f9672fe34b326c2`. macOS `._*` sidecars on
the external volume are filesystem metadata and are excluded from dataset file/byte counts. No
signed Viewer URL is retained, no row has a model score, and the reserve is not yet the final test
manifest until protected-overlap checks and deterministic Q75 pairing pass.

The independent realization decoded all 960 payloads with zero failure and found zero protected-role
overlap. It excluded 26 exact+dHash duplicate Seedream identities before selection; the remaining
headroom still fills all five 160-parent quotas. The final OpenFake component is 800 parents and
1,600 paired observations (publisher original plus deterministic social-Q75), with selected-original
formats 798 JPEG and two PNG. Its external 1,505,891-byte manifest SHA-256 is
`38048803d01b6cb607c491d2391a7f0031e5c1010d7594bd44c767127e637442`. This remains unscored,
non-training final evidence pending the matching REAL component and complete E49 assembly.

#### Gated-source approval checkpoint — metadata only

Authenticated access is now open for Datapoint revision
`e1d8719a2d521eac6c62ee84f329afc2c03ec928` (manual gate, CC-BY-4.0, provider image terms) and
Hugging Face `dkarageo/itw-sm` revision `3060094fb576669927134193de3f517d7e64af86` (manual gate,
research terms). ITW-SM is equivalent in scientific role to the official MediaEval archive already
consumed in E45, so it cannot be reused as independent evidence.

Datapoint exposes 30 models, 14,952 full-resolution generated images and 500 controlled prompts.
Its full pinned inventory is 158 files /34,856,922,710 bytes. The audit downloaded only
`models.parquet` (4,336 bytes), `responses/test.parquet` (1,579,770 bytes) and `prompts.parquet`
(153,603 bytes): 1,737,709 metadata bytes total. No image Parquet or image body was transferred.
The roster confirms GPT Image 2, Nano Banana 2, Seedream 5.0 Pro, FLUX.2 Max and Ideogram 4.0, but
this is a preference benchmark with shared controlled prompts; OpenFake `core/test` is the more
direct OOD detector final and is already identity/byte-frozen. Datapoint remains unscored,
non-training post-final reserve rather than replacing E49-C after selection.

#### E51 REAL-source audit — metadata only, no download authorization

The public Kaggle v2 inventory for `goyalpuneet/sci30iitrpr` contains 9,940 files, of which exactly
9,937 are JPEG images: 5,287 Random and 4,650 Similar, grouped under 30 normalized device ids. Image
payloads total 35,592,810,773 bytes. Kaggle declares CC-BY-NC-ND-4.0 and the authors limit use to
research/education. Because files are addressable individually, a device-disjoint TRAIN/CAL slice
can be frozen without mirroring 35.6 GB; the same publisher is forbidden from E51 DEVELOPMENT.
The complete 921,570-byte external metadata inventory hashes to `fe070411...72c1e`.

Zenodo record `17317613` /DOI `10.5281/zenodo.17317613` exposes SCIMD-17 under CC-BY-4.0: a
174,438,734-byte archive (`md5:37da574c9e8d9c0fd3a7c9bedc5d72a6`) claiming about 17,000 images
from 17 phone models. Every image was uniformly resized to 224x224, so it is eligible only as an
auxiliary resized-real TRAIN hard negative—not native-camera CAL, DEVELOPMENT or final evidence.

RAISE is a legitimate non-commercial research source with 8,156 camera-native RAW images, but its
three cameras and ~350 GB full transfer make it a poor primary route. The official Dresden image
host is unavailable; a third-party mirror's CC0 label is not accepted as a replacement. IMAGINE
claims 2,816 images/60 cameras/~26 GB but the Python client cannot verify its publisher TLS chain and
the page exposes no explicit licence. SOCRatES requires a signed agreement and emailed password.
Therefore no image payload is authorized by this partial audit. Compact evidence SHA-256 is
`9ddab57a5d478102b9b3314f4258f548a98c743377150e302355dc206fdc4883`.

#### E51 frozen TRAIN/CAL/DEVELOPMENT route — identities only

The role contract was frozen before an image transfer or detector score. Its 2,418,484-byte external
JSON hashes to `975e8164477c7234292ba87449007f0ee4c8b65eb582f25a8b0d81140ec315e4`.
Historical legitimate TRAIN parents remain the base; SCIMD-17 may add only its 224x224 resized REAL
images as auxiliary hard negatives.

SCMI30 supplies CAL only: exactly 20 Random and 20 Similar JPEGs from each of 30 normalized devices,
for 1,200 parents /4,247,339,334 expected bytes. The frozen selection identity hash is
`93181852...c1a`. No SCMI30 row may enter TRAIN, DEVELOPMENT or E52 final.

The IEEE Signal Processing Cup 2018 camera-identification competition supplies REAL DEVELOPMENT.
The complete public inventory is 5,391 files /11,447,649,387 bytes; only all 2,640 hidden-camera test
TIFFs are selected, split evenly into 1,320 unaltered and 1,320 publisher-postprocessed rows
(837,665,909 bytes). Its external 513,553-byte inventory hashes to `fdb4fdee...c41b`. Kaggle returns
HTTP 403 for payload until the user accepts the competition rules; the project does not accept legal
terms on the user's behalf and does not substitute training labels for the hidden test cameras.

The user accepted those rules and the transfer is complete. All 2,640 selected files decode as the
publisher's 512x512 RGB PNG bodies despite their `.tif` names; 1,320 are unaltered and 1,320 are
postprocessed. Exact expanded bytes are 837,665,909 and selected compressed ZIP ranges total
836,795,134 bytes. The detailed external 1,621,751-byte receipt hashes to
`09188d4966a19f70a2f9ed34dab052f2d5a9b69f8819b6578fe5421e548b3794`; payload identity hash
`fc3657dd...fb05` reproduces the contract. No training image, signed URL or model score was retained.

Datapoint supplies AI DEVELOPMENT only at pinned revision `e1d8719a...c928`: FLUX.2 Max, Nano
Banana 2, GPT Image 2 high, Ideogram 4.0 quality and Seedream 5.0 Pro. Seven predeclared image
Parquets total 3,220,281,593 bytes. To remove a model/content confound, the reserve freezes the exact
same 23 prompt identities in each of eight categories for every generator: 184/model, 920 rows total,
of which realization may keep 20/category/model =800 clean parents. Bound selected image bodies total
654,005,247 bytes; reserve identity SHA-256 is `43103fac...e22`. This source is now consumed for E51
DEVELOPMENT and cannot become E52 final. Every role still requires decode, exact/perceptual overlap
and original/Q75 pairing checks before admission.

All seven Datapoint source Parquets are now present on LaCie and reproduce exactly 3,220,281,593
bytes. Detailed external receipt SHA-256 is
`18b8326a7972d19094b48ebc1e67d9ece8114ae36baeb258575f803f3a90bad1`. This is a transport
checkpoint only: the image columns were not opened, the 920 reserve rows were not decoded or
selected down to 800, and detector-score count remains zero.

The source contract is now frozen before transfer: 960 exact WebP paths total 23,936,830 bytes,
192 per model. Contract SHA-256 is `170f70db128becfe9986dffc6a3e150ec11b182d51ee496dc40ac530204eed36`;
reserve-identity SHA-256 is `9637626d7ddae037a82c15f1582458beecee1904a27bf59038727dbde85bbf5a`.
Four successful rows omit declared width/height, so missing metadata geometry is recorded as unknown
and must be recovered from the decoded WebP rather than treated as corruption. Downloaded image
bytes and detector scores remain zero at this checkpoint.

Transfer subsequently completed for all 960 bound WebP files /23,936,830 bytes, exactly 192 per
model. Every payload size and SHA-256 matches the pinned Hub inventory. The first validation pass
stopped before receipt because LaCie/exFAT created macOS `._*` AppleDouble sidecars; the validator
now ignores only those and `.cache` metadata while continuing to reject any unexpected image/file.
At that transfer checkpoint no target row had been selected, no image had been scored and the
160-per-model decode/decontamination freeze was still pending.

The score-blind freeze then decoded all 960 reserve files with zero failures and compared exact and
perceptual hashes against 15 protected-role manifests. Six candidate identities were excluded by
the predeclared overlap/duplicate rules; the 32-row headroom still filled every quota. The immutable
diagnostic now contains 800 AI parents, exactly 160 per selected generator, and 1,600 paired
observations: publisher WebP plus deterministic long-side-1080 JPEG-Q75. Its external manifest is
1,552,366 bytes /SHA-256 `048572a41b47b65d3d09bd39bee45a40745a40c2e15c50444b41f8384fdccc9`.
No detector score or training row exists at this checkpoint.

The frozen diagnostic was subsequently consumed once by exact E43-S. It passed all six declared
AI-only checks: pooled recall is 97.38% on publisher WebP and 95.88% after social-Q75; the worst
model is GPT Image 2 at 91.25%/86.88%. The paired transform costs 1.50 recall points. These results
confirm that the selected rows are useful current-generator stress evidence, but the dataset remains
forbidden from training and cannot measure REAL false accusations, AUC or balanced accuracy.

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

**NIST access check, 2026-09-02.** The official GenAI Image portal remains online and offers
participant authentication only through Login.gov. Its currently published Image-D schedule lists
D-Testset-3 release on 2026-02-23, output deadline on 2026-04-03 and results on 2026-04-10; therefore
the completed round cannot be assumed open to a new submission. PixelProof has opened the official
registration route as Plan B but has not authenticated, accepted a data agreement, obtained a team
ID, downloaded a byte or created a NIST score. After user-controlled login, only an explicitly
available late/new-round Image-D route may be bound as an untouched external evaluation.

After successful Login.gov authentication, the participant dashboard exposed a stronger access
constraint: individuals may participate only for a legally registered/incorporated organization;
foreign organizations may request participation and may need NIST IAAO approval. The account has
no associated organization `site`. Profile completion asks for country, full name, affiliation and
academic/government/industry type, after which site creation/joining, track registration and licence
upload would follow. PixelProof stopped before transmitting affiliation data or creating a site.
The user authorizes at most 4 GB of NIST transfer if access is eventually granted; current NIST
payload and score counts remain exactly zero.

### E43 Plan C — untouched DDA-COCO reassigned to the frozen E43 candidate (2026-09-02)

The NIST organization gate and ITW-SM manual review motivate an immediate open benchmark, not a
lower-quality replacement. DDA-COCO remains the official NeurIPS 2025 evaluation release from
`Junwei-Xi/DDA-COCO`, immutable revision `8c9330a3b374bcac46a8045a0e3c09ebcf7868fb`,
Apache-2.0. Its 4,301,452,066-byte `DDA-COCO.zip` is bound to SHA-256
`8cd600779aaecef21605b07bff9ab3963a7fb9b9614a3d9a0588cd4a5e099c24`. It uses MS-COCO
validation reals and five content/frequency-aligned VAE reconstruction variants, making it a test
of shortcut resistance rather than social-media transfer.

No E43 fit, calibration or DEVELOPMENT step accessed an archive member. The LaCie staging area now
contains one 212,860,928-byte prefix and four disjoint range parts of 1,022,147,785,
1,022,147,785, 1,022,147,784 and 1,022,147,784 bytes. They sum exactly to the official archive
size, correcting the earlier intermediate note that 49,069,257 bytes were still missing. This is
not yet a completed dataset claim: the pieces remain unassembled, final SHA/ZIP/CRC verification is
pending, and no member name or pixel has been opened. Expected new transfer is zero; a failed whole-
file hash stops rather than triggering an unplanned download. E43 binding is
`evidence/e43_dda_coco_contract.json`.

Assembly completed without a network request and reproduced the official archive SHA-256. Safe
ZIP and full CRC validation passed: 29,982 members, 29,969 image/files and 4,298,688,287 expanded
bytes, all under `DDA-COCO`. The observed release contains six synthetic folders rather than the
card-level five-subset description: 5,000 each for `sd-vae-ft-ema`, `sd-vae-ft-mse`, `sdxl-vae`
and `stable-diffusion-2-1`; 4,998 for `stable-diffusion-3.5-large`; 4,971 for `FLUX.1`. No original
REAL folder is bundled.

The paired REAL companion is therefore the official COCO 2017 validation archive from the COCO S3
bucket, exactly 815,585,330 bytes /5,000 JPEGs. Before transfer, E43 binds URL
`https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip`, observed Last-Modified
`2018-07-11`, multipart ETag `d366be60d3dc737327160d62453e3973-98` and exact schema
`val2017/<12-digit>.jpg`. Because COCO does not publish a cryptographic digest on this surface, the
first verified download will compute and freeze SHA-256 before pixels or scores. Only parent IDs
present in REAL and all six DDA folders may enter the score-blind candidate manifest.

That transfer completed once: 815,585,330 network bytes, SHA-256
`4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`. All 5,000 members match
the declared JPEG schema and pass full ZIP CRC; expanded bytes are 814,705,164. Cross-archive
structure leaves 4,969 complete seven-view parents /34,783 rows. These are inventory facts, not
decoded samples or model results. DDA-COCO's archive is Apache-2.0; the companion COCO images retain
their individual source/Flickr licence terms and are research evaluation material, not relicensed
by this project.

The score-blind pixel audit decoded all 34,783 complete-group candidate rows. Nineteen exact dHash
hits against 17 protected manifests touched four parent IDs; the protocol therefore excluded all
28 views from those parents. The resulting frozen test manifest contains 4,965 parents and exactly
4,965 rows in each of REAL plus six synthetic conditions (34,755 rows /5,080,919,889 image bytes).
It has zero within-pool exact groups, zero cross-label exact groups and zero cross-parent dHash
diagnostics. Detailed manifest SHA-256 is `e663d679f86ba69a545659203e11528d8998c9a362198a19f5f269a1ef97a3db`;
it is unscored and cannot be changed after seeing model output.

The frozen DDA-COCO manifest has now been scored exactly once and is scientifically consumed. All
34,755 rows completed; score-stream SHA-256 is
`1eefbdb7111154c408f08f84cfe155a0697715a974c965b4ea19d938671642dd`. E43-S failed the aligned
benchmark (AUC 0.54178, balanced accuracy 0.51114, REAL FP 14.44%, AI macro recall 16.67%). These
rows may support diagnosis and future DEVELOPMENT comparisons, but never a second independent
final claim or test-informed row/threshold repair. Any E44 training population must use different
COCO parents/assets and remain parent- and generator-separated from this consumed snapshot.

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

### E41 external-proof sources — frozen before bytes (2026-08-28)

E41 is already frozen at artifact SHA-256
`9bcc021e74b617ee48cf297bd384a8dbe946240ec04822323af1e7c3fe63ab65` and threshold
`0.6195540428161622`. The next sources therefore evaluate this exact candidate; they cannot select
another threshold, crop, model or ensemble.

**B-Free viral-image stress.** The official GRIP-UNINA checkout is pinned at Git revision
`c6a9f898782fb466b29af01f21960b67415afb0e`. Its 260,491-byte
`viral_images_dataset/BFree_viral_images.csv` has SHA-256
`3c727c4f8990ca91e129c97842fbf3c997b25fa6430fc316ffccc756f2373fc8` and declares 1,111 URL
rows: 361 REAL and 750 FAKE versions derived from only 34 source events, balanced as 17 events per
class. Each row supplies filename, label, source id, post time, dimensions, MD5 and source URL.
The latest declared post is 2024-03-29, so this source measures web propagation/robustness rather
than 2026-generator coverage. Use is informational/nonprofit under the included 1,843-byte GRIP
licence (SHA-256 `cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27`);
third-party image rights and dead URLs remain limitations. Destination is
`/Volumes/LaCie/pixelproof-datasets/e42_external/bfree_viral/`. No aggregate byte count is
published, so exact per-row MD5, URL status, decode and source-event grouping are mandatory.

The first complete URL pass verified 811/1,111 declared versions (72.9973% coverage), 162,894,149
bytes: 278 REAL and 533 FAKE. Crucially, all 17 REAL and all 17 FAKE source events retain at least
one verified child, so event-level coverage is 34/34 even though 191 URLs are dead/unreachable and
109 now return bytes that fail the authors' published MD5. Changed bytes were discarded rather
than accepted under their labels. The 913,506-byte detailed acquisition manifest has SHA-256
`e95f514942654107d60b244c8ca47e50da09f1bf49c67c5f5976bebbe0bb221d`; compact evidence is
`evidence/e42_bfree_acquisition.json`. No model was loaded and no score exists at this checkpoint.

The corrected LaCie-root audit compared every verified child against 14 earlier E32/E33/E36/E39
role manifests. It found zero exact/dHash overlap with earlier roles and zero exact/dHash collision
across distinct B-Free events; no source event or row was removed. The frozen unscored manifest
therefore contains all 811 verified rows and all 34 parent events. Its detailed 544,189-byte JSON
has SHA-256 `338a2f2b2135a4bbfcb8ce0ceef7da5d8cbe2a5b1ffbe745c0e05a1248f37ca2`;
compact evidence is `evidence/e42_bfree_manifest.json`.

**RRDataset external robustness.** Reuse the already frozen official Zenodo record `14963880`,
CC BY 4.0 receipt for `RRDataset_test.tar.gz`: exactly 20,117,869,400 bytes, published MD5
`13c3ff3d61986170cc0c8cf76a35cd4b`, destination
`/Volumes/LaCie/pixelproof-datasets/e33_rrdataset/archives/`. It remains zero local bytes at this
freeze. Because RR validation was previously observed, the test is an independent-row robustness
transfer but not collection-independent proof.

**ITW-SM locked external.** Official `dkarageo/itw-sm` revision
`3060094fb576669927134193de3f517d7e64af86` contains 10,000 social-media images (5,000/class,
Facebook/Instagram/LinkedIn/X), declares 3.57 GB and is manual-gated for non-commercial research.
The machine currently has no Hugging Face login and no access approval. No image byte may be
downloaded until the user authenticates and personally accepts the dataset terms; metadata access
does not imply consent. ITW-SM stays the preferred E42 independent final, not a silent dependency
of E41's open-test run.

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

The E36 metadata freeze reproduced every selected contract before image transfer. The five REAL
CAL archives total 2,052,606,020 B; the four locked REAL FINAL archives total 2,038,841,380 B. AI
selection is 600 CAL rows / 468,420,944 B and 240 locked FINAL rows / 311,236,195 B. The detailed
395,300-byte selection has SHA-256 `01eec03e...2dcc`; compact evidence is
`evidence/e36_acquisition.json`. `evidence/e36_qwen_role_amendment.json` records that the older
unscored 40-row scout is superseded before E36 acquisition. Image bytes downloaded by freeze: zero.

### E36-A selected CAL/FINAL sources — metadata decision before image bytes (2026-08-27)

REAL source: Zenodo `18136670`, `sns-homogenization-forensics-dataset` v1.0.0, publication
2026-02-03, record-level CC BY 4.0. The source declares nine device ZIPs and three parent-linked
conditions: `view_000` normal/unprocessed, `view_001` QQ and `view_002` Sina Weibo. PixelProof CAL
uses archives 001/002/003/005/009 only; FINAL reserves 004/006/007/008 before any score. CAL selects
at most 100 normal originals per device and requires at least 80; FINAL uses at most 100 per device.
Social copies remain children of the same parent and never inflate sample size.

CAL transfer reproduced all five MD5 values and all 600 AI blob SHA-256 values: 2,521,026,964 B
total, with zero FINAL bytes. ZIP CRC/safety inventory found normal/QQ/Weibo image counts of
138/143/138 (device 001), 139/139/139 (002), 168/165/168 (003), 100/100/100 (005) and 71/71/71
(009). Because device 009 has only 71 normal originals, the preregistered >=80 availability floor
is amended to >=70 before extraction or any model score. It remains in CAL and each device remains
capped at 100; no easier replacement source is introduced.

CAL realization then passed 1,071/1,071 decodes: 471 REAL normal originals and 600 clean AI
parents. REAL device counts are 100/100/100/100/71; all six AI families contribute exactly 100.
There are zero exact-byte duplicates, zero cross-label dHash collisions and zero exact/dHash match
to prior passed E32 realizations. The unscored detailed manifest is 518,606 B / SHA-256
`4ed1b734...2e03`; compact evidence is `evidence/e36_cal_manifest.json`. FINAL downloaded bytes
remain zero and no DDA score existed when this manifest was frozen.

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

### E36-B CAL consumption and E37 role amendment (2026-08-27)

The unchanged official DDA checkpoint scored all 1,071 CAL parents once. The first threshold that
met REAL device-macro/worst FP budgets retained only 27.67% AI family-macro and 1.0% worst-family
recall; ROC-AUC was 0.58753. E36 is therefore a failed calibration experiment, not a candidate.
No FINAL REAL archive or FINAL AI blob has been downloaded, opened or scored, and the final cells
listed above remain eligible for exactly one later frozen candidate.

From this result onward, the 471 CAL REAL and 600 CAL AI parents are consumed
`E37_ADAPTATION/DEVELOPMENT`. They may fit a new head, but they may not be presented again as an
independent DDA calibration set. E37 threshold selection is permitted only from predictions that
are out-of-fold by complete REAL device and complete AI generator family; no row may be scored by
a head trained on its own source group. The original E32 TRAIN pool may remain the fixed base.
Evidence is `evidence/e36_calibration.json`; detailed score bytes stay outside Git at
`/Volumes/LaCie/pixelproof-datasets/e36/`.

E37 extracted one DINOv2-S embedding for each of these 1,071 consumed rows and generated exactly
one source-held-out score per parent. The resulting feature archive is local-only at
`/Volumes/LaCie/pixelproof-datasets/e37/e36_dinov2s_features.npz`, SHA-256
`3a08e0dc...f178`; OOF scores SHA-256 `5f66c32e...2d1d`. These are DEVELOPMENT derivatives and
cannot be reassigned to FINAL. The four reserved REAL archives and 240 reserved AI blobs still
have zero local bytes and remain the only authorized E38 FINAL cells.

E38 consumes the same rows with uniform sample emphasis and creates no new dataset role. Its fixed
candidate passed DEVELOPMENT and therefore unlocks acquisition of only the already named FINAL
cells. Candidate artifact SHA-256 is `fddbe475...4067`; threshold `0.896190`. Neither may change
after FINAL bytes begin. The FINAL native/clean parent manifest must be frozen and committed before
the first model score; QQ/Weibo copies remain grouped derivatives and cannot enter the headline
parent count.

FINAL acquisition then reproduced all four published REAL archive MD5 values and all 240 pinned AI
blob SHA-256 values: 2,350,077,575 downloaded bytes. Safe ZIP/CRC inventory found at least 100
native `view_000` originals in every reserved device. Before model access, PixelProof extracted and
decoded exactly 100 native originals/device and audited all six AI families at 40 rows each. The
frozen FINAL contains 400 REAL +240 AI =640 parents, zero exact/perceptual match to earlier passed
roles and no within-FINAL cross-label dHash collision. Detailed manifest is 319,091 B / SHA-256
`cad71ff5...66e6`; compact unscored evidence is `evidence/e38_final_manifest.json`.

The one-shot E38 score consumes every one of these 640 parents permanently. They may support E39
diagnosis/calibration but can never again be called LOCKED FINAL. At the frozen E38 threshold, all
400 REAL parents were below the AI boundary and 162/240 AI parents were detected. Detailed scores
are local-only at `/Volumes/LaCie/pixelproof-datasets/e38/final_scores.jsonl`, SHA-256
`dd4f181d...dc2d`; compact result is `evidence/e38_final_result.json`. Any E39 success claim needs
new device- and generator-family-disjoint parents.

### E39-A role ledger (2026-08-27)

All 640 E38 FINAL parents are formally consumed `E39_CALIBRATION`. They remain stored under the
external E38 directory with unchanged score SHA-256 `dd4f181d...dc2d`; no image is copied or
counted again. Their sole new use was selecting the one E39 threshold. They are permanently barred
from E39 FINAL and every later independent success claim.

The resulting E39 decision contract is local-only at
`/Volumes/LaCie/pixelproof-datasets/e39/e39_threshold_candidate.json`, SHA-256
`7d497929...2cef`. It contains no image or new fitted weight and points to the unchanged E38 model
artifact. At this checkpoint E39 FINAL contains **zero bytes and zero rows**. Source research,
licence acceptance and allocation must be committed before any E39 FINAL transfer.

### E39-B independent FINAL source contract (2026-08-27)

REAL is frozen to the University of Florence FloreView dataset, whose publisher describes 6,637
outdoor images from 46 smartphones/11 brands and licenses the download CC BY-SA 4.0. PixelProof
selects only camera-native `Nat/jpeg-h264` JPEG parents: 40 each from D14 Apple iPhone 13 mini,
D27 DOOGEE S96 Pro, D34 Google Pixel 5 and D43 OnePlus 8T. The 955,483-byte official URL catalog
is bound at SHA-256 `90d8408c...186b`; selection is a fixed capture/subject/location ordering that
spreads every device across locations. No flat field, video, social derivative or extra-data row is
eligible.

AI is frozen to CERTH-ITI's AIGenImages2026 release (`sha6th/AIGenImages2026`), revision
`d634f663...c0c5`, dataset-card CC BY 4.0. Its 2026 MAD paper reports 5,439 images from 19 recent
models with generator, prompt and split metadata. PixelProof reserves seven previously unused 2025
cells: Reve Image 1.0, HiDream I1 Dev, Ideogram 3, Midjourney v7, Adobe Firefly Image 5, Z Image
Turbo and Gemini 3 Pro Image, capped at 40 clean parents each. The single 11,138,511,098-byte
archive is pinned at LFS SHA-256 `67c60427...c498` and Xet hash `6ff1c1e7...533a`.

The resulting E39 FINAL target is 160 REAL +280 AI =440 parents. The archive/catalog identities,
licences, candidate hashes, deterministic score-blind selection and fail-closed overlap policy are
recorded in `evidence/e39_source_contract.json` before image transfer. CID2013 was rejected because
its presentation images are scaled and only about six images exist per device; SCIMD-6/17 because
they are 224px resizes; ForensiCam-215K because no adequate downloadable-data licence was found;
VISION/FODB/CSAFE/IPN because those collections are already consumed by earlier roles.

The metadata-only preflight reproduced the exact upstream state and froze 160 individual REAL
URLs plus the single AI archive before transfer. Detailed external selection is 35,473 bytes,
SHA-256 `4253497a...7be4`; compact zero-image-byte receipt is
`evidence/e39_source_preflight.json`.

Physical transfer reproduced the full AI archive hash and acquired all 160 REAL URLs. Archive
safety inventory passed 10,905 members with no unsafe path/link/oversized expansion. Eligible AI
counts are 305 Reve, 305 HiDream, 305 Ideogram, 300 Midjourney, 150 Firefly, 305 Z Image and 307
Gemini 3; deterministic member ranking selected 40 each. The publisher's 1,256,612-byte prompt
metadata CSV is bound at SHA-256 `46e484bd...0b22`. Images remain unscored and the unselected
archive members do not enter PixelProof's parent count.

The frozen E39 FINAL realization contains exactly 440 parents: 40 each from four FloreView phones
and 40 each from seven AIGenImages2026 generators. All 440 decode, all REAL rows are >=2 MP with
EXIF, and all AI rows retain prompt provenance. There are zero exact or dHash overlaps with earlier
roles and zero within-FINAL exact/dHash duplicate parents. Detailed external manifest is 412,914 B
/ SHA-256 `1076df20...7306`; compact unscored evidence is
`evidence/e39_final_manifest.json`. No E39 prediction existed when this ledger entry was committed.

The one-shot result permanently consumes all 440 E39 FINAL parents. Detailed local scores are
`/Volumes/LaCie/pixelproof-datasets/e39/final_scores.jsonl`, 146,705 B / SHA-256
`2a47e8a8...bb86`; compact result is `evidence/e39_final_result.json`. From this point their only
eligible role is E40 adaptation/development. They cannot test E40 or any later candidate, and the
unselected rows from the same FloreView/AIGenImages2026 source collections cannot be called an
independent substitute merely because their filenames differ.

### E40 consumed development role (2026-08-27)

`evidence/e40_role_amendment.json` formally assigns every E39 parent to
`E40_ADAPTATION_DEVELOPMENT` before E40 feature extraction. Counts remain 160 REAL +280 AI; no row
is filtered. The amendment cryptographically binds the 440-row manifest, full one-shot score
stream, compact failed result and E39 decision contract. These bytes may train/select E40 only.
They can never be counted as E40 FINAL, and extra FloreView/AIGenImages2026 rows do not restore
source independence. No new dataset was downloaded at this checkpoint.

E40's historical replay is also fixed before fitting: for every label/source stratum in E32 TRAIN,
select `round(5%)` rows by the lowest SHA-256 of `E40_REPLAY_V1|record_id`. The expected union is
1,067 existing feature rows; no image is copied or downloaded. All 1,071 existing E36 consumed
development rows remain in every E40 fold. Replay record IDs and final feature archives will be
hash-recorded by the experiment, but none of these sources becomes independent evaluation data.

The E39 consumed-image feature cache now exists at
`/Volumes/LaCie/pixelproof-datasets/e40/e39_dinov2s_features.npz`: 440x384 float32, 642,070 bytes,
SHA-256 `ec0501713a966b1ceaef41539907638b26440c4c3b1f39f69a8de5ff0c794e68`. It contains one embedding,
record ID, label and source per frozen parent—no image copy and no extra row. This is development
material only; `evidence/e40_features.json` records the compact binding.

E40 training used exactly the preregistered existing rows: 1,067 E32 TRAIN replay features, all
1,071 E36 consumed development features and all 440 E39 consumed rows under source-held-out OOF.
The sorted replay-ID list binds to SHA-256 `646a85a2...e13b`. No new image was downloaded, copied or
promoted to a test role. The generated score streams and 12,690-byte draft remain under local
`/Volumes/LaCie/pixelproof-datasets/e40/`; they are derived development artifacts, not datasets or
independent evidence.

E40-C introduces no new dataset. It reuses all 440 E39 parents under two deterministic, parent-
linked transport views and the already-consumed 210-photo owner-gallery DEVELOPMENT smoke (identity
SHA-256 `390e3c21...ac09`). The one declared extra WhatsApp reserve remains excluded and unscored.
No derivative changes the unique-parent count, and no gallery or derivative row can become FINAL.

E40-C scored exactly 1,320 parent-linked E39 views (440 unique parents x3 transports) and 210
owner-gallery parents. The output is derived DEVELOPMENT only; score stream is 487,011 B / SHA-256
`a126e814...0e3b`. The one reserve remains excluded. After the sealed failure, only the 440 native
E39 rows and 210 gallery parents entered a post-hoc threshold diagnostic; both populations were
already consumed. No image role was upgraded and no new data was acquired.

For E41, the 440 native E39 rows and 210 owner-gallery rows are now formally consumed
`E41_BROAD_REAL_CALIBRATION` (650 parents). The 880 compressed/resize views stay linked robustness
derivatives and are excluded from threshold selection. None of these bytes can enter E41 FINAL.
This role change creates no copy and downloads no data.

The packaged E41 artifact is a 13,064-byte derived model file at
`/Volumes/LaCie/pixelproof-datasets/e41/e41_dinov2s.joblib`, SHA-256
`9bcc021e74b617ee48cf297bd384a8dbe946240ec04822323af1e7c3fe63ab65`. It contains no dataset bytes.
At this checkpoint E41 FINAL has zero images and zero rows. Future FINAL must exclude E39/FloreView/
AIGenImages2026 and the owner gallery, and cannot begin until exact new source/licence allocations
are frozen separately.

## E42 recovery roles (frozen 2026-08-28 before extraction)

E42 downloads no new image source before development. Its base training population is exactly the
fixed 1,067-parent E32 TRAIN replay, all 1,071 consumed E36 CAL parents and only the 2,500 official
`train/{real,ai}` members inside the already downloaded, MD5-verified CC BY 4.0 RRDataset original
train/validation archive. The RR archive remains 2,163,176,547 bytes with MD5
`2f4498c3690d8f4c7a30d2e41dd34500`; its 500 validation members are not promoted to E42 TRAIN.

E42 source-held-out DEVELOPMENT is fixed at 2,246 previously consumed unique parents: 640 E36
former-final rows, 440 E39 rows, 960 IPN native phone originals across 12 devices and 206 unique
owner-gallery images from the declared 210-file identity SHA-256 `390e3c21...ac09`. The four
duplicate pairs (`IMG_8335` through `IMG_8338`, with and without the ` 2` suffix) are byte-identical
and collapse to one parent each. These rows may select the E42 backbone and
threshold and can never validate it independently afterward. The 811 B-Free viral rows and the
unopened 20.12 GB RR test archive are forbidden from training, calibration and model choice. RR
test remains locked for one candidate; ITW-SM remains untouched and manually gated.

The RR train-only extraction is complete at `/Volumes/LaCie/pixelproof-datasets/e42/rr_train`:
2,500/2,500 decoded images, 1,250 REAL +1,250 AI and 1,860,689,134 image bytes. Seven AI groups are
retained (113–479 rows) rather than collapsed. Detailed receipt SHA-256 is
`ba8f4ab1...a4941813`. The frozen combined manifest contains 6,884 unique parents from 63 declared
sources: TRAIN 4,638 (2,335 REAL /2,303 AI) and DEVELOPMENT 2,246 (1,726 REAL /520 AI). Cross-role
exact SHA-256 and exact dHash overlap are both zero. Manifest SHA-256 is
`15124d93f195d618b00c9cf79bec6151ae26fd4397cd9f5529c41842c4e3e238`; compact tracked receipt is
`evidence/e42_data_manifest.json`. No B-Free or RR-test row appears.

## E42 RR external robustness acquisition (2026-08-28)

The locked CC BY 4.0 Zenodo 14963880 test archive is now complete on LaCie: 20,117,869,400 bytes,
published MD5 `13c3ff3d61986170cc0c8cf76a35cd4b`. Full tar inventory and safe extraction contain 50,999
images /20,354,797,721 expanded image bytes: original REAL/AI 8,500/8,500, transfer REAL/AI
8,500/8,500 and redigital REAL/AI 8,499/8,500. The public package therefore differs from the
paper's described 10,000 REAL +10,000 AI parent population; no missing row is synthesized.

The actual archive layout is `RRDataset_final/{original,transfer,redigital}/{real,ai}`, despite the
repository README's `real_images/ai_images` example. Extraction preserves only declared image
members and creates no model score. The first decoded audit found 35 same-label exact-copy parent
components, 13 protected-role exact REAL overlaps and one protected-dHash AI parent. Final selection
must remove contaminated parents as whole events and deduplicate clean exact components before the
unscored manifest is bound. `evidence/e42_rr_acquisition.json` records archive, receipt and inventory
hashes; final selected counts remain pending until that manifest passes.

Decontamination is complete without model access. Forty-seven entire parents /141 rows are excluded,
leaving 50,858 images from 16,953 parents and 20,341,312,914 bytes, or 99.7235% of official rows.
Selected counts are original REAL/AI 8,454/8,499, transfer 8,454/8,499 and redigital 8,453/8,499.
The detailed unscored manifest is 31,091,691 bytes /SHA-256
`b2d815afab0bbafa339baf70eac19afbaf955e545041c550193340763ac30c98`; tracked receipts are
`evidence/e42_rr_manifest.json` and `evidence/e42_rr_score_contract.json`.

The one-shot E42 run has now consumed every selected RR row: 50,858/50,858 inference successes,
with no new image download or copy. The derived JSONL score stream is 14,572,649 bytes /SHA-256
`c065957e21df795712ae367566f5f86358443d66829739125092a74fee868434`; the 17,498-byte report
SHA-256 is `516c6d92ca8d712aa740bb929ea835bfbc19324c16df3b9f786042589496252e`.
RRDataset is no longer eligible as an independent FINAL for a later candidate. Its images may be
declared only as consumed `E43_DIAGNOSTIC_DEVELOPMENT`; a new final source must exclude all RR,
E42, B-Free and earlier protected parents.

## E43 untouched final — ITW-SM access decision (2026-09-02, zero bytes)

| Item | Frozen fact |
|---|---|
| Source | [`dkarageo/itw-sm`](https://huggingface.co/datasets/dkarageo/itw-sm) |
| Meaning | **In The Wild – Social Media**; a real-world AI-image-detection benchmark |
| Declared size | 10,000 images /3.57 GB: 5,000 REAL +5,000 AI |
| Platforms | Facebook, Instagram, LinkedIn and X |
| Labels | `0_real` /target `0`; `1_fake` /target `1` |
| Intended role | `E43_UNTOUCHED_FINAL`; never TRAIN, CAL or DEVELOPMENT |
| Licence/access | ITW-SM research-use terms; individual gated access, non-commercial research only, no redistribution, privacy/non-identification and citation obligations |
| Current physical state | **Not downloaded**; authenticated request still awaited manual author approval on 2026-09-03; zero payload images and no receipt |

ITW-SM was selected because it preserves the resolution, compression and content distribution of
images encountered on real social platforms. Controlled generator datasets can reward format,
resolution or collection shortcuts that disappear in actual uploads; ITW-SM specifically measures
that deployment gap. It is also balanced by class and exposes platform metadata, allowing both
overall metrics and per-platform REAL false-positive /AI-recall reporting. This makes it a stronger
answer to E42's authentic-photo and redigital-transfer failure than another internal random split.

The dataset is not a new training source and must not be used to choose E43's architecture,
threshold, transforms or stopping point. Before any download the project must preserve the accepted
terms and resolved repository revision. After authenticated transfer to the external dataset root,
the acquisition gate must verify the declared 5,000/5,000 label counts, decode every file, record
exact bytes and hashes, reconcile `metadata.csv`, and decontaminate whole parents against every
prior TRAIN/CAL/DEVELOPMENT/FINAL role. Only a zero-score manifest and a candidate-bound score
contract may unlock one E43 run. No row removal, threshold repair or retry is allowed after a
completed result.

The student submitted the individual access request and accepted the non-commercial terms on
2026-09-02; local OAuth authentication succeeded without recording credentials in Git. The
authenticated Hub
API resolves `dkarageo/itw-sm` to immutable revision
`3060094fb576669927134193de3f517d7e64af86`: 10,004 files /3,573,691,324 bytes. Its exact remote
layout is 5,000 images under `0_real`, 5,000 under `1_fake`, plus `.gitattributes`, `LICENSE`,
`README.md` and `metadata.csv`. These are remote inventory facts, not a completed-download claim.
The revision-pinned acquisition tool is committed before the first image transfer and will write a
receipt only after all local paths and sizes match this inventory.

The first pinned download request returned HTTP 403 with Hugging Face's explicit state
`awaiting manual author review`. Metadata inventory access did not mean content approval. The
attempt created only about 6.3 MB of resumable Hugging Face cache/tree/lock scaffolding under the
external root: no payload image exists, no detector opened a file and no acquisition receipt was
written. The tool now performs a single non-image `.gitattributes` content preflight before starting
the 10,000-image worker pool, so future pending-review checks stop before scheduling image paths.

The authenticated access check was repeated on 2026-09-03 after the student reported receiving
several emails. Hugging Face again returned HTTP 403 with the exact repository state `awaiting a
review from the repo authors` during the non-image `.gitattributes` preflight. The repository still
contains zero local payload files, no acquisition receipt and zero model scores; only the prior
6.3 MB resumable cache scaffolding remains. No part of the 3.57 GB snapshot transfer started.

## E45 official MediaEval validation distribution — frozen before transfer (2026-09-03)

The official MediaEval 2026 SID repository publicly links the labeled validation archive
`itw-sm-sid-val.zip`. The task authors declare 10,000 in-the-wild images in `0_real` and `1_fake`,
5,000 per class. A live source preflight returned HTTP 200, 3,553,693,205 bytes, ETag
`"68555a02-d3d10e15"`, Last-Modified `Fri, 20 Jun 2025 12:54:26 GMT` and byte-range support. These
facts are bound in `evidence/e45_mediaeval_contract.json` before transfer.

This is the preferred immediate E44-D final because it tests online/social-media distribution shift
and is published by the challenge organizers, while Hugging Face manual approval and NIST
organization registration remain blocked. The filename and declared class structure strongly
suggest that it is the official MediaEval distribution associated with ITW-SM. It must therefore
be treated as one candidate final—not as a second independent benchmark beside the gated snapshot—
until post-download inventory/hashes establish their relationship.

The archive retains the already accepted research-only, non-redistribution ITW-SM usage boundary;
the project does not infer broader rights from an open URL. It is `E45_UNTOUCHED_FINAL`, never
TRAIN/CAL/DEVELOPMENT. Download target is LaCie, with resume, a 100 GiB free-space reserve, full
SHA-256 and CRC/schema inventory. At this checkpoint downloaded E45 bytes are zero, decoded images
are zero and model scores are zero.

The transfer subsequently completed at the exact bound size. Local archive SHA-256 is
`18f1806e1cef6bc9f7ed6e49b61379a6cb4bac63cb4f3ed4f9fffffdf177b6e3`; LaCie retained more than
431 GiB free. ZIP structure contains exactly 5,000 declared REAL and 5,000 declared AI paths under
root `ITW-SM`. A complete member-by-member decompression/CRC scan found one unusable publisher
entry: `ITW-SM/1_fake/x_618.jpg`, at ZIP index 9,763. A fresh HTTP range covering its local header
and compressed payload matched the local bytes exactly, proving that the corruption is present in
the published artifact rather than caused by the interrupted connection or disk.

E45 therefore retains 9,999 structurally usable images before decode: 5,000 REAL and 4,999 AI,
99.99% of official rows. The broken AI member is a disclosed technical exclusion fixed before
model access, not a model inference failure or post-score row removal. The inventory remains
explicitly `zip_crc_passed=false`; it is never rewritten as a clean 10,000-row archive. Images
decoded and model scores created remain zero at this checkpoint.

The next local-only audit decoded all 9,999 usable members and derived exact SHA-256, dHash,
geometry, format, label and platform. Nineteen duplicate REAL byte pairs were found; one lexical
record per pair was kept so repeated posts cannot inflate the result. Two AI rows
(`facebook_46.jpg`, `instagram_427.jpg`) matched protected prior dHashes and were excluded before
inference. There were no cross-label exact duplicates and no protected overlap survived.

The frozen `E45_UNTOUCHED_FINAL` manifest therefore has 9,978 rows: 4,981 REAL and 4,997 AI.
Platform cells are REAL/AI: Facebook 1,308/1,032, Instagram 1,206/2,178, LinkedIn 1,265/931 and X
1,202/856. Its detailed 4,489,982-byte manifest SHA-256 is
`3e7c1d7e815a252d454d36c78f2a6ad6381983edb9494c31951bdb683c6d7e03`; official-row coverage is
99.78%. The 141 within-E45 exact-dHash groups are retained only as a disclosed similarity
diagnostic because dHash equality alone is not byte identity. Model scores remain zero.

E45 was subsequently scored exactly once and is now **consumed external FINAL**. All 9,978
manifest rows received both model-arm and fused scores with 100% manifest coverage; official-row
coverage remains 99.78%. The result failed and cannot be repaired by deleting rows, changing cuts,
training on these images or rescoring a revised E44 candidate. MediaEval/ITW-SM is prohibited from
all future TRAIN/CAL/model-selection roles; it may only support disclosed post-hoc diagnosis.

## E43 RR adaptation roles — planned before selection (2026-09-02)

RRDataset's 50,858-row E42 score stream has already consumed its final status, so RR may now support
E43 only as labelled DEVELOPMENT. The fixed local adaptation sample will require complete
original/transfer/redigital parent triplets and will be selected without reading E42 scores:
1,960 pooled REAL parents plus 280 parents from each of the seven AI scenario sources. Independent
SHA-256 keys will select parents and assign 50% TRAIN /25% CAL /25% DEVELOPMENT within every
stratum. Expected totals are 3,920 parents and 11,760 image rows: 1,960/980/980 parents by role,
with both labels balanced in each role.

RR TRAIN triplets may fit the E43 head; RR CAL may select one threshold; RR DEVELOPMENT may only
evaluate the frozen local candidate. All three roles are scientifically consumed and none is a new
external final. ITW-SM remains the only planned untouched E43 final. No new image is downloaded or
copied by this role decision; paths continue to reference the existing decontaminated RR snapshot.

The score-blind role freeze completed exactly as declared. Available complete-parent counts were
8,453 REAL and 8,499 AI across the seven scenarios. Selection retained 3,920 parents /11,760 rows:
TRAIN 1,960 parents /5,880 rows, CAL 980 /2,940 and DEVELOPMENT 980 /2,940. Every role is exactly
class-balanced and each condition contributes 3,920 rows. The 7,645,807-byte detailed manifest
SHA-256 is `29dd9b564061098101bcaf178cda0c75cdacc659113ce8b01cc389371bef4b16`;
tracked receipt is `evidence/e43_rr_roles.json`. It records `score_files_read=0`, creates no model
score and references existing image paths without copying bytes.

E43-S feature extraction adds no dataset and copies no source image. The derived local archive at
`/Volumes/LaCie/pixelproof-datasets/e43/rr_features_small.npz` contains 11,760x3,072 float features
and role/parent/source/condition identifiers: 134,777,581 bytes /SHA-256
`fdc5d4c8b28136898eb1431939b6c38997a6dd501153fd545a0cb092f5ca4aa4`. It was created only from
the fixed RR role manifest with the hash-pinned E42-S encoder and reports zero model scores.

The E43-S fit consumes only the already-declared E42 fit-eligible views and RR `TRAIN`; RR `CAL`
originals select one threshold, while all RR `DEVELOPMENT` rows remain unopened. The fitted local
artifact is 87,916 bytes /SHA-256
`a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390`; its compact fit receipt is
`evidence/e43_small_predev.json`. This is a derived research candidate, not a new dataset or final
result. ITW-SM still contributes zero local payload images and zero scores pending author approval.

The one permitted E43-S run has now consumed all 2,940 RR `DEVELOPMENT` condition rows (980 each
for original, transfer and redigital) and 11,230 earlier E42 development-regression views. The
combined 14,170-row local score stream is 4,192,797 bytes /SHA-256
`8398f763b44f97b3e7ca426b74dd665d72850b740b2ecd2fb6ebbab30df1ccc4`. These roles can never be
reused as independent final evidence. No new source image was downloaded, copied or relabelled;
ITW-SM remains untouched with zero local payload images and zero scores.

## E46 cross-platform recovery sources — frozen before transfer (2026-09-03)

E46 separates model development from final proof with two publisher-controlled sources. The
calibration/development source is **SynthWildX**, published in the official GRIP-UNINA repository
for *Raising the Bar of AI-generated Image Detection with CLIP*. Its immutable source list is
`https://raw.githubusercontent.com/grip-unina/ClipBased-SyntheticImageDetection/main/data/synthwildx/list.csv`.
The list declares 2,000 X-hosted images: 500 REAL and 500 each from DALL-E 3, Midjourney v5 and
Firefly. It is assigned once to `E46_CAL_DEV`; it can never become final evidence. The repository
code is Apache-2.0, but the social-media images have no explicit redistribution grant in the
dataset README. Therefore image bytes stay in the external research store, are not committed or
redistributed, and dead publisher URLs may only be recorded as failures.

The untouched final source is the **TrueFake Facebook** partition from the official UNITN IJCNN
2025 release: `https://drive.usercontent.google.com/download?id=10cQq48JtpRZgrHuckMyeFOwPvZZHDMXd`.
Google Drive advertises `Facebook.tar.gz` as 3.9 GB. The paper states that each platform contains
the same 60,000-source subset after native API sharing: 20,000 REAL, 25,000 diffusion-generated and
15,000 GAN-generated images. The release repository is CC-BY-4.0. The archive is assigned to
`E46_UNTOUCHED_FINAL`; only a score-blind, hash-selected 2,000-row balanced manifest will be scored
after archive integrity, structure, labels and protected-overlap checks. No TrueFake score may
alter model, calibration, threshold, abstention band, gate or row selection.

The two sources differ in both origin and transport: development uses naturally posted X images;
final uses an independently constructed corpus passed through Facebook. Exact SHA-256 and dHash
checks against all protected prior roles are mandatory before scoring. SynthWildX/TrueFake overlap,
corrupt members and unavailable URLs must be excluded and disclosed before any model is loaded.
At this pre-transfer checkpoint, both local payload counts and model-score counts are zero.

SynthWildX acquisition subsequently followed the frozen official list exactly. Of 2,000 declared
X CDN URLs, 1,723 returned structurally valid images and 277 persistently returned HTTP 403/404;
no failed row was substituted. The valid payload is 553,125,164 bytes: DALL-E 3 396, Firefly 474,
Midjourney v5 435 and REAL 418. The preassigned surviving roles are CAL 1,034 and DEVELOPMENT 689
(CAL/DEVELOPMENT by type: DALL-E 3 237/159, Firefly 283/191, Midjourney 264/171, REAL 250/168).
Two exact-payload duplicate groups remain pending the formal identity audit and cannot inflate a
later result. Bytes were relocated unchanged from ignored local scratch to
`/Volumes/LaCie/pixelproof-datasets/e46/synthwildx`; the path-corrected 1,224,028-byte unscored
manifest SHA-256 is `fd8008a6781e3feef632ce7250a6ece08d15457c1f903579c8b9a80da2a89f3f`.
No model was loaded and no score exists.

The TrueFake Facebook transfer is also complete in the external store. Google Drive's exact
4,207,525,545-byte payload hashes to
`413cb7f9664cf5f4e37a2ae0bea5d1a999c47398ca1c267a2173e88c2cda0d63`; Last-Modified is
2025-11-01 18:21:12 GMT and byte-range resume was available. `gzip -t` and a full TAR listing both
pass. The archive contains exactly 60,000 JPG payloads: 10,000 FFHQ REAL, 10,000 FORLAB REAL, and
5,000 each from FLUX.1, Stable Diffusion 1.5/2/3/XL and StyleGAN 1/2/3. The archive remains
unscored and only 3,500 score-blind reserve candidates will be decoded to obtain the frozen 2,000-
row balanced final after contamination checks; the remaining 56,500 images need not be extracted.

The completed SynthWildX identity audit compared every recovered payload with 23 protected prior
manifests. Fifteen rows were excluded: two redundant same-label exact copies and thirteen protected
exact/dHash overlaps (some rows carry both reasons). The clean unscored population is 1,708 rows:
CAL 1,024 and DEVELOPMENT 684; type counts are REAL 415, DALL-E 3 396, Firefly 472 and Midjourney
v5 425. Its 1,258,086-byte audited manifest SHA-256 is
`953490a9c63669fac2305e6abcc4259f2f4066e8c58b7851efddedaa7e2da8d4`.

TrueFake inventory and sample binding also completed without extracting a final image. All 60,000
member facts hash to `b59e78de93d8d6b84f323fbca0329e2b7f60a73828a439fa7a893861198ba28b`.
Contract namespace `E46_TRUEFAKE_FACEBOOK_FINAL_V1` selects a 3,500-row reserve by lowest SHA-256
rank: 750 candidates from each REAL origin and 250 from each AI generator. After decode/overlap
checks, the first clean 500 per REAL origin and 125 per AI generator become final. The detailed
1,106,013-byte selection contract SHA-256 is
`1e77dfbdc69f82a8eb69c40ee93f87428b5f4627fd34928bb589913be37cead3`; scores remain zero.

The reserve was subsequently streamed from the compressed TAR without unpacking the full corpus.
All 3,500 candidates decoded successfully; no exact or dHash overlap was found against 24 protected
manifests (including clean SynthWildX), and there were no exact or dHash duplicate groups inside the
reserve. The final manifest therefore reaches all frozen quotas without replacement: 500 FFHQ,
500 FORLAB and 125 from each of eight AI generators, exactly 1,000 REAL /1,000 AI. The selected
payloads remain in the external candidate pool; non-selected reserve files are never model-scored.
The 1,442,089-byte `E46_UNTOUCHED_FINAL` manifest SHA-256 is
`4572339ebe15821c6c86d50178ed31aa80f60cf98f1bd710d73e7265c15b225b`; score count is zero.

The bound 2,000-row TrueFake Facebook final was consumed exactly once on 2026-09-03. All rows
decoded and scored, so coverage is 100%; no member was removed, repaired or replaced after model
access. The fused score stream is 510,939 bytes /SHA-256
`6a51a9b11163fc8bb45889e38cc400a1b210cff7bcd6f36d6a760edc1fa68c97`; the external report is
6,754 bytes /SHA-256 `4b66bfa94fd82ca885723a439d62330a719dd3cfa4f65bf9a020b523ea3666a7`.
This manifest and all 3,500 decoded reserve candidates are now consumed diagnostic evidence and
must never be reassigned to CAL, DEVELOPMENT or another independent final.

E47-R1 reused the exact consumed 2,000-row final only for post-final architecture triage; it added
no source image and changed no role. The legacy GenImage-ResNet diagnostic stream is 297,858 bytes
/SHA-256 `e4bbcde82d810e95a220ec3265052c22356a1d77db08f24484c84ced06c7cb27` under the external
E47 store. These scores are also consumed and cannot support a final claim.

E47-R2 acquired no image dataset. It pinned the official MIT-licensed UniversalFakeDetect code at
commit `030495aea3300a8b54c0ec37ec7fe1dd7e63c619`, its 4,083-byte ProGAN-trained linear head at
SHA-256 `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847`, and the official
932,768,134-byte OpenAI CLIP ViT-L/14 backbone at publisher SHA-256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. The 72 GB training
corpus and 19 GB benchmark were intentionally not downloaded. Model bytes live on LaCie; only the
small reproducibility receipt is committed.

The UnivFD arm then scored the exact 2,000 consumed E46 identities with 100% coverage; no source
image or role changed. Its 302,379-byte external diagnostic stream hashes to
`faff25929505ad40b2d84fec8fe142cd609c50dc552cb1a8e1c03d6f4909104e`. These post-final scores
are consumed architecture evidence only and cannot fit or validate E47.

E47-R2b acquired the official GRIP-UNINA StyleGAN2-trained ResNet50-NoDown weights only: 282,549,121
bytes /SHA-256 `65467594eeb53945417c909390a3d872d55b6dbd819aa12cf01e4ced9c4d5a08`, repository commit
`543943cdf281df7417751e794109431d0975df88`. No UNINA training or test image was downloaded. Its
licence restricts use to informational/nonprofit purposes, so any future runtime must remain
research-only/opt-in unless a different licence is obtained.

UNINA native-resolution inference produced 655 ordered rows before a score-blind runtime stop; no
metric was calculated. The preserved 88,487-byte partial stream SHA-256 is
`87417d5f86b14b725accd42c776949533d639ec354e283675347fa76117733a4`. It is excluded from all
comparisons. E47-R2b restarts the same identities from row zero with a frozen aspect-preserving
512 px long-side cap; this is an inference policy change, not a data or role change.

The capped UNINA restart completed all 2,000 consumed identities with 100% coverage. Its 298,486-
byte external stream SHA-256 is `1725169559c53bcf8e56a53bbc1cc697844cbef664d3a5cf9b872fc3ae6d99e6`.
Like every E47-R1/R2 stream, it is post-final diagnostic evidence only and cannot train, calibrate,
validate or prove a successor.

E47-R3 pre-registers 2,400 new CAL/DEVELOPMENT identities from the still-unscored remainder of the
existing TrueFake Facebook TAR; network bytes are zero. Every one of the old 3,500 E46 reserve
members is excluded before ranking. CAL quotas are FFHQ REAL 600 plus StyleGAN2/SD1.5/SDXL AI 200
each. DEVELOPMENT quotas are FORLAB REAL 600 plus StyleGAN/StyleGAN3 AI 200 each and FLUX.1/SD3 AI
100 each. A 20% deterministic reserve is decoded so corrupt/overlapping rows can be rejected
without score-dependent replacement. These roles can develop E47 but can never become its final.

The E47 CAL/DEVELOPMENT selection contract is now frozen before payload decoding. It binds 2,880
reserve candidates (1,440 CAL, 1,440 DEVELOPMENT), all outside the complete E46 3,500-member
reserve. The 919,423-byte contract SHA-256 is
`c031ef92f5188536003ad94789195e4f17866c0006e37812e2e576b658d0753a`; target rows remain
2,400 and model-score rows remain zero.

All 2,880 E47 candidates decoded successfully into a 745 MB external research pool. One SD1.5
reserve row (`general/00099.jpg`) matched a protected dHash and was excluded before scoring. Every
quota still filled: 1,200 CAL and 1,200 DEVELOPMENT, each exactly 600 REAL/600 AI across the frozen
sources. The 1,760,652-byte unscored manifest SHA-256 is
`378b83fe56bcf4bbf61d5b626efa71899bea571abeeeca05c74774daa8585739`; model-score count is zero.
The 480 non-selected reserve payloads remain unscored and may not silently replace a later row.

### E48 fresh decision-repair population — planned before selection (2026-09-04)

E48 requires no network transfer. It reuses verified local source archives but selects only fresh
identities outside all E46/E47 candidates and current-candidate training rows. The intended 2,400
rows are balanced within FIT (600), CAL (600) and DEVELOPMENT (1,200).

| role | REAL | AI | purpose |
|---|---|---|---|
| FIT | 150 unused VISION +150 unused CSAFE S21 camera originals | FLUX.1 100, StyleGAN2 100, SD1.5 50, SDXL 50 | fit authentic-score percentile maps only |
| CAL | same quotas, disjoint identities/devices where possible | same quotas, disjoint identities | choose monotone expert set and threshold |
| DEVELOPMENT | 600 unused FODB originals, device-balanced and <=5 cameras/shared scene | fresh FLUX.1, SD3, StyleGAN and StyleGAN3, 150 each | one-shot real-source and mostly generator-held transfer |

VISION (CC BY-SA 4.0), CSAFE S21 (CC BY 4.0) and FODB (research/non-commercial restrictions)
are already decoded on LaCie. TrueFake Facebook is CC BY 4.0 and remains compressed; only the
new hash-ranked AI members need streaming extraction. FODB's repeated 143-scene design prevents
an unseen-camera-and-scene claim, so DEVELOPMENT must report both row-level pooled FP and
camera-pipeline worst FP and cap each scene's representation. This is successor development data,
not a new independent final.

The score-blind E48 selection is now bound. Namespace `E48_MONOTONE_NONVETO_V1` selected
2,880 reserve candidates for the 2,400-row target: FIT 360/360 REAL/AI, CAL 360/360 and
DEVELOPMENT 720/720. It excludes all 6,380 E46/E47 TrueFake reserve candidates before ranking
and explicitly excludes current-candidate E32 training identities from camera candidates.
The 1,565,157-byte contract SHA-256 is `dbb6f4aa...0e6e`; no new payload was extracted and
no model score exists.

The first identity-audit execution stopped before a manifest or model score: legacy
`r1b_role_manifest.json` contains all 22,688 planned C3 roles and therefore marked every fresh
camera candidate as protected even though the current model consumed only the E42-selected
subset. E48 now excludes that broad planning ledger from overlap hashes while still excluding
the exact E42 current-training record IDs and every actual historical CAL/DEVELOPMENT/final role.
The frozen E48 candidate identities and quotas do not change.

A second pre-score stop identified a hash-implementation mismatch rather than changed data:
camera files reproduce their pinned SHA-256, while the current helper and the original audit treat
EXIF orientation differently when deriving dHash. E48 therefore verifies immutable camera bytes
with SHA-256 and reuses the already-decoded audit dHash for perceptual-overlap comparison. It does
not silently replace the pinned dHash with a newly computed value. Identities and quotas remain
unchanged; model scores remain zero.

The corrected audit completed all 2,880 candidates with zero decode failures. One VISION and one
FODB candidate were excluded for protected dHash overlap; deterministic headroom filled every
quota without score-dependent replacement. The frozen manifest contains exactly 600 FIT, 600 CAL
and 1,200 DEVELOPMENT rows, each 50/50 REAL/AI. Its 1,971,148-byte SHA-256 is
`1404a3ff...5b68`; model scores remain zero.

### E51 SCMI30-IITRPR CAL acquisition (2026-09-04)

The pre-registered REAL calibration subset is fully local on LaCie: 1,200 native camera JPEG/MPO
images totaling exactly 4,247,339,334 bytes. Selection is balanced across all 30 publisher devices
(40 each) and Random/Similar branches (600 each); camera make and model EXIF is present for every
row. HTTP Range reads fetched only the selected 4,098,207,862 compressed bytes instead of the
34,429,117,013-byte official v2 container. The 1,111,889-byte external receipt SHA-256 is
`01cc5921f716b4f713efa4955cf2c5c741e3dc18076d42c35ef841975fa2f33e`; compact evidence is
`evidence/e51_scmi30_download.json`. These images are CAL-only, cannot fit representation weights,
and created zero detector scores during acquisition.

### E51 SCIMD-17 auxiliary TRAIN archive (2026-09-04)

The exact Zenodo 17317613 `SCIMD-17.zip` is local: 174,438,734 bytes, publisher MD5
`37da574c9e8d9c0fd3a7c9bedc5d72a6` and SHA-256
`ef1fe3e77e21d44c4cb29f4b44ad69bc3b83bfd3038531605c5ee315a2cb0201`. A model-blind ZIP audit
finds 17,638 members containing 17,620 images (17,618 `.jpg`, 2 `.jpeg`) and 172,781,180 expanded
bytes. No image body was decoded during acquisition. Because all publisher images are uniform
224x224 resizes, this CC-BY-4.0 source is restricted to auxiliary REAL hard negatives in TRAIN;
it cannot calibrate, develop, finally evaluate, or support native-camera claims. External receipt
SHA-256 is `8b38fa826a8f3f51705eff96236cbc5c13602fdba4c124d7bd5ce091ce20b230`.

SCMI30 CAL identity audit found one protected near-duplicate: the publisher-labelled no-content
`D04_black.jpg`. A score-blind amendment downloaded five previously unselected D04-Similar reserves
(7,710,716 bytes); all passed, and the original namespace's first clean row `D04_nat_45.jpg` replaces
it. Receipt SHA-256 is `66e3d063ba530d146c93dc44c5cfdcf4f045ca88b8947798ba0bf90657e59722`.
The CAL contract remains 1,200 parents, 30 devices x40 and Random/Similar 600/600.

E51 TRAIN/CAL realization is now complete. The score-blind SCIMD reserve decoded without failure;
two conservative protected-dHash collisions were skipped and headroom retained exactly 100 images
from each of 17 devices (1,700). Combined TRAIN contains 5,978 parents: 4,035 REAL and 1,943 AI,
after removing all 360 held-out AI-CAL parents. CAL contains 1,200 SCMI30 REAL and 360 AI parents
from 18 historical TRAIN sources, expanded deterministically to 1,560 original +1,560 Q75 rows.
TRAIN/CAL manifest SHA-256 values are `41444640...77ef` and `60688291...2356`; scores are zero.

### E51 provenance and admission correction — 2026-09-07

The above frozen manifests remain intact; "realization complete" is not blanket permission to fit.
The follow-up audit verifies TRAIN/CAL bytes and canonical decoded identity, and explicitly retains
the outstanding protected-pixel/reserve checks in its readiness state. A stale Dotting protection
path was corrected to `e49_d1_dotting/manifest_unscored.json`; missing required manifests now abort
instead of silently reducing coverage. A metadata-only check of 22 pinned protected inputs finds
no exact-key overlap; v2 Commons/StyleGAN2/OpenFake reserves are included, not just final selections.

For SCIMD, filename screening found eight `chatgpt-*` files (two in selected TRAIN). We fetched only
the publisher's 2,388,083-byte metadata table, not new image data:
[official Zenodo record](https://zenodo.org/records/17317613), `merged_common17.csv`, verified
MD5 `47279dd6c20ba1e9da8bef11623b9da2`, 17,620 metadata rows. All eight record an INFINIX X6851
camera. Visual review of the two selected files shows apparent laptop-screen camera photographs;
there is no basis for flipping labels from the names alone. Labels and source quotas are unchanged.
This is targeted provenance support, not a dataset-wide proof of label purity. The compact receipt
is `evidence/e51_scimd_filename_review.json`; image-transfer cost for this checkpoint is zero.

### E51 offline recovery — 2026-09-09 (zero image downloads)

The completed 9,098-observation TRAIN/CAL check found zero cross-role matched parent pairs; its
compact receipt is `evidence/e51_prefit_identity.json`. A separate local locator inventory now
resolves 117,898 protected body locations: 73,165 ordinary files and 44,733 members inside existing
DDA/COCO/ITW-SM ZIP archives. This represents 50,577,346,337 image bytes already on disk, **not a
50 GB download**. Every merged locator has an expected SHA-256; none of the body-bearing records
has an unresolved path. Relative paths are resolved against each source manifest's directory.

`evidence/e51_offline_bodies.json` records locator hash `a51cb457...eb81`. It preserves 9,800
metadata-only parent/reserve references separately; they still need identity joins and cannot be
silently treated as image-level audit coverage. This inventory does not decode or score images;
canonical protected-pixel checking and superseded-reserve coverage remain pending before training.

Protected-pixel execution update (2026-09-09): the offline reader reached 64,000 locations before
six existing RR test images exceeded its 100 MP safety cap. Those six exact SHA-256 identities
(largest 178,562,880 pixels) now have a protected-only, serialized full-resolution exception; no
general cap is disabled and no payload is downloaded. Their identities remain test-protected,
not training candidates. The restarted comparison is still running; an available path or cached
fingerprint alone is not reported as a completed separation result.

E51 admission closure (2026-09-09): the completed protected-pixel report `2f070e7d...c1e54`
verifies all 117,898 locations /117,888 distinct encoded bodies and all 9,098 TRAIN/CAL
observations. Both new-source-versus-protected and all-E51-versus-E49 screens find zero matched
parent pairs. Reserve report `2c2cdb07...9ad36` joins all 9,800 metadata references and adds
434 byte-verified local reserve bodies, also with zero matches or reserved-identity hits.
419 superseded never-downloaded Commons reserves remain identity-forbidden; their pixels are
unavailable, not certified disjoint. These checks are duplicate screens, not proof of universal
semantic independence or publisher-label correctness. No new data was downloaded.

### E51 fresh DEVELOPMENT realized locally — 2026-09-09

Opened only the image bodies of the seven previously downloaded Datapoint Parquets after
E51-A passed CAL. Verified all 920 reserved publisher hashes/byte counts; created original/Q75
pairs locally. The score-blind shared-prompt reserve policy selects 800 AI: 160 each FLUX.2 max,
Nano Banana 2, GPT Image 2 high, Ideogram v4.0q and Seedream 5.0 Pro, with the same 160 prompts
across models. Paired with all 2,640 already-local IEEE test REAL images, the admitted manifest
is 3,440 image parents /6,880 observations, SHA `1b1882f3...ea224`.

Canonical screening against 122,782 protected bodies found no protected REAL overlap, one AI
perceptual candidate, and seven internal pair candidates. AI exclusions remove whole shared prompt
groups before selecting fixed ranked reserves. Five IEEE REAL pairs are retained but grouped after
score-blind visual review; real detected scene groups=2,635. Failed first admission audit
`975569d3...258e8` and grouping amendment `10792cb7...a495a` are retained. Camera ids are hidden
and IEEE images are 512px publisher-processed, not native-camera proof. No images downloaded in
this work, no test image enters TRAIN/CAL, and the new DEV cannot also serve as E52 final.

DEV usage completion: all 6,880 observations were evaluated once with frozen E51-A. Original/Q75
balanced accuracy is 92.85%/92.08%; REAL false-AI 1.67%/1.97%, AI recall 87.38%/86.13%.
Raw score SHA `088718f8...724be`. This population is now consumed DEVELOPMENT and remains forbidden
from future TRAIN/CAL or a relabelled E52 final. IEEE native/worst-device proof is still unavailable;
positive results here do not override the weaker consumed-E49 native/transport results. No download.

### E53 planning inventory — existing bytes only (2026-09-09)

Purpose: identify plausible balanced REAL/AI training expansion without downloading or contaminating
tests. Rehashed `/Volumes/LaCie/pixelproof-datasets/e32/c3_role_manifest.json`:
`0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb`, 13,430,131 bytes.
All historical TRAIN locators are physically resolvable, but **not yet E53-admitted**:

| Source | Historical TRAIN parents | Reason to consider / boundary |
|---|---:|---|
| CSAFE S21 | 3,197 REAL | Native phone pipelines; preserve physical-device groups |
| FODB | 3,078 REAL | Native camera/content diversity; scene splits do not hold out cameras |
| VISION | 2,798 REAL | Additional devices; preserve device/source roles |
| Community Forensics Small AI | 1,781 AI | 240 historical TRAIN generator identities, not 4,803 local generators |
| FLUX.2 Klein 9B | 1,788 AI | Modern-family replay; shared prompts stay grouped |
| GPT Image 1 | 1,782 AI | Preserve eligible commercial-family examples, not proof about newer GPT models |
| Nano Banana | 1,782 AI | Preserve eligible family coverage; check prompt provenance |
| Nano Banana Pro (ash) | 160 AI | Small family supplement, not enough alone to establish broad robustness |
| Qwen Image 2512 | 1,788 AI | Additional family; shared prompts stay grouped |
| **Total** | **18,154: 9,073 REAL +9,081 AI** | **Historical role counts, not approved new training or test counts** |

Method: reuse the existing E32 source-path/Parquet mappings, stat loose files, and validate row
indices against 35 local Parquet footers. No image column decoded, new score produced or remote
body requested. Loose files sum to 42,135,059,473 logical bytes (~42.14 decimal GB), excluding
Parquet image bodies; this is not the full disk size or the E53 training budget. Evidence:
`evidence/e53_local_inventory_plan.json`. Existing source terms/receipts remain authoritative.

Before reuse, intersect with all subsequent CAL/DEV/test/reserve and duplicate-component protection;
verify body hashes, decoding, class orientation, source terms and group metadata. Later exclusion
supersedes historical TRAIN eligibility. Keep the 4,534 C3 CAL parents separate. E51 DEV, E49,
ITW-SM, DDA test pairs and Module 2 protected data cannot become convenient training negatives.
`theminji` and unofficial `34data` provenance/licence restrictions from the later E32 audit still
override the earlier E31 representation-only recommendations. CF's FFHQ REAL half is not a
substitute for everyday native-camera photographs. No additional data was downloaded in this audit.

### E53 original-body audit and bounded native replay — 2026-09-09

The subsequent audit opens original image bodies, not only locators: 16,292 conservatively
unprotected historical TRAIN candidates +4,534 C3 CAL originals =20,826 verified bodies.
Against 134,196 canonical protected references, 15 match observations and internal propagation
exclude 17 parents; 16,275 remain. All 83 internal near-duplicate pairs are retained in the audit.
This is heuristic duplicate screening, not a guarantee of every semantic relationship. Evidence:
`evidence/e53_native_inventory.json`, detailed report SHA `a84405de...26d55`.

A separate closure fingerprints the 120 unselected E51 AI reserve parents and their Q75 versions,
in addition to the 800 selected AI already protected. Zero matches against the native candidate
pool; all 920 reserve parents are accounted for (`evidence/e53_latest_reserves.json`).

The bounded expansion removes both endpoints of every detected internal pair and chooses 5,652
parents under existing research licences: 3,000 REAL (1,000 each CSAFE S21, FODB, VISION) and
2,652 AI (500 each CF, FLUX.2 Klein, GPT Image 1, Nano Banana, Qwen Image 2512; 152 NBP).
Deterministic round-robin selection across recorded source groups preserves diversity. Historical
CF remains non-commercial/per-model-terms restricted. No image or resulting model redistribution.
These are extra TRAIN replay candidates, not independent final publishers. Added rows enter only
their matching publisher's FIT fold; existing CAL/validation receive no new images. Native features
are being extracted from originals rather than recycling old 224px export features. No download.

Completion: all 5,652 selected original bodies produced 16,956 clean/assigned-transport/Q75 feature
views, feature SHA `82ab703a641aebb4b08cc1175e08b272732d0a91e6825b19bdc394941482c1c5`.
The two native arms were fitted under both the original and coverage-balanced source protocols;
added rows remain FIT-only, and fold 2 adds none because all matching publishers are held out.
All eleven source components train at least once across v2; this does not turn them into new
independent publishers. All images stay on the existing disk; no new image body was downloaded.

Input-geometry audit (`evidence/e53_shortcut_audit.json`) finds 2,314/4,035 REAL and 353/1,943 AI
in the original E51 TRAIN pool are exact 224x224 (57.35% versus 18.17%). Every selected native
addition has short side >=512; combined inventory exact-224 fractions are 32.89% REAL/7.68% AI.
Native additions are 0/3,000 square REAL versus 2,152/2,652 square AI, another processing/content
association to control rather than exploit. These known source associations are not proof that
every error is caused by resolution or that downloaded data alone will solve generalization.

Reuse the verified local original-based inputs before requesting more of the same sources. Future
downloads require explicit permission, source licence/provenance and missing-coverage specification,
group-disjoint role assignment, protected-overlap screening, byte cap and resumable receipts.
No acquisition is authorized by this documentation entry. Existing consumed DEV/final and Module 2
held-out data remain protected. No public redistribution of CF-restricted images or derived weights.

### Home source review and MNW test-reserve plan — 2026-09-09

[Microsoft–Northwestern–WITNESS](https://github.com/microsoft/MNW) is an official, non-commercial,
evaluation-only collection; it explicitly prohibits training. Its current image folders include
recent text-to-image generators. Pin revision `c93abf43e8157558a0e60aab7df4278b2c539253` and select
50 metadata-hash-ranked images each from OpenAI_GPTimage2, Midjourney_v8, Google_Imagen4,
Adobe_Firefly_v4, MAI_image2, Flux_2_pro: 300 AI parents planned, <=768 MiB image budget.
Only complete per-folder Git trees are authoritative; the initial recursive root listing is truncated.
Resolve and hash-check LFS pointers to freeze actual image hashes/sizes before any image download.

Purpose: new-publisher current-AI evaluation reserve, never model training or threshold tuning.
Exclude explicit editing/inpainting/adversarial folders from the fully-generated Module 1 label.
The entire publisher stays protected; filename UUIDs do not establish independent prompts.
Acquisition/decode/overlap validation is not scoring or final admission. An AI-only reserve cannot
measure REAL safety or constitute the balanced E52 final. User has authorized public downloads;
no paid API spend or redistribution is included. Actual received counts/bytes will be appended.

Other reviewed options: the official CERTH AIGenImages2026 release overlaps the already-consumed
E39 collection despite a changed Hugging Face account, so it is not a new independent test source.
[RealHD](https://github.com/Hanzhe-yu/RealHD) currently says coming soon, not a usable published
download. [SOCRatES](https://socrates.eurecom.fr/) still requires a signed agreement/password; no
signature or access bypass is attempted. [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/)
offers research-only original DNGs and documented SLR provenance; it is a possible REAL supplement,
but not a modern-phone substitute. Its research licences must accompany any selected files and
RAW development would require a separately fixed decoder/input contract. No FiveK images downloaded
at this planning checkpoint. These source limitations prevent claiming that more downloads alone
will solve the representation problem.

MNW acquisition completed: 300/300 selected parents, 50 per declared family; 229,880,188 verified
image bytes (229.88 MB /219.23 MiB). Exact identities derive from publisher Git-LFS SHA256/size,
manifest SHA `51b2c5e3f18315912cf89a5c99da759da65eb72f76e5e909d40a69a41a31a4b7`.
Original bodies and source terms reside in `e52/mnw_reserve_v1/` on LaCie, not in Git. Immutable
download/overlap reports are retained alongside them; compact receipts are in `evidence/`.
Screened against 150,483 unique protected/current-TRAIN reference bodies: zero byte/RGB or
radius-4 dHash+pHash cross/internal matches, 300 retained, zero detector scores. Heuristic screening
cannot certify every shared prompt/scene; unknown publisher prompt structure remains protected.
This adds evaluation coverage for recent generators, not training material or REAL-class evidence.

### Google HDR+ REAL reserve preregistration — 2026-09-09

Official source: [HDR+ Burst Photography Dataset](https://www.hdrplusdata.org/dataset.html),
Hasinoff et al., SIGGRAPH Asia 2016. [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Public Google Cloud bucket `hdrplusdata` permits anonymous downloads. Use only full-resolution
`20171106/results_20171023/<burst>/final.jpg` (publisher Q95 finishing pipeline), one per burst.
These are camera-derived HDR computational photographs, not generative-AI samples; retain this
processing distinction in evaluation. The source covers older Nexus/Pixel phones, not 2026 phones.

Source audit found 20 explicit `synthetic_*` result folders among 3,640 final JPEGs. Exclude them
before selection; a reputable publisher name does not justify blindly labelling every file REAL.
Plan 100 metadata-hash-ranked photographic burst parents, <=768 MiB image transfer and <=12 MiB
per image, with pinned object generations/MD5/sizes before bytes. No 765 GiB full data, DNGs or
new decoder dependency. Retain licence/attribution locally; no image redistribution. Whole HDR+
publisher is a reserved evaluation source by project policy. Burst/session/day groups do not prove
independent scenes; device coverage and protected overlap still need verification. This is a
complementary REAL stress reserve, not an official AI-detector pass threshold or balanced final.

Completed HDR+ transfer: 100 parents /423,955,391 bytes; manifest SHA
`417dee80f915c983ebb6bc6bb195e9579f2fb712b28a8ca85449e46f04dfc176`. Originals and attribution
are in LaCie `e52/hdrplus_reserve_v1/`; no local pixel modification to originals. GCS generation
and publisher MD5 match; SHA256/canonical RGB fingerprints are recorded after verified decoding.
Selected images are full-resolution (roughly 8–13 MP), not gallery thumbnails. EXIF labels include
Nexus/Pixel names and aliases (`bullhead`, `angler`, `sailfish`, `marlin`); do not count nine strings
as nine independent device models or physical devices. Eighty-one session/day proxies remain
only approximate grouping metadata. Zero cross-protected/internal heuristic matches against
150,783 historical/native/reserve observations including MNW. All 100 stay unscored and excluded
from future TRAIN/CAL; final E52 admission still requires broader coverage and a frozen candidate.

MNW+HDR+ new image payload total is 653,835,579 bytes /400 parents; metadata is additional.
Neither source was used to optimize E54. See `evidence/e54_hdrplus_{manifest,download,overlap}.json`
and the accompanying overlap contract. The reproduced canonical reference snapshot is bound to
the prior MNW audit, not an unversioned mutable cache of convenient negatives.

### E56 coverage review and FiveK metadata-only acquisition plan — 2026-09-10

Admitted inventory remains 11,630 TRAIN parents. Ten of eleven frozen publisher components contain
only one class (9,270 parents); fold-0 FIT has 9/5,314 monochrome REAL versus 229/1,250 held-out
RR REAL. Topic fields are absent in this normalized manifest; this does not mean the photos have
no topics or that original source metadata cannot be recovered. No source label becomes a detector.
Receipt: `evidence/e56_coverage_audit.json`. No new image body or model score for this audit.

[B-Free's official training release](https://github.com/grip-unina/B-Free/tree/main/training_data)
offers paired COCO/SD2.1 data, but the COCO publisher is already protected by DDA/Module 2 roles;
do not silently admit that release. Its background-restored variants also mix real/edited regions,
so they cannot automatically receive fully-generated Module 1 labels. Source/content matching is
a useful research principle, not permission to recycle our tests. E40 already tried content-cluster
weighting, so a repeat is not a novel fix.

FiveK source review: official MIT-Adobe 2011 SLR photographs with subject/light/location/time
annotations and human tonal adjustments. It may add processing/content diversity, but is not
modern-phone evidence or a matched-AI dataset. Archive only its official index, two research
licences and two filename-assignment lists (12 MiB total cap; 8 MiB/response) under actual LaCie
`e56/fivek_metadata/`. Retain notices and citation; research-only, non-commercial, no endorsement.
Exact licence assignment and complete 5,000-parent metadata are required. Zero images or model
weights planned in this acquisition. Before any separate image pilot, freeze parent selection,
byte cap, RAW/ProPhoto-to-sRGB processing, canonical overlap screening and TRAIN-only role.
Sources: [FiveK](https://data.csail.mit.edu/graphics/fivek/),
[Adobe licence](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt),
[Adobe/MIT licence](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt).

Acquisition outcome: 4,725,181 archived metadata bytes; zero image bytes. Index 4,629,128 B,
licences 1,841/1,994 B, filename lists 45,282/46,936 B. V1 exact-basename join correctly rejects
extensionless official lists; v2 exact-stem amendment (no refetch) validates a unique complete
licence assignment for all 5,000 rows: 2,690 Adobe /2,310 Adobe+MIT. Subjects: nature 1,089,
people 1,341, man-made objects 1,869, animals 300, unknown 366, abstract 35; lighting: sun/sky
3,404, mixed 831, artificial 765. These are publisher annotations, not independently verified truth.
Manifest SHA `ace6999013bf359db1d7d6b8b22361b9cfc2b9bb97aaf55c71e5f5b7ced10487`; compact receipts
`evidence/e56_fivek_metadata.json` and `evidence/e56_fivek_review_contract_v2.json`.

Planned next, not downloaded: 12 original DNGs (two hash-ranked per subject, including unknown),
<=32 MiB each/384 MiB total. Quarantine for fixed, isolated RAW decoding and full protected-overlap
screening; no TRAIN admission or model scoring until checks pass. Retain original bodies/licences,
do not manufacture modern-camera claims or label human tonal edits as AI-generated.

Pilot frozen before image GET: 12 subject-stratified parents, official HEAD sizes total
122,074,000 bytes (~122.1 MB), each within 32 MiB. Preserve observed ETag/Last-Modified and locally
computed SHA-256; upstream provides no per-file cryptographic digest. Contract:
`evidence/e56_fivek_pilot_contract.json`. Isolated decoder rawpy 0.27.1 /LibRaw 0.22.1, NumPy 2.5.1,
Pillow 12.3.0; binary wheel installation is separate from frozen ML dependencies. AHD, camera WB,
sRGB primaries/gamma(2.4,12.92), full resolution, auto-bright off, brightness 1, highlight clipping,
8-bit PNG. This is a fixed local rendition, not native camera JPEG or an expert's Lightroom output.
All originals and derived pixels remain one parent and quarantined; no training or score yet.

Pilot completion: all 12 official DNGs verified against pinned HTTP identity/size, local SHA-bound,
122,074,000 B received; 166,400,845 B local PNG derivatives (not additional downloaded photographs).
All 12 decodes reproduce identical RGB on replay. Licences retained beside originals. Protected
overlap checks against 150,883 observations including MNW/HDR+ find zero matches/internal pairs;
heuristic limitations remain. Quarantine/zero detector scores unchanged. Receipt
`evidence/e56_fivek_pilot_download.json`; overlap `evidence/e56_fivek_pilot_audit.json`.

E57 proposed acquisition, not yet selected/downloaded: min(12,N) per subject x light cell from
frozen official index, expected 201 parents/18 cells, deterministic new salt `FIVEK_E57_TRAIN_V1|`.
<=48 MiB each, <=4 GiB total original bodies, on external storage with bounded supervised runs.
Reuse exact verified pilot bodies where selected; same fixed decoder. Exclude protected/internal
matches without replacement, retain whole publisher as FIT-only research after admission. No new
final source, modern-device claim, paid API, or licence expansion. Purpose is testing REAL content/
lighting coverage while preserving existing AI replay, not assuming more data guarantees success.

E57 selection frozen before image GET: 201 parents, 18 subject/light cells, HEAD-pinned total
1,999,000,262 original bytes (~2.00 GB); all selected files satisfy the 48 MiB cap. Receipt
`evidence/e57_data_contract.json`. Exact pilot body reuse is verified against the same HTTP identity,
decoder recipe and SHA. No image substitution for resource failure and no detector score. A separate
completed acquisition/audit receipt will establish actual new bytes and eligible count.

E57 v1 stops after 61 decoded parents because `a1854-kme_290.dng` (DCS460D) lacks valid
as-shot WB ([0,1,0,0]). Preserve this original and failed-v1 evidence. Preregistered v2 f840136
keeps the same 201 identities, caps, licences and decoder, but applies the decoder's existing
metadata predicate before admission: four finite coefficients, first RGB three positive.
Unsupported cases remain quarantined, with RAW hash and rejection reason, no replacement.
Other errors still stop. Reuse verified original bodies and completed PNGs; no silent auto-WB
or daylight rendering. V2 records live on LaCie under `e57/v2/`, with raw/PNG cache retained
under `e57/`. Actual valid/excluded counts, lost strata, body bytes and protected-overlap results
must come from completed v2 receipts, not the planned 201 count. No admission or accuracy claim
at this in-progress checkpoint; as-shot metadata coverage is a stated limitation.

E57 v2 completion: all 201 selected original bodies total 1,999,000,262 B; two pilot reuses
24,508,438 B, so new originals across the whole E57 acquisition total 1,974,491,824 B.
199 supported RAWs yield 3,022,249,526 B local PNG derivatives (not network download).
Excluded `a1854-kme_290.dng` and `a2638-LS_060107_0717.dng` retain RAW/hash/reason; lost
strata animals/sun and unknown/sun each one, without replacement. All 18 strata still present.
Completed receipt `evidence/e57_v2_download.json`, data contract
`72781bf8672ab7b9debf3c92c4f20f383745fe63ec825420b4b3393c734e817b`.
Protected screen 150,883 observations gives zero cross/internal matches; admission manifest
SHA `2733bfe3c5397cbb26aa2faceb0cf44018bba2688079aae7642b0514561f3fb0` admits 199 FIT-only
parents, never CAL/test. Six-fit study modestly reduces pooled REAL errors but loses supported
AI sources and fails acceptance. No automatic enlargement; retain licensed originals and
provenance for research. No final-reserve images scored or training-role changes.

E59 acquires no new source images, weights or dependencies. It reuses the admitted E54
11,630-parent pool and already archived/licensed CLIP backbone from E47. Additional files
under LaCie `e59/` are local derived embedding caches, not new photographs: three existing
views and three crops per parent, raw768-D CLIP vectors plus fixed1536-D mean/std aggregates.
Upstream parent/source/role boundaries unchanged; no E57 FiveK addition or protected reserve
reuse. Contract evidence/e59_feature_contract.json binds source crop/teacher and encoder hashes.
Full extraction starts06:22 and remains in progress at this checkpoint; no new download bytes.

### E60 offline champion exposure/admission audit — 2026-09-10

E59 was subsequently paused; its 9,599 completed parent chunks remain preserved. E60 uses
only the already admitted E54 pool: 11,630 parents, 7,035 REAL and 4,595 AI, with 34,890 cached
views. No source, model-weight or dependency downloads. All admitted AI parents enter the one
registered correction; E57 FiveK and protected evaluation reserves are not added.

Reconstructed E43 FIT contains 8,844 parents/19,648 views, including 3,803 AI. Of these, 4,278
parents match current TRAIN with identical encoded-body hashes; 1,860 old AI are absent from
current admission, including 360 now-protected E51 CAL. Missing historical rows are not restored.
E54 fold validation contains 2,360/480/1,438 teacher-seen parents and is not clean validation
for warm-started E43. The unused audited native pool has 10,508 parents after prior duplicate
endpoint exclusion; only 2,385 AI (GPT Image1/Nano Banana) are outside current recorded groups,
with no corresponding unseen REAL group. This does not establish a balanced group-disjoint
fresh DEV in that pool. No global claim about every unindexed file on disk is made.

E51 DEV remains consumed; E49 is consumed regression-only, never tuning/final evidence. E52,
MNW/HDR+ and Module2 roles remain protected. Audit summary: `evidence/e60_audit.json`; detailed
teacher membership, missing-parent reasons and input hashes reside in LaCie `e60/audit.json`.


## 2026-09-10 — Metadata-only home acquisition queue

User disallows downloads now and requests new sources be recorded for home. PLAN's dated
home queue lists WIFD first, RawNIND second, SIDD conditionally, and RealHD watch-only.
Official source links, intended diagnostic roles and uncertainties live there. None was
acquired or admitted. Source/scene/parent splits and licenses must be frozen before scoring;
RAW/JPEG, burst/exposure and noisy/clean pairs inherit their common scene lineage. Do not
mistake file count for independent scene/publisher count or REAL-only evidence for a balanced
AI-retention test. Existing protected publisher and E52/Module2 boundaries remain intact.


## E65 — score-blind camera/noise pilot registered (2026-09-13)

User now authorizes needed data downloads and explicitly requested starting while connecting
power; subsequent pmset confirmed AC/charging. WIFD full tree is101,432,609,110B/6,944 blobs,
not a sensible automatic full download. Its1,838 SDR JPEGs cover10 cameras (not all14 cameras
in the entire corpus). Select at most4 distinct exposure/aperture setting tuples at each
camera's min/max recorded ISO; hash-rank with fixedE65 seed, omit reference/burst/AEB files.
SDR scene independence is unknown; do not equate files or cameras with independent scenes.
Pinned revision3f577edf0b14c686aa08e8d0d8ae07a83ba44f26; MIT README/license blobs verified.

RawNIND official Dataverse metadata v1.0:2,845 files/120,172,531,162B, dataset-level CC-BY-SA-4.0.
Its permissions.txt contains per-scene exceptions, including research-only entries: therefore
select ONLY explicitly CC0 scenes with author attribution, excluding official test reserves
from both dataset.yaml and test_reserve.yaml. Four hash-ranked scenes per Bayer/X-Trans,
one clean RAW and one highest available ISO<=6400 RAW per scene, preserving filename SHA1.
No RAW/scene selection uses pixels/model scores. Known-field hashes and unrestricted flags
are required. This pilot is REAL-only DIAGNOSTIC_DEV, whole publishers reserved, never TRAIN
or balanced final; future separate decoding/overlap and scoring contracts required.

Freeze all metadata/code and selected file identities before pixels. Total payload ceiling2GiB,
two download workers,40min cap, external volume with10GiB reserve. Verify Git blob SHA1 for WIFD,
MD5 plus filename SHA1 for RawNIND, then record SHA256. Resume partial HTTP ranges; if the
server ignores Range, truncate/restart rather than append. No existing payload/receipt overwrite.
No source code, executable or full repository clone is downloaded. Originals stay outside git.

Metadata recovery: initial gh recursive-tree response was incomplete; curl retry obtained the
complete non-truncated tree. System Python lacked trusted CA setup; requests in the existing
venv succeeded with normal TLS verification (no certificate checking disabled). No image bytes
had been acquired at that point. E43/E59/serving and existing protected datasets remain unchanged.


### E65 payload complete / score-blind audit plan (2026-09-13)

83 files (67 WIFD JPEG,16 RawNIND RAW),1,009,998,251B downloaded and publisher-checksum
verified. Receipt SHA `0aab1b2fb544e415b9018084ab13af2cd0bd17761ee3a163b68cdf601c750fd1`.
User said start while connecting power; AC/charging subsequently confirmed. No model scores.
Next: fixed isolated E56 RAW decoder, pinned runtime, canonical E51 fingerprints against
150,483 protected reference bodies plus300 MNW,100 HDR+ and199 FiveK records. Failures and
cross-reference exact/confirmed perceptual matches quarantine; RawNIND quarantine propagates
across its scene pair. Internal similar observations remain linked descriptive repetitions,
never independent scene evidence. No score/brightness exclusions or replacement draws.
Whole WIFD/RawNIND publishers remain diagnostic-only; no TRAIN or final admission.


### E65 decode/identity audit complete; diagnostic preregistration (2026-09-13)

All83 originals decoded successfully, including16 RAWs with frozen E56/LibRaw0.22.1 runtime.
151,082 reference records screened;0 cross-reference matches under fixed exact/pHash+dHash
criteria. Eight internal matched pairs are recorded, not discarded or counted independent.
83 diagnostic observations remain, RawNIND8 scenes. Audit SHA
`db8b3134886b62dc31d4d39be5b7eb8015550260fa0359bd92af8e18a8316caa`.
Initial audit CLI omitted PIXELPROOF_DATA_ROOT and failed before opening any payload/contract;
correct external-root environment succeeded. No frozen code/manifest changed.

Separate diagnostic protocol, before any E65 model score: unchanged E43-S head and backbone,
AI cut0.07940196245908739 and REAL cut0.011505939625203613; all83 admitted observations in
original and E49-convention1080px/JPEG75 views (166 views). Global+2texture crops, four frozen
DINO blocks and mean/std aggregation. Network disabled, batch8,20min cap, no training.
Lock full scores/features before descriptive source/camera/CFA/capture/ISO and paired RAW
scene/transport reports. No threshold search, E49 read, AI-retention claim or promotion.
11 focused E65 tests passed: source eligibility, checksums/resume, identity screening, pair
quarantine, transport orientation and complete metric pairing/boundary conventions.


### E65 diagnostic complete (2026-09-13)

All166 original/Q75 views scored with frozen E43 in146.33s; scores locked before metrics.
WIFD REAL false AI11/67 (16.42%) original and13/67 (19.40%) Q75; RawNIND3/16 (18.75%) and
5/16 (31.25%). Transport creates8 WIFD and3 RawNIND errors, rescues6 and1. High-ISO scores
fall in5/8 and rise in3/8 RAW scenes; brightness/development/scene differences prohibit a
noise-causality claim.7D-6 soil/seedling pair visually inspected after score lock; no exclusions.
No new candidate or AI recall measurement; target remains unmet, E43 and serving unchanged.
Whole publishers and feature cache remain consumed diagnostic-only, never TRAIN/fresh final.
697 Python tests pass (11 new); compile/pip/diff checks pass. Source/code/hash details and
next valid TRAIN/DEV acquisition requirements are in `evidence/e65_diagnostic.md`.


## 2026-09-13 overnight continuation — E66 acquisition preregistration

User requests continuous active-session experiments/development and commit/push checkpoints,
explicitly no30-minute monitor or scheduled heartbeat. No automation created; E59 remains
paused. Work proceeds while the live session can execute; interruption/network/app limits
cannot be guaranteed away. AC power confirmed51%. E65 commit2c6524e is present on origin/main.
Its GitHub Python job passed; web lint/typecheck/tests passed, existing critical Next.js audit
failed again (run34722064922), independently of ML. No audit threshold weakened.

Next E66: acquire only official SIDD Small sRGB archive,6,615,978,508B, published
MD5 796971867583bf14677dcae510e52538 and SHA1 5a4aa6aa7abcf7b0b56c88ad578fea9a4ff77935.
MIT stated on official SIDD page. One stream,2h ceiling,30GiB reserve; resume verified ranges;
no executables or official benchmark. Codalab mirror HEAD403, direct official mirror supports
ranges. Keep full publisher out of TRAIN/final; no model scoring before decode/group/overlap
admission.160 noisy/GT pairs are at most10 underlying scenes,5 older phones; processed GT
is not an additional independent photo. Prefer160 noisy sRGB for a grouped DEV candidate.

E60's2,385 unused recorded-group AI candidates remain under audit: GPT Image1 (current source
card CC-BY4.0) and Nano Banana (MIT). Their groups mostly identify individual files, not known
prompts. Plan a hash-ranked80 per family paired with160 noisy SIDD observations only after
provenance, old-role/teacher/current-TRAIN and newer-reserve overlap checks. This may support
a limited balanced DEV screen, never unseen-publisher/independent-prompt or final claims.
Model recipe/feature contract is a subsequent step; no candidate fitted yet.


E66 local AI admission protocol frozen before materialization/scoring:80 per GPT Image1/Nano
Banana, seedE66, chosen only from the2,385 E60-unused recorded groups after all E53 old-role
and duplicate exclusions. Explicitly reject E43 FIT/current TRAIN identity/body overlap.
The150,483-body snapshot includes the native candidates themselves: remove only selected
exact self-body entries after those eligibility checks, preserve every other hash/pixel and
all newer MNW/HDR+/FiveK/E65 references. Current TRAIN has100% body coverage in that snapshot.
Any confirmed cross-reference or internal overlap quarantines selected endpoints, with no
replacement draws. Reserve the entire2,385 unused pool from future TRAIN. Local materialization
and source-checksum verification only; no new AI downloads or inference. Unknown prompt
independence and previously seen families prohibit a fresh-final claim. Balanced DEV remains
pending SIDD. Native selection and archive range-resume tests pass; E67's30-source original
crop replay passed with exact0 score error and0 decision changes; blur extraction is running.


### E66 AI audit complete; SIDD group audit registered (2026-09-13)

160 selected local AI originals materialized and verified;160 admitted,0 cross-reference or
internal matches. Report SHA914c5eb4a4db7bea693965cca3e1e0f076dd34986cae71ae47be0ebb1cba51b7.
151,005 references after160 exact candidate self-body snapshot exemptions plus later protected
sources. No AI image downloaded or scored. All2,385 unused candidates now reserved from TRAIN.

SIDD remote ZIP directory required68,753 metadata bytes (no image members extracted),484
entries:320 PNGs,160 scene-instance directories,2 text metadata files and container directories.
160 noisy observations span10 underlying scenes; camera countsIP54,S635,GP33,N622,G416.
This directory is bound before full archive decode. Archive selection unchanged: complete
Small sRGB only, official benchmark untouched. Await whole-archive published checksum/SHA256
receipt, then exact directory validation and160 NOISY PNG decodes; GT stays unextracted.
Cross-reference overlaps or cross-scene internal matches quarantine complete underlying scenes,
including other cameras/settings; same-scene internal similarity stays explicitly dependent.
Only complete160 REAL+160 AI, no shared id/body/cross-label match, can freeze limited DEV.
No replacement draws or model-score-based exclusions.10 new E66/E67 pipeline tests pass.


### E66 archive verified (2026-09-13)

SIDD Small sRGB archive acquisition is complete: 6,615,978,508 bytes, published MD5 and
SHA1 matched; SHA256 855c375ae20312386cd961e7fdbbeaeb33efdecf0f6f115c01e09f45ec471fdf.
The frozen scene/decode/overlap audit is running. No SIDD model score or DEV admission yet.
Archive and image payloads remain on the external dataset volume, outside git.


### E66 limited DEV admitted; E67 fit can start (2026-09-13)

All 160 SIDD NOISY images decoded. Against 151,165 protected/reference records: zero
cross-reference matches, zero cross-scene or cross-label matches, no quarantine. The
223 internal matched pairs all stay within their known scene and are explicitly dependent.
Ten underlying REAL scenes across five cameras plus the 160 audited AI observations form
the frozen 320-observation development manifest, SHA bf4c3586126edf91588cc72b375504fa089c3d235abdce7f660686a82c2a6301.
No model scores exist for this population. This limited DEV is not independent final data.
Starting the already preregistered E67 fit: freeze exact artifact/code/DEV identity bindings,
then one constrained TRAIN fit with the unchanged 10% REAL ceiling and per-image AI guard.


### Additional data lead: Boon_or_Bane (metadata only, 2026-09-13)

The author-linked public Nextcloud share opens without authentication in the browser:
https://cloud.digfor.code.unibw-muenchen.de/s/BEEzsMnZLFCXw9S . Its root shows only two
folders, ChatGPT5 (145 MB) and FireflyImage4 (139 MB), rounded total 284 MB. No licence
file is visible at the root. ChatGPT5 has folders named 1024_1024, 1024_1536 and
1536_1024; the first contains fingerprint/test subfolders. These names are not verified
pixel dimensions or generator identities. They differ from the paper's GPT size table,
so exact published-file provenance needs clarification before use. No image download
was triggered and no image was opened/scored. A legacy read-only DAV request returned
401; the ordinary public browser listing succeeded, without login or access changes.

Keep as an unadmitted acquisition lead pending explicit dataset licence, exact manifest,
prompt/source/processing provenance and full protected overlap checks. Do not infer an
image licence from the paper's CC BY-NC-ND licence. This lead contributes no E69 data and
does not resolve modern REAL scene coverage; it is not a ready TRAIN/DEV/final pool.


### First E66 DEV score stream locked (2026-09-13)

E70 completed all 640 paired DEV scores in 274.49 seconds. Full score SHA
9cf625b6f2123c5a6bd421f37d6415030cc386ccdc7cd8b809a6dba36b879521. Scores are now locked before the first metric report. E66's
admission manifest is an immutable historical unscored snapshot; the population has
NOW been exposed to model scoring and must be treated as consumed DEVELOPMENT in
future plans. Never infer current freshness from its original model_scores_created=0
field. No fitting, data exclusion, threshold change or E49 access took place in this step.


## E70 rejected on first E66 DEV comparison (2026-09-13)

All 640 scores locked in 274.49 seconds before metrics; unchanged candidate and cuts.
Original REAL FPR worsens from 68/160 (42.50%) to 75/160 (46.88%): eight rescues and
15 new errors. Social-Q75 REAL FPR improves from 68/160 to 49/160 (30.63%): 19 rescues,
zero new errors. AI recall declines from 156/160 (97.50%) to 153/160 (95.63%) original,
and from 154/160 (96.25%) to 152/160 (95.00%) social Q75. Paired AI losses are four
original and three Q75 views, partially offset by one rescue in each condition; both
GPT Image 1 and Nano Banana lose previously caught examples. Do not mask losses by net
counts. Original AUC .96668 -> .95379; Q75 .94430 -> .95031. Both reference and candidate
pass only 12/20 numeric gates on this limited DEV. E70 fails all additional original
checks and Q75 AI retention, so no E49 regression is permitted and no promotion occurs.

The TRAIN pass does not generalize to the new SIDD rendering domain or every new AI
observation. Its ten scene groups and unknown AI prompt relationships remain explicit
limitations. Preserve E70 candidate, full scores and failed report. E66 is now consumed
DEVELOPMENT, never TRAIN or a fresh final; its admission snapshot stays immutable. No
post-score threshold, interaction-map or source-specific route adjustment is allowed.
Next assess a complementary pretrained representation with the same E43-preserving
TRAIN/DEV rules, reusing eligible cached work where possible. Target remains unmet.

### 2026-09-13 — SIDL research lead, not admitted or downloaded

[SIDL official page](https://sidl-benchmark.github.io/) and
[AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/32257) describe physical lens
contamination pairs on iPhone12Pro. Research/education use is permitted; commercial use or
redistribution requires author permission. This can support optical-degradation research,
not broad camera coverage or an independent final by itself.

The official [full-resolution TRAIN folder](https://drive.google.com/drive/folders/1chLRpBtGzTkAX_7v-ycPgL_X1-mezv2r)
shows `train_full.tar`, rounded25.15GB, and six alternative split pieces.
[RAW folder](https://drive.google.com/drive/folders/1FTbgb43Eq_CTYTb_gTowQ8bPnPh6qftg) and
[metadata](https://drive.google.com/file/d/1o_O3ua5zhOv6vrsF6KwI9O19E5axUXED/view) are linked
by the authors. Browser metadata preview confirms sample4032x3024 linear ProRAW records,
iOS17.5.1 and applied denoising. No image was downloaded/scored, no complete metadata
inventory is claimed. Exact archive bytes/hash, RGB derivation, complete scene mapping and
protected-overlap audit remain prerequisites. Keep paired conditions within whole scenes;
never mix the author's validation/test scenes into TRAIN. No source role assigned yet.

Metadata-only acquisition completed:8,874,397 bytes, SHA-256
`649e408b82d139b1ff3b5aeeeb9ad7e73f4ec92fdb6d0abece35b8256baf2527`,
1605 unique DNG filenames, all iPhone12Pro/4032x3024/iOS17.5.1. Filename grouping yields253
scenes, not the webpage's300: D/F/S/O/W each253, C252, plus88 mixed-suffix records.
All records report Linear Raw and NoiseReductionApplied≈0.95. These are1605 RAW metadata
records, not1588 verified pairs. Reconcile release completeness/splits and RGB derivation
before selecting images. Compact receipt: `evidence/sidl_metadata_lead.json`; raw metadata
stays on LaCie, no personal/location fields copied into the repository. Image downloads0.

### 2026-09-13 — MIDD availability lead, no image acquisition

The [official CVPR2024 implementation](https://github.com/rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024)
links to an accessible [MIDD public share](https://download.ai-benchmark.com/s/Gq3n2cS7QkH7ZMz).
Its browser inventory reports20 sensor ZIP files, total331.3GB (rounded display sizes).
Visible examples include ISOCELL_3P9.zip5.4GB, ISOCELL_3J1.zip5.8GB,
ISOCELL_3T2.zip8GB, Sony_IMX258.zip9.3GB. No file was downloaded, decoded or scored.
The code repository declares academic/research CC BY-NC-SA4.0; dataset-specific terms,
full-resolution versus patch packaging, noisy/ground-truth identity and scene/device
mapping must be checked in the release before any admission. Do not treat hundreds of
thousands of denoising pairs/patches as independent camera scenes or use denoised targets
as native ISP photos. Bulk331GB acquisition is not justified for the active E71 experiment.

MIDD ISOCELL_3P9 metadata probe: direct public endpoint advertises5,774,265,212 bytes and
ETag4173e7cad09c6a62571845663336995b. Exact HTTP ranges read232,439 ZIP-directory bytes,
revealing842 training/original JPEGs plus842 denoised partners,79 test/original PNGs and79
partners. No image member was read. The archive's own LICENSE.txt is CC BY-NC-SA4.0,
20,850 bytes /SHAe66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd;
license-only range pass238,707 bytes including repeated directory metadata. Native dimensions,
scene grouping and originals' processing remain unverified until admitted pixel audit.
Catalog SHA608a0822d29ab42736849a74392cb36f832d0b50650746759d7634ba4a2fb3df,
external `research/midd/ISOCELL_3P9_directory.json`. Sparse member acquisition is technically
available, making a bounded TRAIN-only source expansion feasible without331GB bulk transfer.


MIDD four-vendor metadata verification completed: Hynix_SL846868, ISOCELL_3P9842,
Sony_IMX258485 and OmniVision_OV32A763 official TRAIN originals (2,958 total).
Exact archive identities and directory digests are in `evidence/midd_catalog_lead.json`;
all four embedded LICENSE.txt bodies match CC BY-NC-SA4.0. Total directory/license
range transfer874,873B; image members read0. Separate official test and denoised paths
are excluded from any proposed TRAIN acquisition. Next: bounded score-blind original
selection, per-member integrity/resume and scene/overlap audit before TRAIN admission.


### E72 bounded MIDD research TRAIN acquisition plan (2026-09-13)

Freeze128 SHA256(E72|filename)-ranked publisher-original TRAIN JPEGs per preselected
Hynix_SL846/ISOCELL_3P9/Sony_IMX258/OmniVision_OV32A sensor:512 members,
3,821,581,226 uncompressed original bytes. No reserve/refill, official test or denoised
partners. Selection uses only filenames/metadata; no detector scores or content filtering.
Two workers,90min execution budget, AC and20GiB free reserve. Pinned archive ETags,
If-Match exact206 ranges, member ZIP CRC/length and SHA256 receipts, verified member resume.
Bulk fallback rejected before reading its body. Code's8 focused tests passed, including
wrong range/ETag, truncation, excess transfer, entry identity and test/denoised/path exclusion.
All images initially QUARANTINE; separately freeze canonical decode/overlap audit including
E52/later reserves/E65 and consumed E66 before TRAIN admission. Unknown filename scene links
are not independence; whole MIDD publisher excluded from DEV/fresh final. Research license
CC BY-NC-SA4.0, no automatic serving eligibility. E71 recipe/population stays frozen.


E72 admission code prepared: canonical whole-body/RGB plus dHash/pHash overlap against
all frozen references and explicit320-row consumed E66; verify all2,385 reserved AI bodies
and current TRAIN are covered by protected snapshot. Internal similarity and exact same-sensor
EXIF capture-second links form transitive components. Propagate protected matches through the
whole component; keep one deterministic representative, no replacement. Scene independence
remains unverified. Three focused transitivity/time-link tests passed; full suite765passed.
Audit freeze waits for complete acquisition receipt. E71 DEV failure evidence is now checkpointed.


E72 bounded MIDD acquisition complete:512 publisher original TRAIN JPEGs,128 per four
sensors,3,821,581,226B. All archive/member identities verified; no official test/denoised
members acquired. Quarantine pending canonical protected-role/component audit; no scores.


### E72 score-blind MIDD TRAIN admission complete (2026-09-13)

All512 originals decoded;151,485 protected references, zero cross-reference matches,
zero decode failures. One internal perceptual pair formed a component; retain one hash-ranked
representative,511 admitted (Hynix127, other3 sensors128). No refill. Native publisher
JPEG dimensions:256 at3264x2448,128 at2320x1744,128 at4208x3120. EXIF make/model and capture
timestamps absent in all512, so sensor provenance comes from publisher packaging and scene
independence remains unverified. Whole MIDD publisher is research TRAIN only, never fresh
DEV/final; no detector scores yet. Audit SHA`23021f21e0ff5fcb72c972c8293a0c18188ddf18cdf8fdfb33ec3046ac5d8e04`;
TRAIN manifest SHA`a258b4362e45016762ce1d557cb2ac87f82fe29f78d598559caf216b551e0d8e`.
E75 feature contract may now freeze for all511 admitted representatives. Existing E54 TRAIN
remains immutable; a future expanded fit would have12,141 parents /7,546 REAL /4,595 AI.

DEAR lead (2026-09-13): metadata/code/terms only, no new images or weights. Model revision
5b57350b0ee75553109b3844f3c8a9341fc7e707 offers94,372,114B c/r checkpoints under
CC BY-NC4.0 plus stated SD1.5 use restrictions; NOTICE records unresolved explicit licensing
for some upstream AlignedForensics sources. This is a possible research expert, not an
admitted dataset/model or serving replacement. Complete pinned receipt in evidence.


### Future authentic-capture source leads, no pixel acquisition (2026-09-13)

The metadata-only E84B inventory records REAL7,034 JPEG/1 PNG/511 unspecified and
AI786 JPEG/3,809 PNG. This association motivates examining lossless/RAW capture coverage;
it does not establish a shortcut or alter the active E84B/E85/E86 protocol.

- [SID official repository](https://github.com/cchen156/Learning-to-See-in-the-Dark)
  publishes short/long RAW pairs and scene/burst identifiers. Sony/Fuji archives are
  advertised25/52GB; RAWpy-rendered16-bit reference RGB alternatives12/22GB. Multiple
  short exposures share a long reference, so image count is not independent scene count.
  README states MIT; [license](https://raw.githubusercontent.com/cchen156/Learning-to-See-in-the-Dark/master/LICENSE.md)
  uses software/documentation wording. Record that scope rather than inventing separate
  image terms. Original captures or explicitly documented RAWpy references could supply
  REAL; network-enhanced outputs must not. The September2025 storage-cost note asks
  users to retain one local copy and avoid repeated download per experiment. Source tree
  revision0448b58d8c57459b1550ddf68c160d9212046a25 has six split lists (~367KB).
  Next: metadata-only scene/pair inventory; no images acquired or role admission yet.
- [PolyU original release](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
  lists40 scenes/five cameras and100 crops. Original noisy and averaged reference files
  are JPEG, so this is not direct lossless coverage. Its
  [license](https://raw.githubusercontent.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/master/License.txt)
  restricts use/redistribution to noncommercial purposes and also refers to software.
  Prefer original captures; averages/crops are dependent derivatives. Not acquired.
- [RENOIR author page](https://ani.stat.fsu.edu/~abarbu/Renoir.html) is a lead for CanonT3i,
  CanonS90 and XiaomiMi3 capture groups. Search exposes raw/aligned archives but two
  direct reads timed out. Exact current inventory, rights and rendering provenance remain
  unresolved; no download or admission. Do not rely on mirror terms or claim full review.

Whole SIDD remains consumed DEVELOPMENT; WIFD/RawNIND remain diagnostic-only. No source
above has been added to TRAIN, DEV or final. No author contact or external messages.


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


### E99 — four additional MIDD sensors, acquisition selected before pixels (2026-09-14)

Why: enlarge REAL capture-pipeline coverage with original mobile photographs from four
sensors not present in E72's four-sensor acquisition. Use [official MIDD release](https://download.ai-benchmark.com/s/Gq3n2cS7QkH7ZMz),
linked by the [CVPR2024 authors](https://github.com/rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024).
Research terms: CC BY-NC-SA4.0; preserve attribution, no production/weight redistribution
claim. This is the same publisher already used in TRAIN, so never fresh source-OOD evidence.

| Sensor package | Full archive bytes (metadata only) | Requested image subset | Status |
|---|---:|---:|---|
| ISOCELL_GN1 |21,661,904,344|64 original TRAIN JPEGs|Directory validation pending|
| ISOCELL_HM3 |19,242,794,655|64 original TRAIN JPEGs|Directory validation pending|
| OmniVision_OV64B |27,181,871,080|64 original TRAIN JPEGs|Directory validation pending|
| Sony_IMX766 |20,864,102,442|64 original TRAIN JPEGs|Directory validation pending|

Full archives will not be downloaded. Hash-ranked selection, at most4GiB image bodies,
no test/denoised partners, no quota refill. Exact bytes/ETags/member CRCs are frozen before
image transfer. Root: `$PIXELPROOF_DATA_ROOT/e99`; all images stay outside Git. Initial
role QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN; protected-overlap/group/decode admission
and a new experiment contract are required before fitting. Completed image count is0
at this registration. New download and failure receipts will be appended below.


### E99 transfer started — exact selection frozen (2026-09-14)

All four central directories and archive LICENSE.txt files verified before image bytes.
Contract SHA256 `0f67891068521533355ec39a90e3d4b396c14bb77b42fff9fc29a77745703729`. Requested256 files /3,323,883,735B
(3.324 decimal GB). Two exact-range workers; no full archive download. Sensor body totals:
GN1 913,893,564B; HM3 729,865,949B; OV64B 974,869,735B; IMX766 705,254,487B.
The four archive identities include exact size, ETag and Last-Modified in
`evidence/e99_acquisition_contract.json`; complete selected member names/CRC/offsets stay
under `$PIXELPROOF_DATA_ROOT/e99/acquisition_contract.json`. Stage status: TRANSFERRING,
not admitted or scored. Each verified body receives a persistent SHA256 receipt; failed
runs retain verified members for resume. Completion totals will supersede this entry.

CSAFE additional-package probes returned403; institutional endpoint returned a WAF
challenge. No bypass or image transfer attempted after that response. SIDL remains
pending the already-documented253-versus300 scene-release reconciliation; no new SIDL
metadata or image acquisition is claimed. E98 current metadata transfer: CSAFE article
6,622B plus official MIDD DAV listing and ZIP directories/licence text, not photos.


<!-- E99-LIVE-START -->
### E99 current download snapshot

Updated UTC: 2026-09-14T14:56:37+00:00. **COMPLETE — quarantined, admission separate**.

| Sensor | Requested files | Verified files | Verified body bytes |
|---|---:|---:|---:|
|ISOCELL_GN1|64|64|913,893,564|
|ISOCELL_HM3|64|64|729,865,949|
|OmniVision_OV64B|64|64|974,869,735|
|Sony_IMX766|64|64|705,254,487|

Total: 256/256 files; 3,323,883,735 verified body bytes.
Counts use exact selected receipt paths; AppleDouble sidecars/partial files do not count.
This snapshot verifies receipt binding and current file size; original CRC/SHA checks are
performed by the acquisition worker and are rechecked during admission. Image bodies stay
under `$PIXELPROOF_DATA_ROOT/e99/images`. No new model score or quality claim.
<!-- E99-LIVE-END -->


E98 role audit completion: all6,619 E32-derived E92 TRAIN records map to original TRAIN
by record identity/body; no mapped CAL record or shared available source-scoped group.
R1b CAL5,333 remains historically consumed, and C3 CAL4,534 is an overlapping earlier
manifest. No images were downloaded for this role audit. Do not sum both CAL counts
or call either a newly independent test. No role admission or model fit follows solely
from an exact-identity pass. Receipts are evidence/e98_{role,lineage}_audit.json.


### E99 download complete — 2026-09-14

All256/256 original MIDD JPEGs verified,64 per registered sensor. Stored source bodies
3,323,883,735B; exact compressed ZIP ranges received3,310,702,089B in the completed run.
No whole archives or denoised/test partners downloaded. All archive size/ETag checks,
member CRC/length checks and stored SHA256 receipts passed. Receipt SHA256
`6dab93cf4bddac6f670f927e8d53ff217efd9ba4c5bf80f6a1f9261f8f5bfacc`;
public summary `evidence/e99_download.json`. These256 files are still quarantined at
this transfer checkpoint, not256 admitted training examples or correct predictions.
E100 next repeats body hashing, native decoding and protected/scene-group checks.


### E100 native admission complete — 2026-09-14

All256 downloaded originals decoded successfully and retained native resolution. Against
152,124 protected/reference records, including every206 unique owner-gallery image:
zero byte/RGB/declared perceptual overlaps, zero internal perceptual links and zero
quarantined/removed images. All256 admitted to separate research TRAIN,64 per sensor;
137.118s. These heuristic checks do not establish256 independent scenes. No classifier
scores, new fit, calibration change or independent-final evaluation. Audit SHA256
0a403cb8c9acf755622f625c42e0a91a67b187ec77e63c999bbebbeabc9c031c;
TRAIN manifest SHA256c5abc769d1b430bba52e8e5ae8c7ae90fdeb1cbbc56c7a40d4bc01f458ee496e.
Public receipt: evidence/e100_audit.json. Local bodies remain under e99/images;
admission records under e100/training_manifest.json. Training availability is not a
model-quality improvement. The earlier quarantined snapshot describes acquisition only;
this later admission receipt supersedes that role status for all256 originals.


E100 decoded image geometry:64 each at4080x3072,4000x3000,4576x3168 and4096x3072
(after the fixed orientation convention). All are full-photo dimensions; no resizing
was performed for admission. Camera model EXIF was absent from all256 records, so sensor
identity relies on publisher packaging. Do not report256 verified physical devices or
independent scenes. The E101 derived crops are separate training views; originals remain
unchanged. New training features/model fits will be logged in EXPERIMENTS/HISTORY.


E101 derived-feature availability: all256 admitted originals now have four fixed training
views (1024 total), with DINO/CLIP/DEAR caches complete. This adds no source photos and
must not inflate the downloaded count. Feature receipt: evidence/e101_features.json.
The unchanged original3,323,883,735B remain the acquisition total.


E102 use/outcome annotation: all256 newly admitted parents were used with the12,269
previous TRAIN parents in one registered frozen-map correction. On this TRAIN source,
both pre-fit E92 and fitted E102 give0 false AI indications in each of4 conditions.
This is a neutral data-coverage observation, not256 newly corrected gallery errors,
independent validation or proof of a universal detector. Full fit and later regression
results belong to EXPERIMENTS/HISTORY; original file counts and acquisition rationale
above remain unchanged.


### E102 downstream evaluation outcome — 2026-09-14

The full 256-image E99/E100 acquisition was used in the registered correction fit;
no further source photos were downloaded for E101 features or E102 evaluation.
Consumed DEV social-Q75 REAL false alerts changed from 14/160 to 12/160; original
REAL false alerts stayed 0/160 and AI detection stayed 159/160 in both conditions.
The candidate passes 20 numeric gates but fails mandatory retention of all E43-caught
original AI (one unresolved miss). It was not promoted or tested on the owner gallery.
This does not isolate a causal benefit of these 256 photos: E92 already made zero
false AI calls on them, and the correction was fitted on the combined TRAIN population.
Source, licence, 256 originals / 3,323,883,735 stored bytes and research-TRAIN role remain
as recorded above. More downloads are allowed, but another acquisition must address an
identified gap and receive its own requested/completed ledger and admission checks.


### E103 planned reuse — 2026-09-14

No dataset download or new source admission. Reuse the identical E102 TRAIN population:
12,269 prior parents plus all 256 E99/E100 MIDD originals, four fixed views per parent.
Roles, licences and original-byte counts remain unchanged. E103 tests a joint REAL/AI
objective; derivatives are not new downloaded photographs. Existing DEV/gallery/CAL
records stay excluded from fitting. E103 results will be logged after the frozen run.


E103 completion: no new photographs or bytes downloaded. All 256 added MIDD originals
remain TRAIN-only. The joint objective clears missed AI on TRAIN but leaves consumed
DEV binary results unchanged from E102 and fails the same E43 retention requirement.
E104 is a score-blind audit of previously downloaded legacy TRAIN raw CLIP feature chunks;
no new source acquisition, licence change, image decoding or role reassignment.


E104 completed using 30 previously stored TRAIN parent chunks / 90 views only. Exact
raw/aggregate identity checks passed. Download count and bytes remain zero for E103–E104;
no source photos changed roles and no gallery/DEV/final images were read. The proposed
next raw-feature cache extension uses existing original images, pending complete inventory
and a separately registered extraction cost/parity probe. Feature vectors are not new
source photographs and must not be counted as additional downloaded data.


### E105 acquisition registration — DiffSeg30k, 2026-09-14

Purpose: expand Model2's mask/provenance audit beyond the old GLIDE-only collection,
covering multi-turn local diffusion edits and masks encoded with generator labels1–8.
Primary paper: https://arxiv.org/abs/2511.19111; author-linked release:
https://huggingface.co/datasets/Chaos2629/Diffseg30k . Pinned revision
`8049819755ff430dcf4701636b14e6f1df6c4542`, dataset-card licence Apache-2.0.
Underlying COCO image and generator-specific terms/ancestry remain separate; initial role
QUARANTINE_MODEL2_RESEARCH, not TRAIN/CAL/final and never automatic Model1 REAL negatives.

Select512 hash-first image identities from official train.zip with their corresponding
masks; no validation images, model-score filtering or refill. Official archive size
21,563,023,972B, publisher LFS SHA256
9ef27f4c1c35ef21bce150435312de18fa35cf00df6d622271219455a4ee3d02.
Only exact selected ranges will be fetched, not the whole archive. ZIP central-directory
metadata:57,998 members /6,644,552 transferred bytes, no image member read at inventory.
Store at `/Volumes/LaCie/pixelproof-datasets/e105`; image/mask originals never enter Git.
Record exact selected body bytes and receipts when frozen/completed. Admission requires
mask values/geometry, grouping and cross-module protected-overlap checks. Public samples
are not independent merely because their hosting publisher is new.

Unselected leads: TGIF2 official FLUX public-share DAV returned401 through documented
access, no image downloaded; stopped without bypass or interactive login. NTIRE2026 TRAIN
metadata inspected at revision700b6d08a3268b1e7a191306dec7321dd953b12f; no explicit licence
found in inspected card metadata, so no image acquisition or role assignment. Its test
labels are not publicly supplied in the inspected official instructions. PLAN.md contains
primary references and the evidence implications of both leads.

E105 exact selection frozen:512 images +512 corresponding masks;374,620,852 source-body
bytes requested (about375MB). Contract SHA256 `c8c66ed19f9b93d5a0acc7aefab98f5bc5deccf794df99fda911e3ea4148f303`.
Two bounded workers started; no complete download or model admission claimed yet.

### E105 completion / E109 schema outcome — 2026-09-14

Author-hosted DiffSeg30k acquisition completed automatically without login or manual
computer interaction: **512 images +512 masks**, **374,620,852 source-body bytes**;
387,608,597 range-transfer bytes in the execution plus6,644,552 earlier catalog bytes.
Elapsed1,449.92seconds. Repository `Chaos2629/Diffseg30k`, pinned revision
`8049819755ff430dcf4701636b14e6f1df6c4542`; native files under
`/Volumes/LaCie/pixelproof-datasets/e105/files`. ZIP member CRC and individual SHA256
verified; the21.56GB whole archive was not downloaded and its publisher SHA was not
recomputed. Public receipt: `evidence/e105_download.json`.

Why: the author's multi-turn, eight-editor image/mask corpus can expose generator and
editing-scope gaps in Model2 that single-generator CocoGlide cannot. Requested TRAIN
subset chosen by fixed filename hash, without inspecting scores; no validation images.
Author card Apache-2.0, with underlying COCO/model rights and ancestry still unresolved.

E109 decoded all512 pairs at native geometry and checked the documented uint8 IDs0–8.
Edited pixels are mask>0, not mask>127. **201 partial masks,224 full-image positive masks,
87 empty masks**; no duplicate image bodies or decoded RGB arrays inside this subset.
All eight positive model IDs appear; a pair may contain several IDs. Among partial masks,
7 cover1–5%,50 cover5–20%,80 cover20–50%,64 cover50–100%; none cover<=1%.
These are materially different localization cases. Full-positive masks cannot supply
within-image positive-versus-background AUC; empty masks cannot be automatically labelled
as authentic photographs. Keep every file and report these strata, not a refilled or
silently filtered subset. No data corruption was inferred merely from a degenerate mask.

**Role remains QUARANTINE_MODEL2_RESEARCH**, zero training/validation/classifier use.
Next: establish original COCO/prompt ancestry and the interpretation of empty/full masks,
audit cross-role overlap, then assign eligible whole-parent research roles. This download
is not fresh independent proof and not a new pool of authentic Model1 negatives. Audit:
`evidence/e109_diffseg_audit.json`; detailed local rows remain outside Git.

### E111 cross-role audit — 2026-09-14

Compared1,536 candidate Model2 images (512 CocoGlide edits,512 authentic pointers,
512 DiffSeg30k derivatives) against152,380 frozen existing reference fingerprints.
All decoded. **45 CocoGlide parent groups** produced252 joint perceptual matches
(dHash<=4 and pHash63<=4); none were exact body/RGB matches. Conservatively quarantine
those parents and all derivatives:14 in the previously exposed120,31 in the other392.
No DiffSeg image matched the snapshot, and no internal cross-parent matches were found.
These negative findings do not prove ancestry independence; numeric compiled authentic
names do not identify original COCO samples. Keep the entire new acquisition in research
quarantine until ancestry and empty/full-mask interpretation are resolved. No classifier
training or independent test admission occurred. `evidence/e111_model2_lineage.json`.

### E112/E113 existing-data usage — 2026-09-14

No new download. Reuse the complete12,525 previously admitted TRAIN parents across
legacy/MIDD/SID/additional-MIDD cohorts. Extract15,210 missing raw CLIP views and preserve
34,890 old views; the50,100 views are transformations, not50,100 independent photographs.
Raw vectors, context features and candidate weights stay under the ignored external
E112/E113 work directories. No gallery, CAL, DEV or final images are used for these fits.
DiffSeg30k/Model2 candidates retain their existing quarantine status.

### E115/E116 author CocoGlide provenance recovery — 2026-09-14

E115 inspected the ZIP linked by the [official TruFor repository](https://github.com/grip-unina/TruFor).
Only167,412 range bytes were transferred; zero image members. The123,161,479-byte
archive is bound to ETag `"7574b87-60269d48b6230"` and Last-Modified2023-08-08.
It contains1,536 PNGs, `table.csv`, `README.txt`, and `licenses.json`. The table provides
512 real/edit/mask triples and prompts with COCO val2017 IDs. License metadata covers
512 unique originals:201 CC BY-NC-SA,131 CC BY,113 CC BY-NC,67 CC BY-SA. These labels
must be preserved with each source image, not replaced by the software licence.

E116 is registered to acquire this complete~123MB author package for **provenance only**.
Why download: verify exact canonical authentic/edited/mask pixels against the already
compiled512 triples and recover original COCO IDs/prompt/licence mappings. This repairs
lost lineage; it is not512 new independent training examples. Join only unique exact
three-component pixel identities, never filenames or table order. Keep E111's45 near-match
parent quarantines and no automatic TRAIN/CAL/final admission. Archive body will be bound
by local full SHA and CRC plus author headers; no publisher full SHA is advertised.
Local storage: `/Volumes/LaCie/pixelproof-datasets/e116`. Download outcome recorded below
when verified; E115 receipt is `evidence/e115_cocoglide_metadata.json`.

DiffSeg30k follow-up: the [author paper](https://arxiv.org/html/2511.19111v1) explicitly
includes both real COCO bases and generated bases from COCO prompts. This reinforces the
need to retain separate base/scope labels; do not reinterpret every full/empty mask as
corruption or an authentic negative. Original-parent linkage is still unresolved there.

### E116 completion and E117 original-parent overlap — 2026-09-14

Author package download completed:123,161,479bytes in14.98seconds; local full SHA256
`fc848643ad0a81010ba2a6ab8c055762782f54e4fb9e407210dbfe1c3b538af9`.
All ZIP member CRCs/sizes and catalog identities verified. E116 recovered **512/512
unique exact authentic/edit/mask pixel triples**, joining compilation records to author
COCO val2017 IDs, prompts and per-original licence metadata. The earlier45 near-match
quarantines were retained. No ordering/filename guess was used to link the compilations.
Receipts: `evidence/e116_cocoglide_download.json`, `evidence/e116_cocoglide_lineage.json`.

E117 compared these recovered originals with the already protected, SHA-bound DDA-COCO
manifest:34,755 records representing4,965 COCO val2017 parents. **506/512 CocoGlide
parents overlap exactly by original COCO ID**, including all120 previously measured
pairs; each has7 protected original/derived records. **461 additional parent groups**
were missed by the earlier perceptual-only test because different generated derivatives
need not resemble each other. The6 unmatched originals are not certified independent
of all other data and do not justify a six-image training/validation claim.

Keep506 original groups and every descendant out of new Model2 TRAIN/CAL admission
under the current protected-parent policy. The corpus remains consumed diagnostic
material. This is a lineage overlap finding, not a claim that the historical crop model
was necessarily trained on those exact images. No Model2 fitting or protected pixel
reads occurred in E117. Receipt: `evidence/e117_coco_ancestry.json`.

## E118 — generator assets for traceable Model2 research pairs (2026-09-14)

Why: recovered CocoGlide original identities overlap protected parents, while DiffSeg base ancestry is unresolved. A local inpainting pilot can preserve known TRAIN-only parent and mask provenance. Acquire only16 required files from the public `stable-diffusion-v1-5/stable-diffusion-inpainting` mirror at revision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` (under3GiB, exact bytes in the contract). Download four fp16 safetensors with publisher LFS SHA256 checks, required tokenizer/configs with Git blob checks, and retain the safety checker. No pickle or remote repository code is loaded. Resumable downloads validate final full hashes.

This [community mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) explicitly disclaims affiliation with Runway. Retain the [CreativeML OpenRAIL-M licence](https://huggingface.co/spaces/CompVis/stable-diffusion-license) at licence-space revision14d42d09bffd871b1666a084fc954a50cff72ac0,14385bytes. Model assets are for a local research pilot; original MIDD image terms remain CC BY-NC-SA4.0. Record all derived assets outside Git. No generator or input-image upload, paid API, automatic publication or new evaluation-role admission. Asset download is not generated data or a detector improvement; actual byte totals follow the receipt.

### E118 acquisition / E119 preparation outcome

Downloaded and verified all16 model files:2,742,261,613bytes transferred in344.59seconds, plus14,385bytes of the pinned licence text and5,911,654bytes for the isolated Diffusers0.40.0 wheel (SHA256 `5b5da7c3ddb62152fa4afc577f02e050af688c797375c34fb2d01006da3f3541`). The wheel matches PyPI metadata; it is installed only under ignored `ml/work/model2-runtime/site-packages`, leaving the active detector environment untouched. E118 contract `b07af76fb6961c63482c7163e703676808b8c83d6c617fee4895ad1c5318890c`; public asset receipt `evidence/e118_inpaint_assets.json`.

E119 prepared16 existing native MIDD TRAIN originals, two per eight sensors, without downloading original images or using detector scores. Source manifest, parent/scene identities, source-body hashes, exact512px crop, deterministic mask, same-mask GaussianBlur6 control and zero AI targets are retained locally. Mask fractions10.10–16.99%; all variants remain research TRAIN descendants. Preparation SHA256 `2721b06c3e2ab866caee8fa87fc5b2ff36b756422afe0399c6a19d5bc26d5695`. Generation is registered but pending Model1 GPU release; no generated pairs are claimed yet.

E119 isolated-runtime completion: downloaded importlib-metadata9.0.1 (27920bytes) and zipp4.1.0 (10238bytes), each checked against PyPI SHA256; these are software dependencies, not image data. The supplemental receipt is evidence/e119_runtime_dependencies.json and the reproducible isolated pins are ml/requirements-model2-pilot.lock. No base-environment upgrade or change to registered E119 image/generation settings.

### E119 automatic engineering pilot result

{"contract_sha256": "80721e3928d94f618ce4525e50f7f3b51de04bdd15ec31f0c36217120ea85e80", "detector_scores": 0, "limits": "16 already-used TRAIN parents, one old editor, correlated sensor scenes and fixed prompt. This is an engineering pilot. No calibration, independent evaluation, universal claim or automatic training/serving admission.", "mean_raw_background_changed_fraction": 0.9998254416167689, "parents": 16, "passed": false, "passed_parents": 1, "peak_mps_bytes": 4143857664, "promotion_allowed": false, "result_sha256": "bba363a9c710dd094c09e9717a648b4a5b3024acd7d4c3002a482e227ef61abf", "seconds": 273.0622565409867, "state": "E119_inpainting_engineering_pilot_complete", "training_admission": false}

This measures generation feasibility only. Spatial-head training requires a new protocol; no independent detector evidence or serving change.

E119 generated16 attempted outputs, only1 meeting its engineering checks.15 black outputs failed; logs show nonfinite conversion warnings. No generated data enters training. Raw out-of-mask changed fraction~99.98% includes invalid black outputs and must not be described as clean VAE drift evidence. Existing image/mask/source identities are retained. E120 reuses two fixed diagnostic parents without downloads to identify numerical failure before any complete re-generation.

E120 numerical diagnosis used only existing parents0/1. The fp16+sliced failure originates in UNet tensors; retained guard stops before rendering, so no content-based safety conclusion is made. E122 replays all16 original parents with the preregistered fp16+SDPA correction; same input data and no downloads. Neither failed black outputs nor diagnostic raw full-frame outputs are admitted as local-mask training pairs.

### E122 complete numerical replay result

{"branch": "fp16_sdpa", "contract_sha256": "353836702bcf414116aa3f2669de92a90081ba1f930b1cd85858afe3c8746789", "detector_scores": 0, "limits": "Engineering correction of numerical failure, not improved detector performance or a semantic-quality audit. Only after16/16 pass may a separate TRAIN-only spatial learning experiment be registered. All original research-only licences and ancestry roles remain.", "mean_valid_raw_background_changed_fraction": 0.9994174467463207, "parents": 16, "passed": false, "passed_parents": 11, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "report_sha256": "cb0c713de246cb20e8d4822948aa5dd34303d845d85deefdffeb6fb9c54cd373", "seconds": 487.82744045800064, "state": "E122_complete16_numerical_replay_finished", "training_allowed": false}

Original E119 remains failed; this is a separately registered correction. No detector performance or independent evidence is claimed.

E122 completes16 numerical replays:11 valid engineering outputs,5 safety-filter rejections, no nonfinite tensor events. No new downloads and no training admission. For the11 valid raw outputs only, mean out-of-mask changed pixel fraction is99.94%; raw diffusion outputs cannot be treated as locally masked truth. Only explicit composites preserve background exactly. E123/E124 reuse existing complete TRAIN originals for new full-frame features, not new images or changed dataset roles.

### E126 data-use plan — 2026-09-15

No new download. Reuse the existing SHA-bound 640 consumed-development views only after complete passing E125 TRAIN guards. Add full-frame CLIP features without reassigning roles, fitting on DEV, reapplying social compression, or opening gallery/final data. E114 raw features serve only as an identity/numerical replay reference. Model2 E122 remains quarantined after its failed 16/16 gate; a planned read-only pixel audit will not admit its 11 accepted generations to TRAIN.

### E127 completed — pixel magnitude and compositing controls (2026-09-15)

Frozen contract `8b5e1efb01339e92d7450d7ae5e04fc9a24c8bc580c3bc362a33cdae81bf082d`; all 16 attempts accounted for, 11 accepted pairs measured and 5 safety-filter/blank placeholders explicitly excluded from generated-content metrics. CPU-only audit finished in 1.73s. All 16 classical controls preserve background exactly and have zero AI targets; all 11 accepted composites reproduce the exact registered hard-mask operation.

On the matched 11-parent subset, raw generator outside-mask MAE is 0.038381 in [0,1] (about 9.79/255 channel levels), with per-parent MAE range 0.015261–0.115276. Outside changed-pixel fraction remains 99.9417%, exactly replaying E122. These measure different properties: many changed pixels do not by themselves imply large or semantic changes, while the measured magnitude still prevents labelling raw background untouched. Composite outside MAE is exactly 0.

Mean mask-crossing RGB gradient is 0.030659 in authentic inputs, 0.042073 in classical controls, 0.028218 in raw generator outputs and 0.044219 in hard composites. Composite gradient exceeds the corresponding classical control in only 5/11 pairs. Classical controls therefore also expose the boundary shortcut; these small correlated-sample summaries do not establish separability, semantic quality or actual learned detector bias. Statistics are equal-parent summaries, not independent population estimates.

Decision: retain classical-edit negatives and require boundary-versus-interior localization evaluation in a future spatial-head protocol. Keep raw outputs diagnostic, and preserve exact mask/provenance/encoding matching. The failed E122 16/16 gate remains failed; no output is newly admitted to training and no safety filter is disabled. No detector score, GPU operation, download or serving change. Aggregate receipt: `evidence/e127_pixel_audit.json`; per-case measurements stay local.

### E128 same-parent resized-context preparation — 2026-09-15

No download or new source admission. Reused all16 admitted E119 native MIDD TRAIN originals, preserving original parent/body identities, research-only licences,16 seeds, masks and zero AI targets. Produced32 lossless PNG inputs (resized authentic plus classical-edit control), totaling15,370,680bytes. Each starts from the complete native center square (69.23–75.29% of source area), LANCZOS-resized to512 without aspect distortion. These are resized research derivatives, not additional independent/native-camera photographs. Prepared manifest SHA256 `d2114aee3c1c2fa579bb1db25cf456beaf571119b6d080405f41d773ce785dcb`.

Neighbour-difference texture proxy averages0.038142 on original native crops and0.032809 on resized context; this is neither a calibrated noise estimate nor a selection/quality gate. The transformation changes both field of view and resampling. E129 has separately registered all16 generation attempts with the same E122 fp16+SDPA weights, checker, prompt, masks and seeds; it waits for Model1 GPU release. Neither preparation nor future generation automatically admits training data.

Read-only external research: RAISE's official small/custom download route requires a name/affiliation/email form despite its public research terms. Do not invent registration details or bypass the form; no RAISE payload acquired. Its older three-camera RAW coverage is supplementary, not a modern-phone solution. MNW and HDR+ are already acquired protected evaluation reserves; retain their existing roles and avoid a duplicate acquisition. A newly found AI Detector Arena collection has mixed per-component terms and Unsplash-derived REAL labels; provenance and metadata would need separate assessment before use. No images from these leads downloaded.

### Overnight source-access findings — 2026-09-15

The [SIDL author-linked train folder](https://drive.google.com/drive/folders/1chLRpBtGzTkAX_7v-ycPgL_X1-mezv2r) lists `train_full.tar` at27,003,873,280bytes. The normal anonymous download endpoint returned HTTP200 HTML titled "Google Drive - Quota exceeded", not an archive. Downloaded only362,334bytes of folder HTML and2,009bytes of quota-response HTML with TLS verification enabled; **zero image/archive payload bytes**. Preserve these local access receipts and public aggregate `evidence/sidl_access_2026-09-15.json`. Do not evade the provider quota through alternate split-file IDs. The prior1605-entry/253-group metadata versus300-scene release discrepancy remains unresolved; no role admission or repeated immediate retry.

[DND](https://noise.visinf.tu-darmstadt.de/) offers50 paired real noisy/reference images from SonyA7R, OlympusOMD E-M10, SonyRX100IV and Nexus6P. Its1000 evaluation patches are descendants of50 images, not1000 independent scenes. The [download route](https://noise.visinf.tu-darmstadt.de/downloads/) requires a login; no account or image download was attempted. [Terms](https://noise.visinf.tu-darmstadt.de/license/) restrict use to noncommercial research/teaching/personal work and prohibit image redistribution. Reference processing/alignment and custom sRGB development prevent treating every reference as an untouched camera JPEG. DND is a potential supplementary noise diagnostic, not broad modern-camera evidence. RAISE's small route likewise requires identity/affiliation/email registration. Neither access limitation blocks experiments on existing admitted data.

### TRAIL implementation reference — 2026-09-15

Read five small code/config/license/aggregate-result files from the [author repository](https://github.com/VishalJ99/trail-image-edit-localization) at revision`0fce0f1de2ac0d219baf91592ee2bca95ec0a219`:27,393bytes total, exact individual SHA256 in `evidence/trail_reference_receipt.json`. No image, checkpoint or remote code execution during acquisition. Adapted numerical primitives retain the authors' MIT notice at `ml/third_party/TRAIL_LICENSE.txt`; upstream dataset/model terms are separate. E130 will reuse existing local DINO weights and E128/E129 TRAIN-research descendants only, retaining all16 authentic/classical controls and explicitly accounting for rejected generations. No protected CocoGlide/TGIF2 pixel acquisition, no new source independence and no training admission.

### E131 internal-use plan — 2026-09-15

Reuse all12525 existing TRAIN parents and four cached conditions without new image acquisition. Keep RR topics and REAL pool in one publisher group, E36 shared AI prompts plus declared cross-version family links together, MIDD all eight sensors together and SID both cameras together. Preserve inherited E53 near-duplicate components plus known original-body/pixel/scene links. The resulting10 declared groups include only3 AI-bearing groups. This is not10 certified independent datasets; unknown CommunityForensics/RR generator or semantic ancestry and encoder pretraining remain limitations. Internal fold FIT/held-out usage stays under global TRAIN, with no admission to CAL/DEV/final and no protected or gallery pixels used. Numerical fitting is pending a separately frozen E131 contract.

### E131 pre-registration scope clarification — 2026-09-15

Before any E131 contract or fit, rechecked the existing E79 provenance note and [RR's primary paper, sections3.2.1–3.2.2](https://arxiv.org/html/2509.09172v1). RR explicitly includes FLUX, DALL-E and SD-family generations, COCO/CC3M-derived prompts and Chameleon material at corpus level. The local admitted rows lack the file-level mapping needed to assign those origins. CommunityForensics is also a mixed-generator corpus. Therefore the10 declared row/publisher/prompt components **do not establish generator-family independence**. Conservatively connecting every possible mixed-corpus relation could collapse the apparent AI-bearing groups; their count is not a certificate of unseen families.

E131 remains useful as a publisher-group internal transfer diagnostic with explicitly incomplete upstream ancestry. Its contract and report now state `generator_family_holdout_supported:false` and retain the corpus-level limitation. Do not describe its grouped folds as unseen-generator proof, a fresh source benchmark or a qualified deployment candidate. This correction precedes registration/scoring; no result or threshold was selected. A true generator-held-out experiment requires file-level generator/prompt lineage or independently acquired clean families.

2026-09-15: E124 completed12525 TRAIN parents /50100 full-frame feature views with zero download. E125 fitted both branches on the existing complete TRAIN cache and passed its guards. E126 now reuses the existing640 consumed-development views under a separate frozen protocol; no source role is changed and no new independent benchmark is claimed.

### E129 fixed input-context replay completed

{"comparison": {"context_passed": 16, "full16_engineering_gate_passed": true, "new_failed_attempts": 0, "previous_passed": 11, "rescued_attempts": 5}, "contract_sha256": "15524ea7f66a1e52fd7a539b53f2b218968cf8899d383623aba4b18025a7116a", "detector_scores": 0, "downloads": 0, "limits": "TRAIN research engineering, one old generator and correlated MIDD parents. Resampling and scene context change together. A pass is not semantic-quality, localization-accuracy or independent-OOD evidence; a new spatial-learning contract remains necessary.", "parents": 16, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "report_sha256": "b07aaf1b361c6603ebcee78d3e6054198ab6d6d4f3022deaac94128f1106e76e", "seconds": 458.7882085829042, "spatial_protocol_may_be_registered": true, "state": "E129_context_generation_complete", "training_admission": false}

This is paired generation engineering, not detector accuracy. Preserve E122 and all failures; no automatic training/serving admission.

### E130 patch-drift diagnostic completed

{"accepted_composites": 16, "contract_sha256": "b9eb9ed0e1f1240e5f846cf7deec635b6db57fff48529f6d03e1d6f3a51aea78", "excluded_generations": 0, "limits": "Small correlated TRAIN research, one old editor, simple masks, conditional accepted-generation subset. This is an adapted edit-response baseline, not an exact TRAIL reproduction, calibrated AI probability, semantic annotation or independent detector evidence.", "max_duplicate_token_error": 0.0, "measurements_sha256": "c235cb22f5000405974063b951bb1c7a93f87d435acb3d455332d19bb1776bb0", "parents": 16, "peak_mps_bytes": 1219198976, "promotion_allowed": false, "scoring_sha256": "0b54f326768b00ce90029c2d87ef088acdd4a42825bf9d9edf796049c95616df", "seconds": 25.685287708998658, "state": "E130_patch_drift_audit_complete", "summary": {"jpeg75": {"accepted_parent_ranking_baselines": {"constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.5291163098010457, "radial_center_auc": 0.7975643577862963}, "accepted_parent_spatial_contrast": {"AI_minus_authentic_auc": 0.12430492180855968, "AI_minus_classical_edit_auc": -0.18437926993454085, "interpretation": "Within-image aligned-region ranking differences, not REAL-versus-AI classification accuracy.", "parents": 16}, "ai_composite": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.459066698329999}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.50054931640625}, "interior_background_auc": {"available_parents": 16, "mean": 0.7056244594884324}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.7439364244365543}, "pixel_auc": {"available_parents": 16, "mean": 0.6973202554169273}}, "parents": 16}, "authentic": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.4412181806683595}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.45697021484375}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.5513152973102817}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}, "classical_edit": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.48944667454250673}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.5535888671875}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.925899659695963}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}}, "original": {"accepted_parent_ranking_baselines": {"constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.47224683746742646, "radial_center_auc": 0.7975643577862963}, "accepted_parent_spatial_contrast": {"AI_minus_authentic_auc": 0.1468308237980383, "AI_minus_classical_edit_auc": -0.20524348577118698, "interpretation": "Within-image aligned-region ranking differences, not REAL-versus-AI classification accuracy.", "parents": 16}, "ai_composite": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.5346556152692453}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.57806396484375}, "interior_background_auc": {"available_parents": 16, "mean": 0.7321641102415646}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.8157760800408127}, "pixel_auc": {"available_parents": 16, "mean": 0.7259450807081305}}, "parents": 16}, "authentic": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.5161144963519939}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.53826904296875}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.6631707335390167}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}, "classical_edit": {"metrics": {"background_flagged_fraction": {"available_parents": 16, "mean": 0.6090485126157333}, "flagged_area_fraction": {"available_parents": 16, "mean": 0.66693115234375}, "interior_background_auc": {"available_parents": 0, "mean": null}, "interior_flagged_fraction": {"available_parents": 16, "mean": 0.991369405449905}, "pixel_auc": {"available_parents": 0, "mean": null}}, "parents": 16}}}, "training_admission": false}

No trained localizer, calibrated probability, independent proof or training admission is claimed.

### CO3D candidate metadata audit — 2026-09-15

Retrieved three text files (65,546 bytes total) from the official facebookresearch/co3d repository at revision `eb51d7583c56ff23dc918d9deafee50f4d8178c3`: LICENSE (19,332 bytes), README.md (11,584 bytes), and co3d/links.json (34,630 bytes). Exact URLs and SHA256 values are retained in evidence/co3d_metadata_receipt.json; local storage: /Volumes/LaCie/pixelproof-datasets/research/co3d. No image/archive payload, account interaction or dependency installation occurred.

Reason for review: a possible additional real-video source for content and acquisition variation, and potentially future eligible Model2 source parents. The official README describes improved video decoding and common within-sequence cropping. Therefore this lead does not establish native camera JPEG traces, recent phone coverage, or independent frames. If eventually acquired, all frames from a sequence must share one ancestry group; photographer/device and cross-corpus lineage still need review. No TRAIN/CAL/DEV/final role is assigned.

The README explicitly applies CC BY-NC4.0 to the codebase. That statement alone has not resolved the image dataset's applicable terms and provenance. Image acquisition/admission is deferred pending that evidence. The documented full archive is5.5TB and is unsuitable for this machine; the documented single-sequence subset is8.9GB, but neither was requested or downloaded. This is a metadata-only lead, not new evaluation evidence.

### E131 internal source-holdout result

{"contract_sha256": "6db23ea917a7428d015e0453efe99b16bbcbe3fa2822e6b540e8c31b645ea421", "corpus_level_overlap_limit": "RR paper sections3.2.1-3.2.2 explicitly name FLUX, DALL-E and SD families, COCO/CC3M prompts and Chameleon. Local RR rows do not map each file to those origins. CommunityForensics also mixes generators. The10 components preserve declared row/prompt links but do not isolate all corpus-level possible families; this is publisher-group transfer only, not unseen-generator evidence.", "downloads": 0, "fold_counts": [{"0": 2467, "1": 2916}, {"0": 2875, "1": 1110}, {"0": 2588, "1": 569}], "generator_family_holdout_supported": false, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence.", "locked_scores_sha256": "e3dbcbc1fa28d26e58e48805c1d97eb6575b60462b7d3894b8ec2a9d86b45121", "new_pixels_read": 0, "paired_changes_fullframe_vs_center": {"assigned_transport": {"0": {"new_errors": 84, "rescued_errors": 78}, "1": {"new_errors": 67, "rescued_errors": 152}}, "clean": {"0": {"new_errors": 79, "rescued_errors": 58}, "1": {"new_errors": 55, "rescued_errors": 116}}, "q75": {"0": {"new_errors": 77, "rescued_errors": 46}, "1": {"new_errors": 71, "rescued_errors": 162}}, "social_q75": {"0": {"new_errors": 81, "rescued_errors": 66}, "1": {"new_errors": 68, "rescued_errors": 152}}}, "parents": 12525, "promotion_allowed": false, "seconds": 42.25651224993635, "source_components": 10, "state": "E131_paired_internal_source_holdout_complete"}

Full component/condition metrics: evidence/e131_source_holdout.json. This is consumed TRAIN analysis using separately fitted fold heads, not an independent final or serving candidate.

## E132 — paired patch learning with source exclusion (planned 2026-09-15)

E130 completed in25.69seconds with zero duplicate-token error. Original/JPEG75 mean composite pixel AUC is0.72595/0.69732, below the same-mask radial-center baseline0.79756. The transferred diagnostic cut falsely flags53.83%/45.70% of authentic area and66.69%/55.36% of classical-edit area. AI-versus-classical aligned-region AUC contrasts are negative (-0.20524/-0.18438). Thus this drift map cannot be presented as an AI-specific detector. No threshold sweep or deployment follows.

Next implement a separately registered E132 internal learning diagnostic using the existing E130 original patch-token caches (384 dimensions,32x32). Admit only the fixed16 E129 hard composites and their16 authentic/16 same-mask Gaussian-blur controls to this scoped TRAIN research experiment; global dataset roles remain unchanged. Their16/16 engineering gate passes. A contact-sheet inspection of all16 pairs found scene-preserving local reconstructions and some visible object/detail changes, not a semantic-mask annotation or a guarantee of realistic edits. Keep all16; do not cherry-pick attractive outputs.

Use leave-one-declared-sensor-group-out folds, joining any known scene/source-body ancestry across sensors first. Fit normalization and a zero-initialized regularized logistic patch head solely on the remaining parents. No position, true-mask, difference-to-original or drift map enters inference. Train on fully interior AI patches as positives and equal-mass authentic, classical-edit and composite-background negatives; discard composite boundary patches using the fixed8px band. Both original and JPEG75 views of a parent stay together. Fix BCE+0.005||w||², zero-start L-BFGS and0.5 diagnostic cut before fitting, without held-out tuning. Reuse the tested finite/convergence guards.

Save all fold heads and lock every held-out map before metrics. Report authentic/classical false-flag area, full and interior/background localization, per-parent paired contrast and the same E130/radial/constant baselines, including every parent and condition. This compares internal held-out sensors within one small, already inspected MIDD research population and one SD1.5 editor; it cannot demonstrate unseen-editor, new-dataset or universal performance. Scores are uncalibrated. No final full-data head, web heatmap or deployment is authorized by this diagnostic's outcome.

### E132 internal patch-learning result

{"contract_sha256": "c5f07c1e38b8b932c3e63b1ff16ce3ffbb9d529d5013a348d4a8af85dd7a4363", "downloads": 0, "folds": 8, "limits": "Sixteen inspected MIDD parents, one SD1.5 editor, hard composites, simple masks and two views. Global token context may carry synthetic cues into background. Intended generation masks are not semantic-change annotations. Unknown scene ancestry and encoder pretraining remain. Fold heads are not one deployable model; raw scores are not calibrated probabilities.", "locked_scores_sha256": "65329652dca7ee42648170b86788f02769bd6d0ef793d8d38398b113684182aa", "measurements_sha256": "18bdb2a002574c21c5da148d5d38d4a0d9a54403ed063b75a9ad11f1e3955f46", "parents": 16, "promotion_allowed": false, "seconds": 25.856737582944334, "state": "E132_internal_patch_learning_complete", "summary": {"jpeg75": {"AI_minus_authentic_aligned_auc": 0.07112007203130111, "AI_minus_classical_aligned_auc": 0.13904100934913213, "ai_composite": {"mean_flagged_area": 0.1927490234375, "mean_interior_background_auc": 0.7491296090659942, "mean_iou": 0.20270705884133267, "mean_pixel_auc": 0.7342234512939676}, "authentic": {"mean_flagged_area": 0.174560546875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "classical_edit": {"mean_flagged_area": 0.17108154296875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "reference_rankings": {"E130_drift_mean_pixel_auc": 0.6973202554169273, "constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.5291163098010457, "radial_center_auc": 0.7975643577862963}}, "original": {"AI_minus_authentic_aligned_auc": 0.07381122188737699, "AI_minus_classical_aligned_auc": 0.17412843479263335, "ai_composite": {"mean_flagged_area": 0.1910400390625, "mean_interior_background_auc": 0.7570250862785984, "mean_iou": 0.22249568870308084, "mean_pixel_auc": 0.7422432027316962}, "authentic": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "classical_edit": {"mean_flagged_area": 0.16009521484375, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null}, "reference_rankings": {"E130_drift_mean_pixel_auc": 0.7259450807081305, "constant_auc": 0.5, "interpretation": "Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.", "parents": 16, "pixel_response_auc": 0.47224683746742646, "radial_center_auc": 0.7975643577862963}}}}

No serving change, calibrated probability or independent generalization claim.

### E132 interpretation and next steps — 2026-09-15

All eight sensor/known-ancestry fold heads completed in25.86seconds; every source pair was scored only by a head that excluded its component. Original/JPEG75 mean composite pixel AUC is0.74224/0.73422 (E130:0.72595/0.69732), interior/background AUC0.75703/0.74913, and IoU0.22250/0.20271. AI-minus-classical aligned-region AUC contrast is now positive0.17413/0.13904 (E130 negative). This supports a limited learned distinction between these synthetic composites and Gaussian blur in this pilot.

Authentic flagged area at the E1320.5 diagnostic cut is17.11%/17.46%, and classical-edit flagged area16.01%/17.11%. Both remain far too large to present as a reliable AI heatmap. These cut-specific false-area rates cannot be compared as matched-operating-point improvement over E130: its external drift cut and positive coverage differ. The threshold-free composite ranking remains below the radial-center baseline0.79756 in both conditions. The pilot has not defeated a location prior or demonstrated robust localization. No full-data final head, gallery score, web change or deployment follows; exact E92 is restored.

Next Model2 mechanism test: pre-register masks with matched area but varied location/shape independent of image content, including off-center edits; keep authentic and same-mask classical controls. Hold complete source/known-ancestry groups out and compare frozen E132 fold heads before any retraining. Record generator/checker failures without replacements, and separate interior from boundary success. Do not infer causal center bias merely from the present baseline gap. A second editor and independent source population are required later for editor/source transfer evidence. The present16-set SD1.5/MIDD pilot cannot provide that proof by further threshold tuning.

Next Model1 investigation: use the locked E131 training-source predictions to characterize publisher/content/processing differences and per-source score shifts before choosing another representation or grouped calibration experiment. The excluded RR error rate is a diagnostic, not permission to tune on consumed DEV or protected reserves. Any calibration experiment needs a nested source split, retaining outer held-out sources untouched by normalization, fitting and cutoff selection. Preserve individual AI non-regression as well as REAL errors; do not accept a higher aggregate AUC as a replacement for those safeguards.

## E133 — frozen-head mask-location challenge (planned 2026-09-15)

Test E132's location sensitivity without retraining or choosing another cutoff. Reuse the exact16 E128 authentic inputs and E132 source-excluding fold heads. Translate each original binary mask's tight bounding box, unchanged in shape and pixel area, to a fixed corner with16px image margin. Corner assignment is index modulo4 (top-left, top-right, bottom-left, bottom-right), four parents per corner, fixed independently of pixels and scores. This isolates the mask-position intervention more narrowly than simultaneously varying shape; varied shapes/editors remain later work. Each parent receives one new placement, not a fully crossed parent-by-corner experiment.

Keep source images, seeds119000+i, prompt, SD1.5 weights,30 steps,7.5 guidance, fp16+SDPA and enabled safety checker exactly as E129. Generate one attempt per parent, retaining all failures and no rerolls/refills. Rebuild matching GaussianBlur6 classical negatives using the new mask and retain authentic negatives. Locally synthesized descendants are TRAIN research only; no download, new independent source, global training admission or deployment.

Freeze preparation, generation, head binding and scoring before any new generation or score. Score all16 authentic/classical controls and only accepted composites; explicitly count rejected attempts. Encode with the exact E130 DINO adapter and require authentic-token replay<=1e-5 and frozen-head authentic-score replay<=1e-6 against the previous cache. The head for each image must still exclude its original source/known-ancestry component. Preserve the0.5 diagnostic cut. Lock every map before new-mask metrics.

Report all-negative false-flag area; composite full and interior/background AUC/IoU; aligned AI-authentic/classical contrast; radial-center and constant ranking baselines on the same accepted new masks; paired old/new composite metrics on exactly the accepted parents; and every failure. A position intervention also changes edited semantic content and diffusion output, so any score difference is not pure causal proof of positional bias. Sixteen previously inspected parents, one old editor and unknown upstream ancestry cannot establish independent generalization.

### E133 preparation complete

{"contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "parents": 16, "prepared_sha256": "37f6855fb56972980d35bf1753aaf3c0e86755a9bb9608459b3119ca53cde3c0", "training_admission": false}

### E133 generation complete

{"contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "full16_engineering_gate_passed": true, "generation_sha256": "c9f75a64948f0364a4e0990cb33d3344997b4c1d6fcba38754ae2dd0c0efd369", "parents": 16, "passed": 16, "peak_mps_bytes": 6788562944, "promotion_allowed": false, "seconds": 555.461868665996, "state": "E133_location_generation_complete", "training_admission": false}

### E133 frozen-head location result

{"accepted_composites": 16, "contract_sha256": "74c627b17a69a65095b372b368aaf1ed17199532ff9e443d0949bbdee38bc037", "downloads": 0, "limits": "Consumed sixteen-parent MIDD research with one old editor and known source-excluding heads. New placement changes edited semantic content and diffusion output; not pure causal position attribution or a fully crossed corner experiment. Intended masks are not semantic-change annotations. No fresh-source/editor proof, calibration or serving promotion.", "locked_scores_sha256": "db0cb8362cfd34ee32ab6e81efe31eaf27ae81c257cf48ce4b5c1e736192b79e", "max_authentic_score_error": 0.0, "max_authentic_token_error": 0.0, "measurements_sha256": "6edfdcd3c03b92b7c71dc5d3c708c22da2dc12b4fa7f75819670d534167bca91", "parents": 16, "peak_mps_bytes": 1219198976, "promotion_allowed": false, "rejected_attempts": 0, "seconds": 18.19654425000772, "state": "E133_frozen_head_location_challenge_complete", "summary": {"jpeg75": {"accepted_parents": 16, "ai_composite": {"mean_flagged_area": 0.183349609375, "mean_interior_background_auc": 0.5696380503109616, "mean_iou": 0.09461821008149006, "mean_pixel_auc": 0.5618493555000308, "parents": 16}, "all_parents": 16, "authentic": {"mean_flagged_area": 0.174560546875, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "classical_edit": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "matched_accepted_comparison": {"AI_minus_authentic_aligned_auc": 0.08085072641406385, "AI_minus_classical_aligned_auc": 0.16855089642605717, "constant_auc": 0.5, "new_radial_center_auc": 0.4421728665547357, "old_E132_mean_pixel_auc": 0.7342234512939676, "parents": 16}}, "original": {"accepted_parents": 16, "ai_composite": {"mean_flagged_area": 0.182373046875, "mean_interior_background_auc": 0.5746209190475555, "mean_iou": 0.10165574918839779, "mean_pixel_auc": 0.5658123794602998, "parents": 16}, "all_parents": 16, "authentic": {"mean_flagged_area": 0.171142578125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "classical_edit": {"mean_flagged_area": 0.16778564453125, "mean_interior_background_auc": null, "mean_iou": 0.0, "mean_pixel_auc": null, "parents": 16}, "matched_accepted_comparison": {"AI_minus_authentic_aligned_auc": 0.09192518625126159, "AI_minus_classical_aligned_auc": 0.1851799676281684, "constant_auc": 0.5, "new_radial_center_auc": 0.4421728665547357, "old_E132_mean_pixel_auc": 0.7422432027316962, "parents": 16}}}, "training_admission": false}

### E133 completed location challenge — 2026-09-15

All16 generation attempts passed the unchanged engineering checks,555.46seconds and6,788,562,944 peak MPS bytes. No rejected/repeated/replaced attempt. Prepared32 PNG payloads occupy7,413,473 bytes (16 masks,16 classical controls);32 raw/composite generation PNG payloads occupy15,646,384 bytes. Authentic inputs are reused. macOS AppleDouble sidecars are excluded from image counts. Downloads remain0; these are local research derivatives, not new independent parents.

Frozen E132 heads on the new locations produce original/JPEG75 composite mean pixel AUC0.56581/0.56185, down from0.74224/0.73422 on the exact matched16 original placements. IoU is0.10166/0.09462, down from0.22250/0.20271. The new radial-center baseline is0.44217; fixed heads beat that weak corner ranking but only modestly exceed the constant0.5 baseline. AI-minus-classical aligned AUC contrast remains positive0.18518/0.16855. Thus some edit-related discrimination persists while absolute localization degrades substantially.

Authentic token and score replay errors are both exactly0. Authentic flagged area is unchanged17.11%/17.46%. The measurement does not identify pure causal positional bias because moving the mask also changes edited semantic content and the generated output. It does expose failure under the registered location/content intervention. All maps were locked before mask metrics; scorer took18.20seconds with1,219,198,976 peak MPS bytes. E92 restored exactly. No serving promotion.

## E135 — learn from both registered mask placements (planned 2026-09-15)

Apply a bounded development response to E133: train the same E132384-D frozen-token logistic head with both the original and corner placements, equal weight per parent/placement/condition. Preserve exact E132 source/known-ancestry folds; all variants/placements of a source component stay together. Use both authentic and same-mask classical negatives, the fixed8px pure-patch exclusion, weighted FIT-only normalization, zero-start BCE+0.005||w||², solver guards and0.5 cut. No hyperparameter or threshold search.

Cache only the missing E133 classical/composite patch tokens with the exact E130 adapter; replay unchanged authentic tokens<=1e-5. Reuse the original E130 tokens and authentic entries; duplicate bookkeeping across the two placements does not create independent authentic observations. Only the E129/E13316/16 accepted hard composites enter this scoped TRAIN research. No role changes outside E135, new editor or image download.

Fit eight new fold heads, each excluding all data from its original held-out sensor component. Lock all192 view maps (16 parents x2 placements x3 variants x2 processing conditions) before held-out metrics. Report original/corner separately, authentic/classical false-flag area, full/interior-background AUC and IoU; retain paired E132/E133 frozen-head comparisons and new/rescued parent-level outcomes where applicable. This is an adaptive internal development comparison after seeing E133, not fresh independent validation, and no model is promoted from its outcome.

### E135 cache complete

{"contract_sha256": "0e240bbe58c5672b5a1568be315d509c4c49196cb68f0d7d594aa5b65e24c98c", "downloads": 0, "max_authentic_error": 0.0, "peak_mps_bytes": 1219198976, "seconds": 27.1236343330238, "state": "E135_two_placement_tokens_complete", "token_sha256": "a684dcc96b6540fdfc842e106a3418515bc92e1778bcbcdf5d92feb9d737416b", "unique_parents": 16, "views": 192}

### E135 learning result

{"contract_sha256": "0e240bbe58c5672b5a1568be315d509c4c49196cb68f0d7d594aa5b65e24c98c", "downloads": 0, "folds": 8, "limits": "Adaptive internal development after observing E133; sixteen previously inspected MIDD parents, one SD1.5 editor, two masks per parent. Source exclusion prevents declared parent/scene leakage, not unknown ancestry or adaptive validation bias. More views are not more independent parents. Raw scores uncalibrated; no deployment or universal/editor-transfer proof.", "locked_scores_sha256": "6872c17f604ed874c1a0d4d0c97a17db30d81926784f012ed1bb786e49d27962", "measurements_sha256": "2434046ea207c1ca450d80e6f6e1f647b01ee7ce08e200dce2803de85c41de11", "placement_records": 32, "promotion_allowed": false, "seconds": 48.97825333289802, "state": "E135_two_placement_internal_learning_complete", "summary": {"corner": {"jpeg75": {"ai_composite": {"mean_auc_change": 0.07067495358935735, "mean_flagged_area": 0.28759765625, "mean_flagged_area_change": 0.104248046875, "mean_interior_background_auc": 0.6476919664012324, "mean_iou": 0.17331742888558427, "mean_pixel_auc": 0.6325243090893882, "parents_higher_auc": 12, "parents_lower_auc": 4}, "authentic": {"mean_flagged_area": 0.26605224609375, "mean_flagged_area_change": 0.09149169921875, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.2734375, "mean_flagged_area_change": 0.102294921875, "mean_iou": 0.0}, "unique_parents": 16}, "original": {"ai_composite": {"mean_auc_change": 0.07221693225085765, "mean_flagged_area": 0.28076171875, "mean_flagged_area_change": 0.098388671875, "mean_interior_background_auc": 0.6538335879592014, "mean_iou": 0.18619654171877348, "mean_pixel_auc": 0.6380293117111575, "parents_higher_auc": 13, "parents_lower_auc": 3}, "authentic": {"mean_flagged_area": 0.2574462890625, "mean_flagged_area_change": 0.0863037109375, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.25482177734375, "mean_flagged_area_change": 0.0870361328125, "mean_iou": 0.0}, "unique_parents": 16}}, "original": {"jpeg75": {"ai_composite": {"mean_auc_change": -0.0440433199883205, "mean_flagged_area": 0.29193115234375, "mean_flagged_area_change": 0.09918212890625, "mean_interior_background_auc": 0.7017308809755072, "mean_iou": 0.18385901662410248, "mean_pixel_auc": 0.690180131305647, "parents_higher_auc": 3, "parents_lower_auc": 13}, "authentic": {"mean_flagged_area": 0.26605224609375, "mean_flagged_area_change": 0.09149169921875, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.2701416015625, "mean_flagged_area_change": 0.09906005859375, "mean_iou": 0.0}, "unique_parents": 16}, "original": {"ai_composite": {"mean_auc_change": -0.04322624339088158, "mean_flagged_area": 0.28692626953125, "mean_flagged_area_change": 0.09588623046875, "mean_interior_background_auc": 0.7096534175195824, "mean_iou": 0.19590418623135836, "mean_pixel_auc": 0.6990169593408145, "parents_higher_auc": 3, "parents_lower_auc": 13}, "authentic": {"mean_flagged_area": 0.2574462890625, "mean_flagged_area_change": 0.0863037109375, "mean_iou": 0.0}, "classical_edit": {"mean_flagged_area": 0.24737548828125, "mean_flagged_area_change": 0.0872802734375, "mean_iou": 0.0}, "unique_parents": 16}}}, "unique_parents": 16}

### E135 result and disposition — 2026-09-15

Completed192 token views in27.12seconds with exact authentic replay (maximum error0) and1,219,198,976 peak MPS bytes. All eight fold heads completed in48.98seconds. Post-run receipt review confirms each fold excluded its expected parent/source component, all16 parent IDs are accounted for exactly once per fold's FIT/held union, saved-head maximum score error0, and identical authentic maps across placement bookkeeping. All192 maps were locked before evaluation. Downloads0; E92 is restored exactly.

The corner original/JPEG75 mean composite AUC improves0.56581->0.63803 /0.56185->0.63252. Per-parent corner AUC improves for13/16 original and12/16 JPEG75 cases. Corner IoU improves0.10166->0.18620 /0.09462->0.17332. However, original-placement AUC drops0.74224->0.69902 /0.73422->0.69018, with13/16 parents worsening in each condition. Original-placement IoU also falls0.22250->0.19590 /0.20271->0.18386.

Authentic falsely flagged area rises17.11%->25.74% original and17.46%->26.61% JPEG75. Classical-negative flagged area also rises:24.74%/27.01% for original placement and25.48%/27.34% for corners. Thus the candidate trades corner improvement for original-placement loss and substantially more false alarms. It is not a successful non-regressing replacement and is rejected for serving. Raw-cut comparisons are descriptive; no matched-coverage calibration claim is made.

Next investigate two mechanisms separately: richer location-independent low-level/residual features with exactly matched classical and authentic controls; and a nested source-separated calibration diagnostic to determine whether score offsets can be corrected without destroying localization. The present experiment does not prove that either will work or that the encoder alone caused the failure. Keep the current folds, previous failures and per-placement reports; any new experiment must freeze its features, fitting population and evaluation before scoring. More independent parents and a second editor remain necessary for transfer evidence; repeated tuning on this small pilot cannot supply it. Do not promote E135 or alter the demo's score/decision policy.

Implementation commit123d626 passed GitHub CI34976937916 (Python and web). The full local1116-test suite and compilation checks passed before execution. These software checks establish implementation health, not detector reliability.

### E136 Model1 consistency result

{"contract_sha256": "a94d48cbfe540464698b7c555feca35468ce14a6de12aaa2fc262c20ff6cf77d", "downloads": 0, "four_view_score_stability": {"center_control": {"0": {"baseline_mean_four_view_score_range": 0.07215121479782, "candidate_mean_four_view_score_range": 0.058017973338941625}, "1": {"baseline_mean_four_view_score_range": 0.14756244758067666, "candidate_mean_four_view_score_range": 0.11504566218030453}}, "full_frame": {"0": {"baseline_mean_four_view_score_range": 0.07063796249434952, "candidate_mean_four_view_score_range": 0.057147595881528526}, "1": {"baseline_mean_four_view_score_range": 0.14259441615715251, "candidate_mean_four_view_score_range": 0.10958736853573167}}}, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E134, not an E92 serving-model evaluation. Consistency can reduce useful discrimination or create new errors; no automatic candidate promotion.", "locked_scores_sha256": "06cb526a3ee7f2c5dfe7f99da6feb8072117095998e441a77972078b3c1201a7", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control": false, "full_frame": false}, "promotion_allowed": false, "seconds": 31.591304500005208, "state": "E136_Model1_consistency_complete"}

Full source/condition and individual-transition results: evidence/e136_transport_consistency.json. E92 remains unchanged.

E136 acquisition closeout: zero downloads and zero new image reads. Reused only the
existing12525-parent TRAIN feature cache and FIT-only E131 maps on the external data
volume; generated heads/predictions remain local research artifacts. No data roles,
rights or admission decisions changed. Next acquisition work prioritizes file-level
provenance and independent Model1 evaluation coverage; Model2 acquisition is secondary
and no new download has been initiated by this priority update.

## E137 acquisition plan — 2026-09-16

No acquisition: reuse the existing12525-parent TRAIN feature cache and source-excluding
E131 maps on /Volumes/LaCie/pixelproof-datasets. Four processing views per parent are
repeated measurements, not50100 independent images. Purpose: isolate expert dependence
before spending data or compute on another representation. No new pixels, roles, licences,
external uploads or reserve admissions; new head/score artifacts remain local research.

### E137 Model1 expert-ablation result

{"context_omitted_geometry_max_error": 0.0, "contract_sha256": "63c2993345b5d4181988da87212ce7c2fadf26bf9062567f8a26be4173089545", "downloads": 0, "fits": 24, "generator_family_holdout_supported": false, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E134, not an E92 serving-model evaluation. Consistency can reduce useful discrimination or create new errors; no automatic candidate promotion. Adaptive repeated-fold expert-ablation diagnostic. Refit differences measure conditional utility of an expert plus changed capacity, not causal dataset bias, standalone expert quality or independent selection evidence.", "locked_scores_sha256": "3ed493830780064a951ca737d27c50b0bdc11e4591e54c1b6b44677479e911b1", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control_without_clip": false, "center_control_without_context": false, "center_control_without_dear": false, "center_control_without_dino": false, "full_frame_without_clip": false, "full_frame_without_context": false, "full_frame_without_dear": false, "full_frame_without_dino": false}, "promotion_allowed": false, "seconds": 39.361219875048846, "state": "E137_Model1_expert_ablation_complete"}

All eight branch/omission reports: evidence/e137_expert_ablation.json. No serving change.

## E138 acquisition plan — 2026-09-16

Zero new acquisition or image decoding. Reuse the same12525-parent TRAIN feature cache
and source/class components to test loss weighting; no new independent observations are
created. Literature webpages were read for method context, with no source-code/package,
weights or image dataset downloaded. Roles and external-volume storage stay unchanged.

### E138 Model1 source-risk result

{"contract_sha256": "a98371486c6423918dfbaf68dc7cbd9e14df3380a3fcf350addb7f023a70edd6", "downloads": 0, "four_view_score_stability": {"center_control": {"0": {"baseline_mean_four_view_score_range": 0.07215121479782, "candidate_mean_four_view_score_range": 0.07523635573952739}, "1": {"baseline_mean_four_view_score_range": 0.14756244758067666, "candidate_mean_four_view_score_range": 0.15987321354486975}}, "full_frame": {"0": {"baseline_mean_four_view_score_range": 0.07063796249434952, "candidate_mean_four_view_score_range": 0.07306581163107048}, "1": {"baseline_mean_four_view_score_range": 0.14259441615715251, "candidate_mean_four_view_score_range": 0.15570828479970486}}}, "limits": "Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence. Adaptive development after E137, not an E92 serving-model evaluation. Entropic source risk is a fixed convex adaptation, not a reproduction of neural group DRO or a guarantee for unseen groups. No automatic promotion.", "locked_scores_sha256": "1603c1a77424c61a47bb34e9e749e5d0bb7846387c4748d6768300ca09d83336", "new_pixels_read": 0, "parents": 12525, "passes_internal_individual_nonregression": {"center_control": false, "full_frame": false}, "promotion_allowed": false, "seconds": 31.994000499951653, "state": "E138_Model1_source_risk_complete"}

Full source/condition and individual-transition results: evidence/e138_source_risk.json. E92 remains unchanged.

## E137/E138 acquisition closeout — 2026-09-16

Completed24 ablation fits and6 source-risk fits using only existing TRAIN features.
Downloads:0; new image reads:0; new independent parents:0; data-role changes:0. No Model2
acquisition/generation occurred. New heads, full parent-level predictions, contracts and
logs remain under external-volume e137/e138 and pipeline directories. Small source-level
summaries and contracts are retained in evidence/; no raw images, weights, private gallery
or parent-level scores are committed. Next: provenance/coverage audit before a new dataset
admission; no acquisition is currently running and no new dataset is declared qualified.

## E139 offline audit plan — 2026-09-16

No acquisition. Inspect active TRAIN metadata and existing aggregate MNW/HDR+ reserve records only; no image/feature/weight reads, scores, role changes or new independent parents.

### E139 metadata audit result

{"AI_bearing_components": 3, "components_crossing_folds": 0, "declared_components": 10, "independent_test_parents_created": 0, "mixed_corpus_AI_rows_without_explicit_generator_field": 1179, "model_scores_created": 0, "parents": 12525, "pixels_read": 0, "stored_identity_links": {"body": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 0, "repeated_identity_groups": 0}, "declared_scene": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 1000, "repeated_identity_groups": 114}, "pixel": {"cross_component_groups": 0, "cross_fold_groups": 0, "cross_label_groups": 0, "cross_source_groups": 0, "parents_in_repeated_groups": 0, "repeated_identity_groups": 0}}}

Full report: evidence/e139_provenance_audit.json. No new independent test or serving change.

## E139 and offline runtime closeout — 2026-09-16

No dataset/package/model-weight download and no role admission. E139 read only the active
TRAIN contract and aggregate reserve metadata: no image bodies, feature arrays, weights
or protected parent lists. The separate demo restart loaded existing local model assets;
HTTP input checks used invalid bytes and one in-memory32x32 synthetic PNG, not a dataset
or detector evaluation. E139 found no stored identity collisions; only895 parents had
stored pixel hashes, so do not claim12525 fresh pixel comparisons. Preserve1179 missing
mixed-corpus generator/model fields as gaps, and the500 CommunityForensics model-name
records as partial provenance rather than240 proven independent generator families.

## E140 offline acquisition status — 2026-09-16

Zero acquisition or role change. Verify metadata hashes and admitted TRAIN membership for the895 post-E54 additions; protected reference JSONs are hashed without parsing their records. No image, weight, feature or score reads.

### E140 admission-chain audit result

{"active_parents": 12525, "balanced_final_admitted": false, "base_parents": 11630, "contract_sha256": "c0c121dfb96ac82ed65588e6cd30fab45aeec09bf60ee1611dd406ef4877dfe0", "downloads": 0, "generator_independence_proven": false, "historical_admission_coverage_complete": true, "image_reads": 0, "later_admitted_parents": 895, "limits": "Historical exact/perceptual screening lineage, not fresh pixel comparison, semantic deduplication, generator independence, balanced final admission or accuracy evaluation.", "model_scores": 0, "new_independent_test_admitted": false, "next": "Membership/reference gap for the895 later additions is closed under stored admission policy. Prompt/scene/base-generator ancestry, protected evaluation protocol and candidate gates remain separate; do not score/tune on reserves automatically.", "stages": {"e100": {"admitted": 256, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 8}, "e72": {"admitted": 511, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 6}, "e88": {"admitted": 128, "historical_cross_matches": 0, "protected_reserves_bound": 2, "reference_documents": 7}}, "state": "E140_admission_chain_complete", "verified_metadata_files": 32}

## E140 acquisition closeout — 2026-09-16

No downloads or role changes. Verified32 metadata files; all895 post-E54 TRAIN parents
are covered by admissions binding both existing protected reserve receipts. Historical
reference documents were hashed only, never parsed as individual protected records.
The reserve coverage-chain question is closed under the recorded fingerprint policy;
semantic/scene/base-generator independence and balanced final admission remain unproven.
MNW/HDR+ already exist locally; do not redownload them or repurpose them for tuning.


## E141 plan — recover existing TRAIN lineage (2026-09-16)

Recover E32 AI metadata already stored on the external disk into an additive overlay.
Bind current parent ID/source/class to original c3 TRAIN records and native body SHA;
processed inputs also require the original R0 input receipt SHA. Bind each original
source key/body to its realization audit, already referenced by eligibility metadata.
Recover declared model names, source-family declarations, dataset revisions and stored
prompt hashes. Exclude the empty-string hash from prompt links. Reject conflicting or
ambiguous identities; do not guess generator ancestry from model names. Report source
coverage and prompt links across current components/folds, without changing those folds.

Only active E32 AI TRAIN rows join to the private output. No photos, features, weights,
model scores, protected reserve records or new data are opened. Schema/role exploration
preceded registration and is not a blind experiment. Roles, serving E92 and Model2 stay
unchanged. No downloads, training or threshold search. The scientific purpose is to find
missing provenance and possible evaluation dependence before further candidate fitting.


## E141 result — recoverable lineage and remaining coverage gaps (2026-09-16)

Contract SHA: ee668f66549618d6c3a9251c56db33c3195a363fdfbfb5df833ae45695a2653b.
All3005 active E32 AI parents joined without identity conflict through the historical
c3 TRAIN/native-body/processed-receipt/audit chain. The separate overlay is stored only
on the external volume at e141/train_lineage_overlay.json; its SHA is
 d0d636ec405eb0ecee77437cdc8036b84f736e9f544446249b9c89fcd08d5e1e.
No old manifest, fold assignment, artifact or guard was edited.

| Declared source | Active AI parents | Recovered prompt hashes | Declared model identity |
| --- | ---: | ---: | --- |
| CommunityForensics |569|569|569 populated rows,240 distinct names;69 previously omitted names recovered |
| FLUX.2 Klein 9B Base |569|569|Source declaration only |
| GPT Image1 |569|569|Source declaration only |
| Qwen Image2512 |569|569|Source declaration only |
| Gemini2.5 Flash Image Preview / Nano Banana |569|0|Source declaration only |
| Nano Banana Pro |160|0|Source declaration only |
| E36 consumed TRAIN AI |480|Not recovered in this stage|Six declared source labels; scope does not inspect their upstream records |
| RR TRAIN AI |1110|Not recovered in this stage|Per-file generator ancestry remains unresolved |

Recovered2276 nonempty stored prompt hashes among4595 active AI parents.2319 remain
without a recovered hash in this overlay, not necessarily absent upstream.418 repeated
hash groups involve1050 parents;380 groups cross source labels. None crosses the current
frozen E131 component or fold. Existing conservative grouping covers these observed
links. Different hashes do not establish different semantics; preprocessing/normalization
may differ, and the check does not audit all hidden common training ancestry.

The4026 rows without a recovered explicit model_name field are NOT4026 wholly unknown
generators: many have explicit source-level declarations shown above. Conversely, the240
CommunityForensics names are NOT240 verified independent base-generator families. Dataset
revision fields refer to release metadata, not model weight revisions. CF's historical
revision is a local E31-pinned description, not a recovered upstream commit in this stage.

Interpretation: E139's claim that no prompt IDs were stored applies to the old active
manifest. E141 recovers metadata from upstream records without rewriting that evidence.
This closes69 missing CommunityForensics name fields and quantifies known prompt links;
it does not improve AI recall, REAL false positives, abstention coverage or calibration.
The mixed-corpus explicit-model gap is now1110 RR AI parents in the new overlay, down
from1179 when the69 older CF rows lacked names. No independent test parents created.

Next: inspect eligible E36 consumed-TRAIN metadata for the remaining480 known-source AI
parents and document whether local RR metadata can map the1110 mixed-generator parents.
Keep unavailable prompt/model ancestry explicit. A later candidate/evaluation design must
use this coverage evidence and independent groups; do not run another adaptive sweep on
the consumed source folds or score the protected reserve to fill a dashboard.

Downloads, fits, model scores and image reads:0. All existing source roles and licences
are retained; no acquisition/admission or additional use rights are implied. E92 stays
serving with unchanged thresholds, and Model2 remains experimental. The prior UI commit
85590ea61410d8695d64ed48d6278b92866175ad passed GitHub CI run35072884465.

Validation: all1160 Python tests passed in21.05seconds (one existing upstream
Starlette/httpx deprecation warning); compileall and diff checks passed. No frontend
or inference code changed in E141. These checks establish implementation integrity,
not improved detection performance.


## E142 plan — E36 TRAIN prompt identity recovery (2026-09-16)

After making the demo current, resume Model1 provenance work without downloads.
Existing E36 consumed CAL metadata retains prompt_id fields omitted by the current
TRAIN manifest. Register a source/parent/body join restricted to currently active E36
AI TRAIN rows. Recover local declared prompt IDs and count cross-component/fold links.
Reject role/body/ID mismatches and duplicate keys. A shared numeric prompt ID is only
meaningful within this publisher namespace, not proof of identical text across corpora.
Do not open E36 FINAL, protected reserves, pixels, model scores or weights. Preserve all
frozen manifests and create only a separate private overlay and aggregate public report.


## E142 result — remaining E36 prompt metadata recovered (2026-09-16)

Main Model1 work resumed after the demo update. Contract SHA
 d183adbda1fa99e989300859b06d12179915a7f6e2330a3240f5eddfea627511 binds current E131
TRAIN and historical E36 CAL metadata.480 existing AI TRAIN parents (80 per each of
six declared sources) match source/parent/body/old-role identities. Recovered100
publisher-scoped prompt groups:28 groups have6 parents,37 have5,24 have4,9 have3,
and2 have2. No group crosses a registered component or fold. The private overlay SHA
is a349aca026ec85e14c9a658be51689424de6b4713da56841a9195f43c7040895; no inherited
manifest or role was rewritten. The original consumed-CAL role is historical; eligibility
comes only from the current TRAIN contract, not a new admission made by this audit.

Together with E141,2276 active AI parents now have recovered stored prompt hashes and
another480 have publisher-scoped prompt IDs. These are different kinds of evidence and
cannot be compared across namespaces without prompt text or authoritative mappings.
1839 active AI parents remain without either recovered link in these overlays:729 Nano
source parents and1110 RR parents. Base-generator ancestry remains unverified. Next is
local RR per-file metadata availability, without inferring model families from filenames
or consuming protected test reserves. No images, fits, scores or downloads in E142;
Model2 remains experimental and E92 stays serving under the new display-only policy.


## E143 plan — local RR TRAIN metadata availability (2026-09-16)

Resume Model1 work after the site update. Existing RR train/validation inventory records
3000 images and no non-image sidecars; this is a historical inventory, not a new archive
scan. Verify its acquisition/extraction metadata chain and bind only currently active
RR TRAIN parents to their original train member/source/class/body identities. Read those
bounded local bodies to recheck SHA/size, then inspect lazy image headers and EXIF already
present in the header. No pixel decoding, model/feature extraction, gallery, official
validation/test image or protected reserve access. Use a separate private record file;
only counts/hashes reach Git. No datasets, weights or packages downloaded.

Selected PNG text keys and EXIF software/camera/description fields are untrusted
self-declarations. A camera Model tag is not an AI generator. Empty/missing metadata
cannot certify authenticity, and header-only inspection may miss later container fields
such as PNG post-IDAT text. Count failures without removing parents. Freeze code and
input metadata before this image-header pass; do not modify TRAIN roles/folds or infer
generator families from topic/filename strings. This audit must lead to an explicit
local-provenance availability conclusion, not repeated unbounded searches or a new
claim of unseen-generator evaluation.


## E143 completed — RR local provenance availability (2026-09-16)

Verified all2360 active RR TRAIN bodies (1250 REAL,1110 AI), totaling1,721,292,228
bytes, against their original extraction/acquisition identities. Header/EXIF parse
errors:0. Selected fields occur in525 REAL and135 AI parents. The original containers
are REAL:1249 JPEG/1 PNG; AI:113 JPEG/997 PNG. These strongly unequal distributions
motivate a bounded processing-confound diagnostic; they do not prove that a model uses
container identity or that generator/content differences are unimportant.

A separately recorded post-hoc parser finds110 exact DALL-E producer declarations and
one model-name declaration in generation settings. These111 strings remain unverified
file self-declarations;999 RR AI parents lack either parsed convention. Camera and
Matplotlib software tags are not generator identities. No verified checkpoint/base-family
ancestry was recovered, no folds/labels changed, and1839 active AI parents still lack a
recovered prompt hash or publisher-scoped prompt ID. Header-only inspection can miss
later container fields; this closes the bounded local-header availability question,
not all provenance uncertainty. Do not repeat unconstrained metadata searches.

Public aggregate receipts: evidence/e143_rr_metadata.json and
 evidence/rr_header_declarations_20260916.json. The frozen E143 contract SHA256 is
9bb8135d43ef4c5c0848f689406fc55e27a982717aa9fa649b2335d85eddb4f6.
Raw metadata text and per-parent overlays remain on the external disk, outside Git.
No downloads, pixel decoding, model scoring, fits, protected-image access or promotion.
E92 remains the live primary demo; Model2 remains experimental. The preceding site/code
commit b1ac736 passed GitHub CI35076273276.


### E144 acquisition/role accounting — 2026-09-16

Joined the already recorded2360 RR TRAIN header formats to locked E131 scores; no new
image access, download, data admission or role/fold changes. Per-parent metadata stays
private. Only aggregate32-stratum measurements are published. Container labels do not
become authenticity or generator-family labels; protected reserves remain unopened.


## E145/E146 data accounting — 2026-09-16

E145 reads only current TRAIN metadata; E146 reuses exactly12525 TRAIN parents and
four existing cached processing views. No acquisition, raw image read, role admission,
new independent sample or protected-reserve access. FIT/CAL/EVAL are internal roles
within existing research TRAIN, not a relabeling of the project-wide final reserve.
Preserve all known source/prompt components and unresolved upstream ancestry.


### E146 completed reuse — 2026-09-16

Three heads reused existing12525-parent/four-view TRAIN caches; no image bytes or new
samples were acquired. Six internal role permutations preserve every declared component.
All artifacts and per-parent predictions remain on the external disk under e146; public
receipts contain aggregate metrics/hashes only. No admission or protected-reserve change.


## Deep project audit — data accounting (2026-09-16)

Read bound TRAIN and consumed E66 DEV metadata and locked prediction records, not dataset
image pixels. Checked stored body/pixel identities for12269 historical/12525 current
TRAIN versus320 DEV parents. Zero observed intersections; pixel coverage is639/895 TRAIN
and320 DEV, so differently encoded copies remain incompletely covered. No independent
reserve was opened, no roles changed and no acquisition occurred. Four synthetic images
were generated in memory only for preprocessing parity, not admitted as training/test
examples or scored for detector quality. Raw per-parent records remain outside Git.
