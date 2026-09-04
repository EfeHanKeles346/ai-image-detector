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

### E49 comprehensive-final registry — frozen intent before payload transfer (2026-09-04)

E49 is the last independent promotion test for Module 1. Its target is **2,000 unique parents**,
exactly 1,000 REAL and 1,000 AI. No third-party image is committed to Git; payloads belong under
`/Volumes/LaCie/pixelproof-datasets/e49/`. The 4 GiB network ceiling excludes the already-local
AIGC archive and includes every newly transferred image byte.

| Source | Pinned identity / licence | Frozen E49 role and quota | Why it is here | Boundary |
|---|---|---:|---|---|
| Wikimedia Commons camera-category originals | Commons API page/revision ids and per-file licence/attribution will be frozen by the binder | FINAL REAL, 1,000: 100 each across ten declared current phone/camera categories | Independent public publisher; original-upload transport, multiple devices and uploaders; directly stresses the false-AI failure on real photography | JPEG only; camera category and EXIF must agree; uploader capped; “original upload” is not claimed to be untouched sensor-native data |
| `datapointai/text-2-image-human-preferences-2m` | revision `e1d8719a2d521eac6c62ee84f329afc2c03ec928`; metadata CC BY 4.0; image-output rights remain provider-specific | FINAL AI, 800: 160 each GPT Image 2 / Nano Banana 2 / Seedream 5 Pro / FLUX 2 / Ideogram 4 | August-2026 independent collection, fixed seed, 500 prompts in ten categories, current provider breadth and full-resolution output table | Contact-sharing gate must be accepted by the user first; evaluation/research only; no post-score source substitution |
| `TheKernel01/AIGC-Detection-Benchmark` | revision `c91d9024a5a77ef06e2ec681b53f9caf08675663`; Apache-2.0; local 60-shard test release | FINAL AI, 200 StyleGAN2 (generator code 14) | Adds a GAN family from a source not used for E43-S training or E48/E50 selection; prevents “current diffusion only” success | Test-only; select bytes without reading scores; source-native format preserved; exact/dHash protected-role audit required |

The local AIGC copy now reproduces that pinned Hub revision and all 60 shards. A metadata-only scan
read exactly the `label` and `generator` columns: 125,026 total rows, of which 1,997 are label=AI and
generator code 14 (StyleGAN2). The deterministic 240-row reserve identity digest is
`15e5c1315d1411c1c106dc457166b2b097dd3753bcb357e35178f279a95cc731`. No image column or model
score was read, and no network byte was required for this source.

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
