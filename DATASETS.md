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
