# Runtime artifacts

The Git repository intentionally does not contain model weights. Runtime identity is instead
frozen in `artifacts.manifest.json`: origin, licence, revision, expected path, SHA-256 and the
model/feature schema are recorded together.

## Clean environment

The serving lock was verified with Python 3.13 on macOS arm64. Re-resolve deliberately on a
different platform because PyTorch wheels are platform-specific.

```bash
cd ml
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-serving.lock
.venv/bin/pip install --no-deps -e .
```

Copy the five project-trained files from the owner's controlled artifact storage to their
manifest paths:

- `artifacts/tile_resnet18_seed2024.pt` — canonical E20 project model
- `artifacts/best.pt`
- `artifacts/best_genimage.pt`
- `artifacts/feature_full.joblib`
- `artifacts/feature_crop128.joblib`

There is deliberately no anonymous download URL: these files have no redistribution licence.
The servable MIT-licensed Community-Forensics snapshot can then be fetched at its pinned commit
and every non-optional artifact checked offline:

```bash
.venv/bin/pixelproof-artifacts prepare
.venv/bin/pixelproof-artifacts check
```

`prepare` downloads only manifest entries of kind `huggingface`; it never invents or replaces
project-trained weights. `check` exits 2 on a missing or mismatched file. The service repeats the
same verification before deserialising anything: a core failure yields `status=unavailable`; a
missing/mismatched verdict snapshot yields `status=degraded` while research methods remain usable.

The canonical E20 checkpoint is also schema-checked after its hash passes: the runtime requires
the recorded ResNet-18 arm, seed, state dict, validation metadata, 128 px tile size, ImageNet
normalization, texture floor, selected aggregation, calibration threshold and split provenance.
It never deserialises an unverified checkpoint.

## Optional B-Free arm

B-Free is not fetched by the preparation command and is not part of the default servable setup.
To use it within its upstream research/non-commercial terms:

1. place the official checkout at `external/B-Free` on the pinned manifest commit;
2. place the official `BFREE_dino2reg4` config and weights at the manifest paths;
3. run `.venv/bin/pixelproof-artifacts check --include-optional --group bfree`;
4. explicitly acknowledge the scope with `PIXELPROOF_BFREE=1`.

Revision and both file hashes are checked before the arm can load. CORS, authentication and rate
limits remain separate deployment controls described in `SERVING.md`.

## Local E31 training archive

`data/e31/train_v2_tiles.npz` is an ignored, reproducible training input rather than a serving
artifact. E31/B2 realized 11,300 unique 128 px RGB tiles into a 395,082,960-byte NPZ with SHA-256
`508330c2d8318bcd4c8a92c86a86a627ff98ee1bdc97a67772540a68c8569f2b`. Its frozen selection,
source/role counts and compact realization receipt live under `../evidence/`; source images and
protected hashes do not enter Git. B3 code must verify this SHA before extracting features.

E31/B3's ignored `data/e31/b3_features.npz` cache is 19,002,342 bytes with SHA-256
`f59e1fb616d9bcf7384bd571f92570b9cd70f8a043e1b659bf1c258bb97c4c49`. It contains aligned E20
scores, 384-dimensional frozen DINOv2 embeddings and 68-dimensional forensic vectors for the
accepted tile archive. Seed-specific convex heads under `artifacts/e31/` remain experimental,
ignored and non-servable until B4/B5 gates choose and freeze a candidate.

B4 selected the ignored `artifacts/e31/b4_candidate.joblib` single-DINO package (12,759 bytes,
SHA-256 `99901219ec47e49a36fca7edd35a1c1737eb1cd9088f6465893054023914d860`). It embeds the fitted
384-dimensional linear head, threshold `0.7090073824`, selection/tile/cache identities and the
encoder contract: timm `vit_small_patch14_dinov2.lvd142m`, 224 px, pretrained blob SHA
`04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081`. It is a B5 candidate,
not yet a serving artifact; no E30 result was used to choose it.

## Local E32 runnable research artifact

With `PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets`, E32 stores its ignored fitted head at
`e32/models/e32_r0_dinov2s.joblib` (12,720 bytes, SHA-256
`7f170340ab050543f60ceda129224a67b5adaf22a628e52049d401bc96e8a85e`). It binds the standardized
input receipt SHA `2255b123...5199`, frozen feature archive SHA `716df956...be3b`, DINOv2-S model
identity, selected C=0.1 and threshold 0.141444. The feature cache is 33,439,283 bytes at
`e32/features/r0_dinov2s_features.npz`.

The artifact is research-runnable but rejected for serving: its owner-gallery DEVELOPMENT REAL
recall is only 24.29%. Use `PIXELPROOF_DATA_ROOT=... HF_HUB_OFFLINE=1
PYTHONPATH=ml/src ml/.venv/bin/python -m pixelproof.e32_candidate /path/to/image`; do not copy it
into the canonical artifact registry or treat its verdict as authenticity proof.

R1a stores `e32/models/e32_r1a_cfvit.joblib` (12,703 bytes, SHA-256
`6288acba5e50f11588b48907351cbd0fd1b741d3dab079376491bae5938ed670`) and its frozen 22,688x384
feature cache at `e32/features/r1a_cfvit_features.npz` (33,436,875 bytes, SHA-256
`c170a1f6688421f73c72c3b9ed6f1de10a57bf9850a535246e64a15bc71bbc6b`). It binds CF revision
`ac6ee457...db00`, weight SHA `275ba982...1692`, C=0.01 and threshold 0.118110. R1a is also
rejected for serving after only 26.67% owner-gallery REAL recall; the reproducibility CLI is
`pixelproof.e32_cfvit_candidate`.

R1b stores two controlled ignored heads after adding 3,994 audited iPhone rows. DINO is
`e32/models/e32_r1b_dinov2s.joblib`, SHA-256 `aca41dd86d3bfb2d6c6cb722c0d4882e5866640149fc5a684535f5eca8da8e86`;
CF is `e32/models/e32_r1b_cfvit.joblib`, SHA-256
`68a54aa2a5a7f85713669302ba5151983e3a9217cb2f49cc894e2a802802701c`. The preregistered internal
rule selected CF at C=0.01 / threshold 0.1259349137544632. Its complete and appended-only feature
archive SHAs are `e3408691...4592` and `579d302a...d5c6`; the artifact binds the R1b input receipt
and pinned CF weights. The external gate then failed at 40.0% worst-device IPN FP and 68.57%
owner-gallery FP. Both heads remain ignored research evidence and must not enter the canonical
registry or an official decision. The selected CF head may be loaded from the external data root
by the local demo only as a hash-verified, non-voting `r1b_research` payload.

### E60 offline correction — research only (2026-09-10)

External `e60/correction.npz` stores a3,072-element additive-logit vector, bound to its frozen
contract and original E43 SHA. Candidate SHA-256:
`582d6c4f020ce309c8e44a88383ef9d69c559773d6332694bd203e1132415bc5`.
It depends on unchanged `e43/e43_small_predev.joblib` with SHA
`a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390`; it is not a standalone
detector, registry replacement or serving promotion. Exact zero-init and saved-vector replay
pass on TRAIN. Frozen recipe/results live in `evidence/e60_correction_contract.json` and
`evidence/e60_fit.json`; any consumed E49 regression remains separate from independent final proof.


### E61 replay guard — engineering evidence only (2026-09-10)

`evidence/e61_replay_contract.json` binds the admitted E54 TRAIN manifest/features,
frozen E43/E60 artifacts and new guard/runner code. `evidence/e61_replay_gate.json`
records the successful software check: unchanged E43 passes, E60 is rejected for5
newly missed AI parents across its3 TRAIN conditions. No model artifact produced;
no E49 read, new training or serving promotion.37 focused tests pass. Never interpret
this evidence as improved detection quality or a guarantee on unseen generators.


### E62–E64 correction artifacts (2026-09-10; research only)

All arrays remain under `/Volumes/LaCie/pixelproof-datasets/e62`, `e63`, `e64`.
E62/E64 NPZ stores TRAIN PCA mean/components/scales and65 additive-logit coefficients;
E63 adds64 RBF centers, bandwidth and kernel normalization. Every artifact binds its
fit contract and original E43 hash. Zero initialization and saved score replay are exact.
E62/E63 each have locked4000-view consumed E49 scores and failed reports. E64 has only
TRAIN evidence and must not be evaluated after failing its preregistered TRAIN ceiling.
No serving artifact/registry changed. Versioned compact evidence is in `evidence/e6[234]*`;
see EXPERIMENTS for identities, counts and limitations. Do not overwrite or promote.


### E65 diagnostic artifacts (2026-09-13)

External `e65/` contains83 publisher-verified originals,16 fixed RAW-development PNGs,
83 Q75 derivatives, frozen acquisition/audit/diagnostic contracts, complete reports and
166x3072 diagnostic features. No new classifier. Feature archive SHA
`25768622659036089fe7ab45e6a95af3aacdcabbabdba5bf3abf594d84ada170`; full score SHA
`4ed6222a5e40afd84e88f2cdeb86059728ad84e763908a0656afefc6b757680b`.
Every view remains DIAGNOSTIC_DEV_ONLY; features cannot silently enter TRAIN/final.
Compact evidence and human-readable findings are versioned in `evidence/e65_*`.


### E67 TRAIN processing-response features (2026-09-13)

`/Volumes/LaCie/pixelproof-datasets/e67/blur_features.npz`:11,630x3x3,072 TRAIN-only frozen
DINOv2S features of GaussianBlur0.8 applied to fixed E54 crops. SHA
f48474422616c0b3776d6749c0b03796493f8b349a97ce355749011e1b715b4b. Feature contract and
91 chunk receipts bind original crop identities. No classifier artifact yet; do not confuse
feature extraction with accepted quality. E43/serving unchanged.


### E66 acquisition and limited DEV bindings (2026-09-13)

Verified 6.62 GB SIDD archive and 160 decoded NOISY images remain under the external `e66/`
directory, alongside 160 source-verified local AI originals. The frozen development manifest
SHA is bf4c3586126edf91588cc72b375504fa089c3d235abdce7f660686a82c2a6301. Zero model scores
at admission; whole SIDD publisher excluded from TRAIN/final. Compact `evidence/e66_*`
receipts bind acquisition, scene/overlap audit and the limited DEV population.


E67 rejected TRAIN candidate: external `e67/correction.npz`, SHA d0491ceb962915f096bdfe6cb0555402b16e78085b2a9f2e141c6d10d98f9a1b.
Bound to its frozen fit contract and E43 reference. `evidence/e67_fit.json` records the
failed TRAIN guard; no DEV or E49 scores may be created for this candidate.


E68 rejected TRAIN candidate: external `e68/correction.npz`, SHA f1dd50fdee88cb00de543308a8fef08d72e1e9e06121b580d7f93170f5278d6b.
The bound original/blur bases are unchanged from E67; only correction weights were fit.
`evidence/e68_fit.json` records failure. No DEV/test scoring or promotion allowed.


### E69 TRAIN patch-shuffle features complete (2026-09-13)

All 11,630 parents x three conditions x 3,072 features completed in 871.99s
across 91 checked chunks. Original parity: zero maximum score error and zero
decision changes on 30 source-selected parents. Feature SHA fb1028dc15e7c4ef7b85c54253edf9e3a9e04a184a60c4da07e8aa9cbc7a1819;
contract SHA abb66f00bacf05bfd290d3318ec4cf273b2ad8ea6d39341212ba976ecb2b4dc0. No DEV/final rows read and no candidate yet.
Proceed to bind and run the already fixed 129-coefficient E69 recipe.


E69 rejected TRAIN candidate SHA 7f41b84ec80ee8c54839565804227c17644aca1bf9dd5e004a5deb518e63670d (`e69/correction.npz`, external).
Its original/texture PCA bases and 129 coefficients are frozen with the fit contract.
No E69 DEV or E49 scores exist; the candidate must not be promoted or re-evaluated.


E70 TRAIN-passing research candidate: external `e70/correction.npz`, SHA 2a82d30c86b975a0293d2ab0af99345240d99905fffbd9c403530d0a18db0fd2.
Original/blur bases, fixed bilinear hash/sign arrays, TRAIN sketch normalization and 257
coefficients are bound to the fit contract. TRAIN pass only; no deployment eligibility.


E70 is rejected after its first limited DEV evaluation. Full external scores SHA 9cf625b6f2123c5a6bd421f37d6415030cc386ccdc7cd8b809a6dba36b879521;
compact report `evidence/e70_development.json`. Preserve all artifacts; no E49 or serving
promotion. E66 has now been scored; its old unscored-admission flag is historical only.
