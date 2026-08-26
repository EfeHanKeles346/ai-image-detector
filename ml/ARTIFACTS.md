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
