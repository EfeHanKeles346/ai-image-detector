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
