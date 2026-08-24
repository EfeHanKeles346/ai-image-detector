# PixelProof — AI image evidence demo

PixelProof is a research system that looks for evidence consistent with AI generation. Its
official decision is deliberately asymmetric: **`AI detected`** or **`insufficient evidence`**.
It never certifies that an image is real.

## Current scientific contract (2026-08-24)

The served decision layer is the E26 OR rule over frozen external detectors:

- **Community-Forensics ViT-S** is the default arm (MIT, pinned local snapshot).
- **B-Free** is optional because its upstream terms are research/non-commercial; it loads only
  after artifact verification and explicit `PIXELPROOF_BFREE=1` acknowledgement.
- Each arm uses a threshold fitted on source-wise calibration halves. Evaluation halves never
  select a threshold, model or gate.
- On the measured 12-source evaluation library, the two-arm union's worst-source false-positive
  point estimate is **10.7%** on iPhone originals (11/103; Wilson 95% CI 6.1–18.1%). This is a
  limited-population estimate, not a guarantee for a new camera, editor or upload pipeline.

E27's project-trained GPT-family arm is **not served**. A 2026-08-24 audit found that its union
threshold procedure could inspect evaluation halves. The corrected calibration-only rerun raised
the candidate threshold from 15.38 to 21.71 and reduced its in-collection GPT recall from 40.5%
to 14.5%, below its pre-registered 40% gate. The append-only correction is in
[`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md).

The UI also exposes four older project-trained methods (`auto`, `cnn`, `stats`, `tiles`) as
**uncalibrated research scores**. They are not the verdict: unseen-camera false positives reached
79–100% in E13/E24. The tile overlay is a detector-score map, not validated localisation evidence.

Module 2 (“where was it edited?”) is parked. It does not resume until a localisation model is
measured against pixel masks on the relevant manipulation family; current signal was limited to
diffusion inpainting and did not generalise to classic splicing.

## Repository map

| Path | Purpose |
|---|---|
| `app/` | Turkish web client |
| `ml/src/pixelproof/` | Models, input policy, decision protocol, CLI and FastAPI service |
| `ml/experiments/` | Runnable E20–E27 protocol scripts |
| `ml/EXPERIMENTS.md` | Append-only measured experiment log |
| `ml/artifacts.manifest.json` | Pinned model identity, hashes, licences and schemas |
| `PLAN.md` | Living roadmap and measured hardening results |
| `HISTORY.md` | Append-only chronological project and decision archive |
| `DATASETS.md` | Dataset inventory, allowed uses and portable path contract |
| `rapor/` | Historical report/talk snapshot; see its boundary note |

Label convention everywhere is `1 = AI-generated`, `0 = real`.

## Reproducible setup

The checked serving lock targets Python 3.13 on macOS arm64. PyTorch wheels are platform-specific,
so another platform must deliberately re-resolve the declared ranges in `ml/pyproject.toml`.

```bash
cd ml
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-serving.lock
.venv/bin/pip install --no-deps -e .
```

Weights are intentionally not committed. Follow [`ml/ARTIFACTS.md`](ml/ARTIFACTS.md), place the
five project-owned files (including the canonical E20 model), prepare the pinned MIT snapshot,
then verify everything before serving:

```bash
.venv/bin/pixelproof-artifacts prepare
.venv/bin/pixelproof-artifacts check
PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --host 127.0.0.1 --port 8799
```

Missing core artifacts produce a truthful `status=unavailable`; missing verdict artifacts produce
`status=degraded`. Neither condition crashes module import. Input limits, CORS and required edge
authentication/rate controls are documented in [`ml/SERVING.md`](ml/SERVING.md).

Prepared datasets default to `ml/work/` and acquired sources to `ml/data/`. Existing layouts are
selected without code edits:

```bash
export PIXELPROOF_WORK_ROOT=/path/to/prepared-work
export PIXELPROOF_DATA_ROOT=/path/to/source-datasets
```

## Web client

```bash
npm ci
npm run dev
```

Development deliberately falls back to `http://127.0.0.1:8799`. A production build posts to
same-origin `/predict` unless `NEXT_PUBLIC_PIXELPROOF_API_URL` is set to an absolute HTTP(S)
origin; see [`.env.example`](.env.example).

## Verification

```bash
npm test
npm run lint
npm run typecheck

PYTHONPATH=ml/src ml/.venv/bin/pytest -q
PYTHONPATH=ml/src ml/.venv/bin/python -m pixelproof.artifact_registry check
ml/.venv/bin/pip check
```

Known dependency debt: `npm audit` currently reports two high entries in one development/build
chain, `vinext@0.0.50 -> image-size@2.0.2`. npm's offered remediation is the breaking
`vinext@1.0.0-beta.8` line; CI fails on critical advisories while this migration remains explicit.

This repository currently grants no open-source licence. See [`LICENSE.md`](LICENSE.md) for the
project and external-model boundaries.
