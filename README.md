# PixelProof — AI image evidence demo

PixelProof is a research system that looks for evidence consistent with AI generation. Its
official decision is deliberately asymmetric: **`AI detected`** or **`insufficient evidence`**.
It never certifies that an image is real.

## Current scientific contract (2026-08-26)

The primary API, CLI and web-demo path is now the project-owned E20 ResNet-18 checkpoint
(`e20-tile-resnet18-seed2024`). It scores native 128 px tiles, aggregates the three highest
texture-qualified tile scores and compares the result with its stored calibration threshold.
Every load verifies the checkpoint's SHA-256 and inference schema. This is an experimental AI
signal, not an authenticity verdict: an under-threshold result is reported as uncertain, never
as proof that an image is real. On E20's three-seed evaluation, worst-source authentic false
positives were **86.2% +/- 3.1**.

The web UI presents the E26 OR rule over frozen external detectors in a separate comparison card:

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

E31 is also **not served**. It produced the project's cleanest new training contract (11,300
balanced, source-capped, protected-overlap-free native tiles) and a frozen DINOv2 probe reached
90.72% current-generator macro recall at 4.67% CALIBRATION macro real FP. The independent 900-view
E30 DEVELOPMENT gate then falsified it: 80.67% AI recall came with **83.63% macro real FP** and
AUC 0.385. A diagnostic threshold that restores the real budget leaves 0.33% AI recall, so this is
representation/source shift rather than a fixable threshold. Qwen LOCKED FINAL remained unscored.

E32 is the first new end-to-end candidate that is genuinely runnable after the modern-data rebuild,
but it is **not served**. A balanced 22,688-parent frozen-DINOv2-S screen reached 0.9964 internal
CALIBRATION AUC, then mislabeled 159/210 authentic owner-gallery DEVELOPMENT stills as AI. This
24.29% REAL recall exposes source/pipeline shift despite the strong internal score. The research
CLI remains useful for reproducible diagnosis; the model cannot certify authenticity and did not
consume LOCKED FINAL.

The follow-up R1a control reused the exact data/roles with a forensic CF-ViT representation. It
reached 0.9982 internal AUC but still mislabeled 154/210 owner stills (26.67% REAL recall). Because
two different encoders fail almost identically, further encoder shopping is paused until a fourth
licensed, provenance-complete authentic camera source and a REAL-source-held-out gate are added.

R1b completed that controlled data ablation with 3,994 audited CSAFE iPhone 14 photos. The selected
CF head retained 99.82% internal current-AI macro recall but failed untouched authentic transfer:
249/960 IPN-NFID photos and 144/210 owner-gallery photos were falsely flagged. It is therefore not
an official decision and no threshold was repaired on either set. The local demo can display its
frozen score as a clearly non-voting research card. The result shifts the next work from simply
downloading more volume to source-invariant objectives and a newly reserved multi-camera REAL gate.

Four older project-trained methods (`auto`, `cnn`, `stats`, `tiles`) remain API-compatible only in
the `full` profile as **uncalibrated research scores**. They are neither the primary project
model nor the external comparison verdict: unseen-camera false positives reached 79–100% in
E13/E24. The tile overlay is a detector-score map, not validated localisation evidence.

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
| `MODEL_CARD.md` | Canonical E20 identity, training data, metrics, limits and allowed uses |
| `PRESENTATION_EVIDENCE.md` | Current internship presentation ledger and reproducible demo scenario |
| `evidence/` | Machine-readable presentation evidence bound to artifact/input hashes |
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

### One-command local demo

After the reproducible setup above, start the verified E20/E26 API and Turkish web UI together from
the repository root:

```bash
./tools/pixelproof-demo start
```

`start` first checks the venv, imports, dependency graph, canonical checkpoint hash, installed
project CLIs, Node/npm installation and loopback ports. It starts the API in lightweight demo mode,
waits for truthful readiness, submits the tracked smoke image and validates the project-model
response before starting the web UI at `http://127.0.0.1:3000`. Press `Ctrl+C` once to stop both
processes. Retired legacy models are not loaded.

To add the frozen R1b score as a visibly non-voting research card from an external dataset disk:

```bash
./tools/pixelproof-demo start \
  --r1b-data-root /path/to/pixelproof-datasets
```

The command refuses a missing artifact and `/health` reports `r1b_research_ready`; the page keeps
E26 as its measured decision even when R1b disagrees.

The same checks and API smoke can be run separately when diagnosing a startup:

```bash
./tools/pixelproof-demo check
./tools/pixelproof-demo smoke --api-url http://127.0.0.1:8799
```

Use `--api-port` and `--web-port` with `check`/`start` when the defaults are intentionally occupied.
Errors identify the missing dependency, artifact or port and include the corrective command.

### Evaluate the project model on labelled folders

Place supported JPG, PNG or WEBP files under separate class roots. Immediate or nested folder
names are retained as source groups in the report.

```bash
cd ml
.venv/bin/pixelproof-evaluate-project \
  --real /path/to/evaluation/real \
  --ai /path/to/evaluation/ai \
  --output artifacts/my-evaluation
```

The command verifies and loads the canonical E20 checkpoint, scores each discovered image once,
then writes `results.json` and `predictions.csv`. The JSON includes image/error counts, ROC-AUC,
recall and false-positive rate at the checkpoint's stored threshold, confusion counts, per-folder
AI-signal rates, full checkpoint/configuration/environment/command provenance and every individual
row. Decode and inference failures remain in both files and make the command exit non-zero after
writing the report. A non-empty output directory is never overwritten.

The rejected-but-runnable E31 candidate can be inspected separately without changing the service:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=ml/src ml/.venv/bin/python \
  ml/experiments/e31_score_folder.py /path/to/images \
  --output ml/data/e31/my_folder_scores.json
```

This research-only JSON never fits a new threshold and always carries the measured 83.63% real-FP
warning. An under-threshold score is `insufficient_evidence`, never a claim that the image is real.

The newer E32 research candidate is also runnable without changing the API/web model:

```bash
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
HF_HUB_OFFLINE=1 PYTHONPATH=ml/src ml/.venv/bin/python \
  -m pixelproof.e32_candidate /path/to/image-or-folder
```

It emits one JSON score/verdict per image and verifies the head plus DINO weights before loading.
Its owner-gallery REAL recall is only 24.29%, so the output is diagnostic rather than actionable.

The rejected R1a forensic-representation control is also reproducible:

```bash
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
HF_HUB_OFFLINE=1 PYTHONPATH=ml/src ml/.venv/bin/python \
  -m pixelproof.e32_cfvit_candidate /path/to/image-or-folder
```

R1a verifies its head, pinned CF revision and weight hash. Its owner-gallery REAL recall is 26.67%,
so it is not an authenticity verdict and is not integrated into the API or web application.

The rejected R1b clean-real-data ablation is reproducible with its separately verified CLI:

```bash
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
HF_HUB_OFFLINE=1 PYTHONPATH=ml/src ml/.venv/bin/python \
  -m pixelproof.e32_r1b_candidate /path/to/image-or-folder
```

R1b binds artifact SHA `68a54aa2...701c` and threshold 0.125935. Its 40.0% worst-device IPN FP
and 68.57% owner-gallery FP make every output research-only.

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
