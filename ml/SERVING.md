# Serving boundary

PixelProof's FastAPI process is a model worker, not a public internet edge. It accepts
multipart image data and performs CPU/GPU-heavy inference, so a non-local deployment must
put an authenticated, rate-limited reverse proxy or API gateway in front of it. CORS is a
browser policy and is not authentication.

## Runtime contract

```bash
cd ml
PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --host 127.0.0.1 --port 8799
```

- `PIXELPROOF_CORS_ORIGINS` is a comma-separated allow-list. Its local default is
  `http://localhost:3000,http://127.0.0.1:3000`; wildcard origins are rejected. Same-origin
  deployments do not need CORS.
- `PIXELPROOF_RUNTIME_PROFILE=project` loads only E20; `demo` loads E20 plus the E26 comparison but
  skips retired legacy models; the default `full` also attempts those legacy methods. The local
  one-command demo uses `demo`. R1b additionally requires `PIXELPROOF_R1B=1` plus the correct
  `PIXELPROOF_DATA_ROOT`; load or inference failure affects only `r1b_research`.
- `POST /predict` accepts `JPG`, `PNG` or `WEBP`. Its default `project_model` method runs the
  verified E20 native-tile ResNet-18. `auto`, `cnn`, `stats` and `tiles` remain API-compatible in
  `full`, but are removed from the simplified browser flow.
- Default limits are 12 MiB, 16 million decoded pixels, 16,384 pixels per side and a 20:1
  maximum aspect ratio. EXIF orientation is applied, and transparent pixels are composited
  onto white before every detector sees the image.
- At most 256 native tiles are evaluated and each worker serializes inference. Horizontal
  capacity therefore comes from a bounded number of worker processes/replicas, not an
  unbounded request queue.
- Inputs below 48 pixels on either axis can still return a research score, but the official
  `decision` is suppressed and `enough_evidence` is false.
- `/health` distinguishes `starting`, `unavailable`, `degraded` and `ready`. `ready` now means
  the canonical project model is verified and loaded; `project_model_ready`, legacy `core_ready`
  and external `decision_ready` are reported separately so one optional subsystem cannot mask
  another's state.

## E31 non-integration decision

The E31 single-DINOv2 candidate is deliberately absent from `/predict`, readiness and the web UI.
Its frozen E30 DEVELOPMENT run measured 83.63% macro / 100% worst-group authentic false positives,
so the B6 condition “replace only after B5 passes” evaluated false. The verified E20 serving
contract remains unchanged. E31 may be run only through `experiments/e31_score_folder.py`; that
output is marked research-only and cannot claim authenticity. Qwen LOCKED FINAL was not scored.

## E32 non-integration decision

E32's rebuilt data contract and frozen DINOv2-S head are runnable, but the candidate is deliberately
absent from `/predict`, readiness and the web UI. It passed source-stratified CALIBRATION and then
misclassified 159/210 authentic owner-gallery DEVELOPMENT stills. The verified research CLI
`pixelproof-predict-e32` exposes scores for diagnosis only; below threshold is not proof of reality
and above threshold is not reliable enough for product action. No gallery-derived recalibration is
allowed for the frozen artifact.

R1a's frozen forensic CF-ViT representation is also excluded. Despite 0.9982 internal AUC, its
untouched threshold mislabeled 154/210 owner-gallery authentic stills (26.67% REAL recall). The
`pixelproof-predict-e32-cf` CLI exists for reproducibility only; it does not change `/predict`,
readiness or the web UI, and the owner gallery cannot recalibrate it.

R1b is excluded from the official decision for the same reason after adding 3,994 audited iPhone 14
photos. Its selected CF head mislabeled 249/960 independent IPN-NFID authentic photos (40.0%
worst-device FP) and 144/210 owner-gallery stills. The local `demo` profile may expose the exact
frozen head as an optional `r1b_research` card. It is explicitly `research_only`, cannot affect
E26's decision, readiness or the canonical registry, and below threshold means only insufficient
evidence. IPN/owner results cannot recalibrate or route it.

The local browser deliberately gives this optional R1b card visual priority so a demonstration
answers “what did the new model say?” directly. It shows why the label fired (score distance from
the frozen threshold) and identifies the percentage as a raw signal rather than a calibrated
probability. E26, E20 and external false-positive measurements sit under collapsed technical
details. This is presentation only: API fields, decision voting and promotion status are unchanged.

## Required edge controls

Before binding outside loopback, require all of the following at the proxy/gateway:

1. authentication or a scoped service token;
2. per-principal and per-IP request limits plus a small bounded upstream queue;
3. a request-body limit no larger than the worker's 12 MiB limit;
4. TLS, explicit timeouts and removal of uploaded bodies from access logs;
5. monitoring for 413/429/5xx rates, queue time, inference latency and worker readiness.

The worker intentionally does not persist uploads. Infrastructure must not add body logging
or durable request capture without a separate privacy/retention decision.
