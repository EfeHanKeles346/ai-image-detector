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
- `POST /predict` accepts `JPG`, `PNG` or `WEBP`. Its default `project_model` method runs the
  verified E20 native-tile ResNet-18. `auto`, `cnn`, `stats` and `tiles` retain the older
  research paths for comparison until M3 moves them behind research details.
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

## Required edge controls

Before binding outside loopback, require all of the following at the proxy/gateway:

1. authentication or a scoped service token;
2. per-principal and per-IP request limits plus a small bounded upstream queue;
3. a request-body limit no larger than the worker's 12 MiB limit;
4. TLS, explicit timeouts and removal of uploaded bodies from access logs;
5. monitoring for 413/429/5xx rates, queue time, inference latency and worker readiness.

The worker intentionally does not persist uploads. Infrastructure must not add body logging
or durable request capture without a separate privacy/retention decision.
