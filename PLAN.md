# Plan — the living document

Everything that was decided, measured or abandoned lives in [`HISTORY.md`](HISTORY.md)
(frozen) and [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) (append-only log). This file holds
only what is *next*, so there is exactly one place to look and one place to update.

## Where the project stands (2026-08-18, after E20-v2 / E21 protocol work)

- **Best detector:** ResNet-18 fine-tuned on 128px native tiles — Defactify AUC 0.770,
  61.4% AI recall on the untouched evaluation half. Best numbers the project has produced.
- **Why it is not deployable:** a threshold fitted for 10% false positives reaches 19% on
  Defactify's own held-out half and up to **96% on the worst unseen camera source**. The
  bottleneck is no longer data (Phase 1) or representation (E20) — it is the operating
  point under source/pipeline shift.
- **Module 2** (where was it edited?) is measured and parked: tile localisation carries
  signal only on diffusion inpainting (CocoGlide); the classic-splice line is closed.

## Next, in order

1. **External baselines through our protocol** — run frozen B-Free, then CLIP, through the
   E20-v2 evaluator (`ml/experiments/e21_external_detector_benchmark.py` is built for
   this). No training. If neither fixes cross-source specificity, that is a finding.
2. **Source-robust decision rule** — source-balanced real calibration, or an explicit
   "insufficient evidence" band keyed on measured high-frequency content. Only pay for a
   three-seed retrain once a configuration passes the cross-source gate.
3. **Report + demo polish** — the internship report writes itself from HISTORY.md +
   EXPERIMENTS.md; the demo stays as-is (three signals, unblended) until 1–2 change what
   it should say.

## Standing rules (unchanged)

- Headline metric = AI recall at a fixed FP budget on **unseen real sources**; AUC is
  reported alongside, never alone.
- ≥3 seeds on anything reported. Audit every dataset before use (`ml/tools/audit_datasets.py`).
- Thresholds are chosen on calibration halves and measured on untouched halves — always.

## Repo conventions after the 2026-08-18 tidy-up

- `ml/experiments/` holds only runnable protocol scripts (e20, e21); finished evidence
  scripts are frozen in `ml/experiments/archive/`.
- `ml/src/pixelproof/archive/` holds retired modules (E2–E4 analysis, ELA, DINOv2
  extraction). Nothing in the live path imports them.
- `ml/artifacts/archive/` (not committed) holds superseded artifacts, including the
  poisoned `*.BOZUK_etiket.bak` evidence files. Nothing is deleted.
