# Runnable experiment index

These scripts are the reproducible protocol surface. E7–E18 are frozen under `archive/`; E20–E28
remain runnable because later decisions reuse their cached scores and source-wise splits.

Run from `ml/` with the declared environment and portable dataset roots:

```bash
export PIXELPROOF_DATA_ROOT=/path/to/source-datasets
export PIXELPROOF_WORK_ROOT=/path/to/prepared-work
PYTHONPATH=src .venv/bin/python experiments/e20_tile_model_shootout.py --help
```

| Script | State | Purpose |
|---|---|---|
| `e20_tile_model_shootout.py` | measured baseline | Three native-tile model families; image-level aggregation, disjoint calibration/evaluation, multi-seed and worst-source FP |
| `e21_external_detector_benchmark.py` | measured baseline | Pinned Community-Forensics and opt-in B-Free through the same protocol |
| `e22_source_robust_calibration.py` | adopted | Worst-source calibration and asymmetric decision operating points |
| `e22b_bootstrap_ci.py` | adopted reporting rule | Source-wise bootstrap intervals for headline operating points |
| `e23a_midjourney_diagnostic.py` | measured | Why a symmetric “real” verdict is unsafe |
| `e23b_megapixel_policy.py` | adopted for B-Free | 2048px B-Free input cap on megapixel authentic sources |
| `e23c_compression_column.py` | measured limitation | Clean/degraded threshold domains; bytes-per-pixel remains only a heuristic |
| `e24_library_promise.py` | measured | New iPhone pipeline, threshold-only calibration and held-out transfer |
| `e25_modern_generator_probe.py` | measured limitation | Modern-generator recall probes, including GPT-family blindness |
| `e27_gpt_family_arm.py` | **rejected** | GPT specialist experiment; corrected calibration-only union threshold fails G1 (14.5% < 40%), so it is not served |
| `pixelproof-train-stay-positive` | candidate, not served | E28 independent Stay-Positive head training over the frozen E20 backbone; N2/N3 gates decide whether it advances |
| `e29_saneval_2025_probe.py` | pre-registered diagnostic | Pinned, balanced, sub-100 MB 2025-generator JPEG subset and frozen-threshold CF-ViT recall probe |
| `e30_data_system.py` | active data/OOD system | Five-role manifests, deterministic capped/resumable acquisition, shortcut audit and sealed current-generator selection |
| `e31_ssd_audit.py` | active TRAIN-v2 audit | Read-only attached-disk inventory, verified label direction, bounded shard-spread decode/shortcut probes and explicit hash-coverage boundary |
| `e31_train_v2.py` | active TRAIN-v2 contract | Metadata freeze, full-pool protected eligibility screening, exact row freeze and deterministic native-tile realization |
| `e31_representation_ladder.py` | active representation screen | SHA-pinned E31 tiles; E20 control vs frozen DINOv2 vs 68 forensic features with TRAIN-OOF thresholds and untouched CALIBRATION |
| `e31_ensemble.py` | active fusion gate | Five-fold group-cross-fitted max/stack rules, row-level complementarity and paired group-bootstrap acceptance |
| `e31_qwen_locked.py` | sealed final scout | Refuses access without a committed passing E31 DEVELOPMENT receipt; then scores the fixed 40 native + 40 standardized Qwen rows once |
| `e31_score_folder.py` | research-only E31 inspection | Hash-verified rejected DINO candidate over a folder; immutable threshold, asymmetric verdicts and explicit DEVELOPMENT warning |

The current served decision contract is E26's OR rule, implemented in `pixelproof/verdict.py`:
CF-ViT is default; B-Free is optional and licence-gated. E27 is retained to reproduce the
rejection, not as a deployment recipe. Exact results and the append-only correction are in
`../EXPERIMENTS.md`.

Feature extraction caches under `artifacts/` are not committed. Runtime artifacts must pass
`pixelproof-artifacts check`; see `../ARTIFACTS.md`.

E28 is an installed research command rather than a stand-alone script. From any working directory:

```bash
pixelproof-train-stay-positive --help
```

Its default output is `artifacts/e28/stay_positive_seed2024.pt`. It never overwrites or registers
the served E20 checkpoint; evaluation remains a separate, gated phase.
