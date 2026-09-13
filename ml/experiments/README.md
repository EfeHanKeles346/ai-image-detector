# Runnable experiment index

These scripts are the reproducible protocol surface. E7–E18 are frozen under `archive/`; E20–E32
remain runnable because later decisions reuse their receipts, cached scores and source-wise splits.
Runnable means reproducible research, not served or scientifically accepted.

E60's current offline path is `e60_audit.py` (teacher exposure/admission audit),
`e60_correction.py` (bounded additive logit mechanism) and `e60_run.py` (single frozen fit and
consumed E49 diagnosis). Run stages only in PLAN order; immutable completed stages are not
rerun. `e60_run` denies network connections before importing model dependencies. No acquisition
helper is needed. E43 remains frozen and serving promotion is explicitly disabled.

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
| `e32_ai_inventory.py`, `e32_ai_pool_selection.py` | E32 AI metadata boundary | Audits generator identity/licence and freezes the exact source-capped modern-AI selection before bytes |
| `e32_archive_inventory.py`, `e32_data_system.py`, `e32_source_realization.py`, `e32_eligibility_overlay.py`, `e32_role_manifest.py` | E32 data contract | Safe archive inventory, role-free decode/decontamination, global eligibility and group-disjoint TRAIN/CALIBRATION roles |
| `e32_gap_acquisition.py`, `e32_gpt_acquisition.py` | receipt-bound acquisition | Resumable exact-row transfer tools; network work is allowed only after a committed selection and decoder gate |
| `e32_r0_input.py`, `e32_r0_train.py`, `e32_r0_loco.py` | rejected E32 R0 | Standardized input receipt, frozen-DINO head and leave-one-real-source-out diagnostic |
| `e32_cfvit_train.py`, `e32_owner_gallery_smoke.py`, `e32_r1a_gallery_smoke.py` | rejected R1a control | CF-ViT representation control and consumed owner-gallery DEVELOPMENT gate |
| `e32_r1b_acquisition.py`, `e32_r1b_csafe_iphone14.py`, `e32_r1b_ipn_audit.py` | R1b corrective data path | Frozen iPhone/IPN selections, safe extraction and independent authentic-pipeline realization |
| `e32_r1b_iphone14_audit.py`, `e32_r1b_iphone14_eligibility.py`, `e32_r1b_role_manifest.py`, `e32_r1b_input.py` | R1b role/input contract | Removes one confirmed burst, extends roles and reproduces the shared standardized input |
| `e32_r1b_train.py`, `e32_r1b_select.py`, `e32_r1b_external.py` | rejected R1b experiment | DINO/CF controlled heads, preregistered CF selection and one consumed external DEVELOPMENT run |
| `e36_acquisition.py`, `e36_calibrate.py`, `e37_source_heldout.py` | modern balanced recovery | Frozen disjoint CAL/FINAL acquisition, rejected DDA threshold transfer and fixed source-held-out DINO adaptation |
| `e38_fixed_adaptation.py` | fixed pre-FINAL candidate | Uniformly emphasizes every consumed modern adaptation row and freezes one development-selected DINO head |
| `e38_final.py` | one-shot untouched FINAL | Hash-binds the E38 candidate/threshold and scores the family/device-disjoint native/clean parents exactly once |

The canonical project model is the hash-verified E20 tile ResNet-18. Runtime profiles may also
load E26's comparison OR rule and the separately opted-in, non-voting R1b research card; these
must not be confused with the E43/E51 research experiments. See `../SERVING.md` for the actual
API/runtime contract. E27 is retained to reproduce its rejection, not as a deployment recipe.

### E53 offline research — completed office slice (2026-09-09)

| Scripts | Result/boundary |
|---|---|
| `e53_offline.py`, `e53_report.py` | Immutable six-arm core and expanded reports; TRAIN-only folds, no promotion |
| `e53_native_inventory.py`, `e53_latest_reserves.py`, `e53_expansion.py` | Original-body/reserve admission and 5,652 native TRAIN additions; no download |
| `e53_head_controls.py`, `e53_coverage.py` | Twelve configurations in each of two protocols, 72 fold fits; no guard survivor |
| `e53_diagnostics.py`, `e53_shortcut_audit.py` | Bound-prediction source/ROC diagnostics and input associations; no threshold tuning |
| `e53_weight_benchmark.py` | Bitwise-identical linear-time training weight helper |
| `e53_crop_dedup.py`, `e53_crop_dedup_full.py` | Optional exact-crop prototype, numerical parity on all current TRAIN clean/Q75 views |
| `e53_adaptation_probe.py` | Last-two-block autograd/resource feasibility only; no adapted candidate saved |
| `e53_artifact_replay.py` | All 72 saved heads reproduce 286,944 archived predictions exactly; not a new final |

Run E53 modules from the repository root with `PYTHONPATH=ml:ml/src`, the existing `ml/.venv`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and the correct `PIXELPROOF_DATA_ROOT`. Completed
freeze/extract/report phases deliberately refuse overwriting immutable receipts. Inspect existing
`evidence/e53_*.json` and disk outputs before any resume. Do not edit hash-pinned implementations
and then expect old contracts to validate; use a separately documented successor protocol.
No E53 job is a deployment command. Native order/scale expansion and an actual backbone-adaptation
study remain next-step work, not completed training. Exact methods/failures live in `../EXPERIMENTS.md`.

Feature extraction caches under `artifacts/` are not committed. Runtime artifacts must pass
`pixelproof-artifacts check`; see `../ARTIFACTS.md`.

E28 is an installed research command rather than a stand-alone script. From any working directory:

```bash
pixelproof-train-stay-positive --help
```

Its default output is `artifacts/e28/stay_positive_seed2024.pt`. It never overwrites or registers
the served E20 checkpoint; evaluation remains a separate, gated phase.


### E61 strict TRAIN replay check (2026-09-10)

`e61_replay_gate.py freeze` binds inputs/code, then `run` verifies the reusable
`pixelproof.retention_gate` against E43 self-replay and the known frozen E60 TRAIN result.
This is an engineering test, not a fit or E49 evaluation. Commands use the existing venv:
`PYTHONPATH=ml:ml/src ml/.venv/bin/python -m experiments.e61_replay_gate freeze` (then `run`),
with the external DATA_ROOT environment configured. CLI disables network connections before
third-party imports; evidence is write-once. Read PLAN and EXPERIMENTS before execution.


### E62 constrained correction (2026-09-10)

`e62_run.py freeze` then `fit` runs one registered TRAIN-only constrained logit correction
using cached features. `e62_constrained.py` contains the objective, linear preservation
constraints and saved projection. Read PLAN/EXPERIMENTS before use. Network denied at CLI
startup; outputs are write-once under the external e62 directory and compact git evidence.


### E62–E64 completed outcomes (2026-09-10)

E62 and E63 `freeze/fit` and separate `*_regression freeze/score/report` are finished;
both failed external AI retention. E64 `freeze/fit` is finished and failed its10% TRAIN
ceiling, so no regression runner/contract exists. Do not rerun or overwrite. All CLIs
block network connects before third-party imports and use immutable bound inputs.
49 focused tests pass; source code, registration and evidence remain for reproduction
review, not a new evaluation opportunity. Frozen historical scripts remain unchanged.


### E80–E86 current overnight continuation (2026-09-13)

| Modules | State and boundary |
|---|---|
| `e80_fit`, `e81_fit` | Frozen TRAIN rejections; same multimodal map, different declared objectives |
| `e82_representation`, `e83_fit` | Supervised64 features and constrained head; full TRAIN/runtime pass |
| `e83_development`, `e83_diagnostic` | Consumed DEV rejected despite REAL improvement; exact read-only component replay |
| `e84_features` | Probe stopped on native source-key schema; original code/64 chunks retained |
| `e84_sources` | All12,141 original TRAIN bodies verified,1,000 Parquet bodies materialized locally |
| `e84b_features` | Same encoder/transform with verified paths; probe passed, full social-view extraction active |
| `e85_data`, `e85_representation` | Prepared four-condition pairing and same fixed supervised recipe; not frozen/fitted |
| `e86_gates`, `e86_model`, `e86_fit`, `e86_development` | Prepared80-check TRAIN screen and consumed cached DEV; no fit or DEV contract yet |

Use the repo-root module form with the external environment:
`PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets PYTHONPATH=ml/src:ml ml/.venv/bin/python -m experiments.<module> <stage>`.
Inspect existing receipts and active processes first. Do not rerun completed `freeze`,
fit or DEV stages. E84B `extract` may resume missing fixed windows only when the existing
execution is no longer active. No E49 access, serving change or promotion in this chain.

E87: `e87_acquisition freeze|download` acquires128 fixed SID long-RAW originals into
quarantine with source/CRC/SHA/range bounds. No decode, admission or model scoring.

E88: prepared `e88_audit freeze|audit` reuses the existing isolated RAW renderer,
protects original/derived identities and resolves SID duplicate components before admission.

E86 consumed regression: prepared `e86_regression freeze|score|report`; every E49
access requires the complete TRAIN/runtime and E66 DEV pass. Not frozen/executed yet.

E89: prepared `e89_features freeze|extract` creates the separate SID four-condition
TRAIN cache only after E84B/E85/E86; no new-cohort classifier scores or automatic fit.

E90: registered `e90_head_diagnostic freeze|run` reads the existing E82 scalar-head
sign within unchanged E83 consumed-DEV transition bins; no new fit or detection candidate.

E90 complete: exact640-view E83 replay; the original newly missed AI also has a
negative E82 scalar logit. A sign-agreement fallback is not pursued; details in the receipt.
