"""Freeze the fixed E38 development-selected head before untouched FINAL access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_MODEL_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e37_source_heldout as e37  # noqa: E402


E38_ROOT = DATA_ROOT / "e38"
CONTRACT = ML_ROOT.parent / "evidence" / "e38_fixed_contract.json"
CONTRACT_SHA256 = "c61ec08006a6fe6e433dd6ea6a7cafa90b92d55a13ddbcc26a33260d95b2eedd"
OOF_SCORES = E38_ROOT / "oof_scores.jsonl"
REPORT = E38_ROOT / "development_report.json"
CANDIDATE = E38_ROOT / "e38_dinov2s.joblib"
EVIDENCE = ML_ROOT.parent / "evidence" / "e38_development.json"
C_VALUE = 0.0003
HISTORICAL_WEIGHT = 1.0
ADAPTATION_WEIGHT = 100.0
SEED = 42


def make_head() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_VALUE,
            class_weight="balanced",
            max_iter=3_000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )


def training_weights(base_count: int, adaptation_count: int) -> np.ndarray:
    if base_count <= 0 or adaptation_count <= 0:
        raise ValueError("both historical and adaptation rows are required")
    return np.concatenate([
        np.full(base_count, HISTORICAL_WEIGHT, dtype=np.float64),
        np.full(adaptation_count, ADAPTATION_WEIGHT, dtype=np.float64),
    ])


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    temporary.replace(path)


def run() -> dict[str, Any]:
    if any(path.exists() for path in (OOF_SCORES, REPORT, CANDIDATE, EVIDENCE)):
        raise FileExistsError("E38 output already exists; no silent rerun")
    if e37._sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("E38 fixed contract changed")
    manifest = e37._load_manifest()
    rows = manifest["rows"]
    base = e37._load_base()
    adaptation = e37._load_or_extract(rows, e37.BATCH_SIZE)
    if e37._sha256_file(e37.E36_FEATURES) != "3a08e0dcacce6441efa0d422531871b964d100b126a6884715f5616c1a26f178":
        raise ValueError("E36 DINOv2-S adaptation feature archive changed")
    base_train = base["roles"].astype(str) == "TRAIN"
    base_x = base["features"][base_train]
    base_y = base["labels"][base_train].astype(np.int64)
    adapt_x = adaptation["features"]
    adapt_y = adaptation["labels"].astype(np.int64)
    adapt_sources = adaptation["sources"].astype(str)
    assignments = e37.fold_ids(adapt_y, adapt_sources)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold_index in range(len(e37.FOLDS)):
        held = assignments == fold_index
        included = ~held
        train_x = np.concatenate([base_x, adapt_x[included]])
        train_y = np.concatenate([base_y, adapt_y[included]])
        weights = training_weights(len(base_y), int(included.sum()))
        head = make_head()
        head.fit(train_x, train_y, logisticregression__sample_weight=weights)
        oof[held] = head.predict_proba(adapt_x[held])[:, 1]
        fold_reports.append({
            "fold": fold_index,
            "held_rows": int(held.sum()),
            "fit_historical_rows": int(len(base_y)),
            "fit_adaptation_rows": int(included.sum()),
            "held_real_sources": sorted(e37.FOLDS[fold_index]["real"]),
            "held_ai_sources": sorted(e37.FOLDS[fold_index]["ai"]),
        })
        print(f"E38 fixed fold {fold_index + 1}/{len(e37.FOLDS)} scored {int(held.sum())}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("E38 did not produce one score per adaptation row")
    score_rows = [
        {
            "parent_id": row["parent_id"],
            "path": row["path"],
            "label": int(row["label"]),
            "source": row["source"],
            "condition": row["condition"],
            "fold": int(assignments[index]),
            "status": "ok",
            "score": float(oof[index]),
        }
        for index, row in enumerate(rows)
    ]
    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in score_rows)
    score_part = OOF_SCORES.with_suffix(OOF_SCORES.suffix + ".part")
    score_part.parent.mkdir(parents=True, exist_ok=True)
    score_part.write_bytes(score_raw)
    score_part.replace(OOF_SCORES)
    selected = e37.select_threshold(score_rows)
    metrics = evaluate_binary_scores(score_rows, threshold=float(selected["threshold"]))
    result_gate = e37.gate(metrics, selected)
    report = {
        "schema_version": 1,
        "experiment": "E38/fixed-adaptation-emphasis",
        "state": "development_gate_passed_candidate_frozen" if result_gate["passed"] else "development_gate_failed",
        "provenance_caveat": "Hyperparameters were selected from consumed E36/E37 DEVELOPMENT diagnostics; only untouched FINAL can validate this candidate.",
        "contract_sha256": CONTRACT_SHA256,
        "model": {
            "backbone": DINO_MODEL_ID,
            "backbone_weight_sha256": DINO_WEIGHT_SHA256,
            "head": "StandardScaler + LogisticRegression",
            "c": C_VALUE,
            "class_weight": "balanced",
            "historical_sample_weight": HISTORICAL_WEIGHT,
            "adaptation_sample_weight": ADAPTATION_WEIGHT,
            "seed": SEED,
        },
        "bindings": {
            "e32_feature_archive_sha256": e37.E32_FEATURES_SHA256,
            "e36_manifest_sha256": e37.E36_MANIFEST_SHA256,
            "e36_feature_archive_sha256": e37._sha256_file(e37.E36_FEATURES),
            "oof_scores_sha256": hashlib.sha256(score_raw).hexdigest(),
        },
        "counts": {"base_train": int(len(base_y)), "adaptation": int(len(adapt_y)), "oof": int(len(oof))},
        "folds": fold_reports,
        "selected_frontier": selected,
        "metrics": metrics,
        "bootstrap": {
            "real_macro_fp": e37._bootstrap_macro(score_rows, threshold=float(selected["threshold"]), label=0),
            "ai_macro_recall": e37._bootstrap_macro(score_rows, threshold=float(selected["threshold"]), label=1),
        },
        "gate": result_gate,
        "boundary": "Consumed DEVELOPMENT only; no FINAL archive/blob downloaded or scored.",
    }
    if result_gate["passed"]:
        final_head = make_head()
        final_x = np.concatenate([base_x, adapt_x])
        final_y = np.concatenate([base_y, adapt_y])
        final_head.fit(
            final_x,
            final_y,
            logisticregression__sample_weight=training_weights(len(base_y), len(adapt_y)),
        )
        artifact = {
            "schema_version": 1,
            "model_name": "E38 DINOv2-S fixed adaptation emphasis",
            "model_id": DINO_MODEL_ID,
            "model_weight_sha256": DINO_WEIGHT_SHA256,
            "input_size": e37.INPUT_SIZE,
            "positive_label": "ai",
            "threshold": float(selected["threshold"]),
            "head": final_head,
            "preprocessing": {
                "orientation": "PIL ImageOps.exif_transpose",
                "mode": "RGB",
                "resize_short_side": 256,
                "crop": "center-224",
                "encoding": "JPEG quality=90 subsampling=0 optimize=false progressive=false",
            },
            "contract_sha256": CONTRACT_SHA256,
            "bindings": report["bindings"],
        }
        temporary = CANDIDATE.with_suffix(".joblib.part")
        CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, temporary)
        temporary.replace(CANDIDATE)
        report["candidate"] = {
            "path": CANDIDATE.relative_to(E38_ROOT).as_posix(),
            "bytes": CANDIDATE.stat().st_size,
            "sha256": e37._sha256_file(CANDIDATE),
            "threshold": float(selected["threshold"]),
        }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
