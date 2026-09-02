"""Fit the E43-S transport-adapted head without scoring RR DEVELOPMENT."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e42_features import BLOCKS, MODEL_IDS
from experiments.e42_train import _fit_view_mask, select_threshold
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43"
E42_FEATURES = DATA_ROOT / "e42" / "features" / "small.npz"
E42_FEATURES_SHA256 = "452fec98990aaf425a2ae2a494a16d9c0d9c111d02cdda60e9e31529a069ac5a"
RR_FEATURES = ROOT / "rr_features_small.npz"
RR_FEATURES_SHA256 = "fdc5d4c8b28136898eb1431939b6c38997a6dd501153fd545a0cb092f5ca4aa4"
CANDIDATE = ROOT / "e43_small_predev.joblib"
REPORT = ROOT / "e43_small_predev.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e43_small_predev.json"
C_VALUE = 0.01
SEED = 42


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if _digest(path) != expected_sha256:
        raise ValueError(f"E43 feature archive changed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    if values["features"].ndim != 2 or values["features"].shape[1] != 3_072:
        raise ValueError(f"E43 feature width changed: {path}")
    return values


def parent_source_weights(
    labels: np.ndarray, sources: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(sources).astype(str)
    parents = np.asarray(parents).astype(str)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("E43 fit requires both labels")
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in (0, 1):
        label_mask = labels == label
        label_sources = sorted(set(sources[label_mask].tolist()))
        for source in label_sources:
            source_mask = label_mask & (sources == source)
            source_parents = sorted(set(parents[source_mask].tolist()))
            for parent in source_parents:
                mask = source_mask & (parents == parent)
                weights[mask] = 1.0 / (
                    2 * len(label_sources) * len(source_parents) * int(mask.sum())
                )
    if not np.all(weights > 0) or not np.isfinite(weights).all():
        raise ValueError("invalid E43 parent/source weights")
    return weights * (len(weights) / weights.sum())


def fit_population(
    e42: Mapping[str, np.ndarray], rr: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    e42_mask = _fit_view_mask(e42)
    rr_mask = rr["roles"].astype(str) == "train"
    return {
        "features": np.concatenate([e42["features"][e42_mask], rr["features"][rr_mask]]),
        "labels": np.concatenate([e42["labels"][e42_mask], rr["labels"][rr_mask]]),
        "sources": np.concatenate([e42["sources"][e42_mask], rr["sources"][rr_mask]]).astype(str),
        "parents": np.concatenate([e42["parent_ids"][e42_mask], rr["parent_ids"][rr_mask]]).astype(str),
        "origin": np.concatenate([
            np.full(int(e42_mask.sum()), "e42_consumed", dtype="U16"),
            np.full(int(rr_mask.sum()), "rr_train", dtype="U16"),
        ]),
    }


def fit() -> dict[str, Any]:
    if any(path.exists() for path in (CANDIDATE, REPORT, EVIDENCE)):
        raise FileExistsError("E43 pre-development candidate already exists; no silent replacement")
    e42 = _load(E42_FEATURES, E42_FEATURES_SHA256)
    rr = _load(RR_FEATURES, RR_FEATURES_SHA256)
    population = fit_population(e42, rr)
    if population["features"].shape != (19_648, 3_072):
        raise ValueError(f"E43 fit population changed: {population['features'].shape}")
    weights = parent_source_weights(
        population["labels"], population["sources"], population["parents"]
    )
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C_VALUE, max_iter=1_000, random_state=SEED, solver="lbfgs"),
    )
    head.fit(
        population["features"],
        population["labels"],
        standardscaler__sample_weight=weights,
        logisticregression__sample_weight=weights,
    )

    cal = (rr["roles"].astype(str) == "calibration") & (rr["conditions"].astype(str) == "original")
    if int(cal.sum()) != 980:
        raise ValueError("E43 original CAL population changed")
    cal_scores = head.predict_proba(rr["features"][cal])[:, 1]
    cal_rows = [
        {
            "record_id": str(record),
            "parent_id": str(parent),
            "label": int(label),
            "source": str(source),
            "condition": "original",
            "score": float(score),
            "status": "ok",
        }
        for record, parent, label, source, score in zip(
            rr["record_ids"][cal], rr["parent_ids"][cal], rr["labels"][cal],
            rr["sources"][cal], cal_scores, strict=True
        )
    ]
    threshold_report = select_threshold(cal_rows)
    threshold = float(threshold_report["threshold"])
    cal_metrics = evaluate_binary_scores(cal_rows, threshold=threshold)
    artifact = {
        "schema_version": 1,
        "model_name": "E43-S RR transport-adapted DINOv2 intermediate head",
        "status": "research_candidate_frozen_before_consumed_development",
        "positive_label": "ai",
        "model_id": MODEL_IDS["small"],
        "block_indices": BLOCKS["small"],
        "head": head,
        "threshold": threshold,
        "feature_contract": "global plus two deterministic texture crops; per-block crop mean+std",
        "e42_feature_sha256": E42_FEATURES_SHA256,
        "rr_feature_sha256": RR_FEATURES_SHA256,
    }
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANDIDATE.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(CANDIDATE)
    report = {
        "schema_version": 1,
        "state": "e43_small_candidate_frozen_before_consumed_development",
        "candidate_path": str(CANDIDATE),
        "candidate_bytes": CANDIDATE.stat().st_size,
        "candidate_sha256": _digest(CANDIDATE),
        "fit": {
            "views": len(population["labels"]),
            "parents": len(set(population["parents"].tolist())),
            "by_origin": dict(sorted(Counter(population["origin"].tolist()).items())),
            "by_label": {str(label): int(np.sum(population["labels"] == label)) for label in (0, 1)},
            "source_parent_balanced": True,
            "C": C_VALUE,
        },
        "calibration": {
            "rows": len(cal_rows),
            "threshold": threshold,
            "threshold_rates": threshold_report,
            "metrics": cal_metrics,
        },
        "rr_development_scores_created": 0,
        "itwsm_scores_created": 0,
        "boundary": "Candidate and threshold frozen using TRAIN/CAL only; RR DEVELOPMENT and ITW-SM remain unscored.",
    }
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    REPORT.write_bytes(raw)
    EVIDENCE.write_bytes(raw)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit",))
    args = parser.parse_args(argv)
    result = fit() if args.command == "fit" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
