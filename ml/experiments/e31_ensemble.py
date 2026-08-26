"""E31/B4 cross-fitted fusion and complementarity gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e31_representation_ladder import (
    CURRENT_AI_SOURCES,
    MACRO_REAL_FP_BUDGET,
    SEED,
    SELECTION_SHA256,
    TILE_SHA256,
    linear_oof,
    load_contract,
    sha256_file,
    threshold_at_source_fp_budget,
)

META_FOLDS = 5
BOOTSTRAP_REPLICATES = 2000
MIN_GAIN = 0.05
DINO_MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
DINO_INPUT_PX = 224
DINO_WEIGHT_SHA256 = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"


def assign_meta_folds(sources: np.ndarray, groups: np.ndarray, n_folds: int = META_FOLDS) -> np.ndarray:
    """Stable source-stratified group folds; no group crosses a fold."""
    sources = np.asarray(sources).astype(str)
    groups = np.asarray(groups).astype(str)
    result = np.full(len(groups), -1, dtype=np.int8)
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        for index, group in enumerate(sorted(set(groups[selected].tolist()))):
            result[selected & (groups == group)] = index % n_folds
    if (result < 0).any():
        raise RuntimeError("meta-fold assignment is incomplete")
    return result


def metrics_from_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    sources = np.asarray(sources).astype(str)
    real = labels == 0
    ai = labels == 1
    real_fp = {
        source: float(np.mean(predictions[real & (sources == source)]))
        for source in sorted(set(sources[real].tolist()))
    }
    ai_recall = {
        source: float(np.mean(predictions[ai & (sources == source)]))
        for source in sorted(set(sources[ai].tolist()))
    }
    current = [ai_recall[source] for source in CURRENT_AI_SOURCES]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ai_recall": float(np.mean(predictions[ai])),
        "real_fp": float(np.mean(predictions[real])),
        "macro_real_fp": float(np.mean(list(real_fp.values()))),
        "worst_real_fp": float(max(real_fp.values())),
        "real_fp_by_source": real_fp,
        "ai_recall_by_source": ai_recall,
        "current_ai_macro_recall": float(np.mean(current)),
        "current_ai_worst_source_recall": float(min(current)),
    }


def _stack_model(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=seed),
    )


def crossfit_rule(
    kind: str,
    component_scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    meta_folds: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float], Any | None, float]:
    """Cross-fit coefficients and thresholds, then fit the final CALIBRATION rule."""
    scores = np.asarray(component_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    oof_scores = np.full(len(labels), np.nan, dtype=np.float64)
    predictions = np.zeros(len(labels), dtype=bool)
    thresholds: list[float] = []
    for fold in sorted(set(meta_folds.tolist())):
        fit = meta_folds != fold
        held = meta_folds == fold
        if kind == "single":
            fit_scores = scores[fit, 0]
            held_scores = scores[held, 0]
        elif kind == "max":
            fit_scores = scores[fit].max(axis=1)
            held_scores = scores[held].max(axis=1)
        elif kind == "stack":
            model = _stack_model(seed + int(fold))
            model.fit(scores[fit], labels[fit])
            fit_scores = model.predict_proba(scores[fit])[:, 1]
            held_scores = model.predict_proba(scores[held])[:, 1]
        else:
            raise ValueError(f"unknown rule kind: {kind}")
        real_fit = labels[fit] == 0
        threshold, _ = threshold_at_source_fp_budget(
            fit_scores[real_fit], sources[fit][real_fit]
        )
        thresholds.append(threshold)
        oof_scores[held] = held_scores
        predictions[held] = held_scores >= threshold
    if np.isnan(oof_scores).any():
        raise RuntimeError("fusion out-of-fold score coverage is incomplete")

    final_model = None
    if kind == "single":
        full_scores = scores[:, 0]
    elif kind == "max":
        full_scores = scores.max(axis=1)
    else:
        final_model = _stack_model(seed)
        final_model.fit(scores, labels)
        full_scores = final_model.predict_proba(scores)[:, 1]
    real = labels == 0
    final_threshold, _ = threshold_at_source_fp_budget(full_scores[real], sources[real])
    return oof_scores, predictions, thresholds, final_model, final_threshold


def group_bootstrap_gain(
    labels: np.ndarray,
    sources: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int = SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    """Paired source-stratified group bootstrap for current-AI macro recall gain."""
    labels = np.asarray(labels)
    sources = np.asarray(sources).astype(str)
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    group_rows: dict[str, list[np.ndarray]] = {}
    for source in CURRENT_AI_SOURCES:
        mask = (labels == 1) & (sources == source)
        group_rows[source] = [np.flatnonzero(mask & (groups == group)) for group in sorted(set(groups[mask]))]
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences = []
        for source in CURRENT_AI_SOURCES:
            units = group_rows[source]
            chosen = rng.integers(0, len(units), size=len(units))
            rows = np.concatenate([units[value] for value in chosen])
            differences.append(float(np.mean(candidate[rows]) - np.mean(baseline[rows])))
        samples[index] = float(np.mean(differences))
    return {
        "replicates": replicates,
        "mean_gain": float(samples.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, default=Path("ml/data/e31/train_v2_tiles.npz"))
    parser.add_argument(
        "--selection", type=Path, default=Path("evidence/e31_train_v2_selection_v3.json")
    )
    parser.add_argument("--features", type=Path, default=Path("ml/data/e31/b3_features.npz"))
    parser.add_argument("--scores", type=Path, default=Path("ml/data/e31/b4_scores.npz"))
    parser.add_argument("--output", type=Path, default=Path("ml/data/e31/b4_ensemble.json"))
    parser.add_argument("--candidate", type=Path, default=Path("ml/artifacts/e31/b4_candidate.joblib"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    contract = load_contract(args.tiles, args.selection)
    with np.load(args.features, allow_pickle=False) as stored:
        matrices = {name: stored[name] for name in stored.files}
    labels = contract["y"].astype(np.int64)
    roles = contract["roles"].astype(str)
    sources = contract["sources"].astype(str)
    train = roles == "train"
    calibration = roles == "calibration"

    base_scores: dict[str, np.ndarray] = {"r0_e20": matrices["r0_e20"].astype(np.float64)}
    b3_thresholds: dict[str, float] = {}
    base_models: dict[str, Any] = {}
    for arm in ("r1_dinov2", "r2_forensic68"):
        oof, model = linear_oof(
            matrices[arm], labels, roles, contract["folds"], seed=args.seed
        )
        scores = np.full(len(labels), np.nan, dtype=np.float64)
        scores[train] = oof[train]
        scores[calibration] = model.predict_proba(matrices[arm][calibration])[:, 1]
        base_scores[arm] = scores
        base_models[arm] = model
    for arm, scores in base_scores.items():
        real_train = train & (labels == 0)
        b3_thresholds[arm], _ = threshold_at_source_fp_budget(
            scores[real_train], sources[real_train]
        )

    cal_labels = labels[calibration]
    cal_sources = sources[calibration]
    cal_groups = contract["groups"].astype(str)[calibration]
    cal_ids = contract["record_ids"].astype(str)[calibration]
    meta_folds = assign_meta_folds(cal_sources, cal_groups)
    cal_base = {arm: scores[calibration] for arm, scores in base_scores.items()}

    dino_b3 = cal_base["r1_dinov2"] >= b3_thresholds["r1_dinov2"]
    complementarity: dict[str, Any] = {}
    current_ai = (cal_labels == 1) & np.isin(cal_sources, CURRENT_AI_SOURCES)
    real = cal_labels == 0
    for arm in ("r0_e20", "r2_forensic68"):
        other = cal_base[arm] >= b3_thresholds[arm]
        complementarity[arm] = {
            "dino_current_ai_misses": int(np.sum(current_ai & ~dino_b3)),
            "incremental_current_ai_true_positives": int(np.sum(current_ai & ~dino_b3 & other)),
            "incremental_real_false_positives": int(np.sum(real & ~dino_b3 & other)),
            "decision_phi": float(np.corrcoef(dino_b3.astype(float), other.astype(float))[0, 1]),
            "score_pearson": float(np.corrcoef(cal_base["r1_dinov2"], cal_base[arm])[0, 1]),
        }

    definitions = [
        ("single_dinov2", "single", ("r1_dinov2",)),
        ("max_dinov2_e20", "max", ("r1_dinov2", "r0_e20")),
        ("stack_dinov2_e20", "stack", ("r1_dinov2", "r0_e20")),
        ("max_dinov2_forensic68", "max", ("r1_dinov2", "r2_forensic68")),
        ("stack_dinov2_forensic68", "stack", ("r1_dinov2", "r2_forensic68")),
    ]
    results: list[dict[str, Any]] = []
    saved_scores: dict[str, np.ndarray] = {
        "record_ids": cal_ids,
        "labels": cal_labels,
        "sources": cal_sources,
        "groups": cal_groups,
        "meta_folds": meta_folds,
        **{arm: values for arm, values in cal_base.items()},
    }
    final_contracts: dict[str, Any] = {}
    for name, kind, components in definitions:
        matrix = np.column_stack([cal_base[arm] for arm in components])
        oof_scores, predictions, thresholds, final_model, final_threshold = crossfit_rule(
            kind, matrix, cal_labels, cal_sources, meta_folds, seed=args.seed
        )
        metrics = metrics_from_predictions(cal_labels, oof_scores, predictions, cal_sources)
        saved_scores[f"score:{name}"] = oof_scores
        saved_scores[f"prediction:{name}"] = predictions
        final_contracts[name] = {
            "kind": kind,
            "components": components,
            "model": final_model,
            "threshold": final_threshold,
        }
        results.append(
            {
                "name": name,
                "kind": kind,
                "components": list(components),
                "fold_thresholds": thresholds,
                "final_threshold": final_threshold,
                "metrics": metrics,
            }
        )

    baseline = next(row for row in results if row["name"] == "single_dinov2")
    baseline_predictions = saved_scores["prediction:single_dinov2"]
    for row in results:
        row["gain_over_dinov2"] = (
            row["metrics"]["current_ai_macro_recall"]
            - baseline["metrics"]["current_ai_macro_recall"]
        )
        row["bootstrap_gain"] = group_bootstrap_gain(
            cal_labels,
            cal_sources,
            cal_groups,
            baseline_predictions,
            saved_scores[f"prediction:{row['name']}"] ,
            seed=args.seed,
        )
        row["fusion_acceptance"] = bool(
            row["name"] != "single_dinov2"
            and row["gain_over_dinov2"] >= MIN_GAIN
            and row["metrics"]["macro_real_fp"] <= MACRO_REAL_FP_BUDGET
            and row["metrics"]["worst_real_fp"] <= 0.10
            and row["bootstrap_gain"]["lower_95"] > 0.0
        )

    accepted = [row for row in results if row["fusion_acceptance"]]
    winner = max(accepted, key=lambda row: row["metrics"]["current_ai_macro_recall"]) if accepted else baseline
    contract = final_contracts[winner["name"]]
    artifact = {
        "schema_version": 1,
        "experiment": "E31/B4",
        "rule": winner["name"],
        "kind": contract["kind"],
        "components": contract["components"],
        "component_models": {
            arm: base_models[arm] for arm in contract["components"] if arm in base_models
        },
        "meta_model": contract["model"],
        "threshold": contract["threshold"],
        "dinov2": {
            "model_id": DINO_MODEL_ID,
            "input_px": DINO_INPUT_PX,
            "weight_sha256": DINO_WEIGHT_SHA256,
            "feature_dim": 384,
        },
        "selection_sha256": SELECTION_SHA256,
        "tile_archive_sha256": TILE_SHA256,
        "feature_cache_sha256": sha256_file(args.features),
    }
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.candidate)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.scores, **saved_scores)
    payload = {
        "schema_version": 1,
        "experiment": "E31/B4",
        "state": "fusion_accepted" if accepted else "fusion_rejected_single_arm_selected",
        "boundaries": {
            "selection_sha256": SELECTION_SHA256,
            "tile_archive_sha256": TILE_SHA256,
            "feature_cache_sha256": sha256_file(args.features),
            "meta_folds": META_FOLDS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "minimum_recall_gain": MIN_GAIN,
            "e30_opened": False,
        },
        "calibration": {
            "count": int(len(cal_labels)),
            "role_counts": {"calibration": int(len(cal_labels))},
            "meta_fold_counts": dict(Counter(map(int, meta_folds))),
        },
        "complementarity": complementarity,
        "rules": results,
        "winner": winner["name"],
        "candidate": {
            "path": str(args.candidate),
            "bytes": args.candidate.stat().st_size,
            "sha256": sha256_file(args.candidate),
        },
        "score_cache": {
            "path": str(args.scores),
            "bytes": args.scores.stat().st_size,
            "sha256": sha256_file(args.scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in results:
        metrics = row["metrics"]
        print(
            row["name"],
            f"current={metrics['current_ai_macro_recall']:.4f}",
            f"macro_fp={metrics['macro_real_fp']:.4f}",
            f"worst_fp={metrics['worst_real_fp']:.4f}",
            f"gain={row['gain_over_dinov2']:+.4f}",
            f"ci=[{row['bootstrap_gain']['lower_95']:+.4f},{row['bootstrap_gain']['upper_95']:+.4f}]",
            f"accept={row['fusion_acceptance']}",
        )
    print(f"winner={winner['name']} wrote {args.output}")


if __name__ == "__main__":
    main()
