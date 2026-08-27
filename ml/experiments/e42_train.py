"""Fit and evaluate the frozen E42 source-held-out head ladder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e42_features import BLOCKS, MODEL_IDS, assigned_transport
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E42_ROOT = DATA_ROOT / "e42"
FEATURE_ROOT = E42_ROOT / "features"
CONTRACT = REPO_ROOT / "evidence" / "e42_fixed_contract.json"
CONTRACT_SHA256 = "859d4ba812ba22678c0a6ec5e299244b999cab5d3b8ad72888d7c40f309ed279"
FEATURE_BINDINGS = {
    "small": {
        "archive_sha256": "452fec98990aaf425a2ae2a494a16d9c0d9c111d02cdda60e9e31529a069ac5a",
        "evidence_sha256": "a46f5138717199fa966513a1611b1314a3a32a7cb0643728da460834d99643cf",
        "width": 3072,
    },
    "large": {"archive_sha256": None, "evidence_sha256": None, "width": 8192},
}
FOLDS = 5
C_VALUE = 0.01
SEED = 42


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def source_folds(roles: np.ndarray, labels: np.ndarray, sources: np.ndarray) -> dict[str, int]:
    """Greedily balance whole development sources within each class over fixed folds."""
    counts = Counter(
        (int(label), str(source))
        for role, label, source in zip(roles, labels, sources, strict=True)
        if str(role) == "development"
    )
    assignments: dict[str, int] = {}
    for label in (0, 1):
        totals = [0] * FOLDS
        candidates = [(source, count) for (found, source), count in counts.items() if found == label]
        candidates.sort(
            key=lambda item: (
                -item[1],
                hashlib.sha256(f"E42_FOLD|{label}|{item[0]}".encode()).digest(),
            )
        )
        for source, count in candidates:
            fold = min(range(FOLDS), key=lambda index: (totals[index], index))
            assignments[source] = fold
            totals[fold] += count
        if any(total == 0 for total in totals):
            raise ValueError(f"E42 label {label} cannot populate every fold")
    return assignments


def source_balanced_weights(labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    sources = sources.astype(str)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("E42 fit requires both labels")
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in (0, 1):
        label_mask = labels == label
        label_sources = sorted(set(sources[label_mask]))
        for source in label_sources:
            mask = label_mask & (sources == source)
            weights[mask] = 1.0 / (2 * len(label_sources) * int(mask.sum()))
    if not np.all(weights > 0) or not np.isfinite(weights).all():
        raise ValueError("invalid E42 source-balanced weights")
    return weights * (len(weights) / weights.sum())


def _head() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_VALUE,
            class_weight=None,
            max_iter=1_000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )


def _fit(x: np.ndarray, y: np.ndarray, sources: np.ndarray) -> Any:
    weights = source_balanced_weights(y, sources)
    head = _head()
    head.fit(
        x,
        y,
        standardscaler__sample_weight=weights,
        logisticregression__sample_weight=weights,
    )
    return head


def source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), str(row["source"]))].append(float(row["score"]) >= threshold)
    real = {source: float(np.mean(values)) for (label, source), values in sorted(grouped.items()) if label == 0}
    ai = {source: float(np.mean(values)) for (label, source), values in sorted(grouped.items()) if label == 1}
    if not real or not ai:
        raise ValueError("E42 rates require both classes")
    return {
        "threshold": float(threshold),
        "real_fp_by_source": real,
        "real_macro_fp": float(np.mean(list(real.values()))),
        "real_worst_source_fp": max(real.values()),
        "ai_recall_by_source": ai,
        "ai_macro_recall": float(np.mean(list(ai.values()))),
        "ai_worst_source_recall": min(ai.values()),
    }


def select_threshold(clean_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        {float(row["score"]) for row in clean_rows}
        | {float(np.nextafter(float(row["score"]), np.inf)) for row in clean_rows}
    )
    for threshold in candidates:
        rates = source_rates(clean_rows, threshold)
        if rates["real_macro_fp"] <= 0.10 + 1e-12 and rates["real_worst_source_fp"] <= 0.20 + 1e-12:
            return rates
    raise ValueError("no E42 OOF threshold satisfies authentic-source budgets")


def development_gate(
    clean_metrics: Mapping[str, Any], clean_rates: Mapping[str, Any], robust_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "clean_auc_gte_0_90": float(clean_metrics["roc_auc"]) >= 0.90,
        "clean_tpr_at_fpr10_gte_0_80": float(clean_metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
        "clean_eer_lte_0_15": float(clean_metrics["eer"]) <= 0.15,
        "clean_balanced_accuracy_gte_0_85": float(clean_metrics["balanced_accuracy"]) >= 0.85,
        "real_macro_fp_lte_0_10": float(clean_rates["real_macro_fp"]) <= 0.10 + 1e-12,
        "real_worst_source_fp_lte_0_20": float(clean_rates["real_worst_source_fp"]) <= 0.20 + 1e-12,
        "ai_macro_recall_gte_0_80": float(clean_rates["ai_macro_recall"]) >= 0.80,
        "ai_worst_source_recall_gte_0_60": float(clean_rates["ai_worst_source_recall"]) >= 0.60,
        "clean_coverage_eq_1": float(clean_metrics["coverage"]) == 1.0,
        "robust_auc_gte_0_85": float(robust_metrics["roc_auc"]) >= 0.85,
        "robust_balanced_accuracy_gte_0_80": float(robust_metrics["balanced_accuracy"]) >= 0.80,
        "robust_coverage_eq_1": float(robust_metrics["coverage"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _load(backbone: str) -> dict[str, np.ndarray]:
    binding = FEATURE_BINDINGS[backbone]
    if not binding["archive_sha256"] or not binding["evidence_sha256"]:
        raise ValueError(f"E42 {backbone} feature binding has not been preregistered")
    archive = FEATURE_ROOT / f"{backbone}.npz"
    evidence = REPO_ROOT / "evidence" / f"e42_features_{backbone}.json"
    if _sha256_file(archive) != binding["archive_sha256"] or _sha256_file(evidence) != binding["evidence_sha256"]:
        raise ValueError(f"E42 {backbone} features changed")
    if _sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("E42 fixed contract changed")
    with np.load(archive, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    if values["features"].shape != (20_506, int(binding["width"])):
        raise ValueError(f"E42 {backbone} feature shape changed")
    return values


def _fit_view_mask(values: Mapping[str, np.ndarray]) -> np.ndarray:
    output = np.zeros(len(values["roles"]), dtype=bool)
    for index, (role, parent, condition) in enumerate(
        zip(values["roles"], values["parent_ids"], values["conditions"], strict=True)
    ):
        output[index] = str(role) == "train" or str(condition) in {
            "clean", assigned_transport(str(parent))
        }
    return output


def oof_scores(values: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    assignments = source_folds(values["roles"], values["labels"], values["sources"])
    fit_views = _fit_view_mask(values)
    output = []
    for fold in range(FOLDS):
        held_sources = {source for source, assigned in assignments.items() if assigned == fold}
        held = (values["roles"].astype(str) == "development") & np.isin(values["sources"].astype(str), list(held_sources))
        fit = fit_views & ~held
        head = _fit(values["features"][fit], values["labels"][fit], values["sources"][fit])
        scores = head.predict_proba(values["features"][held])[:, 1]
        indices = np.flatnonzero(held)
        for index, score in zip(indices, scores, strict=True):
            output.append({
                "record_id": str(values["record_ids"][index]),
                "parent_id": str(values["parent_ids"][index]),
                "label": int(values["labels"][index]),
                "source": str(values["sources"][index]),
                "condition": str(values["conditions"][index]),
                "fold": fold,
                "score": float(score),
                "status": "ok",
            })
        print(f"E42 fold {fold + 1}/{FOLDS}: fit={int(fit.sum())}, held={int(held.sum())}", flush=True)
    output.sort(key=lambda row: row["record_id"])
    if len(output) != 11_230 or len({row["record_id"] for row in output}) != len(output):
        raise ValueError("E42 OOF coverage changed")
    return output, assignments


def evaluate(backbone: str) -> dict[str, Any]:
    scores_path = E42_ROOT / f"oof_{backbone}.jsonl"
    report_path = E42_ROOT / f"development_{backbone}.json"
    evidence_path = REPO_ROOT / "evidence" / f"e42_development_{backbone}.json"
    candidate_path = E42_ROOT / f"e42_{backbone}.joblib"
    if any(path.exists() for path in (scores_path, report_path, evidence_path, candidate_path)):
        raise FileExistsError(f"E42 {backbone} training output already exists")
    values = _load(backbone)
    rows, assignments = oof_scores(values)
    clean = [row for row in rows if row["condition"] == "clean"]
    robust = [row for row in rows if row["condition"] != "clean"]
    threshold_rates = select_threshold(clean)
    threshold = float(threshold_rates["threshold"])
    clean_metrics = evaluate_binary_scores(clean, threshold=threshold)
    robust_metrics = evaluate_binary_scores(robust, threshold=threshold)
    robust_rates = source_rates(robust, threshold)
    by_condition = {
        condition: {
            "metrics": evaluate_binary_scores(
                [row for row in robust if row["condition"] == condition], threshold=threshold
            ),
            "rates": source_rates(
                [row for row in robust if row["condition"] == condition], threshold
            ),
        }
        for condition in sorted({str(row["condition"]) for row in robust})
    }
    result_gate = development_gate(clean_metrics, threshold_rates, robust_metrics)
    score_bytes, score_sha256 = _write_jsonl(scores_path, rows)
    report = {
        "schema_version": 1,
        "experiment": "E42/texture-intermediate-source-heldout-recovery",
        "state": "development_passed" if result_gate["passed"] else "development_failed",
        "backbone": backbone,
        "feature_archive_sha256": FEATURE_BINDINGS[backbone]["archive_sha256"],
        "fold_assignments": dict(sorted(assignments.items())),
        "threshold": threshold,
        "clean": {"metrics": clean_metrics, "rates": threshold_rates},
        "robust_combined": {"metrics": robust_metrics, "rates": robust_rates},
        "robust_by_condition": by_condition,
        "gate": result_gate,
        "score_stream": {"bytes": score_bytes, "sha256": score_sha256},
        "boundary": "Consumed source-held-out DEVELOPMENT only; B-Free and RR test remain excluded.",
    }
    _write_atomic(report_path, report)
    _write_atomic(evidence_path, report)
    if result_gate["passed"]:
        fit = _fit_view_mask(values)
        head = _fit(values["features"][fit], values["labels"][fit], values["sources"][fit])
        artifact = {
            "schema_version": 1,
            "model_name": f"E42 DINOv2-{backbone[0].upper()} texture-intermediate candidate",
            "status": "research_candidate_awaiting_rr_external",
            "positive_label": "ai",
            "model_id": MODEL_IDS[backbone],
            "backbone": backbone,
            "block_indices": BLOCKS[backbone],
            "threshold": threshold,
            "head": head,
            "feature_contract": {
                "crops": "global plus two deterministic highest-texture crops",
                "aggregation": "per-block crop mean+std",
                "max_long_side": 2048,
            },
            "development_report_sha256": _sha256_file(report_path),
            "feature_archive_sha256": FEATURE_BINDINGS[backbone]["archive_sha256"],
        }
        temporary = candidate_path.with_suffix(candidate_path.suffix + ".part")
        joblib.dump(artifact, temporary)
        temporary.replace(candidate_path)
        report["candidate"] = {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha256_file(candidate_path),
        }
        _write_atomic(report_path, report)
        _write_atomic(evidence_path, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=("small", "large"))
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.backbone), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
