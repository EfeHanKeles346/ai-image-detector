"""Fit the E46 calibration candidate without reading DEVELOPMENT or final scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.e44_fusion import CANDIDATE as FUSION, _feature
from experiments.e46_development import (
    GENERALIST_SCORES,
    GENERALIST_SHA256,
    MANIFEST,
    MANIFEST_SHA256,
    SPECIALIST_SCORES,
)
from pixelproof.dda_candidate import CHECKPOINT_SHA256
from pixelproof.evaluation_protocol import threshold_at_fpr
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e46"
SPLIT_NAMESPACE = "E46_CAL_INTERNAL_V1"
CONTRACT = ROOT / "calibration_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_calibration_contract.json"
CANDIDATE = ROOT / "e46_candidate.joblib"
CANDIDATE_REPORT = ROOT / "candidate_calibration_report.json"
CANDIDATE_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_candidate_calibration.json"
GENERALIST_SCORE_SHA256 = "8be0aefdd995ab16e409c070c897ee91ac102dd61ad281dd8da7114d1c0ce88d"
SPECIALIST_SCORE_SHA256 = "a7fbd7e2075d9687a042995a5b30412460564bf6f931d8974b440527b2257eda"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
EPSILON = 1e-5
RIDGE = 1.0


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def internal_roles(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Make a fixed 60/40 QUALITY_FIT/OPERATING_CAL split within every source."""
    output: dict[str, str] = {}
    sources = sorted({str(row["typ"]) for row in rows if row["role"] == "CAL"})
    for source in sources:
        ids = [str(row["record_id"]) for row in rows if row["role"] == "CAL" and row["typ"] == source]
        ids.sort(key=lambda value: (hashlib.sha256(
            f"{SPLIT_NAMESPACE}|{source}|{value}".encode()
        ).digest(), value))
        if len(ids) < 10:
            raise ValueError(f"too few E46 CAL identities for {source}")
        cut = 3 * len(ids) // 5
        output.update({value: "QUALITY_FIT" if index < cut else "OPERATING_CAL"
                       for index, value in enumerate(ids)})
    return output


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E46 calibration contract already exists")
    manifest_raw = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("E46 manifest changed before calibration binding")
    manifest = json.loads(manifest_raw)
    roles = internal_roles(manifest["rows"])
    counts = Counter(roles.values())
    by_source_role = Counter(
        (row["typ"], roles[row["record_id"]])
        for row in manifest["rows"] if row["record_id"] in roles
    )
    identities = {
        "manifest_sha256": MANIFEST_SHA256,
        "generalist_score_sha256": _digest(GENERALIST_SCORES),
        "specialist_score_sha256": _digest(SPECIALIST_SCORES),
        "generalist_model_sha256": GENERALIST_SHA256,
        "specialist_model_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": _digest(FUSION),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "generalist_score_sha256": GENERALIST_SCORE_SHA256,
        "specialist_score_sha256": SPECIALIST_SCORE_SHA256,
        "generalist_model_sha256": GENERALIST_SHA256,
        "specialist_model_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": FUSION_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E46 calibration identity changed: {identities}")
    payload = {
        "schema_version": 1,
        "state": "e46_calibration_method_frozen_before_score_read",
        "identities": identities,
        "internal_split": {
            "namespace": SPLIT_NAMESPACE,
            "rule": "within-source SHA-256 rank, first floor(60%) QUALITY_FIT, rest OPERATING_CAL",
            "counts": dict(sorted(counts.items())),
            "by_source_role": {f"{source}:{role}": count
                               for (source, role), count in sorted(by_source_role.items())},
            "roles": roles,
        },
        "methods": {
            "dda_global": "official DDA probability + OPERATING_CAL REAL-10% threshold",
            "fusion_global": "frozen E44 fusion probability + OPERATING_CAL REAL-10% threshold",
            "fusion_quality_gaussian": (
                "QuAD-inspired class-conditional Gaussian of frozen-fusion logit given three "
                "standardized quality proxies; ridge=1.0; OPERATING_CAL REAL-10% threshold"
            ),
        },
        "eligibility": {"operating_cal_ai_recall_min": 0.75,
                        "operating_cal_worst_generator_recall_min": 0.50},
        "selection": {
            "global": "eligible DDA/fusion ordered by worst-generator recall, AI recall, AUC, then DDA simplicity",
            "quality_replacement": (
                "eligible quality method must be non-inferior to best global in REAL FP, AI recall, "
                "worst-generator recall and AUC, and improve at least one recall/AUC metric by >=0.01"
            ),
            "fallback": "if none eligible, choose same ordering but mark calibration failure",
        },
        "selective_policy": (
            "AI cut is binary cut; choose CAL-only REAL cut maximizing automatic coverage with "
            "covered accuracy >=0.95 and uncertainty <=0.20; otherwise REAL cut equals AI cut"
        ),
        "forbidden": ["DEVELOPMENT read", "TrueFake read", "E45 use", "method addition",
                      "threshold change after OPERATING_CAL", "backbone refit"],
        "score_rows_read": 0,
        "development_scores_read": 0,
        "final_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {key: value for key, value in payload.items() if key != "internal_split"}
    evidence["internal_split"] = {key: value for key, value in payload["internal_split"].items()
                                  if key != "roles"}
    evidence.update({"contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()})
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(values: Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def fit_quality_gaussian(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quality = np.asarray([row["quality"] for row in rows], dtype=np.float64)
    center = quality.mean(axis=0)
    scale = quality.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.column_stack(((quality - center) / scale, np.ones(len(rows))))
    scores = _logit([float(row["fusion_score"]) for row in rows])
    models = {}
    penalty = np.eye(x.shape[1]) * RIDGE
    penalty[-1, -1] = 0.0
    for label in (0, 1):
        mask = np.asarray([int(row["label"]) == label for row in rows])
        subset_x, subset_y = x[mask], scores[mask]
        mean_coef = np.linalg.solve(subset_x.T @ subset_x + penalty, subset_x.T @ subset_y)
        residual = subset_y - subset_x @ mean_coef
        log_residual = np.log(residual**2 + 1e-3)
        variance_coef = np.linalg.solve(
            subset_x.T @ subset_x + penalty, subset_x.T @ log_residual
        )
        models[str(label)] = {"mean": mean_coef.tolist(), "log_variance": variance_coef.tolist()}
    return {"center": center.tolist(), "scale": scale.tolist(), "classes": models, "ridge": RIDGE}


def apply_quality_gaussian(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    quality = np.asarray([row["quality"] for row in rows], dtype=np.float64)
    x = np.column_stack(((quality - np.asarray(model["center"])) / np.asarray(model["scale"]),
                         np.ones(len(rows))))
    observed = _logit([float(row["fusion_score"]) for row in rows])
    terms = []
    for label in (0, 1):
        params = model["classes"][str(label)]
        mean = x @ np.asarray(params["mean"])
        log_variance = np.clip(x @ np.asarray(params["log_variance"]), -10.0, 10.0)
        terms.append((observed - mean) ** 2 / np.exp(log_variance) + log_variance)
    llr = 0.5 * (terms[0] - terms[1])
    return _sigmoid(llr)


def score_metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    predicted = s >= threshold
    real_fp = float(predicted[y == 0].mean())
    ai_recall = float(predicted[y == 1].mean())
    by_generator = {
        source: float(predicted[np.asarray([row["source"] == source for row in rows])].mean())
        for source in sorted({row["source"] for row in rows if int(row["label"]) == 1})
    }
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "balanced_accuracy": 0.5 * ((1.0 - real_fp) + ai_recall),
        "real_false_ai": real_fp,
        "ai_recall": ai_recall,
        "ai_recall_by_generator": by_generator,
        "worst_generator_recall": min(by_generator.values()),
        "threshold": float(threshold),
        "rows": len(rows),
    }


def _operating_threshold(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> float:
    real = [float(score) for row, score in zip(rows, scores, strict=True) if int(row["label"]) == 0]
    return float(threshold_at_fpr(real, 0.10))


def _selective(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, ai_cut: float) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows])
    candidates = sorted(set(float(value) for value in scores if value <= ai_cut))
    feasible = []
    for real_cut in candidates:
        automatic = (scores < real_cut) | (scores >= ai_cut)
        coverage = float(automatic.mean())
        uncertainty = 1.0 - coverage
        if not automatic.any():
            continue
        decisions = scores[automatic] >= ai_cut
        accuracy = float((decisions == labels[automatic]).mean())
        if accuracy >= 0.95 and uncertainty <= 0.20:
            feasible.append((coverage, accuracy, real_cut))
    if feasible:
        coverage, accuracy, real_cut = max(feasible, key=lambda item: (item[0], item[1], item[2]))
        return {"real_if_score_lt": float(real_cut), "ai_if_score_gte": float(ai_cut),
                "cal_coverage": coverage, "cal_covered_accuracy": accuracy,
                "cal_uncertainty": 1.0 - coverage, "cal_constraint_passed": True}
    metrics = score_metrics(rows, scores, ai_cut)
    return {"real_if_score_lt": float(ai_cut), "ai_if_score_gte": float(ai_cut),
            "cal_coverage": 1.0, "cal_covered_accuracy": metrics["balanced_accuracy"],
            "cal_uncertainty": 0.0, "cal_constraint_passed": False}


def _joined_cal(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = {row["record_id"]: row for row in json.loads(MANIFEST.read_text())["rows"]}
    generalist = _jsonl(GENERALIST_SCORES)
    specialist = {row["record_id"]: row for row in _jsonl(SPECIALIST_SCORES)}
    fusion = joblib.load(FUSION)
    roles = contract["internal_split"]["roles"]
    output = []
    for row in generalist:
        if row["record_id"] not in roles:
            continue
        peer = specialist[row["record_id"]]
        source = manifest[row["record_id"]]
        features = np.asarray([_feature(float(row["score"]), float(peer["score"]))])
        fused = float(fusion["head"].predict_proba(features)[0, 1])
        output.append({
            "record_id": row["record_id"], "internal_role": roles[row["record_id"]],
            "label": int(row["label"]), "source": source["typ"], "quality": row["quality"],
            "generalist_score": float(row["score"]), "specialist_score": float(peer["score"]),
            "fusion_score": fused,
        })
    if len(output) != len(roles) or any(row["internal_role"] == "DEVELOPMENT" for row in output):
        raise ValueError("E46 CAL join changed or DEVELOPMENT leaked")
    return output


def choose_method(results: Mapping[str, Mapping[str, Any]]) -> tuple[str, bool]:
    def eligible(item: Mapping[str, Any]) -> bool:
        return item["ai_recall"] >= 0.75 and item["worst_generator_recall"] >= 0.50

    globals_ = [name for name in ("dda_global", "fusion_global") if eligible(results[name])]
    calibration_passed = bool(globals_ or eligible(results["fusion_quality_gaussian"]))
    pool = globals_ or ["dda_global", "fusion_global"]
    best_global = max(pool, key=lambda name: (
        results[name]["worst_generator_recall"], results[name]["ai_recall"],
        results[name]["roc_auc"], name == "dda_global",
    ))
    quality = results["fusion_quality_gaussian"]
    baseline = results[best_global]
    noninferior = (
        eligible(quality)
        and quality["real_false_ai"] <= baseline["real_false_ai"] + 1e-12
        and quality["ai_recall"] >= baseline["ai_recall"] - 1e-12
        and quality["worst_generator_recall"] >= baseline["worst_generator_recall"] - 1e-12
        and quality["roc_auc"] >= baseline["roc_auc"] - 1e-12
    )
    material = max(
        quality["ai_recall"] - baseline["ai_recall"],
        quality["worst_generator_recall"] - baseline["worst_generator_recall"],
        quality["roc_auc"] - baseline["roc_auc"],
    ) >= 0.01
    return ("fusion_quality_gaussian" if noninferior and material else best_global), calibration_passed


def fit() -> dict[str, Any]:
    if any(path.exists() for path in (CANDIDATE, CANDIDATE_REPORT, CANDIDATE_EVIDENCE)):
        raise FileExistsError("E46 calibration candidate already exists")
    contract_raw = CONTRACT.read_bytes()
    contract = json.loads(contract_raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if hashlib.sha256(contract_raw).hexdigest() != evidence["contract_sha256"]:
        raise ValueError("E46 calibration contract changed")
    if _digest(GENERALIST_SCORES) != GENERALIST_SCORE_SHA256 or _digest(SPECIALIST_SCORES) != SPECIALIST_SCORE_SHA256:
        raise ValueError("E46 score stream changed")
    rows = _joined_cal(contract)
    fit_rows = [row for row in rows if row["internal_role"] == "QUALITY_FIT"]
    operating = [row for row in rows if row["internal_role"] == "OPERATING_CAL"]
    quality_model = fit_quality_gaussian(fit_rows)
    method_scores = {
        "dda_global": np.asarray([row["specialist_score"] for row in operating]),
        "fusion_global": np.asarray([row["fusion_score"] for row in operating]),
        "fusion_quality_gaussian": apply_quality_gaussian(quality_model, operating),
    }
    results = {}
    for name, values in method_scores.items():
        threshold = _operating_threshold(operating, values)
        results[name] = score_metrics(operating, values, threshold)
    selected, calibration_passed = choose_method(results)
    selected_scores = method_scores[selected]
    threshold = float(results[selected]["threshold"])
    selective = _selective(operating, selected_scores, threshold)
    artifact = {
        "schema_version": 1,
        "model_name": "E46 cross-platform calibrated detector",
        "positive_label": "ai",
        "selected_method": selected,
        "threshold": threshold,
        "selective_policy": selective,
        "quality_model": quality_model if selected == "fusion_quality_gaussian" else None,
        "identities": contract["identities"],
        "selection_contract_sha256": evidence["contract_sha256"],
    }
    temporary = CANDIDATE.with_suffix(CANDIDATE.suffix + ".part")
    joblib.dump(artifact, temporary)
    temporary.replace(CANDIDATE)
    report = {
        "schema_version": 1,
        "state": "e46_candidate_frozen_before_development",
        "quality_fit_rows": len(fit_rows),
        "operating_cal_rows": len(operating),
        "method_results": results,
        "selected_method": selected,
        "calibration_eligibility_passed": calibration_passed,
        "selected_threshold": threshold,
        "selective_policy": selective,
        "candidate_bytes": CANDIDATE.stat().st_size,
        "candidate_sha256": _digest(CANDIDATE),
        "development_rows_read": 0,
        "final_scores_created": 0,
        "boundary": "QUALITY_FIT and OPERATING_CAL only; DEVELOPMENT and TrueFake remain unread.",
    }
    raw = _write(CANDIDATE_REPORT, report)
    _write(CANDIDATE_EVIDENCE, {**report, "report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "fit"))
    args = parser.parse_args(argv)
    result = bind() if args.command == "bind" else fit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
