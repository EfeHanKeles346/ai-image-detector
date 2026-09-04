"""Freeze and fit the E48 monotone non-veto decision layer on FIT/CAL only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from experiments.e43_train import _digest
from experiments.e44_fusion import CANDIDATE as E44_FUSION, _feature as e44_feature
from experiments.e48_score import ARM_PATHS, MANIFEST_SHA256
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e48"
CONTRACT = ROOT / "decision_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e48_decision_contract.json"
CAL_REPORT = ROOT / "calibration_report.json"
CANDIDATE = ROOT / "e48_monotone_candidate.joblib"
CANDIDATE_EVIDENCE = ML_ROOT.parent / "evidence" / "e48_calibration_result.json"
STREAM_SHA256 = {
    "generalist": "f2a1be3bf428952a2f8ae166d6bc0f47ba13559607aebb109e4c5adfdd137cf7",
    "dda": "7ebc7831fe59bfcc91fe3bef6d2a0b3ed6504dc038785b63e9c0132a3cb4a23b",
    "univfd": "e768d591839644c4570ae52bd195bc0b03b14bd22695e7b3bcce99769e81d635",
    "unina": "e3d47527fffe98a7bec44cef8255339fe86db1cac9e71ad214ed51b6704b3c01",
}
E44_FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
CANDIDATES = ("e46", "e46_univfd", "e46_unina", "e46_univfd_unina")
LICENSE_MARGIN = 0.02
GATES = {
    "coverage": 1.0, "roc_auc_min": 0.90, "balanced_accuracy_min": 0.85,
    "pooled_real_false_ai_max": 0.10, "worst_camera_false_ai_max": 0.20,
    "pooled_ai_recall_min": 0.80, "worst_ai_recall_min": 0.60,
}


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E48 decision contract already exists")
    found = {name: _digest(path) for name, path in ARM_PATHS.items()}
    if found != STREAM_SHA256 or _digest(E44_FUSION) != E44_FUSION_SHA256:
        raise ValueError(f"E48 decision inputs changed: {found}")
    payload = {
        "schema_version": 1, "state": "e48_decision_contract_frozen_before_score_interpretation",
        "manifest_sha256": MANIFEST_SHA256, "score_stream_sha256": STREAM_SHA256,
        "e44_fusion_sha256": E44_FUSION_SHA256,
        "fit": {
            "rows_allowed": "FIT REAL only", "rows": 300,
            "per_arm_map": "sorted authentic scores; evidence=(left insertion rank+1)/(n+1)",
            "direction": "higher detector score is more AI-like",
        },
        "candidates": {
            "e46": ["e46"], "e46_univfd": ["e46", "univfd"],
            "e46_unina": ["e46", "unina"],
            "e46_univfd_unina": ["e46", "univfd", "unina"],
        },
        "fusion": "maximum empirical authentic-percentile evidence; monotone in every arm",
        "threshold": "lowest CAL threshold with pooled REAL FP<=10% and worst camera-pipeline FP<=20%",
        "eligibility_and_development_gates": GATES,
        "selection": {
            "ranking": ["worst AI-source recall", "pooled AI recall", "ROC AUC", "balanced accuracy"],
            "tie_break": list(CANDIDATES),
            "licence_rule": "prefer eligible e46_univfd if within 0.02 of an UNINA winner on both AI recall measures",
            "licence_margin": LICENSE_MARGIN,
        },
        "forbidden": ["FIT AI use", "CAL fit", "DEVELOPMENT access", "signed/veto weight",
                      "post-CAL rule change", "backbone update"],
        "calibration_metrics_opened": 0, "development_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> dict[str, Any]:
    raw = CONTRACT.read_bytes(); payload = json.loads(raw); evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
            or payload.get("calibration_metrics_opened") != 0
            or payload.get("development_scores_created") != 0):
        raise ValueError("E48 decision contract changed")
    if {name: _digest(path) for name, path in ARM_PATHS.items()} != STREAM_SHA256:
        raise ValueError("E48 score stream changed")
    return payload


def _load_rows(role: str) -> list[dict[str, Any]]:
    streams = {name: [json.loads(line) for line in path.read_text().splitlines() if line]
               for name, path in ARM_PATHS.items()}
    if any(len(rows) != 1_200 for rows in streams.values()):
        raise ValueError("E48 FIT+CAL score count changed")
    lookup = {name: {row["record_id"]: row for row in rows} for name, rows in streams.items()}
    output = []
    for base in streams["generalist"]:
        peers = {name: values.get(base["record_id"]) for name, values in lookup.items()}
        if any(peer is None for peer in peers.values()):
            raise ValueError(f"E48 join missing {base['record_id']}")
        identity = (base["role"], int(base["label"]), base["source"], base.get("camera_pipeline"))
        if any((peer["role"], int(peer["label"]), peer["source"], peer.get("camera_pipeline")) != identity
               for peer in peers.values()):
            raise ValueError(f"E48 score identity drift {base['record_id']}")
        if base["role"] != role:
            continue
        output.append({
            "record_id": base["record_id"], "role": role, "label": int(base["label"]),
            "source": base["source"], "camera_pipeline": base.get("camera_pipeline"),
            **{f"{name}_score": float(peer["score"]) for name, peer in peers.items()},
        })
    if len(output) != 600:
        raise ValueError(f"E48 {role} score count changed")
    return output


def _e46_scores(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    fusion = joblib.load(E44_FUSION)
    x = np.asarray([e44_feature(row["generalist_score"], row["dda_score"]) for row in rows])
    return fusion["head"].predict_proba(x)[:, 1]


def empirical_evidence(sorted_real: Sequence[float], value: float) -> float:
    real = np.asarray(sorted_real, dtype=np.float64)
    if len(real) < 2 or not np.all(real[:-1] <= real[1:]):
        raise ValueError("E48 empirical map must be sorted and nontrivial")
    return float((np.searchsorted(real, float(value), side="left") + 1) / (len(real) + 1))


def _score(name: str, row: Mapping[str, Any], maps: Mapping[str, Sequence[float]]) -> float:
    arms = ["e46"]
    if "univfd" in name: arms.append("univfd")
    if "unina" in name: arms.append("unina")
    return max(empirical_evidence(maps[arm], float(row[f"{arm}_score"])) for arm in arms)


def _rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    real_rows = [row for row in rows if int(row["label"]) == 0]
    ai_rows = [row for row in rows if int(row["label"]) == 1]
    by_camera: dict[str, list[bool]] = defaultdict(list)
    by_ai: dict[str, list[bool]] = defaultdict(list)
    for row in real_rows:
        by_camera[str(row.get("camera_pipeline") or row["source"])].append(float(row["score"]) >= threshold)
    for row in ai_rows:
        by_ai[str(row["source"])].append(float(row["score"]) >= threshold)
    real_rates = {key: float(np.mean(value)) for key, value in sorted(by_camera.items())}
    ai_rates = {key: float(np.mean(value)) for key, value in sorted(by_ai.items())}
    return {
        "real_false_ai_by_camera": real_rates, "ai_recall_by_source": ai_rates,
        "pooled_real_false_ai": float(np.mean([float(row["score"]) >= threshold for row in real_rows])),
        "worst_camera_false_ai": max(real_rates.values()),
        "pooled_ai_recall": float(np.mean([float(row["score"]) >= threshold for row in ai_rows])),
        "worst_ai_recall": min(ai_rates.values()),
    }


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    scores = [float(row["score"]) for row in rows]
    candidates = sorted(set(scores) | {float(np.nextafter(value, np.inf)) for value in scores})
    for threshold in candidates:
        rates = _rates(rows, threshold)
        if (rates["pooled_real_false_ai"] <= 0.10 + 1e-12
                and rates["worst_camera_false_ai"] <= 0.20 + 1e-12):
            return threshold
    raise RuntimeError("no E48 CAL threshold satisfies authentic safety")


def _gate(metrics: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "coverage_eq_1": float(metrics["coverage"]) == 1.0,
        "roc_auc_gte_0_90": float(metrics["roc_auc"]) >= 0.90,
        "balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= 0.85,
        "pooled_real_false_ai_lte_0_10": float(rates["pooled_real_false_ai"]) <= 0.10,
        "worst_camera_false_ai_lte_0_20": float(rates["worst_camera_false_ai"]) <= 0.20,
        "pooled_ai_recall_gte_0_80": float(rates["pooled_ai_recall"]) >= 0.80,
        "worst_ai_recall_gte_0_60": float(rates["worst_ai_recall"]) >= 0.60,
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_candidate(summaries: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, str]:
    eligible = [name for name in CANDIDATES if summaries[name]["gate"]["passed"]]
    if not eligible:
        return None, "no E48 CAL candidate passes the frozen gate"
    def rank(name: str) -> tuple[float, float, float, float, int]:
        value = summaries[name]
        return (value["rates"]["worst_ai_recall"], value["rates"]["pooled_ai_recall"],
                value["metrics"]["roc_auc"], value["metrics"]["balanced_accuracy"], -CANDIDATES.index(name))
    raw = max(eligible, key=rank); mit = "e46_univfd"
    if "unina" in raw and mit in eligible:
        a, b = summaries[raw]["rates"], summaries[mit]["rates"]
        if (b["worst_ai_recall"] >= a["worst_ai_recall"] - LICENSE_MARGIN
                and b["pooled_ai_recall"] >= a["pooled_ai_recall"] - LICENSE_MARGIN):
            return mit, "MIT preference within 0.02 on worst and pooled AI recall"
    return raw, "highest frozen CAL ranking"


def fit_calibration() -> dict[str, Any]:
    if any(path.exists() for path in (CAL_REPORT, CANDIDATE, CANDIDATE_EVIDENCE)):
        raise FileExistsError("E48 CAL output already exists")
    contract = _validate_contract(); fit = _load_rows("FIT"); cal = _load_rows("CAL")
    fit_e46 = _e46_scores(fit); cal_e46 = _e46_scores(cal)
    fit_rows = [{**row, "e46_score": float(score)} for row, score in zip(fit, fit_e46, strict=True)]
    cal_rows = [{**row, "e46_score": float(score)} for row, score in zip(cal, cal_e46, strict=True)]
    maps = {arm: np.sort(np.asarray([row[f"{arm}_score"] for row in fit_rows if row["label"] == 0]))
            for arm in ("e46", "univfd", "unina")}
    if any(len(values) != 300 for values in maps.values()):
        raise ValueError("E48 FIT REAL map count changed")
    summaries = {}
    for name in CANDIDATES:
        scored = [{"record_id": row["record_id"], "role": "CAL", "label": row["label"],
                   "source": row["source"], "camera_pipeline": row["camera_pipeline"],
                   "status": "ok", "score": _score(name, row, maps)} for row in cal_rows]
        threshold = select_threshold(scored); metrics = evaluate_binary_scores(scored, threshold=threshold)
        rates = _rates(scored, threshold)
        summaries[name] = {"threshold": threshold, "metrics": metrics, "rates": rates,
                           "gate": _gate(metrics, rates)}
    selected, reason = choose_candidate(summaries)
    if selected is None:
        report = {
            "schema_version": 1, "state": "e48_calibration_failed_no_candidate",
            "fit_real_rows": 300, "fit_ai_rows_used": 0, "cal_rows": 600,
            "candidate_summaries": summaries, "selected_method": None,
            "selection_reason": reason, "selected_threshold": None,
            "candidate_bytes": 0, "candidate_sha256": None,
            "contract_sha256": _digest(CONTRACT), "development_scores_created": 0,
        }
        raw = _write(CAL_REPORT, report)
        _write(CANDIDATE_EVIDENCE, {**report, "cal_report_sha256": hashlib.sha256(raw).hexdigest()})
        return report
    artifact = {
        "schema_version": 1, "state": "e48_candidate_frozen_before_development",
        "selected_method": selected, "threshold": summaries[selected]["threshold"],
        "empirical_real_maps": maps, "contract_sha256": _digest(CONTRACT),
        "input_stream_sha256": STREAM_SHA256, "map_rule": contract["fit"]["per_arm_map"],
        "fusion": contract["fusion"], "selection_reason": reason,
    }
    temporary = CANDIDATE.with_suffix(".joblib.part"); joblib.dump(artifact, temporary); temporary.replace(CANDIDATE)
    report = {
        "schema_version": 1, "state": "e48_candidate_frozen_before_development",
        "fit_real_rows": 300, "fit_ai_rows_used": 0, "cal_rows": 600,
        "candidate_summaries": summaries, "selected_method": selected, "selection_reason": reason,
        "selected_threshold": summaries[selected]["threshold"], "candidate_bytes": CANDIDATE.stat().st_size,
        "candidate_sha256": _digest(CANDIDATE), "contract_sha256": _digest(CONTRACT),
        "development_scores_created": 0,
    }
    raw = _write(CAL_REPORT, report)
    _write(CANDIDATE_EVIDENCE, {**report, "cal_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "fit-cal")); args = parser.parse_args(argv)
    result = bind() if args.command == "bind" else fit_calibration()
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
