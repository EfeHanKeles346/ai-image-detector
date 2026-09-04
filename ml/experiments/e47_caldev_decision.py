"""Freeze, fit on CAL, and evaluate the E47 GAN-aware decision layer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e43_train import _digest
from experiments.e44_fusion import CANDIDATE as E44_FUSION, _feature as e44_feature
from experiments.e47_caldev_score import ARM_PATHS, MANIFEST_SHA256
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
CONTRACT = ROOT / "caldev_decision_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_caldev_decision_contract.json"
CAL_REPORT = ROOT / "calibration_report.json"
CANDIDATE = ROOT / "e47_gan_aware_candidate.joblib"
CANDIDATE_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_calibration_result.json"
DEVELOPMENT_SCORES = ROOT / "development_candidate_scores.jsonl"
DEVELOPMENT_REPORT = ROOT / "development_report.json"
DEVELOPMENT_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_development_result.json"

STREAM_SHA256 = {
    "generalist": "073110f4b9cd3b4056db185798bcf6808e39ef5c467f44c4a348b9f0bcf30c03",
    "dda": "8001c60b793afed8592ecd41547050eb975b95e58a90a7a067cdc5ab6f9d75f5",
    "univfd": "67b7b94c41a3ed9a732d7c82b412dd25c4f6879226517b82351ed50fc6ee2829",
    "unina": "7efb36c0dcc6bba15528d3fb58973c0d13eb09873481d05b487b6a2c6ed5e16d",
}
E44_FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
EPSILON = 1e-5
C_VALUE = 0.1
SEED = 47
LICENSE_MARGIN = 0.02
CANDIDATES = ("e46", "e46_univfd", "e46_unina", "e46_univfd_unina")
GATES = {
    "coverage": 1.0,
    "roc_auc_min": 0.90,
    "balanced_accuracy_min": 0.85,
    "pooled_real_false_ai_max": 0.10,
    "worst_real_false_ai_max": 0.20,
    "pooled_ai_recall_min": 0.80,
    "worst_ai_recall_min": 0.60,
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E47 decision contract already exists")
    found = {name: _digest(path) for name, path in ARM_PATHS.items()}
    if found != STREAM_SHA256 or _digest(E44_FUSION) != E44_FUSION_SHA256:
        raise ValueError(f"E47 decision inputs changed: {found}")
    payload = {
        "schema_version": 1,
        "state": "e47_decision_contract_frozen_before_score_interpretation",
        "manifest_sha256": MANIFEST_SHA256,
        "score_stream_sha256": STREAM_SHA256,
        "e44_fusion_sha256": E44_FUSION_SHA256,
        "roles": {"fit_and_select": "CAL", "one_shot_gate": "DEVELOPMENT"},
        "candidates": {
            "e46": "frozen E44 generalist+DDA fusion; no refit",
            "e46_univfd": "E46 score + UnivFD probability",
            "e46_unina": "E46 score + capped UNINA raw logit",
            "e46_univfd_unina": "E46 score + both GAN specialists",
        },
        "learned_heads": {
            "features": "StandardScaler over frozen input scores; probabilities use clamped logits; UNINA raw logit is clipped to [-100,100]",
            "head": "LogisticRegression(lbfgs)", "C": C_VALUE, "seed": SEED,
            "max_iter": 2_000, "sample_weight": "equal label/source mass within CAL",
        },
        "threshold": {
            "rule": "lowest CAL score cut satisfying pooled REAL FP <=10% and worst REAL-source FP <=20%",
            "ties": "scores equal to threshold are AI",
        },
        "eligibility_and_development_gates": GATES,
        "selection": {
            "eligible_only": True,
            "ranking": ["worst AI-source recall", "pooled AI recall", "ROC AUC", "balanced accuracy"],
            "tie_break": list(CANDIDATES),
            "licence_rule": "if the raw winner uses UNINA, choose e46_univfd when eligible and within 0.02 on both worst and pooled AI recall",
            "licence_margin": LICENSE_MARGIN,
        },
        "forbidden": ["DEVELOPMENT-informed fit", "DEVELOPMENT-informed threshold",
                      "backbone update", "candidate retry", "post-DEVELOPMENT repair"],
        "calibration_metrics_opened": 0,
        "development_metrics_opened": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> dict[str, Any]:
    raw = CONTRACT.read_bytes()
    payload = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (payload.get("state") != "e47_decision_contract_frozen_before_score_interpretation"
            or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
            or payload.get("calibration_metrics_opened") != 0
            or payload.get("development_metrics_opened") != 0):
        raise ValueError("E47 decision contract changed")
    if {name: _digest(path) for name, path in ARM_PATHS.items()} != STREAM_SHA256:
        raise ValueError("E47 decision score stream changed")
    return payload


def _load_rows(role: str) -> list[dict[str, Any]]:
    streams = {
        name: [json.loads(line) for line in path.read_text().splitlines() if line]
        for name, path in ARM_PATHS.items()
    }
    reference = streams["generalist"]
    if any(len(rows) != 2_400 for rows in streams.values()):
        raise ValueError("E47 decision score count changed")
    by_arm = {name: {row["record_id"]: row for row in rows} for name, rows in streams.items()}
    output = []
    for base in reference:
        peers = {name: rows.get(base["record_id"]) for name, rows in by_arm.items()}
        if any(peer is None for peer in peers.values()):
            raise ValueError(f"E47 decision join missing {base['record_id']}")
        identity = (base["role"], int(base["label"]), base["source"])
        if any((peer["role"], int(peer["label"]), peer["source"]) != identity for peer in peers.values()):
            raise ValueError(f"E47 decision identity drift {base['record_id']}")
        if base["role"] != role:
            continue
        output.append({
            "record_id": base["record_id"], "role": base["role"],
            "label": int(base["label"]), "source": base["source"],
            **{f"{name}_score": float(peer["score"]) for name, peer in peers.items()},
        })
    if len(output) != 1_200:
        raise ValueError(f"E47 {role} row count changed")
    return output


def _clamped_logit(value: float) -> float:
    bounded = float(np.clip(value, EPSILON, 1.0 - EPSILON))
    return float(np.log(bounded / (1.0 - bounded)))


def _e46_scores(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    fusion = joblib.load(E44_FUSION)
    features = np.asarray([e44_feature(row["generalist_score"], row["dda_score"]) for row in rows])
    return fusion["head"].predict_proba(features)[:, 1]


def _candidate_features(name: str, row: Mapping[str, Any]) -> list[float]:
    values = [_clamped_logit(float(row["e46_score"]))]
    if "univfd" in name:
        values.append(_clamped_logit(float(row["univfd_score"])))
    if "unina" in name:
        values.append(float(np.clip(float(row["unina_score"]), -100.0, 100.0)))
    return values


def _source_label_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(int(row["label"]), str(row["source"]))].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for label in (0, 1):
        label_groups = [indices for key, indices in groups.items() if key[0] == label]
        if not label_groups:
            raise ValueError("E47 CAL needs both labels")
        for indices in label_groups:
            weights[indices] = 1.0 / (2.0 * len(label_groups) * len(indices))
    return weights * (len(weights) / weights.sum())


def select_real_safe_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    scores = [float(row["score"]) for row in rows]
    candidates = sorted(set(scores) | {float(np.nextafter(value, np.inf)) for value in scores})
    for threshold in candidates:
        rates = _source_rates(rows, threshold)
        if rates["pooled_real_false_ai"] <= 0.10 + 1e-12 and rates["worst_real_false_ai"] <= 0.20 + 1e-12:
            return threshold
    raise ValueError("no E47 CAL threshold satisfies REAL safety")


def _source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), str(row["source"]))].append(float(row["score"]) >= threshold)
    real = {source: float(np.mean(values)) for (label, source), values in grouped.items() if label == 0}
    ai = {source: float(np.mean(values)) for (label, source), values in grouped.items() if label == 1}
    if not real or not ai:
        raise ValueError("E47 rates need REAL and AI sources")
    real_rows = [float(row["score"]) >= threshold for row in rows if int(row["label"]) == 0]
    ai_rows = [float(row["score"]) >= threshold for row in rows if int(row["label"]) == 1]
    return {
        "real_false_ai_by_source": dict(sorted(real.items())),
        "ai_recall_by_source": dict(sorted(ai.items())),
        "pooled_real_false_ai": float(np.mean(real_rows)), "worst_real_false_ai": max(real.values()),
        "pooled_ai_recall": float(np.mean(ai_rows)), "worst_ai_recall": min(ai.values()),
    }


def _gate(metrics: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "coverage_eq_1": float(metrics["coverage"]) == GATES["coverage"],
        "roc_auc_gte_0_90": float(metrics["roc_auc"]) >= GATES["roc_auc_min"],
        "balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= GATES["balanced_accuracy_min"],
        "pooled_real_false_ai_lte_0_10": float(rates["pooled_real_false_ai"]) <= GATES["pooled_real_false_ai_max"],
        "worst_real_false_ai_lte_0_20": float(rates["worst_real_false_ai"]) <= GATES["worst_real_false_ai_max"],
        "pooled_ai_recall_gte_0_80": float(rates["pooled_ai_recall"]) >= GATES["pooled_ai_recall_min"],
        "worst_ai_recall_gte_0_60": float(rates["worst_ai_recall"]) >= GATES["worst_ai_recall_min"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_candidate(summaries: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    eligible = [name for name in CANDIDATES if summaries[name]["gate"]["passed"]]
    if not eligible:
        raise RuntimeError("no E47 CAL candidate passes the frozen gate")
    def rank(name: str) -> tuple[float, float, float, float, int]:
        value = summaries[name]
        return (value["rates"]["worst_ai_recall"], value["rates"]["pooled_ai_recall"],
                value["metrics"]["roc_auc"], value["metrics"]["balanced_accuracy"],
                -CANDIDATES.index(name))
    raw = max(eligible, key=rank)
    mit = "e46_univfd"
    if "unina" in raw and mit in eligible:
        raw_rates, mit_rates = summaries[raw]["rates"], summaries[mit]["rates"]
        if (mit_rates["worst_ai_recall"] >= raw_rates["worst_ai_recall"] - LICENSE_MARGIN
                and mit_rates["pooled_ai_recall"] >= raw_rates["pooled_ai_recall"] - LICENSE_MARGIN):
            return mit, f"MIT preference within {LICENSE_MARGIN:.2f} on worst and pooled AI recall"
    return raw, "highest frozen CAL ranking"


def fit_calibration() -> dict[str, Any]:
    if any(path.exists() for path in (CAL_REPORT, CANDIDATE, CANDIDATE_EVIDENCE)):
        raise FileExistsError("E47 calibration output already exists")
    contract = _validate_contract()
    rows = _load_rows("CAL")
    e46 = _e46_scores(rows)
    prepared = [{**row, "e46_score": float(score)} for row, score in zip(rows, e46, strict=True)]
    heads: dict[str, Any] = {"e46": None}
    summaries: dict[str, Any] = {}
    weights = _source_label_weights(prepared)
    labels = np.asarray([row["label"] for row in prepared], dtype=np.int64)
    for name in CANDIDATES:
        if name == "e46":
            scores = e46
        else:
            x = np.asarray([_candidate_features(name, row) for row in prepared])
            head = make_pipeline(StandardScaler(), LogisticRegression(
                C=C_VALUE, max_iter=2_000, random_state=SEED, solver="lbfgs"))
            head.fit(x, labels, standardscaler__sample_weight=weights,
                     logisticregression__sample_weight=weights)
            heads[name] = head
            scores = head.predict_proba(x)[:, 1]
        scored = [{"record_id": row["record_id"], "role": "CAL", "label": row["label"],
                   "source": row["source"], "status": "ok", "score": float(score)}
                  for row, score in zip(prepared, scores, strict=True)]
        threshold = select_real_safe_threshold(scored)
        metrics = evaluate_binary_scores(scored, threshold=threshold)
        rates = _source_rates(scored, threshold)
        summaries[name] = {"threshold": threshold, "metrics": metrics,
                           "rates": rates, "gate": _gate(metrics, rates)}
    selected, reason = choose_candidate(summaries)
    artifact = {
        "schema_version": 1, "state": "e47_candidate_frozen_before_development",
        "selected_method": selected, "head": heads[selected],
        "threshold": summaries[selected]["threshold"], "C": C_VALUE, "seed": SEED,
        "contract_sha256": _digest(CONTRACT), "input_stream_sha256": STREAM_SHA256,
        "feature_policy": contract["learned_heads"], "selection_reason": reason,
    }
    temporary = CANDIDATE.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(CANDIDATE)
    report = {
        "schema_version": 1, "state": "e47_candidate_frozen_before_development",
        "cal_rows": len(prepared), "candidate_summaries": summaries,
        "selected_method": selected, "selection_reason": reason,
        "selected_threshold": summaries[selected]["threshold"],
        "candidate_bytes": CANDIDATE.stat().st_size, "candidate_sha256": _digest(CANDIDATE),
        "contract_sha256": _digest(CONTRACT), "development_metrics_opened": 0,
    }
    raw = _write(CAL_REPORT, report)
    _write(CANDIDATE_EVIDENCE, {**report, "cal_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def evaluate_development() -> dict[str, Any]:
    if any(path.exists() for path in (DEVELOPMENT_SCORES, DEVELOPMENT_REPORT, DEVELOPMENT_EVIDENCE)):
        raise FileExistsError("E47 DEVELOPMENT output already exists")
    _validate_contract()
    cal = json.loads(CAL_REPORT.read_text())
    if _digest(CANDIDATE) != cal.get("candidate_sha256") or cal.get("development_metrics_opened") != 0:
        raise ValueError("E47 frozen candidate changed")
    artifact = joblib.load(CANDIDATE)
    rows = _load_rows("DEVELOPMENT")
    e46 = _e46_scores(rows)
    prepared = [{**row, "e46_score": float(score)} for row, score in zip(rows, e46, strict=True)]
    method = artifact["selected_method"]
    if method == "e46":
        scores = e46
    else:
        x = np.asarray([_candidate_features(method, row) for row in prepared])
        scores = artifact["head"].predict_proba(x)[:, 1]
    scored = [{"record_id": row["record_id"], "role": "DEVELOPMENT", "label": row["label"],
               "source": row["source"], "status": "ok", "score": float(score)}
              for row, score in zip(prepared, scores, strict=True)]
    size, digest = _write_jsonl(DEVELOPMENT_SCORES, scored)
    threshold = float(artifact["threshold"])
    metrics = evaluate_binary_scores(scored, threshold=threshold)
    rates = _source_rates(scored, threshold)
    gate = _gate(metrics, rates)
    report = {
        "schema_version": 1,
        "state": "e47_development_passed" if gate["passed"] else "e47_development_failed",
        "selected_method": method, "threshold": threshold, "rows": len(scored),
        "candidate_sha256": _digest(CANDIDATE),
        "score_stream": {"bytes": size, "sha256": digest},
        "metrics": metrics, "rates": rates, "gate": gate,
        "boundary": "One-shot frozen DEVELOPMENT; no refit, threshold repair or retry permitted.",
    }
    raw = _write(DEVELOPMENT_REPORT, report)
    _write(DEVELOPMENT_EVIDENCE, {**report, "development_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "fit-cal", "evaluate-development"))
    args = parser.parse_args(argv)
    result = {"bind": bind, "fit-cal": fit_calibration,
              "evaluate-development": evaluate_development}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
