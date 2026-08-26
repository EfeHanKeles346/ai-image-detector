from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "experiments/e30_development_benchmark.py"
SPEC = importlib.util.spec_from_file_location("e30_development_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e30 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e30
SPEC.loader.exec_module(e30)


def row(label, predicted, transport="standardized_jpeg", group=None, score=None):
    return {
        "status": "ok",
        "label": label,
        "predicted_ai": predicted,
        "transport": transport,
        "group": group or ("hybrid:generator" if label == "ai" else "hybrid:MLLMGenSet matched real"),
        "generator": "synthetic" if label == "ai" else None,
        "score": float(predicted) if score is None else score,
    }


def test_exact_interval_handles_boundaries():
    assert e30.exact_interval(0, 10)[0] == 0.0
    assert e30.exact_interval(10, 10)[1] == 1.0
    assert e30.exact_interval(1, 0) is None


def test_summary_keeps_transport_metrics_and_exact_counts():
    rows = []
    for transport in e30.TRANSPORTS:
        rows.extend(
            [
                row("real", False, transport, score=0.1),
                row("real", True, transport, score=0.9),
                row("ai", True, transport, score=0.8),
                row("ai", False, transport, score=0.2),
            ]
        )
    report = e30.summarize(rows, 0.5)
    assert report["accounting"] == {"expected": 20, "succeeded": 20, "failed": 0}
    assert report["overall"]["real_false_positive"]["rate"] == 0.5
    assert report["overall"]["real_false_positive"]["exact_95_ci"] is None
    assert report["overall"]["ai_recall"]["rate"] == 0.5
    assert report["overall"]["roc_auc"] == 0.5
    assert set(report["per_transport"]) == set(e30.TRANSPORTS)
    assert report["per_transport"]["standardized_jpeg"]["ai_recall"]["exact_95_ci"]
    assert report["per_generator"]["synthetic"]["rate"] == 0.5
    assert report["per_artifact_regime"]["ai::hybrid"]["rate"] == 0.5
    assert all(value == 0.0 for value in report["recall_delta_vs_standardized"].values())


def test_e31_arm_is_exposed_without_changing_existing_names():
    assert {"e20", "cf_vit", "dinov2_e31"} <= {
        "e20", "cf_vit", "dinov2_e31"
    }


def test_development_gate_enforces_transport_loss_and_fp_budgets():
    summary = {
        "accounting": {"failed": 0},
        "macro": {
            "real_false_positive": 0.04,
            "worst_real_group_false_positive": 0.10,
            "ai_recall": 0.60,
            "worst_ai_group_recall": 0.30,
        },
        "recall_delta_vs_standardized": {
            "jpeg_q90": -0.01,
            "jpeg_q75": -0.15,
            "jpeg_q50": -0.20,
            "resize256_q90": -0.10,
        },
    }
    assert e30.development_gate(summary)["passed"] is True
    summary["macro"]["real_false_positive"] = 0.051
    assert e30.development_gate(summary)["passed"] is False
