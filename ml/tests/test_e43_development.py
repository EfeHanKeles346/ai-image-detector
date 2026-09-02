from experiments.e43_development import historical_gate, rr_gate


def _metrics(auc=0.95, balanced=0.90, coverage=1.0, tpr=0.85, eer=0.10):
    return {
        "roc_auc": auc,
        "balanced_accuracy": balanced,
        "coverage": coverage,
        "tpr_at_fpr": {"tpr": tpr},
        "eer": eer,
    }


def _rates(real_macro=0.08, real_worst=0.15, ai_macro=0.85, ai_worst=0.65):
    return {
        "real_macro_fp": real_macro,
        "real_worst_source_fp": real_worst,
        "ai_macro_recall": ai_macro,
        "ai_worst_source_recall": ai_worst,
    }


def test_rr_gate_requires_every_condition_and_full_original_gate():
    report = {
        "original": {"metrics": _metrics(), "rates": _rates()},
        "transfer": {"metrics": _metrics(auc=0.85, balanced=0.80), "rates": _rates()},
        "redigital": {"metrics": _metrics(auc=0.85, balanced=0.80), "rates": _rates()},
    }
    assert rr_gate(report)["passed"]
    report["redigital"]["metrics"]["balanced_accuracy"] = 0.799
    assert not rr_gate(report)["passed"]


def test_historical_gate_enforces_frozen_regression_tolerance():
    clean = {"metrics": _metrics(auc=0.98, balanced=0.91), "rates": _rates()}
    robust = {"metrics": _metrics(auc=0.98, balanced=0.90), "rates": _rates()}
    assert historical_gate(clean, robust)["passed"]
    robust["metrics"]["balanced_accuracy"] = 0.88
    assert not historical_gate(clean, robust)["passed"]
