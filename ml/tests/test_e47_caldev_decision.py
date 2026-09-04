import pytest

from experiments.e47_caldev_decision import _gate, choose_candidate, select_real_safe_threshold


def _summary(worst, pooled, *, passed=True, auc=0.95, balanced=0.90):
    return {"rates": {"worst_ai_recall": worst, "pooled_ai_recall": pooled},
            "metrics": {"roc_auc": auc, "balanced_accuracy": balanced},
            "gate": {"passed": passed}}


def test_real_safe_threshold_uses_boundary_above_tied_score():
    rows = [
        {"label": 0, "source": "real", "score": 0.2},
        {"label": 0, "source": "real", "score": 0.2},
        {"label": 1, "source": "ai", "score": 0.9},
    ]
    threshold = select_real_safe_threshold(rows)
    assert threshold > 0.2
    assert threshold <= 0.9


def test_candidate_selection_prefers_worst_source_then_mit_margin():
    summaries = {
        "e46": _summary(0.70, 0.82),
        "e46_univfd": _summary(0.80, 0.89),
        "e46_unina": _summary(0.81, 0.90),
        "e46_univfd_unina": _summary(0.82, 0.91),
    }
    selected, reason = choose_candidate(summaries)
    assert selected == "e46_univfd"
    assert "MIT" in reason


def test_candidate_selection_rejects_license_override_outside_margin():
    summaries = {
        "e46": _summary(0.70, 0.82),
        "e46_univfd": _summary(0.77, 0.88),
        "e46_unina": _summary(0.82, 0.91),
        "e46_univfd_unina": _summary(0.85, 0.93),
    }
    selected, _ = choose_candidate(summaries)
    assert selected == "e46_univfd_unina"


def test_gate_requires_every_real_and_ai_condition():
    metrics = {"coverage": 1.0, "roc_auc": 0.95, "balanced_accuracy": 0.90}
    rates = {"pooled_real_false_ai": 0.10, "worst_real_false_ai": 0.20,
             "pooled_ai_recall": 0.80, "worst_ai_recall": 0.60}
    assert _gate(metrics, rates)["passed"] is True
    rates["worst_ai_recall"] = 0.59
    assert _gate(metrics, rates)["passed"] is False


def test_no_eligible_candidate_is_a_hard_stop():
    summaries = {name: _summary(0.1, 0.2, passed=False) for name in
                 ("e46", "e46_univfd", "e46_unina", "e46_univfd_unina")}
    with pytest.raises(RuntimeError, match="no E47 CAL candidate"):
        choose_candidate(summaries)
