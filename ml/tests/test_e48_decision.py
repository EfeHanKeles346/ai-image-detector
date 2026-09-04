import pytest

from experiments.e48_decision import choose_candidate, empirical_evidence, select_threshold


def test_empirical_evidence_is_monotone_and_bounded():
    real = [0.1, 0.2, 0.3]
    values = [empirical_evidence(real, value) for value in (0.0, 0.15, 0.25, 0.9)]
    assert values == sorted(values)
    assert all(0.0 < value <= 1.0 for value in values)


def test_empirical_evidence_rejects_unsorted_map():
    with pytest.raises(ValueError, match="sorted"):
        empirical_evidence([0.2, 0.1], 0.5)


def test_threshold_enforces_each_camera_pipeline():
    rows = []
    for camera in ("a", "b"):
        rows.extend({"label": 0, "source": "real", "camera_pipeline": camera, "score": score}
                    for score in (0.1, 0.2, 0.3, 0.4, 0.5))
    rows.extend({"label": 1, "source": "ai", "camera_pipeline": None, "score": score}
                for score in (0.6, 0.7, 0.8, 0.9, 1.0))
    threshold = select_threshold(rows)
    assert threshold > 0.4
    assert threshold <= 0.6


def _summary(*, passed: bool, worst: float, pooled: float, auc: float = 0.95, ba: float = 0.90):
    return {
        "gate": {"passed": passed},
        "rates": {"worst_ai_recall": worst, "pooled_ai_recall": pooled},
        "metrics": {"roc_auc": auc, "balanced_accuracy": ba},
    }


def test_candidate_selection_records_a_clean_failure():
    summaries = {
        name: _summary(passed=False, worst=0.99, pooled=0.99)
        for name in ("e46", "e46_univfd", "e46_unina", "e46_univfd_unina")
    }
    selected, reason = choose_candidate(summaries)
    assert selected is None
    assert "no E48 CAL candidate" in reason


def test_candidate_selection_applies_licence_preference_within_margin():
    summaries = {
        "e46": _summary(passed=True, worst=0.70, pooled=0.82),
        "e46_univfd": _summary(passed=True, worst=0.79, pooled=0.89),
        "e46_unina": _summary(passed=True, worst=0.80, pooled=0.90),
        "e46_univfd_unina": _summary(passed=False, worst=0.99, pooled=0.99),
    }
    selected, reason = choose_candidate(summaries)
    assert selected == "e46_univfd"
    assert "MIT preference" in reason
