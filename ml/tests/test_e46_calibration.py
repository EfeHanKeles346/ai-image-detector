import numpy as np

from experiments.e46_calibration import (
    apply_quality_gaussian,
    choose_method,
    fit_quality_gaussian,
    internal_roles,
    selective_metrics,
)


def test_internal_roles_are_source_stratified_and_deterministic():
    rows = [
        {"record_id": f"{source}-{index}", "typ": source, "role": "CAL"}
        for source in ("real", "ai") for index in range(20)
    ]
    roles = internal_roles(rows)
    assert sum(value == "QUALITY_FIT" for value in roles.values()) == 24
    assert sum(value == "OPERATING_CAL" for value in roles.values()) == 16
    assert roles == internal_roles(rows)


def test_quality_gaussian_returns_finite_probabilities():
    rows = []
    for label in (0, 1):
        for index in range(20):
            rows.append({
                "label": label,
                "quality": [float(index), float(index % 3), float(index % 5)],
                "fusion_score": 0.2 + 0.5 * label + 0.001 * index,
            })
    model = fit_quality_gaussian(rows)
    scores = apply_quality_gaussian(model, rows)
    assert scores.shape == (40,)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))


def _result(fp, recall, worst, auc):
    return {"real_false_ai": fp, "ai_recall": recall,
            "worst_generator_recall": worst, "roc_auc": auc}


def test_choose_method_keeps_simple_global_without_noninferior_quality_gain():
    results = {
        "dda_global": _result(0.1, 0.85, 0.70, 0.90),
        "fusion_global": _result(0.1, 0.83, 0.65, 0.92),
        "fusion_quality_gaussian": _result(0.1, 0.86, 0.69, 0.94),
    }
    assert choose_method(results) == ("dda_global", True)


def test_choose_method_allows_material_noninferior_quality_gain():
    results = {
        "dda_global": _result(0.1, 0.85, 0.70, 0.90),
        "fusion_global": _result(0.1, 0.83, 0.65, 0.92),
        "fusion_quality_gaussian": _result(0.09, 0.87, 0.72, 0.93),
    }
    assert choose_method(results) == ("fusion_quality_gaussian", True)


def test_selective_metrics_counts_uncertain_rows():
    rows = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]
    found = selective_metrics(
        rows, [0.1, 0.5, 0.7, 0.9],
        {"real_if_score_lt": 0.4, "ai_if_score_gte": 0.8},
    )
    assert found["automatic_coverage"] == 0.5
    assert found["covered_accuracy"] == 1.0
    assert found["uncertain_rows"] == 2
