import numpy as np

from experiments.e46_e45_diagnostic import pattern_counts, threshold_at_real_budget


def test_threshold_at_real_budget_uses_conservative_higher_quantile():
    rows = [
        {"label": 0, "score": score} for score in np.linspace(0.0, 0.99, 100)
    ] + [{"label": 1, "score": 0.95}]
    threshold = threshold_at_real_budget(rows, 0.10)
    false_ai = np.mean([row["score"] >= threshold for row in rows if row["label"] == 0])
    assert false_ai <= 0.101


def test_pattern_counts_preserves_arm_disagreement():
    generalist = [
        {"record_id": "a", "label": 0, "score": 0.99},
        {"record_id": "b", "label": 1, "score": 0.01},
    ]
    specialist = [
        {"record_id": "a", "label": 0, "score": 0.01},
        {"record_id": "b", "label": 1, "score": 0.99},
    ]
    fused = [
        {"record_id": "a", "label": 0, "score": 0.99},
        {"record_id": "b", "label": 1, "score": 0.01},
    ]
    found = pattern_counts(generalist, specialist, fused)
    assert found["fusion_false_ai"] == {"generalist_1_specialist_0": 1}
    assert found["fusion_false_real"] == {"generalist_0_specialist_1": 1}
