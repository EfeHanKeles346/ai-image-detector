from __future__ import annotations

import numpy as np

from experiments import e32_r0_loco as loco


def test_source_metric_real_reports_false_positive_rate() -> None:
    result = loco.source_metric(0, np.array([0.1, 0.6, 0.9]), 0.5)
    assert result["real_false_positive_rate"] == 2 / 3
    assert "ai_recall" not in result


def test_source_metric_ai_reports_recall() -> None:
    result = loco.source_metric(1, np.array([0.1, 0.6, 0.9]), 0.5)
    assert result["ai_recall"] == 2 / 3
    assert "real_false_positive_rate" not in result
