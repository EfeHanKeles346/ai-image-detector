from __future__ import annotations

from experiments.e35_dda_threshold_diagnostic import threshold_summary


def test_threshold_summary_keeps_real_populations_separate() -> None:
    rows = [
        {"population": "rr", "label": 0, "score": 0.1},
        {"population": "rr", "label": 1, "score": 0.9},
        {"population": "owner", "label": 0, "score": 0.8},
        {"population": "ipn", "label": 0, "source": "phone-a", "score": 0.2},
        {"population": "ipn", "label": 0, "source": "phone-b", "score": 0.7},
    ]
    result = threshold_summary(rows, 0.5)
    assert result["rr_real_fp"] == 0.0
    assert result["rr_ai_recall"] == 1.0
    assert result["owner_fp"] == 1.0
    assert result["ipn_macro_device_fp"] == 0.5
    assert result["ipn_worst_device_fp"] == 1.0
