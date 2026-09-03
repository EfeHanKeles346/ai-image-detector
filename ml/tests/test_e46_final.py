import json

import numpy as np
import pytest

from experiments.e46_final import _resume, bootstrap_primary, selective_metrics, source_rates


def _rows():
    return [
        {"record_id": "r1", "label": 0, "source": "real-a", "score": 0.1},
        {"record_id": "r2", "label": 0, "source": "real-b", "score": 0.7},
        {"record_id": "a1", "label": 1, "source": "fake-a", "score": 0.8},
        {"record_id": "a2", "label": 1, "source": "fake-b", "score": 0.4},
    ]


def test_source_rates_reports_each_publisher_source():
    found = source_rates(_rows(), 0.6)
    assert found["pooled_real_false_ai"] == 0.5
    assert found["pooled_ai_recall"] == 0.5
    assert found["worst_real_false_ai"] == 1.0
    assert found["worst_ai_recall"] == 0.0


def test_selective_metrics_are_consistent():
    found = selective_metrics(_rows())
    assert 0.0 <= found["automatic_coverage"] <= 1.0
    assert found["automatic_rows"] + found["uncertain_rows"] == 4


def test_bootstrap_is_seeded_and_finite():
    first = bootstrap_primary(_rows(), samples=100, seed=7)
    second = bootstrap_primary(_rows(), samples=100, seed=7)
    assert first == second
    assert all(np.all(np.isfinite(bounds)) for bounds in first.values())


def test_resume_rejects_wrong_final_prefix(tmp_path):
    rows = [{"record_id": "a", "source": "FFHQ", "label": 0}]
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "wrong", "source": "FFHQ", "label": 0, "score": 0.2}) + "\n")
    with pytest.raises(ValueError):
        _resume(path, rows)
