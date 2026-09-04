import json

import pytest

from experiments.e48_score import _resume


def test_resume_accepts_fit_and_rejects_development_or_identity_drift(tmp_path):
    rows = [{"record_id": "a", "role": "FIT", "label": 0, "source": "camera"}]
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "a", "role": "FIT", "label": 0,
                                "source": "camera", "score": 0.2}) + "\n")
    assert len(_resume(path, rows)) == 1
    path.write_text(json.dumps({"record_id": "a", "role": "DEVELOPMENT", "label": 0,
                                "source": "camera", "score": 0.2}) + "\n")
    with pytest.raises(ValueError, match="changed"):
        _resume(path, rows)


def test_resume_rejects_truncated_jsonl(tmp_path):
    path = tmp_path / "scores.partial"
    path.write_text('{"record_id":"a"}')
    with pytest.raises(ValueError, match="truncated"):
        _resume(path, [{"record_id": "a", "role": "CAL", "label": 1, "source": "ai"}])
