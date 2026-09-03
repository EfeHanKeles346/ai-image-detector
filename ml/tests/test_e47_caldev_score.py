import json

import pytest

from experiments.e47_caldev_score import _resume


def test_resume_rejects_identity_or_role_drift(tmp_path):
    rows = [{"record_id": "a", "role": "CAL", "label": 0, "source": "FFHQ"}]
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "a", "role": "CAL", "label": 0,
                                "source": "FFHQ", "score": 0.2}) + "\n")
    assert len(_resume(path, rows)) == 1
    path.write_text(json.dumps({"record_id": "a", "role": "DEVELOPMENT", "label": 0,
                                "source": "FFHQ", "score": 0.2}) + "\n")
    with pytest.raises(ValueError, match="changed"):
        _resume(path, rows)
