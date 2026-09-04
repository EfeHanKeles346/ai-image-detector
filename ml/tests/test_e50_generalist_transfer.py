import json

import pytest

from experiments.e50_generalist_transfer import _resume


def _rows():
    return [
        {"record_id": "r0", "role": "DEVELOPMENT", "label": 0, "source": "FODB"},
        {"record_id": "a0", "role": "DEVELOPMENT", "label": 1, "source": "FLUX.1"},
    ]


def test_resume_accepts_exact_development_prefix(tmp_path):
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "r0", "role": "DEVELOPMENT", "label": 0,
                                "source": "FODB", "score": 0.1}) + "\n")
    assert len(_resume(path, _rows())) == 1


def test_resume_rejects_non_development_role(tmp_path):
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "r0", "role": "CAL", "label": 0,
                                "source": "FODB", "score": 0.1}) + "\n")
    with pytest.raises(ValueError, match="prefix changed"):
        _resume(path, _rows())


def test_resume_rejects_truncated_line(tmp_path):
    path = tmp_path / "scores.partial"
    path.write_text('{"record_id": "r0"}')
    with pytest.raises(ValueError, match="truncated"):
        _resume(path, _rows())
