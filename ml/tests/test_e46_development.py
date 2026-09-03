import json

import numpy as np
import pytest
from PIL import Image

from experiments.e46_development import _resume, quality_proxies


def test_quality_proxies_are_finite_and_deterministic():
    image = Image.new("RGB", (80, 40), (10, 20, 30))
    first = quality_proxies(image, 1_000)
    second = quality_proxies(image, 1_000)
    assert first == second
    assert len(first) == 3
    assert np.all(np.isfinite(first))


def test_quality_proxies_reject_invalid_bytes():
    with pytest.raises(ValueError):
        quality_proxies(Image.new("RGB", (10, 10)), 0)


def test_resume_accepts_only_exact_manifest_prefix(tmp_path):
    rows = [{"record_id": "a", "role": "CAL", "label": 0},
            {"record_id": "b", "role": "DEVELOPMENT", "label": 1}]
    path = tmp_path / "scores.partial"
    path.write_text(json.dumps({"record_id": "a", "role": "CAL", "label": 0, "score": 0.2}) + "\n")
    assert len(_resume(path, rows)) == 1
    path.write_text(json.dumps({"record_id": "b", "role": "CAL", "label": 0, "score": 0.2}) + "\n")
    with pytest.raises(ValueError):
        _resume(path, rows)
