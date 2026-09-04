from __future__ import annotations

import hashlib

from PIL import Image
import pytest

from experiments.e49_commons_download import inspect_file


def _expected(path):
    return {
        "identity": "commons:1:sha", "rank": "rank", "label": 0,
        "source": "Phone", "category": "Category:Phone", "pageid": 1,
        "title": "File:test.jpg", "revision_timestamp": "2026-01-01T00:00:00Z",
        "uploader": "user", "description_url": "https://example.test/page",
        "license": "CC BY 4.0", "license_url": "https://example.test/license",
        "bytes": path.stat().st_size, "commons_sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        "width": 32, "height": 24,
    }


def test_commons_inspection_binds_sha1_jpeg_and_geometry(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), (20, 30, 40)).save(path, format="JPEG")
    found = inspect_file(path, _expected(path))
    assert found["format"] == "JPEG"
    assert found["sha1"] == hashlib.sha1(path.read_bytes()).hexdigest()
    assert len(found["sha256"]) == 64


def test_commons_inspection_rejects_hash_and_geometry_drift(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), (20, 30, 40)).save(path, format="JPEG")
    expected = _expected(path)
    with pytest.raises(ValueError, match="SHA1 changed"):
        inspect_file(path, {**expected, "commons_sha1": "0" * 40})
    with pytest.raises(ValueError, match="dimensions changed"):
        inspect_file(path, {**expected, "width": 31})
