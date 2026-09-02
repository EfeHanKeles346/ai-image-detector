from pathlib import Path

import pytest

from experiments.e43_itwsm_acquisition import classify_path, validate_local_snapshot


def test_itwsm_classification_keeps_real_and_ai_labels_explicit():
    assert classify_path("0_real/Facebook_real_1.jpg") == ("0_real", "image")
    assert classify_path("1_fake/x_1.jpg") == ("1_fake", "image")
    assert classify_path("metadata.csv") == ("metadata", "metadata")


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "other/x.jpg", "0_real/x.txt"])
def test_itwsm_classification_fails_closed(name):
    with pytest.raises(ValueError):
        classify_path(name)


def test_itwsm_local_snapshot_checks_sizes_and_extra_payload(tmp_path: Path):
    (tmp_path / "0_real").mkdir()
    (tmp_path / "0_real" / "a.jpg").write_bytes(b"abc")
    validate_local_snapshot(tmp_path, {"0_real/a.jpg": 3})

    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="differs"):
        validate_local_snapshot(tmp_path, {"0_real/a.jpg": 3})


def test_itwsm_local_snapshot_rejects_missing_or_wrong_size(tmp_path: Path):
    with pytest.raises(ValueError, match="missing or wrong size"):
        validate_local_snapshot(tmp_path, {"0_real/a.jpg": 3})
