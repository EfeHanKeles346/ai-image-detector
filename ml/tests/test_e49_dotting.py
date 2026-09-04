from __future__ import annotations

import hashlib

import pytest

from experiments.e49_dotting import MODEL_KEYS, RESERVE_PER_MODEL, select_reserve, validate_download


def _population():
    rows = []
    remote = {}
    for model_key in MODEL_KEYS:
        for index in range(210):
            request_id = f"{model_key}-{index}"
            image = f"images/{model_key}/{index}.webp"
            rows.append({"request_id": request_id, "status": "ok", "word_slug": str(index),
                         "model_key": model_key, "prompt_key": "sign", "image": image,
                         "width": 768, "height": 768})
            remote[image] = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    return rows, remote


def test_dotting_reserve_is_deterministic_and_balanced():
    rows, remote = _population()
    first = select_reserve(rows, remote)
    second = select_reserve(list(reversed(rows)), remote)
    assert first == second
    assert len(first) == RESERVE_PER_MODEL * len(MODEL_KEYS)
    assert {source: sum(row["source"] == source for row in first)
            for source in MODEL_KEYS.values()} == {
                source: RESERVE_PER_MODEL for source in MODEL_KEYS.values()
            }


def test_dotting_selection_rejects_population_drift():
    rows, remote = _population()
    rows.pop()
    with pytest.raises(ValueError, match="population changed"):
        select_reserve(rows, remote)


def test_dotting_download_validation_checks_hash_and_unexpected_files(tmp_path):
    path = tmp_path / "images" / "a.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    rows = [{"image_path": "images/a.webp", "bytes": 3,
             "sha256": hashlib.sha256(b"abc").hexdigest()}]
    validate_download(tmp_path, rows)
    (tmp_path / "extra.webp").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected"):
        validate_download(tmp_path, rows)
