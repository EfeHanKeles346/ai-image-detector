from __future__ import annotations

import pytest

from experiments.e49_openfake import (
    MODEL_KEYS,
    REVISION,
    compact_page,
    eligible,
    select_reserve,
)


def _wrapped(index: int, *, model: str = "gpt-image-2", label: str = "fake",
             row_type: str = "image") -> dict:
    return {
        "row_idx": index,
        "truncated_cells": [],
        "row": {
            "image": {
                "src": (
                    "https://datasets-server.huggingface.co/cached-assets/ComplexDataLab/OpenFake/"
                    f"--/{REVISION}/--/core/test/{index}/image/image.jpg?Expires=secret"
                ),
                "width": 1024,
                "height": 768,
            },
            "prompt": "must never enter compact metadata",
            "label": label,
            "model": model,
            "type": row_type,
            "release_date": "2026-01",
        },
    }


def test_compact_page_validates_revision_and_strips_prompt_and_url():
    payload = {"num_rows_total": 91_398, "partial": False, "rows": [_wrapped(91_397)]}
    rows = compact_page(payload, offset=91_397)
    assert rows == [{
        "row_index": 91_397,
        "label": "fake",
        "model": "gpt-image-2",
        "type": "image",
        "release_date": "2026-01",
        "width": 1024,
        "height": 768,
    }]
    assert "prompt" not in rows[0] and "src" not in rows[0]

    broken = _wrapped(91_397)
    broken["row"]["image"]["src"] = broken["row"]["image"]["src"].replace(REVISION, "bad")
    with pytest.raises(ValueError, match="asset identity"):
        compact_page({"num_rows_total": 91_398, "partial": False, "rows": [broken]}, offset=91_397)


def test_eligibility_requires_exact_fake_modern_nonvideo_cell():
    base = {"label": "fake", "model": "gpt-image-2", "type": "image"}
    assert eligible(base)
    assert not eligible({**base, "label": "real"})
    assert not eligible({**base, "model": "gpt-image-1.5"})
    assert not eligible({**base, "type": "video"})


def test_selection_is_deterministic_and_source_stratified():
    rows = []
    for model in MODEL_KEYS:
        rows.extend({
            "row_index": index,
            "label": "fake",
            "model": model,
            "type": "image",
            "release_date": "2026-01",
            "width": 512,
            "height": 512,
        } for index in range(10))
    first = select_reserve(rows, reserve_per_model=3)
    second = select_reserve(list(reversed(rows)), reserve_per_model=3)
    assert first == second
    assert len(first) == len(MODEL_KEYS) * 3
    assert {row["label"] for row in first} == {1}
    assert all(sum(row["model"] == model for row in first) == 3 for model in MODEL_KEYS)
