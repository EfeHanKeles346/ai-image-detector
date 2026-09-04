from __future__ import annotations

import pytest

from experiments.e49_openfake import REVISION
from experiments.e49_openfake_assets import extract_asset_urls


def _row(index: int) -> dict:
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
            "prompt": "not retained",
            "label": "fake",
            "model": "gpt-image-2",
            "type": "image",
            "release_date": "2026-04",
        },
    }


def test_asset_page_requires_frozen_metadata_and_keeps_url_ephemeral():
    index = 91_397
    expected = {
        "record_id": f"openfake:core:test:{index}",
        "row_index": index,
        "model": "gpt-image-2",
        "release_date": "2026-04",
        "width": 1024,
        "height": 768,
    }
    payload = {"num_rows_total": 91_398, "partial": False, "rows": [_row(index)]}
    found = extract_asset_urls(payload, offset=index, wanted={index: expected})
    assert found[0]["row"] == expected
    assert REVISION in found[0]["url"]
    assert "url" not in found[0]["row"]

    changed = {**expected, "width": 512}
    with pytest.raises(ValueError, match="metadata changed"):
        extract_asset_urls(payload, offset=index, wanted={index: changed})
