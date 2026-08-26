from __future__ import annotations

import pytest

from experiments.e32_r1b_ipn_audit import cross_scene_perceptual_groups, scene_group


def test_scene_group_matches_shared_landscape_and_portrait_ids() -> None:
    assert scene_group("iPhone_SE2020_1_01 natural Landscape.JPG") == "ipn:landscape:01"
    assert scene_group("LG_L65_40 natural Portrait.JPG") == "ipn:portrait:40"


def test_scene_group_rejects_unstructured_name() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        scene_group("natural.JPG")


def test_perceptual_duplicates_are_allowed_only_within_scene() -> None:
    records = [
        {"source_key": "a", "scene_group": "ipn:landscape:01"},
        {"source_key": "b", "scene_group": "ipn:landscape:01"},
        {"source_key": "c", "scene_group": "ipn:portrait:02"},
    ]
    assert cross_scene_perceptual_groups(records, [["a", "b"], ["a", "c"]]) == [["a", "c"]]
