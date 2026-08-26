from __future__ import annotations

import pytest

from experiments import e32_role_manifest as manifest


def _rows(group_sizes: list[int]) -> list[dict[str, str]]:
    rows = []
    for group_index, size in enumerate(group_sizes):
        for row_index in range(size):
            rows.append(
                {
                    "source_key": f"g{group_index}/r{row_index}",
                    "prompt_group": f"g{group_index}",
                }
            )
    return rows


def test_select_records_preserves_indivisible_groups() -> None:
    selected = manifest.select_records(_rows([4, 4, 4]), 8, group_field="prompt_group")
    groups = {row["prompt_group"] for row in selected}
    assert len(selected) == 8
    assert all(sum(row["prompt_group"] == group for row in selected) == 4 for group in groups)


def test_select_records_fails_when_target_cuts_every_group() -> None:
    with pytest.raises(ValueError, match="cannot reach exact target"):
        manifest.select_records(_rows([4, 4]), 6, group_field="prompt_group")


def test_calibration_groups_reaches_exact_target_when_possible() -> None:
    sizes = {"a": 4, "b": 3, "c": 2, "d": 1}
    chosen = manifest.calibration_groups(sizes, 3)
    assert sum(sizes[group] for group in chosen) == 3
    assert chosen
    assert chosen != set(sizes)


def test_calibration_groups_is_deterministic() -> None:
    sizes = {"device-1": 100, "device-2": 100, "device-3": 100}
    assert manifest.calibration_groups(sizes, 60) == manifest.calibration_groups(sizes, 60)


def test_role_group_contracts() -> None:
    assert manifest._role_group("vision-base-native", {"device": "D01"}) == "device:D01"
    assert manifest._role_group("forchheim-fodb", {"scene_group": "scene:1"}) == "scene:1"
    assert manifest._role_group("qwen-image-2512", {"prompt_group": "p:1"}) == "p:1"
    assert manifest._role_group(
        "communityforensics-ai-local", {"model_name": "org/model"}
    ) == "generator:org/model"
