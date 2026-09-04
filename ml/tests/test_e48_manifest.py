from collections import Counter

import pytest

from experiments.e48_manifest import _device_split, balanced_select


def _rows(devices=3, scenes=20):
    return [
        {"device": f"d{device}", "scene_group": f"s{scene}",
         "identity": f"{device}:{scene}", "rank": f"{scene:02d}:{device}"}
        for scene in range(scenes) for device in range(devices)
    ]


def test_balanced_selection_round_robins_devices_and_caps_scenes():
    selected = balanced_select(_rows(), 30, max_per_scene=2)
    devices = Counter(row["device"] for row in selected)
    scenes = Counter(row["scene_group"] for row in selected)
    assert max(devices.values()) - min(devices.values()) <= 1
    assert max(scenes.values()) <= 2


def test_balanced_selection_fails_when_constraints_cannot_fill():
    with pytest.raises(ValueError, match="cannot fill"):
        balanced_select(_rows(devices=2, scenes=2), 4, max_per_scene=1)


def test_device_split_is_deterministic_and_disjoint():
    devices = [f"d{i}" for i in range(10)]
    first = _device_split("camera", devices)
    assert first == _device_split("camera", list(reversed(devices)))
    assert set(first.values()) == {"FIT", "CAL"}
    assert Counter(first.values()) == {"FIT": 5, "CAL": 5}
