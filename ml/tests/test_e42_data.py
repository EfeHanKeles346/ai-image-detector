from __future__ import annotations

import pytest

from experiments.e42_data import fixed_replay, rr_source


def test_rr_source_requires_declared_class_filename() -> None:
    assert rr_source("real_000123.jpg", "real") == "rrdataset_real_pool"
    assert rr_source("Culture_&_Religion_000384.png", "ai") == "culture_and_religion"
    assert rr_source("normal_009095.png", "ai") == "everyday_life"
    with pytest.raises(ValueError):
        rr_source("mystery_000001.png", "ai")
    with pytest.raises(ValueError):
        rr_source("normal_000001.png", "real")


def test_fixed_replay_is_group_local_and_deterministic() -> None:
    rows = [
        {
            "record_id": f"{label}-{source}-{index}",
            "label": label,
            "source_id": source,
            "role": "TRAIN",
        }
        for label in ("real", "ai")
        for source in ("a", "b")
        for index in range(40)
    ]
    rows.append({"record_id": "ignored", "label": "real", "source_id": "a", "role": "CALIBRATION"})
    first = fixed_replay(rows)
    second = fixed_replay(list(reversed(rows)))
    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]
    assert len(first) == 8
    assert {row["role"] for row in first} == {"TRAIN"}
