from __future__ import annotations

from collections import Counter
import zipfile

from experiments.e51_realization import select_ai_cal, select_scimd_reserve


def test_internal_ai_cal_is_source_balanced_and_deterministic():
    rows = [
        {"parent_id": f"{source}-{index}", "source": source, "label": 1, "role": "train"}
        for source in ("a", "b") for index in range(5)
    ]
    first = select_ai_cal(rows, per_source=2, minimum_source_rows=4)
    assert first == select_ai_cal(list(reversed(rows)), per_source=2, minimum_source_rows=4)
    assert Counter(row["source"] for row in first) == {"a": 2, "b": 2}
    assert all(row["e51_role"] == "CAL" for row in first)


def test_scimd_reserve_is_device_balanced_and_score_blind():
    infos = []
    for device in ("phone_a", "phone_b"):
        for index in range(4):
            info = zipfile.ZipInfo(f"SCIMD-17/{device}/{index}.jpg")
            info.file_size = index + 10
            info.CRC = index + 20
            infos.append(info)
    selected = select_scimd_reserve(infos, reserve_per_device=2)
    assert Counter(row["device"] for row in selected) == {"phone_a": 2, "phone_b": 2}
    assert all(row["role"] == "TRAIN" and row["label"] == 0 for row in selected)


def test_scimd_reserve_rejects_unbound_layout():
    info = zipfile.ZipInfo("SCIMD-17/device/nested/image.jpg")
    try:
        select_scimd_reserve([info], reserve_per_device=1)
    except ValueError as error:
        assert "unexpected SCIMD-17" in str(error)
    else:
        raise AssertionError("nested SCIMD member should fail closed")
