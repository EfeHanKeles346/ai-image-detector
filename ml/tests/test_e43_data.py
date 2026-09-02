from collections import Counter

import pytest

from experiments.e43_data import assign_roles


def _rows(parent: str, label: int, source: str):
    return [
        {
            "record_id": f"{parent}:{condition}",
            "parent_id": parent,
            "label": label,
            "source": source,
            "condition": condition,
        }
        for condition in ("original", "transfer", "redigital")
    ]


def test_e43_roles_keep_complete_parents_together_and_ignore_incomplete():
    rows = []
    for index in range(4):
        rows.extend(_rows(f"real-{index}", 0, "rrdataset_real_pool"))
        rows.extend(_rows(f"ai-{index}", 1, "everyday_life"))
    rows.extend(_rows("incomplete", 0, "rrdataset_real_pool")[:2])
    caps = {"real": 4, "everyday_life": 4}
    role_counts = {
        "real": {"train": 2, "calibration": 1, "development": 1},
        "everyday_life": {"train": 2, "calibration": 1, "development": 1},
    }
    selected, summary = assign_roles(rows, caps=caps, role_counts=role_counts)
    assert summary["selected_parents"] == 8
    assert summary["selected_rows"] == 24
    assert "incomplete" not in {row["parent_id"] for row in selected}
    roles_by_parent = {}
    for row in selected:
        roles_by_parent.setdefault(row["parent_id"], set()).add(row["e43_role"])
    assert all(len(roles) == 1 for roles in roles_by_parent.values())
    assert Counter(row["e43_role"] for row in selected) == {
        "train": 12, "calibration": 6, "development": 6
    }


def test_e43_role_assignment_is_deterministic_and_score_blind():
    rows = [row for index in range(5) for row in _rows(f"real-{index}", 0, "rrdataset_real_pool")]
    caps = {"real": 4}
    role_counts = {"real": {"train": 2, "calibration": 1, "development": 1}}
    first, _ = assign_roles(rows, caps=caps, role_counts=role_counts)
    reversed_rows = [{**row, "score": index / 100} for index, row in enumerate(reversed(rows))]
    second, _ = assign_roles(reversed_rows, caps=caps, role_counts=role_counts)
    assert [(row["record_id"], row["e43_role"]) for row in first] == [
        (row["record_id"], row["e43_role"]) for row in second
    ]


def test_e43_roles_reject_label_crossing_parent():
    rows = _rows("bad", 0, "rrdataset_real_pool")
    rows[-1] = {**rows[-1], "label": 1}
    with pytest.raises(ValueError, match="label/source changed"):
        assign_roles(rows, caps={"real": 1}, role_counts={"real": {"train": 1, "calibration": 0, "development": 0}})
