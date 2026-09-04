from __future__ import annotations

import pytest

from experiments.e49_openfake import MODEL_KEYS_C
from experiments.e49_openfake_realize import select_clean_candidates


def _rows(per_model: int = 3):
    return [
        {
            "record_id": f"{model}:{index}", "model": model,
            "rank": f"{index:02d}", "sha256": f"sha:{model}:{index}",
            "dhash": f"dhash:{model}:{index}",
        }
        for model in MODEL_KEYS_C for index in range(per_model)
    ]


def test_openfake_clean_selection_is_balanced_and_deterministic():
    rows = _rows()
    first, reasons = select_clean_candidates(rows, set(), set(), target_per_model=2)
    second, _ = select_clean_candidates(list(reversed(rows)), set(), set(), target_per_model=2)
    assert first == second
    assert reasons == {}
    assert {model: sum(row["model"] == model for row in first) for model in MODEL_KEYS_C} == {
        model: 2 for model in MODEL_KEYS_C
    }


def test_openfake_clean_selection_excludes_protected_and_internal_overlap():
    rows = _rows()
    rows[0]["sha256"] = "protected"
    rows[1]["dhash"] = rows[2]["dhash"]
    selected, reasons = select_clean_candidates(rows, {"protected"}, set(), target_per_model=1)
    assert rows[0]["record_id"] in reasons
    assert rows[2]["record_id"] in reasons
    assert all(row["record_id"] not in reasons for row in selected)


def test_openfake_clean_selection_refuses_underfilled_cell():
    rows = _rows(per_model=1)
    with pytest.raises(ValueError, match="clean target unavailable"):
        select_clean_candidates(rows, set(), set(), target_per_model=2)
