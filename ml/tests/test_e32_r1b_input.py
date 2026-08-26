from __future__ import annotations

import pytest

from experiments.e32_r1b_input import merge_receipts


def _row(record_id: str, source: str = "old") -> dict[str, str]:
    return {"record_id": record_id, "source_id": source, "label": "real", "role": "TRAIN"}


def test_merge_receipts_preserves_contract_and_sorts() -> None:
    merged = merge_receipts([_row("b")], [_row("a", "new")], [_row("b"), _row("a", "new")])
    assert [row["record_id"] for row in merged] == ["a", "b"]


def test_merge_receipts_rejects_role_drift() -> None:
    manifest = _row("a")
    manifest["role"] = "CALIBRATION"
    with pytest.raises(ValueError, match="metadata changed"):
        merge_receipts([_row("a")], [], [manifest])
