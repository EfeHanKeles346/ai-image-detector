from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e49_evaluation import CONDITIONS, SOURCE_COUNTS
from experiments.e49_final_score import _identity_sha, validate_scores


def _scores():
    parents = [
        {"parent_id": f"{source}:{index}", "label": 0 if count == 100 else 1, "source": source}
        for source, count in SOURCE_COUNTS.items() for index in range(count)
    ]
    return [
        {**row, "record_id": f"{row['parent_id']}:{condition}", "condition": condition,
         "sha256": "a" * 64, "status": "ok", "score": 0.1}
        for condition in CONDITIONS for row in parents
    ]


def test_final_score_validator_requires_complete_finite_scores():
    rows = _scores()
    validate_scores(rows)
    broken = deepcopy(rows)
    broken[0]["score"] = None
    with pytest.raises((TypeError, ValueError)):
        validate_scores(broken)


def test_final_identity_digest_changes_with_payload_identity():
    rows = _scores()
    first = _identity_sha(rows)
    rows[0]["sha256"] = "b" * 64
    assert _identity_sha(rows) != first
