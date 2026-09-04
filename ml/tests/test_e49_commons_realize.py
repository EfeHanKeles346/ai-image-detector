from __future__ import annotations

from experiments.e49_commons_realize import audit_originals


def test_commons_audit_is_order_independent_and_excludes_protected():
    rows = [
        {"identity": "a", "rank": "1", "sha256": "sha-a", "dhash": "d-a"},
        {"identity": "b", "rank": "2", "sha256": "sha-b", "dhash": "d-b"},
        {"identity": "c", "rank": "3", "sha256": "sha-c", "dhash": "d-c"},
    ]
    first = audit_originals(rows, {"sha-b"}, {"d-c"})
    second = audit_originals(list(reversed(rows)), {"sha-b"}, {"d-c"})
    assert first == second
    assert first == {"b": ["protected_exact_overlap"], "c": ["protected_dhash_overlap"]}


def test_commons_audit_marks_later_internal_duplicates():
    rows = [
        {"identity": "a", "rank": "1", "sha256": "same", "dhash": "d-a"},
        {"identity": "b", "rank": "2", "sha256": "same", "dhash": "d-b"},
        {"identity": "c", "rank": "3", "sha256": "sha-c", "dhash": "d-a"},
    ]
    reasons = audit_originals(rows, set(), set())
    assert reasons["b"] == ["internal_exact_duplicate_of:a"]
    assert reasons["c"] == ["internal_dhash_duplicate_of:a"]
