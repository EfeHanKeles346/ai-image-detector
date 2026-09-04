from __future__ import annotations

from experiments.e49_open_realize import select_clean


def test_stylegan_clean_selection_respects_rank_and_all_overlap_rules():
    rows = [
        {"identity": "protected", "rank": "0", "sha256": "p", "dhash": "0"},
        {"identity": "first", "rank": "1", "sha256": "a", "dhash": "1"},
        {"identity": "duplicate", "rank": "2", "sha256": "b", "dhash": "1"},
        {"identity": "second", "rank": "3", "sha256": "c", "dhash": "3"},
    ]
    selected, reasons = select_clean(rows, {"p"}, set(), target=2)
    assert [row["identity"] for row in selected] == ["first", "second"]
    assert reasons["protected"] == ["protected_exact_overlap"]
    assert reasons["duplicate"] == ["internal_dhash_duplicate_of:first"]


def test_stylegan_clean_selection_tracks_protected_dhash():
    rows = [{"identity": "blocked", "rank": "0", "sha256": "a", "dhash": "x"},
            {"identity": "ok", "rank": "1", "sha256": "b", "dhash": "y"}]
    selected, reasons = select_clean(rows, set(), {"x"}, target=1)
    assert [row["identity"] for row in selected] == ["ok"]
    assert reasons["blocked"] == ["protected_dhash_overlap"]
