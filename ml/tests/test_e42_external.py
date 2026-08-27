from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from experiments import e42_external as external


def _write_registry(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "filename", "label", "source_id", "date", "days_since_1st_post", "w", "h", "md5", "url"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_preserves_label_and_parent_event(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    rows = [
        {
            "filename": "REAL/event-a/a.jpg", "label": "REAL", "source_id": "event-a",
            "date": "2024-01-01", "days_since_1st_post": "0", "w": "10", "h": "20",
            "md5": "a" * 32, "url": "https://example.test/a.jpg",
        },
        {
            "filename": "FAKE/event-b/b.png", "label": "FAKE", "source_id": "event-b",
            "date": "2024-01-02", "days_since_1st_post": "1.5", "w": "30", "h": "40",
            "md5": "b" * 32, "url": "http://example.test/b.png",
        },
    ]
    monkeypatch.setattr(external, "REGISTRY_SHA256", _write_registry(path, rows))
    monkeypatch.setattr(external, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(external, "EXPECTED_BY_LABEL", {"REAL": 1, "FAKE": 1})
    monkeypatch.setattr(external, "EXPECTED_SOURCES_BY_LABEL", {"REAL": 1, "FAKE": 1})
    loaded = external.load_registry(path)
    assert [row["label"] for row in loaded] == [0, 1]
    assert [row["source_id"] for row in loaded] == ["event-a", "event-b"]


def test_registry_rejects_cross_label_parent(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    rows = [
        {
            "filename": f"{label}/shared/{index}.jpg", "label": label, "source_id": "shared",
            "date": "2024-01-01", "days_since_1st_post": "0", "w": "10", "h": "20",
            "md5": str(index) * 32, "url": f"https://example.test/{index}.jpg",
        }
        for index, label in ((1, "REAL"), (2, "FAKE"))
    ]
    monkeypatch.setattr(external, "REGISTRY_SHA256", _write_registry(path, rows))
    with pytest.raises(ValueError, match="crosses labels"):
        external.load_registry(path)


def test_summary_keeps_url_failures_visible() -> None:
    summary = external.summarize([
        {"status": "verified", "label_name": "REAL", "source_id": "r1", "bytes": 7},
        {"status": "request_failed", "label_name": "REAL", "source_id": "r2"},
        {"status": "verified", "label_name": "FAKE", "source_id": "f1", "bytes": 11},
    ])
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["failure_statuses"] == {"request_failed": 1}
    assert summary["covered_sources_by_label"] == {"REAL": 1, "FAKE": 1}


def test_overlap_excludes_whole_parent_event_and_cross_event_duplicates() -> None:
    rows = [
        {"source_id": "r1", "sha256": "a", "dhash": "1"},
        {"source_id": "r1", "sha256": "b", "dhash": "2"},
        {"source_id": "f1", "sha256": "c", "dhash": "3"},
        {"source_id": "f2", "sha256": "d", "dhash": "3"},
    ]
    excluded, cross = external.excluded_event_ids(rows, {"b"}, set())
    assert excluded == {"r1", "f1", "f2"}
    assert cross == {"dhash:3": ["f1", "f2"]}
