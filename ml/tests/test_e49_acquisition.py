from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e49_acquisition import (
    AIGC_GENERATOR_CODE,
    AIGC_RESERVE,
    COMMONS_CATEGORIES,
    COMMONS_RESERVE_PER_DEVICE,
    build_open_components_payload,
    commons_rows,
    read_aigc_coordinates,
    select_aigc_coordinates,
    select_capped,
    validate_hf_identity,
)
from experiments.e49_evaluation import SOURCE_COUNTS


def test_hf_identity_binds_revision_and_licence() -> None:
    payload = {"id": "owner/repo", "sha": "abc", "cardData": {"license": "cc-by-4.0"}}
    validate_hf_identity(payload, repo="owner/repo", revision="abc", license_id="cc-by-4.0")
    changed = deepcopy(payload)
    changed["sha"] = "def"
    with pytest.raises(ValueError, match="identity changed"):
        validate_hf_identity(changed, repo="owner/repo", revision="abc", license_id="cc-by-4.0")


def test_commons_device_contract_matches_final_evaluator() -> None:
    sources = {category.removeprefix("Category:Taken with ") for category in COMMONS_CATEGORIES}
    assert sources == {source for source, count in SOURCE_COUNTS.items() if count == 100}


def test_open_component_contract_binds_every_reserve_without_scores() -> None:
    commons = {}
    for category in COMMONS_CATEGORIES:
        commons[category] = [
            {"identity": f"{category}:{index}", "rank": f"{index:04d}",
             "uploader": f"u{index}", "bytes": 1}
            for index in range(COMMONS_RESERVE_PER_DEVICE)
        ]
    aigc = [{"identity": f"a:{index}", "rank": f"{index:04d}"} for index in range(AIGC_RESERVE)]
    payload = build_open_components_payload(commons, aigc)
    assert payload["commons"]["reserve_rows"] == 1_100
    assert payload["commons"]["target_rows"] == 1_000
    assert payload["aigc"]["reserve_rows"] == 240
    assert payload["new_image_bytes_downloaded"] == payload["model_scores_created"] == 0


def test_open_component_size_cap_is_applied_before_rank_selection() -> None:
    commons = {}
    for category in COMMONS_CATEGORIES:
        commons[category] = [
            {"identity": f"{category}:large", "rank": "0000", "uploader": "large", "bytes": 11},
            *({"identity": f"{category}:{index}", "rank": f"{index + 1:04d}",
               "uploader": f"u{index}", "bytes": 1}
              for index in range(COMMONS_RESERVE_PER_DEVICE)),
        ]
    aigc = [{"identity": f"a:{index}", "rank": f"{index:04d}"} for index in range(AIGC_RESERVE)]
    payload = build_open_components_payload(commons, aigc, commons_file_cap=10)
    assert payload["commons"]["max_file_bytes"] == 10
    assert all(row["bytes"] == 1 for row in payload["commons"]["rows"])


def test_capped_selection_is_deterministic_and_limits_contributors() -> None:
    rows = [
        {"identity": str(index), "rank": f"{index:03d}", "uploader": f"u{index // 3}"}
        for index in range(12)
    ]
    selected = select_capped(list(reversed(rows)), 8, group_key="uploader", max_per_group=2)
    assert [row["identity"] for row in selected] == ["0", "1", "3", "4", "6", "7", "9", "10"]


def test_commons_metadata_accepts_only_licensed_large_jpeg_originals() -> None:
    def page(pageid: int, **overrides):
        info = {
            "url": f"https://upload.wikimedia.org/file-{pageid}.jpg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:test.jpg",
            "size": 200_000, "width": 2_000, "height": 1_500,
            "mime": "image/jpeg", "sha1": f"{pageid:040x}",
            "timestamp": "2026-01-01T00:00:00Z", "user": "photographer",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"}},
        }
        info.update(overrides)
        return {"pageid": pageid, "title": f"File:{pageid}.jpg", "imageinfo": [info]}

    payload = {"query": {"pages": [page(1), page(2, mime="image/png"),
                                     page(3, extmetadata={})]}}
    rows = commons_rows("Category:Taken with Camera", [payload])
    assert len(rows) == 1
    assert rows[0]["identity"].startswith("commons:1:")


def test_aigc_coordinates_are_selected_without_image_payloads() -> None:
    rows = [("a.parquet", index, 1, AIGC_GENERATOR_CODE) for index in range(6)]
    rows += [("b.parquet", 0, 0, 0), ("b.parquet", 1, 1, 13)]
    selected = select_aigc_coordinates(rows, reserve=5)
    assert len(selected) == 5
    assert {row["generator_code"] for row in selected} == {AIGC_GENERATOR_CODE}


def test_aigc_parquet_reader_requests_only_metadata_columns(tmp_path, monkeypatch) -> None:
    class Column:
        def __init__(self, values):
            self.values = values

        def to_pylist(self):
            return self.values

    class Table:
        def column(self, name):
            return Column({"label": [1, 0], "generator": [AIGC_GENERATOR_CODE, 0]}[name])

    requested = []

    def fake_read_table(path, *, columns):
        requested.append((path, columns))
        return Table()

    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, "read_table", fake_read_table)
    coordinates, total = read_aigc_coordinates([tmp_path / "one.parquet"])
    assert requested == [(tmp_path / "one.parquet", ["label", "generator"])]
    assert total == 2
    assert coordinates[0] == ("one.parquet", 0, 1, AIGC_GENERATOR_CODE)
