from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e32_r1b_acquisition import (
    CSAFE,
    parse_content_range,
    range_plan,
    select_csafe,
    select_ipn_natural,
)


def _license() -> dict[str, str]:
    return {"name": "CC BY 4.0"}


def _ipn_payload() -> dict:
    files = []
    for index in range(80):
        files.append(
            {
                "id": 1000 + index,
                "name": f"phone_{index:02d} natural Landscape.JPG",
                "size": 100 + index,
                "download_url": f"https://example.test/{index}",
                "supplied_md5": f"{index:032x}",
                "computed_md5": f"{index:032x}",
            }
        )
    files.append(
        {
            "id": 9999,
            "name": "phone_00 flat.JPG",
            "size": 10,
            "download_url": "https://example.test/flat",
            "supplied_md5": "0" * 32,
            "computed_md5": "0" * 32,
        }
    )
    return {"id": 25137734, "version": 1, "license": _license(), "files": files}


def test_ipn_selection_is_natural_only_and_device_bound() -> None:
    rows = select_ipn_natural(_ipn_payload(), 25137734, "iphone-se")
    assert len(rows) == 80
    assert {row["device"] for row in rows} == {"iphone-se"}
    assert all(row["role"] == "development" and " natural " in row["name"].lower() for row in rows)


def test_ipn_selection_rejects_checksum_drift() -> None:
    payload = _ipn_payload()
    payload["files"][0]["computed_md5"] = "f" * 32
    with pytest.raises(ValueError, match="checksum"):
        select_ipn_natural(payload, 25137734, "iphone-se")


def _csafe_payload() -> dict:
    return {
        "id": CSAFE["article_id"],
        "version": CSAFE["version"],
        "license": _license(),
        "files": [
            {
                "id": CSAFE["file_id"],
                "name": CSAFE["name"],
                "size": CSAFE["bytes"],
                "computed_md5": CSAFE["md5"],
                "download_url": "https://example.test/iphone14.zip",
            }
        ],
    }


def test_csafe_selection_binds_published_archive() -> None:
    selected = select_csafe(_csafe_payload())
    assert selected["bytes"] == 20_428_338_922
    assert selected["role"] == "train_cal_candidate"


def test_csafe_selection_rejects_size_drift() -> None:
    payload = deepcopy(_csafe_payload())
    payload["files"][0]["size"] += 1
    with pytest.raises(ValueError, match="contract changed"):
        select_csafe(payload)


def test_range_plan_is_exhaustive_and_disjoint() -> None:
    plan = range_plan(92_274_688, CSAFE["bytes"], workers=4)
    assert len(plan) == 4
    assert plan[0][0] == 92_274_688
    assert plan[-1][1] == CSAFE["bytes"] - 1
    assert all(left_end + 1 == right_start for (_, left_end), (right_start, _) in zip(plan, plan[1:]))


def test_content_range_parser_rejects_non_exact_shape() -> None:
    assert parse_content_range("bytes 10-19/100") == (10, 19, 100)
    with pytest.raises(ValueError, match="invalid Content-Range"):
        parse_content_range("10-19/100")
