from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e33_rrdataset import FILES, RECORD_ID, validate_record


def _record() -> dict:
    return {
        "id": RECORD_ID,
        "metadata": {"license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": item["name"],
                "size": item["bytes"],
                "checksum": f"md5:{item['md5']}",
                "links": {
                    "self": f"https://zenodo.org/api/records/{RECORD_ID}/files/{item['name']}/content"
                },
            }
            for item in FILES.values()
        ],
    }


def test_rrdataset_record_binds_both_roles_and_label_boundary() -> None:
    selected = validate_record(_record())
    assert selected["cal"]["role"] == "r1c_cal_candidate"
    assert selected["test"]["role"] == "locked_final_test"
    assert selected["test"]["bytes"] > selected["cal"]["bytes"]


def test_rrdataset_record_rejects_licence_drift() -> None:
    payload = _record()
    payload["metadata"]["license"]["id"] = "other"
    with pytest.raises(ValueError, match="licence"):
        validate_record(payload)


def test_rrdataset_record_rejects_checksum_drift() -> None:
    payload = deepcopy(_record())
    payload["files"][0]["checksum"] = "md5:" + "0" * 32
    with pytest.raises(ValueError, match="contract changed"):
        validate_record(payload)
