from __future__ import annotations

from copy import deepcopy
import tarfile

import pytest

from experiments.e33_rrdataset import FILES, RECORD_ID, inspect_members, validate_record


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


def _member(name: str, size: int = 10) -> tarfile.TarInfo:
    item = tarfile.TarInfo(name)
    item.size = size
    item.type = tarfile.REGTYPE
    return item


def test_cal_archive_inventory_requires_explicit_split_and_class() -> None:
    root = "RRDataset_original_train_val"
    result = inspect_members(
        [
            _member(f"{root}/train/real/a.jpg"),
            _member(f"{root}/train/ai/b.png"),
            _member(f"{root}/val/real/c.jpeg"),
            _member(f"{root}/val/ai/d.png"),
        ],
        role="cal",
    )
    assert result["image_count"] == 4
    assert result["by_split_class"] == {
        "train/ai": 1,
        "train/real": 1,
        "val/ai": 1,
        "val/real": 1,
    }


def test_archive_inventory_rejects_traversal_and_links() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        inspect_members([_member("RRDataset_original_train_val/../escape.jpg")], role="cal")
    link = tarfile.TarInfo("RRDataset_original_train_val/val/real/link.jpg")
    link.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="unsupported"):
        inspect_members([link], role="cal")


def test_test_archive_inventory_requires_declared_condition_and_class() -> None:
    root = "RRDataset_test"
    result = inspect_members(
        [
            _member(f"{root}/original/real_images/real_000001.jpg"),
            _member(f"{root}/original/ai_images/normal_000001.png"),
            _member(f"{root}/transfer/real_images/transfer_real_000001.png"),
            _member(f"{root}/redigital/ai_images/redigital_normal_000001.jpg"),
        ],
        role="test",
    )
    assert result["image_count"] == 4
    assert result["by_split_class"] == {
        "original/ai": 1,
        "original/real": 1,
        "redigital/ai": 1,
        "transfer/real": 1,
    }
