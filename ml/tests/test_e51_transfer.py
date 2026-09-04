from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from experiments.e51_data_route import DATAPOINT_SHARD_BYTES
from experiments.e51_transfer import (
    ieee_destination,
    inspect_ieee_file,
    inspect_scmi30_file,
    scmi30_destination,
    validate_datapoint_remote,
    validate_selected_archive_members,
)


def _row(size: int = 1) -> dict:
    return {
        "identity": "id",
        "label": 0,
        "role": "DEVELOPMENT",
        "source": "IEEE-SP-Cup-2018-test",
        "transport_cell": "unaltered",
        "remote_path": "test/test/img_0000000_unalt.tif",
        "expected_bytes": size,
    }


def test_ieee_destination_rejects_unsafe_or_wrong_split():
    assert ieee_destination(_row()).name == "img_0000000_unalt.tif"
    for remote in ("../bad.tif", "train/train/a.tif", "/test/test/a.tif", "test/test/a.jpg"):
        row = _row()
        row["remote_path"] = remote
        with pytest.raises(ValueError, match="unsafe IEEE"):
            ieee_destination(row)


def test_ieee_inspection_verifies_tiff_and_bytes(tmp_path: Path):
    path = tmp_path / "image.tif"
    Image.new("RGB", (512, 512), "red").save(path, format="PNG")
    row = _row(path.stat().st_size)
    found = inspect_ieee_file(path, row)
    assert found["format"] == "PNG"
    assert (found["width"], found["height"]) == (512, 512)
    assert len(found["sha256"]) == 64


def test_ieee_inspection_rejects_byte_drift(tmp_path: Path):
    path = tmp_path / "image.tif"
    Image.new("RGB", (512, 512)).save(path, format="PNG")
    with pytest.raises(ValueError, match="byte length"):
        inspect_ieee_file(path, _row(path.stat().st_size + 1))


def test_datapoint_remote_requires_every_exact_shard_byte():
    files = {
        f"data/images/images-{index:04d}.parquet": size
        for index, size in DATAPOINT_SHARD_BYTES.items()
    }
    assert validate_datapoint_remote(files) == files
    files[next(iter(files))] += 1
    with pytest.raises(ValueError, match="inventory changed"):
        validate_datapoint_remote(files)


def test_ieee_archive_members_require_bound_names_and_bytes():
    rows = [_row(100), {**_row(200), "remote_path": "test/test/img_0000001_manip.tif"}]
    members = {
        rows[0]["remote_path"]: (100, 80),
        rows[1]["remote_path"]: (200, 150),
    }
    found = validate_selected_archive_members(members, rows)
    assert found == {
        "selected_files": 2,
        "selected_uncompressed_bytes": 300,
        "selected_compressed_bytes": 230,
    }
    members[rows[1]["remote_path"]] = (201, 150)
    with pytest.raises(ValueError, match="member changed"):
        validate_selected_archive_members(members, rows)


def test_scmi30_destination_and_jpeg_inspection(tmp_path: Path):
    row = {
        "identity": "id", "rank": "rank", "label": 0, "role": "CAL",
        "source": "SCMI30-IITRPR", "device_id": "D01", "device_folder": "D01_Phone",
        "branch": "Random", "remote_path": "SCMI30-IITRPR/Random/D01_Phone/image.jpg",
        "expected_bytes": 0,
    }
    assert scmi30_destination(row).parts[-3:] == ("Random", "D01_Phone", "image.jpg")
    path = tmp_path / "image.jpg"
    Image.new("RGB", (640, 480), "blue").save(path, format="JPEG")
    row["expected_bytes"] = path.stat().st_size
    found = inspect_scmi30_file(path, row)
    assert found["format"] == "JPEG"
    assert (found["width"], found["height"]) == (640, 480)


def test_scmi30_destination_rejects_wrong_root():
    row = {"remote_path": "Other/Random/D01_Phone/image.jpg"}
    with pytest.raises(ValueError, match="unsafe SCMI30"):
        scmi30_destination(row)
