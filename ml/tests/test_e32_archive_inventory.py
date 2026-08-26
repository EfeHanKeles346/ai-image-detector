from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_archive_inventory.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_archive_inventory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def test_zip_inventory_records_physical_members_and_hash(tmp_path):
    path = tmp_path / "safe.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("D01_Phone/orig/D01_img_orig_0001.jpg", b"jpeg")
    result = e32._inventory_zip(path, path.stat().st_size)
    assert result["file_members"] == 1
    assert result["uncompressed_bytes"] == 4
    assert result["suffix_counts"] == {".jpg": 1}
    assert len(result["archive_sha256"]) == 64


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "safe\\escape.jpg"])
def test_zip_member_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="unsafe"):
        e32._validate_member(ZipInfo(name))


def test_zip_member_rejects_symlink():
    info = ZipInfo("link.jpg")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symlink"):
        e32._validate_member(info)


def test_fodb_member_contract_links_transport_and_scene():
    match = e32.FODB_MEMBER.fullmatch(
        "D01_Motorola_E3_1/orig/D01_img_orig_0063.jpg"
    )
    assert match is not None
    assert match.group("pipeline") == "D01_Motorola_E3_1"
    assert match.group("transport") == "orig"
    assert match.group("scene") == "0063"


def test_fodb_excludes_only_the_precommitted_inspection_root():
    assert e32.FODB_EXCLUDED_ROOTS == {"inspection"}
    assert e32.FODB_MEMBER.fullmatch("inspection/check_devices/helper.jpg") is None
    assert e32.FODB_MEMBER.fullmatch("unknown/helper.jpg") is None
