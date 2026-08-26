from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_ai_inventory.py"
SPEC = importlib.util.spec_from_file_location("e32_ai_inventory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def test_registry_does_not_treat_missing_licence_as_admissible_family():
    sources = e32.registry()
    assert len({source["id"] for source in sources}) == len(sources)
    assert all(
        source["license"] is not None
        for source in sources
        if source["counts_toward_verified_modern_families"]
    )


def test_parquet_loose_and_zip_inventory_are_physical(tmp_path, monkeypatch):
    parquet_folder = tmp_path / "parquet"
    parquet_folder.mkdir()
    pq.write_table(pa.table({"image": [b"a", b"b"], "prompt": ["x", "y"]}), parquet_folder / "data.parquet")

    loose_folder = tmp_path / "loose"
    loose_folder.mkdir()
    (loose_folder / "one.png").write_bytes(b"png")
    (loose_folder / "one.txt").write_text("prompt")
    (loose_folder / "._one.png").write_bytes(b"appledouble")

    zip_folder = tmp_path / "zip"
    zip_folder.mkdir()
    with zipfile.ZipFile(zip_folder / "AI.zip", "w") as archive:
        archive.writestr("a.jpg", b"jpeg")
        archive.writestr("notes.txt", b"notes")

    specs = [
        {
            "id": "p", "repo_id": "p", "revision": "a" * 40, "dirname": "parquet",
            "storage": "parquet", "family": "p", "license": "MIT", "provenance": "p",
            "admission_state": "eligible", "counts_toward_verified_modern_families": True,
        },
        {
            "id": "l", "repo_id": "l", "revision": "b" * 40, "dirname": "loose",
            "storage": "loose_images_with_text_sidecars", "family": "l", "license": "MIT",
            "provenance": "l", "admission_state": "eligible",
            "counts_toward_verified_modern_families": True,
        },
        {
            "id": "z", "repo_id": "z", "revision": "c" * 40, "dirname": "zip",
            "storage": "zip", "family": "z", "license": None,
            "provenance": "z", "admission_state": "conditional",
            "counts_toward_verified_modern_families": False,
        },
    ]
    monkeypatch.setattr(e32, "registry", lambda: specs)
    report = e32.build_report(tmp_path)
    by_id = {source["id"]: source for source in report["sources"]}
    assert by_id["p"]["row_count"] == 2
    assert by_id["l"]["loose_image_count"] == 1
    assert by_id["l"]["image_with_matching_sidecar_count"] == 1
    assert by_id["z"]["zip_image_members"] == 1
    assert report["verified_admissible_modern_family_count"] == 2
    assert report["family_gap"] == 3


def test_report_write_is_atomic_json(tmp_path):
    output = tmp_path / "evidence" / "report.json"
    e32.write_report({"ok": True}, output)
    assert json.loads(output.read_text()) == {"ok": True}
