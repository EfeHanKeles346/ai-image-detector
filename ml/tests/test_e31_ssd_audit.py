from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "experiments/e31_ssd_audit.py"
SPEC = importlib.util.spec_from_file_location("e31_ssd_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e31 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e31
SPEC.loader.exec_module(e31)


def _jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _write_fixture(folder: Path) -> None:
    folder.mkdir(parents=True)
    rows = []
    for index in range(12):
        label = index % 2
        size = (64, 64) if label == 0 else (128, 128)
        rows.append(
            {
                "image_data": {"bytes": _jpeg(size, (index * 10, 80, 120)), "path": None},
                "label": label,
                "model_name": "FFHQ" if label == 0 else f"generator-{index}",
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, folder / "part-000.parquet")
    (folder / "._part-000.parquet").write_bytes(b"AppleDouble")


def test_audit_counts_rows_skips_appledouble_and_marks_sample_scope(tmp_path):
    root = tmp_path / "datasets"
    _write_fixture(root / "OwensLab__CommunityForensics-Small")
    report = e31.build_report(root, tmp_path, sample_limit=10)
    source = next(
        item for item in report["registered_sources"]
        if item["source_id"] == "communityforensics-small"
    )
    assert source["physical_file_count"] == 1
    assert source["parquet_file_count"] == 1
    assert source["parquet_rows"] == 12
    assert source["project_label_counts"] == {"ai": 6, "real": 6}
    assert source["sample_count"] == 10
    assert source["sample_hash_scope"] == "bounded_shard_spread_not_full_dataset"
    assert set(source["native_shortcut_audit"]["groups"]) == {"real", "ai"}


def test_output_cannot_be_written_to_source_disk(tmp_path):
    root = tmp_path / "datasets"
    root.mkdir()
    with pytest.raises(ValueError, match="refusing"):
        e31.write_report({"ok": True}, root / "report.json", root)

    output = tmp_path / "evidence" / "report.json"
    e31.write_report({"ok": True}, output, root)
    assert json.loads(output.read_text()) == {"ok": True}


def test_missing_root_fails_actionably(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset root is unavailable"):
        e31.build_report(tmp_path / "missing", tmp_path)


def test_sample_spreads_over_first_middle_and_last_shards():
    files = [Path(f"part-{index:03d}.parquet") for index in range(293)]
    selected = e31._spread_files(files, 300)
    assert len(selected) == 3
    assert selected[0][0] == files[0]
    assert selected[-1][0] == files[-1]
    assert selected[1][0] == files[146]
    assert sum(per_file for _, per_file in selected) == 300


def test_explicit_folder_role_supplies_label_without_parquet_column():
    spec = next(item for item in e31.SOURCES if item.source_id == "communityforensics-real")
    assert e31._project_label(spec, None) == "real"
    assert e31._validate_label_metadata(spec, object()) == "implicit_real"


def test_protected_hashes_include_parent_and_derived_manifests(tmp_path):
    root = tmp_path / "ml" / "data" / "e30" / "source"
    root.mkdir(parents=True)
    for name, digest in (("manifest.json", "a" * 64), ("derived_manifest.json", "b" * 64)):
        (root / name).write_text(json.dumps({"records": [{"sha256": digest}]}))
    assert e31._known_e30_hashes(tmp_path) == {"a" * 64, "b" * 64}
