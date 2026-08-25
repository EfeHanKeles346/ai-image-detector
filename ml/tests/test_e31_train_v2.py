from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io


MODULE_PATH = Path(__file__).parents[1] / "experiments/e31_train_v2.py"
SPEC = importlib.util.spec_from_file_location("e31_train_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e31 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e31
SPEC.loader.exec_module(e31)


def candidate(group: str, index: int) -> object:
    return e31.Candidate(
        source_id="source",
        dirname="source-dir",
        shard=f"source-dir/{group}.parquet",
        row_index=index,
        image_col="image",
        label="ai" if index % 2 else "real",
        generator="generator" if index % 2 else None,
        group_id=group,
    )


def test_even_selection_is_deterministic_and_spreads_groups():
    rows = [candidate(group, index) for group in ("a", "b", "c") for index in range(10)]
    first = e31.evenly_select(rows, 8)
    second = e31.evenly_select(list(reversed(rows)), 8)
    assert [row.key for row in first] == [row.key for row in second]
    assert Counter(row.group_id for row in first) == {"a": 3, "b": 3, "c": 2}


def test_group_fold_never_depends_on_row():
    assert e31.group_fold("same-group") == e31.group_fold("same-group")
    assert 0 <= e31.group_fold("same-group") < e31.N_FOLDS


def test_stratified_group_folds_give_every_source_calibration_support():
    rows = [candidate(f"group-{index}", index) for index in range(7)]
    folds = e31.assign_group_folds(rows)
    assert set(folds.values()) == set(range(5))
    assert folds == e31.assign_group_folds(list(reversed(rows)))


def test_frozen_selection_sha_detects_row_mutation(tmp_path):
    payload = {
        "state": "frozen_selection_before_image_bytes",
        "records": [{"record_id": "one", "label": "ai"}],
    }
    material = json.dumps(payload["records"], sort_keys=True, separators=(",", ":"))
    payload["selection_sha256"] = e31.hashlib.sha256(material.encode()).hexdigest()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    loaded = json.loads(path.read_text())
    loaded["records"][0]["label"] = "real"
    changed = json.dumps(loaded["records"], sort_keys=True, separators=(",", ":"))
    assert e31.hashlib.sha256(changed.encode()).hexdigest() != payload["selection_sha256"]


def test_report_cannot_be_written_inside_source_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError, match="refusing"):
        e31.write_json_atomic({"ok": True}, root / "selection.json", root)


def test_parquet_protection_hashes_embedded_images(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 40, 60)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    folder = tmp_path / "protected"
    folder.mkdir()
    pq.write_table(pa.Table.from_pylist([{"image": {"bytes": raw, "path": None}}]), folder / "x.parquet")
    exact, perceptual, counts = e31._parquet_protected([folder])
    assert e31.hashlib.sha256(raw).hexdigest() in exact
    assert len(perceptual) == 1
    assert counts["protected"] == 1


def test_eligibility_sha_rejects_mutation(tmp_path):
    keys = ["source:a:1", "source:b:2"]
    payload = {
        "state": "mechanical_input_eligibility_before_selection_v2",
        "source_id": "source",
        "eligible_keys": keys,
        "eligible_set_sha256": e31.hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(payload))
    assert e31.load_eligibility([path]) == {"source": set(keys)}
    payload["eligible_keys"].append("tampered")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA mismatch"):
        e31.load_eligibility([path])


def test_rejected_realization_state_is_non_success():
    assert "rejected" != "realized_train_v2"


def test_combined_screen_eligibility_loads_each_source(tmp_path):
    sources = []
    for source_id, keys in (("a", ["a:1"]), ("b", ["b:1", "b:2"])):
        sources.append(
            {
                "source_id": source_id,
                "eligible_keys": keys,
                "eligible_set_sha256": e31.hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            }
        )
    path = tmp_path / "screen.json"
    path.write_text(
        json.dumps(
            {
                "state": "protected_mechanical_eligibility_before_selection_v3",
                "sources": sources,
            }
        )
    )
    assert e31.load_eligibility([path]) == {"a": {"a:1"}, "b": {"b:1", "b:2"}}
