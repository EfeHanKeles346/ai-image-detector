from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_gpt_acquisition.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_gpt_acquisition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


@pytest.mark.parametrize("key", ("../escape.png", "/absolute.png", "safe/../../escape.png"))
def test_safe_join_rejects_escape(tmp_path, key):
    with pytest.raises(ValueError):
        e32._safe_join(tmp_path, key)


def test_selection_loader_rejects_changed_record_hash(tmp_path, monkeypatch):
    payload = {
        "state": "frozen_ai_parent_selection_before_remaining_bytes_or_decode",
        "selection_sha256": "wrong",
        "sources": [
            {
                "source_id": "gpt-image-1",
                "family": "GPT",
                "records": [{"source_key": "image.png"}],
            }
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(e32.pool, "DETAILED_SELECTION", path)
    with pytest.raises(ValueError, match="SHA mismatch"):
        e32._load_selection()


def test_bulk_requires_current_smoke(tmp_path, monkeypatch):
    smoke = tmp_path / "smoke.json"
    monkeypatch.setattr(e32, "SMOKE_EVIDENCE", smoke)
    payload = {"selection_sha256": "current"}
    with pytest.raises(PermissionError, match="missing"):
        e32._require_smoke(payload)
    smoke.write_text(
        json.dumps({"state": "gpt_decoder_smoke_passed", "selection_sha256": "old"})
    )
    with pytest.raises(PermissionError, match="different"):
        e32._require_smoke(payload)


def test_available_asset_prefers_matching_local_file(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    image = local / "image.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(e32, "_local_asset", lambda root, key: local / key)
    monkeypatch.setattr(e32, "_e32_asset", lambda key: tmp_path / "e32" / key)
    assert e32._available_asset(tmp_path, "image.png", 5) == image
    assert e32._available_asset(tmp_path, "image.png", 6) is None
