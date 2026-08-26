from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "experiments/e31_qwen_locked.py"
SPEC = importlib.util.spec_from_file_location("e31_qwen_locked", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)


def test_summary_keeps_native_generator_cells_separate() -> None:
    rows = []
    for transport in ("native_source", "standardized_jpeg"):
        for generator in ("a", "b"):
            for predicted in (True, False):
                rows.append({"status": "ok", "transport": transport, "generator": generator,
                             "predicted_ai": predicted})
    report = qwen.summarize(rows)
    assert report["native_recall"]["recall"] == 0.5
    assert report["standardized_minus_native"] == 0.0
    assert set(report["native_per_generator"]) == {"a", "b"}


def test_locked_gate_rejects_nonpassing_development(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"state": "development_failed", "candidate_sha256": "x"}))
    with pytest.raises(PermissionError):
        qwen.load_development_gate(path)
