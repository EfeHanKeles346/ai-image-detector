from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "experiments/e31_score_folder.py"
SPEC = importlib.util.spec_from_file_location("e31_score_folder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
folder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = folder
SPEC.loader.exec_module(folder)


def test_file_discovery_is_recursive_and_ignores_appledouble(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "nested/b.PNG").write_bytes(b"x")
    (tmp_path / "._a.jpg").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x")
    assert [path.name for path in folder.image_files(tmp_path)] == ["a.jpg", "b.PNG"]


def test_verdict_never_claims_real() -> None:
    assert folder.verdict(True) == "ai_signal_detected"
    assert folder.verdict(False) == "insufficient_evidence"
