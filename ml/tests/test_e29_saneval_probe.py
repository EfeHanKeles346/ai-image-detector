from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e29_saneval_2025_probe.py"
SPEC = importlib.util.spec_from_file_location("e29_saneval_2025_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e29 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e29
SPEC.loader.exec_module(e29)


def source_rows(per_group: int = 3):
    rows = []
    row_idx = 0
    for model in (*e29.TARGET_MODELS, "Imagen 3.0"):
        for prompt_type in e29.TARGET_TYPES:
            for split in e29.TARGET_SPLITS:
                for _ in range(per_group):
                    rows.append(
                        e29.SourceRow(
                            row_idx=row_idx,
                            model=model,
                            split=split,
                            prompt_type=prompt_type,
                            image_url=f"https://example.invalid/{row_idx}.jpg",
                            width=1024,
                            height=1024,
                        )
                    )
                    row_idx += 1
    return rows


def test_frozen_selection_is_balanced_and_excludes_pre_2025_imagen():
    selected = e29.select_rows(source_rows())
    assert len(selected) == 100
    assert len({row.row_idx for row in selected}) == 100
    assert {row.model for row in selected} == set(e29.TARGET_MODELS)
    for model in e29.TARGET_MODELS:
        assert sum(row.model == model for row in selected) == 20
        for prompt_type in e29.TARGET_TYPES:
            for split in e29.TARGET_SPLITS:
                group = [
                    row
                    for row in selected
                    if (row.model, row.prompt_type, row.split) == (model, prompt_type, split)
                ]
                assert len(group) == 2


def test_download_size_gate_is_strict_and_requires_every_row():
    assert e29.validate_content_lengths([999_999] * 100) == 99_999_900
    with pytest.raises(RuntimeError, match="above"):
        e29.validate_content_lengths([1_000_001] * 100)
    with pytest.raises(RuntimeError, match="expected 100"):
        e29.validate_content_lengths([1] * 99)
    with pytest.raises(RuntimeError, match="positive"):
        e29.validate_content_lengths([1] * 99 + [0])
