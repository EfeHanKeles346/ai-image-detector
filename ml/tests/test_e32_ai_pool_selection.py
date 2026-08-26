from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_ai_pool_selection.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_ai_pool_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def test_registry_freezes_15k_and_five_current_families():
    payload = e32.registry()
    assert payload["target_parents"] == 15_000
    assert sum(source["target"] for source in payload["sources"]) == 15_000
    assert max(source["target"] for source in payload["sources"]) == 3_000
    assert sum(source["counts_as_current_family"] for source in payload["sources"]) == 5


def test_stable_selection_is_independent_of_input_order():
    rows = [{"id": str(index)} for index in range(20)]
    forward = e32._stable_select(
        rows, source_id="source", key_field="id", seed=7, limit=8
    )
    backward = e32._stable_select(
        list(reversed(rows)), source_id="source", key_field="id", seed=7, limit=8
    )
    assert forward == backward


def test_round_robin_spreads_rows_across_model_identities():
    rows = [
        {"shard": "a", "row_index": row, "model_name": model}
        for model in ("m1", "m2", "m3")
        for row in range(5)
    ]
    selected = e32._model_round_robin(
        rows, source_id="community", seed=11, limit=8
    )
    counts = __import__("collections").Counter(row["model_name"] for row in selected)
    assert set(counts) == {"m1", "m2", "m3"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_gpt_selection_does_not_depend_on_which_pairs_are_local(tmp_path):
    siblings = []
    for index in range(6):
        for suffix, size in (("png", 100 + index), ("txt", 10 + index)):
            siblings.append(
                SimpleNamespace(rfilename=f"GPTIMG_{index:02d}.{suffix}", size=size)
            )
    info = SimpleNamespace(
        sha="a" * 40, tags=["license:cc-by-4.0"], siblings=siblings
    )
    spec = {
        "id": "gpt",
        "repo_id": "fixture/repo",
        "revision": "a" * 40,
        "license_tag": "license:cc-by-4.0",
        "expected_available_pairs": 6,
        "local_dirname": "gpt-local",
        "target": 4,
        "family": "GPT",
        "counts_as_current_family": True,
        "selection_kind": "huggingface_stable_hash_pairs",
    }
    first = e32._freeze_gpt(tmp_path, spec, 9, lambda repo, revision: info)
    selected_keys = [row["source_key"] for row in first["records"]]
    local = tmp_path / "gpt-local"
    local.mkdir()
    for row in first["records"][:2]:
        image = local / row["source_key"]
        prompt = local / row["prompt_key"]
        image.write_bytes(b"x" * row["expected_image_bytes"])
        prompt.write_bytes(b"x" * row["expected_prompt_bytes"])
    second = e32._freeze_gpt(tmp_path, spec, 9, lambda repo, revision: info)
    assert [row["source_key"] for row in second["records"]] == selected_keys
    assert second["selected_pairs_already_local"] == 2


def test_gpt_freeze_rejects_changed_licence(tmp_path):
    info = SimpleNamespace(sha="a" * 40, tags=[], siblings=[])
    spec = {
        "id": "gpt",
        "repo_id": "fixture/repo",
        "revision": "a" * 40,
        "license_tag": "license:cc-by-4.0",
        "expected_available_pairs": 0,
        "local_dirname": "gpt-local",
        "target": 0,
        "family": "GPT",
        "counts_as_current_family": True,
        "selection_kind": "huggingface_stable_hash_pairs",
    }
    with pytest.raises(ValueError, match="licence"):
        e32._freeze_gpt(tmp_path, spec, 9, lambda repo, revision: info)
