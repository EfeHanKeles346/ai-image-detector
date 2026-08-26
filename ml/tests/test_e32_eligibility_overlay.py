from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_eligibility_overlay.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_eligibility_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def _row(source, key, unit, label, sha, dhash=None, phash=None):
    independent = hashlib.sha256(f"{source}:{key}".encode()).hexdigest()[:16]
    return {
        "source_id": source,
        "source_key": key,
        "unit_id": unit,
        "label": label,
        "sha256": sha,
        "dhash": dhash or independent,
        "phash": phash or independent,
    }


def test_duplicate_overlay_preserves_groups_and_rejects_cross_label_components():
    rows = [
        _row("ai-a", "a1", "group-a", "ai", "same-internal"),
        _row("ai-a", "a2", "group-a", "ai", "same-internal"),
        _row("ai-b", "b", "group-b", "ai", "same-ai"),
        _row("ai-c", "c", "group-c", "ai", "same-ai"),
        _row("ai-d", "d", "group-d", "ai", "cross-label"),
        _row("real", "r", "group-r", "real", "cross-label"),
    ]
    components = e32._row_duplicate_components(rows)
    excluded, counts = e32._duplicate_unit_exclusions(rows, components)

    assert excluded[("ai-a", "group-a")] == "within_parent_duplicate"
    assert ("ai-b", "group-b") not in excluded
    assert excluded[("ai-c", "group-c")] == "same_label_duplicate_noncanonical"
    assert excluded[("ai-d", "group-d")] == "cross_label_duplicate"
    assert excluded[("real", "group-r")] == "cross_label_duplicate"
    assert counts == {
        "cross_label_duplicate": 2,
        "same_label_duplicate_noncanonical": 1,
        "within_parent_duplicate": 1,
    }


def test_perceptual_duplicate_requires_matching_dhash_and_close_phash():
    rows = [
        _row("a", "1", "1", "ai", "sha1", "f" * 16, "0" * 16),
        _row("b", "2", "2", "ai", "sha2", "f" * 16, "0000000000000003"),
        _row("c", "3", "3", "ai", "sha3", "f" * 16, "f" * 16),
    ]
    components = e32._row_duplicate_components(rows)
    assert components == [[0, 1]]


def test_source_cap_targets_respect_indivisible_four_row_groups():
    sizes = {
        "qwen": [4] * 747,
        "flux": [4] * 729,
        "nano": [1] * 3000,
        "gpt": [1] * 2994,
        "nbp": [1] * 200,
        "community": [1] * 2800,
    }
    targets = e32._max_source_cap_targets(sizes)
    total = sum(targets.values())

    assert targets["qwen"] % 4 == 0
    assert targets["flux"] % 4 == 0
    assert all(5 * count <= total for count in targets.values())
    assert total == 14786
    assert targets == {
        "qwen": 2956,
        "flux": 2916,
        "nano": 2957,
        "gpt": 2957,
        "nbp": 200,
        "community": 2800,
    }
