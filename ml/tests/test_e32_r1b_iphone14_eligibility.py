from __future__ import annotations

from experiments.e32_r1b_iphone14_eligibility import duplicate_component_keys


def test_duplicate_component_exclusion_takes_every_side() -> None:
    assert duplicate_component_keys([["a", "b"], ["c", "d", "e"]]) == {"a", "b", "c", "d", "e"}
