from __future__ import annotations

from experiments.e49_evaluation import SOURCE_COUNTS


def test_final_source_contract_is_exactly_balanced_two_thousand_parents():
    assert sum(SOURCE_COUNTS.values()) == 2_000
    assert sum(count == 100 for count in SOURCE_COUNTS.values()) == 10
    assert sum(count == 160 for count in SOURCE_COUNTS.values()) == 5
    assert SOURCE_COUNTS["StyleGAN2"] == 200
