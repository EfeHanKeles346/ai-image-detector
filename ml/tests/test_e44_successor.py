from __future__ import annotations

from experiments.e44_successor import EXPECTED_E42, EXPECTED_RR, EXPECTED_ROWS, SUCCESSOR_THRESHOLD
from experiments.e44_fusion import EPSILON


def test_successor_population_arithmetic_and_cut_are_frozen():
    assert EXPECTED_RR == 2_940
    assert EXPECTED_E42 == 2_160
    assert EXPECTED_ROWS == EXPECTED_RR + EXPECTED_E42
    assert SUCCESSOR_THRESHOLD > 0.34238504933894964
    assert EPSILON == 1e-5
