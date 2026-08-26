from __future__ import annotations

from experiments.e32_r1b_external import external_gate


def test_external_gate_requires_every_real_boundary_and_ai_retention() -> None:
    assert external_gate({"a": 0.1, "b": 0.2}, 0.2, 0.9)["passed"]
    assert not external_gate({"a": 0.21}, 0.1, 0.9)["passed"]
    assert not external_gate({"a": 0.1}, 0.21, 0.9)["passed"]
    assert not external_gate({"a": 0.1}, 0.1, 0.89)["passed"]
