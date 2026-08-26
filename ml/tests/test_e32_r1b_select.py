from __future__ import annotations

from experiments.e32_r1b_select import choose


def _arm(name: str, auc: float, c_value: float, passed: bool = True) -> dict:
    return {"arm": name, "metrics": {"auc": auc}, "selected_c": c_value, "screen_gate": {"passed": passed}}


def test_selection_prefers_auc_then_smaller_c_then_dino() -> None:
    assert choose([_arm("dino", 0.9, 1), _arm("cf", 0.91, 10)])["arm"] == "cf"
    assert choose([_arm("dino", 0.9, 0.1), _arm("cf", 0.9, 1)])["arm"] == "dino"
    assert choose([_arm("cf", 0.9, 0.1), _arm("dino", 0.9, 0.1)])["arm"] == "dino"
