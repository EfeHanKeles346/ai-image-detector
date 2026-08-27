from __future__ import annotations

import io

from PIL import Image

from experiments.e40_robustness import robustness_gate, transport_image


def _transport_result(*, auc: float = 0.9, real_macro: float = 0.1) -> dict:
    return {
        "metrics": {
            "coverage": 1.0,
            "roc_auc": auc,
            "tpr_at_fpr": {"tpr": 0.9},
            "balanced_accuracy": 0.9,
        },
        "source_rates": {
            "real_macro_fp": real_macro,
            "real_worst_device_fp": 0.2,
            "ai_macro_recall": 0.9,
            "ai_worst_family_recall": 0.8,
        },
        "score_summary": {"real": {"mean": 0.2}, "ai": {"mean": 0.8}},
    }


def test_transport_contract_is_deterministic() -> None:
    image = Image.new("RGB", (400, 200), (20, 80, 140))
    first = transport_image(image, "resize75_q50")
    second = transport_image(image, "resize75_q50")
    assert first.size == (300, 150)
    a, b = io.BytesIO(), io.BytesIO()
    first.save(a, format="PNG")
    second.save(b, format="PNG")
    assert a.getvalue() == b.getvalue()


def test_robustness_gate_requires_every_population() -> None:
    transports = {
        "native": _transport_result(),
        "jpeg_q50": _transport_result(),
        "resize75_q50": _transport_result(),
    }
    agreements = {
        name: {"real": 0.9, "ai": 0.9}
        for name in transports
    }
    assert robustness_gate(transports, agreements, owner_fp=0.1)["passed"] is True
    transports["jpeg_q50"] = _transport_result(auc=0.7)
    assert robustness_gate(transports, agreements, owner_fp=0.1)["passed"] is False
    transports["jpeg_q50"] = _transport_result()
    assert robustness_gate(transports, agreements, owner_fp=0.3)["passed"] is False
