from __future__ import annotations

from experiments.e35_dda_development import development_gate


def test_official_dda_development_gate_requires_every_population() -> None:
    passed = development_gate(
        rr_auc=0.90,
        rr_real_fp=0.10,
        rr_ai_macro=0.85,
        rr_ai_worst=0.65,
        ipn_worst_fp=0.20,
        owner_fp=0.20,
    )
    assert passed["passed"]
    failed = development_gate(
        rr_auc=0.90,
        rr_real_fp=0.10,
        rr_ai_macro=0.85,
        rr_ai_worst=0.65,
        ipn_worst_fp=0.20,
        owner_fp=0.21,
    )
    assert not failed["passed"]
    assert not failed["checks"]["owner_fp_lte_0.20"]
