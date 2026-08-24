# =============================================================================
# test_verdict.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Unit tests for the decision layer's pure logic: the asymmetric band (E23a)
# and the megapixel input policy (E23b). No model is loaded — these guard the
# contract, not the weights.
# =============================================================================
import numpy as np
from PIL import Image

from pixelproof.predict import official_label
from pixelproof.verdict import CAP_PX, capped, combine, decide, input_caveats


def test_band_is_asymmetric():
    # Above the line: AI. Below it: insufficient — never "real" (E23a).
    assert decide(5.0, 1.0) == "ai"
    assert decide(1.0, 1.0) == "ai"          # ties go to the threshold rule >=
    assert decide(0.99, 1.0) == "insufficient"
    assert decide(-10.0, 1.0) == "insufficient"


def test_or_rule_any_arm_decides():
    # E26: one confident arm is enough — a blind primary cannot veto it.
    assert combine({"cf_vit": "ai", "bfree": "insufficient"}) == ("ai", ["cf_vit"])
    assert combine({"cf_vit": "insufficient", "bfree": "ai"}) == ("ai", ["bfree"])
    assert combine({"cf_vit": "ai", "bfree": "ai"}) == ("ai", ["cf_vit", "bfree"])
    assert combine({"cf_vit": "insufficient", "bfree": "insufficient"}) == (
        "insufficient", [],
    )


def test_cap_only_touches_megapixel_input():
    small = Image.fromarray(np.zeros((500, 700, 3), dtype="uint8"))
    unchanged, was_capped = capped(small)
    assert not was_capped and unchanged.size == (700, 500)

    big = Image.fromarray(np.zeros((3024, 4032, 3), dtype="uint8"))
    shrunk, was_capped = capped(big)
    assert was_capped
    assert max(shrunk.size) == CAP_PX
    # aspect ratio preserved within rounding
    assert abs(shrunk.size[0] / shrunk.size[1] - 4032 / 3024) < 0.01


def test_caveats_describe_only_an_enabled_policy_and_a_coarse_byte_density_flag():
    # A large image does not earn the cap caveat when no B-Free arm consumed
    # the capped copy. Low byte density is still flagged as a heuristic.
    assert input_caveats(0.8, bfree_was_capped=False) == []
    assert input_caveats(0.2, bfree_was_capped=False) == ["sikistirilmis-girdi"]
    assert input_caveats(0.8, bfree_was_capped=True) == ["megapiksel-siniri"]


def test_cli_never_turns_a_research_score_into_a_real_certificate():
    assert official_label({"enough_evidence": True, "decision": {"label": "ai"}}) == "AI detected"
    assert official_label({"enough_evidence": True, "decision": {"label": "insufficient"}}) == "insufficient evidence"
    assert official_label({"enough_evidence": True, "decision": None}) == "insufficient evidence"
    assert official_label({"enough_evidence": False, "decision": {"label": "ai"}}) == "insufficient evidence"
