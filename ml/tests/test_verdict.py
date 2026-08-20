# =============================================================================
# test_verdict.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Unit tests for the decision layer's pure logic: the asymmetric band (E23a)
# and the megapixel input policy (E23b). No model is loaded — these guard the
# contract, not the weights.
# =============================================================================
import numpy as np
from PIL import Image

from pixelproof.verdict import CAP_PX, capped, decide


def test_band_is_asymmetric():
    # Above the line: AI. Below it: insufficient — never "real" (E23a).
    assert decide(5.0, 1.0) == "ai"
    assert decide(1.0, 1.0) == "ai"          # ties go to the threshold rule >=
    assert decide(0.99, 1.0) == "insufficient"
    assert decide(-10.0, 1.0) == "insufficient"


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
