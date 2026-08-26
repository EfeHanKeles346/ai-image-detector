from __future__ import annotations

from pixelproof.e32_cfvit_candidate import ARTIFACT_SHA256, MODEL_REVISION, MODEL_WEIGHT_SHA256


def test_frozen_r1a_identities_are_full_hashes() -> None:
    assert len(ARTIFACT_SHA256) == 64
    assert len(MODEL_WEIGHT_SHA256) == 64
    assert len(MODEL_REVISION) == 40
