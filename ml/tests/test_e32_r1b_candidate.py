from __future__ import annotations

from pixelproof.e32_r1b_candidate import ARTIFACT_SHA256


def test_r1b_candidate_artifact_identity_is_full_sha() -> None:
    assert len(ARTIFACT_SHA256) == 64
