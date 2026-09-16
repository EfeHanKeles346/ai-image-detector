import pytest
from pixelproof.population_audit import compare_populations


def test_body_aliases_and_reencoded_pixels_cannot_hide_overlap():
    a = [dict(parent_id='train', sha256='a'*64, pixel_sha256='c'*64)]
    b = [dict(parent_id='eval', original_sha256='A'*64, sha256='b'*64, pixel_sha256='c'*64)]
    r = compare_populations(a, b)
    assert not r['stored_identity_disjoint']
    assert r['matches']['body']['shared_identity_values'] == 1
    assert r['matches']['pixel']['shared_identity_values'] == 1
    assert r['matches']['parent']['shared_identity_values'] == 0


def test_missing_hashes_are_uncovered_not_verified_independence():
    r = compare_populations([dict(parent_id='a', sha256='a'*64)], [dict(parent_id='b', pixel_sha256='a'*64)])
    assert r['stored_identity_disjoint']
    assert r['left']['pixel_parents'] == r['right']['body_parents'] == 0
    with pytest.raises(ValueError):
        compare_populations([dict(parent_id='a', sha256='bad')], [dict(parent_id='b')])
    with pytest.raises(ValueError):
        compare_populations([dict(parent_id='a'), dict(parent_id='a')], [dict(parent_id='b')])
