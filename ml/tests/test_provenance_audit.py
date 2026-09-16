import pytest
from pixelproof.provenance_audit import audit


def test_duplicate_body_aliases_reveal_cross_label_and_fold_link_without_opening_images():
    rows = [dict(parent_id='a', source='one', label=0, role='TRAIN', sha256='1'*64, path='/does/not/exist'),
            dict(parent_id='b', source='two', label=1, role='TRAIN', sha256='2'*64, original_sha256='1'*64)]
    r = audit(rows, {'a':'x','b':'y'}, {'a':0,'b':1})
    assert r['stored_identity_links']['body']['cross_fold_groups'] == 1
    assert r['stored_identity_links']['body']['cross_label_groups'] == 1
    assert r['pixels_read'] == 0 and r['independent_test_parents_created'] == 0


def test_body_and_pixel_domains_are_not_conflated_and_missing_provenance_is_not_certified():
    rows = [dict(parent_id='a', source='rr:topic', label=1, role='TRAIN', sha256='1'*64),
            dict(parent_id='b', source='camera', label=0, role='TRAIN', sha256='2'*64, pixel_sha256='1'*64)]
    r = audit(rows, {'a':'x','b':'y'}, {'a':0,'b':1})
    assert r['stored_identity_links']['body']['repeated_identity_groups'] == 0
    assert r['stored_identity_links']['pixel']['repeated_identity_groups'] == 0
    assert r['mixed_corpus_AI_rows_without_explicit_generator_field'] == 1
    assert r['classes']['1']['scene_independence_verified'] == 0


def test_training_only_and_component_integrity():
    rows = [dict(parent_id='a', source='one', label=0, role='TRAIN', sha256='1'*64),
            dict(parent_id='b', source='one', label=0, role='TRAIN', sha256='2'*64)]
    assert audit(rows, {'a':'x','b':'x'}, {'a':0,'b':1})['components_crossing_folds'] == 1
    rows[0]['role'] = 'FINAL'
    with pytest.raises(ValueError): audit(rows, {'a':'x','b':'x'}, {'a':0,'b':1})
