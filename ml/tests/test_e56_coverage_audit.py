import pytest
from experiments.e56_coverage_audit import summarize


def fixture():
    rows = [{'parent_id': str(i), 'label': i % 2, 'source': source, 'native': False,
             'width': 224, 'height': 224} for i, source in enumerate(('a', 'b', 'rr:real', 'rr:ai'))]
    facts = [{k: r[k] for k in ('parent_id', 'label', 'source', 'native')} |
             {'near_monochrome_global_crop': i == 2} for i, r in enumerate(rows)]
    folds = [{'fold': 0, 'roles': ['FIT', 'FIT', 'VALIDATION', 'VALIDATION']}]
    return rows, facts, folds


def test_counts_and_missing_metadata_are_explicit():
    result = summarize(*fixture())
    assert result['single_class_publishers'] == ['a', 'b']
    assert result['parents_in_single_class_publishers'] == 2
    assert result['classes']['0']['monochrome'] == 1
    assert result['classes']['0']['scene_metadata_present'] == 0
    assert result['classes']['1']['exact224'] == 2


def test_rejects_bad_join_or_metadata():
    rows, facts, folds = fixture()
    facts[0]['label'] = 1
    with pytest.raises(ValueError): summarize(rows, facts, folds)
    with pytest.raises(ValueError): summarize(rows, facts[:-1], folds)


def test_rejects_publisher_cross_role_leakage():
    rows, facts, folds = fixture()
    folds[0]['roles'][-1] = 'FIT'
    with pytest.raises(ValueError, match='publisher crosses'): summarize(rows, facts, folds)
