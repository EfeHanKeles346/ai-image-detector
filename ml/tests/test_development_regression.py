import pytest
from pixelproof.development_regression import compare_development


def population():
    return [{'parent_id':str(i), 'condition':'original', 'label':i//2,
             'source':'owner' if i<2 else 'generator', 'role':'CONSUMED_DEVELOPMENT',
             'sha256':str(i)*64,'score':s} for i,s in enumerate([.01,.9,.9,.01])]


def test_reference_passes_without_implying_promotion():
    result=compare_development(population(),list(reversed(population())),.1)
    assert result['passes_paired_regression'] and not result['promotion_allowed']


def test_equal_aggregate_counts_cannot_hide_real_or_ai_regressions():
    old=population();new=[r|{'score':old[1-i]['score'] if i<2 else old[5-i]['score']} for i,r in enumerate(old)]
    result=compare_development(old,new,.1)
    assert not result['passes_paired_regression']
    assert result['new_real_false_ai']==1 and result['newly_missed_ai']==1


@pytest.mark.parametrize('change',[{'score':float('nan')},{'label':1},{'sha256':'f'*64},{'source':'changed'},{'role':'TRAIN'}])
def test_invalid_or_relabelled_candidate_is_rejected(change):
    old=population();new=[r.copy() for r in old];new[0].update(change)
    with pytest.raises(ValueError):compare_development(old,new,.1)


def test_missing_or_duplicate_views_fail_closed():
    old=population()
    for new in [old[:-1],old+[old[0]]]:
        with pytest.raises(ValueError):compare_development(old,new,.1)
