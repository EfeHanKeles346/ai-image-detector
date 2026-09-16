import copy
import pytest
from experiments.e140_admission_chain import reconcile


def fixture():
    base = [dict(parent_id='old', sha256='a', source='old', label=1, role='TRAIN')]
    new = [dict(parent_id='new', sha256='b', source='camera', label=0, role='TRAIN')]
    stage = dict(contract={'reference_files':['MNW','HDR'],'inputs':{'MNW':'m','HDR':'h'}},
        manifest={'rows':new,'model_scores':0},
        report={'records':new,'admitted':1,'model_scores':0,'cross_matches':[]})
    return base+new, base, {'audit':stage}, {'MNW':'m','HDR':'h'}


def test_current_membership_is_covered_without_promoting_a_final_test():
    r = reconcile(*fixture())
    assert r['historical_admission_coverage_complete'] is True
    assert r['later_admitted_parents'] == 1 and r['new_independent_test_admitted'] is False


@pytest.mark.parametrize('change', ['missing_reference','changed_reference','uncovered_parent','wrong_identity','overlap','cross_match','wrong_role'])
def test_broken_admission_chain_cannot_pass(change):
    active, base, stages, hashes = copy.deepcopy(fixture()); item = stages['audit']
    if change == 'missing_reference': item['contract']['reference_files'].remove('MNW')
    elif change == 'changed_reference': item['contract']['inputs']['MNW'] = 'different'
    elif change == 'uncovered_parent': active.append(dict(parent_id='extra',sha256='z',source='extra',label=0,role='TRAIN'))
    elif change == 'wrong_identity': active[1] = dict(active[1], sha256='changed')
    elif change == 'overlap': base.append(copy.deepcopy(active[1]))
    elif change == 'cross_match': item['report']['cross_matches'] = [{'match':'body'}]
    elif change == 'wrong_role': active[1] = dict(active[1], role='TEST')
    with pytest.raises(ValueError): reconcile(active, base, stages, hashes)
