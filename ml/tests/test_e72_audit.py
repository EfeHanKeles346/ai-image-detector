import pytest
from experiments.e72_audit import resolve


def rows():
    return [{'parent_id':p,'sensor':'A','capture_second':None} for p in ['a','b','c','d','e']]


def link(a,b):return {'train_parent':a,'cal_parent':b}


def test_protected_overlap_propagates_through_transitive_component():
    result=resolve(rows(),[link('a','b'),link('b','c')],{'c','failed_decode'})
    assert result['quarantined']==['a','b','c','failed_decode']
    assert result['kept']==['d','e']
    assert result['lineage']['a']==result['lineage']['c']


def test_internal_dedup_keeps_one_deterministic_representative_without_role_split():
    matches=[link('a','b'),link('b','c')]
    result=resolve(rows(),matches,set())
    assert len(result['kept'])==3 and len(result['duplicate_members_removed'])==2
    assert result==resolve(rows()[::-1],matches[::-1],set())
    assert len(set(result['kept']) & {'a','b','c'})==1


def test_capture_second_links_only_known_same_sensor_times():
    records=rows()
    for r in records[:3]:r['capture_second']='2024:01:01 12:00:00'
    records[2]['sensor']='B'
    result=resolve(records,[],{'a'})
    assert result['quarantined']==['a','b']
    assert result['kept']==['c','d','e']
    with pytest.raises(ValueError,match='unknown'):resolve(records,[link('a','missing')],set())
    with pytest.raises(ValueError,match='duplicate'):resolve(records+[records[0]],[],set())
