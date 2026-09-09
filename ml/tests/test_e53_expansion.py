import pytest
from experiments.e53_expansion import select_rows


def rows(label='real',count=1100):
    return [{'record_id':str(i),'source_id':'source','role_group':f'g{i%5}',
             'role':'TRAIN','label':label} for i in range(count)]


def test_source_budget_and_deterministic_group_coverage():
    data=rows();allowed={r['record_id'] for r in data}
    selected=select_rows(data,allowed,[])
    assert len(selected)==1000 and len({r['role_group'] for r in selected})==5
    assert selected==select_rows(list(reversed(data)),allowed,[])
    assert len(select_rows(rows('ai'),allowed,[]))==500


def test_both_internal_duplicate_endpoints_are_excluded():
    data=rows(count=100);allowed={r['record_id'] for r in data}
    selected=select_rows(data,allowed,[{'train_parent':'e32:0','cal_parent':'e32:1'}])
    assert len(selected)==98
    assert not {'0','1'}&{r['record_id'] for r in selected}


def test_non_train_and_depleted_sources_stop():
    data=rows(count=100);data[0]['role']='CAL'
    with pytest.raises(ValueError):select_rows(data,{r['record_id'] for r in data},[])
    with pytest.raises(ValueError):select_rows(rows(count=49),{str(i) for i in range(49)},[])
