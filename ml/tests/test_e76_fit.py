import numpy as np
import pytest
from experiments import e76_fit as m


def test_new_camera_failure_cannot_hide_in_large_old_population():
    rows=[{'label':0,'source':'old'} for _ in range(100)]+[{'label':0,'source':'MIDD:A'} for _ in range(2)]
    scores=np.zeros((102,3));scores[-1]=1
    result=m.real_slice_gates(rows,scores,100)
    for condition in m.CONDITIONS:
        value=result[condition]
        assert value['checks']['expanded_real'] and value['checks']['old_real']
        assert not value['checks']['new_midd_real'] and not value['passed']


def test_pooled_new_camera_pass_cannot_hide_worst_sensor_failure():
    rows=[{'label':0,'source':'old'} for _ in range(100)]
    rows += [{'label':0,'source':'MIDD:A'}]+[{'label':0,'source':'MIDD:B'} for _ in range(19)]
    scores=np.zeros((120,3));scores[100]=1
    value=m.real_slice_gates(rows,scores,100)['clean']
    assert value['checks']['new_midd_real']
    assert not value['checks']['worst_new_sensor_lte_20'] and not value['passed']
    with pytest.raises(ValueError,match='coverage'):m.real_slice_gates(rows,scores[:-1],100)


def test_expansion_keeps_every_existing_ai_and_rejects_overlap_or_role_leakage():
    old=[{'parent_id':f'old:{i}','sha256':f'{i:064x}','label':int(i>=7035),'role':'TRAIN','source':'old'} for i in range(11630)]
    new=[{'parent_id':f'MIDD:{i}','sha256':f'{i+11630:064x}','label':0,'role':'TRAIN','source':'MIDD:A'} for i in range(511)]
    result=m.combine_rows(old,new)
    assert len(result)==12141 and sum(r['label']==1 for r in result)==4595
    for changed in [new[:-1],[new[0]|{'sha256':old[0]['sha256']}]+new[1:],[new[0]|{'role':'DEVELOPMENT'}]+new[1:]]:
        with pytest.raises(ValueError,match='disjoint'):m.combine_rows(old,changed)
