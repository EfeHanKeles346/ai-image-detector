import numpy as np
from experiments import e86_gates as m
from experiments.e76_fit import real_slice_gates
from experiments.e80_fit import full_training_gates


def test_fourth_condition_cannot_mask_old_condition_failure():
    rows=[{'label':0,'source':'A','parent_id':'a'},{'label':1,'source':'AI','parent_id':'b'},
          {'label':0,'source':'MIDD:S','parent_id':'c'}]
    scores=np.array([[.99,0,0,0],[.99,.99,.99,.99],[0,0,0,0]])
    new=m.population_gates(rows,scores,2);old=real_slice_gates(rows,scores[:,:3],2)
    assert {k:new[k] for k in list(new)[:3]}==old
    assert not new['clean']['passed'] and new['social_q75']['passed']
    expanded=m.full_training_gates(rows,scores,2)
    assert {k:expanded[k] for k in list(expanded)[:3]}==full_training_gates(rows,scores[:,:3],2)
    assert not expanded['clean']['old_TRAIN']['gate']['passed']
    assert expanded['social_q75']['old_TRAIN']['gate']['passed']
