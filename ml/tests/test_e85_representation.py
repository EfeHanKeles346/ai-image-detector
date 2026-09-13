import numpy as np
import pytest
from experiments import e85_representation as m,e82_representation as old


def test_four_view_weights_keep_parent_and_class_mass_with_same_recipe():
    rows=[{'label':0,'source':'A','parent_id':'a'},{'label':0,'source':'A','parent_id':'b'},
          {'label':0,'source':'B','parent_id':'c'},{'label':1,'source':'AI','parent_id':'d'}]
    baseline=np.zeros(16);baseline[:4]=1;labels,w=m.loss_weights(rows,baseline)
    assert w.mean()==pytest.approx(1) and w[labels==0].sum()==pytest.approx(w[labels==1].sum())
    assert w[0]==pytest.approx(2*w[4]) and w[8]==pytest.approx(2*w[4])
    assert m.build_network is old.build_network and m.epoch_order is old.epoch_order
    with pytest.raises(ValueError,match='aligned'):m.loss_weights(rows,baseline[:-1])


def test_input_layout_guard_rejects_tiny_decision_changes():
    cut=m.input_model.AI_CUT
    assert m.check_old_scores(np.array([.2,.8]),np.array([.2+1e-8,.8]))['passed']
    with pytest.raises(ValueError,match='decisions'):
        m.check_old_scores(np.array([cut-1e-9]),np.array([cut+1e-9]))
