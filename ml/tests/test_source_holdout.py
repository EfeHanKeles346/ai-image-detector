import numpy as np
import pytest
from pixelproof.source_holdout import components,outer_folds,balanced_weights


def row(p,source,label=0,**extra):
    return {'parent_id':p,'source':source,'role':'TRAIN','label':label,**extra}


def test_shared_prompts_and_known_families_form_transitive_component():
    rows=[row('a','e32:flux2-klein-9b',1),row('b','e32:qwen-image-2512',1),
          row('c','e36:FLUX.2_max',1),row('d','e36:gpt-image-2',1),row('e','e32:gpt-image-1',1)]
    assert len(set(components(rows,{}).values()))==1


def test_topics_and_sensors_are_not_independent_publishers():
    rows=[row('a','rr:topic_a',1),row('b','rr:topic_b',1),row('c','rr:rrdataset_real_pool'),
          row('d','MIDD:sensor1'),row('e','MIDD:sensor2'),row('f','SID:Sony'),row('g','SID:Fuji')]
    g=components(rows,{})
    assert g['a']==g['b']==g['c']
    assert g['d']==g['e'] and g['f']==g['g']
    assert len(set(g.values()))==3


def test_inherited_scene_and_body_edges_survive_new_publishers():
    rows=[row('a','A'),row('b','B',scene_group='scene'),row('c','C',scene_group='scene',original_sha256='same'),
          row('d','D',sha256='same')]
    g=components(rows,{'a':'prior','b':'prior'})
    assert len(set(g.values()))==1


def test_every_component_has_one_outer_fold_and_both_classes():
    rows=[row(str(i),str(i),i%2) for i in range(12)]
    g=components(rows,{})
    folds=outer_folds(rows,g)
    assert len(folds)==12
    for f in range(3):
        assert {r['label'] for r in rows if folds[r['parent_id']]==f}=={0,1}
        assert {r['label'] for r in rows if folds[r['parent_id']]!=f}=={0,1}


def test_fold_failure_cannot_be_fixed_by_splitting_connected_ai_group():
    rows=[row('a','A'),row('b','B'),row('c','C'),row('d','rr:topic1',1),row('e','rr:topic2',1)]
    with pytest.raises(ValueError):outer_folds(rows,components(rows,{}))


def test_class_component_parent_view_weights_do_not_reward_extra_sensors():
    rows=[row('a','A'),row('b','A'),row('c','B'),row('d','C',1)]
    g=components(rows,{})
    w=balanced_weights(rows,g,4).reshape(4,4)
    np.testing.assert_allclose(w.sum(axis=1),[.125,.125,.25,.5])
    assert w.sum()==1


@pytest.mark.parametrize('bad',[row('a','A',role='FINAL'),row('a','A',True)])
def test_protected_or_ambiguous_roles_are_rejected(bad):
    with pytest.raises(ValueError):components([bad],{})
