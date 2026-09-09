import numpy as np
from PIL import Image
import pytest

from experiments.e53_offline import ARMS, assign_folds, fixed_write, ordered_views, publisher


def test_shared_publishers_and_prompts_are_not_independent_folds():
    assert publisher('rr:everyday_life') == publisher('rr:rrdataset_real_pool')
    assert publisher('e32:qwen-image-2512') == publisher('e32:flux2-klein-9b')
    assert publisher('e36:gpt-image-2') == publisher('e36:FLUX.2_max')
    assert publisher('e32:csafe-mcsidb-iphone14') == publisher('e32:csafe-mcsidb-s21')


def test_all_six_arms_are_declared_before_results():
    assert len(ARMS)==6
    assert set(ARMS.values()) == {(r,v) for r in ('full','mean') for v in ('old2','e51_3','order3')}


def test_transport_is_deterministic_and_does_not_modify_original():
    im=Image.fromarray(np.random.default_rng(53).integers(0,256,(240,360,3),dtype=np.uint8))
    original=np.asarray(im).copy()
    a=ordered_views(im,'parent'); b=ordered_views(im,'parent')
    assert len(a)==2
    assert np.array_equal(original,np.asarray(im))
    for x,y in zip(a,b,strict=True):
        assert np.array_equal(x,y)
        assert x.mode=='RGB' and min(x.size)>0
    assert a[0].size==a[1].size


def test_source_disjoint_inner_and_outer_folds_keep_all_parents():
    rows=[{'parent_id':f'{label}:{source}:{i}','source':f'source:{label}:{source}','label':label}
          for label in (0,1) for source in range(9) for i in range(100)]
    groups,folds=assign_folds(rows,[])
    _,again=assign_folds(rows,[])
    assert folds==again
    for fold in folds:
        assert len(fold['roles'])==len(rows)
        for group in set(groups.values()):
            assert len({fold['roles'][p] for p,g in groups.items() if g==group})==1
        for role in ('FIT','CAL','VALIDATION'):
            assert {r['label'] for r in rows if fold['roles'][r['parent_id']]==role}=={0,1}
    assert all(sum(f['roles'][r['parent_id']]=='VALIDATION' for f in folds)==1 for r in rows)


def test_insufficient_groups_fail_closed():
    rows=[{'parent_id':str(i),'source':'same','label':i%2} for i in range(100)]
    with pytest.raises(ValueError): assign_folds(rows,[])


def test_frozen_outputs_cannot_be_overwritten(tmp_path):
    path=tmp_path/'contract.json'
    fixed_write(path,{'frozen':True}); fixed_write(path,{'frozen':True})
    with pytest.raises(ValueError):fixed_write(path,{'frozen':False})
