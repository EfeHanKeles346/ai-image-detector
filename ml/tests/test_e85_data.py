import numpy as np
import pytest
from experiments import e85_data as m


def test_new_transport_appends_without_reordering_or_changing_old_views():
    rows=[{'parent_id':str(i),'role':'TRAIN'} for i in range(3)]
    rng=np.random.default_rng(85)
    old={k:rng.normal(size=(3,3,n)).astype(np.float32) for k,n in m.WIDTHS.items()}
    social={k:rng.normal(size=(3,n)).astype(np.float32) for k,n in m.WIDTHS.items()}|{
        'binding':np.array('frozen'),'condition':np.array('social_q75'),'parents':np.array(['0','1','2']),'roles':np.repeat('TRAIN',3)}
    actual=m.append_social(old,social,rows,'frozen')
    for key,width in m.WIDTHS.items():
        assert actual[key].shape==(3,4,width)
        assert np.array_equal(actual[key][:,:3],old[key]) and np.array_equal(actual[key][:,3],social[key])
    for key,value in [('parents',np.array(['1','0','2'])),('roles',np.repeat('DEVELOPMENT',3)),('condition',np.array('q75'))]:
        with pytest.raises(ValueError,match='TRAIN'):m.append_social(old,social|{key:value},rows,'frozen')
