import numpy as np
import pytest
from experiments.e83_diagnostic import check_cache


def test_diagnostic_rejects_role_and_order_leakage_before_scoring():
    rows=[{'record_id':str(i),'sha256':f'h{i}'} for i in range(2)]
    a={'binding':np.array('contract'),'roles':np.repeat('DEVELOPMENT',2),
       'record_ids':np.array(['0','1']),'image_sha256':np.array(['h0','h1']),
       **{k:np.zeros((2,n),np.float32) for k,n in [('dino',3072),('clip',1536),('dear',1640)]}}
    check_cache(a,rows,'contract')
    for key,value in [('roles',np.repeat('TRAIN',2)),('record_ids',np.array(['1','0'])),
                      ('image_sha256',np.array(['h1','h0'])),('binding',np.array('other'))]:
        with pytest.raises(ValueError,match='DEVELOPMENT'):check_cache(a|{key:value},rows,'contract')
    with pytest.raises(ValueError,match='finite'):check_cache(a|{'dear':a['dear'][:1]},rows,'contract')
