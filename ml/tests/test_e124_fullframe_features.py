import numpy as np
import pytest
from experiments.e124_fullframe_features import verify_chunk,CONDITIONS
from experiments.e71_features import array_sha


def test_complete_fullframe_chunk_rejects_role_source_order_and_array_changes():
    rows=[{'parent_id':'a','sha256':'s1'},{'parent_id':'b','sha256':'s2'}]
    x=np.zeros((2,4,768),np.float32)
    a={'full':x,'binding':'bound','parents':np.array(['a','b']),'sources_sha256':np.array(['s1','s2']),
        'roles':np.array(['TRAIN','TRAIN']),'conditions':np.array(CONDITIONS),'array_sha256':array_sha(x)}
    assert verify_chunk(a,rows,'bound') is x
    for key,value in [('roles',np.array(['TRAIN','DEVELOPMENT'])),('parents',np.array(['b','a'])),
        ('sources_sha256',np.array(['s1','changed'])),('conditions',np.array(CONDITIONS[::-1])),('full',x+1)]:
        with pytest.raises(ValueError):verify_chunk(a|{key:value},rows,'bound')
