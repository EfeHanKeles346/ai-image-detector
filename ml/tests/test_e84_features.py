import hashlib
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from experiments import e84_features as m


def test_social_crops_preserve_frozen_cap_before_jpeg_transport():
    rng=np.random.default_rng(84);pixels=rng.integers(0,256,(1200,1600,3),dtype=np.uint8)
    body=BytesIO();Image.fromarray(pixels).save(body,format='PNG');body=body.getvalue()
    crops,qsha=m.social_crops(body)
    q75=m.social_q75_bytes(BytesIO(body))
    with Image.open(BytesIO(q75)) as im:
        assert im.size==(1080,810)
        expected=np.stack(m.dino.texture_crops(m.dino.transport_image(im,'clean')))
    assert qsha==hashlib.sha256(q75).hexdigest()
    assert crops.dtype==np.uint8 and crops.shape==(3,224,224,3)
    assert np.array_equal(crops,expected)
    # Reversing the transport order is a materially different input, not an equivalent JPEG view.
    encoded=BytesIO();Image.fromarray(pixels).save(encoded,format='JPEG',quality=75,subsampling=2,optimize=False)
    with Image.open(BytesIO(encoded.getvalue())) as im:
        wrong=np.stack(m.dino.texture_crops(im.resize((1080,810),Image.Resampling.LANCZOS)))
    assert not np.array_equal(crops,wrong)


def test_fixed_windows_cover_all_parents_once_and_keep_partial_tail():
    rows=list(range(12141));parts=m.windows(rows)
    assert len(parts)==1518 and len(parts[-1])==5
    assert [x for part in parts for x in part]==rows


def test_chunk_rejects_role_order_and_feature_corruption():
    rows=[{'parent_id':str(i),'sha256':f'h{i}'} for i in range(2)]
    values={k:np.zeros((2,n),np.float32) for k,n in m.WIDTHS.items()}
    a=values|{k+'_sha256':np.array(m.array_sha(v)) for k,v in values.items()}|{
        'binding':np.array('bound'),'parents':np.array(['0','1']),'source_sha256':np.array(['h0','h1']),
        'roles':np.repeat('TRAIN',2),'transport_sha256':np.repeat('a'*64,2)}
    m.check_chunk(a,'bound',rows)
    for key,value in [('roles',np.repeat('DEVELOPMENT',2)),('parents',np.array(['1','0'])),
                      ('source_sha256',np.array(['h1','h0'])),('binding',np.array('other'))]:
        with pytest.raises(ValueError,match='TRAIN'):m.check_chunk(a|{key:value},'bound',rows)
    changed=a['clip'].copy();changed[0,0]=1
    with pytest.raises(ValueError,match='feature body'):m.check_chunk(a|{'clip':changed},'bound',rows)
