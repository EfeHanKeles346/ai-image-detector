from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from experiments.e123_fullframe_probe import fullframe_array,transported_images,selected
from experiments.e75_features import crop_views
from experiments.e84_features import social_crops
from pixelproof.e32_candidate import standardized_array


def test_fullframe_keeps_opposite_borders_and_native_center_transport_parity():
    a=np.zeros((240,720,3),dtype=np.uint8);a[:,:160,0]=255;a[:,-160:,2]=255
    image=Image.fromarray(a);full=fullframe_array(image)
    assert full.shape==(224,224,3)
    assert full[:,2,0].mean()>200 and full[:,-3,2].mean()>200
    stream=BytesIO();image.save(stream,format='PNG');body=stream.getvalue()
    expected=np.concatenate([crop_views(body,'test'),social_crops(body)[0][None]])[:,0]
    actual=np.stack([standardized_array(im) for im in transported_images(body,'test')])
    assert np.array_equal(expected,actual)


def test_fullframe_source_selection_keeps_legacy_case_and_denies_dev():
    rows=[{'role':'train','label':0,'source':'a','parent_id':'a1'},
          {'role':'TRAIN','label':1,'source':'b','parent_id':'b1'}]
    assert selected(rows)==[0,1]
    rows[0]['role']='DEVELOPMENT'
    with pytest.raises(ValueError,match='TRAIN'):selected(rows)
