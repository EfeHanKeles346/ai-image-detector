from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from pixelproof import source_pixel_views as old, source_pixel_views_b as new


@pytest.mark.parametrize('orientation',range(1,9))
def test_all_orientation_pixels_and_features_match_frozen_e150(orientation):
    image=Image.fromarray(np.random.default_rng(150).integers(0,256,(183,301,3),dtype=np.uint8))
    exif=Image.Exif();exif[274]=orientation
    stream=BytesIO();image.save(stream,format='JPEG',exif=exif)
    parent=next(str(i) for i in range(100) if old.assigned_transport(str(i))!='resize_jpeg')
    before,shape=old.extract(stream.getvalue(),parent);after,new_shape=new.extract(stream.getvalue(),parent)
    np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(shape,new_shape)


@pytest.mark.parametrize('mode',['RGB','L','RGBA'])
def test_large_image_all_transports_and_modes_have_exact_parity(mode):
    image=Image.fromarray(np.random.default_rng(2).integers(0,256,(1510,2110,3),dtype=np.uint8)).convert(mode)
    if mode=='RGBA':image.putalpha(100)
    raw=BytesIO();image.save(raw,format='PNG')
    parents={}
    for i in range(100):parents.setdefault(old.assigned_transport(str(i)),str(i))
    for parent in parents.values():
        before,shape=old.extract(raw.getvalue(),parent);after,new_shape=new.extract(raw.getvalue(),parent)
        np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(shape,new_shape)
