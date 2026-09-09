from io import BytesIO
import numpy as np
from PIL import Image

from experiments.e54_data import crop_views


def test_crop_cache_preserves_fixed_view_shape_and_is_deterministic():
    stream=BytesIO()
    Image.fromarray(np.random.default_rng(54).integers(0,256,(250,320,3),dtype=np.uint8)).save(stream,format='PNG')
    one=crop_views(stream.getvalue(),'fixed-parent')
    assert one.shape==(3,3,224,224,3)
    assert one.dtype==np.uint8
    np.testing.assert_array_equal(one,crop_views(stream.getvalue(),'fixed-parent'))
