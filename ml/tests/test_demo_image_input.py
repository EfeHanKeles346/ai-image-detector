import io
import numpy as np
import pytest
from PIL import Image
from pixelproof.demo_image_input import decode_photo
from pixelproof.research_serve import decode_demo
from pixelproof.image_input import ImagePolicyError


def encoded(size):
    stream = io.BytesIO()
    Image.new('RGB', size, (101, 82, 37)).save(stream, format='JPEG', quality=80)
    return stream.getvalue()


def test_real_phone_24mp_supported_without_rescaling():
    raw = encoded((5712, 4284))
    with pytest.raises(ImagePolicyError):
        decode_demo(raw)
    result = decode_photo(raw)
    assert result.size == (5712, 4284)
    with Image.open(io.BytesIO(raw)) as native:
        np.testing.assert_array_equal(np.asarray(result), np.asarray(native.convert('RGB')))


def test_old_supported_decode_remains_identical():
    raw = encoded((1200, 1600))
    np.testing.assert_array_equal(np.asarray(decode_photo(raw)), np.asarray(decode_demo(raw)))


def test_over_32mp_is_rejected_by_headers_before_pixels_load():
    raw = encoded((8001, 4000))
    with pytest.raises(ImagePolicyError) as caught:
        decode_photo(raw)
    assert caught.value.status_code == 413
