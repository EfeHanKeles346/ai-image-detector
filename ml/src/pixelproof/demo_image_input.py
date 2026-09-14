"""Bounded local-demo support for full 24MP phone photographs.

This changes admission only. No resizing or recompression is applied during decode;
frozen E92 preprocessing still caps the image inside its feature pipeline.
"""
import io
from PIL import Image
from pixelproof.image_input import ImageLimits, ImagePolicyError, decode_image

PHOTO_LIMITS = ImageLimits(max_pixels=32_000_000)


def decode_photo(raw: bytes):
    image = decode_image(raw, PHOTO_LIMITS)
    with Image.open(io.BytesIO(raw)) as original:
        if original.format != 'MPO' and getattr(original, 'is_animated', False):
            raise ImagePolicyError(415, 'Hareketli görseller desteklenmiyor. Tek bir fotoğraf seçin.')
        if original.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
            raise ImagePolicyError(415, 'Bu renk biçimi desteklenmiyor. Standart bir JPG veya PNG seçin.')
        if original.mode in ('RGBA', 'LA') or 'transparency' in original.info:
            if original.convert('RGBA').getchannel('A').getextrema()[0] < 255:
                raise ImagePolicyError(415, 'Şeffaf görseller desteklenmiyor. Fotoğrafın asıl dosyasını seçin.')
    return image
