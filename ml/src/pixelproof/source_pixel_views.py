"""Four fixed source-pixel inputs; semantic encoder caches remain unchanged."""
import hashlib
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pixelproof.source_pixel_residual import center_patch, paired_patch_features

CONDITIONS = ('clean', 'assigned_transport', 'q75', 'social_q75')
TRANSPORTS = ('jpeg', 'webp', 'resize_jpeg', 'mild_blur')


def assigned_transport(parent):
    value = hashlib.sha256(('E42_TRANSPORT|'+parent).encode()).digest()[:4]
    return TRANSPORTS[int.from_bytes(value, 'big') % 4]


def capped_size(size, limit):
    if max(size) <= limit:
        return tuple(size)
    scale = limit / max(size)
    return tuple(max(1, round(d*scale)) for d in size)


def geometry(size, parent):
    assigned = capped_size(size, 2048)
    if assigned_transport(parent) == 'resize_jpeg':
        assigned = tuple(max(1, round(d*.65)) for d in assigned)
    return (tuple(size), assigned, tuple(size), capped_size(size, 1080))


def encoded(image, format_name, quality):
    stream = BytesIO()
    options = dict(quality=quality)
    if format_name == 'JPEG':
        options.update(subsampling=2, optimize=False, progressive=False)
    image.save(stream, format=format_name, **options)
    with Image.open(BytesIO(stream.getvalue())) as opened:
        return opened.convert('RGB')


def cap(image, limit):
    size = capped_size(image.size, limit)
    return image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image.copy()


def assigned_image(image, parent):
    with cap(image, 2048) as base:
        condition = assigned_transport(parent)
        if condition == 'jpeg':
            return encoded(base, 'JPEG', 55)
        if condition == 'webp':
            return encoded(base, 'WEBP', 60)
        if condition == 'mild_blur':
            return base.filter(ImageFilter.GaussianBlur(.8))
        size = tuple(max(1, round(d*.65)) for d in base.size)
        with base.resize(size, Image.Resampling.LANCZOS) as resized:
            return encoded(resized, 'JPEG', 65)


def extract(raw, parent):
    with Image.open(BytesIO(raw)) as opened:
        if opened.width*opened.height > 100000000:
            raise ValueError('Source exceeds100MP budget')
        with ImageOps.exif_transpose(opened) as oriented:
            image = oriented.convert('RGB')
    sizes = geometry(image.size, parent)
    if any(min(size) < 128 for size in sizes):
        image.close()
        raise ValueError('Every condition must support128 without padding or upsampling')
    values = []
    try:
        values.append(paired_patch_features(center_patch(image)))
        with assigned_image(image, parent) as view:
            if view.size != sizes[1]:raise ValueError('Assigned geometry differs')
            values.append(paired_patch_features(center_patch(view)))
        # Q75 source pixels precede the semantic encoder's2048 cap.
        with encoded(image, 'JPEG', 75) as view:
            if view.size != sizes[2]:raise ValueError('Q75 geometry differs')
            values.append(paired_patch_features(center_patch(view)))
        with cap(image, 1080) as small:
            with encoded(small, 'JPEG', 75) as view:
                if view.size != sizes[3]:raise ValueError('Social geometry differs')
                values.append(paired_patch_features(center_patch(view)))
    finally:
        image.close()
    result = np.stack(values)
    if result.shape != (4, 2, 300) or result.dtype != np.float32 or not np.isfinite(result).all() or \
            (result < 0).any() or not np.allclose(result.reshape(4, 2, 12, 25).sum(axis=-1), 1, rtol=0, atol=1e-6):
        raise ValueError('Incomplete normalized four-condition histograms')
    return result, np.asarray(sizes, dtype=np.int32)
