"""Fixed local residual-pair histograms; no learned weights or authenticity claims."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps

SIDE = 128
DIMENSIONS = 300  # RGB x orders1/2 x axes y/x x 5x5 joint histogram.


def center_patch(image):
    if min(image.size) < SIDE or image.width * image.height > 100000000:
        raise ValueError('Source must support an unpadded128 crop within100MP')
    oriented = ImageOps.exif_transpose(image)
    left, top = (oriented.width-SIDE)//2, (oriented.height-SIDE)//2
    # Same RGB conversion convention as existing models; alpha is discarded, not composited.
    patch = oriented.crop((left, top, left+SIDE, top+SIDE)).convert('RGB')
    return np.asarray(patch, dtype=np.uint8).copy()


def residual_pairs(patch):
    patch = np.asarray(patch)
    if patch.shape != (SIDE, SIDE, 3) or patch.dtype != np.uint8:
        raise ValueError('Exact128x128 RGB uint8 source patch required')
    features = []
    for channel in range(3):
        for order in (1, 2):
            for axis in (0, 1):
                residual = np.diff(patch[:, :, channel].astype(np.int16), n=order, axis=axis)
                quantized = np.clip(np.rint(residual.astype(np.float64)/4), -2, 2).astype(np.int16)+2
                first, second = (quantized[:-1, :], quantized[1:, :]) if axis == 0 else (quantized[:, :-1], quantized[:, 1:])
                hist = np.bincount((first*5+second).ravel(), minlength=25).astype(np.float64)
                features.append(hist/hist.sum())
    result = np.concatenate(features).astype(np.float32)
    if result.shape != (DIMENSIONS,) or not np.isfinite(result).all():
        raise ValueError('Invalid residual histogram')
    return result


def paired_patch_features(patch):
    original = residual_pairs(patch)
    low = Image.fromarray(patch).resize((64, 64), Image.Resampling.LANCZOS)
    reconstructed = np.asarray(low.resize((SIDE, SIDE), Image.Resampling.LANCZOS), dtype=np.uint8)
    return np.stack([original, residual_pairs(reconstructed)])


def extract(raw):
    with Image.open(BytesIO(raw)) as image:
        patch = center_patch(image)
    return patch, paired_patch_features(patch)
