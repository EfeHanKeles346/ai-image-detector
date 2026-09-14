"""Bounded TRAIL-style numerical primitives, not a calibrated AI probability.

Adapted from TRAIL authors' MIT-licensed core.py at revision
0fce0f1de2ac0d219baf91592ee2bca95ec0a219:
https://github.com/VishalJ99/trail-image-edit-localization
Copyright (c) 2026 TRAIL authors. See ml/third_party/TRAIL_LICENSE.txt.
This adaptation adds strict finite/geometry checks and exposes half-cosine drift
on [0,1] for a common metric interface; dividing by two does not calibrate it.
"""
import numpy as np


def haar_attenuate(image):
    """Fixed two-level global Haar detail attenuation, alpha=.2, no mask input."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 4 or \
            any(x % 4 for x in value.shape[:2]) or value.dtype != np.uint8:
        raise ValueError('Haar input must be uint8 RGB with sides divisible by four')
    value = value.astype(np.float32) / 255.
    h, w, c = value.shape
    low = value.reshape(h//4, 4, w//4, 4, c).mean(axis=(1, 3))
    low = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
    return np.clip(value - .2*(value-low), 0., 1.).astype(np.float32)


def half_cosine_drift(left, right):
    left = np.asarray(left, dtype=np.float32); right = np.asarray(right, dtype=np.float32)
    if left.ndim != 3 or left.shape != right.shape or min(left.shape) < 1 or \
            not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Finite aligned HWC patch-token grids required')
    a = np.linalg.norm(left, axis=-1, keepdims=True)
    b = np.linalg.norm(right, axis=-1, keepdims=True)
    if np.any(a == 0) or np.any(b == 0) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonzero finite patch-token norms required')
    cosine = ((left/a)*(right/b)).sum(axis=-1)
    return (.5*(1-np.clip(cosine, -1., 1.))).astype(np.float32)


def median3(value):
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 2 or value.dtype.kind != 'f' or \
            not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Finite bounded two-dimensional score grid required')
    padded = np.pad(value, ((1, 1), (1, 1)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def expand_grid(value, shape):
    """Repeat exact grid cells; no interpolation or mask-aware smoothing."""
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1 or len(shape) != 2 or \
            any(type(x) is not int or x < 1 for x in shape) or \
            any(total % cells for total, cells in zip(shape, value.shape, strict=True)):
        raise ValueError('Exact integer expansion from grid to image geometry required')
    if value.dtype.kind != 'f' or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Finite bounded grid required')
    return np.repeat(np.repeat(value, shape[0]//value.shape[0], axis=0), shape[1]//value.shape[1], axis=1)
