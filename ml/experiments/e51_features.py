"""Fixed, lightweight E51-B residual/DCT branch; no weights or network required."""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import uniform_filter

from experiments.e42_features import texture_crops, transport_image

EPS = 1e-12
BANDS = 8
CROP_SIZE = 224
PER_CROP = 16
DIMENSIONS = 32
BASE_NAMES = tuple(f'dct_energy_{i}' for i in range(BANDS)) + (
    'residual_abs_mean', 'residual_std', 'residual_abs_p50', 'residual_abs_p90',
    'residual_abs_p99', 'gradient_x_abs_mean', 'gradient_y_abs_mean',
    'residual_energy_fraction',
)
FEATURE_NAMES = tuple(f'{aggregation}_{name}' for aggregation in ('mean', 'std') for name in BASE_NAMES)


def crop_statistics(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape != (CROP_SIZE, CROP_SIZE, 3) or array.dtype != np.uint8:
        raise ValueError('E51-B requires a 224x224 RGB uint8 crop')
    grey = array.astype(np.float64).mean(axis=2) / 255.0
    centered = grey - grey.mean()
    spectrum = dctn(centered, norm='ortho') ** 2
    spectrum[0, 0] = 0.0
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt(xx**2 + yy**2) / (np.sqrt(2) * (CROP_SIZE - 1))
    bins = np.minimum((radius * BANDS).astype(int), BANDS - 1)
    energy = np.bincount(bins.ravel(), weights=spectrum.ravel(), minlength=BANDS)
    energy /= max(float(energy.sum()), EPS)
    residual = grey - uniform_filter(grey, size=3, mode='reflect')
    absolute = np.abs(residual)
    stats = np.asarray([
        absolute.mean(), residual.std(), *np.quantile(absolute, [.5, .9, .99]),
        np.abs(np.diff(grey, axis=1)).mean(), np.abs(np.diff(grey, axis=0)).mean(),
        np.mean(residual**2) / max(float(np.mean(centered**2)), EPS),
    ])
    result = np.concatenate([energy, stats]).astype(np.float32)
    if result.shape != (PER_CROP,) or not np.isfinite(result).all():
        raise ValueError('invalid residual/DCT feature')
    return result


def aggregate_crops(crops) -> np.ndarray:
    if len(crops) != 3:
        raise ValueError('E51-B requires exactly three shared E42 crops')
    features = np.stack([crop_statistics(crop) for crop in crops])
    return np.concatenate([features.mean(axis=0), features.std(axis=0)]).astype(np.float32)


def extract(image, condition='clean') -> np.ndarray:
    return aggregate_crops(texture_crops(transport_image(image, condition)))


def candidate_matrix(dino: np.ndarray, residual: np.ndarray, candidate: str) -> np.ndarray:
    dino, residual = np.asarray(dino), np.asarray(residual)
    if (dino.ndim != 2 or dino.shape[1] != 3072 or residual.shape != (len(dino), DIMENSIONS)
            or not np.isfinite(dino).all() or not np.isfinite(residual).all()):
        raise ValueError('E51 candidate feature contract changed')
    if candidate == 'A':
        return dino.copy()
    if candidate == 'B':
        return np.concatenate([dino, residual], axis=1)
    raise ValueError('only preregistered candidates A/B are permitted')
