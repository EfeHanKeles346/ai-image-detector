# =============================================================================
# features.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Turns ANY image, of ANY resolution, into a FIXED-LENGTH vector of numbers that
# a classical ML model can consume. No resizing, no cropping, no pixel skipped:
# every number is a statistic computed over the whole image.
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# Our CNN pipeline downscales every image to 224x224 before looking at it, and
# experiment E7 (2026-07-27) measured the cost: detection quality fell
# monotonically with how much an image had been downscaled (270px -> AUC 0.896,
# 1024px -> AUC 0.670). Generation artifacts live in fine texture; downscaling
# is a low-pass filter that removes exactly that.
#
# A statistic sidesteps the problem. "Average high-frequency energy per pixel"
# means the same thing on a 300x200 image and on a 4000x3000 one, and both
# produce ONE number. Resolution changes how many pixels we average over --
# never how many numbers come out.
#
# THE RULE EVERY FEATURE HERE OBEYS
# -----------------------------------------------------------------------------
# Ratios and per-pixel averages only -- never totals. A total (e.g. "sum of
# gradient energy") grows with pixel count and would smuggle resolution into
# the vector, which is the shortcut we are trying to avoid. See
# IMAGE_STRUCTURE_NOTES.md Q4.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# load_array()     Opens an image, canonicalises it to RGB float in [0,1].
#                  Optionally takes a fixed-size NATIVE crop (no resampling) so
#                  that real and AI images can be compared at identical
#                  dimensions -- this kills the "all AI images are square 1024"
#                  shortcut present in both GenImage and Defactify.
#
# f_channel_stats  Per-channel tone/colour behaviour: mean, std, skew, kurtosis
#                  for R, G, B. Generated images sit in a measurably different
#                  colour regime than camera output.
#
# f_cross_channel  Correlations and differences BETWEEN channels. This is the
#                  CFA / demosaicing trace: a real sensor measures one colour
#                  per photosite and interpolates the other two from
#                  neighbours, leaving a structured inter-channel dependency.
#                  A latent-diffusion image never passed through a sensor, so
#                  it has no such dependency. Analysing channels in isolation
#                  would throw this signal away.
#
# f_cfa_lattice    The same physics, measured directly: split the noise
#                  residual into the four Bayer sub-lattices (even/odd rows x
#                  even/odd columns). In a demosaiced photo these four
#                  populations have systematically different variance, because
#                  some values were measured and some were interpolated. In a
#                  generated image they are statistically identical.
#
# f_noise          Noise-residual statistics (image minus a denoised copy).
#                  Real photos carry sensor shot/read noise everywhere;
#                  diffusion output carries whatever the VAE decoder invented.
#
# f_spectrum       Radial FFT power spectrum in 16 normalised frequency bands.
#                  Upsamplers and VAE decoders leave periodic spectral traces,
#                  and diffusion suppresses local high-frequency variance.
#                  Frequencies are normalised to cycles/pixel and power is
#                  normalised by the total, so the bands mean the same thing at
#                  any resolution.
#
# f_local_variance Texture consistency: local variance measured in small
#                  windows, summarised as percentiles. Generated images tend to
#                  be more uniformly smooth than photographs.
#
# f_jpeg_grid      Blockiness across the 8x8 JPEG grid versus inside blocks.
#                  Reveals compression history -- and mismatched history is the
#                  basis of ELA and double-JPEG analysis.
#
# f_saturation     HSV saturation/value statistics; generated images occupy a
#                  different saturation distribution than camera output.
#
# extract()        Concatenates every group into one flat vector.
# FEATURE_NAMES    Human-readable name per position, so a trained model can be
#                  inspected ("which features actually mattered?").
# =============================================================================

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

EPS = 1e-8
FFT_BANDS = 16
BLOCK = 8  # JPEG's DCT block size


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def array_from_image(image: Image.Image, crop: int | None = None) -> np.ndarray:
    """RGB float array in [0,1]. `crop` takes a centre crop at NATIVE resolution.

    Cropping (never resizing) is how we give both classes identical dimensions
    without resampling a single pixel.
    """
    image = image.convert("RGB")
    if crop is not None:
        width, height = image.size
        if width < crop or height < crop:  # too small: pad by reflection
            array = np.asarray(image, dtype=np.float32) / 255.0
            pad_y, pad_x = max(0, crop - height), max(0, crop - width)
            array = np.pad(array, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
            return array[:crop, :crop]
        left, top = (width - crop) // 2, (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
    return np.asarray(image, dtype=np.float32) / 255.0


def load_array(path: Path, crop: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        return array_from_image(image, crop)


def _moments(values: np.ndarray) -> list[float]:
    """mean, std, skew, kurtosis — all scale-free summaries of a distribution."""
    values = values.ravel()
    mean = float(values.mean())
    std = float(values.std())
    centred = (values - mean) / (std + EPS)
    return [mean, std, float((centred ** 3).mean()), float((centred ** 4).mean())]


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #
def f_channel_stats(rgb: np.ndarray) -> list[float]:
    return [v for c in range(3) for v in _moments(rgb[:, :, c])]


def f_cross_channel(rgb: np.ndarray) -> list[float]:
    """Correlation between channels — the demosaicing fingerprint."""
    flat = rgb.reshape(-1, 3)
    flat = (flat - flat.mean(0)) / (flat.std(0) + EPS)
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        out.append(float((flat[:, a] * flat[:, b]).mean()))          # correlation
        out.append(float((rgb[:, :, a] - rgb[:, :, b]).std()))        # difference spread
    return out


def _residual(gray: np.ndarray) -> np.ndarray:
    """High-frequency residual: what a 3x3 median filter cannot explain.

    This is where sensor noise lives — and where a generated image has only
    whatever texture the decoder synthesised.
    """
    return gray - ndimage.median_filter(gray, size=3)


def f_cfa_lattice(gray: np.ndarray) -> list[float]:
    """Variance of the residual on the four Bayer sub-lattices.

    A demosaiced photograph: the four populations differ (some values were
    physically measured, some interpolated). A generated image: identical.
    We report the spread between them, normalised — so it is scale-free.
    """
    residual = _residual(gray)
    variances = [float(residual[y::2, x::2].var()) for y in (0, 1) for x in (0, 1)]
    mean_variance = float(np.mean(variances)) + EPS
    normalised = [v / mean_variance for v in variances]
    return normalised + [float(np.std(normalised)), float(max(normalised) - min(normalised))]


def f_noise(rgb: np.ndarray, gray: np.ndarray) -> list[float]:
    residual = _residual(gray)
    out = _moments(residual)
    out.append(float(np.abs(residual).mean()))                        # noise energy per pixel
    for c in range(3):                                                # per-channel noise level
        out.append(float(np.abs(_residual(rgb[:, :, c])).mean()))
    return out


def f_spectrum(gray: np.ndarray) -> list[float]:
    """Radial power spectrum in FFT_BANDS normalised frequency bands.

    Bands are defined in cycles/pixel (0 to 0.5), so band k means the same
    physical scale on any image size. Power is normalised by the total, so the
    result is a shape, not an amount.
    """
    windowed = gray - gray.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(windowed))) ** 2
    height, width = power.shape
    yy, xx = np.ogrid[:height, :width]
    # radius in normalised frequency: 0 at DC, 1.0 at Nyquist along each axis
    radius = np.sqrt(((yy - height / 2) / (height / 2)) ** 2 + ((xx - width / 2) / (width / 2)) ** 2)
    total = power.sum() + EPS
    edges = np.linspace(0, 1.0, FFT_BANDS + 1)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        bands.append(float(power[mask].sum() / total) if mask.any() else 0.0)
    high = sum(bands[FFT_BANDS // 2:])                                # high-frequency share
    slope = float(np.polyfit(np.arange(1, FFT_BANDS + 1), np.log(np.array(bands) + EPS), 1)[0])
    return bands + [high, slope]


def f_local_variance(gray: np.ndarray) -> list[float]:
    """Distribution of local texture energy — percentiles are scale-free."""
    mean = ndimage.uniform_filter(gray, size=8)
    variance = np.maximum(ndimage.uniform_filter(gray ** 2, size=8) - mean ** 2, 0.0)
    percentiles = np.percentile(variance, [10, 25, 50, 75, 90, 99])
    spread = float(percentiles[4] / (percentiles[1] + EPS))            # heavy-tail indicator
    return [float(v) for v in percentiles] + [spread, float(variance.mean())]


def f_jpeg_grid(gray: np.ndarray) -> list[float]:
    """Energy at 8x8 block boundaries vs inside blocks — JPEG's footprint."""
    dy = np.abs(np.diff(gray, axis=0))
    dx = np.abs(np.diff(gray, axis=1))
    out = []
    for diff, axis in ((dy, 0), (dx, 1)):
        index = np.arange(diff.shape[axis])
        on = np.isin((index + 1) % BLOCK, [0])
        boundary = diff[on] if axis == 0 else diff[:, on]
        inside = diff[~on] if axis == 0 else diff[:, ~on]
        out.append(float(boundary.mean() / (inside.mean() + EPS)))     # >1 means visible grid
    return out


def f_saturation(rgb: np.ndarray) -> list[float]:
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (maximum - minimum) / (maximum + EPS)
    return _moments(saturation) + _moments(maximum)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def tile_positions(width: int, height: int, tile: int, max_tiles: int) -> list[tuple[int, int]]:
    """Grid covering the whole image, thinned evenly if it exceeds max_tiles.

    Thinning picks an evenly spaced subset of rows and columns rather than the
    first N, so coverage stays spread across the picture instead of bunching in
    one corner.
    """
    columns, rows = max(1, width // tile), max(1, height // tile)
    side = int(np.floor(np.sqrt(max_tiles)))
    keep_x = np.unique(np.linspace(0, columns - 1, min(columns, side)).round().astype(int))
    keep_y = np.unique(np.linspace(0, rows - 1, min(rows, side)).round().astype(int))
    # Centre the grid so leftover pixels are split between the edges.
    offset_x, offset_y = (width - columns * tile) // 2, (height - rows * tile) // 2
    return [(int(offset_x + cx * tile), int(offset_y + cy * tile)) for cy in keep_y for cx in keep_x]


def extract_tiles(image: Image.Image, tile: int = 128, max_tiles: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Cut the image into `tile`x`tile` squares at NATIVE resolution, describe each.

    Returns (features (n_tiles, N_FEATURES), texture (n_tiles,)).

    No retraining is needed to use this: the crop128 model was fitted on
    128x128 native crops, and every tile here IS a 128x128 native crop — the
    same input distribution, just evaluated several times per image.

    `texture` is each tile's grey-level standard deviation. Flat tiles (sky, a
    wall, plain clothing) contain no measurable trace, so a caller can drop them
    instead of letting them dilute the informative ones.

    The per-tile scores are also exactly what a localisation heat-map is built
    from, so this is shared machinery with Module 2.
    """
    image = image.convert("RGB")
    width, height = image.size
    if width < tile or height < tile:              # too small to tile: one padded crop
        vector = extract_from_image(image, tile)
        grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return np.stack([vector]), np.array([grey.std()], dtype=np.float32)

    rows, texture = [], []
    for x, y in tile_positions(width, height, tile, max_tiles):
        patch = image.crop((x, y, x + tile, y + tile))
        rows.append(extract_from_image(patch))
        texture.append(float((np.asarray(patch.convert("L"), dtype=np.float32) / 255.0).std()))
    return np.stack(rows), np.array(texture, dtype=np.float32)


def extract_from_image(image: Image.Image, crop: int | None = None) -> np.ndarray:
    """Same as extract(), for an already-open PIL image (used by the web service)."""
    rgb = array_from_image(image, crop)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    values = (f_channel_stats(rgb) + f_cross_channel(rgb) + f_cfa_lattice(gray)
              + f_noise(rgb, gray) + f_spectrum(gray) + f_local_variance(gray)
              + f_jpeg_grid(gray) + f_saturation(rgb))
    return np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract(path: Path, crop: int | None = None) -> np.ndarray:
    rgb = load_array(path, crop)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    values = (f_channel_stats(rgb) + f_cross_channel(rgb) + f_cfa_lattice(gray)
              + f_noise(rgb, gray) + f_spectrum(gray) + f_local_variance(gray)
              + f_jpeg_grid(gray) + f_saturation(rgb))
    return np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _names() -> list[str]:
    names = [f"ch{c}_{m}" for c in "rgb" for m in ("mean", "std", "skew", "kurt")]
    names += [f"cross_{p}_{k}" for p in ("rg", "rb", "gb") for k in ("corr", "diffstd")]
    names += [f"cfa_lat{i}" for i in range(4)] + ["cfa_spread", "cfa_range"]
    names += [f"noise_{m}" for m in ("mean", "std", "skew", "kurt")] + ["noise_energy"]
    names += [f"noise_ch{c}" for c in "rgb"]
    names += [f"fft_band{i:02d}" for i in range(FFT_BANDS)] + ["fft_high_share", "fft_slope"]
    names += [f"lvar_p{p}" for p in (10, 25, 50, 75, 90, 99)] + ["lvar_spread", "lvar_mean"]
    names += ["jpeg_grid_y", "jpeg_grid_x"]
    names += [f"sat_{m}" for m in ("mean", "std", "skew", "kurt")]
    names += [f"val_{m}" for m in ("mean", "std", "skew", "kurt")]
    return names


FEATURE_NAMES = _names()
N_FEATURES = len(FEATURE_NAMES)
