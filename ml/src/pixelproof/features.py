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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

EPS = 1e-8
FFT_BANDS = 16
BLOCK = 8  # JPEG's DCT block size
THREADS = 6         # measured optimum on an M3 Pro; see extract_tiles()
PARALLEL_FROM = 16  # below this the pool costs more than it saves


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


def f_cfa_lattice(gray: np.ndarray, residual: np.ndarray | None = None) -> list[float]:
    """Variance of the residual on the four Bayer sub-lattices.

    A demosaiced photograph: the four populations differ (some values were
    physically measured, some interpolated). A generated image: identical.
    We report the spread between them, normalised — so it is scale-free.

    `residual` may be supplied by a caller that has already computed it. The
    3x3 median filter behind _residual() is 63% of a tile's total cost and this
    function and f_noise() were each computing the grey one independently.
    """
    if residual is None:
        residual = _residual(gray)
    variances = [float(residual[y::2, x::2].var()) for y in (0, 1) for x in (0, 1)]
    mean_variance = float(np.mean(variances)) + EPS
    normalised = [v / mean_variance for v in variances]
    return normalised + [float(np.std(normalised)), float(max(normalised) - min(normalised))]


def f_noise(rgb: np.ndarray, gray: np.ndarray, residual: np.ndarray | None = None) -> list[float]:
    if residual is None:
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
def _axis_positions(length: int, tile: int) -> np.ndarray:
    """Start offsets covering `length` completely, last one flush with the edge.

    The previous rule laid tiles on a `length // tile` grid and centred it, which
    left `length mod tile` pixels unexamined as a frame around the image. The
    loss is `C ~ 2/k` for k tiles per axis: 3% on a 4032px photo, but **41% at
    500px** — and every real photograph in GenImage is exactly 500px. It lands on
    the border, so an edit near the edge was systematically invisible.

    Here the count is rounded UP and the offsets are spread from 0 to
    `length - tile`, so the first tile starts at the edge and the last ends at
    it. Adjacent tiles overlap by whatever the remainder demands. Overlap is
    cheap and useful: the shared pixels get scored more than once, which makes a
    localisation heat-map smoother rather than noisier.
    """
    if length <= tile:
        return np.array([0])
    count = int(np.ceil(length / tile))
    return np.unique(np.linspace(0, length - tile, count).round().astype(int))


def _thin(positions: np.ndarray, keep: int) -> np.ndarray:
    """An evenly spaced subset that always retains both ends."""
    if keep >= len(positions):
        return positions
    picked = np.linspace(0, len(positions) - 1, max(keep, 2)).round().astype(int)
    return positions[np.unique(picked)]


def tile_positions(width: int, height: int, tile: int,
                   max_tiles: int | None = None) -> list[tuple[int, int]]:
    """Tile origins covering the whole image. `max_tiles=None` means full coverage.

    The cap used to default to 36, which sounds generous and is not: a
    4032x3024 photograph yields 713 tiles and scoring 36 of them inspects
    **4.8% of the pixels**. (The "~100% coverage" claim in HISTORY 9b holds only
    up to 768px.) Full coverage is affordable now — measured 2026-08-05, a
    ResNet-18 scores a tile in 0.44 ms, so all 713 take 0.31 s.

    When a cap IS applied the ends are always kept, so the frame stays covered
    and only the interior is sampled.
    """
    xs = _axis_positions(width, tile)
    ys = _axis_positions(height, tile)
    if max_tiles is not None and len(xs) * len(ys) > max_tiles:
        side = max(1, int(np.floor(np.sqrt(max_tiles))))
        xs, ys = _thin(xs, side), _thin(ys, side)
    return [(int(x), int(y)) for y in ys for x in xs]


def select_tiles(image: Image.Image, tile: int = 128, max_tiles: int | None = None,
                 texture_floor: float = 0.0, min_tiles: int = 3
                 ) -> tuple[list[Image.Image], list[float], list[tuple[int, int]]]:
    """Which tiles are worth analysing — crops, textures and boxes, no features.

    Split out from extract_tiles() so a consumer that does NOT want the 68
    statistics can still get the same geometry and the same texture screening.
    A CNN arm wants the pixels; making it pay for feature extraction just to
    learn which tiles to keep would be measuring the wrong thing and paying
    8.4 ms a tile for the privilege.
    """
    image = image.convert("RGB")
    width, height = image.size
    boxes = tile_positions(width, height, tile, max_tiles)
    patches = [image.crop((x, y, x + tile, y + tile)) for x, y in boxes]
    textures = [float((np.asarray(p.convert("L"), dtype=np.float32) / 255.0).std())
                for p in patches]

    keep = [i for i, t in enumerate(textures) if t >= texture_floor]
    if len(keep) < min_tiles:
        # Never starve the aggregation. A mostly-flat photograph can lose 19 of
        # 20 tiles to the floor, and a top-3 mean over one surviving tile is one
        # tile's opinion wearing an average's clothes. Fall back to the
        # highest-texture `min_tiles` — they are the only ones carrying anything
        # measurable anyway.
        keep = sorted(np.argsort(textures)[-min_tiles:].tolist())
    return ([patches[i] for i in keep], [textures[i] for i in keep], [boxes[i] for i in keep])


def extract_tiles(image: Image.Image, tile: int = 128, max_tiles: int | None = None,
                  texture_floor: float = 0.0, min_tiles: int = 3
                  ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Cut the image into `tile`x`tile` squares at NATIVE resolution, describe each.

    Returns (features (n, N_FEATURES), texture (n,), positions (n,)) — all three
    aligned. Positions are returned rather than re-derived by the caller: E17 and
    E18 were calling tile_positions() separately and zipping the two by index,
    which is only correct as long as nothing filters, and something does now.

    No retraining is needed to use this: the crop128 model was fitted on 128x128
    native crops, and every tile here IS one — the same input distribution, just
    evaluated several times per image.

    `texture_floor` skips the expensive extraction for tiles whose grey-level
    standard deviation falls below it. Measuring texture costs 0.029 ms against
    8.4 ms for the full 68 features — **1/290** — and a flat tile (sky, a wall,
    plain clothing) scores around 0.5, so it can never enter a top-k aggregate.
    E11 measured 21% of tiles below 0.04. Skipped tiles are dropped from all
    three returned arrays, so aggregation never sees them.

    The per-tile scores are also exactly what a localisation heat-map is built
    from, so this is shared machinery with Module 2.
    """
    image = image.convert("RGB")
    width, height = image.size
    if width < tile or height < tile:              # too small to tile: one padded crop
        grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return (np.stack([extract_from_image(image, tile)]),
                np.array([grey.std()], dtype=np.float32), [(0, 0)])

    patches, textures, boxes = select_tiles(image, tile, max_tiles, texture_floor, min_tiles)
    arrays = [array_from_image(p) for p in patches]
    if len(arrays) >= PARALLEL_FROM:
        # THREADS, not processes. Most of a tile's cost is inside scipy's median
        # filter and numpy's FFT, both of which release the GIL, so threads give
        # 3.2x on an M3 Pro (6.6 s -> 2.1 s over 768 tiles) with no pickling and
        # nothing to go wrong inside a web worker. More than ~6 workers does not
        # help; this is memory-bandwidth bound, not core bound.
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            vectors = list(pool.map(_vector, arrays))
    else:
        vectors = [_vector(a) for a in arrays]

    return np.stack(vectors), np.array(textures, dtype=np.float32), boxes


def _vector(rgb: np.ndarray) -> np.ndarray:
    """The 68 numbers, from an RGB float array in [0,1].

    The grey noise residual is computed ONCE here and handed to both consumers.
    It used to be computed inside f_cfa_lattice and again inside f_noise; the
    3x3 median filter behind it is 63% of a tile's cost, so the duplicate was
    ~12% of every extraction. The output is bit-identical.
    """
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    residual = _residual(gray)
    values = (f_channel_stats(rgb) + f_cross_channel(rgb)
              + f_cfa_lattice(gray, residual) + f_noise(rgb, gray, residual)
              + f_spectrum(gray) + f_local_variance(gray)
              + f_jpeg_grid(gray) + f_saturation(rgb))
    return np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_from_image(image: Image.Image, crop: int | None = None) -> np.ndarray:
    """Same as extract(), for an already-open PIL image (used by the web service)."""
    return _vector(array_from_image(image, crop))


def extract(path: Path, crop: int | None = None) -> np.ndarray:
    return _vector(load_array(path, crop))


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
