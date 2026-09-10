"""Standalone isolated RAW decoder; never imports the model-training environment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image
import rawpy


def versions():
    return {'python': sys.version, 'rawpy': rawpy.__version__, 'numpy': np.__version__,
            'pillow': PIL.__version__, 'libraw': list(rawpy.libraw_version)}


def decode(source, target):
    with rawpy.imread(source) as raw:
        if raw.sizes.width * raw.sizes.height > 100_000_000:
            raise ValueError('RAW exceeds pixel budget')
        wb = raw.camera_whitebalance
        if not np.isfinite(wb).all() or min(wb[:3]) <= 0:
            raise ValueError('missing valid as-shot white balance')
        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            half_size=False, four_color_rgb=False, use_camera_wb=True, use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB, output_bps=8, no_auto_bright=True,
            no_auto_scale=False, bright=1., gamma=(2.4, 12.92),
            highlight_mode=rawpy.HighlightMode.Clip, median_filter_passes=0,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 224:
        raise ValueError('invalid RGB output')
    pixels = hashlib.sha256(f'RGB:{rgb.shape[1]}:{rgb.shape[0]}:'.encode()+rgb.tobytes()).hexdigest()
    target = Path(target)
    if target.exists():
        with Image.open(target) as saved:
            if not np.array_equal(np.asarray(saved), rgb): raise ValueError('decoded replay differs')
    else:
        part = target.with_suffix('.png.part')
        Image.fromarray(rgb).save(part, format='PNG', optimize=False, compress_level=6)
        part.replace(target)
    return {'versions': versions(), 'width': rgb.shape[1], 'height': rgb.shape[0],
            'pixel_sha256': pixels, 'camera_whitebalance': wb,
            'minimum': int(rgb.min()), 'maximum': int(rgb.max()), 'mean': float(rgb.mean()),
            'black_pixel_fraction': float(np.all(rgb == 0, axis=2).mean()),
            'white_pixel_fraction': float(np.all(rgb == 255, axis=2).mean())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--versions', action='store_true')
    parser.add_argument('source', nargs='?'); parser.add_argument('target', nargs='?')
    args = parser.parse_args()
    print(json.dumps(versions() if args.versions else decode(args.source, args.target)))
