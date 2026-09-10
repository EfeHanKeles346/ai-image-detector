"""Isolated metadata eligibility probe; no rendered image or model score."""
import json
import sys
import math
import rawpy


def valid_wb(values):
    return len(values)==4 and all(math.isfinite(v) for v in values) and min(values[:3])>0


if __name__=='__main__':
    with rawpy.imread(sys.argv[1]) as raw:
        wb=raw.camera_whitebalance
        print(json.dumps({'camera_whitebalance':wb,'valid_as_shot_wb':valid_wb(wb),
                          'num_colors':raw.num_colors,'color_desc':raw.color_desc.decode(),
                          'rawpy':rawpy.__version__,'libraw':list(rawpy.libraw_version)},allow_nan=False))
