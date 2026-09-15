"""Deterministic mask translations; never choose locations from model scores."""
import numpy as np

CORNERS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')


def corner_mask(value, index):
    mask = np.asarray(value)
    if mask.shape != (512, 512) or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all() or \
            not mask.any() or mask.all() or type(index) is not int or not 0 <= index < 16:
        raise ValueError('Exact binary512 mask and fixed16-parent index required')
    yy, xx = np.where(mask)
    left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
    width, height = right-left, bottom-top
    if width > 480 or height > 480:
        raise ValueError('Mask must fit without resizing or clipping')
    corner = index % 4
    x = 16 if corner % 2 == 0 else 496-width
    y = 16 if corner < 2 else 496-height
    translated = np.zeros_like(mask)
    translated[y:y+height, x:x+width] = mask[top:bottom, left:right]
    if np.count_nonzero(translated) != np.count_nonzero(mask):
        raise ValueError('Mask area changed')
    return translated, {'corner': CORNERS[corner], 'old_bbox': [left, top, right, bottom],
                        'new_bbox': [x, y, x+width, y+height], 'shift': [x-left, y-top],
                        'mask_pixels': int(np.count_nonzero(mask))}


def accepted_indices(cases):
    if len(cases) != 16 or [r['index'] for r in cases] != list(range(16)):
        raise ValueError('All sixteen generation attempts must be retained')
    accepted = []
    for row in cases:
        if type(row.get('passed')) is not bool or row.get('seed') != 119000+row['index'] or row.get('branch') != 'fp16_sdpa':
            raise ValueError('Generation identity differs')
        if 'checks' in row:
            if set(row.get('checks', {})) != {'safety_check_negative', 'nonblank_output', 'masked_pixels_changed', 'background_exact'} or \
                    any(type(v) is not bool for v in row['checks'].values()) or row['passed'] != all(row['checks'].values()):
                raise ValueError('Generation lacks consistent engineering guards')
        elif row['passed'] or not row.get('first_nonfinite_stage') or row.get('rendered_images') != 0 or row.get('safety_classification_performed') is not False:
            raise ValueError('Rejected nonfinite attempt requires its original failure receipt')
        if row['passed']:
            accepted.append(row['index'])
    return accepted
