"""Decode the entire E105 quarantine subset without inferring admission from mask IDs."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def mask_labels(array):
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype != np.uint8 or not set(np.unique(array)).issubset(set(range(9))):
        raise ValueError('Expected native uint8 mask IDs 0..8')
    return array > 0


def audit():
    root = DATA_ROOT/'e105'; evidence = ML_ROOT.parent/'evidence'
    download = root/'download.json'; public = read(evidence/'e105_download.json')
    if digest(download) != public['receipt_sha256']:
        raise ValueError('Download receipt differs')
    report = read(download); rows = []; values = Counter(); problems = Counter()
    image_hashes = Counter(); decoded_hashes = Counter()
    for pair in report['rows']:
        row = {'id': pair['id'], 'problems': []}
        try:
            files = {f['member'].split('.')[-2]: f for f in pair['files']}
            for f in files.values():
                if digest(f['path']) != f['sha256']:
                    raise ValueError('Body digest differs')
            with Image.open(files['image']['path']) as image:
                w, h = image.size
                if w*h > 32_000_000:
                    raise ValueError('Over 32MP decode limit')
                image.load(); rgb = np.asarray(image.convert('RGB'))
                import hashlib
                decoded = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
            with Image.open(files['mask']['path']) as mask:
                if mask.size != (w, h):
                    raise ValueError('Native mask/image geometry mismatch')
                mask.load(); array = np.asarray(mask); positive = mask_labels(array)
            label_values = np.unique(array).tolist(); values.update(label_values)
            fraction = float(positive.mean())
            if fraction == 0: row['problems'].append('empty_edit_mask_origin_unresolved')
            if fraction == 1: row['problems'].append('full_image_edit_mask')
            row.update(width=w, height=h, mask_values=label_values, mask_fraction=fraction,
                       image_sha256=files['image']['sha256'], decoded_rgb_sha256=decoded)
            image_hashes[files['image']['sha256']] += 1; decoded_hashes[decoded] += 1
        except (OSError, ValueError, SyntaxError) as exc:
            row['problems'].append(type(exc).__name__+': '+str(exc))
        problems.update(row['problems']); rows.append(row)
    out = DATA_ROOT/'e109'; out.mkdir(exist_ok=True)
    write_once(out/'rows.json', {'rows': rows, 'download_sha256': digest(download)})
    fractions = [r['mask_fraction'] for r in rows if 'mask_fraction' in r]
    result = {'state': 'E109_DiffSeg30k_quarantine_schema_audit_complete',
        'code_sha256': digest(Path(__file__)), 'download_sha256': digest(download),
        'rows_sha256': digest(out/'rows.json'), 'requested_pairs': report['images'],
        'decoded_pairs': len(fractions), 'nondegenerate_pairs': sum(0 < f < 1 for f in fractions),
        'mask_ids_pair_counts': dict(values), 'problems': dict(problems),
        'duplicate_image_body_groups': sum(n > 1 for n in image_hashes.values()),
        'duplicate_decoded_RGB_groups': sum(n > 1 for n in decoded_hashes.values()),
        'mask_area_bins': {f'{lo}_{hi}': sum(lo < f <= hi for f in fractions)
            for lo, hi in [(0, .01), (.01, .05), (.05, .2), (.2, .5), (.5, 1)]},
        'training_allowed': False, 'classifier_scores': 0,
        'limits': 'Native labels use >0, never >127. Empty masks retained and quarantined, not relabelled as authentic. Image bodies are edited derivatives; original COCO ancestry/prompt and cross-role near duplicates unresolved. No CAL/final or universal evidence.',
        'preregistration_note': 'Schema audit after one arbitrary downloaded pair was inspected and found to have an empty mask. Entire fixed512 population retained; no score-based selection/refill.'}
    write_once(out/'audit.json', result); write_once(evidence/'e109_diffseg_audit.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
