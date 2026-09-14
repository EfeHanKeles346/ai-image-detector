"""Read existing CocoGlide pairs/masks; no extraction, classifier or role reassignment."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pixelproof.project_paths import LEGACY_DATA_ROOT, ML_ROOT, WORK_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_auth(root, pointer):
    # The existing compilation preserves the original extension then adds .png.
    # Resolve this documented encoding only; never guess by a partial basename.
    requested = (root / pointer).resolve()
    if not requested.is_relative_to(root.resolve()):
        raise ValueError('unsafe_auth_pointer')
    candidates = [p for p in (requested, Path(str(requested) + '.png')) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError('ambiguous_auth_pointer' if candidates else 'missing_auth_file')
    return candidates[0]


def audit():
    root = LEGACY_DATA_ROOT / 'manipulation_test'
    images = sorted(p for p in (root / 'CocoGlide/manip').glob('*.png')
                    if '.mask.' not in p.name and not p.name.startswith('._'))
    if not images:
        raise FileNotFoundError('No previously prepared CocoGlide pairs found')
    rows, errors, parent_hashes = [], Counter(), {}
    for index, path in enumerate(images):
        mask, metadata = path.with_suffix('.mask.png'), path.with_suffix('.json')
        row = {'image': str(path), 'image_sha256': sha(path), 'old_first120_exposure_possible': index < 120,
               'source_exposed': True, 'problems': []}
        try:
            with Image.open(path) as image:
                size = image.size
            row['width'], row['height'] = size
            if not metadata.is_file():
                row['problems'].append('missing_metadata')
            else:
                m = json.loads(metadata.read_text())
                row['metadata_sha256'] = sha(metadata)
                pointer = m.get('auth')
                if not isinstance(pointer, str) or not pointer:
                    row['problems'].append('missing_auth_pointer')
                else:
                    try:
                        auth = resolve_auth(root, pointer)
                    except ValueError as exc:
                        row['problems'].append(str(exc))
                    else:
                        row['auth_pointer'] = pointer
                        row['auth'] = str(auth.relative_to(root))
                        row['auth_sha256'] = sha(auth)
                        parent_hashes.setdefault(row['auth_sha256'], []).append(path.name)
            if not mask.is_file():
                row['problems'].append('missing_mask')
            else:
                row['mask_sha256'] = sha(mask)
                with Image.open(mask) as opened:
                    if opened.size != size:
                        row['problems'].append('mask_geometry_mismatch')
                    array = np.asarray(opened.convert('L'))
                row['mask_fraction'] = float(np.mean(array > 127))
                row['mask_values'] = np.unique(array).tolist()
                if row['mask_fraction'] == 0:
                    row['problems'].append('empty_mask')
                elif row['mask_fraction'] == 1:
                    row['problems'].append('full_image_mask')
                if not set(row['mask_values']).issubset({0, 255}):
                    row['problems'].append('nonbinary_mask')
        except (OSError, ValueError, SyntaxError):
            row['problems'].append('decode_or_metadata_error')
        errors.update(row['problems'])
        rows.append(row)
    detailed = WORK_ROOT / 'model2_local_inventory_2026-09-14.json'
    detailed.parent.mkdir(parents=True, exist_ok=True)
    with detailed.open('x') as stream:
        json.dump({'rows': rows}, stream, indent=2, sort_keys=True)
        stream.write('\n')
    fractions = [r['mask_fraction'] for r in rows if 'mask_fraction' in r]
    auth = [p for p in (root / 'CocoGlide/auth').glob('*.png') if not p.name.startswith('._')]
    result = {'state': 'existing_CocoGlide_inventory_complete', 'images': len(rows), 'auth_files': len(auth),
              'complete_nondegenerate_pairs': sum(not r['problems'] for r in rows),
              'problems': dict(errors), 'unique_exact_auth_parents': len(parent_hashes),
              'multiple_edits_same_exact_parent': sum(len(v) > 1 for v in parent_hashes.values()),
              'old_first120_exposure_possible': min(120, len(rows)),
              'mask_area_bins': {'0_to_1_percent': sum(0 < x <= .01 for x in fractions),
                 '1_to_5_percent': sum(.01 < x <= .05 for x in fractions),
                 '5_to_20_percent': sum(.05 < x <= .20 for x in fractions),
                 '20_to_50_percent': sum(.20 < x <= .50 for x in fractions),
                 '50_to_100_percent': sum(.50 < x < 1 for x in fractions)},
              'mask_fraction_median': float(np.median(fractions)),
              'inventory_sha256': sha(detailed), 'code_sha256': sha(Path(__file__)),
              'auth_resolution': 'Exact metadata path or its documented added .png encoding; reject missing/ambiguous/outside-root mappings.',
              'classifier_calls': 0, 'downloads': 0, 'training_roles_assigned': False,
              'limits': 'Exact parent hashes only; near duplicates/semantic scenes/generator provenance still require audit. Source historically consumed by E17/E18. Unselected files are not an independent unseen-source final. No localisation performance measured.'}
    out = ML_ROOT.parent / 'evidence/model2_local_inventory_2026-09-14.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
