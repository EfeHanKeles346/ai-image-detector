"""Inventory pinned SID split lists only; never request image payloads or assign data roles."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import ssl
import urllib.request
import certifi

from experiments.e65_acquisition import digest, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REVISION = '0448b58d8c57459b1550ddf68c160d9212046a25'
BASE = f'https://raw.githubusercontent.com/cchen156/Learning-to-See-in-the-Dark/{REVISION}/'
EXPECTED = {
    'Sony_train_list.txt': 133675, 'Sony_val_list.txt': 16879, 'Sony_test_list.txt': 42927,
    'Fuji_train_list.txt': 119756, 'Fuji_val_list.txt': 15742, 'Fuji_test_list.txt': 38051,
    'README.md': 5610, 'LICENSE.md': 1108,
}
NAME = re.compile(r'(?P<scene>[012]\d{4})_(?P<burst>\d+)_(?P<seconds>\d+(?:\.\d+)?)s\.(ARW|RAF)')


def inventory(camera, split, body):
    scenes, shorts, longs, iso, apertures = set(), set(), set(), Counter(), Counter()
    pairs = []
    for line in body.decode('utf-8').splitlines():
        if not line.strip():
            continue
        short, long, sensitivity, aperture = line.split()
        parsed = []
        for role, value in [('short', short), ('long', long)]:
            p = PurePosixPath(value)
            match = NAME.fullmatch(p.name)
            if not match or p.parts[:2] != (camera, role) or len(p.parts) != 3:
                raise ValueError('unexpected original-capture path')
            if p.suffix != ('.ARW' if camera == 'Sony' else '.RAF'):
                raise ValueError('camera/RAW extension mismatch')
            parsed.append(match.group('scene'))
        if parsed[0] != parsed[1] or parsed[0][0] != {'train': '0', 'val': '2', 'test': '1'}[split]:
            raise ValueError('scene pairing or published split prefix differs')
        if not re.fullmatch(r'ISO\d+', sensitivity) or not re.fullmatch(r'F\d+(?:\.\d+)?', aperture):
            raise ValueError('unexpected capture metadata')
        scenes.add(parsed[0]); shorts.add(short); longs.add(long)
        iso[sensitivity] += 1; apertures[aperture] += 1; pairs.append((short, long))
    if len(pairs) != len(set(pairs)):
        raise ValueError('duplicate listed pair')
    return {
        'listed_pairs': len(pairs), 'unique_short_files': len(shorts),
        'unique_long_files': len(longs), 'filename_scene_groups': len(scenes),
        'scene_ids': sorted(scenes), 'ISO_by_listed_pair': dict(sorted(iso.items())),
        'aperture_by_listed_pair': dict(sorted(apertures.items())),
    }


def main():
    root = DATA_ROOT / 'source_research/sid_metadata'
    root.mkdir(parents=True, exist_ok=True)
    inputs, groups, downloaded = {}, {}, 0
    for name, size in EXPECTED.items():
        remote = ('dataset/' if name.endswith('_list.txt') else '') + name
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(BASE + remote, timeout=30, context=ssl.create_default_context(cafile=certifi.where())) as response:
                body = response.read(size + 1)
            if len(body) != size:
                raise ValueError('pinned metadata size differs')
            with path.open('xb') as output:
                output.write(body)
            downloaded += len(body)
        body = path.read_bytes()
        if len(body) != size:
            raise ValueError('cached metadata size differs')
        inputs[name] = {'url': BASE + remote, 'bytes': size, 'sha256': hashlib.sha256(body).hexdigest()}
        if name.endswith('_list.txt'):
            camera, split, _ = name.split('_')
            groups[f'{camera}/{split}'] = inventory(camera, split, body)
    result = {
        'state': 'SID_pinned_metadata_only_inventory', 'revision': REVISION,
        'inputs': inputs, 'groups': groups, 'metadata_downloaded_bytes_this_run': downloaded,
        'code_sha256': digest(Path(__file__)), 'image_payload_bytes': 0, 'model_scores': 0,
        'data_role': 'UNASSIGNED_NOT_ADMITTED',
        'limits': 'Filename scene groups are not verified independent scenes, especially across cameras. '
                  'Published splits describe upstream use only; none is our TRAIN/DEV/final. '
                  'No original RAW or rendered/model-output image acquired. README states MIT; '
                  'LICENSE wording covers software/documentation. No additional image terms inferred.',
    }
    write_once(root / 'inventory.json', result)
    write_once(ML_ROOT.parent / 'evidence/sid_metadata_inventory.json', result)
    print(json.dumps({k: {f: v[f] for f in ['listed_pairs', 'unique_short_files', 'unique_long_files', 'filename_scene_groups']} for k, v in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
