"""Bounded remote ZIP directories for SID; no archive image member is opened."""
import json
from pathlib import Path
import time
import zipfile

from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import PinnedRanges, head_identity
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    root = DATA_ROOT / 'source_research/sid_metadata'
    metadata = read(root / 'inventory.json')
    if digest(root / 'inventory.json') != digest(ML_ROOT.parent / 'evidence/sid_metadata_inventory.json'):
        raise ValueError('metadata receipt differs')
    reports = []
    for camera in ['Sony', 'Fuji']:
        target = root / f'{camera}_archive.json'
        if target.exists():
            raise FileExistsError('archive inventory already recorded')
        url = f'https://storage.googleapis.com/isl-datasets/SID/{camera}2025.zip'
        identity = head_identity(url)
        source = PinnedRanges(url, identity, 2 * 1024**2, time.monotonic() + 120)
        with zipfile.ZipFile(source) as archive:
            entries = [{'filename': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size,
                        'compression': x.compress_type, 'crc32': f'{x.CRC:08x}',
                        'header_offset': x.header_offset, 'directory': x.is_dir()} for x in archive.infolist()]
        by_name = {x['filename']: x for x in entries}
        if len(entries) != len(by_name):
            raise ValueError('duplicate ZIP member names')
        groups = {}
        for split in ['train', 'val', 'test']:
            name = f'{camera}_{split}_list.txt'
            path = root / name
            if digest(path) != metadata['inputs'][name]['sha256']:
                raise ValueError('split list changed')
            pairs = [line.split()[:2] for line in path.read_text().splitlines() if line.strip()]
            short = sorted({p[0].removeprefix('./') for p in pairs})
            long = sorted({p[1].removeprefix('./') for p in pairs})
            missing = [p for p in short + long if p not in by_name]
            groups[split] = {
                'short_files': len(short), 'long_files': len(long), 'missing_listed_members': missing,
                'long_raw_bytes': sum(by_name[p]['bytes'] for p in long if p in by_name),
                'long_compressed_bytes': sum(by_name[p]['compressed_bytes'] for p in long if p in by_name),
                'short_raw_bytes': sum(by_name[p]['bytes'] for p in short if p in by_name),
            }
        if head_identity(url) != identity:
            raise ValueError('remote archive changed')
        result = {'camera': camera, 'url': url, 'identity': identity, 'entries': entries,
                  'split_coverage': groups, 'range_bytes': source.used, 'image_members_opened': 0,
                  'metadata_sha256': digest(root / 'inventory.json'), 'code_sha256': digest(Path(__file__)),
                  'role': 'UNASSIGNED_NOT_ADMITTED'}
        write_once(target, result)
        reports.append({k: v for k, v in result.items() if k != 'entries'} |
                       {'entries': len(entries), 'catalog_path': str(target), 'catalog_sha256': digest(target)})
        print(json.dumps(reports[-1]), flush=True)
    write_once(ML_ROOT.parent / 'evidence/sid_archive_inventory.json', {
        'state': 'SID_remote_directories_only_complete', 'archives': reports,
        'image_members_opened': 0, 'model_scores': 0, 'data_admission': False,
        'limits': 'No RAW decode, scene independence, overlap or model-quality audit. '
                  'Counts are source metadata, not a training selection or claim of data rights.',
    })


if __name__ == '__main__':
    main()
