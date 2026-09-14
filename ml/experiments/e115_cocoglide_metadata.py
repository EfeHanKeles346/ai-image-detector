"""Read author CocoGlide ZIP catalog and small text metadata only, without image members."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
import requests
from experiments.e65_acquisition import digest, write_once
from experiments.e72_acquisition import PinnedRanges
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

URL = 'https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip'
EXPECTED = {'bytes': 123161479, 'etag': '"7574b87-60269d48b6230"',
            'last_modified': 'Tue, 08 Aug 2023 14:03:40 GMT'}


def head():
    with requests.head(URL, allow_redirects=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        return {'bytes': int(r.headers['Content-Length']), 'etag': r.headers['ETag'],
                'last_modified': r.headers.get('Last-Modified')}


def audit():
    root = DATA_ROOT/'e115'; root.mkdir(exist_ok=True)
    if (root/'catalog.json').exists(): raise FileExistsError('Author metadata already recorded')
    if head() != EXPECTED: raise ValueError('Observed author archive identity differs')
    source = PinnedRanges(URL, EXPECTED, 8*1024**2, time.monotonic()+180)
    texts = []; records = []
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename:
                raise ValueError('Unsafe archive name')
            records.append({'name': info.filename, 'bytes': info.file_size,
                            'crc32': f'{info.CRC:08x}', 'directory': info.is_dir()})
            if not info.is_dir() and name.suffix.lower() in ('.txt', '.csv', '.json', '.md'):
                if info.file_size > 2*1024**2: continue
                body = archive.read(info)  # ZIP CRC checked; no image member is read.
                local = root/f'metadata_{len(texts):03d}.txt'; local.write_bytes(body)
                texts.append({'name': info.filename, 'path': str(local), 'bytes': len(body),
                              'sha256': digest(local)})
    if head() != EXPECTED: raise ValueError('Archive identity changed')
    write_once(root/'catalog.json', {'url': URL, 'identity': EXPECTED, 'records': records, 'texts': texts})
    result = {'state': 'E115_author_CocoGlide_metadata_inspected', 'url': URL, 'identity': EXPECTED,
        'code_sha256': digest(Path(__file__)), 'catalog_sha256': digest(root/'catalog.json'),
        'catalog_entries': len(records), 'extensions': dict(Counter(PurePosixPath(r['name']).suffix.lower()
            for r in records if not r['directory'])), 'text_metadata_members': texts,
        'range_bytes': source.used, 'image_members_downloaded': 0, 'training_allowed': False,
        'purpose': 'Search author package for original COCO identity/prompt/parent mappings lost in compilation. Catalog/text inspection only; no new model scores or full ZIP download.',
        'limits': 'Publisher ETag/length/Last-Modified and ZIP-member CRC only, not whole-archive SHA verification. Metadata presence alone does not admit TRAIN or independent evaluation.'}
    write_once(root/'report.json', result)
    write_once(ML_ROOT.parent/'evidence/e115_cocoglide_metadata.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
