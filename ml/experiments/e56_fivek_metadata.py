"""Archive official FiveK metadata/licences only; never download image bodies."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

import certifi
import requests

from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e55_resume import inspect_safety
from pixelproof.project_paths import DATA_ROOT

BASE = 'https://data.csail.mit.edu/graphics/fivek/'
ROOT = DATA_ROOT/'e56/fivek_metadata'
PATHS = ('', 'legal/LicenseAdobe.txt', 'legal/LicenseAdobeMIT.txt',
         'legal/filesAdobe.txt', 'legal/filesAdobeMIT.txt')
CONTRACT = ROOT/'contract.json'


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.current = None; self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.current = {'text': [], 'links': []}
        if self.current is not None and tag == 'a':
            self.current['links'].extend(v for k, v in attrs if k == 'href')

    def handle_data(self, data):
        if self.current is not None: self.current['text'].append(data)

    def handle_endtag(self, tag):
        if tag == 'tr' and self.current is not None:
            self.rows.append(self.current); self.current = None


def parse_index(html):
    parser = IndexParser(); parser.feed(html); rows = []
    for row in parser.rows:
        links = [urljoin(BASE, link) for link in row['links'] if link.lower().endswith('.dng')]
        if not links: continue
        if len(links) != 1 or not links[0].startswith(BASE):
            raise ValueError('ambiguous/off-site original URL')
        name = Path(urlsplit(links[0]).path).name
        match = re.fullmatch(r'a(\d{4})-.+\.dng', name, re.IGNORECASE)
        if not match: raise ValueError('unrecognized FiveK identity')
        text = ' '.join(' '.join(row['text']).split())
        fields = re.search(r'Subject:\s*(.*?)\s*Light:\s*(.*?)\s*Location:\s*(.*?)\s*Time:\s*(.*?)\s*EXIF:\s*(.+)', text)
        if not fields or not all(fields.groups()): raise ValueError('incomplete semantic metadata')
        rows.append({'id': int(match[1]), 'filename': name, 'original_url': links[0],
                     **dict(zip(('subject', 'light', 'location', 'time', 'exif_text'), fields.groups(), strict=True))})
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate FiveK identity')
    return rows


def license_for(filename, lists):
    # Compare exact basenames; no substring or numeric-id-only licence guessing.
    matches = [name for name, names in lists.items() if filename in names]
    if len(matches) != 1: raise ValueError(f'missing/ambiguous licence for {filename}')
    return matches[0]


def freeze():
    if inspect_safety(): raise RuntimeError('power/storage safety failed')
    value = {'state': 'FiveK_metadata_only_frozen', 'code_sha256': digest(__file__),
             'urls': [BASE+p for p in PATHS], 'total_cap': 12*1024**2, 'per_response_cap': 8*1024**2,
             'expected_parents': 5000, 'image_downloads': 0, 'model_scores': 0,
             'role': 'source review; no parent admitted to TRAIN/CAL/test yet'}
    fixed_write(CONTRACT, value)
    fixed_write(EVIDENCE/'e56_fivek_metadata_contract.json', value|{'contract_sha256': digest(CONTRACT)})
    return value


def acquire():
    config = json.loads(CONTRACT.read_text())
    receipt = json.loads((EVIDENCE/'e56_fivek_metadata_contract.json').read_text())
    if digest(CONTRACT) != receipt['contract_sha256'] or digest(__file__) != config['code_sha256']:
        raise ValueError('frozen metadata contract changed')
    total = 0; bodies = {}; downloads = []
    for relative in PATHS:
        if inspect_safety(): raise RuntimeError('power/storage safety failed')
        url = BASE+relative; name = Path(relative).name if relative else 'index.html'
        path = ROOT/name
        if path.exists():
            raw = path.read_bytes()
        else:
            chunks = []; size = 0
            with requests.get(url, stream=True, timeout=(10, 30), verify=certifi.where()) as response:
                response.raise_for_status()
                if not response.url.startswith(BASE): raise ValueError('unexpected metadata redirect')
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > config['per_response_cap'] or total+size > config['total_cap']:
                        raise ValueError('metadata byte cap exceeded')
                    chunks.append(chunk)
            raw = b''.join(chunks)
            if inspect_safety(): raise RuntimeError('power/storage safety failed')
            with path.open('xb') as stream: stream.write(raw)
        total += len(raw)
        if len(raw) > config['per_response_cap'] or total > config['total_cap']:
            raise ValueError('cached metadata cap exceeded')
        bodies[relative] = raw.decode('utf-8')
        downloads.append({'url': url, 'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    rows = parse_index(bodies[''])
    if len(rows) != config['expected_parents'] or {r['id'] for r in rows} != set(range(1, 5001)):
        raise ValueError('incomplete FiveK index')
    lists = {name: {Path(line.strip()).name for line in bodies['legal/files'+name+'.txt'].splitlines() if line.strip()}
             for name in ('Adobe', 'AdobeMIT')}
    for row in rows: row['license'] = 'License'+license_for(row['filename'], lists)+'.txt'
    value = {'state': 'FiveK_metadata_complete', 'contract_sha256': digest(CONTRACT),
             'downloads': downloads, 'total_metadata_bytes': total, 'image_bytes': 0,
             'parents': len(rows), 'rows': rows, 'licence_counts': dict(Counter(r['license'] for r in rows)),
             'subject_counts': dict(Counter(r['subject'] for r in rows)),
             'restrictions': ['Research-only, non-commercial; retain both licences/copyright and cite original dataset.',
                              'Old SLR data, not modern phones; no paired synthetic counterparts.',
                              'Metadata identity/license check is not image integrity, duplicate screening or TRAIN admission.']}
    fixed_write(ROOT/'manifest.json', value)
    fixed_write(EVIDENCE/'e56_fivek_metadata.json', {k: v for k, v in value.items() if k != 'rows'} |
                {'manifest_sha256': digest(ROOT/'manifest.json')})
    return {k: v for k, v in value.items() if k != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'acquire'])
    args = parser.parse_args(); print(json.dumps({'freeze': freeze, 'acquire': acquire}[args.phase](), indent=2))
