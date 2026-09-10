"""Offline schema amendment: official FiveK licence lists use exact filename stems."""
from collections import Counter
import json
from pathlib import Path

from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments import e56_fivek_metadata as original


def assign(rows, lists):
    stems = [{line.strip() for line in text.splitlines() if line.strip()} for text in lists]
    if len(stems) != 2 or stems[0] & stems[1]:
        raise ValueError('ambiguous licence lists')
    names = [Path(r['filename']).stem for r in rows]
    if len(set(names)) != len(names) or set(names) != stems[0] | stems[1]:
        raise ValueError('incomplete/extra/duplicate licence identities')
    result = []
    for row, stem in zip(rows, names, strict=True):
        if Path(row['filename']).suffix.lower() != '.dng': raise ValueError('DNG originals only')
        result.append(row | {'license': 'LicenseAdobe.txt' if stem in stems[0] else 'LicenseAdobeMIT.txt'})
    return result


def run():
    root = original.ROOT
    previous = json.loads(original.CONTRACT.read_text())
    if digest(original.__file__) != previous['code_sha256']:
        raise ValueError('original metadata code changed')
    if digest(original.CONTRACT) != json.loads((EVIDENCE/'e56_fivek_metadata_contract.json').read_text())['contract_sha256']:
        raise ValueError('original metadata contract changed')
    files = [root/(Path(p).name if p else 'index.html') for p in original.PATHS]
    sizes = [p.stat().st_size for p in files]
    if sum(sizes) > previous['total_cap'] or max(sizes) > previous['per_response_cap']:
        raise ValueError('metadata cap exceeded')
    config = {'state': 'FiveK_offline_schema_amendment_v2', 'code_sha256': digest(__file__),
              'original_contract_sha256': digest(original.CONTRACT),
              'original_code_sha256': digest(original.__file__),
              'input_sha256': {str(p): digest(p) for p in files},
              'v1_failure': 'Official licence lists use extensionless filenames; exact basename join rejected a0001-jmac_DSC1459.dng.',
              'method': 'Exact validated DNG filename stem, unique complete disjoint licence union; no fuzzy matching.',
              'image_downloads': 0, 'new_network_bytes': 0}
    fixed_write(root/'review_contract_v2.json', config)
    fixed_write(EVIDENCE/'e56_fivek_review_contract_v2.json', config)
    rows = original.parse_index(files[0].read_text())
    if len(rows) != 5000 or {r['id'] for r in rows} != set(range(1, 5001)):
        raise ValueError('incomplete index')
    rows = assign(rows, [p.read_text() for p in files[3:]])
    downloads = [{'url': original.BASE+p, 'path': str(file), 'bytes': size, 'sha256': config['input_sha256'][str(file)]}
                 for p, file, size in zip(original.PATHS, files, sizes, strict=True)]
    value = {'state': 'FiveK_metadata_complete_v2', 'contract_sha256': digest(root/'review_contract_v2.json'),
             'downloads': downloads, 'total_metadata_bytes': sum(sizes), 'image_bytes': 0, 'parents': len(rows),
             'rows': rows, 'licence_counts': dict(Counter(r['license'] for r in rows)),
             'strata': {key: dict(Counter(r[key] for r in rows)) for key in ('subject', 'light', 'location', 'time')},
             'restrictions': ['Research only, non-commercial; retain notices and cite the original dataset.',
                              'Old SLR source, not modern phones; no matched AI.',
                              'Metadata completion does not admit images to TRAIN or establish duplicate-free data.']}
    fixed_write(root/'manifest_v2.json', value)
    receipt = {k: v for k, v in value.items() if k != 'rows'} | {'manifest_sha256': digest(root/'manifest_v2.json')}
    fixed_write(EVIDENCE/'e56_fivek_metadata.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
