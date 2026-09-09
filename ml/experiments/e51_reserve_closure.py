"""Close local reserve identities and supplementary pixel checks without network access."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.e51_offline_bodies import OUTPUT as LOCATORS
from experiments.e51_prefit_audit import fingerprint, load_inputs, cross_role_matches
from experiments.e51_protected_inventory import identity_rows, keys
from experiments.e51_protected_pixels import LOCATOR_SHA256
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e51/audit'
REPORT = ROOT / 'reserve_closure_v1.json'
EVIDENCE = ML_ROOT.parent / 'evidence/e51_reserve_closure.json'
PINS = {
    'e49/open_components_contract.json': 'c6f2cfb054a52816581d523b3d3ef04311add28487a5cc9ed6d4150d6b27f794',
    'e49/open_components_contract_v2.json': '1d4e184c27cb87cf832045a23b6966f382673c3bcd8342a900c07130bd9182aa',
    'e49/openfake/source_contract_e49c_unscored.json': '0abae56af862c9b402ef5ef594a21181cbbb7f72ba7495a491b0389bfdfcd702',
    'e49/open_components_v2/commons_download_receipt_unscored.json': '2511f0ad0ad3f22e72ab5bf04da69fed0e9efac2bb4768d5365cf734fb5a7e04',
    'e49/openfake/download_receipt_unscored.json': '4dfb942c215bbe15e15aea81e9a6c6873670a0bb5019c1c64f9672fe34b326c2',
    'e49_d1_dotting/source_contract.json': '170f70db128becfe9986dffc6a3e150ec11b182d51ee496dc40ac530204eed36',
}


def verify_json(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'input changed: {path}')
    return json.loads(raw)


def connect(index, row, digest):
    for key in keys(row):
        index[key].add(digest)


def close():
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError('reserve closure already exists')
    locators = verify_json(LOCATORS, LOCATOR_SHA256)
    index = defaultdict(set)
    main_sha = {r['sha256'] for r in locators['rows']}
    for row in locators['rows']:
        connect(index, row, row['sha256'])
        for reference in row['protected_by']:
            connect(index, reference, row['sha256'])
    meta = []
    for source in locators['sources']:
        for row in identity_rows(verify_json(Path(source['path']), source['sha256'])):
            if row.get('sha256') in main_sha:
                connect(index, row, row['sha256'])
            if not row.get('path') and not row.get('member'):
                meta.append((source['path'], row))
    documents = {rel: verify_json(DATA_ROOT / rel, sha) for rel, sha in PINS.items()}
    reserves, extras = [], {}
    for rel, document in documents.items():
        for row in identity_rows(document):
            reserves.append(row)
            item = dict(row)
            if rel == 'e49_d1_dotting/source_contract.json':
                item['path'] = str(DATA_ROOT / 'e49_d1_dotting/repository' / row['image_path'])
            if item.get('path') and item.get('sha256'):
                path = Path(item['path'])
                if not path.is_file():
                    raise FileNotFoundError(f'previously downloaded reserve missing: {path}')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['bytes']:
                    raise ValueError(f'reserve bytes changed: {path}')
                connect(index, row, item['sha256'])
                if item['sha256'] not in main_sha:
                    extras[item['sha256']] = {**fingerprint(raw), 'parent_id': str(
                        row.get('parent_id') or row.get('identity') or row.get('record_id')),
                        'condition': 'protected_reserve', 'label': row.get('label', -1)}
    # All 240 StyleGAN2 files were extracted during E49, although only 200 were selected.
    style_root = DATA_ROOT / 'e49/open_components_v2/stylegan2_reserve'
    for row in documents['e49/open_components_contract_v2.json']['aigc']['rows']:
        prefix = f"{Path(row['shard']).stem}-{int(row['row_index']):06d}"
        paths = [p for p in style_root.glob(prefix + '.*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        if len(paths) != 1:
            raise ValueError(f'local StyleGAN2 reserve is missing/ambiguous: {row["identity"]}')
        raw = paths[0].read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        connect(index, row, sha)
        if sha not in main_sha:
            # Unselected file hashes were not retained in E49's selected manifest. Validate
            # the bytes against the original, local Parquet coordinate before accepting them.
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(DATA_ROOT / 'TheKernel01__AIGC-Detection-Benchmark/data' / row['shard'])
            offset = int(row['row_index'])
            for group in range(pf.num_row_groups):
                count = pf.metadata.row_group(group).num_rows
                if offset < count:
                    original = pf.read_row_group(group, columns=['image','label','generator']).slice(offset,1).to_pylist()[0]
                    if (original['label'] != 1 or original['generator'] != 14
                            or original['image']['bytes'] != raw):
                        raise ValueError('StyleGAN2 reserve disagrees with publisher coordinate')
                    break
                offset -= count
            else:
                raise ValueError('StyleGAN2 coordinate out of range')
            extras[sha] = {**fingerprint(raw), 'parent_id': row['identity'],
                           'condition': 'protected_reserve', 'label': 1}
    unjoined = []
    joined = 0
    for source, row in meta:
        if any(index.get(key) for key in keys(row)):
            joined += 1
        else:
            unjoined.append({'source':source, 'identity':str(row.get('identity') or row.get('record_id') or row.get('parent_id'))})
    reserve_keys = set().union(*(keys(row) for row in reserves))
    train, cal = load_inputs()
    identity_hits = [r['parent_id'] for r in train + cal if keys(r) & reserve_keys]
    # Reuse only the already verified E51 canonical cache, bound to the frozen input digests.
    queries = []
    for row in train + cal:
        facts = json.loads((ROOT / 'fingerprints_v1' / (row['sha256'] + '.json')).read_text())
        if facts['sha256'] != row['sha256']:
            raise ValueError('query fingerprint binding changed')
        queries.append({**row, **facts})
    matches = cross_role_matches(queries, list(extras.values()))
    old_rows = documents['e49/open_components_contract.json']['commons']['rows']
    old_unmaterialized = [row['identity'] for row in old_rows if not any(index.get(k) for k in keys(row))]
    report = {'schema_version':1, 'state':'reserve_identity_and_local_pixel_audit_complete',
              'input_sha256':PINS, 'joined_metadata_rows':joined, 'unjoined_metadata_rows':unjoined,
              'metadata_rows_total':len(meta), 'additional_local_bodies':len(extras),
              'additional_fingerprints':list(extras.values()), 'reserved_identity_overlap':identity_hits,
              'supplementary_matches':matches, 'unmaterialized_superseded_commons_identities':old_unmaterialized,
              'model_scores_created':0, 'image_bytes_downloaded':0, 'training_authorized':False,
              'limitation':'Superseded never-downloaded Commons reserves remain identity-forbidden; their unavailable pixels are not claimed as checked. Requires completed main protected-pixel audit and overlap review.'}
    raw = _write_atomic(REPORT, report)
    summary = {k:v for k,v in report.items() if k not in {'additional_fingerprints','supplementary_matches','unmaterialized_superseded_commons_identities'}}
    summary.update(report_sha256=hashlib.sha256(raw).hexdigest(),
                   supplementary_matched_parent_pairs=len({(r['train_parent'],r['cal_parent']) for r in matches}),
                   unmaterialized_superseded_commons_count=len(old_unmaterialized))
    _write_atomic(EVIDENCE, summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(close(), indent=2))
