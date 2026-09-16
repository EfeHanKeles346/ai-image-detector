"""Offline, score-blind geometry and recorded-processing audit of all E131 TRAIN bodies."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import time
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from pixelproof import processing_inventory as inventory
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e148'
CONTRACT = ROOT / 'contract.json'
EVIDENCE = ML_ROOT.parent / 'evidence'
TRAIN = DATA_ROOT / 'e131/contract.json'


def validate_rows(rows):
    if not rows or len({r['parent_id'] for r in rows}) != len(rows):
        raise ValueError('Unique complete TRAIN parents required')
    for row in rows:
        if row['role'].upper() != 'TRAIN' or type(row['label']) is not int or row['label'] not in (0, 1) or \
                len(row['sha256']) != 64 or not Path(row['path']).resolve().is_relative_to(DATA_ROOT.resolve()):
            raise ValueError('Bound local TRAIN body required')


def freeze():
    if digest(TRAIN) != read(EVIDENCE / 'e131_source_holdout_contract.json')['contract_sha256']:
        raise ValueError('TRAIN contract differs')
    prior = read(TRAIN)
    validate_rows(prior['rows'])
    code = [Path(__file__), Path(inventory.__file__), ML_ROOT/'experiments/e65_acquisition.py',
            ML_ROOT/'experiments/e72_acquisition.py']
    # Pin the code establishing the meanings of the historical provenance fields.
    provenance_code = [ML_ROOT/'experiments'/name for name in (
        'e54_data.py', 'e42_data.py', 'e32_r0_input.py', 'e51_train_cal_realize.py', 'e88_audit.py', 'e42_features.py')]
    for path in provenance_code:
        if str(path) in prior['inputs'] and digest(path) != prior['inputs'][str(path)]:
            raise ValueError('Historical bound code differs')
    inputs = [TRAIN, EVIDENCE/'e131_source_holdout_contract.json'] + code + provenance_code
    c = dict(state='E148_processing_inventory_registered', inputs={str(p): digest(p) for p in inputs},
        parents=len(prior['rows']), max_seconds=1800, max_body_bytes=100*1024**2,
        operation='All existing12525 TRAIN source bodies, full SHA256 verification and lazy geometry/container headers. No image pixels, metadata text, EXIF camera/GPS/serials, model, feature caches, predictions, gallery, CAL/DEV or protected reserves.',
        processing='Classify only explicit publisher-224-resize, e32_fixed_replay and SID original-hash/RAW-render records. All others remain admitted source bodies with unverified upstream processing. E54 native boolean is reported solely as a historical cohort flag.',
        geometry='Report format, width/height, header-only orientation, match to recorded raw or oriented dimensions and source-pixel224/256/512/1024 crop capacity. Capacity is not proof of original camera/generator resolution. Record header errors without excluding parents. SHA/size/role mismatch aborts.',
        metrics='Aggregate by label, source+label and existing source fold+label; no detector scores or outcome-based selections. Do not remove small or derived images or change roles. Header reads cannot certify a complete decode or original processing history.',
        downloads=0, pixel_decodes=0, fits=0, model_scores=0, promotion_allowed=False)
    ROOT.mkdir(exist_ok=True)
    write_once(CONTRACT, c)
    write_once(EVIDENCE/'e148_processing_inventory_contract.json',
               {k:v for k,v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'parents': c['parents'], 'contract_sha256': digest(CONTRACT)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e148_processing_inventory_contract.json')['contract_sha256']:
        raise ValueError('Inventory contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound audit input differs')
    train = read(TRAIN)
    rows = train['rows']
    validate_rows(rows)
    start = time.monotonic()
    deadline = start + c['max_seconds']
    resource_check(deadline)
    write_once(ROOT/'started.json', {'contract_sha256': digest(CONTRACT)})
    records, verified_bytes = [], 0
    for i, row in enumerate(rows):
        if i % 128 == 0:
            resource_check(deadline)
        path = Path(row['path'])
        size = path.stat().st_size
        if not 0 < size <= c['max_body_bytes'] or ('bytes' in row and row['bytes'] != size):
            raise ValueError('Body size differs or exceeds budget')
        with path.open('rb') as stream:
            sha = hashlib.sha256()
            for block in iter(lambda: stream.read(4*1024**2), b''):
                sha.update(block)
            if sha.hexdigest() != row['sha256']:
                raise ValueError('TRAIN body SHA differs')
            verified_bytes += size
            stream.seek(0)
            try:
                details = inventory.inspect_geometry(stream)
                actual = (details['width'], details['height'])
                oriented = actual[::-1] if details['header_orientation'] in (5, 6, 7, 8) else actual
                details['declared_geometry_matches'] = (row['width'], row['height']) in (actual, oriented)
            except (ValueError, OSError, SyntaxError) as exc:
                details = {'header_error': type(exc).__name__}
        records.append(dict(parent_id=row['parent_id'], source=row['source'], label=row['label'],
            fold=train['outer_fold'][row['parent_id']], legacy_native_flag=row.get('native'),
            processing_history=inventory.processing_history(row), **details))
        if (i+1) % 512 == 0:
            print(json.dumps({'E148_verified': i+1, 'seconds': round(time.monotonic()-start)}), flush=True)
    write_once(ROOT/'private_records.json', {'rows': records, 'contract_sha256': digest(CONTRACT)})
    report = dict(state='E148_processing_inventory_complete', contract_sha256=digest(CONTRACT),
        total=inventory.summarize(records),
        by_class={str(label): inventory.summarize([r for r in records if r['label']==label]) for label in (0,1)},
        by_source_class=[dict(source=source, label=label, **inventory.summarize([r for r in records if (r['source'],r['label'])==(source,label)]))
                         for source,label in sorted({(r['source'],r['label']) for r in records})],
        by_fold_class=[dict(fold=fold, label=label, **inventory.summarize([r for r in records if (r['fold'],r['label'])==(fold,label)]))
                       for fold in range(3) for label in (0,1)],
        verified_body_bytes=verified_bytes, private_records_sha256=digest(ROOT/'private_records.json'),
        seconds=time.monotonic()-start, downloads=0, pixel_decodes=0, fits=0, model_scores=0, promotion_allowed=False,
        limits=c['metrics']+' Large source files and native=True do not certify native sensor/generator pixels. No independent-source admission or model-quality improvement claim.')
    write_once(ROOT/'report.json', report)
    write_once(EVIDENCE/'e148_processing_inventory.json', report)
    return {k:v for k,v in report.items() if k not in ('by_source_class','by_fold_class')}


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('Processing inventory is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze, 'run':run}[parser.parse_args().stage](), indent=2))
