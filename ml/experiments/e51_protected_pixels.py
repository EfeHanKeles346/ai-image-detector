"""Offline, resumable canonical protected-image audit. No detector is imported."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import zipfile

from experiments.e51_offline_bodies import OUTPUT as LOCATORS, safe_zip_info
from experiments.e51_prefit_audit import (
    VERSION, fingerprint, cross_role_matches, load_inputs,
)
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT, WORK_ROOT

LOCATOR_SHA256 = "a51cb45736641fa4e93c1d2171f17c06bf939cd4257effd78c8f6498c711eb81"
ROOT = DATA_ROOT / "e51/audit"
DATABASE = WORK_ROOT / "e51/canonical_fingerprints.sqlite3"
PROGRESS = ROOT / "protected_pixels_progress.json"
REPORT = ROOT / "protected_pixels_v1.json"
EVIDENCE = ML_ROOT.parent / "evidence/e51_protected_pixels.json"
LARGE_PROTECTED = {
    '9e8012f22251f348ff77dd68d39bbfef36ed63ef3e7c23fea5b25eb5bf95d001': 102960000,
    '71dde8f819a7bff749e66dd30761a28a5ae36ba3c4e67f3c63761d37c399cd46': 109721600,
    '0546996836547dc4ccafc76ca1d0d351c749c8943212decfd215f669ef86a3c8': 178562880,
    'e76b5f0ba9c1f524433d674a124d38bb8d42fbfbe8b1e6d89186c773efc015ec': 102960000,
    '7c74e8fecf0336f43fdf130d3ba5815fd9b77b9402a9aae895a5a780fa52a36b': 109721600,
    '884e6d8a3a76c64bb7f20dac03fe57ca6f9c990bc1aaa736b2a17c59c94db889': 134657146,
}
LARGE_LOCK = threading.Lock()


def read_body(row, bundles):
    if not 0 < int(row['bytes']) <= 200 * 1024**2:
        raise ValueError('body exceeds byte budget')
    if row['kind'] == 'file':
        path = Path(row['path'])
        if path.stat().st_size != row['bytes']:
            raise ValueError(f"protected file size changed: {path}")
        raw = path.read_bytes()
    elif row['kind'] == 'zip':
        bundle = bundles[row['archive']]
        info = safe_zip_info(bundle, row['member'])
        if info.file_size != row['bytes'] or f'{info.CRC:08x}' != row['crc32']:
            raise ValueError('protected ZIP member size/CRC changed')
        raw = bundle.read(row['member'])
    else:
        raise ValueError('unbound body kind')
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError(f"protected bytes changed: {row.get('identity')}")
    return raw


def cached_fingerprint(raw, previous=None):
    sha = hashlib.sha256(raw).hexdigest()
    if previous is not None:
        if previous.get('version') != VERSION or previous.get('sha256') != sha:
            raise ValueError('canonical cache binding mismatch')
        if int(previous['bytes']) != len(raw):
            raise ValueError('canonical cache byte count mismatch')
        return previous
    return fingerprint(raw)


def protected_fingerprint(raw, previous=None):
    sha = hashlib.sha256(raw).hexdigest()
    if sha not in LARGE_PROTECTED:
        return cached_fingerprint(raw, previous)
    # Full-resolution identity, not a thumbnail approximation. No global PIL limit changes.
    with LARGE_LOCK:
        result = (cached_fingerprint(raw, previous) if previous is not None
                  else fingerprint(raw, max_pixels=LARGE_PROTECTED[sha]))
        if result['width'] * result['height'] != LARGE_PROTECTED[sha]:
            raise ValueError('allowlisted protected geometry changed')
        return result


def audit(workers=4, batch_size=128):
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError('protected pixel report already frozen')
    raw = LOCATORS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCATOR_SHA256:
        raise ValueError('protected locator manifest changed')
    bound = json.loads(raw)
    if bound['unresolved_body_rows'] or bound['bodies_without_prior_sha256']:
        raise ValueError('protected locator coverage incomplete')
    rows = bound['rows']
    train, cal = load_inputs()
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS fingerprints (version TEXT, sha TEXT, facts TEXT, PRIMARY KEY(version,sha))')
        cache = {sha: json.loads(facts) for sha, facts in db.execute(
            'SELECT sha,facts FROM fingerprints WHERE version=?', (VERSION,))}
        done = []
        with ExitStack() as stack, ThreadPoolExecutor(max_workers=workers) as pool:
            bundles = {path: stack.enter_context(zipfile.ZipFile(path))
                       for path in sorted({r['archive'] for r in rows if r['kind'] == 'zip'})}
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                def process(row):
                    body = read_body(row, bundles)
                    try:
                        return protected_fingerprint(body, cache.get(row['sha256']))
                    except ValueError as error:
                        raise ValueError(f"{row['identity']}: {error}") from error
                facts = list(pool.map(process, batch))
                with db:
                    db.executemany('INSERT OR REPLACE INTO fingerprints VALUES (?,?,?)',
                                   [(VERSION, f['sha256'], json.dumps(f, sort_keys=True)) for f in facts])
                cache.update({f['sha256']: f for f in facts})
                for row, fact in zip(batch, facts, strict=True):
                    done.append({**fact, 'parent_id': row['identity'], 'label': -1,
                                 'condition': 'protected_body',
                                 'e49': any(Path(ref['manifest']).relative_to(DATA_ROOT).parts[0].startswith('e49')
                                            for ref in row['protected_by'])})
                progress = {'state': 'verifying_protected_bodies', 'verified_locations': len(done),
                            'total_locations': len(rows), 'model_scores_created': 0,
                            'training_authorized': False}
                _write_atomic(PROGRESS, progress)
                print(json.dumps(progress), flush=True)
            queries = []
            for start in range(0, len(train + cal), batch_size):
                batch = (train + cal)[start:start + batch_size]
                def process_query(row):
                    raw = read_body({**row, 'kind': 'file'}, {})
                    return {**row, **cached_fingerprint(raw, cache.get(row['sha256']))}
                result = list(pool.map(process_query, batch))
                queries.extend(result)
                with db:
                    db.executemany('INSERT OR REPLACE INTO fingerprints VALUES (?,?,?)',
                                   [(VERSION, f['sha256'], json.dumps({k: f[k] for k in (
                                       'version','sha256','bytes','width','height','pixel_sha256',
                                       'canonical_dhash','phash63')}, sort_keys=True)) for f in result])
                print(json.dumps({'state': 'verifying_queries', 'verified': len(queries),
                                  'total': len(train + cal)}), flush=True)
        new_queries = [r for r in queries if r['source'] in {'SCIMD-17', 'SCMI30-IITRPR'}]
        all_unique = {r['sha256']: r for r in done}
        e49_unique = {r['sha256']: r for r in done if r['e49']}
        matches = {
            'new_sources_vs_protected': cross_role_matches(new_queries, list(all_unique.values())),
            'all_e51_vs_e49': cross_role_matches(queries, list(e49_unique.values())),
        }
        report = {'schema_version': 1, 'state': 'protected_pixels_checked_review_required',
                  'locator_manifest_sha256': LOCATOR_SHA256, 'canonical_version': VERSION,
                  'verified_locations': len(done), 'unique_protected_bodies': len(all_unique),
                  'unique_e49_bodies': len(e49_unique), 'verified_query_observations': len(queries),
                  'matched_parent_pairs': {name: len({(r['train_parent'], r['cal_parent']) for r in values})
                                           for name, values in matches.items()},
                  'matches': matches, 'metadata_only_rows_not_closed': bound['metadata_only_rows'],
                  'model_scores_created': 0, 'image_bytes_downloaded': 0,
                  'training_authorized': False,
                  'large_protected_only_exceptions': LARGE_PROTECTED,
                  'boundary': 'Review candidates and remaining reserve identities before fitting; no label or model decision is inferred from perceptual matches.'}
        raw = _write_atomic(REPORT, report)
        summary = {k: v for k, v in report.items() if k != 'matches'}
        summary['report_sha256'] = hashlib.sha256(raw).hexdigest()
        _write_atomic(EVIDENCE, summary)
        _write_atomic(PROGRESS, summary)
        return summary
    except Exception as error:
        _write_atomic(PROGRESS, {'state': 'failed_closed', 'error': f'{type(error).__name__}: {error}',
                                'verified_locations': len(done) if 'done' in locals() else 0,
                                'total_locations': len(rows),
                                'training_authorized': False, 'model_scores_created': 0})
        raise
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    print(json.dumps(audit(args.workers), indent=2))
