"""Registered format strata on existing E131 held-out RR scores; no inference."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from experiments.e134_source_scores import distribution, paired, vector
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e144'
BASE = DATA_ROOT / 'e131'
HEADERS = DATA_ROOT / 'e143/private_header_records.json'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
CONDITIONS = ['clean', 'assigned_transport', 'q75', 'social_q75']
BRANCHES = ['center_control', 'full_frame']


def align_headers(rows, headers):
    """Require the complete active RR roster, matching identity, label and source."""
    ids = [r['parent_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate active parent')
    expected = {r['parent_id']: (i, r) for i, r in enumerate(rows)
                if r['source'].startswith('rr:')}
    actual = {r['parent_id']: r for r in headers}
    if not expected or len(actual) != len(headers) or set(actual) != set(expected):
        raise ValueError('Complete unique RR header roster required')
    result = []
    for parent, (i, row) in expected.items():
        header = actual[parent]
        if str(row['role']).upper() != 'TRAIN' or row['label'] not in (0, 1) or \
                header['source'] != row['source'] or header['label'] != row['label']:
            raise ValueError('Header identity/class/role differs')
        if header.get('header_error') or header.get('format') not in ('JPEG', 'PNG'):
            raise ValueError('Registered complete JPEG/PNG header formats required')
        result.append((i, row['label'], header['format']))
    return result


def freeze():
    report = read(EVIDENCE / 'e131_source_holdout.json')
    metadata = read(EVIDENCE / 'e143_rr_metadata.json')
    if digest(BASE / 'contract.json') != report['contract_sha256'] or \
            digest(BASE / 'locked_scores.json') != report['locked_scores_sha256'] or \
            digest(BASE / 'scores.npz') != read(BASE / 'locked_scores.json')['scores_sha256'] or \
            digest(HEADERS) != metadata['private_records_sha256'] or \
            read(HEADERS)['contract_sha256'] != metadata['contract_sha256']:
        raise ValueError('Locked score/header evidence differs')
    roster = align_headers(read(BASE / 'contract.json')['rows'], read(HEADERS)['rows'])
    if len(roster) != metadata['parents']:
        raise ValueError('Metadata count differs')
    files = [Path(__file__), Path(distribution.__code__.co_filename),
             Path(digest.__code__.co_filename), HEADERS, BASE / 'contract.json',
             BASE / 'scores.npz', BASE / 'locked_scores.json',
             EVIDENCE / 'e131_source_holdout.json', EVIDENCE / 'e143_rr_metadata.json']
    c = dict(state='E144_RR_format_score_audit_registered',
             inputs={str(p): digest(p) for p in files}, parents=len(roster),
             conditions=CONDITIONS, branches=BRANCHES, cut=0.5,
             scope='Every RR class/container stratum, both branches and all four existing conditions; fixed-cut alert rates, raw-score distributions and paired clean-to-processed new/rescued errors. No subgroup selection.',
             limits='Retrospective consumed E131 TRAIN source-held-out scores, not E92 serving performance or fresh independent validation. Only one REAL PNG. Container, publisher, generator and content are confounded; this cannot identify causal codec reliance. Models receive decoded pixels, not filename extensions. No fitting, cutoff search, calibration, family reassignment or promotion.',
             downloads=0, new_image_reads=0, new_fits=0, new_inferences=0,
             promotion_allowed=False)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e144_rr_format_scores_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'contract_sha256': digest(CONTRACT), 'parents': len(roster)}


def audit():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e144_rr_format_scores_contract.json')['contract_sha256']:
        raise ValueError('E144 contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound E144 input differs')
    prior = read(BASE / 'contract.json')
    rows = prior['rows']
    roster = align_headers(rows, read(HEADERS)['rows'])
    folds = np.asarray([prior['outer_fold'][r['parent_id']] for r in rows])
    indices = [i for i, _, _ in roster]
    if len(roster) != c['parents'] or len(set(folds[indices].tolist())) != 1 or \
            len({prior['components'][rows[i]['parent_id']] for i in indices}) != 1:
        raise ValueError('Complete RR single-component held-out population required')
    result = []
    with np.load(BASE / 'scores.npz', allow_pickle=False) as data:
        if str(data['contract_sha256']) != digest(BASE / 'contract.json') or \
                list(data['parents']) != [r['parent_id'] for r in rows] or \
                list(data['conditions']) != CONDITIONS or prior['conditions'] != CONDITIONS or \
                not np.array_equal(data['outer_fold'], folds) or \
                str(data['global_role']) != 'TRAIN' or str(data['usage']) != 'INTERNAL_HELD_OUT':
            raise ValueError('Score parent/fold/condition/role identity differs')
        for branch in BRANCHES:
            values = data[branch]
            if values.shape != (len(rows), len(CONDITIONS)):
                raise ValueError('Complete score matrix required')
            vector(values.ravel())
            for label in (0, 1):
                for fmt in ('JPEG', 'PNG'):
                    selected = [i for i, y, f in roster if y == label and f == fmt]
                    if not selected:
                        raise ValueError('Registered nonempty format stratum missing')
                    for j, condition in enumerate(CONDITIONS):
                        stats = distribution(values[selected, j])
                        errors = stats['parents'] - stats['alerts_at_fixed_half'] if label else stats['alerts_at_fixed_half']
                        result.append(dict(branch=branch, label=label, format=fmt,
                                           condition=condition, distribution=stats,
                                           errors=errors, error_fraction=errors / len(selected),
                                           change_from_clean=paired(values[selected, 0], values[selected, j], label)))
    report = dict(state='E144_RR_format_score_audit_complete',
                  contract_sha256=digest(CONTRACT), parents=len(roster), cut=c['cut'],
                  held_out_fold=int(folds[indices][0]), rows=result, limits=c['limits'],
                  downloads=0, new_image_reads=0, new_fits=0, new_inferences=0,
                  promotion_allowed=False)
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e144_rr_format_scores.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('Locked format-score audit is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
