"""Reconcile current TRAIN membership with historical reserve-screen admission receipts.

This follows a preliminary metadata review; it is not a blind performance experiment.
Protected reference files are hashed as opaque metadata, never parsed or scored.
"""
import argparse
import fcntl
import json
from pathlib import Path
import socket
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e140'; EVIDENCE = ML_ROOT.parent / 'evidence'; CONTRACT = ROOT / 'contract.json'
STAGES = ('e72', 'e88', 'e100')
RESERVES = {'MNW': 'e52/mnw_reserve_v1/download.json', 'HDRPLUS': 'e52/hdrplus_reserve_v1/download.json'}
PUBLIC = ('e131_source_holdout_contract.json', 'e54_mnw_reserve_download.json',
          'e54_hdrplus_download.json', 'e54_mnw_overlap_contract.json', 'e54_hdrplus_overlap_contract.json',
          'e54_mnw_overlap.json', 'e54_hdrplus_overlap.json', 'e54_reserve_reference.json')


def identities(rows):
    result = {}
    for row in rows:
        parent = row['parent_id']
        if parent in result or str(row.get('role', '')).upper() != 'TRAIN' or \
                type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Unique explicit TRAIN parents required')
        result[parent] = {k: row.get(k) for k in ('sha256', 'source', 'label', 'pixel_sha256', 'original_sha256')}
    return result


def reconcile(active, base, stages, reserve_hashes):
    active_ids = identities(active); covered = identities(base); summaries = {}
    for stage, item in stages.items():
        contract, manifest, report = item['contract'], item['manifest'], item['report']
        for path, sha in reserve_hashes.items():
            if path not in contract['reference_files'] or contract['inputs'].get(path) != sha:
                raise ValueError('Protected reserve absent or wrong reference identity')
        admitted = identities(manifest['rows'])
        if admitted != identities(report['records']) or len(admitted) != report['admitted'] or \
                report.get('model_scores') != 0 or manifest.get('model_scores') != 0 or \
                report.get('cross_matches') != [] or report.get('raw_cross_matches', []) != []:
            raise ValueError('Historical accepted rows or no-overlap outcome differs')
        if set(covered) & set(admitted): raise ValueError('Admission cohorts overlap')
        covered.update(admitted)
        summaries[stage] = {'admitted': len(admitted), 'protected_reserves_bound': len(reserve_hashes),
                            'historical_cross_matches': 0, 'reference_documents': len(contract['reference_files'])}
    if set(covered) != set(active_ids): raise ValueError('Uncovered or obsolete active TRAIN parents')
    if covered != active_ids: raise ValueError('Current TRAIN identity/label/source differs from admission')
    return {'active_parents': len(active), 'base_parents': len(base),
            'later_admitted_parents': len(active) - len(base), 'stages': summaries,
            'historical_admission_coverage_complete': True,
            'new_independent_test_admitted': False}


def inputs():
    # This explicit list excludes images, feature arrays, weights and score files.
    files = [Path(__file__), DATA_ROOT / 'e131/contract.json', DATA_ROOT / 'e54/data_contract_v2.json']
    files += [EVIDENCE / name for name in PUBLIC]
    for stage in STAGES:
        files += [EVIDENCE / f'{stage}_audit.json', EVIDENCE / f'{stage}_audit_contract.json']
        files += [DATA_ROOT / stage / name for name in ('audit_contract.json', 'audit.json', 'training_manifest.json')]
        files += [Path(p) for p in read(DATA_ROOT / stage / 'audit_contract.json')['reference_files']]
    return sorted(set(files))


def freeze():
    files = inputs()
    c = {'state': 'E140_admission_chain_audit_registered', 'inputs': {str(p): digest(p) for p in files},
         'scope': 'Reproduce metadata-chain checks after preliminary review. Join E54 base11630 and E72/E88/E100 admissions to current E131 TRAIN; verify receipt/reference hashes and accepted identities. Protected reference JSON files hashed only, not parsed.',
         'limits': 'Historical exact/perceptual screening lineage, not fresh pixel comparison, semantic deduplication, generator independence, balanced final admission or accuracy evaluation.',
         'downloads': 0, 'model_scores': 0, 'promotion_allowed': False}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e140_admission_chain_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT), 'metadata_files': len(files)})
    return {'contract_sha256': digest(CONTRACT), 'metadata_files': len(files)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e140_admission_chain_contract.json')['contract_sha256']:
        raise ValueError('Admission-chain contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Bound metadata changed')
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    active_path = DATA_ROOT / 'e131/contract.json'; base_path = DATA_ROOT / 'e54/data_contract_v2.json'
    active = read(active_path); base = read(base_path)
    if digest(active_path) != read(EVIDENCE / 'e131_source_holdout_contract.json')['contract_sha256'] or \
            digest(base_path) != active['inputs'][str(base_path)]:
        raise ValueError('Current/base population binding differs')
    mnw = read(EVIDENCE / 'e54_mnw_overlap_contract.json')
    if mnw['inputs'][str(base_path)] != digest(base_path): raise ValueError('MNW historical base binding differs')
    reserve_hashes = {}
    for name, public in (('MNW', 'e54_mnw_reserve_download.json'), ('HDRPLUS', 'e54_hdrplus_download.json')):
        path = DATA_ROOT / RESERVES[name]; expected = read(EVIDENCE / public)['report_sha256']
        if digest(path) != expected: raise ValueError('Protected reserve receipt changed')
        reserve_hashes[str(path)] = expected
    hdr = read(EVIDENCE / 'e54_hdrplus_overlap_contract.json')
    snapshot = DATA_ROOT / 'e52/protected_reference_v1.json'
    if digest(snapshot) != hdr['inputs'][str(snapshot)] or \
            digest(snapshot) != read(EVIDENCE / 'e54_reserve_reference.json')['snapshot_sha256']:
        raise ValueError('Historical protected reference binding differs')
    stages = {}
    for stage in STAGES:
        public = read(EVIDENCE / f'{stage}_audit.json')
        contract_path = DATA_ROOT / stage / 'audit_contract.json'
        report_path = DATA_ROOT / stage / 'audit.json'; manifest_path = DATA_ROOT / stage / 'training_manifest.json'
        for p, key in ((contract_path, 'contract_sha256'), (report_path, 'report_sha256'), (manifest_path, 'manifest_sha256')):
            if digest(p) != public[key]: raise ValueError('Admission public receipt differs')
        contract, report, manifest = read(contract_path), read(report_path), read(manifest_path)
        if report['contract_sha256'] != digest(contract_path) or manifest['audit_sha256'] != digest(report_path):
            raise ValueError('Admission contract/report/manifest chain differs')
        if digest(contract_path) != read(EVIDENCE / f'{stage}_audit_contract.json')['contract_sha256']:
            raise ValueError('Admission registration receipt differs')
        for path in contract['reference_files']:
            if digest(path) != contract['inputs'][path]: raise ValueError('Historical reference document differs')
        stages[stage] = dict(contract=contract, report=report, manifest=manifest)
    summary = reconcile(active['rows'], base['rows'], stages, reserve_hashes)
    report = summary | {'state':'E140_admission_chain_complete', 'contract_sha256':digest(CONTRACT),
        'verified_metadata_files':len(c['inputs']), 'downloads':0, 'image_reads':0, 'model_scores':0,
        'balanced_final_admitted':False, 'generator_independence_proven':False,
        'limits':c['limits'], 'next':'Membership/reference gap for the895 later additions is closed under stored admission policy. Prompt/scene/base-generator ancestry, protected evaluation protocol and candidate gates remain separate; do not score/tune on reserves automatically.'}
    write_once(ROOT / 'report.json', report); write_once(EVIDENCE / 'e140_admission_chain.json', report)
    note = '\n### E140 admission-chain audit result\n\n' + json.dumps(report, sort_keys=True) + '\n'
    for path in (ML_ROOT.parent / 'HISTORY.md', ML_ROOT / 'EXPERIMENTS.md', ML_ROOT.parent / 'DATASETS.md'):
        with path.open('a') as f: fcntl.flock(f, fcntl.LOCK_EX); f.write(note)
    return report


if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('Admission-chain audit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze, 'run':run}[parser.parse_args().stage](), indent=2))
