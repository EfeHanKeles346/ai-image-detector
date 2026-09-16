"""Offline active-TRAIN lineage inventory and existing public reserve-status audit."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
from experiments.e65_acquisition import digest, read, write_once
from pixelproof import provenance_audit
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e139'; EVIDENCE = ML_ROOT.parent / 'evidence'; CONTRACT = ROOT / 'contract.json'
TRAIN = DATA_ROOT / 'e131/contract.json'
RESERVES = ('e54_mnw_reserve_manifest.json', 'e54_hdrplus_manifest.json',
            'e54_mnw_overlap.json', 'e54_hdrplus_overlap.json')


def freeze():
    if digest(TRAIN) != read(EVIDENCE / 'e131_source_holdout_contract.json')['contract_sha256']:
        raise ValueError('Frozen E131 metadata identity differs')
    files = [Path(__file__), Path(provenance_audit.__file__), TRAIN,
             EVIDENCE / 'e131_source_holdout_contract.json'] + [EVIDENCE / n for n in RESERVES]
    c = dict(state='E139_metadata_audit_registered', inputs={str(p): digest(p) for p in files},
             scope='Active E131 TRAIN metadata and four existing public E54 reserve summaries only. No image/feature/weight reads, new hashes of image bodies, scoring, fitting, downloads or directory-wide reserve traversal.',
             checks='Count populated lineage fields by source/class; exact stored body/pixel links, declared scene links and component/fold crossings. Distinguish declared identifiers from verified independence. Reserve records remain historical snapshots, not newly certified current eligibility.',
             promotion_allowed=False, independent_final=False, downloads=0)
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e139_provenance_contract.json', {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'contract_sha256': digest(CONTRACT)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e139_provenance_contract.json')['contract_sha256']:
        raise ValueError('Audit contract differs')
    for p, expected in c['inputs'].items():
        if digest(p) != expected: raise ValueError('Frozen audit metadata differs')
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    train = read(TRAIN)
    result = provenance_audit.audit(train['rows'], train['components'], train['outer_fold'])
    reserve = {}
    for name in RESERVES:
        old = read(EVIDENCE / name)
        reserve[name] = {k: old[k] for k in ('state', 'parents', 'query_parents',
            'current_train_parents_covered', 'model_scores_created', 'balanced_final_admitted',
            'training_allowed', 'training_allowed_by_project', 'threshold_calibration_allowed', 'limits') if k in old}
    report = dict(state='E139_metadata_audit_complete', contract_sha256=digest(CONTRACT),
                  training=result, historical_reserve_records=reserve, downloads=0,
                  new_model_scores=0, promotion_allowed=False,
                  reserve_conclusion='Existing MNW300 AI and HDR+100 REAL reserves are protected and not admitted as a balanced final test in these records. No automatic combination, tuning or current independence certification follows. Follow-up source-admission audits are needed to extend older reference coverage; no protected pixels or individual reserve identities opened.')
    write_once(ROOT / 'report.json', report); write_once(EVIDENCE / 'e139_provenance_audit.json', report)
    summary = {k: v for k, v in result.items() if k not in ('sources', 'classes', 'limits')}
    note = '\n### E139 metadata audit result\n\n' + json.dumps(summary, sort_keys=True) + '\n\nFull report: evidence/e139_provenance_audit.json. No new independent test or serving change.\n'
    for path in (ML_ROOT.parent / 'HISTORY.md', ML_ROOT / 'EXPERIMENTS.md', ML_ROOT.parent / 'DATASETS.md'):
        with path.open('a') as f: fcntl.flock(f, fcntl.LOCK_EX); f.write(note)
    return summary


if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('Metadata audit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
