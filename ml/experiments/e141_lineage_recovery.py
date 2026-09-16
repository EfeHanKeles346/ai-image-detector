"""Offline recovery of E32 TRAIN lineage omitted from downstream manifests."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
from experiments.e65_acquisition import digest, read, write_once
from pixelproof import lineage_recovery
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e141'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
TRAIN = DATA_ROOT / 'e131/contract.json'
MANIFEST = DATA_ROOT / 'e32/c3_role_manifest.json'
RECEIPT = DATA_ROOT / 'e32/r0_input_receipt.json'
ELIGIBILITY = DATA_ROOT / 'e32/eligibility_overlay.json'


def freeze():
    if digest(TRAIN) != read(EVIDENCE / 'e131_source_holdout_contract.json')['contract_sha256']:
        raise ValueError('Frozen TRAIN identity differs')
    if read(RECEIPT)['manifest_sha256'] != digest(MANIFEST) or read(MANIFEST)['eligibility_overlay_sha256'] != digest(ELIGIBILITY):
        raise ValueError('E32 receipt/role/eligibility binding differs')
    sources = sorted({r['source'].removeprefix('e32:') for r in read(TRAIN)['rows']
                      if r['label'] == 1 and r['source'].startswith('e32:')})
    audits = {s: DATA_ROOT / f'e32/audits/{s}.json' for s in sources}
    bindings = read(ELIGIBILITY)['audit_bindings']
    for source, path in audits.items():
        if digest(path) != bindings[source]['sha256']:
            raise ValueError('Original E32 audit identity differs')
    files = [Path(__file__), Path(lineage_recovery.__file__), TRAIN, MANIFEST, RECEIPT,
             ELIGIBILITY, EVIDENCE / 'e131_source_holdout_contract.json', *audits.values()]
    c = dict(state='E141_lineage_recovery_registered', inputs={str(p): digest(p) for p in files},
        audits={s: str(p) for s, p in audits.items()},
        scope='Bind active E32 AI TRAIN parent IDs to original c3 TRAIN rows, native or processed receipt body identities, then exact source-key/native-body realization metadata. Write separate private overlay and public aggregate coverage; do not modify inherited rows/folds.',
        checks='Reject role/class/source/body/declared-model conflicts and ambiguous keys. Recover stored model names, dataset revisions, declared families and nonempty prompt hashes. Count exact prompt links across frozen components/folds. No substring family inference.',
        limits='Metadata schema exploration preceded registration. This is a reproducible metadata recovery, not a blind experiment or model-quality evaluation. Historical manifests include consumed non-target rows; only current E32 AI TRAIN parents join to the output. No protected reserve documents, images, features, scores or weights are read.',
        downloads=0, model_scores=0, fits=0, promotion_allowed=False)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e141_lineage_recovery_contract.json',
        {k: v for k, v in c.items() if k not in ('inputs', 'audits')} | {'contract_sha256': digest(CONTRACT)})
    return {'contract_sha256': digest(CONTRACT), 'audit_sources': len(audits)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e141_lineage_recovery_contract.json')['contract_sha256']:
        raise ValueError('Registered contract differs')
    for path, expected in c['inputs'].items():
        if digest(path) != expected:
            raise ValueError('Registered metadata/code identity differs')
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    train = read(TRAIN)
    overlay = lineage_recovery.recover(train['rows'], read(MANIFEST)['records'], read(RECEIPT)['records'],
        {s: read(p) for s, p in c['audits'].items()}, train['components'], train['outer_fold'])
    write_once(ROOT / 'train_lineage_overlay.json', {'rows': overlay, 'contract_sha256': digest(CONTRACT)})
    report = lineage_recovery.summarize(train['rows'], overlay, train['components'], train['outer_fold'])
    report.update(state='E141_lineage_recovery_complete', contract_sha256=digest(CONTRACT),
        overlay_sha256=digest(ROOT / 'train_lineage_overlay.json'), downloads=0, model_scores=0, fits=0)
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e141_lineage_recovery.json', report)
    return report


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('Lineage recovery is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
