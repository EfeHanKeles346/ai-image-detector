"""Bounded offline RR TRAIN header availability audit, not model inference."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import socket
from experiments.e65_acquisition import digest, read, write_once
from pixelproof import header_metadata
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e143'; EVIDENCE=ML_ROOT.parent/'evidence'; CONTRACT=ROOT/'contract.json'
TRAIN=DATA_ROOT/'e131/contract.json'; RECEIPT=DATA_ROOT/'e42/rr_train_receipt.json'
INVENTORY=DATA_ROOT/'e33_rrdataset/cal_archive_inventory.json'
DOWNLOAD=DATA_ROOT/'e33_rrdataset/cal_download_receipt.json'
SELECTION=DATA_ROOT/'e33_rrdataset/acquisition_selection.json'


def bound_rows(train, receipt, root):
    by_path={r['relative_path']:r for r in receipt}
    if len(by_path)!=len(receipt): raise ValueError('Ambiguous RR receipt path')
    selected=[]; seen=set()
    for row in train:
        if not row['source'].startswith('rr:'): continue
        if str(row['role']).upper()!='TRAIN' or row['parent_id'] in seen or row['label'] not in (0,1):
            raise ValueError('Unique active TRAIN RR parents required')
        seen.add(row['parent_id'])
        path=Path(row['path']).resolve()
        if not path.is_relative_to((root/'e42/rr_train').resolve()): raise ValueError('Outside eligible RR TRAIN directory')
        relative=path.relative_to((root/'e42').resolve()).as_posix()
        old=by_path[relative]
        expected_prefix='RRDataset_original_train_val/train/'+('ai/' if row['label']==1 else 'real/')
        if not old['upstream_member'].startswith(expected_prefix) or old['label']!=row['label'] or \
                old['sha256']!=row['sha256'] or old['bytes']!=row['bytes'] or 'rr:'+old['source']!=row['source']:
            raise ValueError('Original RR member/body/source/class mismatch')
        if not 0 < row['bytes'] <= 100*1024**2: raise ValueError('Unbounded image body')
        selected.append(row)
    if not selected: raise ValueError('No active RR TRAIN parents')
    return selected


def freeze():
    if digest(TRAIN)!=read(EVIDENCE/'e131_source_holdout_contract.json')['contract_sha256'] or \
            digest(RECEIPT)!=read(EVIDENCE/'e42_data_manifest.json')['rr_receipt_sha256']:
        raise ValueError('Active TRAIN/extraction identity differs')
    if read(RECEIPT)['inventory_sha256']!=digest(INVENTORY) or read(INVENTORY)['download_receipt_sha256']!=digest(DOWNLOAD) or \
            read(INVENTORY)['selection_sha256']!=digest(SELECTION) or read(DOWNLOAD)['selection_sha256']!=digest(SELECTION):
        raise ValueError('Historical inventory/acquisition chain differs')
    rows=bound_rows(read(TRAIN)['rows'],read(RECEIPT)['rows'],DATA_ROOT)
    files=[TRAIN,RECEIPT,INVENTORY,DOWNLOAD,SELECTION,Path(__file__),Path(header_metadata.__file__),
           EVIDENCE/'e131_source_holdout_contract.json', EVIDENCE/'e42_data_manifest.json']
    c=dict(state='E143_RR_TRAIN_header_audit_registered',inputs={str(p):digest(p) for p in files},
        parents=len(rows),class_counts=dict(Counter(str(r['label']) for r in rows)),
        scope='Read only bound active RR TRAIN image bodies, verify complete body SHA/size, then inspect lazy PNG/JPEG/WEBP header info and already-present EXIF. No pixel decode, feature extraction, model, gallery, validation/test archive or protected reserve access.',
        limits='Header-only metadata availability; PNG post-IDAT fields and other container sections can remain unread. Missing fields do not prove no embedded metadata anywhere. Text is untrusted and never establishes generator/authenticity labels.',
        downloads=0,fits=0,model_scores=0,promotion_allowed=False)
    write_once(CONTRACT,c)
    write_once(EVIDENCE/'e143_rr_metadata_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {k:c[k] for k in ['parents','class_counts','scope']}


def run():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e143_rr_metadata_contract.json')['contract_sha256']: raise ValueError('Contract changed')
    for p,sha in c['inputs'].items():
        if digest(p)!=sha: raise ValueError('Bound code/metadata changed')
    train=read(TRAIN)
    rows=bound_rows(train['rows'],read(RECEIPT)['rows'],DATA_ROOT)
    write_once(ROOT/'started.json',{'contract_sha256':digest(CONTRACT)})
    records=[]; bytes_read=0
    for row in rows:
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes']: raise ValueError('RR body size changed')
        raw=path.read_bytes();bytes_read+=len(raw)
        if hashlib.sha256(raw).hexdigest()!=row['sha256']: raise ValueError('RR body digest changed')
        try: details=header_metadata.inspect_header(raw)
        except (OSError,ValueError,SyntaxError) as exc: details={'header_error':type(exc).__name__,'fields':{}}
        records.append(dict(parent_id=row['parent_id'],label=row['label'],source=row['source'],**details))
    write_once(ROOT/'private_header_records.json',{'rows':records,'contract_sha256':digest(CONTRACT)})
    classes={}
    for label in (0,1):
        part=[r for r in records if r['label']==label]
        classes[str(label)]=dict(parents=len(part),header_errors=sum('header_error' in r for r in part),
            exif_parse_errors=sum(bool(r.get('exif_parse_error')) for r in part),
            parents_with_selected_fields=sum(bool(r['fields']) for r in part),
            field_counts=dict(Counter(k for r in part for k in r['fields'])),
            formats=dict(Counter(r.get('format','error') for r in part)))
    report=dict(state='E143_RR_TRAIN_header_audit_complete',contract_sha256=digest(CONTRACT),
        parents=len(records),by_class=classes,verified_body_bytes=bytes_read,
        historical_archive_nonimage_files=read(INVENTORY)['other_files'],
        historical_inventory_not_fresh_archive_scan=True,
        active_RR_components=len({train['components'][r['parent_id']] for r in rows}),
        active_RR_folds=len({train['outer_fold'][r['parent_id']] for r in rows}),
        private_records_sha256=digest(ROOT/'private_header_records.json'),
        downloads=0,fits=0,model_scores=0,pixel_decodes=0,promotion_allowed=False,limits=c['limits'])
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e143_rr_metadata.json',report)
    return report


if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('RR metadata audit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](),indent=2))
