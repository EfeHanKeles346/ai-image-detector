"""Frozen four-condition source-pixel probe and separately admitted full TRAIN extraction."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import socket
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e149_source_pixel_probe as previous
from pixelproof import source_pixel_views as views, source_pixel_residual
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e150'; EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'probe_contract.json'; FULL=ROOT/'full_contract.json'
TRAIN=DATA_ROOT/'e131/contract.json'


def peak_MiB():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)


def geometry_audit(records):
    output={name:dict(parents=0,minimum_short_side=None,unsupported=0) for name in views.CONDITIONS}
    for row in records:
        if 'header_error' in row:raise ValueError('Verified headers required')
        size=(row['width'],row['height'])
        if row['header_orientation'] in (5,6,7,8):size=size[::-1]
        for name,shape in zip(views.CONDITIONS,views.geometry(size,row['parent_id']),strict=True):
            r=output[name];short=min(shape);r['parents']+=1;r['unsupported']+=int(short<128)
            r['minimum_short_side']=short if r['minimum_short_side'] is None else min(short,r['minimum_short_side'])
    return output


def freeze():
    prior=read(previous.CONTRACT);report=read(previous.ROOT/'report.json')
    if digest(previous.CONTRACT)!=read(EVIDENCE/'e149_source_pixel_probe_contract.json')['contract_sha256'] or \
            digest(previous.ROOT/'report.json')!=digest(EVIDENCE/'e149_source_pixel_probe.json') or \
            digest(previous.ROOT/'features.npz')!=report['features_sha256'] or not report['all_engineering_gates_passed']:
        raise ValueError('Complete passing E149 required')
    for path,sha in prior['inputs'].items():
        if digest(path)!=sha:raise ValueError('Inherited probe input differs')
    records=read(DATA_ROOT/'e148/private_records.json')['rows'];geometry=geometry_audit(records)
    if any(r['parents']!=12525 or r['unsupported'] for r in geometry.values()):
        raise ValueError('All12525 parents must support every condition without upsampling/exclusion')
    files=[Path(__file__),Path(views.__file__),Path(source_pixel_residual.__file__),Path(previous.__file__),
           previous.CONTRACT,previous.ROOT/'report.json',previous.ROOT/'features.npz',
           EVIDENCE/'e149_source_pixel_probe.json',ML_ROOT/'experiments/e42_features.py',
           ML_ROOT/'experiments/e54_data.py',ML_ROOT/'experiments/e65_diagnostic.py']
    c=dict(state='E150_four_condition_probe_registered',inputs=prior['inputs']|{str(p):digest(p) for p in files},
        selected=prior['selected'],parents=12525,probe_parents=len(prior['selected']),conditions=views.CONDITIONS,
        versions=previous.versions(),geometry=geometry,max_seconds=900,max_rss_MiB=1536,max_body_bytes=100*1024**2,
        full_budget_seconds=7200,chunk_parents=64,cpu_threads=2,
        transforms='EXIF-oriented RGB. clean retains source resolution; q75 is full-resolution JPEG75/subsampling2 before semantic2048 cap. assigned_transport reproduces E42 hash choice/cap2048 then JPEG55,WEBP60,resize.65/JPEG65 or blur.8. social_q75 cap1080 then JPEG75/subsampling2. Only the new source-pixel branch bypasses encoder-internal capping on clean/Q75. Existing semantic caches stay unchanged.',
        features='Frozen E149 center128 source-pixel crop and128->64->128 same-patch LANCZOS control; two300-coordinate residual histograms per condition. RGB convention including alpha discard unchanged. No padding, source replacement, quantization or crop search.',
        probe='Same53 E149 parents; complete four views twice, exact numeric replay and exact E149 clean-feature replay. Pin serialized features. Every stratum completes; measure process peak RSS and first-pass read/hash/extract times. Forecast full time by source/class/history-stratum counts times mean measured cost; require2x forecast<=7200s and all other guards before separate full registration. Forecast is not a guarantee.',
        full='Separately frozen after passing probe. All12525 parents/four views, immutable64-parent chunks, body/parent/condition/binding receipts and finite normalized arrays. Resume only receipted valid chunks; orphan chunks fail closed. CPU2, AC/20GiB reserve,7200s each execution,1536MiB peak. Preserve all failures; no replacement/drop.',
        later_fit='One E131 comparator recipe: append FIT-only unweighted standardizer/whitened PCA64 of[source300,control-minus-source300] to frozen E131320-coordinate maps, seed131/power3 and existing floors. Three folds/two geometries,six zero-start balanced BCE+.005||w||^2 heads, existing solver tolerances, fixed0.5 diagnostic cut. Replay E131 then lock all new predictions before all-condition/source/fold and paired new/rescued errors. Require zero new AI misses and REAL alerts. No sweep or promotion.',
        downloads=0,fits=0,model_scores=0,promotion_allowed=False,
        limits='Consumed TRAIN engineering/representation work, not an E92 improvement or independent unseen-family evidence. Only three AI-bearing components and unknown upstream/pretraining overlap. No gallery/DEV/protected-reserve scoring or ancestor substitution.')
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e150_source_pixel_probe_contract.json',{k:v for k,v in c.items() if k not in ('inputs','selected')}|{'contract_sha256':digest(CONTRACT)})
    return {'probe_parents':c['probe_parents'],'geometry':geometry,'contract_sha256':digest(CONTRACT)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e150_source_pixel_probe_contract.json')['contract_sha256'] or c['versions']!=previous.versions():
        raise ValueError('Probe contract/runtime differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha:raise ValueError('Frozen feature input differs')
    return c


def extract_row(row,c):
    path=Path(row['path'])
    if row['role'].upper()!='TRAIN' or not path.resolve().is_relative_to(DATA_ROOT.resolve()) or not 0<path.stat().st_size<=c['max_body_bytes']:
        raise ValueError('Bound local TRAIN body required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('TRAIN body differs')
    value,sizes=views.extract(raw,row['parent_id'])
    if peak_MiB()>c['max_rss_MiB']:raise RuntimeError('Feature memory ceiling exceeded')
    return value,sizes


def probe():
    c=validate();start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    write_once(ROOT/'probe_started.json',{'contract_sha256':digest(CONTRACT)})
    rows={r['parent_id']:r for r in read(TRAIN)['rows']}
    records=read(DATA_ROOT/'e148/private_records.json')['rows'];meta={r['parent_id']:r for r in records}
    with np.load(previous.ROOT/'features.npz',allow_pickle=False) as old:
        if list(old['parents'])!=c['selected']:raise ValueError('E149 selection differs')
        clean=old['features'].copy()
    values=[];timings=defaultdict(list)
    with threadpool_limits(limits=2):
        for i,parent in enumerate(c['selected']):
            resource_check(deadline);before=time.monotonic();value,size=extract_row(rows[parent],c)
            elapsed=time.monotonic()-before
            again,again_size=extract_row(rows[parent],c)
            if not np.array_equal(value,again) or not np.array_equal(size,again_size) or not np.array_equal(value[0],clean[i]):
                raise ValueError('Four-condition or E149 clean replay differs')
            r=meta[parent];timings[(r['source'],r['label'],r['processing_history'])].append(elapsed)
            values.append(value)
            if (i+1)%8==0:print(json.dumps({'E150_probe':i+1,'seconds':round(time.monotonic()-start)}),flush=True)
    counts=Counter((r['source'],r['label'],r['processing_history']) for r in records)
    if set(counts)!=set(timings):raise ValueError('Complete cost strata required')
    forecast=sum(counts[k]*float(np.mean(timings[k])) for k in counts)
    array=np.stack(values);save_npz(ROOT/'probe_features.npz',features=array,parents=np.asarray(c['selected']),conditions=c['conditions'],binding=digest(CONTRACT))
    with np.load(ROOT/'probe_features.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['features'],array):raise ValueError('Probe serialization differs')
    resource_check(deadline)
    report=dict(state='E150_four_condition_probe_complete',contract_sha256=digest(CONTRACT),parents=len(values),
        exact_replay=True,exact_E149_clean_replay=True,peak_RSS_MiB=peak_MiB(),seconds=time.monotonic()-start,
        forecast_full_seconds=forecast,forecast_with_2x_margin_seconds=2*forecast,
        passes_full_resource_gate=2*forecast<=c['full_budget_seconds'],features_sha256=digest(ROOT/'probe_features.npz'),
        downloads=0,fits=0,model_scores=0,promotion_allowed=False,limits=c['limits'])
    write_once(ROOT/'probe.json',report);write_once(EVIDENCE/'e150_source_pixel_probe.json',report)
    return report


def register_full():
    c=validate();p=read(ROOT/'probe.json')
    if digest(ROOT/'probe.json')!=digest(EVIDENCE/'e150_source_pixel_probe.json') or not p['passes_full_resource_gate'] or \
            not p['exact_replay'] or not p['exact_E149_clean_replay'] or p['peak_RSS_MiB']>c['max_rss_MiB'] or \
            digest(ROOT/'probe_features.npz')!=p['features_sha256']:
        raise ValueError('Passing complete probe required')
    full=dict(state='E150_full_source_pixel_features_registered',probe_contract_sha256=digest(CONTRACT),
        probe_report_sha256=digest(ROOT/'probe.json'),probe_features_sha256=digest(ROOT/'probe_features.npz'),
        parents=c['parents'],conditions=c['conditions'],max_seconds=c['full_budget_seconds'],chunk_parents=c['chunk_parents'],
        max_rss_MiB=c['max_rss_MiB'],later_fit=c['later_fit'],downloads=0,model_scores=0,promotion_allowed=False)
    write_once(FULL,full);write_once(EVIDENCE/'e150_source_pixel_full_contract.json',full|{'contract_sha256':digest(FULL)})
    return full


def check_chunk(a,rows,binding):
    if str(a['binding'])!=binding or list(a['parents'])!=[r['parent_id'] for r in rows] or \
            list(a['source_sha256'])!=[r['sha256'] for r in rows] or list(a['conditions'])!=list(views.CONDITIONS):
        raise ValueError('Chunk body/parent/condition identity differs')
    value=a['features']
    if value.shape!=(len(rows),4,2,300) or value.dtype!=np.float32 or not np.isfinite(value).all() or (value<0).any() or \
            not np.allclose(value.reshape(len(rows),4,2,12,25).sum(axis=-1),1,rtol=0,atol=1e-6):
        raise ValueError('Incomplete finite normalized chunk required')
    return value


def extract():
    c=validate();full=read(FULL);binding=digest(FULL)
    if binding!=read(EVIDENCE/'e150_source_pixel_full_contract.json')['contract_sha256'] or \
            full['probe_contract_sha256']!=digest(CONTRACT) or full['probe_report_sha256']!=digest(ROOT/'probe.json') or \
            full['probe_features_sha256']!=digest(ROOT/'probe_features.npz'):
        raise ValueError('Full extraction admission differs')
    if (ROOT/'report.json').exists():raise FileExistsError('Complete feature extraction is immutable')
    rows=read(TRAIN)['rows'];start=time.monotonic();deadline=start+full['max_seconds'];resource_check(deadline)
    folder=ROOT/'chunks';folder.mkdir(exist_ok=True)
    all_values=np.empty((len(rows),4,2,300),dtype=np.float32);created=resumed=0;receipts={}
    with threadpool_limits(limits=2):
        for offset in range(0,len(rows),full['chunk_parents']):
            resource_check(deadline);part=rows[offset:offset+full['chunk_parents']]
            path=folder/f'{offset:05d}.npz';receipt=path.with_suffix('.json')
            if path.exists():
                info=read(receipt)
                if info['sha256']!=digest(path) or info['binding']!=binding:raise ValueError('Chunk receipt differs')
                with np.load(path,allow_pickle=False) as saved:value=check_chunk(saved,part,binding).copy()
                resumed+=len(part)
            else:
                values=[]
                for row in part:
                    resource_check(deadline);value,_=extract_row(row,c);values.append(value)
                value=np.stack(values)
                save_npz(path,features=value,parents=np.asarray([r['parent_id'] for r in part]),
                    source_sha256=np.asarray([r['sha256'] for r in part]),conditions=views.CONDITIONS,binding=binding)
                with np.load(path,allow_pickle=False) as saved:
                    if not np.array_equal(value,check_chunk(saved,part,binding)):raise ValueError('Chunk replay differs')
                write_once(receipt,{'sha256':digest(path),'binding':binding});created+=len(part)
            all_values[offset:offset+len(part)]=value;receipts[path.name]=digest(path)
            print(json.dumps({'E150_parents':offset+len(part),'created':created,'resumed':resumed,'seconds':round(time.monotonic()-start)}),flush=True)
    with np.load(ROOT/'probe_features.npz',allow_pickle=False) as probe:
        index={r['parent_id']:i for i,r in enumerate(rows)}
        replay=all_values[[index[p] for p in c['selected']]]
        if not np.array_equal(replay,probe['features']):raise ValueError('Full/probe replay differs')
    resource_check(deadline)
    save_npz(ROOT/'features.npz',features=all_values,parents=np.asarray([r['parent_id'] for r in rows]),conditions=views.CONDITIONS,binding=binding,role='TRAIN')
    with np.load(ROOT/'features.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['features'],all_values):raise ValueError('Full serialization differs')
    if peak_MiB()>c['max_rss_MiB']:raise RuntimeError('Full memory ceiling exceeded')
    report=dict(state='E150_full_source_pixel_features_complete',contract_sha256=binding,parents=len(rows),conditions=views.CONDITIONS,
        feature_shape=list(all_values.shape),features_sha256=digest(ROOT/'features.npz'),chunks=receipts,
        created_this_execution=created,resumed_this_execution=resumed,seconds_this_execution=time.monotonic()-start,
        peak_RSS_MiB=peak_MiB(),exact_probe_and_serialization_replay=True,downloads=0,fits=0,model_scores=0,promotion_allowed=False,limits=c['limits'])
    write_once(ROOT/'report.json',report)
    public={k:v for k,v in report.items() if k!='chunks'}|{'report_sha256':digest(ROOT/'report.json'),'chunks':len(receipts)}
    write_once(EVIDENCE/'e150_source_pixel_features.json',public)
    return public


if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('Source-pixel features are offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','probe','register-full','extract'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe,'register-full':register_full,'extract':extract}[parser.parse_args().stage](),indent=2))
