"""Resolve every E84 TRAIN body through existing loose/parquet source keys, no image decoding."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E84 source resolution offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import pyarrow as pa
import pyarrow.parquet as pq
from experiments import e32_r0_input as legacy
from experiments.e84_features import population,validate as validate_features,CONTRACT as ORIGINAL_CONTRACT
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
ROOT=DATA_ROOT/'e84_sources';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'source_contract.json';INDEX=ROOT/'source_index.json';REPORT=ROOT/'sources.json'


def selected_cells(path,column,wanted):
    if not wanted or any(not isinstance(i,int) or i<0 for i in wanted):raise ValueError('nonnegative selected row indices required')
    parquet=pq.ParquetFile(path);offset=0;found=set()
    for group in range(parquet.metadata.num_row_groups):
        count=parquet.metadata.row_group(group).num_rows
        selected=sorted(i for i in wanted if offset<=i<offset+count)
        if selected:
            # Parquet row-group I/O may include other encoded cells. Only admitted TRAIN indices are converted to Python bytes.
            table=parquet.read_row_group(group,columns=[column]).take(pa.array([i-offset for i in selected],type=pa.int64()))
            for i,value in zip(selected,table.column(0).to_pylist(),strict=True):
                found.add(i);yield i,legacy.raw_image(value)
        offset+=count
    if found!=set(wanted):raise ValueError('selected TRAIN row absent from source parquet')


def verify_body(body,row):
    if row['role'].upper()!='TRAIN' or hashlib.sha256(body).hexdigest()!=row['sha256']:
        raise ValueError('TRAIN role or exact original image body differs')


def freeze():
    ROOT.mkdir(exist_ok=True);validate_features();rows=population();missing=[r for r in rows if not r.get('path')]
    if len(missing)!=5652 or any(not r.get('native') or not r.get('source_key') for r in missing):raise ValueError('native source schema differs')
    c={'state':'E84_complete_TRAIN_source_resolution_registered',
       'inputs':{str(p):digest(p) for p in [Path(__file__),Path(legacy.__file__),ORIGINAL_CONTRACT,
            DATA_ROOT/'e54/data_contract_v2.json',DATA_ROOT/'e72/training_manifest.json']},
       'parents':12141,'missing_direct_paths':5652,'parquet_parents':1000,'max_seconds':1800,
       'source_counts':dict(Counter(r['source_id'] for r in missing)),
       'method':'Existing explicit paths, legacy e32 source_path for native loose files, selected source_key row '
                'and image column for native Parquet. Exact original body SHA required for every parent. '
                'Materialize only1000 admitted TRAIN Parquet bodies to hash-named external files; retain all other paths.',
       'limits':'Parquet row-group I/O can physically include unselected encoded cells; only admitted TRAIN indices '
                'are converted to Python payloads. No unselected labels/pixels decoded, scored or admitted. '
                'No standardized224JPEG fallback and no missing-row omission. No change to source/image identity.',
       'resume':'Existing materialized bodies must match declared SHA; source index written only after all12141 verify.',
       'downloads':0,'image_decodes':0,'model_scores':0,'training_fit_allowed':False,'e49_read_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e84_source_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows)}


def validate():
    validate_features();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e84_source_contract.json')['contract_sha256']:raise ValueError('source contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('source resolver input changed')
    return c


def materialize():
    c=validate()
    if REPORT.exists():raise FileExistsError('sources already complete')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows=population();paths={};grouped=defaultdict(dict);(ROOT/'bodies').mkdir(exist_ok=True);bytes_materialized=0
    for i,row in enumerate(rows):
        if i%128==0:resource_check(deadline)
        if not row.get('path') and row['source_id'] in legacy.PARQUET_SOURCES:
            folder,column=legacy.PARQUET_SOURCES[row['source_id']];shard,index=row['source_key'].rsplit(':',1)
            grouped[(folder,column,shard)][int(index)]=row;continue
        path=Path(row['path']) if row.get('path') else legacy.source_path(row['source_id'],row['source_key'])
        body=path.read_bytes();verify_body(body,row);paths[row['parent_id']]=str(path)
    for (folder,column,shard),selected in sorted(grouped.items()):
        resource_check(deadline);missing={}
        for index,row in selected.items():
            path=ROOT/'bodies'/(row['sha256']+'.bin')
            if path.exists():verify_body(path.read_bytes(),row);paths[row['parent_id']]=str(path)
            else:missing[index]=row
        if missing:
            for index,body in selected_cells(DATA_ROOT/folder/shard,column,set(missing)):
                resource_check(deadline);row=missing[index];verify_body(body,row);path=ROOT/'bodies'/(row['sha256']+'.bin')
                temporary=path.with_suffix('.bin.part');temporary.write_bytes(body);temporary.replace(path)
                verify_body(path.read_bytes(),row);paths[row['parent_id']]=str(path);bytes_materialized+=len(body)
        print(json.dumps({'E84_source_parents_verified':len(paths),'total':len(rows),'seconds':round(time.monotonic()-started)}),flush=True)
    if len(paths)!=len(rows):raise ValueError('complete TRAIN source coverage required')
    index={'state':'E84_verified_TRAIN_source_paths','contract_sha256':digest(CONTRACT),'paths':paths}
    if INDEX.exists():
        if read(INDEX)!=index:raise ValueError('unreceipted source index differs')
    else:write_once(INDEX,index)
    result={'state':'E84_TRAIN_sources_complete','contract_sha256':digest(CONTRACT),'index_sha256':digest(INDEX),
            'parents':len(rows),'parquet_parents':sum(len(v) for v in grouped.values()),'bytes_created_this_run':bytes_materialized,
            'seconds_this_run':time.monotonic()-started,'image_decodes':0,'model_scores':0,'downloads':0}
    write_once(REPORT,result);write_once(EVIDENCE/'e84_sources.json',result);return result


def materialized_rows():
    c=validate();result=read(REPORT);index=read(INDEX)
    if digest(REPORT)!=digest(EVIDENCE/'e84_sources.json') or digest(INDEX)!=result['index_sha256'] or \
            index['contract_sha256']!=digest(CONTRACT) or result['parents']!=c['parents']:
        raise ValueError('complete verified source receipt required')
    rows=population()
    if set(index['paths'])!={r['parent_id'] for r in rows}:raise ValueError('complete source-parent identity required')
    return [r|{'path':index['paths'][r['parent_id']]} for r in rows]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','materialize'))
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'materialize':materialize}[parser.parse_args().stage](),indent=2))
