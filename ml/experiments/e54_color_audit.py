"""Exploratory colour-coverage audit on existing TRAIN inputs, never a classifier."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.e53_offline import EVIDENCE,digest,fixed_write
from experiments.e54_data import ROOT,CONTRACT as DATA_CONTRACT,INDEX,load

CONTRACT=ROOT/'color_contract.json'
DESCRIPTORS=ROOT/'color_descriptors.json'


def colour_stats(rgb):
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:
        raise ValueError('expected exact global model-input RGB crop')
    channel_range=rgb.max(axis=2)-rgb.min(axis=2)
    mean=float(channel_range.mean())
    return {'mean_channel_range_255':mean,'near_monochrome_global_crop':mean<=2.}


def prepare():
    data=load()
    if digest(INDEX)!=json.loads((EVIDENCE/'e54_crop_cache.json').read_text())['index_sha256']:
        raise ValueError('unbound crop index')
    config={'state':'E54_exploratory_colour_audit_frozen','code_sha256':digest(__file__),
            'data_contract_sha256':digest(DATA_CONTRACT),'index_sha256':digest(INDEX),
            'criterion':'Mean max-minus-min RGB channel value <=2 on the clean global 224 crop.',
            'motivation':'Four identity-ordered RR false positives inspected after fold-0 completion; exploratory, not causal.',
            'new_model_scores':0,'final_reserves_opened':False}
    fixed_write(CONTRACT,config)
    fixed_write(EVIDENCE/'e54_color_contract.json',config|{'contract_sha256':digest(CONTRACT)})
    index=json.loads(INDEX.read_text())['records']
    def compute(row):
        record=index[row['parent_id']]
        raw=Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('crop bytes changed')
        with np.load(BytesIO(raw),allow_pickle=False) as a:
            if str(a['binding'])!=config['data_contract_sha256']:
                raise ValueError('crop binding changed')
            facts=colour_stats(a['crops'][0,0])
        return {k:row[k] for k in ('parent_id','label','source','native')}|facts
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for fact in pool.map(compute,data['rows']):
            records.append(fact)
            if len(records)%1000==0:
                print(json.dumps({'phase':'E54_colour_coverage','parents':len(records),'total':len(data['rows'])}),flush=True)
    value={'state':'E54_colour_descriptors_complete','contract_sha256':digest(CONTRACT),'records':records,'new_model_scores':0}
    fixed_write(DESCRIPTORS,value)
    return {'parents':len(records),'descriptors_sha256':digest(DESCRIPTORS)}


def report():
    data=load();config=json.loads(CONTRACT.read_text())
    if config['code_sha256']!=digest(__file__) or config['data_contract_sha256']!=digest(DATA_CONTRACT):
        raise ValueError('colour audit code/data changed')
    facts=json.loads(DESCRIPTORS.read_text())
    if facts['contract_sha256']!=digest(CONTRACT):raise ValueError('descriptors not bound')
    by_parent={r['parent_id']:r for r in facts['records']}
    source_counts=defaultdict(lambda:{'parents':0,'near_monochrome':0})
    for r in facts['records']:
        key=f"{r['label']}:{r['source']}:{'native_added' if r['native'] else 'base'}"
        source_counts[key]['parents']+=1
        source_counts[key]['near_monochrome']+=int(r['near_monochrome_global_crop'])
    exposure=[]
    for fold in data['folds']:
        groups=defaultdict(lambda:{'parents':0,'near_monochrome':0})
        for row,role in zip(data['rows'],fold['roles'],strict=True):
            key=f"{role}:{row['label']}";groups[key]['parents']+=1
            groups[key]['near_monochrome']+=int(by_parent[row['parent_id']]['near_monochrome_global_crop'])
        exposure.append({'fold':fold['fold'],'groups':dict(groups)})
    result_path=ROOT/'result.json';results=json.loads(result_path.read_text());errors={}
    for arm in ('head_only','last2_anchor','full_expanded3'):
        counts=defaultdict(lambda:{'parents':0,'errors':0})
        for f in results['arms'][arm]['folds']:
            path=next(p for p in results['result_bindings'] if p.endswith(f"/{arm}_fold{f['fold']}.json"))
            if digest(path)!=results['result_bindings'][path]:raise ValueError('prediction result changed')
            for row in json.loads(Path(path).read_text())['observations']:
                group='near_monochrome' if by_parent[row['parent_id']]['near_monochrome_global_crop'] else 'other'
                key=f"{row['condition']}:{row['label']}:{group}"
                counts[key]['parents']+=1
                counts[key]['errors']+=int(row['predicted_ai']!=bool(row['label']))
        errors[arm]=dict(counts)
    value={'state':'E54_exploratory_colour_coverage_complete','contract_sha256':digest(CONTRACT),
           'descriptors_sha256':digest(DESCRIPTORS),'result_sha256':digest(result_path),
           'source_counts':dict(source_counts),'fold_exposure':exposure,'fixed_prediction_errors':errors,
           'new_model_scores':0,'classifier_changed':False,'final_reserves_opened':False,
           'limits':['Error-selected motivation and reused TRAIN validation: exploratory association only.',
                     'Global crop monochrome indicator is not a source truth label or an authenticity rule.',
                     'Correlation with error cannot isolate colour from source/content/processing confounds.']}
    fixed_write(ROOT/'color_result.json',value);fixed_write(EVIDENCE/'e54_color_audit.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','report'])
    args=parser.parse_args();print(json.dumps({'prepare':prepare,'report':report}[args.phase](),indent=2))
