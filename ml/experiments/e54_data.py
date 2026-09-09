"""E54 immutable, admitted TRAIN crop cache; no acquisition or new test access."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageOps

from experiments import e42_features as dino
from experiments.e32_r0_input import source_path, PARQUET_SOURCES, _parquet_raw
from experiments.e51_pipeline import FEATURES
from experiments.e53_coverage import load as coverage, CONTRACT as COVERAGE_CONTRACT
from experiments.e53_expansion import NATIVE_FEATURES, CONTRACT as NATIVE_CONTRACT
from experiments.e53_offline import EVIDENCE, digest, fixed_write, publisher
from pixelproof.project_paths import DATA_ROOT

ROOT = DATA_ROOT/'e54'
CONTRACT = ROOT/'data_contract_v2.json'
RECEIPT = EVIDENCE/'e54_data_contract_v2.json'
TEACHER = ROOT/'teacher.npz'
INDEX = ROOT/'crop_index.json'


def freeze():
    if CONTRACT.exists():
        raise FileExistsError('E54 data already frozen')
    base, extra, v2 = coverage()
    rows = [dict(r, native=False) for r in base['rows']]
    rows += [dict(r, native=True, parent_id='e32:'+r['record_id'],
                  source='e32:'+r['source_id'], label=int(r['label']=='ai')) for r in extra['rows']]
    if len({r['parent_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate population identity')
    folds = []
    for fold in v2['folds']:
        mapping = defaultdict(set)
        for r in base['rows']:
            mapping[publisher(r['source'])].add(fold['roles'][r['parent_id']])
        if any(len(roles)!=1 for roles in mapping.values()):
            raise ValueError('publisher crosses roles')
        roles = [fold['roles'][r['parent_id']] if not r['native'] else
                 ('FIT' if mapping[publisher(r['source'])]=={'FIT'} else 'EXCLUDED') for r in rows]
        folds.append({'fold': fold['fold'], 'roles': roles})
    value = {'state': 'E54_data_frozen', 'code_sha256': digest(__file__),
        'coverage_contract_sha256': digest(COVERAGE_CONTRACT),
        'native_contract_sha256': digest(NATIVE_CONTRACT), 'rows': rows, 'folds': folds,
        'base_features_sha256': base['features_sha256'],
        'native_features_sha256': json.loads((EVIDENCE/'e53_native_features.json').read_text())['feature_sha256'],
        'views': ['clean', 'assigned_transport', 'q75'], 'source_downloads': 0,
        'policy': 'Admitted E53 TRAIN only; native rows FIT-only; no protected external CAL/test.'}
    fixed_write(CONTRACT, value)
    fixed_write(RECEIPT, {k:v for k,v in value.items() if k not in {'rows','folds'}} |
                {'parents': len(rows), 'contract_sha256': digest(CONTRACT)})
    return {'parents': len(rows), 'contract_sha256': digest(CONTRACT)}


def load():
    coverage()
    value = json.loads(CONTRACT.read_text())
    if value['code_sha256'] != digest(__file__) or digest(CONTRACT) != json.loads(
            RECEIPT.read_text())['contract_sha256']:
        raise ValueError('E54 data binding changed')
    if value['coverage_contract_sha256'] != digest(COVERAGE_CONTRACT) or value['native_contract_sha256'] != digest(NATIVE_CONTRACT):
        raise ValueError('upstream role contract changed')
    return value


def crop_views(raw, parent):
    with Image.open(BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=75, subsampling=2, optimize=False)
    with Image.open(BytesIO(stream.getvalue())) as opened:
        q75 = opened.convert('RGB')
    views = [dino.transport_image(image,'clean'),
             dino.transport_image(image,dino.assigned_transport(parent)), dino.transport_image(q75,'clean')]
    return np.stack([np.stack(dino.texture_crops(view)) for view in views])


def prepare():
    value = load(); binding = digest(CONTRACT); rows = value['rows']
    if INDEX.exists():
        raise FileExistsError('E54 crop cache complete')
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise ValueError('less than 20 GiB free on external work disk')
    for path, expected in ((FEATURES,value['base_features_sha256']), (NATIVE_FEATURES,value['native_features_sha256'])):
        if digest(path) != expected:
            raise ValueError('teacher feature archive changed')
    if not TEACHER.exists():
        with np.load(FEATURES,allow_pickle=False) as a:
            features,roles,parents,conditions=a['dino'],a['roles'],a['parents'],a['conditions']
            index = {(str(p),str(c)): i for i,(p,c) in enumerate(zip(parents,conditions,strict=True)) if roles[i]=='TRAIN'}
            teacher = [features[[index[(r['parent_id'],c)] for c in ('clean',dino.assigned_transport(r['parent_id']),'q75')]]
                       for r in rows if not r['native']]
        with np.load(NATIVE_FEATURES,allow_pickle=False) as a:
            if list(a['record_ids']) != [r['record_id'] for r in rows if r['native']]:
                raise ValueError('native teacher order changed')
            teacher.extend(a['features'])
        dino._save_npz(TEACHER, {'features': np.stack(teacher), 'binding': np.asarray(binding)})
    records = {}; started = time.monotonic()
    def process(pair):
        row, raw = pair
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('source bytes changed')
        name = hashlib.sha256(row['parent_id'].encode()).hexdigest()+'.npz'
        path = ROOT/'crops'/name
        if not path.exists():
            dino._save_npz(path, {'crops': crop_views(raw,row['parent_id']), 'binding': np.asarray(binding)})
        else:
            with np.load(path,allow_pickle=False) as a:
                if str(a['binding'])!=binding or a['crops'].shape!=(3,3,224,224,3):
                    raise ValueError('cached crop mismatch')
        return row['parent_id'], {'path': str(path), 'sha256': digest(path)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        def batch(pairs):
            records.update(pool.map(process,pairs))
            print(json.dumps({'phase':'E54_crop_cache','parents':len(records),'total':len(rows),
                              'seconds':round(time.monotonic()-started)}),flush=True)
        loose = [r for r in rows if not r['native'] or r['source_id'] not in PARQUET_SOURCES]
        for start in range(0,len(loose),32):
            batch([(r, (source_path(r['source_id'],r['source_key']) if r['native'] else Path(r['path'])).read_bytes())
                   for r in loose[start:start+32]])
        for source,(folder,column) in PARQUET_SOURCES.items():
            pending=[]
            for pair in _parquet_raw([r for r in rows if r['native'] and r['source_id']==source],folder,column):
                pending.append(pair)
                if len(pending)==32:
                    batch(pending);pending=[]
            if pending:
                batch(pending)
    result = {'state':'E54_crop_cache_complete','contract_sha256':binding,'teacher_sha256':digest(TEACHER),
              'parents':len(records),'records':records,'source_downloads':0}
    fixed_write(INDEX,result)
    fixed_write(EVIDENCE/'e54_crop_cache.json',{k:v for k,v in result.items() if k!='records'}|{'index_sha256':digest(INDEX)})
    return {k:v for k,v in result.items() if k!='records'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','prepare']);args=parser.parse_args()
    print(json.dumps({'freeze':freeze,'prepare':prepare}[args.phase](),indent=2))
