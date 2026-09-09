"""Close the 120 unselected E51 AI reserve parents before native expansion."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import sqlite3

from PIL import Image,ImageOps

from experiments.e51_prefit_audit import fingerprint,cross_role_matches
from experiments.e51_train_cal_realize import _q75
from experiments.e53_offline import ROOT,EVIDENCE,digest,fixed_write
from pixelproof.project_paths import DATA_ROOT


def close(records,allowed):
    route_path=DATA_ROOT/'e51/route/contract_untransferred.json'
    if digest(route_path)!='975e8164477c7234292ba87449007f0ee4c8b65eb582f25a8b0d81140ec315e4':raise ValueError('latest reserve contract changed')
    route=json.loads(route_path.read_text())['roles']['development_ai']['rows']
    selected=json.loads((DATA_ROOT/'e51/development/manifest_unscored.json').read_text())['rows']
    ids={r['parent_id'] for r in selected if r['label']==1}
    remaining=[r for r in route if r['identity'] not in ids]
    if len(route)!=920 or len(ids)!=800 or len(remaining)!=120:raise ValueError('E51 reserve coverage changed')
    def verify(row):
        path=DATA_ROOT/'e51/development/ai_reserve'/(row['expected_sha256']+'.image')
        raw=path.read_bytes()
        if len(raw)!=row['expected_bytes'] or hashlib.sha256(raw).hexdigest()!=row['expected_sha256']:raise ValueError('reserve original changed')
        with Image.open(BytesIO(raw)) as image:derived=_q75(ImageOps.exif_transpose(image).convert('RGB'))
        qpath=DATA_ROOT/'e51/development/q75'/(row['expected_sha256']+'.jpg')
        if qpath.read_bytes()!=derived:raise ValueError('reserve Q75 changed')
        return [{**fingerprint(body),'parent_id':row['identity'],'condition':condition,'label':1}
                for body,condition in ((raw,'original'),(derived,'q75'))]
    with ThreadPoolExecutor(max_workers=4) as pool:protected=[r for pair in pool.map(verify,remaining) for r in pair]
    with sqlite3.connect(f'file:{ROOT}/native_fingerprints.sqlite3?mode=ro',uri=True) as db:
        facts={s:json.loads(f) for s,f in db.execute('SELECT sha,facts FROM fingerprints')}
    queries=[{**facts[r['sha256']],'parent_id':'e32:'+r['record_id'],'label':int(r['label']=='ai'),'condition':'original'}
             for r in records if r['record_id'] in allowed]
    matches=cross_role_matches(queries,protected)
    excluded={m['train_parent'].removeprefix('e32:') for m in matches}
    result={'state':'latest_E51_unselected_reserves_closed','original_reserve_parents':920,
            'already_protected_selected_parents':800,'supplementary_parents':120,'supplementary_observations':240,
            'query_parents':len(queries),'matched_parent_ids':sorted(excluded),'matches':matches,
            'model_scores_created':0,'image_bytes_downloaded':0,
            'route_sha256':digest(route_path),'code_sha256':digest(__file__)}
    fixed_write(ROOT/'latest_reserves.json',result)
    fixed_write(EVIDENCE/'e53_latest_reserves.json',{k:v for k,v in result.items() if k!='matches'}|{'report_sha256':digest(ROOT/'latest_reserves.json')})
    return allowed-excluded
