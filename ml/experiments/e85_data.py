"""Complete four-condition TRAIN pairing after verified E84B source/transport extraction."""
import joblib
import numpy as np
from experiments.e65_acquisition import read,digest
from experiments.e76_fit import combine_rows
from pixelproof.project_paths import DATA_ROOT
CONDITIONS=['clean','assigned_transport','q75','social_q75']
WIDTHS={'dino':3072,'clip':1536,'dear':1640}


def append_social(old,social,rows,binding):
    if str(social['binding'])!=binding or str(social['condition'])!='social_q75' or \
            list(social['parents'])!=[r['parent_id'] for r in rows] or len(social['roles'])!=len(rows) or \
            set(social['roles'])!={'TRAIN'} or any(r['role'].upper()!='TRAIN' for r in rows):
        raise ValueError('complete ordered TRAIN social pairing required')
    result={}
    for key,width in WIDTHS.items():
        if old[key].shape!=(len(rows),3,width) or social[key].shape!=(len(rows),width) or \
                old[key].dtype!=np.float32 or social[key].dtype!=np.float32 or \
                not np.isfinite(old[key]).all() or not np.isfinite(social[key]).all():
            raise ValueError('complete finite old3/new1 feature views required')
        result[key]=np.concatenate([old[key],social[key][:,None,:]],axis=1)
        if not np.array_equal(result[key][:,:3],old[key]):raise ValueError('original three views changed')
    return result


def load():
    previous=read(DATA_ROOT/'e71/fit_contract.json');paths={k:v['path'] for k,v in previous['inputs'].items()}
    old_rows=read(paths['manifest'])['rows'];new_rows=read(DATA_ROOT/'e72/training_manifest.json')['rows'];rows=combine_rows(old_rows,new_rows)
    old={}
    with np.load(paths['features'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(paths['manifest']):raise ValueError('teacher manifest changed')
        old['dino']=a['features']
    with np.load(paths['clip'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e71/features_contract.json') or list(a['parents'])!=[r['parent_id'] for r in old_rows] or set(a['roles'])!={'TRAIN'}:
            raise ValueError('original CLIP feature pairing changed')
        old['clip']=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e75/features_contract.json') or list(a['parents'])!=[r['parent_id'] for r in new_rows] or set(a['roles'])!={'TRAIN'}:
            raise ValueError('MIDD feature pairing changed')
        for key in ['dino','clip']:old[key]=np.concatenate([old[key],a[key]])
    with np.load(DATA_ROOT/'e79/dear_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e79/features_contract.json') or list(a['parents'])!=[r['parent_id'] for r in rows] or set(a['roles'])!={'TRAIN'}:
            raise ValueError('complete DEAR TRAIN pairing changed')
        old['dear']=a['features']
    with np.load(DATA_ROOT/'e84b/social_features.npz',allow_pickle=False) as a:features=append_social(old,a,rows,digest(DATA_ROOT/'e84b/features_contract.json'))
    return rows,features,joblib.load(paths['reference'])['head']
