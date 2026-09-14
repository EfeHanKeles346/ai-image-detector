"""Conservative declared-source/prompt/family components for internal TRAIN audits.

This cannot certify unknown semantic dependencies or pretrained-encoder ancestry.
"""
from collections import defaultdict
import numpy as np


def publisher(source):
    if source.startswith('rr:'): return 'rr_publisher_all_topics'
    if source.startswith('e36:'):
        return 'e36_real_publisher' if source.startswith('e36:device_') else 'e36_ai_shared_prompt_batch'
    if source.startswith('e32:csafe-'): return 'csafe_publisher'
    if source in {'e32:flux2-klein-9b','e32:qwen-image-2512'}: return 'e32_flux_qwen_shared_prompts'
    if source.startswith('e32:nano-banana'): return 'e32_nano_family'
    if source.startswith('MIDD:'): return 'MIDD_all_sensors'
    if source.startswith('SID:'): return 'SID_all_cameras'
    return source


def family(source):
    if source in {'e32:flux2-klein-9b','e36:FLUX.2_max'}: return 'declared_FLUX'
    if source in {'e32:qwen-image-2512','e36:Qwen-Image-2.0-pro'}: return 'declared_Qwen'
    if source in {'e32:gpt-image-1','e36:gpt-image-2'}: return 'declared_OpenAI_image'
    if source.startswith('e32:nano-banana') or source=='e36:nano-banana-2.0': return 'declared_Gemini_image'
    return None


def components(rows, inherited):
    ids=[r['parent_id'] for r in rows]
    if not rows or len(set(ids))!=len(ids) or any(not isinstance(p,str) or not p for p in ids) or \
            any(r['role'].upper()!='TRAIN' or type(r['label']) is not int or r['label'] not in (0,1) for r in rows):
        raise ValueError('Unique explicit TRAIN parents and binary integer labels required')
    roots={p:p for p in ids}
    def find(p):
        while roots[p]!=p:
            roots[p]=roots[roots[p]];p=roots[p]
        return p
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: roots[max(a,b)]=min(a,b)
    aliases={}
    for r in rows:
        p=r['parent_id'];keys=[('publisher',publisher(r['source']))]
        f=family(r['source'])
        if f:keys.append(('family',f))
        if p in inherited:keys.append(('inherited',inherited[p]))
        if r.get('scene_group'):keys.append(('scene',r['scene_group']))
        for key in ('sha256','original_sha256','pixel_sha256','source_body_sha256'):
            value=r.get(key)
            if value:
                # Different body encodings must not be conflated with decoded pixels.
                keys.append(('pixel' if key=='pixel_sha256' else 'body',value))
        for key in keys:
            if key in aliases:union(p,aliases[key])
            else:aliases[key]=p
    return {p:find(p) for p in ids}


def outer_folds(rows, groups, n_folds=3):
    if type(n_folds) is not int or n_folds<2 or set(groups)!={r['parent_id'] for r in rows}:
        raise ValueError('Complete parent-component mapping and at least two folds required')
    members=defaultdict(list)
    for r in rows:members[groups[r['parent_id']]].append(r)
    counts={g:np.bincount([r['label'] for r in members[g]],minlength=2) for g in members}
    total=np.bincount([r['label'] for r in rows],minlength=2)
    if np.any(total==0):raise ValueError('Both classes required')
    bins=np.zeros((n_folds,2),dtype=np.int64);assignment={}
    for group in sorted(members,key=lambda g:(-float(np.max(counts[g]/total)),g)):
        def cost(fold):
            candidate=bins.copy();candidate[fold]+=counts[group]
            return float(np.sum((candidate/total)**2)),fold
        fold=min(range(n_folds),key=cost)
        bins[fold]+=counts[group];assignment[group]=fold
    if np.any(bins==0) or np.any(total-bins==0):
        raise ValueError('Conservative components cannot populate every fit/held-out class')
    return {p:assignment[g] for p,g in groups.items()}


def balanced_weights(rows, groups, views):
    """Unit total mass, half per class, equal component/parent/view within class."""
    if type(views) is not int or views<1 or set(groups)!={r['parent_id'] for r in rows}:
        raise ValueError('Complete FIT parent mapping and positive view count required')
    by_class=defaultdict(lambda:defaultdict(int))
    for r in rows:by_class[r['label']][groups[r['parent_id']]]+=1
    if set(by_class)!={0,1}:raise ValueError('Both FIT classes required')
    weights=[]
    for r in rows:
        label=r['label'];group=groups[r['parent_id']]
        weights.extend([1/(2*len(by_class[label])*by_class[label][group]*views)]*views)
    result=np.asarray(weights,dtype=np.float64)
    if not np.isclose(result.sum(),1.):raise ValueError('Invalid class/component weights')
    return result
