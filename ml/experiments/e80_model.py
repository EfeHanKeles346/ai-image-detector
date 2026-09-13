"""Frozen E77 map plus DEAR-r TRAIN PCs and its fixed mean-crop head response."""
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments import e77_model as base
from experiments.e64_constrained import AI_CUT,REAL_CUT

BASE_KEYS=(*base.BASE_KEYS,*base.MANIFOLD_KEYS)


def checkpoint_direction(path):
    """Read only the hash-pinned official gate/head; never execute image inference."""
    import torch
    from experiments.e65_acquisition import digest
    from experiments.e78_acquisition import SHA
    if digest(path)!=SHA:raise ValueError('DEAR-r publisher checksum differs')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if set(checkpoint)!={'model'}:raise ValueError('unexpected DEAR checkpoint schema')
    state=checkpoint['model'];gate=state['gate.gate'];weight=state['backbone.fc.weight'];bias=state['backbone.fc.bias']
    if gate.shape!=(2048,) or set(gate.tolist())!={0.,1.} or int(gate.sum())!=820 or \
            weight.shape!=(1,2048) or bias.shape!=(1,) or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('invalid fixed DEAR gate/head')
    return np.concatenate([weight[0,gate.bool()].double().numpy(),np.zeros(820)]),float(bias[0])


def fit_map(previous,dear,head_direction,head_bias):
    if dear.ndim!=2 or min(dear.shape)<64 or not np.isfinite(dear).all():raise ValueError('complete finite DEAR TRAIN matrix required')
    direction=np.asarray(head_direction,dtype=np.float64)
    if direction.shape!=(dear.shape[1],) or not np.isfinite(direction).all() or not np.isfinite(head_bias):
        raise ValueError('aligned finite frozen DEAR head required')
    a={k:np.array(previous[k],copy=True) for k in BASE_KEYS}
    scaler=StandardScaler().fit(dear.astype(np.float64));z=scaler.transform(dear.astype(np.float64))
    pca=PCA(n_components=64,svd_solver='randomized',random_state=80,iterated_power=3).fit(z)
    scales=np.sqrt(pca.explained_variance_)
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('nondegenerate DEAR map required')
    response=dear.astype(np.float64)@direction+head_bias
    head_scaler=StandardScaler().fit(response[:,None])
    a.update(dear_center=scaler.mean_,dear_scale=scaler.scale_,dear_mean=pca.mean_,
             dear_components=pca.components_,dear_scales=scales,
             dear_head_direction=direction.copy(),dear_head_bias=np.array(head_bias),
             dear_head_center=head_scaler.mean_,dear_head_scale=head_scaler.scale_,weights=np.zeros(386))
    return a


def dear_coordinates(dear,a):
    if dear.ndim!=2 or dear.shape[1]!=len(a['dear_center']) or not np.isfinite(dear).all():
        raise ValueError('aligned finite DEAR inputs required')
    result=((dear.astype(np.float64)-a['dear_center'])/a['dear_scale']-a['dear_mean'])@a['dear_components'].T/a['dear_scales']
    if result.shape!=(len(dear),64) or not np.isfinite(result).all():raise ValueError('invalid DEAR coordinates')
    return result


def project(head,original,clip,dear,a):
    if len(dear)!=len(original):raise ValueError('aligned DEAR parent views required')
    fixed={k:a[k] for k in BASE_KEYS};fixed['weights']=np.zeros(321)
    old=base.project(head,original,clip,fixed)[:,:-1]
    coords=dear_coordinates(dear,a)
    response=(dear.astype(np.float64)@a['dear_head_direction']+a['dear_head_bias']-a['dear_head_center'])/a['dear_head_scale']
    result=np.column_stack([old,coords,response,np.ones(len(old))])
    if result.shape!=(len(original),386) or np.shape(a['weights'])!=(386,) or not np.isfinite(result).all():
        raise ValueError('invalid386D forensic map')
    return result


def predict(head,original,clip,dear,a):
    base.base.linear.aligned(original,clip)
    if len(dear)!=len(original):raise ValueError('aligned DEAR parent views required')
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,clip,dear,a)@a['weights'])
