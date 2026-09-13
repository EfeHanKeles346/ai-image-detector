"""Frozen E77 coordinates plus complementary DEAR-r TRAIN principal components."""
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments import e77_model as base
from experiments.e64_constrained import AI_CUT,REAL_CUT

BASE_KEYS=(*base.BASE_KEYS,*base.MANIFOLD_KEYS)


def fit_map(previous,dear):
    if dear.ndim!=2 or min(dear.shape)<64 or not np.isfinite(dear).all():raise ValueError('complete finite DEAR TRAIN matrix required')
    a={k:np.array(previous[k],copy=True) for k in BASE_KEYS}
    scaler=StandardScaler().fit(dear.astype(np.float64));z=scaler.transform(dear.astype(np.float64))
    pca=PCA(n_components=64,svd_solver='randomized',random_state=80,iterated_power=3).fit(z)
    scales=np.sqrt(pca.explained_variance_)
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('nondegenerate DEAR map required')
    a.update(dear_center=scaler.mean_,dear_scale=scaler.scale_,dear_mean=pca.mean_,
             dear_components=pca.components_,dear_scales=scales,weights=np.zeros(385))
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
    result=np.column_stack([old,dear_coordinates(dear,a),np.ones(len(old))])
    if result.shape!=(len(original),385) or np.shape(a['weights'])!=(385,) or not np.isfinite(result).all():
        raise ValueError('invalid385D forensic map')
    return result


def predict(head,original,clip,dear,a):
    base.base.linear.aligned(original,clip)
    if len(dear)!=len(original):raise ValueError('aligned DEAR parent views required')
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,clip,dear,a)@a['weights'])
