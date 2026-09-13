"""Frozen REAL reconstruction residual appended to the exact E74 coordinate map."""
import numpy as np
from scipy.special import expit
from experiments import e74_model as base
from experiments import e77_representation as manifold
from experiments.e64_constrained import AI_CUT,REAL_CUT

BASE_KEYS=(*base.KEYS,'sketch_hash','sketch_sign','interaction_mean','interaction_scale')
MANIFOLD_KEYS=('real_center','real_scale','residual_center','residual_scale','residual_mean',
               'residual_components','residual_scales',
               *(f'ae_{layer}_{kind}' for layer in (0,2,4,6) for kind in ('weight','bias')))


def assemble(previous,representation):
    a={k:np.array(previous[k],copy=True) for k in BASE_KEYS}
    a.update({k:np.array(representation[k],copy=True) for k in MANIFOLD_KEYS})
    a['weights']=np.zeros(321)
    return a


def project(head,original,clip,a):
    fixed={k:a[k] for k in BASE_KEYS};fixed['weights']=np.zeros(257)
    old=base.project(head,original,clip,fixed)[:,:-1]
    residual=manifold.coordinates(clip,a)
    result=np.column_stack([old,residual,np.ones(len(old))])
    if result.shape!=(len(original),321) or np.shape(a['weights'])!=(321,) or not np.isfinite(result).all():
        raise ValueError('invalid321D manifold map')
    return result


def predict(head,original,clip,a):
    base.linear.aligned(original,clip)
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,clip,a)@a['weights'])
