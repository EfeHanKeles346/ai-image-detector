"""Exact E80 multimodal map plus the frozen supervised E82 latent coordinates."""
import numpy as np
from scipy.special import expit
from experiments import e80_model as base,e82_representation as supervised
from experiments.e64_constrained import AI_CUT,REAL_CUT
BASE_KEYS=(*base.BASE_KEYS,'dear_center','dear_scale','dear_mean','dear_components','dear_scales',
    'dear_head_direction','dear_head_bias','dear_head_center','dear_head_scale')
SUPERVISED_KEYS=('input_center','input_scale','latent_center','latent_scale',
    *(f'nn_{layer}_{kind}' for layer in (0,2,4) for kind in ('weight','bias')))


def assemble(previous,representation):
    a={k:np.array(previous[k],copy=True) for k in BASE_KEYS}
    a.update({k:np.array(representation[k],copy=True) for k in SUPERVISED_KEYS})
    a['weights']=np.zeros(450)
    return a


def project(head,original,clip,dear,a):
    fixed={k:a[k] for k in BASE_KEYS};fixed['weights']=np.zeros(386)
    old=base.project(head,original,clip,dear,fixed)[:,:-1]
    latent=supervised.coordinates(old,a)
    result=np.column_stack([old,latent,np.ones(len(old))])
    if result.shape!=(len(original),450) or np.shape(a['weights'])!=(450,) or not np.isfinite(result).all():
        raise ValueError('invalid450D supervised map')
    return result


def predict(head,original,clip,dear,a):
    base.base.base.linear.aligned(original,clip)
    if len(dear)!=len(original):raise ValueError('aligned DEAR parent views required')
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,clip,dear,a)@a['weights'])
