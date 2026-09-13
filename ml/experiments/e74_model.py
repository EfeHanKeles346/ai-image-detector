"""Original/CLIP bilinear coordinates for full-AI-confidence-preserving correction."""
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from experiments import e71_model as linear
from experiments.e70_model import sketch_parameters,bilinear_sketch
from experiments.e64_constrained import AI_CUT,REAL_CUT

KEYS=(*linear.BASE_KEYS,'clip_center','clip_scale','clip_mean','clip_components','clip_scales')


def coordinates(head,original,clip,a):
    # E71 validates its129-coordinate width; the outer model has257 coefficients.
    base={k:a[k] for k in KEYS};base['weights']=np.zeros(129)
    return linear.project(head,original,clip,base)[:,:-1]


def fit_map(head,original,clip,previous):
    a={k:np.array(previous[k],copy=True) for k in KEYS}
    z=coordinates(head,original,clip,a)
    a.update(sketch_parameters())
    raw=bilinear_sketch(z[:,:64],z[:,64:],a['sketch_hash'],a['sketch_sign'])
    scaler=StandardScaler().fit(raw)
    a.update(interaction_mean=scaler.mean_,interaction_scale=scaler.scale_,weights=np.zeros(257))
    return a


def project(head,original,clip,a):
    z=coordinates(head,original,clip,a)
    raw=bilinear_sketch(z[:,:64],z[:,64:],a['sketch_hash'],a['sketch_sign'])
    result=np.column_stack([z,(raw-a['interaction_mean'])/a['interaction_scale'],np.ones(len(z))])
    if result.shape[1]!=len(a['weights']) or not np.isfinite(result).all():raise ValueError('invalid257D interaction map')
    return result


def predict(head,original,clip,a):
    linear.aligned(original,clip)
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,clip,a)@a['weights'])
