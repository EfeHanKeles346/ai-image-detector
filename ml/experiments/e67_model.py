"""E43 residual correction with original features plus fixed-crop blur response."""
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments.e64_constrained import AI_CUT,REAL_CUT,objective,fit

RANK=64


def fit_basis(head,original,blurred):
    if original.shape!=blurred.shape or original.ndim!=2 or not np.isfinite(original).all() or not np.isfinite(blurred).all():
        raise ValueError('finite aligned original/blurred features required')
    base=head[0].transform(original).astype(np.float64)
    pca=PCA(n_components=RANK,svd_solver='randomized',random_state=62,iterated_power=3).fit(base)
    arrays={'base_mean':pca.mean_,'base_components':pca.components_,'base_scales':np.sqrt(pca.explained_variance_)}
    base_variance=float(pca.explained_variance_ratio_.sum());del base
    delta=(original-blurred).astype(np.float64)
    scaler=StandardScaler().fit(delta);delta=scaler.transform(delta,copy=False)
    response=PCA(n_components=RANK,svd_solver='randomized',random_state=67,iterated_power=3).fit(delta)
    arrays.update(response_center=scaler.mean_,response_scale=scaler.scale_,response_mean=response.mean_,
                  response_components=response.components_,response_scales=np.sqrt(response.explained_variance_),
                  weights=np.zeros(2*RANK+1))
    if any(not np.isfinite(v).all() for v in arrays.values()) or any(np.any(arrays[k]<=0) for k in
            ['base_scales','response_scale','response_scales']):raise ValueError('invalid fitted response basis')
    return arrays,{'base_variance_explained':base_variance,'response_variance_explained':float(response.explained_variance_ratio_.sum())}


def project(head,original,blurred,a):
    if original.shape!=blurred.shape or original.ndim!=2:raise ValueError('aligned two-dimensional feature views required')
    base=(head[0].transform(original).astype(np.float64)-a['base_mean'])@a['base_components'].T/a['base_scales']
    delta=((original-blurred).astype(np.float64)-a['response_center'])/a['response_scale']
    response=(delta-a['response_mean'])@a['response_components'].T/a['response_scales']
    result=np.column_stack([base,response,np.ones(len(base))])
    if not np.isfinite(result).all():raise ValueError('nonfinite response projection')
    return result


def predict(head,original,blurred,a):
    if not np.any(a['weights']):return head.predict_proba(original)[:,1]
    return expit(head.decision_function(original)+project(head,original,blurred,a)@a['weights'])
