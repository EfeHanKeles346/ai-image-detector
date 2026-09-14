"""FIT-only maps and a fixed convex head for internal source-holdout diagnostics."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA


def fit_map(train, width):
    train=np.asarray(train,dtype=np.float64)
    if train.ndim!=2 or type(width) is not int or width<1 or min(train.shape)<=width or not np.isfinite(train).all():
        raise ValueError('Enough finite FIT rows/coordinates for the fixed PCA width required')
    center=train.mean(axis=0);scale=np.maximum(train.std(axis=0),1e-6)
    normalized=(train-center)/scale
    pca=PCA(n_components=width,svd_solver='randomized',iterated_power=3,random_state=131,copy=False)
    pca.fit(normalized)
    return {'center':center,'scale':scale,'mean':pca.mean_,'components':pca.components_,
            'latent_scale':np.sqrt(np.maximum(pca.explained_variance_,1e-12))}


def project(values,a):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=len(a['center']) or not np.isfinite(values).all():
        raise ValueError('Finite aligned source features required')
    result=(((values-a['center'])/a['scale']-a['mean'])@a['components'].T)/a['latent_scale']
    if not np.isfinite(result).all():raise ValueError('Invalid projected features')
    return result


def objective(parameters,x,y,weights,penalty=.01):
    logit=x@parameters[:-1]+parameters[-1]
    loss=float(weights@(np.logaddexp(0,logit)-y*logit)+.5*penalty*(parameters[:-1]@parameters[:-1]))
    error=weights*(expit(logit)-y)
    gradient=np.r_[x.T@error+penalty*parameters[:-1],error.sum()]
    return loss,gradient


def fit_head(x,y,weights,check=lambda:None):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y);weights=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or weights.shape!=y.shape or set(y.tolist())!={0,1} or \
            not np.isfinite(x).all() or not np.isfinite(weights).all() or np.any(weights<=0) or \
            not np.isclose(weights.sum(),1.):
        raise ValueError('Complete finite class-balanced FIT inputs and unit weight mass required')
    trace=[]
    def callback(p):check();trace.append(objective(p,x,y,weights)[0])
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(x,y,weights),jac=True,method='L-BFGS-B',
                    callback=callback,options={'maxiter':500,'ftol':1e-12,'gtol':1e-7})
    loss,gradient=objective(result.x,x,y,weights)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(loss) or np.max(np.abs(gradient))>1e-5:
        raise RuntimeError('Fixed head did not converge to the registered tolerance')
    return result.x,{'success':bool(result.success),'iterations':int(result.nit),'objective':float(loss),
                    'gradient_max_abs':float(np.abs(gradient).max()),'trace':trace}


def predict(x,parameters):
    x=np.asarray(x,dtype=np.float64);parameters=np.asarray(parameters,dtype=np.float64)
    if x.ndim!=2 or parameters.shape!=(x.shape[1]+1,) or not np.isfinite(x).all() or not np.isfinite(parameters).all():
        raise ValueError('Finite aligned head inputs required')
    return expit(x@parameters[:-1]+parameters[-1])
