"""FIT-only convex logit consistency across complete equal-weight parent views."""
import numpy as np
from scipy.optimize import minimize
from pixelproof import holdout_linear

STRENGTH=.1


def covariance(x,weights,views=4):
    x=np.asarray(x,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or type(views) is not int or views<2 or not len(x) or len(x)%views or \
            weights.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(weights).all() or \
            np.any(weights<=0) or not np.isclose(weights.sum(),1):
        raise ValueError('Complete finite FIT parent views and positive unit weights required')
    grouped=weights.reshape(-1,views)
    if not np.array_equal(grouped,np.repeat(grouped[:,:1],views,axis=1)):
        raise ValueError('All views of each parent require identical weights')
    values=x.reshape(-1,views,x.shape[1]);delta=(values-values.mean(axis=1,keepdims=True)).reshape(x.shape)
    result=delta.T@(weights[:,None]*delta)
    return (result+result.T)/2


def objective(parameters,x,y,weights,matrix):
    loss,gradient=holdout_linear.objective(parameters,x,y,weights)
    regularizer=matrix@parameters[:-1]
    return loss+.5*STRENGTH*float(parameters[:-1]@regularizer),gradient+np.r_[STRENGTH*regularizer,0.]


def fit_head(x,y,weights,check=lambda:None):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y);weights=np.asarray(weights,dtype=np.float64)
    matrix=covariance(x,weights)
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1} or \
            not np.all(y.reshape(-1,4)==y.reshape(-1,4)[:,:1]):
        raise ValueError('Each complete parent needs one fixed binary class')
    check();trace=[]
    def callback(parameters):
        check();trace.append(float(objective(parameters,x,y,weights,matrix)[0]))
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(x,y,weights,matrix),jac=True,method='L-BFGS-B',
        callback=callback,options={'maxiter':500,'ftol':1e-12,'gtol':1e-7})
    loss,gradient=objective(result.x,x,y,weights,matrix)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(loss) or np.abs(gradient).max()>1e-5:
        raise RuntimeError('Registered consistency head did not converge')
    return result.x,{'success':bool(result.success),'iterations':int(result.nit),'objective':float(loss),
        'gradient_max_abs':float(np.abs(gradient).max()),'FIT_weighted_view_logit_variance':float(result.x[:-1]@matrix@result.x[:-1]),
        'trace':trace}
