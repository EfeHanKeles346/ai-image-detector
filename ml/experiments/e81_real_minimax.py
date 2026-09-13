"""Worst-REAL-source loss with unchanged all-AI confidence and correct-REAL guards."""
import numpy as np
from scipy.optimize import minimize
from experiments.e68_minimax import GroupRisks
from experiments.e64_constrained import L2,MAX_ITER,FTOL
from experiments.e73_fit import constraints_for


class RealRisks:
    def __init__(self,*args):
        self.groups=GroupRisks(*args)
        self.real=self.groups.classes==0

    def values(self,w):
        values,jac=self.groups.risks(w)
        return values[self.real],jac[self.real]

    def epigraph(self,v):
        values,jac=self.values(v[:-1])
        return v[-1]-values,np.column_stack([-jac,np.ones(len(values))])


def objective(v):
    # Keep the prior REAL loss coefficient .5 and L2 coefficient .01; no AI loss term.
    return (.5*float(v[-1])+.5*L2*float(v[:-1]@v[:-1]),np.r_[L2*v[:-1],.5])


def fit(x,baseline_logits,labels,sources,parents,conditions,check=lambda:None):
    risk=RealRisks(x,baseline_logits,labels,sources,parents,conditions)
    extended=np.column_stack([x,np.zeros(len(x))])
    bounds,ai,real,slack=constraints_for(extended,baseline_logits,labels)
    initial_values,_=risk.values(np.zeros(x.shape[1]))
    initial=np.r_[np.zeros(x.shape[1]),initial_values.max()+1e-10];trace=[]
    def callback(v):
        check();trace.append(objective(v)[0])
    result=minimize(objective,initial,jac=True,method='SLSQP',
        constraints=[*bounds,{'type':'ineq','fun':lambda v:risk.epigraph(v)[0],
                              'jac':lambda v:risk.epigraph(v)[1]}],
        callback=callback,options={'maxiter':MAX_ITER,'ftol':FTOL,'disp':False})
    w=result.x[:-1];shift=x@w
    linear=max(0.,float(np.max(-shift[ai])),float(np.max(shift[real]-slack)) if real.any() else 0.)
    nonlinear=max(0.,float(-risk.epigraph(result.x)[0].min()))
    final_values,_=risk.values(w)
    return w,{'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
        'objective':float(result.fun),'trace':trace,'max_constraint_violation':max(linear,nonlinear),
        'linear_constraint_violation':linear,'epigraph_violation':nonlinear,
        'all_ai_views_protected':int(ai.sum()),'minimum_ai_logit_shift':float(shift[ai].min()),
        'worst_real_bound':float(result.x[-1]),
        'groups':[{'source':str(k[1]),'condition':str(k[2]),
                   'initial_mean_cut_bce':float(initial_values[i]),'final_mean_cut_bce':float(final_values[i])}
                  for i,k in enumerate(k for k in risk.groups.keys if k[0]==0)]}
