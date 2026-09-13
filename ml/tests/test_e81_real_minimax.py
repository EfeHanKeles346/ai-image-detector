import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit
from experiments.e64_constrained import AI_CUT
from experiments import e81_real_minimax as m


def population():
    x=np.array([[1.,1.],[2.,1.],[-1.,1.],[-2.,1.]])
    logits=logit(AI_CUT)+np.array([1.,-1.,1.,-1.]);labels=np.array([1,1,0,0])
    return x,logits,labels,np.array(['ai','ai','hard','easy']),np.array(['a','b','c','d']),np.array(['clean']*4)


def test_real_only_epigraph_and_objective_gradients():
    risk=m.RealRisks(*population());v=np.array([.1,-.2,1.])
    np.testing.assert_allclose(risk.epigraph(v)[1],approx_derivative(lambda z:risk.epigraph(z)[0],v),atol=1e-8)
    np.testing.assert_allclose(m.objective(v)[1],approx_derivative(lambda z:m.objective(z)[0],v).ravel(),atol=1e-8)
    assert len(risk.epigraph(v)[0])==2
    args=list(population());args[1]=args[1].copy();args[1][:2]+=100
    # Already-protected AI confidence is a constraint, not an incentive to grow logits.
    np.testing.assert_array_equal(risk.values(v[:-1])[0],m.RealRisks(*args).values(v[:-1])[0])


def test_worst_real_can_improve_without_reducing_even_missed_ai_logits():
    args=population();w,r=m.fit(*args);shift=args[0]@w
    assert r['success'] and r['max_constraint_violation']<=1e-8
    assert min(shift[:2])>=-1e-8 and args[1][2]+shift[2]<logit(AI_CUT)
    assert args[1][3]+shift[3]<logit(AI_CUT)
    assert max(g['final_mean_cut_bce'] for g in r['groups'])<max(g['initial_mean_cut_bce'] for g in r['groups'])
    assert r['all_ai_views_protected']==2


def test_conflicting_missed_ai_cannot_be_sacrificed_for_real_objective():
    x=np.ones((2,1));logits=logit(AI_CUT)+np.array([-1.,1.]);labels=np.array([1,0])
    w,r=m.fit(x,logits,labels,np.array(['ai','real']),np.array(['a','b']),np.array(['clean']*2))
    assert r['success'] and r['minimum_ai_logit_shift']>=-1e-8
    assert abs(w[0])<1e-8
