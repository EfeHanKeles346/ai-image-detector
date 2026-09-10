import numpy as np
import pytest
from experiments.e57_model import weights_with_mass


def test_real_supplement_preserves_ai_replay_weight_and_total_mass():
    y=np.repeat([0,0,1,1],3);s=np.repeat(['real_a','real_b','ai_a','ai_b'],3);p=np.repeat(['r1','r2','a1','a2'],3)
    old=weights_with_mass(y,s,p,12)
    yy=np.r_[y,[0]*6];ss=np.r_[s,['FiveK']*6];pp=np.r_[p,['f1']*3+['f2']*3]
    new=weights_with_mass(yy,ss,pp,12)
    np.testing.assert_allclose(new[yy==1],old[y==1],atol=1e-12)
    assert new.sum()==pytest.approx(12)
    assert new[yy==1].sum()==pytest.approx(6)
    assert np.all(new>0)
    with pytest.raises(ValueError):weights_with_mass(y,s,p,0)
