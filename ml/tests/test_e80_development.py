import numpy as np
import pytest
import torch
from experiments import e80_development as m


def test_padded_microbatch_never_adds_views_or_changes_aggregation():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.gate=torch.nn.Module()
            self.gate.register_buffer('gate',torch.cat([torch.ones(820),torch.zeros(1228)]));self.batch_sizes=[]
        def pooled(self,x):
            self.batch_sizes.append(len(x));return x.mean(dim=(1,2,3))[:,None].repeat(1,2048)
    net=Encoder();crops=np.stack([np.full((224,224,3),i*12,np.uint8) for i in range(15)]).reshape(5,3,224,224,3)
    actual=m.dear_features(net,crops)
    raw=m.dear_model.normalize(crops.reshape(15,224,224,3)).mean(dim=(1,2,3)).numpy().reshape(5,3)
    np.testing.assert_allclose(actual[:,0],raw.mean(axis=1),atol=1e-6)
    np.testing.assert_allclose(actual[:,820],raw.std(axis=1),atol=1e-6)
    assert net.batch_sizes==[9,9] and actual.shape==(5,1640)
    with pytest.raises(ValueError,match='complete'):m.dear_features(net,crops[:0])


def test_development_freeze_refuses_failed_train_before_reference_or_pixels(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'validate_fit',lambda:None)
    monkeypatch.setattr(m,'read',lambda path:{'dev_scoring_permitted':False})
    monkeypatch.setattr(m,'digest',lambda path:'same')
    with pytest.raises(ValueError,match='TRAIN guard failed'):m.freeze()
