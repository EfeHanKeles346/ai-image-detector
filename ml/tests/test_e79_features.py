import numpy as np
import pytest
import torch
from experiments import e79_features as m


def test_parent_views_stay_separate_and_std_is_population_std():
    raw=np.stack([np.stack([np.full(820,v*10+c,dtype=np.float32) for c in range(3)]) for v in range(3)])
    features=m.aggregate(raw)
    np.testing.assert_allclose(features[:,:820],np.repeat([[1.],[11.],[21.]],820,axis=1))
    np.testing.assert_allclose(features[:,820:],np.sqrt(2/3),rtol=1e-6)
    with pytest.raises(ValueError):m.aggregate(raw[:2])


def test_chunk_replay_rejects_parent_source_contract_and_body_corruption(tmp_path):
    row={'parent_id':'p','sha256':'s'};raw=np.zeros((3,3,820),np.float32);features=m.aggregate(raw)
    a={'binding':np.array('binding'),'parent_id':np.array('p'),'source_sha256':np.array('s'),
       'raw':raw,'features':features,'raw_sha256':np.array(m.array_sha(raw)),'features_sha256':np.array(m.array_sha(features))}
    path=tmp_path/'chunk.npz';np.savez_compressed(path,**a)
    assert np.array_equal(m.read_chunk(path,'binding',row)[1],features)
    for binding,changed in [('other',row),('binding',row|{'parent_id':'other'}),('binding',row|{'sha256':'other'})]:
        with pytest.raises(ValueError,match='binding'):m.read_chunk(path,binding,changed)
    a['raw'][0,0,0]=1;np.savez_compressed(path,**a)
    with pytest.raises(ValueError,match='body'):m.read_chunk(path,'binding',row)


def test_feature_encoding_preserves_view_crop_order_without_classifier_head():
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.gate=torch.nn.Module()
            self.gate.register_buffer('gate',torch.cat([torch.ones(820),torch.zeros(1228)]))
        def pooled(self,x):return x.mean(dim=(1,2,3))[:,None].repeat(1,2048)
        def forward(self,x):raise AssertionError('classifier head must never execute')
    crops=np.stack([np.full((224,224,3),i*20,np.uint8) for i in range(9)]).reshape(3,3,224,224,3)
    raw=m.encode(FakeEncoder(),crops)
    expected=m.model.normalize(crops.reshape(9,224,224,3)).mean(dim=(1,2,3)).numpy().reshape(3,3)
    np.testing.assert_allclose(raw[:,:,0],expected,atol=1e-6)
    assert raw.shape==(3,3,820)
