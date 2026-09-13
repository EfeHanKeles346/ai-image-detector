import numpy as np
import pytest
import torch
from torchvision import transforms
from experiments import e78_model as m


def test_normalization_matches_official_tensor_and_imagenet_recipe():
    rgb=np.random.default_rng(78).integers(0,256,size=(3,41,57,3),dtype=np.uint8)
    official=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    expected=torch.stack([official(x) for x in rgb])
    assert torch.equal(m.normalize(rgb),expected)
    for bad in [rgb.astype(np.float32),rgb[:0],rgb[:,:,:20],rgb[:,:,:,:2]]:
        with pytest.raises(ValueError,match='RGB'):m.normalize(bad)


def test_strict_checkpoint_rejects_missing_backbone_and_invalid_gate(tmp_path,monkeypatch):
    torch.set_num_threads(2);net=m.GatedResNet();state=net.state_dict()
    state['gate.gate'].zero_();state['gate.gate'][:820]=1
    path=tmp_path/'model.pth';torch.save({'model':state},path)
    monkeypatch.setattr(m,'digest',lambda path:m.SHA)
    loaded=m.load(path)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert loaded.backbone.conv1.stride==(1,1) and loaded.backbone.maxpool.stride==1
    state.pop('backbone.conv1.weight');torch.save({'model':state},path)
    with pytest.raises(RuntimeError,match='Missing key'):m.load(path)
    state['gate.gate'][0]=.5;torch.save({'model':state},path)
    with pytest.raises(ValueError,match='gate'):m.load(path)


def test_checkpoint_digest_failure_precedes_deserialization(tmp_path):
    path=tmp_path/'bad.pth';path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='checksum'):m.load(path)
