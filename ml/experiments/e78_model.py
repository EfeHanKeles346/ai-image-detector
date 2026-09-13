"""Strict DEAR-r inference equivalent built from the installed torchvision ResNet50."""
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from experiments.e65_acquisition import digest
from experiments.e78_acquisition import SHA


class GatedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet50(weights=None,num_classes=1)
        self.backbone.conv1.stride=(1,1);self.backbone.maxpool.stride=1
        self.gate=nn.Module();self.gate.register_buffer('gate',torch.ones(2048))
    def forward_features(self,x):
        b=self.backbone;x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x=b.layer4(b.layer3(b.layer2(b.layer1(x))))
        return x*self.gate.gate.view(1,-1,1,1)
    def pooled(self,x):
        return self.backbone.avgpool(self.forward_features(x)).flatten(1)
    def forward(self,x):
        return self.backbone.fc(self.pooled(x))


def load(path,device='cpu'):
    if digest(path)!=SHA:raise ValueError('DEAR-r publisher checksum differs')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if set(checkpoint)!={'model'} or not isinstance(checkpoint['model'],dict):raise ValueError('unexpected checkpoint schema')
    model=GatedResNet();state=checkpoint['model']
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite checkpoint tensor')
    gate=state.get('gate.gate')
    if gate is None or gate.shape!=(2048,) or set(gate.tolist())!={0.,1.} or int(gate.sum())!=820:
        raise ValueError('expected fixed DEAR-r bilateral gate')
    model.load_state_dict(state,strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model.to(device)


def normalize(rgb):
    if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[-1]!=3 or not rgb.shape[0] or min(rgb.shape[1:3])<32:
        raise ValueError('nonempty RGB uint8 batch with sides at least32 required')
    x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(0,3,1,2))).float()/255.
    return (x-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)


def encode(model,rgb):
    device=next(model.parameters()).device
    with torch.inference_mode():
        pooled=model.pooled(normalize(rgb).to(device))
        selected=pooled[:,model.gate.gate.bool()]
        logits=model.backbone.fc(pooled)
    features=selected.cpu().numpy();scores=logits.cpu().numpy().reshape(-1)
    if features.shape!=(len(rgb),820) or not np.isfinite(features).all() or not np.isfinite(scores).all():
        raise ValueError('invalid DEAR-r inference output')
    return features,scores
