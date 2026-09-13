"""REAL-only nonlinear CLIP reconstruction residual; no classifier or DEV access."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E77 representation is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest,read,write_once
from experiments.e71_features import save_npz,array_sha
from experiments.e72_acquisition import resource_check
from experiments.e76_fit import validate as validate_previous,combine_rows
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
ROOT=DATA_ROOT/'e77';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'representation_contract.json';REPORT=ROOT/'representation.json'
ARTIFACT=ROOT/'manifold.npz';FEATURES=ROOT/'manifold_features.npz';CHECKPOINT=ROOT/'ae_checkpoint.pt'
LINEAR_LAYERS=(0,2,4,6)


def build_network(width=1536,hidden=256,latent=64):
    return torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.ReLU(),
        torch.nn.Linear(hidden,latent),torch.nn.ReLU(),torch.nn.Linear(latent,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,width))


def epoch_order(n,epoch):
    if n<=0 or not 0<=epoch<100:raise ValueError('invalid fixed epoch population')
    return np.random.default_rng(np.random.SeedSequence([77,epoch])).permutation(n)


def real_weights(rows):
    labels=np.repeat([r['label'] for r in rows],3)
    weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],3),np.repeat([r['parent_id'] for r in rows],3))
    mask=labels==0;selected=weights[mask];selected*=len(selected)/selected.sum()
    return mask,selected


def real_scaler(features,mask,weights):
    if features.ndim!=2 or mask.shape!=(len(features),) or weights.shape!=(int(mask.sum()),) or \
            not np.isfinite(features).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('aligned finite REAL normalization inputs required')
    fitted=StandardScaler().fit(features[mask].astype(np.float64),sample_weight=weights)
    return {'real_center':fitted.mean_,'real_scale':fitted.scale_}


def normalized(clip,a):
    if clip.ndim!=2 or clip.shape[1]!=len(a['real_center']) or not np.isfinite(clip).all() or \
            not np.isfinite(a['real_center']).all() or not np.isfinite(a['real_scale']).all() or np.any(a['real_scale']<=0):
        raise ValueError('invalid finite CLIP normalization')
    return ((clip.astype(np.float64)-a['real_center'])/a['real_scale']).astype(np.float32)


def export_network(net):
    return {'ae_'+key.replace('.','_'):value.detach().cpu().numpy().copy() for key,value in net.state_dict().items()}


def load_network(a):
    width=a['ae_0_weight'].shape[1];hidden=a['ae_0_weight'].shape[0];latent=a['ae_2_weight'].shape[0]
    net=build_network(width,hidden,latent)
    state={key:torch.from_numpy(np.array(a['ae_'+key.replace('.','_')],copy=True)) for key in net.state_dict()}
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):raise ValueError('invalid AE state')
    net.load_state_dict(state,strict=True);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    return net


def residuals(clip,a,batch_size=256):
    if batch_size<=0 or not len(clip):raise ValueError('positive inference batch and population required')
    x=normalized(clip,a);net=load_network(a);parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            batch=torch.from_numpy(x[start:start+batch_size]);parts.append(torch.abs(batch-net(batch)).numpy())
    return np.concatenate(parts)


def coordinates(clip,a):
    residual=residuals(clip,a)
    z=((residual.astype(np.float64)-a['residual_center'])/a['residual_scale']-a['residual_mean'])@a['residual_components'].T/a['residual_scales']
    if z.shape!=(len(clip),64) or not np.isfinite(z).all():raise ValueError('invalid manifold coordinates')
    return z


def cpu_state(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return value


def validate_checkpoint(checkpoint,binding,weights_sha):
    completed=checkpoint['completed_epochs']
    if checkpoint['binding']!=binding or checkpoint['weights_sha256']!=weights_sha or \
            not isinstance(completed,int) or completed not in range(10,101,10) or \
            [v['epoch'] for v in checkpoint['losses']]!=list(range(10,completed+1,10)) or \
            not np.isfinite(checkpoint['initial_loss']) or \
            any(not np.isfinite(v['weighted_REAL_L1']) for v in checkpoint['losses']):
        raise ValueError('resume checkpoint differs')
    return completed


def freeze():
    ROOT.mkdir(exist_ok=True);validate_previous()
    failed=read(DATA_ROOT/'e76/fit.json')
    if digest(DATA_ROOT/'e76/fit.json')!=digest(EVIDENCE/'e76_fit.json') or failed['dev_scoring_permitted']:
        raise ValueError('frozen failed E76 required')
    files=[Path(__file__),Path(__file__).with_name('e65_acquisition.py'),Path(__file__).with_name('e71_features.py'),
        DATA_ROOT/'e76/fit_contract.json',DATA_ROOT/'e76/fit.json',ML_ROOT/'src/pixelproof/training_weights.py']
    c={'state':'E77_REAL_only_CLIP_manifold_registered','inputs':{str(p):digest(p) for p in files},
       'parents':12141,'real_parents':7546,'ai_parents':4595,'real_views':22638,'conditions':['clean','assigned_transport','q75'],
       'architecture':[1536,256,64,256,1536],'activation':'ReLU after first3 linear layers, final linear output.',
       'normalization':'REAL-only source/parent-weighted StandardScaler. No AI in scaler or AE loss.',
       'loss':'Per-feature L1 reconstruction, source/parent weights fixed globally to mean1 over REAL views; '
              'batch mean without within-batch weight renormalization. All3 views per REAL parent.',
       'optimizer':'Adam lr2e-4,betas(.9,.999),eps1e-8,weight_decay0,foreachFalse. CPUseed77 initialization, '
                   'MPSfloat32 if available else CPU. Fixed100 epochs/batch256, final epoch only, no early stop.',
       'order':'NumPy default_rng(SeedSequence([77,zero_based_epoch])).permutation(22638); every view once per epoch.',
       'epochs':100,'batch_size':256,'learning_rate':2e-4,'max_seconds':3600,'cpu_threads':2,
       'resume':'Atomic model+optimizer checkpoint every10 completed epochs, bound to contract and REAL weights; '
                'resume same deterministic epoch order. No altered seed/epochs or selective checkpoint choice.',
       'residual':'Freeze AE, compute abs(standardized_CLIP-AE(standardized_CLIP)) in CPUfloat32 batch256. '
                  'Unweighted all-TRAIN StandardScaler then PCA64(seed77,randomized power3), whiten.',
       'representation_guard':'Finite state/features; final full weighted REAL L1 < initial; serialized CPU residual '
                              'and coordinates replay exactly in the same batch layout. Not a detector-quality gate.',
       'downstream':'Only after complete representation: separately freeze one321-coefficient correction combining '
                    'exact E74 original64/CLIP64/interaction128 plus residual64. All E76 guards retained.',
       'paper':'https://arxiv.org/html/2603.00717v2','reproduction':False,
       'adaptation':'Architecture/epochs/scaling/PCA/teacher-constrained downstream head are our fixed adaptation.',
       'torch_version':torch.__version__,'dev_or_final_rows_read':0,'classifier_scores':0,'downloads':0,
       'candidate_fit_allowed':False,'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e77_representation_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'real_views':22638,'epochs':100}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e77_representation_contract.json')['contract_sha256'] or str(torch.__version__)!=c['torch_version']:
        raise ValueError('E77 representation contract/runtime changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E77 representation input changed: '+p)
    return c


def load_training():
    previous=read(DATA_ROOT/'e71/fit_contract.json')
    rows=combine_rows(read(previous['inputs']['manifest']['path'])['rows'],read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    with np.load(previous['inputs']['clip']['path'],allow_pickle=False) as a:old=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:new=a['clip']
    clip=np.concatenate([old,new]).reshape(-1,1536)
    if clip.shape!=(36423,1536):raise ValueError('complete expanded CLIP TRAIN required')
    return rows,clip


def train():
    c=validate()
    if REPORT.exists():raise FileExistsError('E77 representation already complete')
    torch.set_num_threads(2);started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows,clip=load_training();mask,weights=real_weights(rows)
    if int(mask.sum())!=c['real_views']:raise ValueError('complete REAL-only view population required')
    with threadpool_limits(limits=2):a=real_scaler(clip,mask,weights)
    binding=digest(CONTRACT);weights_sha=array_sha(weights);x=normalized(clip[mask],a)
    torch.manual_seed(77);net=build_network();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net=net.to(device);optimizer=torch.optim.Adam(net.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    values=torch.from_numpy(x).to(device);weight_tensor=torch.from_numpy(weights.astype(np.float32)).to(device)
    def full_loss():
        net.eval();total=0.
        with torch.inference_mode():
            for start in range(0,len(x),c['batch_size']):
                b=values[start:start+c['batch_size']];w=weight_tensor[start:start+c['batch_size']]
                total+=float((torch.abs(b-net(b)).mean(dim=1)*w).sum().cpu())
        net.train();return total/len(x)
    initial=full_loss();losses=[];completed=0
    if CHECKPOINT.exists():
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        validate_checkpoint(checkpoint,binding,weights_sha)
        net.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        completed=checkpoint['completed_epochs'];losses=checkpoint['losses'];initial=checkpoint['initial_loss']
    print(json.dumps({'stage':'E77_REAL_AE_start','views':len(x),'completed_epochs':completed,'initial_L1':initial,'device':str(device)}),flush=True)
    with threadpool_limits(limits=2):
        for epoch in range(completed,c['epochs']):
            resource_check(deadline);net.train();order=epoch_order(len(x),epoch)
            for start in range(0,len(order),c['batch_size']):
                if start%(c['batch_size']*20)==0:resource_check(deadline)
                ids=torch.from_numpy(order[start:start+c['batch_size']]).to(device)
                batch=values[ids];weight=weight_tensor[ids]
                loss=(torch.abs(batch-net(batch)).mean(dim=1)*weight).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite AE loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if (epoch+1)%10==0:
                measured=full_loss();losses.append({'epoch':epoch+1,'weighted_REAL_L1':measured})
                print(json.dumps({'E77_epoch':epoch+1,'weighted_REAL_L1':measured,'seconds':round(time.monotonic()-started)}),flush=True)
                checkpoint={'binding':binding,'weights_sha256':weights_sha,'model':cpu_state(net.state_dict()),
                    'optimizer':cpu_state(optimizer.state_dict()),'completed_epochs':epoch+1,'losses':losses,'initial_loss':initial}
                temp=CHECKPOINT.with_suffix('.pt.part');torch.save(checkpoint,temp);temp.replace(CHECKPOINT)
        final=full_loss()
        if not np.isfinite(final) or not final<initial:raise ValueError('AE reconstruction loss did not improve')
        a.update(export_network(net));net=net.to('cpu');del values,weight_tensor,optimizer
        if device.type=='mps':torch.mps.empty_cache()
        resource_check(deadline);raw=residuals(clip,a)
        scale=StandardScaler().fit(raw.astype(np.float64));standardized=scale.transform(raw.astype(np.float64))
        pca=PCA(n_components=64,svd_solver='randomized',random_state=77,iterated_power=3).fit(standardized)
        a.update(residual_center=scale.mean_,residual_scale=scale.scale_,residual_mean=pca.mean_,
                 residual_components=pca.components_,residual_scales=np.sqrt(pca.explained_variance_))
        if np.any(a['residual_scales']<=0) or any(not np.isfinite(v).all() for v in a.values()):raise ValueError('invalid trained manifold')
        z=coordinates(clip,a)
        if ARTIFACT.exists():
            with np.load(ARTIFACT,allow_pickle=False) as old:
                if str(old['binding'])!=binding or any(not np.array_equal(old[k],v) for k,v in a.items()):raise ValueError('existing manifold differs')
        else:save_npz(ARTIFACT,**a,binding=np.array(binding))
        with np.load(ARTIFACT,allow_pickle=False) as saved:
            if not np.array_equal(raw,residuals(clip,saved)) or not np.array_equal(z,coordinates(clip,saved)):
                raise ValueError('serialized residual/coordinate replay differs')
        parents=np.array([r['parent_id'] for r in rows]);features=z.reshape(len(rows),3,64)
        if FEATURES.exists():
            with np.load(FEATURES,allow_pickle=False) as old:
                if str(old['binding'])!=binding or set(old['roles'])!={'TRAIN'} or not np.array_equal(old['parents'],parents) or not np.array_equal(old['features'],features):
                    raise ValueError('existing feature archive differs')
        else:save_npz(FEATURES,features=features,parents=parents,binding=np.array(binding),roles=np.array(['TRAIN']*len(rows)))
    result={'state':'E77_REAL_manifold_representation_complete','contract_sha256':binding,'initial_weighted_REAL_L1':initial,
        'final_weighted_REAL_L1':final,'loss_trace':losses,'epochs':100,'real_views':len(x),'ai_views_in_AE_loss':0,
        'feature_shape':list(features.shape),'feature_sha256':digest(FEATURES),'artifact_sha256':digest(ARTIFACT),
        'weights_sha256':weights_sha,'PCA_explained_variance':float(pca.explained_variance_ratio_.sum()),
        'serialized_residual_exact':True,'serialized_coordinates_exact':True,'seconds':time.monotonic()-started,
        'dev_or_final_rows_read':0,'classifier_scores':0,'downloads':0,'detector_quality_claim':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e77_representation.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'representation_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'train':train}[parser.parse_args().stage](),indent=2))
