"""One supervised nonlinear TRAIN feature learner; independent constrained head comes later."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E82 representation is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
import sklearn
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiments import e80_model as input_model
from experiments.e81_fit import validate as validate_previous,reset_map
from experiments.e77_representation import cpu_state
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e71_features import save_npz,array_sha
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
ROOT=DATA_ROOT/'e82';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'representation_contract.json';REPORT=ROOT/'representation.json'
ARTIFACT=ROOT/'supervised_map.npz';FEATURES=ROOT/'supervised_features.npz';CHECKPOINT=ROOT/'network_checkpoint.pt'


def build_network(width=385,hidden=256,latent=64):
    return torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.ReLU(),
        torch.nn.Linear(hidden,latent),torch.nn.ReLU(),torch.nn.Linear(latent,1))


def epoch_order(n,epoch):
    if n<=0 or not 0<=epoch<100:raise ValueError('invalid fixed epoch population')
    return np.random.default_rng(np.random.SeedSequence([82,epoch])).permutation(n)


def loss_weights(rows,baseline):
    labels=np.repeat([r['label'] for r in rows],3)
    if baseline.shape!=labels.shape or not np.isfinite(baseline).all():raise ValueError('aligned finite baseline required')
    weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],3),np.repeat([r['parent_id'] for r in rows],3))
    weights[(labels==0)&(baseline>=input_model.AI_CUT)]*=2
    for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
    return labels,weights*len(labels)


def normalized(x,a):
    if x.ndim!=2 or x.shape[1]!=len(a['input_center']) or not np.isfinite(x).all() or \
            not np.isfinite(a['input_center']).all() or not np.isfinite(a['input_scale']).all() or np.any(a['input_scale']<=0):
        raise ValueError('finite aligned supervised inputs required')
    return (x.astype(np.float64)-a['input_center'])/a['input_scale']


def export_network(net):
    return {'nn_'+k.replace('.','_'):v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}


def latent_raw(x,a,batch_size=512):
    if batch_size<=0 or not len(x):raise ValueError('positive inference batch and population required')
    z=normalized(x,a);parts=[]
    width=z.shape[1];hidden=np.shape(a['nn_0_bias'])[0];latent=np.shape(a['nn_2_bias'])[0]
    shapes={'nn_0_weight':(hidden,width),'nn_0_bias':(hidden,),
            'nn_2_weight':(latent,hidden),'nn_2_bias':(latent,),
            'nn_4_weight':(1,latent),'nn_4_bias':(1,)}
    for key,shape in shapes.items():
        if np.shape(a[key])!=shape or a[key].dtype!=np.float32 or not np.isfinite(a[key]).all():
            raise ValueError('invalid supervised network state')
    w0=a['nn_0_weight'].astype(np.float64).T;w2=a['nn_2_weight'].astype(np.float64).T
    for start in range(0,len(z),batch_size):
        h=np.maximum(z[start:start+batch_size]@w0+a['nn_0_bias'],0)
        parts.append(np.maximum(h@w2+a['nn_2_bias'],0))
    result=np.concatenate(parts)
    if not np.isfinite(result).all():raise ValueError('nonfinite supervised latent')
    return result


def coordinates(x,a,batch_size=512):
    raw=latent_raw(x,a,batch_size)
    if np.shape(a['latent_center'])!=(raw.shape[1],) or np.shape(a['latent_scale'])!=(raw.shape[1],) or \
            not np.isfinite(a['latent_center']).all() or not np.isfinite(a['latent_scale']).all() or np.any(a['latent_scale']<=0):
        raise ValueError('invalid latent normalization')
    return (raw-a['latent_center'])/a['latent_scale']


def validate_checkpoint(checkpoint,binding,weights_sha):
    completed=checkpoint['completed_epochs']
    if checkpoint['binding']!=binding or checkpoint['weights_sha256']!=weights_sha or \
            not isinstance(completed,int) or completed not in range(10,101,10) or \
            [v['epoch'] for v in checkpoint['losses']]!=list(range(10,completed+1,10)) or \
            not np.isfinite(checkpoint['initial_loss']) or any(not np.isfinite(v['weighted_BCE']) for v in checkpoint['losses']):
        raise ValueError('resume checkpoint differs')
    return completed


def freeze():
    ROOT.mkdir(exist_ok=True);validate_previous();failed=read(DATA_ROOT/'e81/fit.json')
    if failed['dev_scoring_permitted'] or not failed['solver']['success'] or \
            digest(DATA_ROOT/'e81/fit.json')!=digest(EVIDENCE/'e81_fit.json') or \
            digest(DATA_ROOT/'e81/correction.npz')!=failed['candidate_sha256']:
        raise ValueError('frozen rejected E81 required')
    paths=[Path(__file__),DATA_ROOT/'e81/fit_contract.json',DATA_ROOT/'e81/fit.json',DATA_ROOT/'e81/correction.npz',
        DATA_ROOT/'e80/correction.npz',Path(input_model.__file__),Path(__file__).with_name('e81_fit.py'),
        Path(__file__).with_name('e77_representation.py'),Path(__file__).with_name('e71_features.py'),
        ML_ROOT/'src/pixelproof/training_weights.py']
    c={'state':'E82_supervised_nonlinear_TRAIN_representation_registered','inputs':{str(p):digest(p) for p in paths},
       'parents':12141,'views':36423,'ai_views':13785,'real_views':22638,'input_width':385,
       'input':'Exact E80 map minus intercept; all TRAIN StandardScaler without labels. No old-map refit.',
       'network':'385->256->64->1, ReLU after first two layers; no dropout or batch normalization.',
       'loss':'Binary TRAIN BCE with existing class/source/parent balance; E43 false-AI REAL2x then class mass.5. '
              'Global mean weight1, no within-batch renormalization. Logits are TRAIN learning signals, not an accepted detector.',
       'optimizer':'AdamW2e-4,betas(.9,.999),eps1e-8,weight_decay1e-4,foreachFalse; CPU seed82 init, '
                   'MPSfloat32 if available else CPU. Fixed100 epochs/batch256; final epoch only.',
       'order':'Each epoch NumPy SeedSequence[82,epoch], all views once, no global RNG dependence.',
       'epochs':100,'batch_size':256,'learning_rate':2e-4,'weight_decay':1e-4,'max_seconds':3600,'cpu_threads':2,
       'torch_version':str(torch.__version__),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
       'resume':'Atomic model+optimizer checkpoint every10 complete epochs, contract/weight/trace binding. Same recipe only.',
       'output':'Freeze64 latent ReLU features; no redundant final-logit scalar because it lies in their affine span. '
                'Evaluate float32 weights in CPUfloat64 with float64 input normalization; all-TRAIN latent StandardScaler. '
                'No PCA or rank selection;12141x3x64 parent/role/contract/input-map bound features.',
       'checks':'Input-map replay must reproduce frozen E80 REAL population metrics before learning. '
                'Finite state and improved final weighted TRAIN BCE; exact saved latent/coordinate replay; '
                'all-view batch8 latent-coordinate max error<=1e-10 using the same fixed input coordinates. '
                'This does not certify the older AE stage or an external detector.',
       'next':'Only complete verified representation permits a separate E83 zero450-coefficient fit in E80 plus latent64. '
              'Same E81 worst-REAL objective, all-AI/correct-REAL constraints and all E80 population/metric/runtime gates.',
       'dev_final_rows_read':0,'downloads':0,'promotion_allowed':False,'detector_quality_claim':False,
       'limits':'Supervised TRAIN features may overfit; no OOF/fresh or independent result. E81/RR/MIDD/DEAR lineage limits persist. '
                'No seed/epoch/loss/rank/weight/cut sweep or new data admission.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e82_representation_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':c['views']}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e82_representation_contract.json')['contract_sha256'] or \
            str(torch.__version__)!=c['torch_version'] or np.__version__!=c['numpy_version'] or sklearn.__version__!=c['sklearn_version']:
        raise ValueError('E82 representation contract/runtime changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E82 input changed: '+p)
    return c


def load_training():
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old=read(p['manifest'])['rows'];new=read(DATA_ROOT/'e72/training_manifest.json')['rows'];rows=combine_rows(old,new)
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    with np.load(DATA_ROOT/'e79/dear_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e79/features_contract.json') or list(a['parents'])!=[r['parent_id'] for r in rows] or set(a['roles'])!={'TRAIN'}:
            raise ValueError('complete DEAR TRAIN pairing required')
        dear=a['features'].reshape(-1,1640)
    head=joblib.load(p['reference'])['head']
    with np.load(DATA_ROOT/'e80/correction.npz',allow_pickle=False) as a,threadpool_limits(limits=2):
        full=input_model.project(head,original,clip,dear,reset_map(a));baseline=head.predict_proba(original)[:,1]
        replay=expit(head.decision_function(original)+full@a['weights']).reshape(-1,3)
        if real_slice_gates(rows,replay,len(old))!=read(DATA_ROOT/'e80/fit.json')['real_population_gates']:
            raise ValueError('frozen E80 input-map replay differs')
        x=full[:,:-1]
    if x.shape!=(36423,385):raise ValueError('complete supervised TRAIN input map required')
    return rows,x,baseline


def train():
    c=validate()
    if REPORT.exists():raise FileExistsError('E82 representation already complete')
    torch.set_num_threads(2);started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows,x,baseline=load_training();labels,weights=loss_weights(rows,baseline)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('complete original AI population required')
    with threadpool_limits(limits=2):scaler=StandardScaler().fit(x)
    a={'input_center':scaler.mean_,'input_scale':scaler.scale_};binding=digest(CONTRACT);weights_sha=array_sha(weights)
    values_np=normalized(x,a).astype(np.float32);torch.manual_seed(82);net=build_network()
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');net=net.to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=c['weight_decay'],foreach=False)
    values=torch.from_numpy(values_np).to(device);targets=torch.from_numpy(labels.astype(np.float32)).to(device)
    weight_tensor=torch.from_numpy(weights.astype(np.float32)).to(device)
    def full_loss():
        net.eval();total=0.
        with torch.inference_mode():
            for start in range(0,len(x),c['batch_size']):
                end=start+c['batch_size'];loss=torch.nn.functional.binary_cross_entropy_with_logits(net(values[start:end]).flatten(),targets[start:end],reduction='none')
                total+=float((loss*weight_tensor[start:end]).sum().cpu())
        net.train();return total/len(x)
    initial=full_loss();losses=[];completed=0
    if CHECKPOINT.exists():
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True);validate_checkpoint(checkpoint,binding,weights_sha)
        net.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        completed=checkpoint['completed_epochs'];losses=checkpoint['losses'];initial=checkpoint['initial_loss']
    print(json.dumps({'stage':'E82_supervised_start','views':len(x),'completed_epochs':completed,'initial_BCE':initial,'device':str(device)}),flush=True)
    with threadpool_limits(limits=2):
        for epoch in range(completed,c['epochs']):
            resource_check(deadline);net.train();order=epoch_order(len(x),epoch)
            for start in range(0,len(order),c['batch_size']):
                if start%(c['batch_size']*20)==0:resource_check(deadline)
                ids=torch.from_numpy(order[start:start+c['batch_size']]).to(device)
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(values[ids]).flatten(),targets[ids],reduction='none')*weight_tensor[ids]).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if (epoch+1)%10==0:
                measured=full_loss();losses.append({'epoch':epoch+1,'weighted_BCE':measured})
                print(json.dumps({'E82_epoch':epoch+1,'weighted_BCE':measured,'seconds':round(time.monotonic()-started)}),flush=True)
                checkpoint={'binding':binding,'weights_sha256':weights_sha,'model':cpu_state(net.state_dict()),
                    'optimizer':cpu_state(optimizer.state_dict()),'completed_epochs':epoch+1,'losses':losses,'initial_loss':initial}
                temporary=CHECKPOINT.with_suffix('.pt.part');torch.save(checkpoint,temporary);temporary.replace(CHECKPOINT)
        final=full_loss()
        if not np.isfinite(final) or not final<initial:raise ValueError('supervised loss did not improve')
        a.update(export_network(net));net=net.to('cpu');del values,targets,weight_tensor,optimizer
        if device.type=='mps':torch.mps.empty_cache()
        resource_check(deadline);raw=latent_raw(x,a);latent_scaler=StandardScaler().fit(raw)
        a.update(latent_center=latent_scaler.mean_,latent_scale=latent_scaler.scale_);z=coordinates(x,a)
        if ARTIFACT.exists():
            with np.load(ARTIFACT,allow_pickle=False) as old:
                if str(old['binding'])!=binding or any(not np.array_equal(old[k],v) for k,v in a.items()):raise ValueError('existing supervised map differs')
        else:save_npz(ARTIFACT,**a,binding=np.array(binding),input_map_sha256=np.array(digest(DATA_ROOT/'e80/correction.npz')))
        with np.load(ARTIFACT,allow_pickle=False) as saved:
            if str(saved['input_map_sha256'])!=digest(DATA_ROOT/'e80/correction.npz') or \
                    not np.array_equal(raw,latent_raw(x,saved)) or not np.array_equal(z,coordinates(x,saved)):
                raise ValueError('saved supervised feature replay differs')
            replay=coordinates(x,saved,batch_size=8);batch_error=float(np.max(np.abs(z-replay)))
            if batch_error>1e-10:raise ValueError('supervised latent runtime batch error exceeds1e-10')
        parents=np.array([r['parent_id'] for r in rows]);features=z.reshape(len(rows),3,64)
        if FEATURES.exists():
            with np.load(FEATURES,allow_pickle=False) as old:
                if str(old['binding'])!=binding or set(old['roles'])!={'TRAIN'} or not np.array_equal(old['parents'],parents) or not np.array_equal(old['features'],features):
                    raise ValueError('existing supervised features differ')
        else:save_npz(FEATURES,features=features,parents=parents,roles=np.array(['TRAIN']*len(rows)),binding=np.array(binding))
    result={'state':'E82_supervised_TRAIN_representation_complete','contract_sha256':binding,
        'initial_weighted_BCE':initial,'final_weighted_BCE':final,'loss_trace':losses,'epochs':100,
        'TRAIN_views':len(x),'real_views':int(np.sum(labels==0)),'ai_views':int(np.sum(labels==1)),
        'feature_shape':list(features.shape),'feature_sha256':digest(FEATURES),'artifact_sha256':digest(ARTIFACT),
        'weights_sha256':weights_sha,'serialized_latent_exact':True,'serialized_coordinates_exact':True,
        'fixed_input_batch8_max_error':batch_error,'seconds':time.monotonic()-started,
        'dev_final_rows_read':0,'new_image_inference':0,'downloads':0,'detector_quality_claim':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e82_representation.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'representation_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'train':train}[parser.parse_args().stage](),indent=2))
