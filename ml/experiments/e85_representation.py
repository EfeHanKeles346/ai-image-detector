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
    def denied(*a,**kw):raise RuntimeError('E85 representation is offline')
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
from experiments.e81_fit import reset_map
from experiments.e84b_features import validate as validate_previous
from experiments.e85_data import load as load_four,CONDITIONS
from experiments.e82_representation import build_network,epoch_order,normalized,export_network,latent_raw,coordinates,validate_checkpoint
from experiments.e77_representation import cpu_state
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e71_features import save_npz,array_sha
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
ROOT=DATA_ROOT/'e85';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'representation_contract.json';REPORT=ROOT/'representation.json'
ARTIFACT=ROOT/'supervised_map.npz';FEATURES=ROOT/'supervised_features.npz';CHECKPOINT=ROOT/'network_checkpoint.pt'


def loss_weights(rows,baseline):
    labels=np.repeat([r['label'] for r in rows],4)
    if baseline.shape!=labels.shape or not np.isfinite(baseline).all():raise ValueError('aligned finite baseline required')
    weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],4),np.repeat([r['parent_id'] for r in rows],4))
    weights[(labels==0)&(baseline>=input_model.AI_CUT)]*=2
    for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
    return labels,weights*len(labels)


def freeze():
    ROOT.mkdir(exist_ok=True);validate_previous()
    features=read(DATA_ROOT/'e84b/features.json');receipt=read(EVIDENCE/'e84b_features.json')
    if digest(DATA_ROOT/'e84b/features.json')!=receipt['report_sha256'] or features['parents']!=12141 or \
            features['state']!='E84B_social_TRAIN_features_complete' or \
            features['shapes']!={'dino':[12141,3072],'clip':[12141,1536],'dear':[12141,1640]} or \
            features['new_view_classifier_scores'] or features['dev_final_image_reads'] or \
            digest(DATA_ROOT/'e84b/social_features.npz')!=features['feature_sha256']:
        raise ValueError('complete verified E84B transport features required')
    if digest(DATA_ROOT/'e83/correction.npz')!=read(DATA_ROOT/'e83/fit.json')['candidate_sha256']:
        raise ValueError('frozen E83 candidate changed')
    paths=[Path(__file__),Path(__file__).with_name('e85_data.py'),Path(__file__).with_name('e82_representation.py'),
        DATA_ROOT/'e82/representation_contract.json',DATA_ROOT/'e84b/features_contract.json',
        DATA_ROOT/'e84b/features.json',DATA_ROOT/'e84b/social_features.npz',DATA_ROOT/'e83/correction.npz',
        DATA_ROOT/'e83/fit.json',DATA_ROOT/'e80/correction.npz',Path(input_model.__file__),
        ML_ROOT/'src/pixelproof/training_weights.py']
    old=read(DATA_ROOT/'e82/representation_contract.json')
    c={k:old[k] for k in ['parents','input_width','network','optimizer','order','epochs','batch_size','learning_rate',
        'weight_decay','max_seconds','cpu_threads','torch_version','numpy_version','sklearn_version','resume']}
    c.update(state='E85_four_condition_supervised_TRAIN_representation_registered',inputs={str(p):digest(p) for p in paths},
        views=48564,ai_views=18380,real_views=30184,conditions=CONDITIONS,
        input='Exact E80 multimodal385 coordinates; old maps stay frozen. Add verified E84B social view to every '
              'TRAIN parent. Refit only input/latent StandardScalers on all four TRAIN conditions.',
        loss='Same E82 binary BCE with class/source/parent balance, E43 false-AI REAL2x then class mass.5. '
             'Four views per parent, global mean weight1, no within-batch renormalization.',
        output='Same64 latent ReLU coordinates, CPUfloat64 inference on saved float32 network. No PCA/logit scalar. '
               '12141x4x64, parent/role/condition/contract bindings. Seed82 initialization and epoch ordering retained; '
               'no E82/E83 warm start. Same fixed100 epochs and final-only export.',
        checks='Reproduce E80 original-three TRAIN population metrics exactly, then verify original-three E83 '
               'scores under the four-view input-map batch layout: max error<=1e-6, zero changes at both cuts. '
               'Finite state and improved final BCE, exact saved coordinates, fixed-input batch8 max error<=1e-10. '
               'These are engineering/TRAIN checks; full constrained-head runtime replay remains mandatory later.',
        next='Only complete features permit separately registered E86 zero450-coefficient fit, same E81 objective. '
             'All18380 AI-view E43 logits protected, correct-REAL constraints, original3 and new social TRAIN '
             'conditions each with all population/sensor and10 metric gates, then full runtime batch8 replay.',
        limits='E83 consumed DEV transport gap motivated this data-view extension; E66 labels/features never used '
               'in learning. Existing RR lineage and MIDD/DEAR restrictions persist. No guarantee of unseen AI retention.',
        dev_final_rows_read=0,downloads=0,promotion_allowed=False,detector_quality_claim=False)
    write_once(CONTRACT,c);write_once(EVIDENCE/'e85_representation_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':c['views']}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e85_representation_contract.json')['contract_sha256'] or \
            str(torch.__version__)!=c['torch_version'] or np.__version__!=c['numpy_version'] or sklearn.__version__!=c['sklearn_version']:
        raise ValueError('E85 contract/runtime changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E85 input changed: '+p)
    return c


def check_old_scores(expected,actual):
    if expected.shape!=actual.shape or not len(expected) or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('complete finite old-three score pairing required')
    error=float(np.max(np.abs(expected-actual)))
    changes={str(cut):int(np.sum((expected>=cut)!=(actual>=cut))) for cut in [input_model.REAL_CUT,input_model.AI_CUT]}
    if error>1e-6 or any(changes.values()):raise ValueError('old-three input layout changes reference decisions')
    return {'max_score_error':error,'decision_changes_by_cut':changes,'passed':True}


def load_training():
    rows,features,head=load_four()
    old={k:v[:,:3].reshape(-1,v.shape[-1]) for k,v in features.items()}
    flat={k:v.reshape(-1,v.shape[-1]) for k,v in features.items()}
    with np.load(DATA_ROOT/'e80/correction.npz',allow_pickle=False) as a,threadpool_limits(limits=2):
        old_full=input_model.project(head,old['dino'],old['clip'],old['dear'],reset_map(a))
        old_logits=head.decision_function(old['dino'])
        replay=expit(old_logits+old_full@a['weights']).reshape(-1,3)
        if real_slice_gates(rows,replay,11630)!=read(DATA_ROOT/'e80/fit.json')['real_population_gates']:
            raise ValueError('frozen E80 original-three replay differs')
        full=input_model.project(head,flat['dino'],flat['clip'],flat['dear'],reset_map(a))
        baseline=head.predict_proba(flat['dino'])[:,1]
        old_x=old_full[:,:-1];layout_x=full.reshape(len(rows),4,386)[:,:3,:-1].reshape(-1,385)
        with np.load(DATA_ROOT/'e83/correction.npz',allow_pickle=False) as previous:
            score=lambda x:expit(old_logits+np.column_stack([x,coordinates(x,previous),np.ones(len(x))])@previous['weights'])
            input_replay=check_old_scores(score(old_x),score(layout_x))
        x=full[:,:-1]
    if x.shape!=(48564,385):raise ValueError('complete four-condition supervised TRAIN map required')
    return rows,x,baseline,input_replay


def train():
    c=validate()
    if REPORT.exists():raise FileExistsError('E85 representation already complete')
    torch.set_num_threads(2);started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows,x,baseline,input_replay=load_training();labels,weights=loss_weights(rows,baseline)
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
    print(json.dumps({'stage':'E85_supervised_start','views':len(x),'completed_epochs':completed,'initial_BCE':initial,'device':str(device)}),flush=True)
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
                print(json.dumps({'E85_epoch':epoch+1,'weighted_BCE':measured,'seconds':round(time.monotonic()-started)}),flush=True)
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
        parents=np.array([r['parent_id'] for r in rows]);features=z.reshape(len(rows),4,64)
        if FEATURES.exists():
            with np.load(FEATURES,allow_pickle=False) as old:
                if str(old['binding'])!=binding or set(old['roles'])!={'TRAIN'} or list(old['conditions'])!=CONDITIONS or not np.array_equal(old['parents'],parents) or not np.array_equal(old['features'],features):
                    raise ValueError('existing supervised features differ')
        else:save_npz(FEATURES,features=features,parents=parents,roles=np.array(['TRAIN']*len(rows)),binding=np.array(binding),conditions=np.array(CONDITIONS))
    result={'state':'E85_supervised_TRAIN_representation_complete','contract_sha256':binding,
        'initial_weighted_BCE':initial,'final_weighted_BCE':final,'loss_trace':losses,'epochs':100,
        'TRAIN_views':len(x),'real_views':int(np.sum(labels==0)),'ai_views':int(np.sum(labels==1)),
        'feature_shape':list(features.shape),'feature_sha256':digest(FEATURES),'artifact_sha256':digest(ARTIFACT),
        'weights_sha256':weights_sha,'serialized_latent_exact':True,'serialized_coordinates_exact':True,
        'fixed_input_batch8_max_error':batch_error,'seconds':time.monotonic()-started,
        'input_map_old_three_replay':input_replay,'conditions':CONDITIONS,
        'dev_final_rows_read':0,'new_image_inference':0,'downloads':0,'detector_quality_claim':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e85_representation.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'representation_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'train':train}[parser.parse_args().stage](),indent=2))
