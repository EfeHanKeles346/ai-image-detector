"""One fixed-recipe supervised representation after complete SID TRAIN expansion."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('E91 representation offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
import sklearn
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiments import e80_model as input_model
from experiments.e81_fit import reset_map
from experiments.e82_representation import build_network, epoch_order, normalized, export_network, latent_raw, coordinates, validate_checkpoint
from experiments.e77_representation import cpu_state
from experiments.e85_representation import loss_weights, check_old_scores
from experiments.e86_fit import validate as validate_previous
from experiments.e86_gates import full_training_gates as previous_gates
from experiments.e89_features import validate as validate_features
from experiments.e91_data import load as load_four, CONDITIONS
from experiments.e71_features import save_npz, array_sha
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
ROOT=DATA_ROOT/'e91'; EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'representation_contract.json'; REPORT=ROOT/'representation.json'
ARTIFACT=ROOT/'supervised_map.npz'; FEATURES=ROOT/'supervised_features.npz'; CHECKPOINT=ROOT/'network_checkpoint.pt'


def write_report(result):
    write_once(REPORT,result); write_once(EVIDENCE/'e91_representation.json',result)


def freeze():
    ROOT.mkdir(exist_ok=True); validate_previous(); validate_features()
    features=read(DATA_ROOT/'e89/features.json'); receipt=read(EVIDENCE/'e89_features.json')
    rejected=read(DATA_ROOT/'e86/dev_report.json')
    if digest(DATA_ROOT/'e89/features.json')!=receipt['report_sha256'] or \
            features['state']!='E89_SID_four_condition_TRAIN_features_complete' or features['parents']!=128 or \
            features['views']!=512 or features['conditions']!=CONDITIONS or \
            features['shapes']!={'dino':[128,4,3072],'clip':[128,4,1536],'dear':[128,4,1640]} or \
            not features['parity']['passed'] or features['first_batch_repeat_max_error']>1e-5 or \
            features['new_image_classifier_scores'] or features['dev_final_image_reads'] or \
            digest(DATA_ROOT/'e89/sid_features.npz')!=features['feature_sha256'] or \
            digest(DATA_ROOT/'e86/dev_report.json')!=digest(EVIDENCE/'e86_development.json') or \
            rejected['passes_limited_dev_screen'] or \
            digest(DATA_ROOT/'e86/correction.npz')!=read(DATA_ROOT/'e86/fit.json')['candidate_sha256']:
        raise ValueError('complete verified SID features and frozen rejected E86 required')
    paths=[Path(__file__),Path(__file__).with_name('e91_data.py'),Path(__file__).with_name('e85_representation.py'),
        Path(__file__).with_name('e82_representation.py'),Path(__file__).with_name('e86_gates.py'),
        DATA_ROOT/'e85/representation_contract.json',DATA_ROOT/'e89/features_contract.json',
        DATA_ROOT/'e89/features.json',DATA_ROOT/'e89/sid_features.npz',DATA_ROOT/'e88/training_manifest.json',
        DATA_ROOT/'e86/correction.npz',DATA_ROOT/'e86/fit.json',DATA_ROOT/'e86/dev_report.json',
        DATA_ROOT/'e80/correction.npz',Path(input_model.__file__),ML_ROOT/'src/pixelproof/training_weights.py']
    old=read(DATA_ROOT/'e85/representation_contract.json')
    c={k:old[k] for k in ['input_width','network','optimizer','order','epochs','batch_size','learning_rate',
        'weight_decay','max_seconds','cpu_threads','torch_version','numpy_version','sklearn_version','resume']}
    c.update(state='E91_SID_expanded_supervised_TRAIN_representation_registered',inputs={str(p):digest(p) for p in paths},
        parents=12269,previous_parents=12141,new_sid_parents=128,views=49076,ai_views=18380,real_views=30696,conditions=CONDITIONS,
        input='Exact E80 multimodal385 coordinates and frozen old maps. Append all128 score-blind E88/E89 SID TRAIN '
              'parents with all4 conditions; preserve every prior feature body/order. Refit only input/latent TRAIN scalers.',
        loss=old['loss'],
        output='Same E85 network/seed82/order/optimizer/batch256/100 epochs from scratch, no warm start. '
               'CPUfloat64 inference on saved float32 weights.64 standardized latent coordinates,12269x4x64.',
        checks='Exact replay of all80 previous E86 TRAIN numeric metric reports. Appended input-map batch layout '
               'must preserve all previous E86 scores<=1e-6 with zero changes at both cuts. Saved coordinates exact, '
               'fixed-input latent batch8 error<=1e-10, finite improved final BCE. These are TRAIN/engineering checks.',
        next='Only complete representation permits separately registered E92 zero450-weight fit, same E81 worst-REAL '
             'objective. All18380 AI logits protected. All4 conditions: preserve legacy/MIDD population guards, add '
             'SID pooled<=10% and worst camera<=20%, combined REAL<=10%, and all10 numeric gates on legacy, previous '
             'and expanded TRAIN (120 checks), then full49076 runtime batch8 replay before any consumed DEV scoring.',
        limits='SID filename scenes are not proven independent, whole publisher research TRAIN only. '
               'No SID/DEV score-based selection. Existing RR lineage and MIDD/DEAR restrictions persist. '
               'No new consistency objective or hyperparameter sweep; no external AI-retention guarantee.',
        dev_final_rows_read=0,downloads=0,promotion_allowed=False,detector_quality_claim=False)
    write_once(CONTRACT,c); write_once(EVIDENCE/'e91_representation_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':c['views']}


def validate():
    validate_previous(); validate_features(); c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e91_representation_contract.json')['contract_sha256'] or \
            str(torch.__version__)!=c['torch_version'] or np.__version__!=c['numpy_version'] or sklearn.__version__!=c['sklearn_version']:
        raise ValueError('E91 contract/runtime changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s: raise ValueError('E91 input changed: '+p)
    return c


def load_training():
    rows,features,head=load_four(); previous_count=12141
    old={k:v[:previous_count].reshape(-1,v.shape[-1]) for k,v in features.items()}
    flat={k:v.reshape(-1,v.shape[-1]) for k,v in features.items()}
    with np.load(DATA_ROOT/'e80/correction.npz',allow_pickle=False) as base, \
            np.load(DATA_ROOT/'e86/correction.npz',allow_pickle=False) as previous, threadpool_limits(limits=2):
        old_x=input_model.project(head,old['dino'],old['clip'],old['dear'],reset_map(base))[:,:-1]
        old_logits=head.decision_function(old['dino'])
        def previous_score(x):
            return expit(old_logits+np.column_stack([x,coordinates(x,previous),np.ones(len(x))])@previous['weights'])
        expected=previous_score(old_x)
        if previous_gates(rows[:previous_count],expected.reshape(-1,4),11630)!=read(DATA_ROOT/'e86/fit.json')['full_TRAIN_metric_gates']:
            raise ValueError('frozen E86 previous-four TRAIN replay differs')
        x=input_model.project(head,flat['dino'],flat['clip'],flat['dear'],reset_map(base))[:,:-1]
        input_replay=check_old_scores(expected,previous_score(x[:previous_count*4]))
        baseline=head.predict_proba(flat['dino'])[:,1]
    if x.shape!=(49076,385): raise ValueError('complete SID-expanded four-view TRAIN map required')
    return rows,x,baseline,input_replay


def train():
    c=validate()
    if REPORT.exists():raise FileExistsError('E91 representation already complete')
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
    print(json.dumps({'stage':'E91_supervised_start','views':len(x),'completed_epochs':completed,'initial_BCE':initial,'device':str(device)}),flush=True)
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
                print(json.dumps({'E91_epoch':epoch+1,'weighted_BCE':measured,'seconds':round(time.monotonic()-started)}),flush=True)
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
    result={'state':'E91_supervised_TRAIN_representation_complete','contract_sha256':binding,
        'initial_weighted_BCE':initial,'final_weighted_BCE':final,'loss_trace':losses,'epochs':100,
        'TRAIN_views':len(x),'real_views':int(np.sum(labels==0)),'ai_views':int(np.sum(labels==1)),
        'feature_shape':list(features.shape),'feature_sha256':digest(FEATURES),'artifact_sha256':digest(ARTIFACT),
        'weights_sha256':weights_sha,'serialized_latent_exact':True,'serialized_coordinates_exact':True,
        'fixed_input_batch8_max_error':batch_error,'seconds':time.monotonic()-started,
        'previous_four_view_input_replay':input_replay,'conditions':CONDITIONS,
        'dev_final_rows_read':0,'new_image_inference':0,'downloads':0,'detector_quality_claim':False}
    write_report(result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'representation_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'train':train}[parser.parse_args().stage](),indent=2))
