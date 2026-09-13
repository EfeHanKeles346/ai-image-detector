"""Fixed TRAIN-only visual selection; no changes to labels, fits or thresholds."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from experiments.e77_fit import validate,ROOT,CANDIDATE
from experiments import e77_model as model
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
def run():
    validate();r=read(ROOT/'fit.json');assert digest(CANDIDATE)==r['candidate_sha256']
    p={k:Path(v['path']) for k,v in read(DATA_ROOT/'e71/fit_contract.json')['inputs'].items()}
    old=read(p['manifest'])['rows'];rows=combine_rows(old,read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
     original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    torch.set_num_threads(2);head=joblib.load(p['reference'])['head']
    with threadpool_limits(limits=2),np.load(CANDIDATE,allow_pickle=False) as a:scores=model.predict(head,original,clip,a).reshape(-1,3)
    assert real_slice_gates(rows,scores,len(old))==r['real_population_gates']
    chosen=[]
    for wrong in [True,False]:
     eligible=[(i,row) for i,row in enumerate(rows) if row['label']==0 and row['source']=='rr:rrdataset_real_pool' and bool(scores[i,0]>=model.AI_CUT)==wrong]
     ranked=sorted(eligible,key=lambda pair:hashlib.sha256(('E77_visual|'+pair[1]['parent_id']).encode()).hexdigest())
     for i,row in ranked[:3]:
      assert digest(row['path'])==row['sha256']
      chosen.append({k:row[k] for k in ['parent_id','path','sha256','source','label','role']}|{'E77_clean_score':float(scores[i,0]),'false_ai':wrong})
    result={'state':'E77_RR_TRAIN_visual_selection_locked','candidate_sha256':digest(CANDIDATE),
            'selection':'3 false-AI and3 correct REAL, SHA256 E77_visual|parent_id within bins','rows':chosen,
            'dev_or_final_rows':0,'labels_changed':False,'training_exclusions':0,'new_fit':False,'images_viewed_at_selection':0}
    write_once(ROOT/'visual_selection.json',result);write_once(ML_ROOT.parent/'evidence/e77_visual_selection.json',result)
    print(json.dumps(chosen,indent=2))


if __name__=='__main__':run()
