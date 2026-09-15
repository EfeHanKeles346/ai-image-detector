import copy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import e135_location_learning as e


def population():
    return [dict(index=i,parent_id=str(i),source=f'sensor{i//2}',scene_group=f'scene{i}',
        source_body_sha256=str(i),role='TRAIN_RESEARCH_PILOT',seed=119000+i,
        files={'original':{'path':f'auth{i}','sha256':str(i)}}) for i in range(16)]


def test_both_placements_keep_exact_parent_fold_and_roles():
    original=population();corner=copy.deepcopy(original);folds=[i//2 for i in range(16)]
    rows,expanded=e.expanded_rows(original,corner,folds)
    assert len(rows)==32 and len({r['parent_id'] for r in rows})==16
    for i in range(16):
        assert [r['parent_index'] for r in rows[2*i:2*i+2]]==[i,i]
        assert expanded[2*i:2*i+2]==[folds[i],folds[i]]
        assert [r['placement'] for r in rows[2*i:2*i+2]]==['original','corner']
    corner[7]['source']='other'
    with pytest.raises(ValueError):e.expanded_rows(original,corner,folds)
    corner=copy.deepcopy(original);corner[7]['files']['original']['sha256']='different'
    with pytest.raises(ValueError):e.expanded_rows(original,corner,folds)


def test_duplicate_authentic_bookkeeping_has_same_total_weight():
    tokens={};masks={};rows=[]
    for i in range(2):
        rows.append({'index':i,'parent_index':0})
        mask=np.zeros((512,512),dtype=bool);mask[128:256,128:256]=True;masks[i]=mask
        for condition in e.CONDITIONS:
            for vi,variant in enumerate(e.VARIANTS):
                value=np.zeros((32,32,384),dtype=np.float32);value[:,:,0]=vi;value[:,:,1]=i
                if variant=='authentic':value[:,:,1]=0
                tokens[f'{i:03d}_{variant}_{condition}']=value
    x,y,w=e.patch_learning.training_arrays(rows,tokens,masks)
    assert w[y==1].sum()==pytest.approx(.5)
    assert w[x[:,0]==0].sum()==pytest.approx(1/6)
    assert w[x[:,0]==1].sum()==pytest.approx(1/6)
    assert w[(x[:,0]==2)&(y==0)].sum()==pytest.approx(1/6)


def maps(auc,flag):
    return {'maps':{v:dict(flagged_area_fraction=flag,iou=.2,
        all_pixel_ranking={'auc':auc},interior_background_ranking={'auc':auc}) for v in e.VARIANTS}}


def test_matched_comparisons_keep_placement_and_parent_separate():
    results=[];refs={p:[] for p in e.PLACEMENTS}
    for place in e.PLACEMENTS:
        for condition in e.CONDITIONS:
            for i in range(2):
                results.append(dict(parent_index=i,placement=place,condition=condition,metrics=maps(.7,.1)))
                refs[place].append(dict(index=i,condition=condition,metrics=maps(.8 if place=='original' else .5,.2)))
    summary=e.summarize(results,refs)
    for condition in e.CONDITIONS:
        assert summary['original'][condition]['ai_composite']['mean_auc_change']==pytest.approx(-.1)
        assert summary['corner'][condition]['ai_composite']['mean_auc_change']==pytest.approx(.2)
        assert summary['corner'][condition]['ai_composite']['parents_higher_auc']==2
        assert summary['original'][condition]['ai_composite']['parents_lower_auc']==2


def test_fit_requires_cache_restoration(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT',str(tmp_path))
    import run_registered_research as runner
    for name in ('e132','e133_generation','e133_localization','e135_cache'):
        folder=tmp_path/(name+'_pipeline');folder.mkdir()
        (folder/'status.json').write_text(json.dumps(dict(state='complete',E92_restored=True)))
    runner.location_learning_ready(tmp_path,'e135_fit')
    (tmp_path/'e135_cache_pipeline/status.json').write_text(json.dumps(dict(state='running',E92_restored=False)))
    with pytest.raises(RuntimeError):runner.location_learning_ready(tmp_path,'e135_fit')
