import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from experiments import e133_mask_location as e


def test_preparation_preserves_authentic_identity_and_outside_mask(tmp_path,monkeypatch):
    root=tmp_path/'e133';root.mkdir();evidence=tmp_path/'evidence';evidence.mkdir()
    contract=root/'contract.json';contract.write_text('{}')
    rng=np.random.default_rng(133)
    source=rng.integers(0,256,(512,512,3),dtype=np.uint8)
    mask=np.zeros((512,512),dtype=np.uint8);mask[128:300,140:310]=255
    files={}
    for name,value in [('original',source),('mask',mask),('AI_negative_target',np.zeros_like(mask))]:
        path=tmp_path/(name+'.png');Image.fromarray(value).save(path)
        files[name]={'path':str(path),'sha256':e.digest(path)}
    rows=[{'index':i,'files':files,'seed':119000+i,'role':'TRAIN_RESEARCH_PILOT'} for i in range(16)]
    geometry=[e.mask_location.corner_mask(mask,i)[1] for i in range(16)]
    for name,value in [('ROOT',root),('EVIDENCE',evidence),('CONTRACT',contract)]:monkeypatch.setattr(e,name,value)
    monkeypatch.setattr(e,'validate',lambda:dict(rows=rows,geometry=geometry))
    monkeypatch.setattr(e,'resource_check',lambda *args:None);monkeypatch.setattr(e,'journal',lambda *args:None)
    result=e.prepare();assert result['parents']==16
    observed=e.prepared_rows()
    for row in observed:
        assert row['files']['original']==files['original']
        actual=np.asarray(Image.open(row['files']['classic']['path']))
        intended=np.asarray(Image.open(row['files']['mask']['path']))>0
        np.testing.assert_array_equal(actual[~intended],source[~intended])
    with pytest.raises(FileExistsError):e.prepare()


def metric(auc):
    return dict(flagged_area_fraction=.1,all_pixel_ranking={'auc':auc},interior_background_ranking={'auc':auc},iou=.2)


def test_summary_counts_all_negatives_but_only_matched_accepted_positives():
    results=[];old=[]
    for condition in e.CONDITIONS:
        for i in range(16):
            maps={v:metric(None) for v in ('authentic','classical_edit')};aligned=dict(authentic=.4,classical_edit=.3)
            if i==5:maps['ai_composite']=metric(.6);aligned['ai_composite']=.6
            results.append(dict(index=i,condition=condition,metrics={'maps':maps},aligned_auc=aligned,radial_center_auc=.2))
            old.append(dict(index=i,condition=condition,metrics={'maps':{'ai_composite':metric(.8 if i==5 else .1)}}))
    for value in e.summarize(results,old,[5]).values():
        assert value['authentic']['parents']==16
        assert value['classical_edit']['parents']==16
        assert value['ai_composite']['parents']==1
        assert value['matched_accepted_comparison']['old_E132_mean_pixel_auc']==pytest.approx(.8)
    for value in e.summarize([dict(r,metrics={'maps':{k:v for k,v in r['metrics']['maps'].items() if k!='ai_composite'}}) for r in results],old,[]).values():
        assert value['ai_composite']['mean_pixel_auc'] is None
        assert value['matched_accepted_comparison']['old_E132_mean_pixel_auc'] is None


def test_scoring_requires_generation_to_release_resources(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT',str(tmp_path))
    import run_registered_research as runner
    for name in ('e130','e131','e132','e133_generation'):
        folder=tmp_path/(name+'_pipeline');folder.mkdir()
        (folder/'status.json').write_text(json.dumps(dict(state='complete',E92_restored=True)))
    runner.location_ready(tmp_path,'e133_generation');runner.location_ready(tmp_path,'e133_localization')
    (tmp_path/'e133_generation_pipeline/status.json').write_text(json.dumps(dict(state='running',E92_restored=False)))
    with pytest.raises(RuntimeError):runner.location_ready(tmp_path,'e133_localization')
