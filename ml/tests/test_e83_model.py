import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from experiments import e83_model as m,e83_fit as run
from test_e80_model import fixture as input_fixture


def test_full_old_map_is_preserved_and_supervised_latent_is_appended(tmp_path):
    x,clip,dear,head,_,previous=input_fixture();previous['weights'][:]=10
    old=m.base.project(head,x,clip,dear,previous)[:,:-1]
    torch.manual_seed(83);net=m.supervised.build_network(385,16,64)
    rep=m.supervised.export_network(net)|{'input_center':np.zeros(385),'input_scale':np.ones(385)}
    raw=m.supervised.latent_raw(old,rep);scaler=StandardScaler().fit(raw)
    rep.update(latent_center=scaler.mean_,latent_scale=scaler.scale_)
    a=m.assemble(previous,rep)
    assert a['weights'].shape==(450,) and not a['weights'].any()
    assert np.array_equal(m.predict(head,x,clip,dear,a),head.predict_proba(x)[:,1])
    z=m.project(head,x,clip,dear,a)
    assert np.array_equal(z[:,:385],old)
    assert np.array_equal(z[:,385:449],m.supervised.coordinates(old,rep))
    assert previous['weights'].sum()==3860
    for k in m.BASE_KEYS:assert not np.shares_memory(a[k],previous[k])
    a['weights'][390]=.1;path=tmp_path/'candidate.npz';np.savez_compressed(path,**a)
    expected=m.predict(head,x,clip,dear,a)
    with np.load(path,allow_pickle=False) as saved:assert np.array_equal(m.predict(head,x,clip,dear,saved),expected)
    with pytest.raises(ValueError,match='aligned'):m.predict(head,x,clip,dear[:-1],a)


def test_new_report_never_writes_previous_experiment_evidence(monkeypatch,tmp_path):
    import json
    old=tmp_path/'e81_fit.json';old.write_text('previous immutable report')
    monkeypatch.setattr(run,'EVIDENCE',tmp_path);monkeypatch.setattr(run,'REPORT',tmp_path/'fit.json')
    run.write_report({'state':'new'})
    assert old.read_text()=='previous immutable report'
    assert json.loads((tmp_path/'e83_fit.json').read_text())=={'state':'new'}
    with pytest.raises(FileExistsError):run.write_report({'state':'changed'})
