from experiments import e84b_features as m,e84_features as original


def test_source_only_revision_preserves_original_probe_plan(monkeypatch,tmp_path):
    expected=list(range(16));written={}
    rows=[{'parent_id':str(i),'source':'synthetic'} for i in range(128)]
    monkeypatch.setattr(m,'ROOT',tmp_path/'new');monkeypatch.setattr(m,'validate_previous',lambda:None)
    monkeypatch.setattr(m,'population',lambda:rows);monkeypatch.setattr(m,'digest',lambda p:'unchanged')
    def read(path):
        if str(path).endswith('e84/features_contract.json'):return {'probe_windows':expected}
        if str(path).endswith('component_diagnostic.json'):return {'score_replay_max_error':0}
        return {'passes_limited_dev_screen':False}
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'write_once',lambda path,value:written.update({str(path):value}))
    result=m.freeze()
    assert result['probe_windows']==expected
    assert m.Encoders is original.Encoders and m.create_chunk is original.create_chunk and m.parity is original.parity
    assert any('e84b_features_contract.json' in p for p in written)
