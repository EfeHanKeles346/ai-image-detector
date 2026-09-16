import copy
import pytest
from pixelproof.lineage_recovery import EMPTY_PROMPT, recover, summarize


def fixture():
    rows = [dict(parent_id='e32:a', source='e32:source', label=1, role='TRAIN', native=True,
                 record_id='a', sha256='1'*64),
            dict(parent_id='e32:b', source='e32:source', label=1, role='train', native=False, sha256='3'*64)]
    manifest = [dict(record_id=k, source_id='source', label='ai', role='TRAIN', source_key=k,
                     sha256=d*64, model_name='declared') for k, d in [('a','1'), ('b','2')]]
    receipt = [dict(record_id='b', source_id='source', label='ai', role='TRAIN', input_sha256='3'*64)]
    audits = {'source': dict(source_id='source', records=[dict(source_key=k, sha256=d*64,
        model_name='declared', prompt_sha256='4'*64) for k,d in [('a','1'),('b','2')]])}
    return rows, manifest, receipt, audits, {'e32:a':'A', 'e32:b':'B'}, {'e32:a':0, 'e32:b':1}


def test_native_and_processed_identity_chains_recover_without_mutating_inputs():
    data = fixture(); before = copy.deepcopy(data)
    overlay = recover(*data)
    assert data == before
    report = summarize(data[0], overlay, data[4], data[5])
    assert report['recovered_nonempty_prompt_hash_parents'] == 2
    assert report['prompt_links']['cross_fold_groups'] == 1
    assert report['promotion_allowed'] is False
    assert all(not r['base_generator_ancestry_verified'] for r in overlay)


@pytest.mark.parametrize('failure', ['role', 'label', 'native_hash', 'processed_hash', 'audit_hash',
                                    'duplicate', 'model_conflict', 'source', 'missing_fold', 'bad_prompt'])
def test_broken_or_ambiguous_identity_is_rejected(failure):
    rows, manifest, receipt, audits, components, folds = data = fixture()
    if failure == 'role': rows[0]['role'] = 'TEST'
    if failure == 'label': manifest[0]['label'] = 'real'
    if failure == 'native_hash': rows[0]['sha256'] = '9'*64
    if failure == 'processed_hash': receipt[0]['input_sha256'] = '9'*64
    if failure == 'audit_hash': audits['source']['records'][0]['sha256'] = '9'*64
    if failure == 'duplicate': manifest.append(copy.deepcopy(manifest[0]))
    if failure == 'model_conflict': audits['source']['records'][0]['model_name'] = 'other'
    if failure == 'source': manifest[0]['source_id'] = 'other'
    if failure == 'missing_fold': folds.pop('e32:a')
    if failure == 'bad_prompt': audits['source']['records'][0]['prompt_sha256'] = 'bad'
    with pytest.raises(ValueError): recover(*data)


def test_empty_prompt_sentinel_never_creates_false_overlap():
    data = fixture()
    for r in data[3]['source']['records']: r['prompt_sha256'] = EMPTY_PROMPT
    overlay = recover(*data)
    report = summarize(data[0], overlay, data[4], data[5])
    assert report['recovered_nonempty_prompt_hash_parents'] == 0
    assert report['prompt_links']['cross_fold_groups'] == 0
    assert report['coverage']['e32:source']['empty_prompt_sentinels'] == 2
