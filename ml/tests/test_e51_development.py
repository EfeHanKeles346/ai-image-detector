import pytest

from experiments.e51_development import materialize, select_shared_prompts


def reserve():
    return [{'model_id':f'm{m}','parent_id':f'{c}:{p}:{m}', 'prompt_category':f'c{c}',
             'prompt_ordinal':p,'rank':f'{p:03d}'} for c in range(8) for p in range(23) for m in range(5)]


def test_reject_whole_shared_prompt_without_breaking_generator_balance():
    selected = select_shared_prompts(reserve(),{'0:0:2'})
    assert len(selected)==800
    assert not any(r['prompt_category']=='c0' and r['prompt_ordinal']==0 for r in selected)
    assert {sum(r['model_id']==m for r in selected) for m in {r['model_id'] for r in selected}}=={160}


def test_exhausted_prompt_reserve_cannot_relax_quota():
    with pytest.raises(ValueError): select_shared_prompts(reserve(),{f'0:{p}:0' for p in range(4)})


def test_realized_bytes_are_idempotent_not_overwritten(tmp_path):
    path = tmp_path/'image'
    materialize(b'original',path); materialize(b'original',path)
    with pytest.raises(ValueError): materialize(b'changed',path)
    assert path.read_bytes()==b'original'
