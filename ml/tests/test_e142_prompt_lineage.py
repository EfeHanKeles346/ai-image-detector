import copy
import pytest
from experiments.e142_prompt_lineage import recover


def fixture():
    a = [dict(label=1, parent_id='e36:generator:qwen-bench:101', source='e36:generator', role='train', sha256='a'*64)]
    b = [dict(label=1, parent_id='qwen-bench:101', source='generator', role='cal', sha256='a'*64, prompt_id=101)]
    return a,b


def test_metadata_recovery_preserves_consumed_roles_and_unknown_ancestry():
    a,b = fixture(); before = copy.deepcopy((a,b))
    row = recover(a,b)[0]
    assert row['declared_prompt_group'] == 'e36:qwen-bench:101'
    assert row['role'] == 'TRAIN' and not row['base_generator_ancestry_verified']
    assert (a,b) == before


@pytest.mark.parametrize('failure', ['body', 'active_role', 'old_role', 'prompt', 'duplicate'])
def test_mismatches_and_reserved_roles_fail_closed(failure):
    a,b=fixture()
    if failure=='body': b[0]['sha256']='b'*64
    if failure=='active_role': a[0]['role']='FINAL'
    if failure=='old_role': b[0]['role']='final'
    if failure=='prompt': b[0]['prompt_id']=1
    if failure=='duplicate': b.append(copy.deepcopy(b[0]))
    with pytest.raises(ValueError): recover(a,b)
