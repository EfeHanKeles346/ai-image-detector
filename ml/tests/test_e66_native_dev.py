import pytest
from experiments.e66_native_dev import select


def test_native_dev_excludes_seen_groups_and_duplicate_endpoints_before_ranking():
    records=[]
    for source in ['gpt-image-1','nano-banana-local']:
        for i in range(8):records.append(dict(record_id=f'{source}:{i}',source_id=source,
            role_group='seen' if i<2 else f'group{i}',role='TRAIN',label='ai'))
    current=[{'parent_id':'e32:gpt-image-1:0'},{'parent_id':'e32:nano-banana-local:0'}]
    inventory={'remaining_record_ids':[r['record_id'] for r in records],
        'internal_pairs':[{'train_parent':'e32:gpt-image-1:2','cal_parent':'e32:nano-banana-local:2'}]}
    selected,eligible=select(records,inventory,current,n=2)
    assert len(selected)==4
    assert all(not r['record_id'].endswith((':0',':1',':2')) for r in eligible)
    assert select(list(reversed(records)),inventory,current,n=2)[0]==selected
    for r in records:r['role']='CALIBRATION'
    with pytest.raises(ValueError,match='insufficient'):select(records,inventory,current,n=2)
