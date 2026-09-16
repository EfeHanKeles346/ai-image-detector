from experiments.e149_source_pixel_probe import select
import pytest


def test_selection_is_order_invariant_covers_strata_modes_and_geometry_extremes():
    rows=[dict(parent_id=str(i),source='s'+str(i%3),label=i%2,processing_history='known' if i<6 else 'unknown',
               mode='RGB',width=500,height=500) for i in range(24)]
    rows[0].update(width=500,height=183)
    rows[23].update(width=8192,height=8192)
    rows[4]['mode']='L';rows[9]['mode']='RGBA'
    selected,count=select(rows)
    assert select(list(reversed(rows)))==(selected,count)
    assert select([r|{'model_score':.99} for r in rows])==(selected,count)
    part=[r for r in rows if r['parent_id'] in selected]
    key=lambda r:(r['source'],r['label'],r['processing_history'])
    assert {key(r) for r in part}=={key(r) for r in rows}
    assert {r['mode'] for r in part}=={'RGB','L','RGBA'}
    assert {'0','23'}.issubset(selected)
    assert len(selected)==len(set(selected))<=count+5
    with pytest.raises(ValueError):select(rows+[rows[0]])
    with pytest.raises(ValueError):select([rows[0]|{'header_error':'OSError'}])
