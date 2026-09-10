import pytest
from experiments.e56_fivek_pilot import SUBJECTS
from experiments.e57_data import select, verify_headers


def test_subject_light_selection_includes_small_cells_without_duplicates():
    rows=[{'filename':f'{s}-{l}-{i}.dng','subject':s,'light':l} for s in SUBJECTS for l in ('sun','mixed','artificial') for i in range(15)]
    rows=[r for r in rows if not (r['subject']=='abstract' and r['light']=='sun' and int(r['filename'].split('-')[-1][:-4])>=4)]
    assert len(select(rows))==208
    assert select(rows)==select(rows[::-1])
    with pytest.raises(ValueError):select(rows+[rows[0]])


def test_range_and_source_identity_fail_closed():
    row={'bytes':10,'etag':'x','last_modified':'y'}
    headers={'Content-Length':'6','ETag':'x','Last-Modified':'y','Content-Range':'bytes 4-9/10'}
    verify_headers(headers,row,4,206)
    with pytest.raises(ValueError):verify_headers(headers,row,4,200)
    with pytest.raises(ValueError):verify_headers(headers|{'ETag':'z'},row,4,206)
    with pytest.raises(ValueError):verify_headers(headers|{'Content-Range':'bytes 0-5/10'},row,4,206)
