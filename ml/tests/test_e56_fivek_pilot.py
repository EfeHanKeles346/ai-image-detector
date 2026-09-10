import pytest
from experiments.e56_fivek_pilot import select, SUBJECTS, header_identity, MAX_IMAGE


def test_selection_balanced_deterministic_and_order_independent():
    rows = [{'filename': f'{s}-{i}.dng', 'subject': s} for s in SUBJECTS for i in range(4)]
    picked = select(rows)
    assert len(picked) == 12
    assert picked == select(rows[::-1])
    assert all(sum(r['subject'] == s for r in picked) == 2 for s in SUBJECTS)
    with pytest.raises(ValueError): select(rows+[rows[0]])
    with pytest.raises(ValueError): select([r for r in rows if r['subject'] != 'unknown'])


@pytest.mark.parametrize('headers', [{}, {'Content-Length': '2'},
    {'Content-Length': str(MAX_IMAGE+1), 'ETag': 'x'}, {'Content-Length': '-1', 'ETag': 'x'}])
def test_bad_image_headers_rejected(headers):
    with pytest.raises(ValueError): header_identity(headers)


def test_http_identity_is_explicit():
    assert header_identity({'Content-Length': '23', 'ETag': 'x'}) == {'bytes':23, 'etag':'x', 'last_modified':None}
