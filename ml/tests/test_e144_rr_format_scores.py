import pytest
from experiments.e144_rr_format_scores import align_headers


def test_format_join_uses_identity_not_record_order():
    rows = [dict(parent_id='a', source='rr:topic', label=0, role='TRAIN'),
            dict(parent_id='other', source='other', label=1, role='TRAIN'),
            dict(parent_id='b', source='rr:topic', label=1, role='train')]
    headers = [dict(parent_id='b', source='rr:topic', label=1, format='PNG'),
               dict(parent_id='a', source='rr:topic', label=0, format='JPEG')]
    assert align_headers(rows, headers) == [(0, 0, 'JPEG'), (2, 1, 'PNG')]
    for bad in (headers[:1], headers + headers[:1],
                [{**headers[0], 'label': 0}, headers[1]],
                [{**headers[0], 'source': 'rr:wrong'}, headers[1]],
                [{**headers[0], 'header_error': 'ValueError'}, headers[1]]):
        with pytest.raises(ValueError):
            align_headers(rows, bad)
    with pytest.raises(ValueError):
        align_headers([{**rows[0], 'role': 'FINAL'}, *rows[1:]], headers)
    with pytest.raises(ValueError):
        align_headers(rows + rows[:1], headers)
