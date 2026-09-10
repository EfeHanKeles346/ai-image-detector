import pytest
from experiments.e56_fivek_review import assign


def test_exact_stem_assignment():
    rows = [{'filename': 'a0001-first.dng'}, {'filename': 'a0002-second.dng'}]
    result = assign(rows, ['a0001-first\n', 'a0002-second\n'])
    assert [r['license'] for r in result] == ['LicenseAdobe.txt', 'LicenseAdobeMIT.txt']
    assert 'license' not in rows[0]


@pytest.mark.parametrize('lists', [ ['a0001-first', 'a0001-first'], ['a0001-first-extra', ''], ['', ''] ])
def test_no_ambiguous_incomplete_or_fuzzy_licence(lists):
    with pytest.raises(ValueError): assign([{'filename': 'a0001-first.dng'}], lists)
