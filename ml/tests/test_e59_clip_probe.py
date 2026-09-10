from experiments.e59_clip_probe import select


def test_balanced_fit_only_selection():
    rows = [dict(parent_id=f'p{i}', label=i % 2) for i in range(24)]
    roles = ['FIT'] * 16 + ['CAL'] * 4 + ['VALIDATION'] * 4
    ids = select(rows, roles)
    assert len(ids) == len(set(ids)) == 8
    assert all(roles[i] == 'FIT' for i in ids)
    assert sum(rows[i]['label'] for i in ids) == 4
    shuffled = list(reversed(list(zip(rows, roles))))
    other = select([r for r, _ in shuffled], [s for _, s in shuffled])
    assert [rows[i]['parent_id'] for i in ids] == [shuffled[i][0]['parent_id'] for i in other]
