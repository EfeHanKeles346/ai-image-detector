from experiments.e53_offline import assign_folds
from experiments.e53_coverage import balanced_folds


def test_coverage_balancing_preserves_outer_validation_and_whole_components():
    rows=[{'parent_id':f'{y}:{s}:{i}','source':f'source{y}:{s}','label':y}
          for y in (0,1) for s in range(5) for i in range(100)]
    components,old=assign_folds(rows,[])
    base={'rows':rows,'components':components,'folds':old}
    new=balanced_folds(base)
    assert new==balanced_folds(base)
    for row in rows:
        p=row['parent_id']
        assert any(f['roles'][p]=='FIT' for f in new)
        for a,b in zip(old,new,strict=True):
            assert (a['roles'][p]=='VALIDATION')==(b['roles'][p]=='VALIDATION')
    for fold in new:
        for group in set(components.values()):
            assert len({fold['roles'][p] for p,g in components.items() if g==group})==1
        for role in ('FIT','CAL','VALIDATION'):
            assert {r['label'] for r in rows if fold['roles'][r['parent_id']]==role}=={0,1}
