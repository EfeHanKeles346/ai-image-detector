"""Quantify the different row mass used by E131 PCA and weighted head fitting."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from pixelproof import holdout_linear, source_holdout
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    evidence = ML_ROOT.parent / 'evidence'
    path = DATA_ROOT / 'e131/contract.json'
    expected = read(evidence / 'e131_source_holdout_contract.json')['contract_sha256']
    if digest(path) != expected:
        raise ValueError('E131 population differs')
    c = read(path)
    for helper in (holdout_linear, source_holdout):
        if digest(Path(helper.__file__)) != c['inputs'][str(Path(helper.__file__))]:
            raise ValueError('Frozen map/weight implementation differs')
    report = []
    for fold in range(3):
        rows = [r for r in c['rows'] if c['outer_fold'][r['parent_id']] != fold]
        labels = np.asarray([r['label'] for r in rows])
        groups = {r['parent_id']: c['components'][r['parent_id']] for r in rows}
        weights = source_holdout.balanced_weights(rows, groups, 4).reshape(-1, 4).sum(axis=1)
        counts = Counter((r['label'], groups[r['parent_id']]) for r in rows)
        class_groups = Counter(y for y, _ in counts)
        cells = []
        for (label, group), count in sorted(counts.items()):
            selected = np.asarray([r['label'] == label and groups[r['parent_id']] == group for r in rows])
            expected_mass = 1 / (2 * class_groups[label])
            if not np.isclose(weights[selected].sum(), expected_mass):
                raise ValueError('Independent expected head mass differs')
            cells.append(dict(label=label, parents=count,
                sources=sorted({r['source'] for r, keep in zip(rows, selected, strict=True) if keep}),
                PCA_uniform_row_mass=count / len(rows), head_loss_mass=expected_mass))
        report.append(dict(excluded_fold=fold, FIT_parents=len(rows),
            PCA_REAL_mass=float((labels == 0).mean()), head_REAL_mass=float(weights[labels == 0].sum()),
            class_component_cells=cells))
    result = dict(state='representation_weighting_audit_complete', code_sha256=digest(__file__),
        E131_contract_sha256=expected, rows=report, code_basis='holdout_linear.fit_map uses unweighted mean/std and sklearn PCA.fit; source_holdout.balanced_weights is applied only to the final classifier objective.',
        consequence='Equal class/component loss weight does not imply equally weighted normalization or PCA. E136-E138 retain these maps; E146 refits them but still uses uniform row mass.',
        causal_explanation_proven=False, candidate_trained=False,
        downloads=0, new_image_reads=0, model_scores_read=0,
        next_control='One separately registered FIT-only weighted normalization/covariance PCA comparison at unchanged ranks, folds, head objective and cutoff. Preserve the unweighted control and individual AI/REAL gates; no performance gain is inferred from this mass audit.')
    write_once(evidence / 'representation_weighting_audit_20260916.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
