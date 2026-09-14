"""Paired consumed-DEV regression protection, never a fit or independent-final claim."""
import math


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (row['parent_id'], row['condition'])
        if key in indexed or row.get('role') != 'CONSUMED_DEVELOPMENT':
            raise ValueError('Unique consumed-development parent/condition rows required')
        if type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('Binary integer labels required')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('parent_id', 'condition', 'source', 'sha256')):
            raise ValueError('Complete identity/provenance required')
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('SHA256 identity required')
        if not isinstance(row.get('score'), (int, float)) or isinstance(row['score'], bool) or \
                not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Finite bounded score required')
        indexed[key] = row
    if not indexed or {r['label'] for r in indexed.values()} != {0, 1}:
        raise ValueError('Both REAL and AI populations required')
    return indexed


def compare_development(reference_rows, candidate_rows, ai_cut):
    if not isinstance(ai_cut, (int, float)) or isinstance(ai_cut, bool) or not math.isfinite(ai_cut) or not 0 < ai_cut < 1:
        raise ValueError('Fixed finite AI cut required')
    old, new = index_rows(reference_rows), index_rows(candidate_rows)
    if set(old) != set(new):
        raise ValueError('Complete identical paired population required')
    new_real_errors = lost_ai = 0
    by_source = {}
    for key, row in old.items():
        candidate = new[key]
        if any(candidate[k] != row[k] for k in ('label', 'source', 'sha256')):
            raise ValueError('Candidate labels/source/pixels differ')
        before, after = row['score'] >= ai_cut, candidate['score'] >= ai_cut
        if row['label'] == 1:
            lost_ai += before and not after
        else:
            new_real_errors += not before and after
        group = f"{row['label']}:{row['source']}:{row['condition']}"
        values = by_source.setdefault(group, {'views': 0, 'reference_ai': 0, 'candidate_ai': 0})
        values['views'] += 1; values['reference_ai'] += int(before); values['candidate_ai'] += int(after)
    if not any(r['label'] == 1 and r['score'] >= ai_cut for r in old.values()):
        raise ValueError('Vacuous AI protection')
    passed = not new_real_errors and not lost_ai
    return {'passes_paired_regression': passed, 'views': len(old),
            'new_real_false_ai': int(new_real_errors), 'newly_missed_ai': int(lost_ai),
            'by_source_condition': by_source, 'promotion_allowed': False,
            'limits': 'Consumed-development regression only. Passing does not repair the E92 acceptance failure, prove OOD reliability or authorize promotion. No thresholds/weights may be fit on these rows.'}
