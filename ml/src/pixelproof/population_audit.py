"""Cross-population identity checks with explicit stored-hash coverage limits."""
from collections import defaultdict
import re

BODY_FIELDS = ('sha256', 'original_sha256', 'source_body_sha256')


def identities(rows):
    if not rows:
        raise ValueError('Nonempty population required')
    seen = set()
    index = {key: defaultdict(set) for key in ('parent', 'body', 'pixel')}
    counts = dict(body_parents=0, pixel_parents=0)
    for row in rows:
        parent = row.get('parent_id')
        if not isinstance(parent, str) or not parent or parent in seen:
            raise ValueError('Unique named parents required')
        seen.add(parent)
        index['parent'][parent].add(parent)
        covered = set()
        for field in (*BODY_FIELDS, 'pixel_sha256'):
            value = row.get(field)
            if value is None or value == '':
                continue
            if not isinstance(value, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', value):
                raise ValueError('Malformed stored identity')
            kind = 'pixel' if field == 'pixel_sha256' else 'body'
            index[kind][value.lower()].add(parent)
            covered.add(kind)
        for kind in covered:
            counts[kind + '_parents'] += 1
    return index, dict(parents=len(rows), **counts)


def compare_populations(left, right):
    a, ac = identities(left)
    b, bc = identities(right)
    matches = {}
    for kind in a:
        shared = a[kind].keys() & b[kind].keys()
        matches[kind] = dict(shared_identity_values=len(shared),
            left_parents=len(set().union(*(a[kind][k] for k in shared))) if shared else 0,
            right_parents=len(set().union(*(b[kind][k] for k in shared))) if shared else 0)
    return dict(left=ac, right=bc, matches=matches,
        stored_identity_disjoint=all(not r['shared_identity_values'] for r in matches.values()),
        limits='Stored body/pixel/parent identities only; no fresh image hashing, semantic matching or pretrained-corpus audit. Missing pixel hashes leave comparisons uncovered; zero observed intersections does not prove independence.')
