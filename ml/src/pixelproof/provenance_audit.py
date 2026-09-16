"""Metadata-only diagnostics; never infer independence from missing collisions."""
from collections import Counter, defaultdict
import re

FIELDS = ('sha256', 'pixel_sha256', 'original_sha256', 'device', 'sensor', 'camera',
          'scene_group', 'parent_group', 'model_name', 'generator', 'generator_id',
          'generator_revision', 'prompt', 'prompt_id', 'prompt_sha256',
          'source_id', 'record_id', 'provenance')
BODY_FIELDS = ('sha256', 'original_sha256', 'source_body_sha256')


def present(value):
    return value is not None and value != '' and value != [] and value != {}


def audit(rows, components, folds):
    ids = [r['parent_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or set(ids) != set(components) or set(ids) != set(folds):
        raise ValueError('Unique parents with complete component/fold mappings required')
    if any(str(r.get('role', '')).upper() != 'TRAIN' or type(r['label']) is not int or
           r['label'] not in (0, 1) or not r.get('source') or
           not re.fullmatch(r'[0-9a-fA-F]{64}', r.get('sha256', '')) or
           type(folds[r['parent_id']]) is not int or folds[r['parent_id']] not in (0, 1, 2) for r in rows):
        raise ValueError('Explicit TRAIN roles, binary labels, body hashes and registered folds required')
    def coverage(part):
        return dict(parents=len(part), labels=dict(Counter(str(r['label']) for r in part)),
                    populated_fields={k: sum(present(r.get(k)) for r in part) for k in FIELDS},
                    scene_independence_verified=sum(r.get('scene_independence_verified') is True for r in part))
    aliases = {kind: defaultdict(set) for kind in ('body', 'pixel', 'declared_scene')}
    for i, row in enumerate(rows):
        for key in BODY_FIELDS + ('pixel_sha256',):
            value = row.get(key)
            if not present(value): continue
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
                raise ValueError('Malformed stored digest; no silent coverage claim')
            aliases['pixel' if key == 'pixel_sha256' else 'body'][value.lower()].add(i)
        if present(row.get('scene_group')):
            aliases['declared_scene'][row['scene_group']].add(i)
    links = {}
    for kind, values in aliases.items():
        repeated = [members for members in values.values() if len(members) > 1]
        links[kind] = dict(repeated_identity_groups=len(repeated),
            parents_in_repeated_groups=len(set().union(*repeated)) if repeated else 0,
            cross_label_groups=sum(len({rows[i]['label'] for i in members}) > 1 for members in repeated),
            cross_source_groups=sum(len({rows[i]['source'] for i in members}) > 1 for members in repeated),
            cross_component_groups=sum(len({components[ids[i]] for i in members}) > 1 for members in repeated),
            cross_fold_groups=sum(len({folds[ids[i]] for i in members}) > 1 for members in repeated))
    group_folds = defaultdict(set)
    for parent in ids: group_folds[components[parent]].add(folds[parent])
    sources = {s: coverage([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})}
    return dict(parents=len(rows), sources=sources, classes={str(label): coverage([r for r in rows if r['label'] == label]) for label in (0, 1)},
        declared_components=len(group_folds), AI_bearing_components=len({components[r['parent_id']] for r in rows if r['label'] == 1}),
        components_crossing_folds=sum(len(fs) > 1 for fs in group_folds.values()), stored_identity_links=links,
        mixed_corpus_AI_rows_without_explicit_generator_field=sum(r['label'] == 1 and
            (r['source'].startswith('rr:') or r['source'] == 'e32:communityforensics-ai-local') and
            not any(present(r.get(k)) for k in ('generator', 'generator_id', 'model_name')) for r in rows),
        independent_test_parents_created=0, pixels_read=0, model_scores_created=0,
        limits='Field population is not verified provenance. Stored body hashes are not rehashed image bytes; pixels only where stored. Scene links follow declared identifiers, not a fresh semantic/near-duplicate audit. Zero observed collisions does not certify independence. All rows remain consumed TRAIN; no reserve scoring or role promotion.')
