"""Recover stored E32 metadata into a separate overlay; never change TRAIN roles."""
from collections import defaultdict
from hashlib import sha256
import re

EMPTY_PROMPT = sha256(b'').hexdigest()


def index_unique(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if not key or key in result:
            raise ValueError('Missing or ambiguous metadata identity')
        result[key] = row
    return result


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid stored digest')
    return value


def recover(rows, manifest, receipt, audits, components, folds):
    active = index_unique(rows, 'parent_id')
    if set(active) != set(components) or set(active) != set(folds):
        raise ValueError('Complete registered component/fold maps required')
    for row in rows:
        if str(row.get('role', '')).upper() != 'TRAIN' or type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Only explicit active TRAIN parents allowed')
        checked_hash(row['sha256'])
    original = index_unique(manifest, 'record_id')
    processed = index_unique(receipt, 'record_id')
    upstream = {source: index_unique(audit['records'], 'source_key') for source, audit in audits.items()}
    overlay = []
    for row in rows:
        if row['label'] != 1 or not row['source'].startswith('e32:'):
            continue
        record_id = row['parent_id'].removeprefix('e32:')
        source = row['source'].removeprefix('e32:')
        if row['parent_id'] != 'e32:' + record_id or record_id not in original:
            raise ValueError('Active parent has no original record')
        m = original[record_id]
        if m['source_id'] != source or m['label'] != 'ai' or m['role'] != 'TRAIN':
            raise ValueError('Original source/class/role mismatch')
        if row.get('native') is True:
            if row.get('record_id') != record_id or row['sha256'] != m['sha256']:
                raise ValueError('Native body identity mismatch')
            binding = 'native_body'
        elif row.get('native') is False:
            p = processed[record_id]
            if p['source_id'] != source or p['label'] != 'ai' or p['role'] != 'TRAIN' or p['input_sha256'] != row['sha256']:
                raise ValueError('Processed body identity mismatch')
            binding = 'processed_receipt_to_native_body'
        else:
            raise ValueError('Native/processed status must be explicit')
        audit = audits[source]
        u = upstream[source][m['source_key']]
        if audit['source_id'] != source or u['sha256'] != checked_hash(m['sha256']):
            raise ValueError('Upstream native body/source identity mismatch')
        for key in ('model_name', 'parent_group', 'source_key'):
            if row.get(key) and m.get(key) != row[key]:
                raise ValueError('Active lineage conflicts with original metadata')
        if m.get('model_name') != u.get('model_name'):
            raise ValueError('Conflicting declared model identities')
        prompt = u.get('prompt_sha256')
        if prompt is not None:
            checked_hash(prompt)
        item = dict(parent_id=row['parent_id'], source=row['source'], role='TRAIN',
                    native_sha256=m['sha256'], active_sha256=row['sha256'], binding=binding,
                    declared_model_name=u.get('model_name'), declared_architecture=u.get('architecture'),
                    declared_source_family=audit.get('family'), dataset_revision=audit.get('revision'),
                    source_key=m['source_key'], prompt_sha256=None if prompt == EMPTY_PROMPT else prompt,
                    empty_prompt_sentinel=prompt == EMPTY_PROMPT,
                    base_generator_ancestry_verified=False)
        overlay.append(item)
    return overlay


def summarize(rows, overlay, components, folds):
    recovered = index_unique(overlay, 'parent_id')
    matrix = {}
    ai = [r for r in rows if r['label'] == 1]
    for source in sorted({r['source'] for r in ai}):
        part = [r for r in ai if r['source'] == source]
        extra = [recovered[r['parent_id']] for r in part if r['parent_id'] in recovered]
        matrix[source] = dict(parents=len(part), bound_recovered_parents=len(extra),
            original_model_names_populated=sum(bool(r.get('model_name')) for r in part),
            recovered_model_names_populated=sum(bool(r['declared_model_name']) for r in extra),
            distinct_declared_model_names=len({r['declared_model_name'] for r in extra if r['declared_model_name']}),
            recovered_prompt_hashes=sum(bool(r['prompt_sha256']) for r in extra),
            empty_prompt_sentinels=sum(r['empty_prompt_sentinel'] for r in extra),
            base_generator_ancestry_verified=False)
    by_prompt = defaultdict(list)
    for row in overlay:
        if row['prompt_sha256']:
            by_prompt[row['prompt_sha256']].append(row)
    repeated = [part for part in by_prompt.values() if len(part) > 1]
    links = dict(repeated_prompt_hash_groups=len(repeated),
        parents_in_repeated_prompt_groups=sum(map(len, repeated)),
        cross_source_groups=sum(len({r['source'] for r in part}) > 1 for part in repeated),
        cross_component_groups=sum(len({components[r['parent_id']] for r in part}) > 1 for part in repeated),
        cross_fold_groups=sum(len({folds[r['parent_id']] for r in part}) > 1 for part in repeated))
    return dict(active_parents=len(rows), active_AI_parents=len(ai), recovered_E32_AI_parents=len(overlay),
        coverage=matrix, prompt_links=links,
        recovered_nonempty_prompt_hash_parents=sum(bool(r['prompt_sha256']) for r in overlay),
        AI_parents_without_recovered_prompt_hash=sum(not recovered.get(r['parent_id'], {}).get('prompt_sha256') for r in ai),
        AI_parents_without_recovered_model_name=sum(not recovered.get(r['parent_id'], {}).get('declared_model_name', r.get('model_name')) for r in ai),
        new_independent_test_parents=0, promotion_allowed=False,
        limits='Stored metadata overlay only. Source family and model name are declarations, not verified base-generator ancestry. Dataset revision is not model revision. Prompt hashes retain upstream normalization; equality flags possible shared prompts, inequality does not establish semantic independence. Empty prompt digests are excluded. No image rehash, semantic deduplication, score, fit, role/fold modification or independent evaluation.')
