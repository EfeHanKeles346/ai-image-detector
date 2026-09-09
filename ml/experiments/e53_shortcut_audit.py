"""TRAIN-only input geometry/crop-dispersion audit; no causal or quality claim."""
from collections import defaultdict
import json

import numpy as np

from experiments.e51_pipeline import FEATURES
from experiments.e53_offline import ROOT, EVIDENCE, contract, digest, fixed_write
from experiments.e53_expansion import load as expansion, NATIVE_FEATURES


def summarize(rows, features):
    if features.shape != (len(rows), 3072) or not np.isfinite(features).all():
        raise ValueError('aligned finite E42 features required')
    sources = defaultdict(list)
    for index, row in enumerate(rows):
        label = row['label']
        if label not in (0, 1, 'real', 'ai'):
            raise ValueError('unknown label')
        label = 1 if label in (1, 'ai') else 0
        width, height = row.get('width'), row.get('height')
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError('missing verified original geometry')
        source = row.get('source', row.get('source_id'))
        if not source:
            raise ValueError('missing source')
        for key in (f'class:{label}', f'class:{label}|source:{source}'):
            sources[key].append((index, width, height))
    output = {}
    for key, entries in sorted(sources.items()):
        indices = [r[0] for r in entries]
        zero = np.max(np.abs(features[indices, 1536:]), axis=1) <= 1e-6
        output[key] = {'parents': len(entries),
            'exact_224_square': sum(w == h == 224 for _, w, h in entries),
            'short_side_at_least_512': sum(min(w, h) >= 512 for _, w, h in entries),
            'square': sum(w == h for _, w, h in entries),
            'near_zero_crop_dispersion': int(zero.sum()),
            'median_crop_dispersion_l2': float(np.median(np.linalg.norm(features[indices, 1536:], axis=1)))}
    return output


def report():
    base, binding = contract()
    _, extra, _ = expansion()
    receipts = [(FEATURES, base['features_sha256']), (NATIVE_FEATURES,
        json.loads((EVIDENCE/'e53_native_features.json').read_text())['feature_sha256'])]
    for path, expected in receipts:
        if digest(path) != expected:
            raise ValueError('feature archive changed')
    with np.load(FEATURES, allow_pickle=False) as archive:
        use = (archive['roles'] == 'TRAIN') & (archive['conditions'] == 'clean')
        parents, features = archive['parents'][use], archive['dino'][use]
    known = {r['parent_id']: r for r in base['rows']}
    if len(set(parents)) != len(parents) or set(parents) != set(known):
        raise ValueError('incomplete clean TRAIN alignment')
    rows = [known[str(p)] for p in parents]
    with np.load(NATIVE_FEATURES, allow_pickle=False) as archive:
        if list(archive['record_ids']) != [r['record_id'] for r in extra['rows']]:
            raise ValueError('native identity alignment changed')
        native = archive['features'][:, 0, :]
    result = {'state': 'TRAIN_input_shortcut_audit_complete', 'code_sha256': digest(__file__),
        'base_contract_sha256': binding, 'expansion_contract_sha256': digest(ROOT/'expansion_contract.json'),
        'pools': {'E51_TRAIN': summarize(rows, features),
                  'native_additions': summarize(extra['rows'], native),
                  'combined_candidate_pool': summarize(rows+extra['rows'], np.r_[features, native])},
        'new_model_scores': 0, 'serving_changed': False,
        'limitations': ['Geometry and crop dispersion correlate with source/label; causation is not established.',
            'Combined pool counts are an inventory, not the FIT count of each source-held-out fold.',
            'Original dimensions do not guarantee native acquisition or verified camera/prompt identity.',
            'Dispersion<=1e-6 is descriptive, not a detection rule or proof an image is authentic.',
            'Never use these geometry values as production label heuristics.']}
    fixed_write(EVIDENCE/'e53_shortcut_audit.json', result)
    return {k: v for k, v in result.items() if k != 'pools'}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
