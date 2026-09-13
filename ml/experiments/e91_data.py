"""Append the complete, separately audited SID cohort to immutable four-view TRAIN."""
from collections import Counter
import numpy as np
from experiments.e65_acquisition import read, digest
from experiments.e85_data import load as load_previous, CONDITIONS, WIDTHS
from pixelproof.project_paths import DATA_ROOT


def combine(previous, sid):
    rows = previous + sid
    if len(previous) != 12141 or len(sid) != 128 or \
            len({r['parent_id'] for r in rows}) != len(rows) or \
            len({r['sha256'] for r in rows}) != len(rows) or \
            any(r['role'].upper() != 'TRAIN' or r['label'] not in (0, 1) for r in rows) or \
            sum(r['label'] == 1 for r in rows) != 4595 or \
            any(r['label'] != 0 or not r['training_allowed'] for r in sid) or \
            Counter(r['source'] for r in sid) != {'SID:Sony': 64, 'SID:Fuji': 64} or \
            len({r['original_sha256'] for r in sid}) != 128:
        raise ValueError('complete disjoint 12141+128 admitted TRAIN cohort required')
    return rows


def append_sid(previous, sid, rows, binding):
    if str(sid['binding']) != binding or list(sid['parents']) != [r['parent_id'] for r in rows] or \
            list(sid['roles']) != ['TRAIN'] * len(rows) or list(sid['conditions']) != CONDITIONS or \
            any(r['role'] != 'TRAIN' or r['label'] != 0 for r in rows):
        raise ValueError('ordered four-condition SID TRAIN features required')
    result = {}; previous_count = previous['dino'].shape[0]
    for key, width in WIDTHS.items():
        old, new = previous[key], sid[key]
        if old.shape != (previous_count, 4, width) or new.shape != (len(rows), 4, width) or \
                old.dtype != np.float32 or new.dtype != np.float32 or \
                not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('complete finite previous and SID feature populations required')
        result[key] = np.concatenate([old, new])
        if not np.array_equal(result[key][:previous_count], old):
            raise ValueError('previous four-condition features changed')
    return result


def load():
    previous, features, head = load_previous()
    manifest = read(DATA_ROOT / 'e88/training_manifest.json')
    if manifest['whole_publisher_role'] != 'RESEARCH_TRAIN_ONLY' or manifest['model_scores']:
        raise ValueError('unchanged score-blind SID research TRAIN admission required')
    sid = manifest['rows']; rows = combine(previous, sid)
    with np.load(DATA_ROOT / 'e89/sid_features.npz', allow_pickle=False) as cached:
        combined = append_sid(features, cached, sid, digest(DATA_ROOT / 'e89/features_contract.json'))
    return rows, combined, head
