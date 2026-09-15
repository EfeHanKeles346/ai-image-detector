"""Small paired patch-learning primitives; split parents before fitting anything."""
import numpy as np
from pixelproof.spatial_evaluation import regions


def source_folds(rows):
    if not rows or len({r['parent_id'] for r in rows}) != len(rows):
        raise ValueError('Unique parent identities required')
    group = list(range(len(rows)))
    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ('source', 'scene_group', 'source_body_sha256'):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError('Complete sensor/scene/body ancestry required')
            key = (field, value)
            if key in seen:
                group[root(i)] = root(seen[key])
            seen[key] = i
    roots = sorted({root(i) for i in range(len(rows))})
    if len(roots) < 2:
        raise ValueError('No source-disjoint fold remains')
    return np.asarray([roots.index(root(i)) for i in range(len(rows))])


def pure_patches(mask):
    if np.asarray(mask).shape != (512, 512):
        raise ValueError('Registered 512-square mask required')
    partition = regions(mask, 8)
    def whole(value):
        return value.reshape(32, 16, 32, 16).all(axis=(1, 3))
    positive, negative = whole(partition['interior']), whole(partition['background'])
    if not positive.any() or not negative.any():
        raise ValueError('Every parent needs pure interior/background patches')
    return positive, negative


def training_arrays(rows, tokens, masks, conditions=('original', 'jpeg75')):
    """Half positive mass; negative mass split equally across three variants."""
    xs, ys, ws = [], [], []
    units = len(rows) * len(conditions)
    if not units:
        raise ValueError('Nonempty FIT population required')
    for row in rows:
        positive, negative = pure_patches(masks[row['index']])
        for condition in conditions:
            for variant, selected, target, mass in (
                ('ai_composite', positive, 1, .5),
                ('ai_composite', negative, 0, 1 / 6),
                ('authentic', np.ones((32, 32), dtype=bool), 0, 1 / 6),
                ('classical_edit', np.ones((32, 32), dtype=bool), 0, 1 / 6),
            ):
                value = np.asarray(tokens[f"{row['index']:03d}_{variant}_{condition}"])
                if value.shape != (32, 32, 384) or not np.isfinite(value).all():
                    raise ValueError('Complete finite registered patch tokens required')
                x = value[selected]
                xs.append(x); ys.append(np.full(len(x), target, dtype=np.int64))
                ws.append(np.full(len(x), mass / units / len(x)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit_normalizer(x, weights):
    x = np.asarray(x, dtype=np.float64); weights = np.asarray(weights)
    if x.ndim != 2 or weights.shape != (len(x),) or not np.isfinite(x).all() or \
            not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Finite FIT features and unit positive weights required')
    center = weights @ x
    scale = np.sqrt(np.maximum(weights @ ((x - center) ** 2), 1e-12))
    return center, scale
