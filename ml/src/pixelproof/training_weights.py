"""Linear-time class/source/parent balancing for future training pipelines.

Frozen historical experiments keep their original implementation. This helper does
not choose the total loss mass; callers must freeze that separately from view count.
"""
from collections import Counter,defaultdict

import numpy as np


def balanced_parent_weights(labels,sources,parents):
    labels=np.asarray(labels);sources=np.asarray(sources);parents=np.asarray(parents)
    if labels.ndim!=1 or sources.shape!=labels.shape or parents.shape!=labels.shape:
        raise ValueError('aligned one-dimensional metadata required')
    if set(labels.tolist())!={0,1}:
        raise ValueError('both binary classes required')
    sources=sources.astype(str);parents=parents.astype(str)
    if np.any(sources=='') or np.any(parents==''):raise ValueError('empty source/parent identity')
    triples=list(zip(labels.astype(int).tolist(),sources.tolist(),parents.tolist(),strict=True))
    views=Counter(triples);source_parents=Counter();class_sources=defaultdict(set);identities={}
    for label,source,parent in views:
        identity=(label,source)
        if parent in identities and identities[parent]!=identity:
            raise ValueError('parent crosses a label or source boundary')
        identities[parent]=identity
        source_parents[(label,source)]+=1;class_sources[label].add(source)
    weights=np.asarray([1.0/(2*len(class_sources[label])*source_parents[(label,source)]*views[(label,source,parent)])
                        for label,source,parent in triples],dtype=np.float64)
    weights*=len(weights)/weights.sum()
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('invalid sample weights')
    return weights
