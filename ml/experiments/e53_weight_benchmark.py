"""Compare exact training weights, not model accuracy, on admitted E51 metadata."""
import json
import time
import hashlib
import numpy as np

from experiments.e43_train import parent_source_weights
from experiments.e53_offline import contract,ROOT,EVIDENCE,digest,fixed_write
from pixelproof.training_weights import balanced_parent_weights
import pixelproof.training_weights as implementation


def run():
    value,binding=contract();rows=value['rows']
    labels=np.repeat([r['label'] for r in rows],3)
    sources=np.repeat([r['source'] for r in rows],3)
    parents=np.repeat([r['parent_id'] for r in rows],3)
    protocol={'contract_sha256':binding,'implementation_sha256':digest(implementation.__file__),
              'parents':len(rows),'views':len(labels),'timed_passes':3,
              'require_bitwise_equal_weights':True,'training_or_serving_changed':False}
    fixed_write(ROOT/'weight_benchmark_contract.json',protocol)
    times={'legacy':[],'linear':[]};outputs={};equal=True
    for repetition in range(4):
        methods=[('legacy',parent_source_weights),('linear',balanced_parent_weights)]
        if repetition%2:methods.reverse()
        for name,function in methods:
            start=time.perf_counter();outputs[name]=function(labels,sources,parents);elapsed=time.perf_counter()-start
            if repetition>0:times[name].append(elapsed)
        equal &= np.array_equal(outputs['legacy'],outputs['linear'])
    result={**protocol,'state':'training_weight_equivalence_benchmark_complete','bitwise_equal':bool(equal),
            'weights_sha256':hashlib.sha256(outputs['linear'].tobytes()).hexdigest(),
            'max_error':float(np.max(np.abs(outputs['legacy']-outputs['linear']))),'seconds':times,
            'median_speedup':float(np.median(times['legacy'])/np.median(times['linear'])),
            'limits':['Only sample-weight construction, not whole training runtime or accuracy.',
                      'Old frozen experiment code and current production are unchanged.']}
    fixed_write(EVIDENCE/'e53_weight_benchmark.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
