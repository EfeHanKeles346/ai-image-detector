"""Describe frozen TRAIN source dimensions; no image access, model scores or parameter choice."""
from collections import Counter,defaultdict
from pathlib import Path
import json
from experiments.e65_acquisition import digest,read,write_once
from experiments.e76_fit import combine_rows
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
ROOT=ML_ROOT.parent;E=ROOT/'evidence'


def summarize(rows):
    dimensions=Counter();formats=Counter()
    for row in rows:
        width,height=row.get('width'),row.get('height')
        if not isinstance(width,int) or not isinstance(height,int) or min(width,height)<=0:
            dimensions['unknown']+=1
        elif max(width,height)<=1080:dimensions['at_most1080']+=1
        elif max(width,height)<=2048:dimensions['1081_to2048']+=1
        else:dimensions['above2048']+=1
        formats[str(row.get('decoded_format') or row.get('format') or 'unknown')]+=1
    return {'parents':len(rows),'dimensions':dict(dimensions),'declared_formats':dict(formats)}


def main():
    contract=read(E/'e84b_features_contract.json')
    paths=[DATA_ROOT/'e54/data_contract_v2.json',DATA_ROOT/'e72/training_manifest.json']
    for path in paths:
        if digest(path)!=contract['inputs'][str(path)]:raise ValueError('frozen TRAIN metadata changed')
    rows=combine_rows(*(read(path)['rows'] for path in paths));sources=defaultdict(list)
    for row in rows:sources[(row['label'],row['source'])].append(row)
    result={'state':'E84B_TRAIN_transport_metadata_coverage','inputs':{str(path):digest(path) for path in paths},
        'code_sha256':digest(Path(__file__)),'by_label':{str(y):summarize([r for r in rows if r['label']==y]) for y in (0,1)},
        'by_source':{f'{y}:{source}':summarize(selected) for (y,source),selected in sorted(sources.items())},
        'interpretation':'Recorded dimensions only; long-side size is unchanged by EXIF rotation. The1080 cap is inactive '
                         'for at-most1080 inputs, so their new social policy is expected to duplicate the existing '
                         'source-JPEG75 view. No all-parent pixel equality test is claimed. Counts above1080 describe '
                         'potential resize exposure, not causal effect, performance or independent source coverage.',
        'image_reads':0,'model_scores':0,'parameter_choices':0,'e49_reads':0}
    write_once(E/'e84b_transport_coverage.json',result)
    print(json.dumps(result['by_label'],indent=2))

if __name__=='__main__':main()
