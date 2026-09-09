"""Describe all completed E54 source transitions without new inference or tuning."""
import json
from pathlib import Path

from experiments.e53_diagnostics import describe,transitions
from experiments.e53_offline import EVIDENCE,digest,fixed_write
from experiments.e54_adapt import load
from experiments.e54_data import ROOT


def run():
    load();path=ROOT/'result.json';result=json.loads(path.read_text())
    if digest(path)!=digest(EVIDENCE/'e54_result.json'):
        raise ValueError('research report changed')
    arms={};maps={}
    for arm in ('head_only','last2_anchor','full_old2','full_e51_3','full_expanded3'):
        maps[arm]={'clean':{},'q75':{}};folds=[]
        for f in result['arms'][arm]['folds']:
            source=next(p for p in result['result_bindings'] if p.endswith(f"/{arm}_fold{f['fold']}.json"))
            if digest(source)!=result['result_bindings'][source]:raise ValueError('predictions changed')
            records=json.loads(Path(source).read_text())['observations']
            stats={}
            for condition in ('clean','q75'):
                rows=[r for r in records if r['condition']==condition]
                stats[condition]=describe(rows,f['threshold'])
                for row in rows:
                    if row['parent_id'] in maps[arm][condition]:raise ValueError('duplicate held-out parent')
                    maps[arm][condition][row['parent_id']]=row
            folds.append({'fold':f['fold'],'threshold':f['threshold'],'conditions':stats})
        source_rates={}
        for condition,rows in maps[arm].items():
            by_source={}
            for source in sorted({r['source'] for r in rows.values()}):
                subset=[r for r in rows.values() if r['source']==source]
                labels={r['label'] for r in subset}
                if len(labels)!=1:raise ValueError('source label family changed')
                by_source[source]={'label':labels.pop(),'parents':len(subset),
                                   'predicted_ai':sum(r['predicted_ai'] for r in subset),
                                   'predicted_ai_rate':sum(r['predicted_ai'] for r in subset)/len(subset)}
            source_rates[condition]=by_source
        arms[arm]={'folds':folds,'sources':source_rates}
    changes={}
    for arm in ('head_only','last2_anchor'):
        changes[arm]={ref:{c:transitions(maps[arm][c],maps[ref][c]) for c in ('clean','q75')}
                      for ref in ('full_old2','full_e51_3','full_expanded3')}
    changes['last2_anchor']['head_only']={c:transitions(maps['last2_anchor'][c],maps['head_only'][c]) for c in ('clean','q75')}
    value={'state':'E54_all_source_transitions_described','code_sha256':digest(__file__),
           'helper_sha256':digest(Path(__file__).with_name('e53_diagnostics.py')),
           'result_sha256':digest(path),'arms':arms,'transitions':changes,
           'new_model_scores':0,'serving_changed':False,
           'limits':['TPR@FPR10 is an optimistic held-out-label ranking diagnostic, not a usable threshold.',
                     'No pooled raw-score AUC across independently fitted/calibrated folds.',
                     'Source/topic counts do not establish independent generators, devices or prompt groups.',
                     'These are reused TRAIN outer-validation rows, not the old full-model E49 benchmark or E52.']}
    fixed_write(ROOT/'diagnostics.json',value);fixed_write(EVIDENCE/'e54_diagnostics.json',value)
    return {k:v for k,v in value.items() if k not in {'arms','transitions'}}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
