"""Paired TRAIN out-of-fold summary; cannot authorize deployment or a final pass."""
from collections import defaultdict
import json

import numpy as np

from experiments.e53_offline import ARMS, ROOT, EVIDENCE, contract, digest, fixed_write


def paired_interval(labels, components, candidate, baseline, repetitions=20000):
    """Publisher-component bootstrap, paired across arms and both transports.

    Each prediction matrix is [parents, clean/Q75]. Intervals are Bonferroni-adjusted
    over the four primary deltas for this pair. They condition on trained fold heads.
    """
    labels=np.asarray(labels);components=np.asarray(components)
    candidate=np.asarray(candidate,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if candidate.shape!=baseline.shape or candidate.shape!=(len(labels),2):raise ValueError('pair shape mismatch')
    if set(labels)!={0,1}:raise ValueError('both classes required')
    summaries=[];strata=defaultdict(list)
    for group in sorted(set(components)):
        m=components==group;delta=candidate[m]-baseline[m];y=labels[m]
        row=np.r_[np.sum(y==0),np.sum(y==1),delta[y==0].sum(axis=0),delta[y==1].sum(axis=0)]
        strata[tuple(sorted(set(y)))].append(len(summaries));summaries.append(row)
    summaries=np.asarray(summaries);rng=np.random.default_rng(53)
    accum=np.zeros((repetitions,6))
    for ids in strata.values():
        draws=rng.choice(ids,size=(repetitions,len(ids)),replace=True)
        accum+=summaries[draws].sum(axis=1)
    if np.any(accum[:,:2]<=0):raise ValueError('bootstrap lost a class')
    deltas=np.c_[accum[:,2:4]/accum[:,0,None],accum[:,4:6]/accum[:,1,None]]
    # Four two-sided intervals with joint family coverage at least 95% by Bonferroni.
    quantiles=np.quantile(deltas,[.00625,.99375],axis=0)
    return {key:[float(x) for x in quantiles[:,i]] for i,key in enumerate(
        ('real_fpr_clean','real_fpr_q75','ai_recall_clean','ai_recall_q75'))}


def preservation_guard(labels,sources,candidate,baseline,intervals,min_source_parents=50):
    labels=np.asarray(labels);sources=np.asarray(sources)
    real,ai=labels==0,labels==1
    real_delta=candidate[real].mean(axis=0)-baseline[real].mean(axis=0)
    ai_delta=candidate[ai].mean(axis=0)-baseline[ai].mean(axis=0)
    source_delta={};unsupported=[]
    for source in sorted(set(sources[ai])):
        m=ai&(sources==source)
        if m.sum()<min_source_parents:unsupported.append(source);continue
        source_delta[source]=(candidate[m].mean(axis=0)-baseline[m].mean(axis=0)).tolist()
    checks={'real_fpr_lower_both':bool(np.all(real_delta<0)),
            'ai_recall_no_point_loss_both':bool(np.all(ai_delta>=0)),
            'supported_ai_sources_no_point_loss':all(min(v)>=0 for v in source_delta.values()),
            'real_improvement_interval_both':all(intervals[f'real_fpr_{c}'][1]<0 for c in ('clean','q75')),
            'ai_preservation_interval_both':all(intervals[f'ai_recall_{c}'][0]>=0 for c in ('clean','q75'))}
    return {'passed':all(checks.values()),'checks':checks,'real_fpr_delta':real_delta.tolist(),
            'ai_recall_delta':ai_delta.tolist(),'supported_ai_source_deltas':source_delta,
            'unsupported_ai_sources':unsupported,
            'support_limitation':'50 image parents is a screening floor, not verified independent prompts or proof for small sources.'}


def report(include_expansion=False):
    value,binding=contract();known={r['parent_id']:r for r in value['rows']};parents=sorted(known)
    arm_names=list(ARMS);expansion_sha=None
    if include_expansion:
        from experiments.e53_expansion import ARMS as EXTRA_ARMS, load, CONTRACT
        load();expansion_sha=digest(CONTRACT);arm_names.extend(EXTRA_ARMS)
    labels=np.asarray([known[p]['label'] for p in parents]);sources=np.asarray([known[p]['source'] for p in parents])
    components=np.asarray([value['components'][p] for p in parents]);arms={};predictions={};bindings={}
    for arm in arm_names:
        maps={};folds=[]
        for fold in range(3):
            path=ROOT/f'results/{arm}_fold{fold}.json'
            if not path.exists():raise ValueError(f'all six arms required, missing {path.name}')
            result=json.loads(path.read_text());bindings[str(path)]=digest(path)
            if result['contract_sha256']!=binding:raise ValueError('result contract mismatch')
            if arm not in ARMS and result.get('expansion_contract_sha256')!=expansion_sha:raise ValueError('expansion result mismatch')
            for row in result['observations']:
                p=row['parent_id'];key=(p,row['condition'])
                if key in maps or value['folds'][fold]['roles'].get(p)!='VALIDATION':raise ValueError('OOF leakage or duplicate observation')
                if row['label']!=known[p]['label'] or row['source']!=known[p]['source']:raise ValueError('label/source mismatch')
                if row['predicted_ai']!=(row['score']>=result['threshold']):raise ValueError('prediction/cut mismatch')
                maps[key]=row
            folds.append({k:v for k,v in result.items() if k!='observations'})
        expected={(p,c) for p in parents for c in ('clean','q75')}
        if set(maps)!=expected:raise ValueError('incomplete paired OOF population')
        predicted=np.asarray([[maps[(p,c)]['predicted_ai'] for c in ('clean','q75')] for p in parents])
        predictions[arm]=predicted
        rates={}
        for j,c in enumerate(('clean','q75')):
            fp=predicted[labels==0,j].mean();tp=predicted[labels==1,j].mean()
            by_source={source:float(predicted[sources==source,j].mean()) for source in sorted(set(sources))}
            real_sources=sorted(set(sources[labels==0]));ai_sources=sorted(set(sources[labels==1]))
            rates[c]={'real_false_ai':float(fp),'ai_recall':float(tp),'balanced_accuracy':float((1-fp+tp)/2),
                      'worst_real_source':max(by_source[s] for s in real_sources),
                      'worst_ai_source':min(by_source[s] for s in ai_sources),
                      'source_predicted_ai_rates':by_source,'coverage':1.0,
                      'unweighted_mean_fold_auc':float(np.mean([f['rates'][c]['auc'] for f in folds]))}
        arms[arm]={'conditions':rates,'folds':folds,'comparisons':{}}
    for arm in arm_names:
        for baseline in ('full_old2','full_e51_3'):
            intervals=paired_interval(labels,components,predictions[arm],predictions[baseline])
            arms[arm]['comparisons'][baseline]={'intervals':intervals,**preservation_guard(
                labels,sources,predictions[arm],predictions[baseline],intervals)}
        arms[arm]['research_guard_passed']=all(v['passed'] for v in arms[arm]['comparisons'].values())
        arms[arm]['all_held_out_fold_checks_passed']=all(f['rates'][c]['passed'] for f in arms[arm]['folds'] for c in ('clean','q75'))
    result={'schema_version':1,'state':'E53_TRAIN_source_held_out_complete','contract_sha256':binding,
            'result_bindings':bindings,'parents':len(parents),'publisher_components':len(set(components)),
            'arms':arms,'serving_changed':False,'independent_final_passed':False,
            'research_guard_survivors':[a for a in arms if arms[a]['research_guard_passed']],
            'limitations':['This is TRAIN-derived model-selection evidence, not a fresh final.',
                          'Different fold-specific CAL cuts are used; pooled raw-score AUC is deliberately not reported.',
                          'Intervals condition on 11 observed source components and fitted heads; unknown domains are not covered.',
                          'Small-source/prompt independence is not established by 50-parent support.',
                          'Per-pair multiplicity correction does not certify a winner selected across six arms.',
                          'Old2 versus three-view arms also change mean-one total loss mass; augmentation is not the only causal change.',
                          'Native expansion, when present, holds total weight mass fixed to the original three-view FIT population.',
                          'AI preservation versus frozen deployed/E43/E51 models still needs identical independent final rows.']}
    suffix='_expanded' if include_expansion else ''
    result['expansion_contract_sha256']=expansion_sha
    fixed_write(ROOT/f'summary{suffix}.json',result);fixed_write(EVIDENCE/f'e53_source_held_out{suffix}_result.json',result)
    return {k:v for k,v in result.items() if k not in {'arms','result_bindings'}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--include-expansion',action='store_true');args=parser.parse_args()
    print(json.dumps(report(args.include_expansion),indent=2))
