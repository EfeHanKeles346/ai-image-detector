"""Unchanged numeric/retention budgets applied to all four complete TRAIN conditions."""
import numpy as np
from experiments.e85_data import CONDITIONS
from experiments.e49_evaluation import evaluate_condition
from experiments.e64_constrained import AI_CUT


def population_gates(rows,scores,old_count):
    if scores.shape!=(len(rows),len(CONDITIONS)) or not np.isfinite(scores).all() or not 0<old_count<len(rows):
        raise ValueError('complete finite four-condition TRAIN required')
    real=np.array([r['label']==0 for r in rows]);is_new=np.arange(len(rows))>=old_count
    if not np.all(real[is_new]) or not (real&~is_new).any():raise ValueError('old and new REAL required')
    detected=scores>=AI_CUT;sources=np.array([r['source'] for r in rows]);result={}
    for i,condition in enumerate(CONDITIONS):
        sensor={s:float(detected[is_new&(sources==s),i].mean()) for s in sorted(set(sources[is_new]))}
        rates={'old_real':float(detected[real&~is_new,i].mean()),'expanded_real':float(detected[real,i].mean()),
               'new_midd_real':float(detected[is_new,i].mean()),'new_midd_by_sensor':sensor}
        checks={key:rates[key]<=.10 for key in ['old_real','expanded_real','new_midd_real']}
        checks['worst_new_sensor_lte_20']=max(sensor.values())<=.20
        result[condition]={'rates':rates,'checks':checks,'passed':all(checks.values())}
    return result


def full_training_gates(rows,scores,old_count):
    population_gates(rows,scores,old_count);result={}
    for i,condition in enumerate(CONDITIONS):
        views=[{'parent_id':r['parent_id'],'label':r['label'],'source':r['source'],'score':float(scores[j,i]),'role':'TRAIN'} for j,r in enumerate(rows)]
        result[condition]={'old_TRAIN':evaluate_condition(views[:old_count]),'expanded_TRAIN':evaluate_condition(views)}
    return result
