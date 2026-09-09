"""Fail-closed E54 advancement receipt: relative, absolute and artifact checks."""
import json

from experiments.e53_offline import EVIDENCE,digest,fixed_write
from experiments.e54_adapt import ARMS,load
from experiments.e54_data import ROOT


def decision(arm,replays):
    checks={'relative_ai_preservation_and_real_improvement':bool(arm['research_guard_passed']),
            'all_absolute_fold_gates':len(arm['folds'])==3 and {f['fold'] for f in arm['folds']}=={0,1,2}
                and all(f['rates'][c]['passed'] for f in arm['folds'] for c in ('clean','q75')),
            'three_artifacts_replayed':len(replays)==3 and {r['fold'] for r in replays}=={0,1,2}
                and all(r['passed'] for r in replays)}
    return {'eligible_for_next_research_stage':all(checks.values()),'checks':checks}


def run():
    load();report_path=ROOT/'result.json';replay_path=ROOT/'replay.json'
    report=json.loads(report_path.read_text());replay=json.loads(replay_path.read_text())
    if digest(report_path)!=digest(EVIDENCE/'e54_result.json') or digest(replay_path)!=digest(EVIDENCE/'e54_replay.json'):
        raise ValueError('research/replay evidence differs')
    for path,sha in report['result_bindings'].items():
        if digest(path)!=sha:raise ValueError('archived fold result changed')
    replay_contract=json.loads((ROOT/'replay_contract.json').read_text())
    if digest(ROOT/'replay_contract.json')!=replay['contract_sha256']:
        raise ValueError('replay contract changed')
    for path,sha in replay_contract['inputs'].items():
        if digest(path)!=sha:raise ValueError('replayed artifact changed')
    arms={a:decision(report['arms'][a],[r for r in replay['results'] if r['arm']==a]) for a in ARMS}
    value={'state':'E54_advancement_decided','code_sha256':digest(__file__),
           'result_sha256':digest(report_path),'replay_sha256':digest(replay_path),'arms':arms,
           'eligible':[a for a in ARMS if arms[a]['eligible_for_next_research_stage']],
           'serving_changed':False,'independent_final_passed':False,
           'scope':'Passing only authorizes separately frozen full-data/CAL and diagnostic research; never direct serving promotion or E52 success.'}
    fixed_write(ROOT/'acceptance.json',value);fixed_write(EVIDENCE/'e54_acceptance.json',value)
    return value


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
