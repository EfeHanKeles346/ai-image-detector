"""Blind HDR+ identity and source-metadata checks; never predicts authenticity."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.e51_prefit_audit import cross_role_matches
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_hdrplus_reserve import ROOT, MANIFEST
from experiments.e54_mnw_audit import rejected_parents
from experiments.e54_mnw_reserve import ROOT as MNW
from experiments.e54_reserve_reference import SNAPSHOT

CONTRACT = ROOT/'overlap_contract.json'


def freeze():
    paths = [SNAPSHOT,MANIFEST,ROOT/'download.json',MNW/'download.json',
             Path(__file__).with_name('e51_prefit_audit.py'),Path(__file__).with_name('e54_mnw_audit.py')]
    for path, receipt, key in ((SNAPSHOT,'e54_reserve_reference.json','snapshot_sha256'),
                               (ROOT/'download.json','e54_hdrplus_download.json','report_sha256'),
                               (MNW/'download.json','e54_mnw_reserve_download.json','report_sha256')):
        if digest(path) != json.loads((EVIDENCE/receipt).read_text())[key]:
            raise ValueError('unbound acquisition/reference')
    value = {'state':'HDRPLUS_blind_audit_frozen','code_sha256':digest(__file__),
             'inputs':{str(p):digest(p) for p in paths},'model_scores_created':0,
             'policy':'Withhold all cross-protected and both internal-match endpoints. No replacements or scoring.'}
    fixed_write(CONTRACT,value)
    fixed_write(EVIDENCE/'e54_hdrplus_overlap_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def run():
    config = json.loads(CONTRACT.read_text())
    if digest(__file__) != config['code_sha256'] or digest(CONTRACT) != json.loads(
            (EVIDENCE/'e54_hdrplus_overlap_contract.json').read_text())['contract_sha256']:
        raise ValueError('HDR+ audit contract/code changed')
    for path,sha in config['inputs'].items():
        if digest(path) != sha:
            raise ValueError('HDR+ audit input changed')
    refs = json.loads(SNAPSHOT.read_text())['records']+json.loads((MNW/'download.json').read_text())['records']
    rows = json.loads((ROOT/'download.json').read_text())['records']
    for row in rows:
        if row['label'] != 0 or 'synthetic_' in row['name'] or digest(row['path']) != row['sha256']:
            raise ValueError('HDR+ body/label mismatch')
    cross = cross_role_matches(rows,refs)
    internal = [r for r in cross_role_matches(rows,rows) if r['train_parent'] < r['cal_parent']]
    excluded = rejected_parents(cross,internal)
    eligible = [r for r in rows if r['parent_id'] not in excluded]
    value = {'state':'HDRPLUS_blind_screen_complete_unscored_reserve','contract_sha256':digest(CONTRACT),
             'query_parents':len(rows),'reference_observations':len(refs),
             'withheld_parents':sorted(excluded),'eligible_parent_ids':[r['parent_id'] for r in eligible],
             'cross_matches':cross,'internal_pairs':internal,
             'eligible_camera_models':dict(Counter(r['model'] for r in eligible)),
             'eligible_capture_session_proxies':len({r['capture_session_proxy'] for r in eligible}),
             'eligible_geometries':dict(Counter(f"{r['width']}x{r['height']}" for r in eligible)),
             'model_scores_created':0,'training_allowed_by_project':False,'balanced_final_admitted':False,
             'limits':['All HDR+ publisher derivatives stay reserved; heuristic screening is not exhaustive semantic deduplication.',
                       'Capture-session/day proxy does not guarantee independent scenes; unknown camera ids cannot be invented.',
                       'Small older-phone, one-publisher reserve does not prove modern-device or final E52 coverage.',
                       'Unavailable historical metadata-only bodies cannot receive pixel comparisons.']}
    fixed_write(ROOT/'overlap.json',value)
    summary = {k:v for k,v in value.items() if k not in {'eligible_parent_ids','cross_matches','internal_pairs'}}
    fixed_write(EVIDENCE/'e54_hdrplus_overlap.json',summary|{'report_sha256':digest(ROOT/'overlap.json'),
        'cross_match_observations':len(cross),'internal_pair_observations':len(internal)})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run'])
    args = parser.parse_args();print(json.dumps({'freeze':freeze,'run':run}[args.phase](),indent=2))
