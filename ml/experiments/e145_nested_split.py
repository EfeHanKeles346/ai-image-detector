"""Metadata-only feasibility of FIT/CAL/EVAL source folds, without score access."""
import json
from pathlib import Path
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.nested_calibration import split_roster
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    root = DATA_ROOT / 'e145'
    root.mkdir(exist_ok=True)
    evidence = ML_ROOT.parent / 'evidence'
    prior = DATA_ROOT / 'e131/contract.json'
    if digest(prior) != read(evidence / 'e131_source_holdout_contract.json')['contract_sha256']:
        raise ValueError('Current TRAIN contract differs')
    c = read(prior)
    report = split_roster(c['rows'], c['components'], c['outer_fold'])
    report.update(state='E145_nested_split_feasibility_complete', parents=len(c['rows']),
        code_sha256=digest(__file__), helper_sha256=digest(Path(split_roster.__code__.co_filename)),
        prior_contract_sha256=digest(prior), engineering_feasible=True,
        independent_validation_supported=False, generator_family_holdout_supported=False,
        scores_read=0, new_fits=0, downloads=0, new_image_reads=0,
        limits='Six ordered FIT/CAL/EVAL assignments, three distinct FIT folds. Both classes in every role, but only one AI-bearing component in each role. Unknown RR/CF ancestry and consumed source groups remain. Feasible internal development diagnostic, not reliable independent source-general calibration.')
    write_once(root / 'report.json', report)
    write_once(evidence / 'e145_nested_split.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
