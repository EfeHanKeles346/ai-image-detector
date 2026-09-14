"""Exact finite-sample planning and a fail-closed audit of consumed E103 evidence."""
import json
import math
from numbers import Integral
from pathlib import Path
from scipy.stats import beta
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import ML_ROOT


def bounds(events, trials, alpha=.05, claims=1):
    """One-sided Clopper-Pearson limits, each at alpha/claims (not a two-sided CI)."""
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (events, trials, claims)) or \
            not 0 <= events <= trials or trials < 1 or claims < 1 or not 0 < alpha < 1:
        raise ValueError('Valid binomial counts, alpha and prespecified claim count required')
    tail = alpha/claims
    return {'lower': 0.0 if events == 0 else float(beta.ppf(tail, events, trials-events+1)),
            'upper': 1.0 if events == trials else float(beta.ppf(1-tail, events+1, trials-events))}


def zero_error_size(target=.01, alpha=.05, claims=1):
    if not 0 < target < 1:
        raise ValueError('Target must lie strictly between zero and one')
    bounds(0, 1, alpha, claims)
    n = math.ceil(math.log(alpha/claims)/math.log1p(-target))
    return n


def assess(events, trials, target, *, independent, fresh, complete, claims=1):
    limits = bounds(events, trials, claims=claims)
    blockers = [name for name, flag in [('dependence_not_resolved', independent),
                ('previously_consumed', fresh), ('incomplete_coverage', complete)] if flag is not True]
    if limits['upper'] > target:
        blockers.append('upper_error_limit_above_target')
    return {'conditional_iid_limits_only': limits, 'proof_supported': not blockers, 'blockers': blockers}


def audit():
    source = ML_ROOT.parent/'evidence/e103_development.json'
    x = read(source); results = {}
    for condition, report in x['reports'].items():
        counts = report['new']['binary_metrics']['confusion']
        scenes = report['real_scenes']
        results[condition] = {
            'photo_FPR': assess(counts['fp'], counts['fp']+counts['tn'],
                .01 if condition == 'publisher_original' else .05,
                independent=False, fresh=False, complete=True),
            'AI_recall_conditional_iid_lower': bounds(counts['tp'], counts['tp']+counts['fn'])['lower'],
            'AI_fresh_independence_established': False,
            'scene_any_error_endpoint': {
                'events': sum(r['new_false_ai'] > 0 for r in scenes.values()), 'scenes': len(scenes),
                'conditional_iid_limits_only': bounds(sum(r['new_false_ai'] > 0 for r in scenes.values()), len(scenes)),
                'warning': 'Scene-any-error risk differs from photo FPR; scenes are consumed and independence is not established.'}}
    result = {'state': 'E108_consumed_evidence_limits_audited',
        'code_sha256': digest(Path(__file__)), 'source_sha256': digest(source),
        'method': 'One-sided exact Clopper-Pearson; illustrative single-claim alpha=.05. No claim of iid sampling for these data.',
        'results': results, 'zero_error_required_independent_fresh_units': {
            'one_claim_FPR_1pct': zero_error_size(),
            'twenty_claims_FPR_1pct_Bonferroni': zero_error_size(claims=20)},
        'historical_twenty_numeric_gates': 'Two conditions times ten engineering criteria; not twenty independent datasets or statistical certificates.',
        'universal_proof_supported': False, 'new_pixels_or_scores_read': 0,
        'next': 'Prospectively register independent parent/scene sampling, all claims and failure accounting before collecting new final evidence.'}
    write_once(ML_ROOT.parent/'evidence/e108_evidence_limits.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
