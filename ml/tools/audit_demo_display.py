"""Read-only display-policy audit on already locked E93 scores. No new inference."""
import json
from collections import Counter
from pathlib import Path

from pixelproof.demo_policy import display_result
from pixelproof.e92_demo import AI_CUT, digest
from pixelproof.project_paths import DATA_ROOT, ML_ROOT, WORK_ROOT
from pixelproof.research_serve import decode_demo
from pixelproof.image_input import DEFAULT_LIMITS, ImagePolicyError


def read(path):
    return json.loads(path.read_text())


def main():
    receipt = read(ML_ROOT.parent / 'evidence/e93_scores_receipt.json')
    if digest(DATA_ROOT / 'e93/diagnostic_scores.json') != receipt['scores_sha256']:
        raise ValueError('E93 scores changed')
    if digest(DATA_ROOT / 'e92/dev_scores.json') != read(ML_ROOT.parent / 'evidence/e92_dev_scores.json')['scores_sha256']:
        raise ValueError('E92 scores changed')
    populations = [('consumed_e66', DATA_ROOT / 'e92/dev_scores.json', 'publisher_original'),
                   ('e65_real_diagnostic', DATA_ROOT / 'e93/diagnostic_scores.json', 'original')]
    report, details = {}, []
    for name, path, condition in populations:
        rows = read(path)['rows']
        paired = {}
        for row in rows:
            paired.setdefault(row['parent_id'], {})[row['condition']] = row
        outputs, eligibility = [], Counter()
        for parent, pair in sorted(paired.items()):
            if set(pair) != {condition, 'social_q75'}:
                raise ValueError('Incomplete pair')
            ordered = [pair[condition], pair['social_q75']]
            result = display_result([r['score'] for r in ordered], [r['reference_score'] for r in ordered])
            if (result['outcome'] == 'ai_signal') != (ordered[0]['score'] >= AI_CUT):
                raise ValueError('An original E92 AI indication was lost or added')
            original = Path(ordered[0]['path'])
            eligible = 'accepted'
            if original.stat().st_size > DEFAULT_LIMITS.max_upload_bytes:
                eligible = 'over_12_MB'
            else:
                if digest(original) != ordered[0]['sha256']:
                    raise ValueError('Locked source image changed')
                try:
                    picture = decode_demo(original.read_bytes())
                    if min(picture.size) < 224:
                        eligible = 'too_small_abstention'
                except ImagePolicyError as exc:
                    eligible = f'input_policy_{exc.status_code}'
            eligibility.update([eligible])
            outputs.append(result | {'parent_id': parent, 'source': ordered[0]['source'],
                           'label': ordered[0]['label'], 'input_eligibility': eligible,
                           'path': str(original), 'sha256': ordered[0]['sha256']})
        report[name] = {'parents': len(outputs), 'lost_raw_ai_alerts': 0, 'added_raw_ai_alerts': 0,
            'outcomes_by_label': {str(label): dict(Counter(r['outcome'] for r in outputs if r['label'] == label)) for label in [0, 1]},
            'positive_review_warnings_by_label': {str(label): sum(r['outcome'] == 'ai_signal' and r['review_required'] for r in outputs if r['label'] == label) for label in [0, 1]},
            'api_input_eligibility': dict(eligibility),
            'accepted_outcomes_by_label': {str(label): dict(Counter(r['outcome'] for r in outputs if r['label'] == label and r['input_eligibility'] == 'accepted')) for label in [0, 1]}}
        details.extend(r | {'population': name} for r in outputs)
    body = {'state': 'display_policy_replay_complete', 'display_policy': 'e92-preserve-alerts-v1',
            'code_sha256': digest(ML_ROOT / 'src/pixelproof/demo_policy.py'),
            'api_sha256': digest(ML_ROOT / 'src/pixelproof/internship_serve.py'),
            'e93_scores_sha256': receipt['scores_sha256'], 'populations': report,
            'limits': 'A presentation requirement, not a new classifier or OOD detector. All original E92 alerts retained including its false alerts. Inconsistent positive alerts receive an explicit warning. Negatives may abstain. API eligibility is decoded separately; do not claim native-benchmark counts as API coverage. No new fit, scores, thresholds or independent final.'}
    out = ML_ROOT.parent / 'evidence/e94_display_audit.json'
    with out.open('x') as f:
        json.dump(body, f, indent=2, sort_keys=True); f.write('\n')
    with (WORK_ROOT / 'e94_display_rows.json').open('x') as f:
        json.dump(details, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(body, indent=2))


if __name__ == '__main__':
    main()
