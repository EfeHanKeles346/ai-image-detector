"""Replay both display policies on bound consumed scores, without new inference."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from pixelproof.demo_scores import scored_display_result
from pixelproof.primary_demo_policy import primary_result, DISPLAY_POLICY
from pixelproof.e92_demo import digest
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    evidence = ML_ROOT.parent / 'evidence'
    report = dict(display_policy=DISPLAY_POLICY, populations={}, new_inference=0,
                  weights_changed=False, thresholds_changed=False,
                  limits='Consumed, dependent, source-limited diagnostic replay. Not fresh accuracy or current API eligibility. Unchanged raw AI alerts are a display invariant, not proof of unseen-AI safety.')
    for name, relative, original, receipt in [
        ('consumed_e66', 'e92/dev_scores.json', 'publisher_original', 'e92_dev_scores.json'),
        ('consumed_e65_REAL_only', 'e93/diagnostic_scores.json', 'original', 'e93_scores_receipt.json')]:
        path = DATA_ROOT / relative
        expected = json.loads((evidence / receipt).read_text())['scores_sha256']
        if digest(path) != expected:
            raise ValueError('Locked scores changed')
        pairs = defaultdict(dict)
        for row in json.loads(path.read_text())['rows']:
            if row['condition'] in pairs[row['parent_id']]:
                raise ValueError('Duplicate parent view')
            pairs[row['parent_id']][row['condition']] = row
        counts = {str(y): dict(parents=0, before=Counter(), after=Counter(),
                               reference_only_changes=0, changed_AI_alerts=0) for y in (0, 1)}
        for pair in pairs.values():
            if set(pair) != {original, 'social_q75'}:
                raise ValueError('Incomplete pair')
            first, second = pair[original], pair['social_q75']
            if first['label'] != second['label']:
                raise ValueError('Inconsistent label')
            s = [first['score'], second['score']]
            ref = [first['reference_score'], second['reference_score']]
            old, new = scored_display_result(s, ref), primary_result(s, ref)
            if old['model_score'] != new['model_score'] or (old['outcome'] == 'ai_signal') != (new['outcome'] == 'ai_signal'):
                raise ValueError('Raw scores or AI alerts changed')
            c = counts[str(first['label'])]
            c['parents'] += 1
            c['before'][old['outcome']] += 1
            c['after'][new['outcome']] += 1
            if old['outcome'] != new['outcome']:
                if old['outcome'] != 'uncertain' or new['outcome'] != 'no_clear_signal' or not new['reference_ai_warning']:
                    raise ValueError('Change is not reference-veto removal')
                c['reference_only_changes'] += 1
        report['populations'][name] = dict(scores_sha256=expected, by_label=counts)
    report['code_sha256'] = {str(p.relative_to(ML_ROOT.parent)): digest(p) for p in [
        Path(__file__), ML_ROOT / 'src/pixelproof/primary_demo_policy.py']}
    with (evidence / 'primary_demo_v2_replay.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
