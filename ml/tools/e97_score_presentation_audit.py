"""Read only locked consumed-DEV scores. Never read pixels, fit or select a cut."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pixelproof.demo_policy import display_result
from pixelproof.demo_scores import scored_display_result
from pixelproof.e92_demo import AI_CUT, REAL_CUT
from pixelproof.internship_serve import DemoResult


def locked_rows(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, 'Locked scores changed'
    return json.loads(raw)['rows']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gallery_sha = '82223943ff259365c1aa9dc2d6679cc491dfc6ad0f4086e618b7d8c5dbee2b1a'
    dev_sha = '8de19298e0e62d982a6bae04ef7e922bcb12748cd68056b510ab832d49bb77ae'
    gallery = locked_rows(args.gallery, gallery_sha)
    dev = locked_rows(args.dev, dev_sha)
    assert len(gallery) == 210 and len(dev) == 640
    outcomes, causes = Counter(), Counter()
    checked = 0

    def check(scores, reference, expected):
        nonlocal checked
        result = scored_display_result(scores, reference)
        assert {k: result[k] for k in expected} == expected
        validated = DemoResult(**result, width=256, height=256)
        assert validated.model_score.original == scores[0]
        assert validated.model_score.social_q75 == scores[1]
        checked += 1

    for row in gallery:
        scores = [row['original_score'], row['social_q75_score']]
        refs = [row['reference_original'], row['reference_q75']]
        check(scores, refs, row['display'])
        outcomes[row['display']['outcome']] += 1
        if row['display']['outcome'] == 'uncertain':
            cause = 'social_AI_crossing' if scores[1] >= AI_CUT else \
                'reference_only_veto' if max(scores) < REAL_CUT else 'borderline_E92'
            causes[cause] += 1
    pairs = defaultdict(dict)
    for row in dev:
        assert row['condition'] not in pairs[row['parent_id']]
        pairs[row['parent_id']][row['condition']] = row
    for pair in pairs.values():
        assert set(pair) == {'publisher_original', 'social_q75'}
        views = [pair['publisher_original'], pair['social_q75']]
        scores = [r['score'] for r in views]
        refs = [r['reference_score'] for r in views]
        check(scores, refs, display_result(scores, refs))
    assert checked == 530
    report = {
        'experiment': 'E97', 'scope': 'consumed DEV presentation replay, not model improvement',
        'gallery_scores_sha256': gallery_sha, 'e66_scores_sha256': dev_sha,
        'paired_displays_checked': checked, 'changed_decisions': 0,
        'raw_score_parity': 'exact', 'gallery_files': len(gallery),
        'gallery_outcomes': dict(outcomes), 'gallery_uncertainty_partition': dict(causes),
        'partition_note': 'Mutually exclusive existing guard conditions; not proven image-processing causes.',
        'calibrated_probability_available': False, 'new_fit': False, 'promotion': False,
        'pixel_reads': 0, 'source_sha256': {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                Path(__file__).relative_to(Path.cwd()), Path('ml/src/pixelproof/demo_scores.py'),
                Path('ml/src/pixelproof/internship_serve.py'), Path('app/demo-contract.ts'),
                Path('app/page.tsx')]},
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
