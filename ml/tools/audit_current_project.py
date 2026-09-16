"""Read-only project audit: identities, consumed-DEV recount and synthetic transport parity."""
import io
import json
import socket
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.e92_demo import AI_CUT, CANDIDATE_SHA, MANIFEST, social_view
from pixelproof.primary_demo_policy import primary_result
from pixelproof.population_audit import compare_populations
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
from pixelproof.verified_demo_runtime import verify_manifest, MANIFEST_SHA256


def main():
    evidence = ML_ROOT.parent / 'evidence'
    inputs = {}

    def bound(path, expected):
        if digest(path) != expected:
            raise ValueError('Audit input binding differs')
        inputs[str(path)] = expected
        return read(path)

    verify_manifest()
    manifest = read(MANIFEST)
    verified = Counter()
    for root, key in [(ML_ROOT.parent, 'code_files'), (DATA_ROOT, 'data_files')]:
        for relative, expected in manifest[key].items():
            if digest(root / relative) != expected:
                raise ValueError('Runtime component identity differs')
            verified[key] += 1
    if manifest['data_files']['e92/correction.npz'] != CANDIDATE_SHA:
        raise ValueError('Serving identity and fitted artifact differ')
    train = bound(DATA_ROOT / 'e131/contract.json', read(evidence / 'e131_source_holdout_contract.json')['contract_sha256'])
    older = []
    for name in ('e54/data_contract_v2.json', 'e72/training_manifest.json', 'e88/training_manifest.json'):
        path = DATA_ROOT / name
        older.extend(bound(path, train['inputs'][str(path)])['rows'])
    current = {r['parent_id']: r for r in train['rows']}
    if len(older) != 12269 or len({r['parent_id'] for r in older}) != len(older) or len(current) != 12525:
        raise ValueError('Complete E92/current TRAIN populations required')
    for row in older:
        active = current[row['parent_id']]
        if str(row['role']).upper() != 'TRAIN' or str(active['role']).upper() != 'TRAIN' or \
                any(row[k] != active[k] for k in ('source', 'label', 'sha256')):
            raise ValueError('Historical/current TRAIN identity differs')
    dev_report = bound(DATA_ROOT / 'e92/dev_report.json', digest(evidence / 'e92_development.json'))
    dev_contract = bound(DATA_ROOT / 'e92/dev_contract.json', dev_report['contract_sha256'])
    dev = bound(Path(dev_contract['manifest']), dev_contract['manifest_sha256'])['rows']
    scored = bound(DATA_ROOT / 'e92/dev_scores.json', dev_report['scores_sha256'])['rows']
    overlap = {'E92_TRAIN_vs_consumed_DEV': compare_populations(older, dev),
               'current_TRAIN_vs_consumed_DEV': compare_populations(train['rows'], dev)}
    if not all(r['stored_identity_disjoint'] for r in overlap.values()):
        raise ValueError('Observed TRAIN/DEV identity overlap; investigate before new fitting')
    from experiments.e71_development import validate_pairs
    from experiments.e49_evaluation import evaluate_condition, CONDITIONS
    validate_pairs(scored, dev)
    originals = {r['parent_id']: r for r in scored if r['condition'] == 'publisher_original'}
    if any(originals[r['parent_id']]['sha256'] != r['sha256'] for r in dev):
        raise ValueError('DEV original body lineage differs')
    recount = {}
    for condition in CONDITIONS:
        rows = [r for r in scored if r['condition'] == condition]
        recalculated = evaluate_condition(rows)
        if recalculated != dev_report['reports'][condition]['new']:
            raise ValueError('Published E92 DEV metric differs from locked scores')
        # Independent binary/retention count, separate from the historical evaluator.
        real = [r for r in rows if r['label'] == 0]
        ai = [r for r in rows if r['label'] == 1]
        recount[condition] = dict(REAL_parents=len(real), AI_parents=len(ai),
            REAL_false_alerts=sum(r['score'] >= AI_CUT for r in real),
            AI_detected=sum(r['score'] >= AI_CUT for r in ai),
            newly_missed_reference_AI=sum(r['reference_score'] >= AI_CUT > r['score'] for r in ai),
            numeric_checks_passed=recalculated['gate']['passed_checks'],
            numeric_checks_total=recalculated['gate']['total_checks'])
    paired = {}
    for row in scored:
        paired.setdefault(row['parent_id'], {})[row['condition']] = row
    policy_counts = {str(y): Counter() for y in (0, 1)}
    for pair in paired.values():
        original, social = pair['publisher_original'], pair['social_q75']
        result = primary_result([original['score'], social['score']],
                                [original['reference_score'], social['reference_score']])
        policy_counts[str(original['label'])][result['outcome']] += 1
    from experiments.e65_diagnostic import social_q75_bytes
    parity = []
    rng = np.random.default_rng(147)
    for width, height, orientation in [(224, 224, 1), (1700, 1200, 6), (3000, 2600, 8), (224, 4100, 1)]:
        im = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        stream = io.BytesIO()
        exif = Image.Exif(); exif[274] = orientation
        im.save(stream, format='JPEG', quality=91, exif=exif)
        raw = stream.getvalue()
        with Image.open(io.BytesIO(raw)) as original:
            actual = np.asarray(social_view(original))
        with Image.open(io.BytesIO(social_q75_bytes(io.BytesIO(raw)))) as expected:
            expected_pixels = np.asarray(expected.convert('RGB'))
        if not np.array_equal(actual, expected_pixels):
            raise ValueError('Synthetic runtime/training transport mismatch')
        parity.append(dict(width=width, height=height, exif_orientation=orientation,
                           output_shape=list(actual.shape), exact_pixel_match=True))
    root = DATA_ROOT / 'project_audit_20260916'
    root.mkdir(exist_ok=True)
    write_once(root / 'inputs.json', {'metadata': inputs})
    report = dict(state='project_audit_20260916_complete', code_sha256=digest(__file__),
        runtime_manifest_sha256=MANIFEST_SHA256, verified_runtime_files=dict(verified),
        private_input_bindings_sha256=digest(root / 'inputs.json'),
        overlap=overlap, E92_consumed_DEV_recount=recount, current_policy_counts=policy_counts,
        synthetic_transport_parity=parity, historical_metric_replay_exact=True,
        new_model_scores=0, dataset_image_reads=0, synthetic_inputs=len(parity),
        downloads=0, protected_reserve_reads=0, model_promotion=False,
        limits='No complete semantic/perceptual or pretrained-corpus overlap audit. Partial stored pixel-hash coverage only. E66 is consumed DEV, not an independent final. Numeric gates and repeated sources cannot establish universal reliability. Runtime fixes address integrity/concurrency, not detector accuracy.')
    write_once(root / 'report.json', report)
    write_once(evidence / 'project_audit_20260916.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('Project audit is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    main()
