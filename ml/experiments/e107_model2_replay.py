"""Full-coverage, non-oracle CocoGlide replay of the historical crop128 baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.features import extract_from_image
from pixelproof.project_paths import DATA_ROOT, LEGACY_DATA_ROOT, ML_ROOT, WORK_ROOT

ROOT = DATA_ROOT/'e107'
EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'contract.json'
INVENTORY = WORK_ROOT/'model2_local_inventory_2026-09-14.json'
MODEL = ML_ROOT/'artifacts/feature_crop128.joblib'
TILE = 128


def positions(length):
    if length < TILE:
        raise ValueError('Native dimension below 128')
    return sorted(set(range(0, length-TILE+1, TILE//2)) | {length-TILE})


def dense_map(scores, boxes, size):
    width, height = size
    scores = np.asarray(scores)
    if len(scores) != len(boxes) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid tile probabilities')
    total = np.zeros((height, width), np.float64)
    count = np.zeros((height, width), np.uint16)
    for score, (x, y) in zip(scores, boxes):
        if not 0 <= x <= width-TILE or not 0 <= y <= height-TILE:
            raise ValueError('Tile outside native bounds')
        total[y:y+TILE, x:x+TILE] += score
        count[y:y+TILE, x:x+TILE] += 1
    if np.any(count == 0):
        raise ValueError('Uncovered pixels')
    return total/count


def metrics(truth, score):
    if truth.dtype != bool or truth.shape != score.shape or not truth.any() or truth.all() or not np.isfinite(score).all():
        raise ValueError('Nondegenerate aligned truth and finite scores required')
    predicted = score >= .5
    return {'auc': float(roc_auc_score(truth.ravel(), score.ravel())),
            'ap': float(average_precision_score(truth.ravel(), score.ravel())),
            'iou_at_fixed_half': float(np.sum(truth & predicted)/np.sum(truth | predicted)),
            'prevalence': float(truth.mean())}


def baselines(size, identity):
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    center = np.clip(1-np.sqrt(((xx+.5-width/2)/(width/2))**2 +
                             ((yy+.5-height/2)/(height/2))**2), 0, 1)
    seed = int.from_bytes(hashlib.sha256(('E107|'+identity).encode()).digest()[:8], 'big')
    boxes = [(x, y) for y in positions(height) for x in positions(width)]
    random_map = dense_map(np.random.default_rng(seed).random(len(boxes)), boxes, size)
    return {'center': center, 'constant_zero': np.zeros((height, width)),
            'constant_one': np.ones((height, width)), 'seeded_random_tiles': random_map}


def decode(path, expected):
    if digest(path) != expected:
        raise ValueError('Image hash differs')
    with Image.open(path) as image:
        width, height = image.size
        if min(width, height) < TILE or width*height > 32_000_000:
            raise ValueError('Native geometry outside registered limits')
        image.load()
        return image.convert('RGB')


def score_image(model, image):
    boxes = [(x, y) for y in positions(image.height) for x in positions(image.width)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = np.stack(list(pool.map(extract_from_image,
            (image.crop((x, y, x+TILE, y+TILE)) for x, y in boxes))))
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Model class ordering differs')
    scores = model.predict_proba(features)[:, 1]
    return dense_map(scores, boxes, image.size), len(boxes)


def freeze():
    if digest(INVENTORY) != read(EVIDENCE/'model2_local_inventory_2026-09-14.json')['inventory_sha256']:
        raise ValueError('Inventory receipt differs')
    rows = [r for r in read(INVENTORY)['rows'] if r['old_first120_exposure_possible']]
    if len(rows) != 120 or any(r['problems'] for r in rows) or len({r['auth_sha256'] for r in rows}) != 120:
        raise ValueError('Expected 120 complete unique exposed parents')
    files = [Path(__file__), INVENTORY, MODEL, Path(extract_from_image.__code__.co_filename)]
    c = {'state': 'E107_non_oracle_Model2_replay_registered',
        'inputs': {str(p): digest(p) for p in files}, 'selected_rows': rows,
        'selection': 'All first120 sorted historical CocoGlide edits, each with its metadata-linked authentic control; no mask-coverage filter.',
        'mapping': '128px native tiles, stride64, include final edges, overlap mean; no tile cap/texture filter/resizing.',
        'threshold': .5, 'threshold_status': 'Fixed diagnostic default, not calibrated; no truth-dependent threshold.',
        'metrics': 'Per-image pixel AUC/AP/IoU, image-macro means and area strata. Image detection uses fixed p95 of dense map; authentic flagged area at .5. Paired parent bootstrap 1000 seed107 of mean metrics and IoU differences to all four baselines.',
        'baselines': ['center', 'constant_zero', 'constant_one', 'seeded_random_tiles'],
        'limits': 'Historically consumed single-generator diagnostic, not independent generalization. Auth identities exact only, shared source ancestry unresolved. No training, threshold tuning, promotion or user-image analysis.',
        'failure_policy': 'Retain failures in denominator and report reasons; any failed pair invalidates full-coverage success. No refills.'}
    ROOT.mkdir(parents=True, exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e107_model2_contract.json', {k:v for k,v in c.items() if k != 'selected_rows'} |
        {'contract_sha256': digest(CONTRACT), 'selected_pairs': len(rows)})
    return {'registered_pairs': len(rows), 'checkpoint_sha256': digest(MODEL)}


def summarize(rows):
    good = [r for r in rows if 'error' not in r]
    if not good:
        return {'attempted': len(rows), 'completed': 0, 'full_coverage': False}
    rng = np.random.default_rng(107)
    bootstrap = rng.integers(0, len(good), size=(1000, len(good)))
    def estimate(values):
        values = np.array(values)
        return {'mean': float(values.mean()), 'parent_bootstrap_95pct':
                np.quantile(values[bootstrap].mean(axis=1), [.025, .975]).tolist()}
    names = list(good[0]['metrics'])
    result = {'attempted': len(rows), 'completed': len(good), 'full_coverage': len(rows) == len(good),
        'image_macro': {name: {key: estimate([r['metrics'][name][key] for r in good])
            for key in ('auc', 'ap', 'iou_at_fixed_half', 'prevalence')} for name in names},
        'paired_iou_improvement': {name: estimate([r['metrics']['crop128']['iou_at_fixed_half']-
            r['metrics'][name]['iou_at_fixed_half'] for r in good]) for name in names if name != 'crop128'},
        'authentic_flagged_area': estimate([r['auth_flagged_area'] for r in good]),
        'authentic_fraction_over_5pct_flagged': float(np.mean([r['auth_flagged_area'] > .05 for r in good])),
        'image_detection_auc_fixed_p95': float(roc_auc_score([0]*len(good)+[1]*len(good),
            [r['auth_p95'] for r in good]+[r['edit_p95'] for r in good])),
        'total_native_tiles': sum(r['tiles'] for r in good)}
    result['area_strata'] = {}
    for lo, hi in [(0, .01), (.01, .05), (.05, .2), (.2, .5), (.5, 1)]:
        group = [r for r in good if lo < r['metrics']['crop128']['prevalence'] <= hi]
        result['area_strata'][f'{lo}_{hi}'] = {'parents': len(group), 'image_macro': {
            name: {key: float(np.mean([r['metrics'][name][key] for r in group]))
            for key in ('auc', 'ap', 'iou_at_fixed_half', 'prevalence')}
            for name in names} if group else None}
    return result


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e107_model2_contract.json')['contract_sha256']:
        raise ValueError('Contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('Frozen input differs')
    model = joblib.load(MODEL); start = time.monotonic(); rows = []
    with threadpool_limits(limits=2):
        for index, row in enumerate(c['selected_rows']):
            out = {'index': index, 'parent_sha256': row['auth_sha256']}
            try:
                path = Path(row['image']); image = decode(path, row['image_sha256'])
                auth = decode(LEGACY_DATA_ROOT/'manipulation_test'/row['auth'], row['auth_sha256'])
                mask_path = path.with_suffix('.mask.png')
                if digest(mask_path) != row['mask_sha256']:
                    raise ValueError('Mask hash differs')
                with Image.open(mask_path) as mask:
                    mask.load(); array = np.asarray(mask)
                    if mask.size != image.size or array.ndim != 2 or not set(np.unique(array)).issubset({0, 255}):
                        raise ValueError('Mask encoding/geometry differs')
                    truth = array > 0
                prediction, tiles = score_image(model, image)
                auth_prediction, auth_tiles = score_image(model, auth)
                maps = {'crop128': prediction} | baselines(image.size, row['auth_sha256'])
                out.update(metrics={name: metrics(truth, score) for name, score in maps.items()},
                    edit_p95=float(np.quantile(prediction, .95)), auth_p95=float(np.quantile(auth_prediction, .95)),
                    auth_flagged_area=float(np.mean(auth_prediction >= .5)), tiles=tiles+auth_tiles)
            except (OSError, ValueError, SyntaxError) as exc:
                out['error'] = type(exc).__name__+': '+str(exc)
            rows.append(out)
            if (index+1) % 10 == 0:
                print(json.dumps({'E107_attempted_pairs': index+1, 'seconds': round(time.monotonic()-start)}), flush=True)
    detailed = {'rows': rows, 'contract_sha256': digest(CONTRACT)}
    write_once(ROOT/'rows.json', detailed)
    result = {'state': 'E107_exposed_Model2_replay_complete', 'contract_sha256': digest(CONTRACT),
        'rows_sha256': digest(ROOT/'rows.json'), 'results': summarize(rows),
        'failed_indices': [r['index'] for r in rows if 'error' in r],
        'seconds': time.monotonic()-start, 'limits': c['limits'], 'serving_changed': False,
        'independent_final_passed': False}
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e107_model2_replay.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
