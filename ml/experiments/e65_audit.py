"""E65 score-blind decode and identity audit; no historical acquisition imports."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from experiments.e65_acquisition import ROOT, EVIDENCE, RECEIPT, digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

VERSION = 'e51-prefit-rgb-lanczos-v1'
CONTRACT = ROOT / 'audit_contract.json'
REPORT = ROOT / 'audit.json'
DECODER = ML_ROOT / 'experiments/e56_raw_decode.py'
PYTHON = ML_ROOT / 'work/e56_decoder/bin/python'
REFERENCES = {
    'e52/protected_reference_v1.json': '7fc7f52e8c38e621b71877b394cccac312af333b183492bcf52154381e3ab988',
    'e52/mnw_reserve_v1/download.json': 'a63d7655cea11d18457dbb45bb4419063526e16e90f6d941a2796af3dd17ea4c',
    'e52/hdrplus_reserve_v1/download.json': '95cdc842844443497af26995fb8430cf39540fa33e093b91a63be6fce768ff18',
    'e57/v2/download.json': '879f3d76255bb7b26b9e2d560a1f5d15096e0ac3fc6172662c0ec8d504635010',
}


def verified(path, sha):
    if digest(path) != sha:
        raise ValueError(f'bound file changed: {path}')
    return Path(path)


def decoder_versions():
    return json.loads(subprocess.check_output([str(PYTHON), str(DECODER), '--versions'], text=True, timeout=30))


def propagate_quarantine(rows, rejected):
    """A RAW pair is indivisible; WIFD has no known scene linkage."""
    bad = set(rejected)
    scenes = {r['scene_group'] for r in rows if r['source'] == 'RawNIND' and r['parent_id'] in bad}
    return bad | {r['parent_id'] for r in rows if r['source'] == 'RawNIND' and r['scene_group'] in scenes}


def freeze():
    verified(RECEIPT, read(EVIDENCE / 'e65_download.json')['receipt_sha256'])
    inputs = {str(p): digest(p) for p in (Path(__file__), DECODER, Path(__file__).with_name('e65_acquisition.py'), RECEIPT)}
    for path, sha in REFERENCES.items():
        verified(DATA_ROOT / path, sha); inputs[str(DATA_ROOT / path)] = sha
    c = {'state': 'E65_score_blind_decode_overlap_registered', 'inputs': inputs,
         'decoder_versions': decoder_versions(), 'fingerprint_version': VERSION,
         'policy': 'Quarantine decode failures and any exact RGB/body or dHash<=4 AND pHash63<=4 '
                   'cross-reference matches. Propagate RawNIND failures to both scene members. '
                   'Internal similarity is reported with linked scene identities, not independent evidence; '
                   'keep internally similar observations for descriptive repeated-view diagnostics only. '
                   'No score-based or brightness exclusions; no replacement draws.',
         'limits': 'Snapshot coverage is bounded, perceptual screen is not exhaustive; WIFD scene identities '
                   'unknown. X-Trans processing may be selected internally by LibRaw; no universal Bayer claim.',
         'max_pixels': 100000000, 'raw_timeout_seconds': 120,
         'training_allowed': False, 'model_scoring_allowed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e65_audit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'references': len(REFERENCES)}


def validate():
    verified(CONTRACT, read(EVIDENCE / 'e65_audit_contract.json')['contract_sha256'])
    c = read(CONTRACT)
    for p, sha in c['inputs'].items(): verified(p, sha)
    if decoder_versions() != c['decoder_versions']: raise ValueError('decoder runtime changed')
    return c


def audit():
    c = validate()
    if REPORT.exists(): raise FileExistsError('audit already frozen')
    rows = read(RECEIPT)['rows']; decoded = []; failed = []
    (ROOT / 'decoded').mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        verified(r['path'], r['sha256'])  # identity corruption is fatal, not a selectable exclusion.
        try:
            p = Path(r['path']); detail = {}; original_sha = r['sha256']
            if r['source'] == 'RawNIND':
                p = ROOT / 'decoded' / (r['parent_id'].split(':')[1] + '.png')
                result = subprocess.run([str(PYTHON), str(DECODER), r['path'], str(p)],
                    text=True, capture_output=True, timeout=c['raw_timeout_seconds'],
                    env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                if result.returncode: raise ValueError(result.stderr[-1500:])
                detail['decode'] = json.loads(result.stdout)
            else:
                with Image.open(p) as im:
                    exif = im.getexif()
                    detail['exif'] = {str(k): str(exif[k]) for k in (271, 272, 274, 34855) if k in exif}
            facts = fingerprint(p.read_bytes(), max_pixels=c['max_pixels'])
            if min(facts['width'], facts['height']) < 224: raise ValueError('below224px')
            decoded.append(r | {'original_path': r['path'], 'original_sha256': original_sha,
                'path': str(p), **facts, **detail})
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            failed.append({'parent_id': r['parent_id'], 'reason': str(error)})
        if (i+1)%10 == 0 or i+1 == len(rows):
            print(json.dumps({'decoded': len(decoded), 'failed': len(failed), 'processed': i+1}), flush=True)
    refs = []
    for p in REFERENCES: refs.extend(read(DATA_ROOT / p)['records'])
    if any(r.get('version') != VERSION for r in refs): raise ValueError('reference hash convention differs')
    matches = cross_role_matches(decoded, refs)
    rejected = {r['parent_id'] for r in failed} | {r['train_parent'] for r in matches}
    rejected = propagate_quarantine(rows, rejected)
    internal = [r for r in cross_role_matches(decoded, decoded) if r['train_parent'] < r['cal_parent']]
    admitted = [r for r in decoded if r['parent_id'] not in rejected]
    result = {'state': 'E65_decoded_audited_unscored', 'contract_sha256': digest(CONTRACT),
        'records': admitted, 'decode_failures': failed, 'quarantined_parents': sorted(rejected),
        'reference_records': len(refs), 'cross_matches': matches, 'internal_matches': internal,
        'diagnostic_observations': len(admitted), 'decoded': len(decoded),
        'source_counts': dict(Counter(r['source'] for r in admitted)),
        'rawnind_scene_groups': len({r['scene_group'] for r in admitted if r['source'] == 'RawNIND'}),
        'model_scores_created': 0, 'training_allowed': False, 'balanced_final_admitted': False,
        'limitations': c['limits']}
    write_once(REPORT, result)
    summary = {k:v for k,v in result.items() if k not in ('records','cross_matches','internal_matches')}
    summary.update(report_sha256=digest(REPORT), cross_match_pairs=len(matches), internal_match_pairs=len(internal))
    write_once(EVIDENCE / 'e65_audit.json', summary)
    return summary


# Canonical E51 functions copied below verbatim; avoids historical import-time acquisition side effects.

def _phash_image(image: Image.Image) -> int:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    coefficients = dctn(pixels, norm="ortho")[:8, :8].ravel()
    bits = coefficients[1:] > np.median(coefficients[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def fingerprint(raw: bytes, *, max_pixels: int = 100_000_000) -> dict[str, Any]:
    """Use a single explicit convention; ties are zero, not inverse-complemented."""
    with Image.open(BytesIO(raw)) as image:
        if image.width * image.height > max_pixels:
            raise ValueError("image exceeds pre-decode pixel budget")
        image.verify()
    with Image.open(BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        grey = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        value = 0
        for bit in (grey[:, 1:] > grey[:, :-1]).ravel():
            value = (value << 1) | int(bit)
        pixels = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
        pixels.update(rgb.tobytes())
        return {
            "version": VERSION,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "width": rgb.width, "height": rgb.height,
            "pixel_sha256": pixels.hexdigest(),
            "canonical_dhash": f"{value:016x}",
            "phash63": f"{_phash_image(rgb):016x}",
        }

def cross_role_matches(train: Sequence[Mapping], cal: Sequence[Mapping]) -> list[dict]:
    """Vectorized radius-4 dHash screening, fixed pHash confirmation; not exhaustive similarity."""
    right_hashes = np.array([int(r["canonical_dhash"], 16) for r in cal], dtype=np.uint64)
    matches = {}
    exact_indexes = {}
    for field in ("sha256", "pixel_sha256"):
        index: dict[str, list[int]] = defaultdict(list)
        for j, row in enumerate(cal):
            index[row[field]].append(j)
        exact_indexes[field] = index
    for left in train:
        distances = np.bitwise_count(right_hashes ^ np.uint64(int(left["canonical_dhash"], 16)))
        candidates = set(np.flatnonzero(distances <= 4).tolist())
        for field, index in exact_indexes.items():
            candidates.update(index.get(left[field], []))
        for j in sorted(candidates):
            right = cal[j]
            exact = [field for field in exact_indexes if left[field] == right[field]]
            phash_distance = (int(left["phash63"], 16) ^ int(right["phash63"], 16)).bit_count()
            if not exact and phash_distance > 4:
                continue
            key = (left["parent_id"], right["parent_id"])
            detail = {"train_parent": key[0], "cal_parent": key[1],
                      "train_label": left["label"], "cal_label": right["label"],
                      "cal_condition": right["condition"], "exact_fields": exact,
                      "dhash_distance": int(distances[j]), "phash63_distance": phash_distance}
            # Preserve every paired condition; do not count two transports as two parents.
            matches.setdefault(key, []).append(detail)
    return [item for key in sorted(matches) for item in matches[key]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'audit'))
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
