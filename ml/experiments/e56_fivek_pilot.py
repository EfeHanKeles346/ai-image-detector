"""Twelve score-blind RAW parents: bounded download, isolated decode, protected audit."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import certifi
import requests

from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e55_resume import inspect_safety
from experiments.e51_prefit_audit import fingerprint, cross_role_matches
from experiments.e54_reserve_reference import SNAPSHOT
from experiments.e54_mnw_reserve import ROOT as MNW
from experiments.e54_hdrplus_reserve import ROOT as HDR
from experiments.e56_fivek_metadata import ROOT as META, BASE

ROOT = META.parent/'fivek_pilot_v1'
CONTRACT = ROOT/'contract.json'
DECODER = Path(__file__).with_name('e56_raw_decode.py')
ENV = EVIDENCE.parent/'ml/work/e56_decoder'
PYTHON = ENV/'bin/python'
MAX_IMAGE = 32*1024**2
SUBJECTS = {'nature', 'person(s)', 'man-made object', 'animal(s)', 'unknown', 'abstract'}


def select(rows):
    if len({r['filename'] for r in rows}) != len(rows): raise ValueError('duplicate parent')
    groups = defaultdict(list)
    for row in rows: groups[row['subject']].append(row)
    if set(groups) != SUBJECTS or min(map(len, groups.values())) < 2:
        raise ValueError('incomplete subject strata')
    return [r for subject in sorted(groups) for r in sorted(groups[subject],
            key=lambda r: hashlib.sha256(('FIVEK_E56_PILOT_V1|'+r['filename']).encode()).hexdigest())[:2]]


def header_identity(headers):
    size = int(headers.get('Content-Length', '0'))
    etag, modified = headers.get('ETag'), headers.get('Last-Modified')
    if not 0 < size <= MAX_IMAGE or not (etag or modified):
        raise ValueError('missing HTTP identity or image exceeds pilot cap')
    return {'bytes': size, 'etag': etag, 'last_modified': modified}


def safe(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('bounded pilot deadline reached')
    if inspect_safety(): raise RuntimeError('AC/external disk safety failed')


def decoder(args, deadline):
    safe(deadline)
    child = subprocess.Popen([str(PYTHON), str(DECODER), *map(str, args)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    try:
        while child.poll() is None:
            safe(deadline); time.sleep(2)
        out, err = child.communicate()
        if child.returncode: raise RuntimeError('isolated decoder failed: '+err[-2000:])
        return json.loads(out)
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


def freeze():
    deadline = time.monotonic()+1800; safe(deadline)
    metadata = META/'manifest_v2.json'
    if digest(metadata) != json.loads((EVIDENCE/'e56_fivek_metadata.json').read_text())['manifest_sha256']:
        raise ValueError('metadata changed')
    runtime = decoder(['--versions'], deadline)
    if (runtime['rawpy'], runtime['numpy'], runtime['pillow']) != ('0.27.1', '2.5.1', '12.3.0'):
        raise ValueError('unexpected decoder versions')
    rows = select(json.loads(metadata.read_text())['rows']); selected = []
    for row in rows:
        safe(deadline)
        with requests.head(row['original_url'], timeout=(10, 30), allow_redirects=True,
                           headers={'Accept-Encoding': 'identity'}, verify=certifi.where()) as response:
            response.raise_for_status()
            if not response.url.startswith(BASE): raise ValueError('off-source redirect')
            identity = header_identity(response.headers)
        selected.append(row | identity | {'parent_id': 'FiveK:'+row['filename'], 'label': 0,
                        'role': 'QUARANTINED_TRAIN_CANDIDATE', 'source': 'MIT-Adobe-FiveK'})
    inputs = [metadata, ENV/'install_report.json', DECODER,
              Path(__file__).with_name('e51_prefit_audit.py'), Path(__file__).with_name('e55_resume.py')]
    refs = [(SNAPSHOT, 'e54_reserve_reference.json', 'snapshot_sha256'),
            (MNW/'download.json', 'e54_mnw_reserve_download.json', 'report_sha256'),
            (HDR/'download.json', 'e54_hdrplus_download.json', 'report_sha256')]
    for path, receipt, key in refs:
        if digest(path) != json.loads((EVIDENCE/receipt).read_text())[key]: raise ValueError('protected reference changed')
        inputs.append(path)
    inputs += [META/'LicenseAdobe.txt', META/'LicenseAdobeMIT.txt']
    config = {'state': 'FiveK_pilot_frozen_before_image_GET', 'code_sha256': digest(__file__),
              'inputs': {str(p): digest(p) for p in inputs}, 'rows': selected, 'runtime': runtime,
              'image_bytes_expected': sum(r['bytes'] for r in selected), 'image_cap': MAX_IMAGE,
              'total_image_cap': 12*MAX_IMAGE, 'max_runtime_seconds': 1800,
              'recipe': 'AHD/full resolution/camera WB/auto WB off/sRGB/gamma 2.4,12.92/auto bright off/bright 1/clip/8-bit; default RAW orientation; no added denoise.',
              'source_integrity_limit': 'HTTP validators and local SHA; no publisher per-file cryptographic checksum.',
              'model_scores': 0, 'training_admitted': False}
    fixed_write(CONTRACT, config)
    fixed_write(EVIDENCE/'e56_fivek_pilot_contract.json', config | {'contract_sha256': digest(CONTRACT)})
    return {'parents': len(selected), 'bytes': config['image_bytes_expected'], 'runtime': runtime}


def load():
    config = json.loads(CONTRACT.read_text())
    if config['code_sha256'] != digest(__file__) or digest(CONTRACT) != json.loads((EVIDENCE/'e56_fivek_pilot_contract.json').read_text())['contract_sha256']:
        raise ValueError('frozen pilot changed')
    for path, expected in config['inputs'].items():
        if digest(path) != expected: raise ValueError('pilot input changed: '+path)
    return config


def download():
    config = load(); deadline = time.monotonic()+config['max_runtime_seconds']; safe(deadline)
    for name in ('LicenseAdobe.txt', 'LicenseAdobeMIT.txt'):
        target = ROOT/name
        if not target.exists(): target.write_bytes((META/name).read_bytes())
        if digest(target) != config['inputs'][str(META/name)]: raise ValueError('licence copy mismatch')
    records = []
    for row in config['rows']:
        safe(deadline); target = ROOT/row['filename']; receipt = target.with_suffix('.receipt.json')
        part = target.with_suffix('.dng.part')
        if not target.exists() and part.exists() and part.stat().st_size == row['bytes']:
            saved = {'sha256': digest(part), 'bytes': row['bytes'], 'contract_sha256': digest(CONTRACT)}
            fixed_write(receipt, saved); part.replace(target)
        if target.exists():
            if not receipt.exists(): raise ValueError('unreceipted complete raw; keep quarantined')
            saved = json.loads(receipt.read_text())
            if saved['contract_sha256'] != digest(CONTRACT) or target.stat().st_size != row['bytes'] or digest(target) != saved['sha256']:
                raise ValueError('saved original changed')
        else:
            part = target.with_suffix('.dng.part'); offset = part.stat().st_size if part.exists() else 0
            headers = {'Accept-Encoding': 'identity'}
            if row['etag']: headers['If-Match'] = row['etag']
            elif row['last_modified']: headers['If-Unmodified-Since'] = row['last_modified']
            if offset: headers['Range'] = f'bytes={offset}-'
            with requests.get(row['original_url'], headers=headers, stream=True, timeout=(10, 30), verify=certifi.where()) as response:
                response.raise_for_status()
                if not response.url.startswith(BASE): raise ValueError('unexpected image redirect')
                if offset and (response.status_code != 206 or response.headers.get('Content-Range') != f"bytes {offset}-{row['bytes']-1}/{row['bytes']}"):
                    raise ValueError('server did not honor exact resume range')
                for key, field in (('ETag', 'etag'), ('Last-Modified', 'last_modified')):
                    if row[field] and response.headers.get(key) != row[field]: raise ValueError('HTTP source changed')
                if int(response.headers.get('Content-Length', '0')) != row['bytes']-offset:
                    raise ValueError('image length changed')
                checked = time.monotonic()
                with part.open('ab' if part.exists() else 'xb') as stream:
                    for chunk in response.iter_content(65536):
                        offset += len(chunk)
                        if offset > row['bytes']: raise ValueError('image cap exceeded')
                        if time.monotonic()-checked >= 5: safe(deadline); checked = time.monotonic()
                        stream.write(chunk)
                    stream.flush(); os.fsync(stream.fileno())
            if offset != row['bytes']: raise ValueError('truncated image')
            saved = {'sha256': digest(part), 'bytes': offset, 'contract_sha256': digest(CONTRACT)}
            fixed_write(receipt, saved); part.replace(target)
        png = target.with_suffix('.png')
        result = decoder([target, png], deadline)
        if result['versions'] != config['runtime']: raise ValueError('decoder runtime changed')
        facts = fingerprint(png.read_bytes())
        if facts['pixel_sha256'] != result['pixel_sha256']: raise ValueError('decoded canonical RGB mismatch')
        record = row | facts | {'path': str(png), 'condition': 'fixed_raw_development',
                   'original_sha256': saved['sha256'], 'original_bytes': row['bytes'], 'decode': result}
        records.append(record)
        fixed_write(target.with_suffix('.decoded.json'), record)
        print(json.dumps({'completed': len(records), 'total': len(config['rows'])}), flush=True)
    result = {'state': 'FiveK_pilot_decoded_quarantine', 'contract_sha256': digest(CONTRACT), 'records': records,
              'original_bytes': sum(r['original_bytes'] for r in records), 'png_bytes': sum(r['bytes'] for r in records),
              'model_scores': 0, 'training_admitted': False}
    fixed_write(ROOT/'download.json', result)
    fixed_write(EVIDENCE/'e56_fivek_pilot_download.json', {k:v for k,v in result.items() if k != 'records'} |
                {'parents': len(records), 'report_sha256': digest(ROOT/'download.json')})
    return {k:v for k,v in result.items() if k != 'records'}


def audit():
    load(); safe(time.monotonic()+1800)
    data = ROOT/'download.json'
    if digest(data) != json.loads((EVIDENCE/'e56_fivek_pilot_download.json').read_text())['report_sha256']:
        raise ValueError('pilot download changed')
    rows = json.loads(data.read_text())['records']
    for row in rows:
        if digest(row['path']) != row['sha256'] or digest(ROOT/row['filename']) != row['original_sha256']:
            raise ValueError('pilot bodies changed')
    refs = json.loads(SNAPSHOT.read_text())['records']
    refs += json.loads((MNW/'download.json').read_text())['records']
    refs += json.loads((HDR/'download.json').read_text())['records']
    cross = cross_role_matches(rows, refs)
    internal = [r for r in cross_role_matches(rows, rows) if r['train_parent'] < r['cal_parent']]
    raw_sha = {r['original_sha256'] for r in rows} & {r['sha256'] for r in refs}
    result = {'state': 'FiveK_pilot_blind_screen_complete', 'contract_sha256': digest(CONTRACT),
              'download_sha256': digest(data), 'parents': len(rows), 'reference_observations': len(refs),
              'cross_matches': cross, 'internal_pairs': internal, 'raw_exact_matches': sorted(raw_sha),
              'screen_pass': not (cross or internal or raw_sha), 'training_admitted': False, 'model_scores': 0,
              'limitations': ['Heuristic screen is not exhaustive semantic/scene deduplication.',
                             'One fixed RAW rendition can miss similarity to other edits.',
                             'Passing a 12-parent pilot proves neither detector quality nor camera coverage.']}
    fixed_write(ROOT/'audit.json', result); fixed_write(EVIDENCE/'e56_fivek_pilot_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'download', 'audit'])
    args = parser.parse_args(); print(json.dumps({'freeze': freeze, 'download': download, 'audit': audit}[args.phase](), indent=2))
