"""Fixed16-parent resized-context inpainting replay; preserve E122 parameters and gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from experiments import e128_inpaint_context_inputs as inputs, e120_inpaint_numerics as numerics
from experiments import e119_inpainting_pilot as original, e118_inpaint_assets as assets
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e129'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'


def paired_outcomes(previous, current):
    if len(previous) != 16 or len(current) != 16 or \
            [r['index'] for r in previous] != list(range(16)) or [r['index'] for r in current] != list(range(16)):
        raise ValueError('Complete paired16 attempts required')
    for before, after in zip(previous, current, strict=True):
        if before['seed'] != after['seed'] or before['branch'] != after['branch'] or \
                type(before['passed']) is not bool or type(after['passed']) is not bool:
            raise ValueError('Paired seed/branch/acceptance identity differs')
    return {'previous_passed': sum(r['passed'] for r in previous), 'context_passed': sum(r['passed'] for r in current),
        'rescued_attempts': sum(not a['passed'] and b['passed'] for a,b in zip(previous, current, strict=True)),
        'new_failed_attempts': sum(a['passed'] and not b['passed'] for a,b in zip(previous, current, strict=True)),
        'full16_engineering_gate_passed': all(r['passed'] for r in current)}


def freeze():
    source = inputs.validate(); prior = numerics.validate(); assets.validate()
    prepared = inputs.ROOT/'prepared.json'; public = read(EVIDENCE/'e128_context_inputs.json')
    if digest(prepared) != public['prepared_sha256'] or public['parents'] != 16 or \
            public['state'] != 'E128_same_parent_context_inputs_complete':
        raise ValueError('Complete verified context preparation required')
    rows = read(prepared)['rows']; parents = read(original.CONTRACT)['rows']
    inputs.validate_population(parents, rows)
    previous = DATA_ROOT/'e122/report.json'
    if digest(previous) != read(EVIDENCE/'e122_inpaint_replay.json')['report_sha256']:
        raise ValueError('Frozen E122 comparator differs')
    files = [Path(__file__), Path(inputs.__file__), Path(numerics.__file__), inputs.CONTRACT,
        prepared, previous, EVIDENCE/'e128_context_inputs.json']
    for row in rows:
        if row['preparation_contract_sha256'] != digest(inputs.CONTRACT) or row['rendering'] != 'center_square_resized_512':
            raise ValueError('Context rendering differs')
        for entry in row['files'].values():
            path = Path(entry['path'])
            if digest(path) != entry['sha256']: raise ValueError('Context input body differs')
            files.append(path)
    c = {'state': 'E129_same_parent_context_generation_registered',
        'inputs': prior['inputs'] | source['inputs'] | {str(p): digest(p) for p in files},
        'parents': 16, 'branch': 'fp16_sdpa', 'max_seconds': 2400, 'mps_limit_bytes': 10*1024**3,
        'change': 'Replace E122 native512 source crops with E128 resized center-square context inputs for all16 unchanged original parents. No parent, seed, mask, prompt, weight, checker or generation-parameter change.',
        'generation': read(original.CONTRACT)['generation'] | {'attention_slicing': False, 'attention': 'SDPA', 'safety_checker': True},
        'acceptance': 'Complete16/16 original E122 engineering gate: finite tensors, nonblank outputs, enabled safety checker negative, >=95% intended-mask pixels changed, exact composite background. No output replacements, rerolls or threshold relaxation.',
        'limits': 'TRAIN research engineering, one old generator and correlated MIDD parents. Resampling and scene context change together. A pass is not semantic-quality, localization-accuracy or independent-OOD evidence; a new spatial-learning contract remains necessary.',
        'downloads': 0, 'detector_scores': 0, 'training_admission': False, 'promotion_allowed': False}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e129_context_replay_contract.json', {k:v for k,v in c.items() if k != 'inputs'} |
        {'contract_sha256': digest(CONTRACT)})
    return {'parents': 16, 'branch': c['branch'], 'training_admission': False}


def replay():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e129_context_replay_contract.json')['contract_sha256']:
        raise ValueError('Context generation contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Bound context generation input differs')
    original.validate()
    if (ROOT/'report.json').exists(): raise FileExistsError('Context generation already completed')
    for row in read(assets.CONTRACT)['files']:
        assets.verify(assets.ROOT/'model'/row['rfilename'], row)
    start = time.monotonic()
    rows, peak = numerics.run_cases(c['branch'], read(inputs.ROOT/'prepared.json')['rows'], ROOT,
        start+c['max_seconds'], c['mps_limit_bytes'])
    comparison = paired_outcomes(read(DATA_ROOT/'e122/report.json')['rows'], rows)
    result = {'state': 'E129_context_generation_complete', 'contract_sha256': digest(CONTRACT),
        'rows': rows, 'parents': len(rows), 'comparison': comparison, 'seconds': time.monotonic()-start,
        'peak_mps_bytes': peak, 'downloads': 0, 'detector_scores': 0, 'training_admission': False,
        'spatial_protocol_may_be_registered': comparison['full16_engineering_gate_passed'],
        'promotion_allowed': False, 'limits': c['limits']}
    write_once(ROOT/'report.json', result)
    public = {k:v for k,v in result.items() if k != 'rows'} | {'report_sha256': digest(ROOT/'report.json')}
    write_once(EVIDENCE/'e129_context_replay.json', public)
    text = '\n### E129 fixed input-context replay completed\n\n'+json.dumps(public, sort_keys=True)+ \
        '\n\nThis is paired generation engineering, not detector accuracy. Preserve E122 and all failures; no automatic training/serving admission.\n'
    for path in (ML_ROOT.parent/'PLAN.md', ML_ROOT.parent/'HISTORY.md', ML_ROOT/'EXPERIMENTS.md', ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(text)
    return public


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def denied(*a, **kw): raise RuntimeError('Context replay is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'replay'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'replay': replay}[parser.parse_args().stage](), indent=2))
