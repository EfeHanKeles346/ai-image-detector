"""One dependent local Model2 pilot after the registered Model1 pipeline releases GPU."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e124_e125 as lifecycle
from run_e126_after_training import ready_e92

ROOT = Path(os.environ['PIXELPROOF_DATA_ROOT']); RUN = ROOT/'e129_pipeline'


def ready(first, second):
    if first.get('state') == 'failed' or second.get('state') == 'failed':
        raise RuntimeError('Model1 execution failure requires review before the dependent pilot')
    if first.get('state') != 'complete': return False
    if first.get('E92_restored') is not True: raise RuntimeError('Model1 API restoration failed')
    if second.get('state') == 'skipped_train_failed': return True
    if second.get('state') != 'complete': return False
    if second.get('E92_restored_after_DEV') is not True: raise RuntimeError('DEV API restoration failed')
    return True


def journal(message):
    text = '\n### E129 one-shot execution — '+datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')+'\n\n'+message+'\n'
    for path in (lifecycle.REPO/'HISTORY.md', lifecycle.REPO/'ml/EXPERIMENTS.md'):
        with path.open('a') as f: fcntl.flock(f, fcntl.LOCK_EX); f.write(text)
    print(message, flush=True)


def main():
    RUN.mkdir(exist_ok=True); lifecycle.RUN = RUN
    with (RUN/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (RUN/'status.json').write_text(json.dumps({'state': 'waiting_for_Model1_GPU_release'})+'\n')
        stopped = False; failure = None
        try:
            deadline = time.monotonic()+21600
            while True:
                if time.monotonic() >= deadline: raise TimeoutError('Bounded dependency wait expired')
                try:
                    first = json.loads((ROOT/'e124_pipeline/status.json').read_text())
                    second = json.loads((ROOT/'e126_pipeline/status.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError): time.sleep(5); continue
                if ready(first, second): break
                time.sleep(10)
            from experiments.e65_acquisition import digest, read
            contract_path = ROOT/'e129/contract.json'
            if digest(contract_path) != read(lifecycle.REPO/'evidence/e129_context_replay_contract.json')['contract_sha256']:
                raise ValueError('E129 contract differs')
            for path, sha in read(contract_path)['inputs'].items():
                if digest(path) != sha: raise ValueError('Bound E129 input differs')
            if not (ROOT/'e118/download.json').exists(): raise RuntimeError('Pinned generator assets incomplete')
            current = lifecycle.health()
            if not ready_e92(current):
                raise RuntimeError('Expected restored E92 before pilot')
            if 'AC Power' not in subprocess.check_output(['pmset', '-g', 'batt'], text=True):
                raise RuntimeError('AC power required')
            output = subprocess.check_output(['lsof', '-t', '-iTCP:8800', '-sTCP:LISTEN'], text=True)
            pids = sorted(set(int(p) for p in output.split()))
            if len(pids) != 1: raise RuntimeError('Expected one local API listener')
            pid = pids[0]
            command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
            if '-m uvicorn pixelproof.internship_serve:app' not in command or '--port 8800' not in command:
                raise RuntimeError('Refusing unrelated listener')
            os.kill(pid, signal.SIGTERM); stopped = True
            for _ in range(30):
                if lifecycle.health() is None: break
                time.sleep(1)
            if lifecycle.health() is not None: raise RuntimeError('E92 did not release API')
            journal('Starting registered same16-parent E129 resized-context inpainting ablation after Model1 completion. No detector fitting or evaluation data. E92 is temporarily stopped for GPU memory.')
            lifecycle.run_stage('experiments.e129_inpaint_context_replay', 'replay')
        except BaseException as exc:
            failure = type(exc).__name__+': '+str(exc)
            journal('E129 dependent execution stopped: '+failure+' No automatic reroll or changed contract.')
        finally:
            restored = None
            if stopped:
                previous_path = os.environ.get('PYTHONPATH')
                os.environ['PYTHONPATH'] = 'ml:ml/src'
                try:
                    restored = lifecycle.restore() and ready_e92(lifecycle.health())
                finally:
                    if previous_path is None: os.environ.pop('PYTHONPATH', None)
                    else: os.environ['PYTHONPATH'] = previous_path
            result = {'state': 'failed' if failure or (stopped and not restored) else 'complete',
                'failure': failure, 'E92_restored_after_pilot': restored}
            (RUN/'status.json').write_text(json.dumps(result, indent=2)+'\n')
            journal('E129 pipeline final state: '+json.dumps(result, sort_keys=True))
        if result['state'] == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()
