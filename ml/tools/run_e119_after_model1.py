"""One dependent local Model2 pilot after the registered Model1 pipeline releases GPU."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e112_e113 as lifecycle

ROOT = Path(os.environ['PIXELPROOF_DATA_ROOT']); RUN = ROOT/'e119_pipeline'


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
    text = '\n### E119 one-shot execution — '+datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')+'\n\n'+message+'\n'
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
                    first = json.loads((ROOT/'e112_pipeline/status.json').read_text())
                    second = json.loads((ROOT/'e114_pipeline/status.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError): time.sleep(5); continue
                if ready(first, second): break
                time.sleep(10)
            from experiments.e119_inpainting_pilot import validate
            validate()
            if not (ROOT/'e118/download.json').exists(): raise RuntimeError('Pinned generator assets incomplete')
            current = lifecycle.health()
            if current is None or current.get('status') != 'ready' or current.get('model_id') != 'E92':
                raise RuntimeError('Expected restored E92 before pilot')
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
            journal('Starting registered16-parent E119 local inpainting engineering probe after Model1 completion. No detector fitting or evaluation data. E92 is temporarily stopped for GPU memory.')
            lifecycle.run_stage('experiments.e119_inpainting_pilot', 'generate')
        except BaseException as exc:
            failure = type(exc).__name__+': '+str(exc)
            journal('E119 dependent execution stopped: '+failure+' No automatic reroll or changed contract.')
        finally:
            restored = lifecycle.restore() if stopped else None
            result = {'state': 'failed' if failure or (stopped and not restored) else 'complete',
                'failure': failure, 'E92_restored_after_pilot': restored}
            (RUN/'status.json').write_text(json.dumps(result, indent=2)+'\n')
            journal('E119 pipeline final state: '+json.dumps(result, sort_keys=True))
        if result['state'] == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()
