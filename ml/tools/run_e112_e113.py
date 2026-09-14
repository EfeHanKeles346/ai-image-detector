"""One unattended local extraction/paired-fit run, with logs and E92 restoration."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ['PIXELPROOF_DATA_ROOT'])
RUN = ROOT/'e112_pipeline'


def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=3) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def journal(text):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    for path in [REPO/'HISTORY.md', REPO/'ml/EXPERIMENTS.md']:
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f'\n### E112/E113 unattended execution — {stamp}\n\n{text}\n')
    print(text, flush=True)


def run_stage(module, stage):
    name = module.split('.')[-1]+'_'+stage
    log = RUN/(name+'.log')
    with log.open('a') as f:
        p = subprocess.Popen([sys.executable, '-u', '-m', module, stage], cwd=REPO,
            env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (RUN/'status.json').write_text(json.dumps({'stage': name, 'pid': p.pid,
            'log': str(log), 'state': 'running'}, indent=2)+'\n')
        try:
            code = p.wait()
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=30)
            raise
    if code:
        raise RuntimeError(f'{name} exited {code}; see local pipeline log. No downstream stage started.')


def restore():
    current = health()
    if current is not None:
        return current.get('status') == 'ready' and current.get('model_id') == 'E92'
    env = os.environ.copy()
    env['PIXELPROOF_CORS_ORIGINS'] = 'http://localhost:3002,http://127.0.0.1:3002'
    with (RUN/'restored_api.log').open('a') as f:
        p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'pixelproof.internship_serve:app',
            '--host', '127.0.0.1', '--port', '8800'], cwd=REPO, env=env, stdout=f,
            stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(40):
        current = health()
        if current and current.get('status') == 'ready' and current.get('model_id') == 'E92':
            return True
        if p.poll() is not None: return False
        time.sleep(1)
    return False


def main(api_pid):
    RUN.mkdir(exist_ok=True)
    with (RUN/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = subprocess.check_output(['ps', '-p', str(api_pid), '-o', 'command='], text=True).strip()
        if '-m uvicorn pixelproof.internship_serve:app' not in command or '--port 8800' not in command:
            raise ValueError('Refusing to stop an unrelated process')
        if 'AC Power' not in subprocess.check_output(['pmset', '-g', 'batt'], text=True):
            raise ValueError('AC power required before stopping API')
        os.kill(api_pid, signal.SIGTERM)
        for _ in range(30):
            if health() is None: break
            time.sleep(1)
        failed = None
        try:
            journal('Started the fixed full E112 extraction; temporarily stopped the verified E92 API for GPU memory. Durable chunks and stage logs are local. This is feature extraction, not a completed model improvement.')
            run_stage('experiments.e112_context_features', 'extract')
            result = json.loads((ROOT/'e112/report.json').read_text())
            journal(f"E112 completed {result['parents']} parents / {result['views']} TRAIN views; maximum aggregate error {result['max_aggregate_error']}. Registering the prespecified paired E113 fit; no DEV/final opened.")
            run_stage('experiments.e113_context_fit', 'freeze')
            run_stage('experiments.e113_context_fit', 'fit')
            result = json.loads((ROOT/'e113/report.json').read_text())
            journal('E113 paired TRAIN fits completed. '+json.dumps(result['summary'], sort_keys=True)+
                ' These are TRAIN guards only; serving/promotion and independent evaluation remain unchanged. Full reports are in evidence/e113_context_fit.json.')
        except BaseException as exc:
            failed = type(exc).__name__+': '+str(exc)
            journal('Pipeline stopped: '+failed+' Completed immutable evidence is retained; no automatic quality-gate relaxation or retry.')
        finally:
            restored = restore()
            status = {'state': 'failed' if failed else 'complete', 'failure': failed, 'E92_restored': restored}
            (RUN/'status.json').write_text(json.dumps(status, indent=2)+'\n')
            journal('Pipeline final state: '+json.dumps(status, sort_keys=True))
        if failed or not restored: raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--api-pid', type=int, required=True)
    main(parser.parse_args().api_pid)
