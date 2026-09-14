"""One dependent DEV job: wait for the existing E124/E125 run, preserve its guards."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e124_e125 as lifecycle

ROOT = Path(os.environ['PIXELPROOF_DATA_ROOT'])
RUN = ROOT/'e126_pipeline'
E92_SHA = '3a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35'


def ready_e92(current):
    return bool(current and current.get('status') == 'ready' and
                current.get('model_id') == 'E92' and current.get('artifact_sha256') == E92_SHA)


def journal(text):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    for path in [lifecycle.REPO/'HISTORY.md', lifecycle.REPO/'ml/EXPERIMENTS.md']:
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f'\n### E126 dependent execution — {stamp}\n\n{text}\n')
    print(text, flush=True)


def status(state, **fields):
    temp = RUN/'status.json.part'
    temp.write_text(json.dumps({'state': state, **fields}, indent=2)+'\n')
    temp.replace(RUN/'status.json')


def wait_training(deadline):
    path = ROOT/'e124_pipeline/status.json'
    while time.monotonic() < deadline:
        try:
            old = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5); continue
        if old.get('state') == 'failed':
            raise RuntimeError('Upstream pipeline failed; dependent DEV denied')
        if old.get('state') == 'complete':
            if old.get('E92_restored') is not True:
                raise RuntimeError('Upstream API restoration incomplete; no duplicate lifecycle started')
            return
        time.sleep(10)
    raise TimeoutError('Six-hour bounded dependency wait expired')


def main():
    RUN.mkdir(exist_ok=True); lifecycle.RUN = RUN; lifecycle.journal = journal
    with (RUN/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('waiting_for_E124_E125')
        stopped = False; failed = None; skipped = False
        try:
            wait_training(time.monotonic()+21600)
            from experiments.e126_fullframe_development import eligible
            fit = json.loads((ROOT/'e125/report.json').read_text())
            branches = eligible(fit, allow_none=True)
            if not branches:
                skipped = True
                lifecycle.journal('E126 skipped: both complete E125 branches failed TRAIN permission. No DEV metadata/pixels scored and no API restart required.')
            else:
                lifecycle.run_stage('experiments.e126_fullframe_development', 'freeze')
                current = lifecycle.health()
                if not ready_e92(current):
                    raise RuntimeError('Expected ready E92 before dependent GPU operation')
                if 'AC Power' not in subprocess.check_output(['pmset', '-g', 'batt'], text=True):
                    raise RuntimeError('AC power required before stopping E92')
                output = subprocess.check_output(['lsof', '-t', '-iTCP:8800', '-sTCP:LISTEN'], text=True)
                pids = sorted(set(int(p) for p in output.split()))
                if len(pids) != 1: raise RuntimeError('Expected one local API listener')
                pid = pids[0]
                command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
                if '-m uvicorn pixelproof.internship_serve:app' not in command or '--port 8800' not in command:
                    raise RuntimeError('Refusing to stop unrelated listener')
                os.kill(pid, signal.SIGTERM); stopped = True
                for _ in range(30):
                    if lifecycle.health() is None: break
                    time.sleep(1)
                else:
                    raise RuntimeError('E92 listener did not stop before GPU scoring')
                lifecycle.journal('Started separately registered E126 consumed-DEV scoring for '+', '.join(branches)+
                    '. Frozen640-view comparison; no training or final/gallery access.')
                lifecycle.run_stage('experiments.e126_fullframe_development', 'score')
                lifecycle.run_stage('experiments.e126_fullframe_development', 'report')
        except BaseException as exc:
            failed = type(exc).__name__+': '+str(exc)
            lifecycle.journal('Dependent E126 pipeline stopped: '+failed+' No gate relaxation or automatic promotion.')
        finally:
            restored = lifecycle.restore() if stopped else None
            if restored:
                restored = ready_e92(lifecycle.health())
            status('failed' if failed else ('skipped_train_failed' if skipped else 'complete'),
                   failure=failed, E92_restored_after_DEV=restored)
            if stopped: lifecycle.journal('E126 finalizer E92 restoration ready='+str(restored)+'.')
        if failed or (stopped and not restored): raise SystemExit(1)


if __name__ == '__main__':
    main()
