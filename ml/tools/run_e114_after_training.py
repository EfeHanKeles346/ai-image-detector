"""One dependent DEV job: wait for the existing E112/E113 run, preserve its guards."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e112_e113 as lifecycle

ROOT = Path(os.environ['PIXELPROOF_DATA_ROOT'])
RUN = ROOT/'e114_pipeline'


def status(state, **fields):
    temp = RUN/'status.json.part'
    temp.write_text(json.dumps({'state': state, **fields}, indent=2)+'\n')
    temp.replace(RUN/'status.json')


def wait_training(deadline):
    path = ROOT/'e112_pipeline/status.json'
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
    RUN.mkdir(exist_ok=True); lifecycle.RUN = RUN
    with (RUN/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('waiting_for_E112_E113')
        stopped = False; failed = None; skipped = False
        try:
            wait_training(time.monotonic()+21600)
            from experiments.e114_context_development import eligible
            fit = json.loads((ROOT/'e113/report.json').read_text())
            # Complete reports with no passing branch are an expected scientific stop.
            if fit.get('state') == 'E113_paired_TRAIN_ablation_complete' and \
                    all(r.get('dev_scoring_permitted') is False for r in fit['reports'].values()) and len(fit['reports']) == 2:
                skipped = True
                lifecycle.journal('E114 skipped: both complete E113 branches failed TRAIN permission. No DEV metadata/pixels scored and no API restart required.')
            else:
                branches = eligible(fit)
                lifecycle.run_stage('experiments.e114_context_development', 'freeze')
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
                lifecycle.journal('Started separately registered E114 consumed-DEV scoring for '+', '.join(branches)+
                    '. Frozen640-view comparison; no training or final/gallery access.')
                lifecycle.run_stage('experiments.e114_context_development', 'score')
                lifecycle.run_stage('experiments.e114_context_development', 'report')
        except BaseException as exc:
            failed = type(exc).__name__+': '+str(exc)
            lifecycle.journal('Dependent E114 pipeline stopped: '+failed+' No gate relaxation or automatic promotion.')
        finally:
            restored = lifecycle.restore() if stopped else None
            status('failed' if failed else ('skipped_train_failed' if skipped else 'complete'),
                   failure=failed, E92_restored_after_DEV=restored)
            if stopped: lifecycle.journal('E114 finalizer E92 restoration ready='+str(restored)+'.')
        if failed or (stopped and not restored): raise SystemExit(1)


if __name__ == '__main__':
    main()
