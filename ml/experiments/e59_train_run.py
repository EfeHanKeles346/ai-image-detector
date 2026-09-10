"""Bounded operational training sequence. Requires completed E59 feature receipts."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.e59_model import ROOT, CONTRACT, RESULT, ARMS, require_features
from experiments.e57_data import safe


def pending():
    require_features()
    steps = []
    if not CONTRACT.exists():
        steps.append('freeze')
    completed = [ROOT / 'training' / f'{a}_fold{f}.json' for a in ARMS for f in range(3)]
    if not all(p.exists() for p in completed):
        steps.append('fit')
    if not RESULT.exists():
        steps.append('report')
    return steps


def run(minutes):
    if not 1 <= minutes <= 60:
        raise ValueError('bounded run requires 1-60 minutes')
    deadline = time.monotonic() + minutes*60
    safe(deadline)
    repo = Path(__file__).resolve().parents[2]
    work = repo / 'ml/work'
    env = dict(os.environ, PIXELPROOF_DATA_ROOT=str(ROOT.parent), PYTHONPATH='ml:ml/src',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    # Use the extraction lock too: never train alongside an active feature writer.
    with (work / 'e59_features.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = pending()
        stamp = time.strftime('%Y%m%dT%H%M%S')
        for stage in stages:
            safe(deadline)
            log = work / f'e59_train_{stamp}_{stage}.log'
            with log.open('x') as stream:
                child = subprocess.Popen([sys.executable, '-m', 'experiments.e59_model', stage],
                                         cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                print(f'{stage} started; pid={child.pid}; log={log}', flush=True)
                try:
                    while child.poll() is None:
                        safe(deadline)
                        time.sleep(2)
                    if child.returncode:
                        raise RuntimeError(f'{stage} failed; inspect {log}; no next stage started')
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                print(stage + ' complete', flush=True)
    return {'state': 'E59_training_guard_complete', 'stages': stages}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt('training guard interrupted')
    signal.signal(signal.SIGTERM, stop)
    print(run(args.minutes))
