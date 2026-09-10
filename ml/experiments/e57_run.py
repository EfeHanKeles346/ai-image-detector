"""Operational guard for E57 finalization/features/fits; frozen science stays separate."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.e57_data_v2 import ROOT, MANIFEST, EVIDENCE, safe


def pending():
    steps=[]
    if not MANIFEST.exists(): steps.append(('e57_data_v2','audit'))
    if not (ROOT/'model_contract.json').exists(): steps.append(('e57_model','freeze'))
    feature=(ROOT/'fivek_features.npz').exists();receipt=(ROOT/'features.json').exists()
    if feature!=receipt: raise ValueError('orphaned feature finalization; requires explicit audit')
    if not feature: steps.append(('e57_model','extract'))
    results=[ROOT/'training'/f'{arm}_fold{fold}.json' for arm in ('native64','native64_fivek') for fold in range(3)]
    if not all(p.exists() for p in results): steps.append(('e57_model','fit'))
    if not (ROOT/'result.json').exists(): steps.append(('e57_model','report'))
    return steps


def wait_acquisition(pid,deadline):
    if pid<=0:raise ValueError('positive acquisition PID required')
    while True:
        safe(deadline)
        state=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=10)
        if state.returncode:break
        if '-m experiments.e57_data_v2 download' not in state.stdout:
            raise RuntimeError('acquisition PID now identifies a different process')
        time.sleep(5)
    if not (ROOT/'download.json').exists() or not (EVIDENCE/'e57_v2_download.json').exists():
        raise RuntimeError('acquisition ended without completed receipt; no next stage started')


def run(minutes,acquisition_pid=None):
    if not 1<=minutes<=60: raise ValueError('runtime must be 1-60 minutes')
    deadline=time.monotonic()+minutes*60;safe(deadline)
    repo=Path(__file__).resolve().parents[2];work=repo/'ml/work';stamp=time.strftime('%Y%m%dT%H%M%S')
    env=dict(os.environ,PIXELPROOF_DATA_ROOT=str(ROOT.parent.parent),PYTHONPATH='ml:ml/src',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (work/'e57_followup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if acquisition_pid is not None:
            print(f'Waiting for existing acquisition pid={acquisition_pid}; no duplicate download',flush=True)
            wait_acquisition(acquisition_pid,deadline)
        if not (ROOT/'download.json').exists():raise ValueError('acquisition must finish first')
        for module,phase in pending():
            safe(deadline);log=work/f'e57_{stamp}_{phase}.log'
            with log.open('x') as stream:
                child=subprocess.Popen([sys.executable,'-m','experiments.'+module,phase],cwd=repo,env=env,
                                       stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                print(f'{phase} started; pid={child.pid}; log={log}',flush=True)
                try:
                    while child.poll() is None:
                        safe(deadline);time.sleep(5)
                    if child.returncode:raise RuntimeError(f'{phase} failed with exit {child.returncode}; inspect {log}')
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                print(phase+' complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=int,default=60)
    parser.add_argument('--wait-for-download-pid',type=int);args=parser.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt('guard interrupted')
    signal.signal(signal.SIGTERM,stop)
    run(args.minutes,args.wait_for_download_pid)
