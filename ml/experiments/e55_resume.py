"""Operational guard for unchanged E55 jobs; never modifies scientific contracts."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
MOUNT=Path('/Volumes/LaCie')
DATA=MOUNT/'pixelproof-datasets'
ROOT=DATA/'e55'
WORK=REPO/'ml/work/e55_resume'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def power_safe(output):
    """Unknown/disconnected adapter status is unsafe; no global power settings."""
    match=re.search(r'(\d+)%',output)
    return ("Now drawing from 'AC Power'" in output and match is not None
            and 0<int(match.group(1))<=100 and 'discharging' not in output.lower())


def inspect_safety():
    reasons=[]
    try:
        power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True,timeout=10,check=True).stdout
        if not power_safe(power):reasons.append('AC power/charging not confirmed')
    except (OSError,subprocess.SubprocessError):
        reasons.append('cannot verify power state')
    if not MOUNT.is_mount():
        reasons.append('LaCie is not an actual mounted volume')
    else:
        try:
            info=plistlib.loads(subprocess.run(['diskutil','info','-plist',str(MOUNT)],
                capture_output=True,timeout=10,check=True).stdout)
            if info.get('Internal') is not False or info.get('MountPoint')!=str(MOUNT):
                reasons.append('external volume identity not confirmed')
            if DATA.resolve().parent!=MOUNT:
                reasons.append('dataset root redirected outside external volume')
            if shutil.disk_usage(MOUNT).free<10*1024**3:
                reasons.append('external volume has less than 10 GiB free')
        except (OSError,ValueError,subprocess.SubprocessError):
            reasons.append('cannot verify external volume')
    if not (ROOT/'contract.json').is_file():reasons.append('frozen E55 contract unavailable')
    return reasons


def verify_pause(root=ROOT,receipt_path=REPO/'evidence/e55_pause_checkpoint.json'):
    receipt=json.loads(receipt_path.read_text());root=Path(root)
    if sha(root/'contract.json')!=receipt['contract_sha256']:
        raise ValueError('paused experiment contract changed')
    seen=set();count=0
    for item in receipt['chunks']:
        name=item['file']
        if not re.fullmatch(r'\d{6}\.npz',name) or name in seen:
            raise ValueError('invalid or duplicate pause chunk name')
        seen.add(name);path=root/'chunks'/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('paused feature chunk changed: '+name)
        count+=item['views']
    if count!=receipt['completed_views']:raise ValueError('pause inventory count mismatch')
    return count


def pending_stages(root=ROOT):
    root=Path(root)
    if (root/'result.json').exists():return []
    if (root/'grayscale_features.npz').exists() and not (root/'features.json').exists():
        raise ValueError('final feature archive exists without receipt; inspect interrupted finalization')
    stages=[]
    if not (root/'features.json').exists():stages.append('extract')
    elif not (root/'grayscale_features.npz').exists():
        raise ValueError('feature receipt exists without archive')
    if not all((root/'training'/f'{arm}_fold{fold}.json').exists()
               for arm in ('duplicate_control','grayscale_20') for fold in range(3)):
        stages.append('fit')
    stages.append('report')
    return stages


def stop_owned_child(child):
    """Only the session/group created by this wrapper; never other Python jobs."""
    if child.poll() is not None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)


def interrupt_wrapper(signum,frame):
    # Ensure the child's finally block runs when the app stops the wrapper itself.
    raise KeyboardInterrupt('guard interrupted')


def run(until=None):
    reasons=inspect_safety()
    if reasons:return {'state':'not_started','reasons':reasons}
    if until is not None and datetime.now().astimezone()>=until:
        return {'state':'not_started','reasons':['overnight deadline reached']}
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'state':'not_started','reasons':['guarded E55 job already running']}
        # Direct legacy launches do not take our lock; refuse those too.
        processes=subprocess.run(['ps','-axo','pid,command'],capture_output=True,text=True,timeout=10,check=True).stdout
        if any(re.search(r'\bPython\s+-m experiments\.e55_color\s+(extract|fit|report)\b',line)
               for line in processes.splitlines()):
            return {'state':'not_started','reasons':['unguarded E55 process already running']}
        verify_pause()
        env=dict(os.environ,PIXELPROOF_DATA_ROOT=str(DATA),PYTHONPATH=f'{REPO}/ml:{REPO}/ml/src',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        stamp=datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
        completed=[]
        for stage in pending_stages():
            reasons=inspect_safety()
            if until is not None and datetime.now().astimezone()>=until:reasons.append('overnight deadline reached')
            if reasons:return {'state':'paused','completed_stages':completed,'reasons':reasons}
            logfile=WORK/f'{stamp}_{stage}.log'
            with logfile.open('a') as output:
                child=subprocess.Popen([sys.executable,'-m','experiments.e55_color',stage],
                    cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                print(json.dumps({'phase':stage,'pid':child.pid,'log':str(logfile)}),flush=True)
                try:
                    while child.poll() is None:
                        reasons=inspect_safety()
                        if until is not None and datetime.now().astimezone()>=until:
                            reasons.append('overnight deadline reached')
                        if reasons:
                            stop_owned_child(child)
                            return {'state':'paused','stage':stage,'reasons':reasons,'log':str(logfile)}
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:pass
                finally:
                    stop_owned_child(child)
                if child.returncode!=0:
                    return {'state':'stage_failed','stage':stage,'exit_code':child.returncode,'log':str(logfile)}
            completed.append(stage)
        return {'state':'stages_complete','completed_stages':completed,
                'note':'Inspect scientific result and acceptance checks; this status does not authorize promotion.'}


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupt_wrapper)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','run'])
    parser.add_argument('--until',help='Timezone-aware ISO deadline for this overnight run')
    args=parser.parse_args();deadline=datetime.fromisoformat(args.until) if args.until else None
    if deadline is not None and deadline.tzinfo is None:parser.error('--until requires explicit timezone')
    result={'reasons':inspect_safety()} if args.mode=='check' else run(deadline)
    print(json.dumps(result,indent=2))
    if result.get('reasons') or result.get('state')=='stage_failed':sys.exit(2)
