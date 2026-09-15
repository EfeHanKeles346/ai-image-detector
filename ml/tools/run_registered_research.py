"""Run one already registered research stage now, with exact E92 restoration.

This command does not schedule or wait for future work. Upstream lifecycles must
already be complete; the caller reviews outcomes between runs.
"""
import argparse
import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e124_e125 as lifecycle
from run_e126_after_training import ready_e92

OPERATIONS={'e130':('experiments.e130_patch_drift_audit','audit'),
            'e131':('experiments.e131_source_holdout','fit'),
            'e132':('experiments.e132_patch_learning','fit'),
            'e133_generation':('experiments.e133_mask_location','generate'),
            'e133_localization':('experiments.e133_mask_location','score'),
            'e135_cache':('experiments.e135_location_learning','cache'),
            'e135_fit':('experiments.e135_location_learning','fit'),
            'e136':('experiments.e136_transport_consistency','fit')}


def model1_consistency_ready(root):
    for stage in ('e131_pipeline','e135_fit_pipeline'):
        result=json.loads((root/stage/'status.json').read_text())
        if result.get('state')!='complete' or result.get('E92_restored') is not True:
            raise RuntimeError('Model1 consistency requires prior shared-memory work complete and E92 restored')


def location_learning_ready(root,operation):
    stages=['e132_pipeline','e133_generation_pipeline','e133_localization_pipeline']
    if operation=='e135_fit':stages.append('e135_cache_pipeline')
    for stage in stages:
        result=json.loads((root/stage/'status.json').read_text())
        if result.get('state')!='complete' or result.get('E92_restored') is not True:
            raise RuntimeError('Location learning requires completed preceding stages and exact E92 restoration')


def location_ready(root, operation):
    stages=['e130_pipeline','e131_pipeline','e132_pipeline']
    if operation=='e133_localization':stages.append('e133_generation_pipeline')
    for stage in stages:
        result=json.loads((root/stage/'status.json').read_text())
        if result.get('state')!='complete' or result.get('E92_restored') is not True:
            raise RuntimeError('Location challenge requires completed predecessor lifecycles and exact E92 restoration')


def patch_learning_ready(root):
    for name in ('e130_pipeline', 'e131_pipeline'):
        result=json.loads((root/name/'status.json').read_text())
        if result.get('state')!='complete' or result.get('E92_restored') is not True:
            raise RuntimeError('E132 requires completed E130/E131 lifecycles with E92 restored')


def upstream_ready(first,second,third):
    if first.get('state')!='complete' or first.get('E92_restored') is not True:
        raise RuntimeError('E124/E125 lifecycle must be complete with E92 restored')
    if second.get('state')!='skipped_train_failed' and not (
        second.get('state')=='complete' and second.get('E92_restored_after_DEV') is True):
        raise RuntimeError('E126 lifecycle must have released resources')
    if third.get('state')!='complete' or third.get('E92_restored_after_pilot') is not True:
        raise RuntimeError('E129 lifecycle must be complete with E92 restored')


def journal(operation,message):
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    text=f'\n### {operation.upper()} registered-stage execution — {stamp}\n\n{message}\n'
    for path in (lifecycle.REPO/'HISTORY.md',lifecycle.REPO/'ml/EXPERIMENTS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)
    print(message,flush=True)


def main(operation):
    root=Path(os.environ['PIXELPROOF_DATA_ROOT']);run=root/(operation+'_pipeline');run.mkdir(exist_ok=True)
    lifecycle.RUN=run
    with (root/'registered_research_resource.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        upstream_ready(*(json.loads((root/name/'status.json').read_text()) for name in
            ('e124_pipeline','e126_pipeline','e129_pipeline')))
        if operation=='e132':patch_learning_ready(root)
        if operation.startswith('e133_'):location_ready(root, operation)
        if operation.startswith('e135_'):location_learning_ready(root, operation)
        if operation=='e136':model1_consistency_ready(root)
        module,stage=OPERATIONS[operation]
        importlib.import_module(module).validate()  # Validate frozen evidence before touching the API.
        stopped=False;failure=None;restored=None
        try:
            if not ready_e92(lifecycle.health()):raise RuntimeError('Exact ready E92 identity required')
            if 'AC Power' not in subprocess.check_output(['pmset','-g','batt'],text=True):raise RuntimeError('AC power required')
            pids=set(subprocess.check_output(['lsof','-t','-iTCP:8800','-sTCP:LISTEN'],text=True).split())
            if len(pids)!=1:raise RuntimeError('Exactly one API listener required')
            pid=int(next(iter(pids)));command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True)
            if '-m uvicorn pixelproof.internship_serve:app' not in command or '--port 8800' not in command:
                raise RuntimeError('Refusing unrelated listener')
            os.kill(pid,signal.SIGTERM);stopped=True
            for _ in range(30):
                if lifecycle.health() is None:break
                time.sleep(1)
            else:raise RuntimeError('API did not release resources')
            journal(operation,'Starting the registered '+stage+' stage now. Exact E92 is temporarily stopped for shared memory. No recurring task or automatic promotion.')
            lifecycle.run_stage(module,stage)
        except BaseException as exc:
            failure=type(exc).__name__+': '+str(exc)
        finally:
            if stopped:
                previous=os.environ.get('PYTHONPATH');os.environ['PYTHONPATH']='ml:ml/src'
                try:restored=lifecycle.restore() and ready_e92(lifecycle.health())
                finally:
                    if previous is None:os.environ.pop('PYTHONPATH',None)
                    else:os.environ['PYTHONPATH']=previous
            result={'state':'failed' if failure or (stopped and not restored) else 'complete',
                    'failure':failure,'E92_restored':restored}
            temp=run/'status.json.part';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(run/'status.json')
            journal(operation,'Stage final state: '+json.dumps(result,sort_keys=True))
        if result['state']=='failed':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=OPERATIONS)
    main(parser.parse_args().operation)
