"""Wait for one observed feature guard, then replace self with bounded training guard."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.e57_data import safe
from experiments.e59_model import require_features


def guard_present(pid):
    if pid <= 0:
        raise ValueError('positive observed guard PID required')
    state = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True, timeout=10)
    if state.returncode:
        return False
    if '-m experiments.e59_features run --minutes ' not in state.stdout:
        raise RuntimeError('observed PID now identifies a different process; no handoff')
    return True


def remaining_minutes(deadline, now):
    minutes = int((deadline - now) // 60)
    if minutes < 1:
        raise TimeoutError('no complete bounded minute left for training handoff')
    return min(minutes, 60)


def run(pid, minutes):
    if not 1 <= minutes <= 60:
        raise ValueError('combined wait/training budget must be1-60min')
    deadline = time.monotonic() + minutes*60
    safe(deadline)
    work = Path(__file__).resolve().parents[1] / 'work'
    with (work / 'e59_followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Waiting for existing feature guard pid={pid}; no duplicate extraction', flush=True)
        while guard_present(pid):
            safe(deadline)
            time.sleep(2)
        safe(deadline)
        require_features()
        remaining = remaining_minutes(deadline, time.monotonic())
        print(f'Feature guard exited; complete receipts verified; training budget={remaining}min', flush=True)
        # No nested detached supervisor: exec preserves this PID; training guard owns its children.
        os.execv(sys.executable, [sys.executable, '-m', 'experiments.e59_train_run', '--minutes', str(remaining)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-guard-pid', type=int, required=True)
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    run(args.feature_guard_pid, args.minutes)
