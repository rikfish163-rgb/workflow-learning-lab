"""CPU-only owner of a fixed, one-shot idle-qualified synthetic GPU probe."""
import argparse
import fcntl
import json
import os
import re
import signal
import time
from pathlib import Path

from scripts.probe_chunked_prefill import ID, PYTHON, ROOT, preflight, require
from wll.storage import atomic_json


def make_job(plan_sha):
    require(re.fullmatch('[a-f0-9]{64}', plan_sha) is not None, 'invalid plan hash')
    return {'id': ID, 'plan_sha256': plan_sha, 'max_runtime_seconds': 1200,
            'argv': [PYTHON, '-B', '-m', 'scripts.probe_chunked_prefill',
                     '--plan-sha256', plan_sha]}


def queue_operation(operation, plan_sha, *, root=ROOT):
    require(operation in {'enqueue', 'pulse', 'finish'}, 'unknown queue operation')
    job = make_job(plan_sha)
    art = Path(root) / 'artifacts'
    queue = art / 'gpu-idle-queue.json'
    heartbeat = art / f'gpu-idle-heartbeat-{ID}'
    result = art / f'gpu-idle-job-{ID}.json'
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(queue.read_text()) if queue.exists() else None
        if operation == 'enqueue':
            markers = [result, heartbeat, art / f'gpu-idle-ticket-{ID}.json',
                       art / f'gpu-idle-ticket-{ID}.consumed.json', art / f'{ID}.json']
            if any(p.exists() for p in markers):
                raise FileExistsError('attempt already exists; inspect instead of retrying')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']) is not None,
                        'invalid previous queue identity')
                old_result = art / f"gpu-idle-job-{current['id']}.json"
                require(old_result.exists() and json.loads(old_result.read_text())['status'] == 'finished'
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'cannot replace active or unconfirmed queue')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(archive.read_bytes() == queue.read_bytes(), 'different existing archive')
                else:
                    archive.write_bytes(queue.read_bytes())
            heartbeat.touch(exist_ok=False)
            try:
                atomic_json(queue, job)
            except BaseException:
                heartbeat.unlink(missing_ok=True)
                raise
        else:
            require(current == job, 'queue ownership changed')
            if operation == 'pulse':
                if not result.exists() or json.loads(result.read_text())['status'] != 'finished':
                    heartbeat.touch()
            else:
                heartbeat.unlink(missing_ok=True)
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller required')
    art = ROOT / 'artifacts'
    state_path = art / f'{ID}-controller.json'
    with (art / 'foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise FileExistsError('controller already attempted; inspect instead of restarting')
        preflight(args.plan_sha256)
        state = {'phase': 'preflight_passed', 'pid': os.getpid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
                 'started_epoch': time.time(), 'plan_sha256': args.plan_sha256, 'uses_gpu': False}
        atomic_json(state_path, state)
        owned = False
        try:
            queue_operation('enqueue', args.plan_sha256)
            owned = True
            state['phase'] = 'queued'
            atomic_json(state_path, state)
            while True:
                queue_operation('pulse', args.plan_sha256)
                result_path = art / f'gpu-idle-job-{ID}.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    state.update(phase=result['status'], job=result, checked_epoch=time.time())
                    atomic_json(state_path, state)
                    if result['status'] == 'finished':
                        probe_path = art / f'{ID}.json'
                        probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
                        require(result['exit_code'] == 0 and probe.get('phase') == 'complete',
                                'probe failed; retain partial evidence, no automatic retry')
                        state.update(phase='complete', finished_epoch=time.time())
                        atomic_json(state_path, state)
                        return
                time.sleep(15)
        except BaseException as exc:
            state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1000],
                         finished_epoch=time.time())
            atomic_json(state_path, state)
            raise
        finally:
            if owned:
                queue_operation('finish', args.plan_sha256)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
