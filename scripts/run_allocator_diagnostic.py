"""CPU owner of two independently idle-qualified, serial allocator probes."""
import argparse
import fcntl
import json
import os
import re
import signal
import time
from pathlib import Path

from scripts.probe_allocator import CONFIGS, ID, preflight
from scripts.probe_chunked_prefill import PYTHON, ROOT, digest, require
from wll.gpu_policy import query_smi
from wll.storage import atomic_json


def job_id(mode):
    require(mode in CONFIGS, 'unregistered allocator mode')
    return f'{ID}-{mode}'


def make_job(mode, sha):
    require(re.fullmatch('[a-f0-9]{64}', sha) is not None, 'invalid plan hash')
    return {'id': job_id(mode), 'plan_sha256': sha, 'max_runtime_seconds': 1200,
            'argv': [PYTHON, '-B', '-m', 'scripts.probe_allocator', '--mode', mode,
                     '--plan-sha256', sha]}


def queue_operation(operation, mode, sha, *, root=ROOT):
    require(operation in {'enqueue', 'pulse', 'finish'}, 'unknown queue operation')
    job = make_job(mode, sha)
    art = Path(root) / 'artifacts'
    queue = art / 'gpu-idle-queue.json'
    heartbeat = art / f"gpu-idle-heartbeat-{job['id']}"
    result = art / f"gpu-idle-job-{job['id']}.json"
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(queue.read_text()) if queue.exists() else None
        if operation == 'enqueue':
            markers = [result, heartbeat, art / f"gpu-idle-ticket-{job['id']}.json",
                       art / f"gpu-idle-ticket-{job['id']}.consumed.json", art / f"{job['id']}.json"]
            if any(p.exists() for p in markers):
                raise FileExistsError('attempt exists; inspect instead of retrying')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']) is not None,
                        'invalid previous queue identity')
                old_result = art / f"gpu-idle-job-{current['id']}.json"
                require(old_result.exists() and json.loads(old_result.read_text())['status'] == 'finished'
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'cannot replace active or unconfirmed queue')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(archive.read_bytes() == queue.read_bytes(), 'different queue archive')
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


def process_live(pid, ticks):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().split()
        return proc.stat().st_uid == os.getuid() and stat[21] == str(ticks) and stat[2] != 'Z'
    except FileNotFoundError:
        return False


def verify_release(mode, sha):
    art = ROOT / 'artifacts'
    job = make_job(mode, sha)
    info = json.loads((art / f"gpu-idle-job-{job['id']}.json").read_text())
    probe = json.loads((art / f"{job['id']}.json").read_text())
    require(info['status'] == 'finished' and info['exit_code'] == 0
            and probe['phase'] == 'complete' and probe['plan_sha256'] == sha
            and probe['id'] == info['job_id'] == job['id'] and probe['mode'] == mode,
            'incomplete allocator probe; preserve results and stop')
    require(json.loads((art / 'gpu-idle-queue.json').read_text()) == job
            and digest(art / 'gpu-idle-queue.json') == info['queue_sha256'],
            'unexpected released queue identity')
    require(not process_live(info['process_group'], info['start_ticks'])
            and not process_live(probe['pid'], probe['start_ticks']), 'probe process still alive')
    occupied = {int(line.split(',')[1].strip()) for line in query_smi('gpu_uuid,pid').splitlines()
                if line.strip()}
    require(not occupied.intersection([info['process_group'], probe['pid']]),
            'own process still has a CUDA context')
    return {'mode': mode, 'released': True, 'checked_epoch': time.time(),
            'job_sha256': digest(art / f"gpu-idle-job-{job['id']}.json"),
            'probe_sha256': digest(art / f"{job['id']}.json"),
            'model_pid': probe['pid'], 'process_group': info['process_group']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only owner required')
    art = ROOT / 'artifacts'
    state_path = art / f'{ID}-controller.json'
    with (art / 'foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise FileExistsError('controller attempted; no automatic restart')
        plan, _ = preflight(args.plan_sha256)
        state = {'phase': 'preflight_passed', 'pid': os.getpid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
                 'started_epoch': time.time(), 'plan_sha256': args.plan_sha256,
                 'uses_gpu': False, 'completed_modes': [], 'release_checks': []}
        atomic_json(state_path, state)
        try:
            for mode in plan['mode_order']:
                preflight(args.plan_sha256)
                queue_operation('enqueue', mode, args.plan_sha256)
                try:
                    state.update(phase='queued', current_mode=mode)
                    atomic_json(state_path, state)
                    while True:
                        queue_operation('pulse', mode, args.plan_sha256)
                        path = art / f'gpu-idle-job-{job_id(mode)}.json'
                        if path.exists():
                            result = json.loads(path.read_text())
                            state.update(phase=result['status'], job=result, checked_epoch=time.time())
                            atomic_json(state_path, state)
                            if result['status'] == 'finished':
                                release = verify_release(mode, args.plan_sha256)
                                state['release_checks'].append(release)
                                state['completed_modes'].append(mode)
                                atomic_json(state_path, state)
                                break
                        time.sleep(15)
                finally:
                    queue_operation('finish', mode, args.plan_sha256)
            state.update(phase='complete', finished_epoch=time.time())
            atomic_json(state_path, state)
        except BaseException as exc:
            state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1200],
                         finished_epoch=time.time())
            atomic_json(state_path, state)
            raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
