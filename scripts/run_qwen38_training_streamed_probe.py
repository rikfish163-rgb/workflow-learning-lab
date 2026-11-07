"""CPU owner for one idle-admitted27B streamed training feasibility probe; no automatic retry."""
import argparse
import fcntl
import json
import os
import re
import signal
import time
from pathlib import Path

from scripts.probe_chunked_prefill import PYTHON, ROOT, digest, require
from scripts.probe_qwen38_training_streamed import ID, assess_complete, preflight
from scripts.run_allocator_diagnostic import process_live
from wll.gpu_policy import query_smi
from wll.storage import atomic_json


def make_job(sha):
    require(re.fullmatch('[a-f0-9]{64}', sha) is not None, 'invalid plan hash')
    return {'id': ID, 'plan_sha256': sha, 'max_runtime_seconds': 1800,
            'argv': [PYTHON, '-B', '-m', 'scripts.probe_qwen38_training_streamed', '--plan-sha256', sha]}


def queue_operation(operation, sha, *, root=ROOT):
    require(operation in {'enqueue', 'pulse', 'finish'}, 'unknown queue operation')
    job, art = make_job(sha), Path(root) / 'artifacts'
    queue, heartbeat = art / 'gpu-idle-queue.json', art / f'gpu-idle-heartbeat-{ID}'
    result = art / f'gpu-idle-job-{ID}.json'
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(queue.read_text()) if queue.exists() else None
        if operation == 'enqueue':
            markers = [result, heartbeat, art / f'gpu-idle-ticket-{ID}.json',
                       art / f'gpu-idle-ticket-{ID}.consumed.json', art / f'{ID}.json']
            if any(p.exists() for p in markers):
                raise FileExistsError('attempt exists; inspect instead of retrying')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']) is not None, 'invalid old queue')
                old = art / f"gpu-idle-job-{current['id']}.json"
                require(old.exists() and json.loads(old.read_text())['status'] == 'finished'
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'cannot replace unfinished or owned queue')
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
            if operation == 'finish':
                heartbeat.unlink(missing_ok=True)
            elif not result.exists() or json.loads(result.read_text())['status'] != 'finished':
                heartbeat.touch()
    return job


def verify_terminal(sha, *, root=ROOT):
    art, job = Path(root) / 'artifacts', make_job(sha)
    info = json.loads((art / f'gpu-idle-job-{ID}.json').read_text())
    probe = json.loads((art / f'{ID}.json').read_text())
    require(info['status'] == 'finished' and info['job_id'] == probe['id'] == ID
            and probe['plan_sha256'] == sha, 'different or unfinished terminal identity')
    require(json.loads((art / 'gpu-idle-queue.json').read_text()) == job
            and digest(art / 'gpu-idle-queue.json') == info['queue_sha256'], 'different queue identity')
    require(not process_live(info['process_group'], info['start_ticks'])
            and not process_live(probe['pid'], probe['start_ticks']), 'process still alive')
    occupied = {int(line.split(',')[1].strip()) for line in query_smi('gpu_uuid,pid').splitlines()
                if line.strip()}
    require(not occupied.intersection([info['process_group'], probe['pid']]), 'CUDA context still alive')
    complete = info['exit_code'] == 0 and probe['phase'] == 'complete'
    if complete:
        assess_complete(probe)
    return {'released': True, 'complete': complete, 'checked_epoch': time.time(),
            'job_sha256': digest(art / f'gpu-idle-job-{ID}.json'),
            'probe_sha256': digest(art / f'{ID}.json'),
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
            raise FileExistsError('controller exists; no automatic restart')
        preflight(args.plan_sha256)
        state = {'phase': 'preflight_passed', 'pid': os.getpid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
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
                        state['release'] = verify_terminal(args.plan_sha256)
                        atomic_json(state_path, state)
                        require(state['release']['complete'], 'probe failed; no retry or automatic fallback')
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
