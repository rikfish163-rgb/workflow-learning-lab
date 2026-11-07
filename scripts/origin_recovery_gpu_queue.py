"""Fixed origin-pilot GPU jobs with a remote, one-use admission deadline.

The existing idle monitor and GPU guard remain unchanged. This entrypoint
rechecks the remote deadline after the GPU guard, before importing Torch or
allocating model memory. A cancelled/expired waiting job cannot enter CUDA even
if the monitor observed an earlier heartbeat. Queue status is not release proof.
"""
import argparse
import fcntl
import json
import math
import os
import re
import signal
import socket
import time
from pathlib import Path

from scripts import origin_recovery_cohort as cohort
from scripts.run_foundation_pilot import process_matches, read, verify_frozen_plan
from scripts.run_qwen38_b3_dev import PYTHON, SERVICE_SHA, static_backend
from scripts.run_qwen38_capability import runtime_state
from scripts.run_timed_tasks_v3 import _atomic, _sync_directory
from scripts.serve_qwen38_b3 import MODEL_IDS
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
ID = 'qwen38-origin-unattempted-001'
PLAN = 'artifacts/qwen38-origin-unattempted-plan-001.json'
DESIGN = 'artifacts/qwen38-origin-unattempted-design-001.json'
DESIGN_SHA = 'f8643987d4bc4ce7f5ef4e9e6f90762062ba99cdbdc3f961088d4ac7bd903ba0'
TASKS = 'artifacts/origin-recovery-inputs-001/tasks.json'
TASKS_SHA = '7ebc9cebe3422b1750bed40854c246b177095b68d2a8895a522a9c64a6d01a3f'
GATE_B3 = 'artifacts/b3-inference-gate-result-001/remote-artifacts/qwen38-b3-inference-gate-b3-001.json'
GATE_B3_SHA = '740ec1c024671228fe06084504f68c57c787167fc60b313ea907b2472e749016'
WAIT_SECONDS = 86400
PORT = 18979


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    import hashlib

    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular bound file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(Path(path).parent)


def load_registration(root, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, root / PLAN, sha)
    require(plan.get('id') == ID and plan.get('status') == 'registered'
            and plan.get('design_sha256') == DESIGN_SHA and digest(root / DESIGN) == DESIGN_SHA
            and plan.get('tasks_sha256') == TASKS_SHA and digest(root / TASKS) == TASKS_SHA
            and type(plan.get('gpu_seconds')) is int and plan['gpu_seconds'] == 7200
            and type(plan.get('admission_wait_seconds')) is int
            and plan['admission_wait_seconds'] == WAIT_SECONDS
            and plan.get('automatic_retry') is False, 'origin registration differs')
    required = {'scripts/origin_recovery_gpu_queue.py', 'scripts/run_origin_recovery_attempt.py',
                'scripts/audit_origin_recovery_attempt.py', 'scripts/origin_recovery_schedule.py',
                'scripts/run_origin_recovery_batch.py', 'scripts/prepare_origin_recovery_metadata.py',
                'scripts/gpu_idle_watch_b3.py', 'config/gpu-policy.json',
                'scripts/run_foundation_pilot.py', 'scripts/run_qwen38_capability.py',
                'scripts/run_qwen38_b3_dev.py', 'scripts/run_timed_tasks_v3.py',
                'scripts/origin_recovery_cohort.py'}
    required.update(str(path.relative_to(root)) for path in (root / 'wll').glob('*.py'))
    require(required <= plan['source_sha256'].keys(), 'incomplete origin source bindings')
    service_path = 'artifacts/qwen38-b3-service-plan-001.json'
    require(digest(root / service_path) == SERVICE_SHA, 'B3 service plan changed')
    service = read(root / service_path)
    require(all(plan['source_sha256'].get(k) == v for k, v in service['source_sha256'].items()),
            'service sources not bound')
    policy = read(root / 'config/gpu-policy.json')
    require(plan.get('gpu_policy') == policy and plan.get('gpu_policy_sha256')
            == digest(root / 'config/gpu-policy.json') and policy['idle_seconds'] == 180
            and set(policy['allowed_gpus']) == {'8', '9'}, 'GPU policy differs')
    gate_path = 'artifacts/qwen38-b3-inference-gate-integration-001.json'
    require({DESIGN, TASKS, service_path, gate_path, GATE_B3, cohort.PROOF, cohort.DESIGN}
            <= plan['input_sha256'].keys(),
            'required frozen inputs missing')
    design = read(root / DESIGN)
    require(digest(root / cohort.PROOF) == cohort.SELECTION_SHA256
            and digest(root / cohort.DESIGN) == cohort.OLD_DESIGN_SHA256
            and plan.get('selection') == design['selection']
            and plan['selection']['task_ids'] == cohort.cohort_ids()
            and plan.get('protocol') == design['protocol'] == read(root / cohort.DESIGN)['protocol'],
            'cohort or immutable base protocol differs')
    gate = read(root / gate_path)
    require(gate['phase'] == 'complete' and gate['all_resources_released'] is True
            and gate['plan_sha256'] == SERVICE_SHA, 'inference gate incomplete')
    require(digest(root / GATE_B3) == GATE_B3_SHA, 'B3 gate backend evidence changed')
    measured = static_backend(read(root / GATE_B3)['final_health']['backend'])
    expected = plan.get('expected_backend_static')
    require(type(expected) is dict and set(expected) == {'B3'}
            and fingerprint(expected['B3']) == fingerprint(measured), 'expected backend differs from B3 gate')
    return plan


def make_job(kind, index, owner, plan_sha):
    require(kind in ('source', 'branch') and type(index) is int
            and 0 <= index < (2 if kind == 'source' else 4)
            and type(owner) is str and re.fullmatch('[a-f0-9]{32}', owner)
            and type(plan_sha) is str and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid fixed origin job')
    return {'id': f'{ID}-{kind}-{index}', 'kind': kind, 'batch_index': index,
        'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 7200,
        'argv': [PYTHON, '-B', '-m', 'scripts.origin_recovery_gpu_queue', '--serve',
                 '--plan-sha256', plan_sha, '--kind', kind, '--batch', str(index), '--owner', owner]}


def window_path(art, job):
    return Path(art) / f"origin-admission-window-{job['id']}.json"


def entry_path(art, job):
    return Path(art) / f"origin-GPU-entry-{job['id']}.json"


def cancel_path(art, job):
    return Path(art) / f"origin-admission-cancelled-{job['id']}.json"


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def check_admission_window(root, job):
    art = Path(root) / 'artifacts'
    require(not cancel_path(art, job).exists(), 'admission cancelled')
    require(read(art / 'gpu-idle-queue.json') == job, 'queue ownership changed')
    window = read(window_path(art, job))
    require(window.get('job_sha256') == fingerprint(job) and window.get('boot_id') == boot_id(),
            'admission binding or boot changed')
    for key in ('created_monotonic', 'deadline_monotonic', 'created_epoch'):
        require(type(window.get(key)) in (int, float) and math.isfinite(window[key])
                and window[key] >= 0, 'invalid admission clock')
    require(window['deadline_monotonic'] == window['created_monotonic'] + WAIT_SECONDS
            and window['created_monotonic'] <= time.monotonic() < window['deadline_monotonic'],
            'admission window expired or clock changed')
    return window


def record_gpu_entry(root, job, gpu):
    # Called by the frozen loaded_engine after acquire_gpu and BEFORE Torch.
    art = Path(root) / 'artifacts'
    # Serialize the final check/entry with cancellation and heartbeat refresh.
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        window = check_admission_window(root, job)
        admitted = time.monotonic()
        record = {'job_sha256': fingerprint(job), 'window_sha256': digest(window_path(art, job)),
            'admitted_monotonic': admitted, 'admitted_epoch': time.time(),
            'admission_wait_seconds': admitted - window['created_monotonic'],
            'pid': os.getpid(), 'gpu_policy': gpu, 'before_torch_import': True}
        require(admitted < window['deadline_monotonic'], 'admission expired')
        write_new(entry_path(art, job), record)
    return record


def _admitted_in_window(art, job):
    path = entry_path(art, job)
    if not path.exists():
        return False
    window = read(window_path(art, job))
    record = read(path)
    require(record.get('job_sha256') == fingerprint(job)
            and record.get('window_sha256') == digest(window_path(art, job))
            and type(record.get('admitted_monotonic')) in (int, float)
            and window['created_monotonic'] <= record['admitted_monotonic'] < window['deadline_monotonic'],
            'GPU entry evidence differs')
    return True


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(job == make_job(job['kind'], job['batch_index'], job['owner'], job['plan_sha256']),
            'queued command changed')
    art = Path(root) / 'artifacts'
    queue = art / 'gpu-idle-queue.json'
    heartbeat = art / f"gpu-idle-heartbeat-{job['id']}"
    result = art / f"gpu-idle-job-{job['id']}.json"
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            load_registration(root, job['plan_sha256'])
            markers = [heartbeat, result, window_path(art, job), entry_path(art, job), cancel_path(art, job),
                art / f"gpu-idle-ticket-{job['id']}.json",
                art / f"gpu-idle-ticket-{job['id']}.consumed.json"]
            if any(path.exists() for path in markers):
                raise FileExistsError('origin job already attempted or reserved')
            if queue.exists():
                require(type(current) is dict and type(current.get('id')) is str
                        and re.fullmatch('[a-z0-9_-]+', current['id']),
                        'invalid previous queue')
                previous = runtime_state(current, art)
                require(previous['released'] and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'previous job not confirmed released')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(read(archive) == current, 'previous archive differs')
                else:
                    write_new(archive, current)
            started = time.monotonic()
            write_new(window_path(art, job), {'job_sha256': fingerprint(job), 'boot_id': boot_id(),
                'created_monotonic': started, 'deadline_monotonic': started + WAIT_SECONDS,
                'created_epoch': time.time()})
            _atomic(queue, job)
            heartbeat.touch(exist_ok=False)
            _sync_directory(art)
            current = job
        owned = current == job
        if owned and operation == 'pulse' and (not result.exists() or read(result)['status'] != 'finished'):
            if not _admitted_in_window(art, job):
                try:
                    check_admission_window(root, job)
                except ValueError:
                    if not cancel_path(art, job).exists():
                        write_new(cancel_path(art, job), {'reason': 'admission_window_unavailable',
                            'job_sha256': fingerprint(job), 'epoch': time.time()})
                    heartbeat.unlink(missing_ok=True)
                else:
                    heartbeat.touch()
            elif not cancel_path(art, job).exists():
                heartbeat.touch()
        if owned and operation == 'stop':
            if not cancel_path(art, job).exists():
                write_new(cancel_path(art, job), {'reason': 'controller_stop',
                    'job_sha256': fingerprint(job), 'epoch': time.time()})
            heartbeat.unlink(missing_ok=True)
            if result.exists():
                info = read(result)
                require(info['queue_sha256'] == digest(queue), 'different running queue')
                if process_matches(info, job['argv']):
                    os.killpg(info['process_group'], signal.SIGTERM)
        _sync_directory(art)
    return {'owned': owned, 'admission_cancelled': cancel_path(art, job).exists(),
            **runtime_state(job, art, known_pids)}


def serve(job, *, root=ROOT):
    require(socket.gethostname() == 'amax' and Path.cwd().resolve() == Path(root).resolve()
            and str(Path(root).resolve()) == str(REMOTE), 'unauthorized service host or directory')
    plan = load_registration(root, job['plan_sha256'])
    check_admission_window(root, job)
    # Reuse the already audited B3 model loader and HTTP engine unchanged.
    from http.server import ThreadingHTTPServer

    from scripts.serve_qwen38 import make_handler
    from scripts.serve_qwen38_b3 import loaded_engine

    with loaded_engine(SERVICE_SHA, MODEL_IDS['B3'], arm='B3',
                       admitted=lambda gpu: record_gpu_entry(root, job, gpu)) as engine:
        require(static_backend(engine.backend) == plan['expected_backend_static']['B3'],
                'actual loaded backend differs')
        server = ThreadingHTTPServer(('127.0.0.1', PORT), make_handler(engine))
        write_new(Path(root) / 'artifacts' / f"origin-service-ready-{job['id']}.json", {
            'job_sha256': fingerprint(job), 'ready_epoch': time.time(), 'ready_monotonic': time.monotonic(),
            'backend_sha256': fingerprint(engine.backend), 'pid': os.getpid()})
        try:
            server.serve_forever()
        finally:
            server.server_close()
        require(not engine.failed, 'model service stopped after execution failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--kind', choices=['source', 'branch'], required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    serve(make_job(args.kind, args.batch, args.owner, args.plan_sha256))


if __name__ == '__main__':
    main()
