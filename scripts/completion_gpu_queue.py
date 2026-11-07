"""Fixed completion-check GPU leases; the existing watcher and guard remain in force.

Admission is one-shot and rechecked under the queue lock before Torch is loaded.
This module does not determine CPU/container release or establish study efficacy.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
from os import killpg
from pathlib import Path
from socket import gethostname
from time import monotonic
from time import time as epoch

from scripts.run_foundation_pilot import process_matches, read, verify_frozen_plan
from scripts.run_qwen38_b3_dev import PYTHON, SERVICE_SHA, static_backend
from scripts.run_qwen38_capability import runtime_state
from scripts.run_timed_tasks_v3 import _atomic, _sync_directory
from scripts.serve_qwen38_b3 import MODEL_IDS
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
ID = 'qwen38-completion-check-001'
PLAN = 'artifacts/qwen38-completion-check-plan-001.json'
SERVICE_PLAN = 'artifacts/qwen38-b3-service-plan-001.json'
GATE = 'artifacts/qwen38-b3-inference-gate-integration-001.json'
GATE_SHA = 'ae3e5c90a88a085360ddf2f1f9fa237fb2c0a0a1fbbc541eaa65ac8e8d8eed67'
GATE_B3 = 'artifacts/b3-inference-gate-result-001/remote-artifacts/qwen38-b3-inference-gate-b3-001.json'
GATE_B3_SHA = '740ec1c024671228fe06084504f68c57c787167fc60b313ea907b2472e749016'
REPLAY_INTENT = 'artifacts/completion-replay-probe-intent-001.json'
REPLAY_INTENT_SHA = 'c357f644cf5c3a0f493577877778fd95b09d3b5ddf559f013826afa580120687'
CONTEXT_PROOF = 'artifacts/completion-context-feasibility-CPU-001.json'
CONTEXT_PROOF_SHA = '9a78f3bd6108b06d2298cecc56fd45b62df8fc3319a2cd0d2376eb7f380b6406'
INELIGIBLE = ['27e1026_2', 'b0a8eae_2']
EXCLUDED = '2a163ab_2'
WAIT_SECONDS = 86400
PORT = 18980
REQUIRED_SOURCES = {
    'scripts/completion_gpu_queue.py', 'scripts/gpu_idle_watch_b3.py',
    'config/gpu-policy.json', 'scripts/run_foundation_pilot.py',
    'scripts/run_qwen38_capability.py', 'scripts/run_qwen38_b3_dev.py',
    'scripts/run_timed_tasks_v3.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
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
            and type(plan.get('gpu_seconds')) is int and plan['gpu_seconds'] == 7200
            and type(plan.get('admission_wait_seconds')) is int
            and plan['admission_wait_seconds'] == WAIT_SECONDS
            and plan.get('automatic_retry') is False, 'completion registration differs')
    require(REQUIRED_SOURCES <= plan['source_sha256'].keys(), 'required source missing')
    require({SERVICE_PLAN, GATE, GATE_B3, REPLAY_INTENT, CONTEXT_PROOF}
            <= plan['input_sha256'].keys(), 'required input missing')
    for name, expected in ((SERVICE_PLAN, SERVICE_SHA), (GATE, GATE_SHA),
                           (GATE_B3, GATE_B3_SHA), (REPLAY_INTENT, REPLAY_INTENT_SHA),
                           (CONTEXT_PROOF, CONTEXT_PROOF_SHA)):
        require(plan['input_sha256'][name] == expected and digest(root / name) == expected,
                'required input changed')
    service = read(root / SERVICE_PLAN)
    require(all(plan['source_sha256'].get(k) == v for k, v in service['source_sha256'].items()),
            'service source bindings differ')
    policy = read(root / 'config/gpu-policy.json')
    require(plan.get('gpu_policy') == policy and plan.get('gpu_policy_sha256')
            == service['input_sha256']['config/gpu-policy.json']
            == digest(root / 'config/gpu-policy.json') and policy['idle_seconds'] == 180
            and set(policy['allowed_gpus']) == {'8', '9'}, 'GPU policy differs')
    gate = read(root / GATE)
    require(gate.get('phase') == 'complete' and gate.get('all_resources_released') is True
            and gate.get('plan_sha256') == SERVICE_SHA, 'B3 gate incomplete')
    measured = static_backend(read(root / GATE_B3)['final_health']['backend'])
    expected = plan.get('expected_backend_static')
    require(type(expected) is dict and set(expected) == {'B3'}
            and fingerprint(expected['B3']) == fingerprint(measured), 'B3 backend differs')

    ids = read(root / REPLAY_INTENT)['task_ids']
    require(type(ids) is list and len(ids) == len(set(ids)) == 11
            and all(type(value) is str and re.fullmatch('[a-f0-9]{7}_2', value) for value in ids)
            and ids.count(EXCLUDED) == 1, 'fixed replay IDs differ')
    selected = [value for value in ids if value != EXCLUDED]
    require(len({value.rsplit('_', 1)[0] for value in selected}) == 10
            and plan.get('batch_task_ids') == [selected[:5], selected[5:]],
            'fixed batch order differs')
    proof = read(root / CONTEXT_PROOF)
    rows = proof.get('rows')
    require(proof.get('status') == 'measured_CPU_tokenization' and type(rows) is list
            and len(rows) == 20, 'initial context proof incomplete')
    require([(row.get('task_id'), row.get('arm')) for row in rows]
            == [(task_id, arm) for task_id in selected for arm in ('A', 'B')],
            'initial context proof order differs')
    for row in rows:
        tokens, fits = row.get('input_tokens'), row.get('fits_initial_input')
        require(type(tokens) is int and tokens > 0 and type(fits) is bool
                and fits == (tokens <= 15360), 'initial context measurement differs')
    ineligible = [task_id for task_id in selected
                  if any(not row['fits_initial_input'] for row in rows if row['task_id'] == task_id)]
    require(ineligible == INELIGIBLE and plan.get('context_ineligible_task_ids') == INELIGIBLE,
            'fixed initial context exclusions differ')
    return plan


def make_job(index, owner, plan_sha):
    require(type(index) is int and index in (0, 1)
            and type(owner) is str and re.fullmatch('[a-f0-9]{32}', owner)
            and type(plan_sha) is str and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid fixed completion job')
    return {'id': f'{ID}-comparison-{index}', 'kind': 'comparison', 'batch_index': index,
            'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 7200,
            'argv': [PYTHON, '-B', '-m', 'scripts.completion_gpu_queue', '--serve',
                     '--plan-sha256', plan_sha, '--batch', str(index), '--owner', owner]}


def window_path(art, job):
    return Path(art) / f"completion-admission-window-{job['id']}.json"


def entry_path(art, job):
    return Path(art) / f"completion-GPU-entry-{job['id']}.json"


def cancel_path(art, job):
    return Path(art) / f"completion-admission-cancelled-{job['id']}.json"


def ready_path(art, job):
    return Path(art) / f"completion-service-ready-{job['id']}.json"


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _bound_window(art, job):
    window = read(window_path(art, job))
    require(window.get('job_sha256') == fingerprint(job) and window.get('boot_id') == boot_id(),
            'admission binding or boot changed')
    for key in ('created_monotonic', 'deadline_monotonic', 'created_epoch'):
        require(type(window.get(key)) in (int, float) and math.isfinite(window[key])
                and window[key] >= 0, 'invalid admission clock')
    require(window['deadline_monotonic'] == window['created_monotonic'] + WAIT_SECONDS,
            'admission window differs')
    return window


def check_admission_window(root, job):
    art = Path(root) / 'artifacts'
    require(not cancel_path(art, job).exists(), 'admission cancelled')
    require(read(art / 'gpu-idle-queue.json') == job, 'queue ownership changed')
    window = _bound_window(art, job)
    require(window['created_monotonic'] <= monotonic() < window['deadline_monotonic'],
            'admission window expired or clock changed')
    return window


def record_gpu_entry(root, job, gpu):
    # loaded_engine invokes this only after its GPU guard, before importing Torch.
    art = Path(root) / 'artifacts'
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        window = check_admission_window(root, job)
        admitted = monotonic()
        require(window['created_monotonic'] <= admitted < window['deadline_monotonic'],
                'admission expired or clock changed')
        record = {'job_sha256': fingerprint(job), 'window_sha256': digest(window_path(art, job)),
                  'admitted_monotonic': admitted, 'admitted_epoch': epoch(),
                  'admission_wait_seconds': admitted - window['created_monotonic'],
                  'pid': os.getpid(), 'gpu_policy': gpu, 'before_torch_import': True}
        write_new(entry_path(art, job), record)
    return record


def _admitted_in_window(art, job):
    path = entry_path(art, job)
    if not path.exists():
        return False
    window, record = _bound_window(art, job), read(path)
    admitted = record.get('admitted_monotonic')
    require(record.get('job_sha256') == fingerprint(job)
            and record.get('window_sha256') == digest(window_path(art, job))
            and type(admitted) in (int, float) and math.isfinite(admitted)
            and window['created_monotonic'] <= admitted < window['deadline_monotonic'],
            'GPU entry evidence differs')
    return True


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(type(job) is dict and job == make_job(job.get('batch_index'), job.get('owner'),
                                                 job.get('plan_sha256')), 'queued command changed')
    art = Path(root) / 'artifacts'
    queue = art / 'gpu-idle-queue.json'
    heartbeat = art / f"gpu-idle-heartbeat-{job['id']}"
    result = art / f"gpu-idle-job-{job['id']}.json"
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            load_registration(root, job['plan_sha256'])
            markers = [heartbeat, result, window_path(art, job), entry_path(art, job),
                       cancel_path(art, job), ready_path(art, job),
                       art / f"gpu-idle-ticket-{job['id']}.json",
                       art / f"gpu-idle-ticket-{job['id']}.consumed.json"]
            if any(path.exists() or path.is_symlink() for path in markers):
                raise FileExistsError('completion job already attempted or reserved')
            if queue.exists():
                require(type(current) is dict and type(current.get('id')) is str
                        and re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid previous queue')
                previous = runtime_state(current, art)
                require(previous['released'] and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'previous job not confirmed released')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(read(archive) == current, 'previous archive differs')
                else:
                    write_new(archive, current)
            started = monotonic()
            write_new(window_path(art, job), {'job_sha256': fingerprint(job), 'boot_id': boot_id(),
                      'created_monotonic': started, 'deadline_monotonic': started + WAIT_SECONDS,
                      'created_epoch': epoch()})
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
                                  'job_sha256': fingerprint(job), 'epoch': epoch()})
                    heartbeat.unlink(missing_ok=True)
                else:
                    heartbeat.touch()
            elif not cancel_path(art, job).exists():
                heartbeat.touch()
        if owned and operation == 'stop':
            if not cancel_path(art, job).exists():
                write_new(cancel_path(art, job), {'reason': 'controller_stop',
                          'job_sha256': fingerprint(job), 'epoch': epoch()})
            heartbeat.unlink(missing_ok=True)
            if result.exists():
                info = read(result)
                require(info['queue_sha256'] == digest(queue), 'different running queue')
                if process_matches(info, job['argv']):
                    killpg(info['process_group'], signal.SIGTERM)
        _sync_directory(art)
    return {'owned': owned, 'admission_cancelled': cancel_path(art, job).exists(),
            **runtime_state(job, art, known_pids)}


def serve(job, *, root=ROOT):
    require(job == make_job(job.get('batch_index'), job.get('owner'), job.get('plan_sha256')),
            'queued command changed')
    require(gethostname() == 'amax' and Path.cwd().resolve() == Path(root).resolve()
            and str(Path(root).resolve()) == str(REMOTE), 'unauthorized service host or directory')
    plan = load_registration(root, job['plan_sha256'])
    check_admission_window(root, job)
    from http.server import ThreadingHTTPServer

    from scripts.serve_qwen38 import make_handler
    from scripts.serve_qwen38_b3 import loaded_engine

    with loaded_engine(SERVICE_SHA, MODEL_IDS['B3'], arm='B3',
                       admitted=lambda gpu: record_gpu_entry(root, job, gpu)) as engine:
        require(static_backend(engine.backend) == plan['expected_backend_static']['B3'],
                'loaded B3 backend differs')
        server = ThreadingHTTPServer(('127.0.0.1', PORT), make_handler(engine))
        try:
            write_new(ready_path(Path(root) / 'artifacts', job), {
                'job_sha256': fingerprint(job), 'ready_epoch': epoch(), 'ready_monotonic': monotonic(),
                'backend_sha256': fingerprint(engine.backend), 'pid': os.getpid()})
            server.serve_forever()
        finally:
            server.server_close()
        require(not engine.failed, 'model service stopped after execution failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    serve(make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == '__main__':
    main()
