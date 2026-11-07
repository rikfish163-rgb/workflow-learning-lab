"""One-shot CPU controller for a standalone registered completion lesson writer.

The existing idle monitor owns GPU admission and the independent 1800s timeout.
No model server, task worker, tunnel, retry, or accepted experience is created here.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path
from socket import gethostname

from scripts.origin_recovery_gpu_queue import digest, require, write_new
from scripts.run_foundation_pilot import process_matches, read, verify_frozen_plan
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_qwen38_b3_dev import PYTHON
from scripts.run_qwen38_capability import runtime_state
from scripts.run_timed_tasks_v3 import _atomic, _sync_directory
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
ID = 'qwen38-completion-lesson-002'
PLAN = 'artifacts/qwen38-completion-lesson-plan-002.json'
WAIT_SECONDS = 86400
GPU_SECONDS = 1800
PRIVATE_OUTPUT = Path('/home/hts/.local/share/workflow-learning-lab/completion-lesson-002/output')


def make_job(owner, plan_sha):
    require(type(owner) is str and re.fullmatch('[a-f0-9]{32}', owner)
            and type(plan_sha) is str and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid writer owner or plan hash')
    return {'id': ID, 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': GPU_SECONDS,
            'argv': [PYTHON, '-B', '-m', 'scripts.run_completion_lesson', '--plan-sha256', plan_sha]}


def load_registration(root, sha, *, native=False):
    root = Path(root)
    require((root / PLAN).is_file() and not (root / PLAN).is_symlink(), 'regular writer plan required')
    plan = verify_frozen_plan(root, root / PLAN, sha)
    require(plan.get('id') == ID and plan.get('status') == 'registered'
            and type(plan.get('gpu_seconds')) is int and plan['gpu_seconds'] == GPU_SECONDS
            and plan.get('admission_wait_seconds') == WAIT_SECONDS
            and plan.get('automatic_retry') is False
            and plan.get('private_output') == str(PRIVATE_OUTPUT), 'writer registration differs')
    required = {'scripts/queue_completion_lesson.py', 'scripts/run_completion_lesson.py',
                'scripts/gpu_idle_watch_b3.py', 'wll/gpu_policy.py', 'config/gpu-policy.json'}
    require(required <= plan['source_sha256'].keys(), 'writer queue dependencies not bound')
    require(plan['gpu_policy'] == read(root / 'config/gpu-policy.json')
            and plan['gpu_policy_sha256'] == digest(root / 'config/gpu-policy.json')
            and plan['gpu_policy']['idle_seconds'] == 180
            and set(plan['gpu_policy']['allowed_gpus']) == {'8', '9'}, 'GPU policy differs')
    if native:
        require(type(plan.get('native_bindings')) is dict and plan['native_bindings'],
                'native writer bindings required')
        for name, expected in plan['native_bindings'].items():
            path = Path(name) if Path(name).is_absolute() else root / name
            require(digest(path) == expected, 'native writer input changed')
    return plan


def marker_path(art):
    return Path(art) / 'completion-lesson-enqueued-002.json'


def cancel_path(art):
    return Path(art) / 'completion-lesson-cancelled-002.json'


def _window(art, job):
    marker = read(marker_path(art))
    now = time.monotonic()
    require(marker.get('job_sha256') == fingerprint(job)
            and marker.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'writer admission binding changed')
    require(marker['deadline_monotonic'] == marker['created_monotonic'] + WAIT_SECONDS
            and marker['created_monotonic'] <= now < marker['deadline_monotonic'],
            'writer admission window expired')
    return marker


def validate_worker_entry(root, sha):
    root = Path(root)
    load_registration(root, sha)
    require(gethostname() == 'amax' and Path.cwd().resolve() == root.resolve() == REMOTE.resolve(),
            'unauthorized writer host or directory')
    art = root / 'artifacts'
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = read(art / 'gpu-idle-queue.json')
        require(job == make_job(job.get('owner'), sha), 'writer queue ownership changed')
        require(not cancel_path(art).exists(), 'writer admission cancelled')
        _window(art, job)
        heartbeat = art / f'gpu-idle-heartbeat-{ID}'
        require(heartbeat.is_file() and not heartbeat.is_symlink()
                and 0 <= time.time() - heartbeat.stat().st_mtime <= 60, 'writer heartbeat expired')
    return job


def _observation(state):
    result = {key: state.get(key) for key in ('monitor_live', 'released', 'model_pids')}
    process_fields = {'pid', 'uid', 'start_ticks', 'process_group', 'cwd', 'cuda_visible_devices'}
    result['model_processes'] = [{key: value for key, value in proc.items() if key in process_fields}
                                 for proc in state.get('model_processes', [])]
    info = state.get('job')
    keys = ('job_id', 'status', 'gpu_index', 'queue_sha256', 'started_epoch', 'finished_epoch',
            'process_group', 'start_ticks', 'exit_code', 'stop_reason')
    result['job'] = {key: info[key] for key in keys if key in info} if type(info) is dict else None
    return result


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(type(job) is dict and job == make_job(job.get('owner'), job.get('plan_sha256')),
            'writer queue command changed')
    root = Path(root)
    art = root / 'artifacts'
    queue = art / 'gpu-idle-queue.json'
    heartbeat = art / f'gpu-idle-heartbeat-{ID}'
    result = art / f'gpu-idle-job-{ID}.json'
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            plan = load_registration(root, job['plan_sha256'])
            markers = [result, heartbeat, marker_path(art), cancel_path(art),
                       art / f'gpu-idle-ticket-{ID}.json',
                       art / f'gpu-idle-ticket-{ID}.consumed.json',
                       root / 'artifacts' / ID / 'worker-report.json', Path(plan['private_output'])]
            if any(path.exists() or path.is_symlink() for path in markers):
                raise FileExistsError('writer already attempted or owned')
            if queue.exists():
                require(type(current) is dict and type(current.get('id')) is str
                        and re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid previous queue')
                previous = runtime_state(current, art)
                require(previous['released'] is True
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'previous queue not released')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(not archive.is_symlink() and archive.read_bytes() == queue.read_bytes(),
                            'previous queue archive differs')
                else:
                    with archive.open('xb') as stream:
                        stream.write(queue.read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
            now = time.monotonic()
            write_new(marker_path(art), {'job_sha256': fingerprint(job),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'created_monotonic': now, 'deadline_monotonic': now + WAIT_SECONDS})
            _atomic(queue, job)
            heartbeat.touch(exist_ok=False)
            current = job
        owned = current == job
        if owned and operation == 'pulse' and not cancel_path(art).exists():
            finished = result.exists() and read(result).get('status') == 'finished'
            consumed = art / f'gpu-idle-ticket-{ID}.consumed.json'
            if not finished:
                try:
                    if not consumed.is_file():
                        _window(art, job)
                except ValueError:
                    write_new(cancel_path(art), {'job_sha256': fingerprint(job), 'reason': 'expired'})
                    heartbeat.unlink(missing_ok=True)
                else:
                    require(not heartbeat.is_symlink(), 'unsafe writer heartbeat')
                    heartbeat.touch()
        if owned and operation == 'stop':
            if not cancel_path(art).exists():
                write_new(cancel_path(art), {'job_sha256': fingerprint(job), 'reason': 'controller_stop'})
            heartbeat.unlink(missing_ok=True)
            if result.exists():
                info = read(result)
                require(info.get('queue_sha256') == digest(queue), 'running writer queue changed')
                if process_matches(info, job['argv']):
                    os.killpg(info['process_group'], signal.SIGTERM)
        _sync_directory(art)
    return {'owned': owned, 'heartbeat_absent': not heartbeat.exists(),
            'admission_cancelled': cancel_path(art).exists(),
            **_observation(runtime_state(job, art, known_pids))}


def collect(root, job, known_pids):
    """Read public metadata only after authoritative monitor completion."""
    root = Path(root)
    art = root / 'artifacts'
    require(job == make_job(job.get('owner'), job.get('plan_sha256'))
            and read(art / 'gpu-idle-queue.json') == job, 'writer collection ownership changed')
    state = runtime_state(job, art, known_pids)
    require((state.get('job') or {}).get('status') == 'finished', 'writer not finished')
    files = {}
    names = [f'gpu-idle-job-{ID}.json', f'gpu-idle-ticket-{ID}.consumed.json',
             marker_path(art).name, 'gpu-idle-queue.json']
    names.extend(f'gpu9-admission-{pid}.json' for pid in known_pids)
    for name in names:
        path = art / name
        if path.is_file() and not path.is_symlink():
            files[name] = {'sha256': digest(path), 'value': read(path)}
    report = root / 'artifacts' / ID / 'worker-report.json'
    require(report.is_file() and not report.is_symlink() and report.stat().st_size <= 1000000,
            'public worker report unavailable')
    value = read(report)
    forbidden = {'text', 'messages', 'response', 'candidate', 'experience', 'instruction',
                 'action', 'observation', 'rule'}

    def safe(item):
        if type(item) is dict:
            return not forbidden.intersection(item) and all(safe(v) for v in item.values())
        return type(item) is not list or all(safe(v) for v in item)

    require(type(value) is dict and safe(value), 'unexpected private worker content')
    require(value.get('id') == ID and value.get('plan_sha256') == job['plan_sha256'],
            'worker report binding differs')
    who = value.get('worker_identity')
    require(type(who) is dict and set(who) == {'pid', 'process_group', 'start_ticks'}
            and all(type(v) is int and v > 0 for v in who.values())
            and who['process_group'] == state['job'].get('process_group'),
            'worker report identity differs')
    known_pids = sorted({*known_pids, who['pid']})
    # The worker can start and finish between polls. Its persisted identity still
    # joins the independent GPU release query; the report alone never proves exit.
    admission_name = f"gpu9-admission-{who['pid']}.json"
    admission_path = art / admission_name
    if admission_path.is_file() and not admission_path.is_symlink():
        admission = read(admission_path)
        require(admission.get('pid') == who['pid']
                and admission.get('ticket') == state['job'].get('ticket')
                and (admission.get('ticket') or {}).get('job_id') == ID,
                'worker admission binding differs')
        files[admission_name] = {'sha256': digest(admission_path), 'value': admission}
    log = art / f'gpu-idle-job-{ID}.log'
    return {'worker_report': value, 'worker_report_sha256': digest(report), 'files': files,
            'observed_worker_pids': known_pids,
            'log': {'sha256': digest(log), 'size_bytes': log.stat().st_size} if log.is_file() else None}


REMOTE_SOURCE = '''import json,os,sys
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));os.chdir(root)
from scripts import queue_completion_lesson as q
p=json.load(sys.stdin)
if p['operation']=='collect': r=q.collect(root,p['job'],p['known_pids'])
else: r=q.queue_operation(p['operation'],p['job'],root=root,known_pids=p['known_pids'])
print(json.dumps(r,allow_nan=False))
'''


def remote_queue(operation, job, known_pids=()):
    try:
        return remote_python(REMOTE_SOURCE, {'operation': operation, 'job': job,
                                            'known_pids': sorted(known_pids)})
    except subprocess.TimeoutExpired:
        raise TimeoutError('writer_observation_timeout') from None
    except subprocess.CalledProcessError:
        raise RuntimeError('writer_remote_operation_failed') from None


def validate_preflight(value, sha, request_sha):
    require(type(value) is dict and value.get('id') == ID
            and value.get('status') == 'passed_preflight' and value.get('plan_sha256') == sha
            and value.get('request_sha256') == request_sha
            and value.get('training_authorized') is False and value.get('uses_gpu') is False
            and type(value.get('generation_calls')) is int and value['generation_calls'] == 0,
            'remote writer preflight differs')
    return {'preflight': 'passed', 'uses_gpu': False, 'request_sha256': request_sha}


def verify_remote(sha):
    source = '''import json,os,subprocess,sys
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));os.chdir(root)
from scripts.queue_completion_lesson import PYTHON,load_registration,validate_preflight
p=json.load(sys.stdin)
plan=load_registration(root,p['sha'])
r=subprocess.run([PYTHON,'-B','-m','scripts.run_completion_lesson','--plan-sha256',p['sha'],'--check'],
 cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=85)
assert r.returncode==0
print(json.dumps(validate_preflight(json.loads(r.stdout),p['sha'],plan['request_sha256'])))
'''
    try:
        result = subprocess.run([*SSH, 'CUDA_VISIBLE_DEVICES= python3 -B -c ' + shlex.quote(source)],
            input=json.dumps({'sha': sha}), capture_output=True, text=True, timeout=90, check=True)
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise TimeoutError('writer_preflight_timeout') from None
    except subprocess.CalledProcessError:
        raise RuntimeError('writer_preflight_failed') from None


def run(sha, check_only=False, *, root=ROOT):
    root = Path(root)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU controller required')
    load_registration(root, sha, native=True)
    checked = verify_remote(sha)
    require(checked.get('preflight') == 'passed', 'remote writer preflight incomplete')
    if check_only:
        return checked
    parent = root / 'artifacts' / ID
    parent.mkdir(exist_ok=True)
    stage = parent / 'controller'
    stage.mkdir(mode=0o700)
    job = make_job(uuid.uuid4().hex, sha)
    write_new(stage / 'job.json', job)
    known, interrupted = set(), None
    final = {'id': ID, 'plan_sha256': sha, 'status': 'unknown', 'released': False,
             'unknown_additional_response_cost': True, 'automatic_retry': False}
    started = time.monotonic()

    def observe(operation):
        value = remote_queue(operation, job, known)
        known.update(value.get('model_pids', []))
        known.update(value.get('observed_worker_pids', []))
        with (stage / 'observations.jsonl').open('a') as stream:
            stream.write(json.dumps({'operation': operation, 'epoch': time.time(), 'value': value})+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        return value

    try:
        snapshot = None
        try:
            snapshot = observe('enqueue')
        except Exception as exc:
            final['enqueue_observation_error'] = type(exc).__name__
        running = None
        while True:
            if snapshot is not None:
                require(snapshot.get('owned') is True and snapshot.get('monitor_live') is True,
                        'writer queue or monitor lost')
                info = snapshot.get('job') or {}
                if info.get('status') == 'finished':
                    final['job'] = info
                    evidence = observe('collect')
                    write_new(stage / 'remote-evidence.json', evidence)
                    final['worker_report'] = evidence['worker_report']
                    final['unknown_additional_response_cost'] = evidence['worker_report'].get(
                        'unknown_additional_response_cost', True)
                    if info.get('exit_code') == 0:
                        final['status'] = 'finished'
                    break
                if info.get('status') == 'running':
                    running = running or time.monotonic()
            require(time.monotonic()-started < WAIT_SECONDS + GPU_SECONDS + 120,
                    'writer controller hard deadline')
            require(running is not None or time.monotonic()-started < WAIT_SECONDS,
                    'writer admission wait expired')
            time.sleep(15)
            try:
                snapshot = observe('pulse' if snapshot is not None else 'status')
            except Exception as exc:
                final['observation_error'] = type(exc).__name__
    except BaseException as exc:
        final.update(status='unknown', error_type=type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupted = exc
    finally:
        deadline = time.monotonic()+120
        while time.monotonic() < deadline:
            try:
                released = observe('stop')
                if (released.get('owned') is True and released.get('released') is True
                        and released.get('heartbeat_absent') is True):
                    final['released'] = True
                    write_new(stage / 'release.json', released)
                    break
            except Exception as exc:
                final['cleanup_error_type'] = type(exc).__name__
            time.sleep(15)
        if not final['released']:
            final['status'] = 'unknown'
        final['finished_epoch'] = time.time()
        write_new(stage / 'final.json', final)
    if interrupted is not None:
        raise interrupted
    return final


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run(args.plan_sha256, check_only=args.check)
    print(json.dumps({'status': result.get('status'), 'preflight': result.get('preflight'),
                      'released': result.get('released')}))
    return 0 if args.check or result['status'] == 'finished' and result['released'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
