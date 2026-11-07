"""V2 source-pilot queue with a serialized admission guard and release fence.

The frozen V1 queue and its underlying completion queue run in independent,
authenticated globals. Import does not reserve a job or initialize a model.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
BASE_SOURCE = ROOT / 'scripts/native_source_gpu_queue.py'
BASE_SHA256 = '4417777dae12bb28a2f1a5d08c54b0327312e2ce24597b95d2f4057f4c6567e0'
COMPLETION_SHA256 = '928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8'
ID = 'qwen38-native-source-pilot-002'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-002.json'
PORT = 18991
SERVICE_PLAN = 'artifacts/qwen38-b3-service-plan-001.json'
SERVICE_SHA = 'd7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5'
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200
GUARD_NAME = 'native-source-operations-002.lock'
GUARD_WAIT_SECONDS = 10.0
RELEASE_SECONDS = 90.0
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_GUARD_FLAGS = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class ReleaseFenceError(RuntimeError):
    """Release could not be proved; the caller must keep the outcome unknown."""


def make_job(index, owner, plan_sha):
    if not (type(index) is int and 0 <= index < 3
            and type(owner) is str and re.fullmatch('[a-f0-9]{32}', owner)
            and type(plan_sha) is str and re.fullmatch('[a-f0-9]{64}', plan_sha)):
        raise ValueError('invalid native source-pilot V2 job')
    return {'id': f'{ID}-comparison-{index}', 'kind': 'comparison', 'batch_index': index,
            'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': BATCH_GPU_SECONDS,
            'argv': [PYTHON, '-B', '-m', 'scripts.native_source_gpu_queue_002', '--serve',
                     '--plan-sha256', plan_sha, '--batch', str(index), '--owner', owner]}


def _validate_job(job):
    if (type(job) is not dict or type(job.get('max_runtime_seconds')) is not int
            or job != make_job(job.get('batch_index'), job.get('owner'), job.get('plan_sha256'))):
        raise ValueError('invalid native source-pilot V2 job')


def load_registration(root, sha):
    from scripts import native_source_registration_002 as registration

    if (registration.ID, registration.PLAN, registration.PORT, registration.SERVICE_PLAN,
            registration.SERVICE_SHA, registration.PYTHON, registration.WAIT_SECONDS,
            registration.BATCH_GPU_SECONDS) != (
            ID, PLAN, PORT, SERVICE_PLAN, SERVICE_SHA, PYTHON, WAIT_SECONDS, BATCH_GPU_SECONDS):
        raise ValueError('native source-pilot V2 registration constants differ')
    plan = registration.load_registration(root, sha, private=False)
    if (type(plan) is not dict or type(plan.get('source_sha256')) is not dict
            or plan['source_sha256'].get('scripts/native_source_gpu_queue.py') != BASE_SHA256
            or plan['source_sha256'].get('scripts/completion_gpu_queue.py') != COMPLETION_SHA256):
        raise ValueError('frozen queue source bindings differ')
    return plan


def _isolated_base(*, raw=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/native_source_gpu_queue.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('frozen native source queue changed')
    spec = importlib.util.spec_from_file_location('scripts._native_source_gpu_queue_002_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError('frozen native source queue unavailable')
    base = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    if (base.PYTHON, base.REMOTE, base.SERVICE_PLAN, base.SERVICE_SHA,
            base.WAIT_SECONDS, base.BATCH_GPU_SECONDS) != (
            PYTHON, Path('/home/hts/workflow-learning-lab'), SERVICE_PLAN, SERVICE_SHA,
            WAIT_SECONDS, BATCH_GPU_SECONDS):
        raise ValueError('frozen native source queue constants differ')
    base.ID, base.PLAN, base.PORT = ID, PLAN, PORT
    base.make_job, base.load_registration = make_job, load_registration
    return base


def _open_artifacts(root):
    path = Path(root)
    if not path.is_absolute() or '..' in path.parts or str(path) != os.path.normpath(str(path)):
        raise ValueError('canonical absolute root required')
    descriptor = os.open('/', _DIR_FLAGS)
    try:
        for part in (*path.parts[1:], 'artifacts'):
            child = os.open(part, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _guard_identity(artifacts_fd, descriptor):
    info = os.fstat(descriptor)
    name = os.stat(GUARD_NAME, dir_fd=artifacts_fd, follow_symlinks=False)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (name.st_dev, name.st_ino)
            or not stat.S_ISREG(name.st_mode)):
        raise ValueError('invalid native source V2 operations guard')


@contextlib.contextmanager
def _operation_guard(root):
    artifacts_fd = _open_artifacts(root)
    descriptor = None
    locked = False
    try:
        descriptor = os.open(GUARD_NAME, _GUARD_FLAGS, 0o600, dir_fd=artifacts_fd)
        _guard_identity(artifacts_fd, descriptor)
        deadline = time.monotonic() + min(10.0, GUARD_WAIT_SECONDS)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('native source V2 guard unavailable') from None
                time.sleep(min(0.05, remaining))
        _guard_identity(artifacts_fd, descriptor)
        yield
    finally:
        if descriptor is not None:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        os.close(artifacts_fd)


_CORE = None
_INIT_LOCK = threading.Lock()


def _get_core():
    global _CORE
    with _INIT_LOCK:
        if _CORE is None:
            base = _isolated_base()
            core = base._get_core()
            original_entry = core.record_gpu_entry

            def guarded_entry(root, job, gpu):
                _validate_job(job)
                with _operation_guard(root):
                    return original_entry(root, job, gpu)

            core.record_gpu_entry = guarded_entry
            _CORE = core
        return _CORE


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    if operation not in {'enqueue', 'pulse', 'stop', 'status'}:
        raise ValueError('invalid queue operation')
    _validate_job(job)
    with _operation_guard(root):
        return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def check_admission_window(root, job):
    return _get_core().check_admission_window(root, job)


def record_gpu_entry(root, job, gpu):
    return _get_core().record_gpu_entry(root, job, gpu)


def window_path(art, job):
    return _get_core().window_path(art, job)


def entry_path(art, job):
    return _get_core().entry_path(art, job)


def cancel_path(art, job):
    return _get_core().cancel_path(art, job)


def ready_path(art, job):
    return _get_core().ready_path(art, job)


def runtime_state(job, art, known_pids=()):
    return _get_core().runtime_state(job, art, known_pids)


def serve(job, *, root=ROOT):
    _validate_job(job)
    return _get_core().serve(job, root=root)


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _read_json_file(artifacts_fd, name):
    descriptor = os.open(name, _READ_FLAGS, dir_fd=artifacts_fd)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or not 0 <= before.st_size <= 1_000_000):
            raise ValueError('invalid queue metadata')
        raw = bytearray()
        while True:
            block = os.read(descriptor, 65536)
            if not block:
                break
            raw.extend(block)
            if len(raw) > 1_000_000:
                raise ValueError('invalid queue metadata')
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=artifacts_fd, follow_symlinks=False)
        def identity(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

        if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError('queue metadata changed')
        def unique_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('duplicate queue metadata field')
                value[key] = item
            return value

        def reject_constant(_value):
            raise ValueError('invalid queue metadata constant')

        value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
        return value, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _exists(artifacts_fd, name):
    try:
        os.stat(name, dir_fd=artifacts_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _release_files(job):
    art_fd = _open_artifacts(REMOTE)
    try:
        current, digest = _read_json_file(art_fd, 'gpu-idle-queue.json')
        cancel, _ = _read_json_file(art_fd, f"completion-admission-cancelled-{job['id']}.json")
        fingerprint = _get_core().fingerprint(job)
        if (current != job or type(cancel) is not dict or cancel.get('job_sha256') != fingerprint
                or cancel.get('reason') not in {'controller_stop', 'admission_window_unavailable'}
                or not _finite(cancel.get('epoch'))):
            raise ValueError('queue or cancellation binding changed')
        prefix = job['id']
        names = {'heartbeat': f'gpu-idle-heartbeat-{prefix}',
                 'ticket': f'gpu-idle-ticket-{prefix}.json',
                 'consumed': f'gpu-idle-ticket-{prefix}.consumed.json',
                 'result': f'gpu-idle-job-{prefix}.json',
                 'entry': f'completion-GPU-entry-{prefix}.json',
                 'ready': f'completion-service-ready-{prefix}.json'}
        markers = {key: _exists(art_fd, name) for key, name in names.items()}
        result = _read_json_file(art_fd, names['result'])[0] if markers['result'] else None
        return digest, markers, result
    finally:
        os.close(art_fd)


def _monitor_identity(runtime, expected=None):
    info = runtime.monitor()
    if (type(info) is not dict or type(info.get('pid')) is not int or info['pid'] <= 0
            or type(info.get('start_ticks')) is not str or not info['start_ticks'].isdigit()
            or not _finite(info.get('checked_epoch'))):
        raise ReleaseFenceError('monitor identity unavailable')
    identity = (info['pid'], info['start_ticks'], _boot_id())
    if expected is not None and identity != expected:
        raise ReleaseFenceError('monitor identity changed')
    return identity, info['checked_epoch']


def fenced_release(runtime, job, known_pids):
    """Cancel, then prove completion or two new clean watcher loop publications."""
    _validate_job(job)
    try:
        try:
            started = runtime.now()
            identity, _ = _monitor_identity(runtime)
        except BaseException:
            # An observation failure must not leave a still-fresh heartbeat behind.
            try:
                runtime.queue('stop', job, known_pids)
            except BaseException as cancel_error:
                try:
                    runtime.record({'event': 'release_fence_cancel_failure',
                                    'error_type': type(cancel_error).__name__,
                                    'traceback': ''.join(traceback.format_exception(cancel_error))})
                except BaseException:
                    pass
            raise
        state = runtime.queue('stop', job, known_pids)
        cancelled_epoch = runtime.epoch()
        if not _finite(started) or not _finite(cancelled_epoch):
            raise ReleaseFenceError('invalid release clock')
        last_witness, witnesses, attempted = cancelled_epoch, 0, False
        runtime.record({'event': 'release_fence_started', 'job_sha256': _get_core().fingerprint(job),
                        'monitor_pid': identity[0], 'monitor_start_ticks': identity[1],
                        'boot_id': identity[2], 'cancel_epoch': cancelled_epoch})
        while True:
            _, monitor_checked = _monitor_identity(runtime, identity)
            now, epoch = runtime.now(), runtime.epoch()
            if (not _finite(now) or now < started or now - started > RELEASE_SECONDS
                    or not _finite(epoch) or type(state) is not dict
                    or state.get('owned') is not True or state.get('admission_cancelled') is not True
                    or state.get('monitor_live') is not True or not _finite(state.get('epoch'))
                    or not 0 <= epoch - state['epoch'] < 5):
                raise ReleaseFenceError('release state unavailable')
            queue_sha, markers, disk_result = _release_files(job)
            if markers['heartbeat']:
                raise ReleaseFenceError('heartbeat returned after cancellation')
            info = state.get('job')
            attempted = attempted or info is not None or any(markers[key] for key in (
                'ticket', 'consumed', 'result', 'entry', 'ready'))
            if info is not None and state.get('released') is True:
                if (type(info) is not dict or not markers['result'] or disk_result != info
                        or info.get('status') != 'finished'
                        or info.get('job_id') != job['id'] or info.get('queue_sha256') != queue_sha
                        or state.get('model_pids') != [] or not _finite(info.get('started_epoch'))
                        or not _finite(info.get('finished_epoch'))
                        or not 0 <= info['started_epoch'] <= info['finished_epoch'] <= state['epoch']):
                    raise ReleaseFenceError('finished job release is unverified')
                duration = info['finished_epoch'] - info['started_epoch']
                return {'verified': True, 'GPU_seconds': duration, 'service_wall_seconds': duration,
                        'cuda_seconds': None, 'runtime_overrun': duration > BATCH_GPU_SECONDS,
                        'state': state, 'release_wall_seconds': now - started}
            watch = state.get('watch')
            if (type(watch) is not dict or watch.get('pid') != identity[0]
                    or not _finite(watch.get('checked_epoch'))
                    or not 0 <= epoch - watch['checked_epoch'] < 35
                    or watch['checked_epoch'] > monitor_checked
                    or watch.get('phase') == 'observation_error'):
                raise ReleaseFenceError('watcher publication unavailable')
            if (not attempted and info is None and state.get('model_pids') == []
                    and watch.get('phase') == 'monitoring' and watch.get('queued_job') == job['id']
                    and watch.get('client_ready') is False and watch.get('already_attempted') is False
                    and _finite(watch.get('checked_epoch')) and watch['checked_epoch'] > last_witness):
                last_witness = watch['checked_epoch']
                witnesses += 1
                runtime.record({'event': 'release_fence_monitor_witness', 'job_id': job['id'],
                                'checked_epoch': last_witness, 'witness': witnesses})
                if witnesses >= 2:
                    return {'verified': True, 'GPU_seconds': 0, 'service_wall_seconds': 0,
                            'cuda_seconds': 0, 'runtime_overrun': False,
                            'state': state, 'release_wall_seconds': now - started}
            remaining = RELEASE_SECONDS - (now - started)
            if remaining <= 0:
                raise ReleaseFenceError('release fence timed out')
            runtime.wait(min(5.0, remaining))
            state = runtime.queue('status', job, known_pids)
    except BaseException as error:
        try:
            runtime.record({'event': 'release_fence_failure', 'error_type': type(error).__name__,
                            'traceback': ''.join(traceback.format_exception(error))})
        except BaseException:
            pass  # Preserve the original release error even if the private journal fails.
        if isinstance(error, ReleaseFenceError) or not isinstance(error, Exception):
            raise
        raise ReleaseFenceError('release fence could not be verified') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args(argv)
    serve(make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == '__main__':
    main()
