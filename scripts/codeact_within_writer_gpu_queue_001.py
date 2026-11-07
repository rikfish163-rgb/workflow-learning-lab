"""One future CodeAct writer lease over the existing isolated GPU admission core.

Import, help and job construction do not register, enqueue, or start a model.
The caller separately authenticates private writer inputs and verifies release.
"""

import argparse
import hashlib
import importlib.util
import re
import threading
from importlib import import_module
from pathlib import Path

from scripts import coverage_contract_gpu_queue as previous

ID = 'qwen38-codeact-within-writer-001'
PLAN = 'artifacts/qwen38-codeact-within-writer-plan-001.json'
PORT, BATCH_COUNT = 18993, 1
WAIT_SECONDS, GPU_SECONDS = 86400, 7200
BATCH_GPU_SECONDS = GPU_SECONDS
BASE_SHA256 = '928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8'
FENCE_SHA256 = 'ae221cd0ad1a88c2f4833cb605c501e831c5068ad8e3aeca4491f02c31881419'
FENCE_SOURCE = Path(__file__).with_name('native_source_gpu_queue_002.py')
GUARD_NAME = 'codeact-within-writer-operations-001.lock'
GUARD_WAIT_SECONDS, RELEASE_SECONDS = 10.0, 180.0
ROOT, REMOTE, PYTHON = previous.ROOT, previous.REMOTE, previous.PYTHON
SERVICE_PLAN, SERVICE_SHA = previous.SERVICE_PLAN, previous.SERVICE_SHA
GATE, GATE_SHA = previous.GATE, previous.GATE_SHA
GATE_B3, GATE_B3_SHA = previous.GATE_B3, previous.GATE_B3_SHA
MODEL_IDS = dict(previous.MODEL_IDS)
_CORE = None
_FENCE = None
_CORE_LOCK = threading.Lock()
_FENCE_LOCK = threading.Lock()
_ERROR = 'codeact_within_writer_gpu_queue_invalid'


def make_job(index, owner, plan_sha):
    if (type(index) is not int or index != 0
            or type(owner) is not str or re.fullmatch(r'[a-f0-9]{32}', owner) is None
            or type(plan_sha) is not str or re.fullmatch(r'[a-f0-9]{64}', plan_sha) is None):
        raise ValueError(_ERROR) from None
    return {'id': f'{ID}-comparison-0', 'kind': 'comparison', 'batch_index': 0,
            'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': GPU_SECONDS,
            'argv': [PYTHON, '-B', '-m', 'scripts.codeact_within_writer_gpu_queue_001', '--serve',
                     '--plan-sha256', plan_sha, '--batch', '0', '--owner', owner]}


def _registration():
    return import_module('scripts.codeact_within_writer_registration_001')


def load_registration(root, sha, *, private=False):
    if private is not False:
        raise ValueError(_ERROR) from None
    return _registration().load_registration(root, sha, private=False)


def _isolated(*, raw=None):
    if previous.SOURCE_SHA256 != BASE_SHA256:
        raise ValueError(_ERROR) from None
    core = previous._isolated(raw=raw)
    core.ID, core.PLAN, core.PORT, core.WAIT_SECONDS = ID, PLAN, PORT, WAIT_SECONDS
    core.MODEL_IDS = dict(MODEL_IDS)
    core.make_job, core.load_registration = make_job, load_registration
    original_entry = core.record_gpu_entry

    def guarded_entry(root, job, gpu):
        _validate_job(job)
        with _operation_guard(root):
            return original_entry(root, job, gpu)

    core.record_gpu_entry = guarded_entry
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        with _CORE_LOCK:
            if _CORE is None:
                _CORE = _isolated()
    return _CORE


def _validate_job(job):
    if (type(job) is not dict or set(job) != {'id', 'kind', 'batch_index', 'owner', 'plan_sha256',
                                             'max_runtime_seconds', 'argv'}
            or type(job['id']) is not str or type(job['kind']) is not str
            or type(job['batch_index']) is not int or type(job['owner']) is not str
            or type(job['plan_sha256']) is not str or type(job['max_runtime_seconds']) is not int
            or type(job['argv']) is not list or any(type(value) is not str for value in job['argv'])
            or job != make_job(job['batch_index'], job['owner'], job['plan_sha256'])):
        raise ValueError(_ERROR) from None


def _isolated_fence(*, raw=None):
    if raw is None:
        raw = previous.public_read(ROOT, 'scripts/native_source_gpu_queue_002.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != FENCE_SHA256:
        raise ValueError(_ERROR) from None
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_writer_release_fence', FENCE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR) from None
    module = importlib.util.module_from_spec(spec)
    # Hash-bound human repository code only; no actor or model text is compiled.
    exec(compile(raw, str(FENCE_SOURCE), 'exec', dont_inherit=True), module.__dict__)
    module.ROOT, module.REMOTE, module.ID, module.PLAN, module.PORT = ROOT, REMOTE, ID, PLAN, PORT
    module.WAIT_SECONDS, module.BATCH_GPU_SECONDS = WAIT_SECONDS, BATCH_GPU_SECONDS
    module.GUARD_NAME, module.GUARD_WAIT_SECONDS, module.RELEASE_SECONDS = (
        GUARD_NAME, GUARD_WAIT_SECONDS, RELEASE_SECONDS)
    module.make_job, module._validate_job, module._get_core = make_job, _validate_job, _get_core
    return module


def _get_fence():
    global _FENCE
    if _FENCE is None:
        with _FENCE_LOCK:
            if _FENCE is None:
                _FENCE = _isolated_fence()
    return _FENCE


def _operation_guard(root):
    return _get_fence()._operation_guard(root)


def _check_predecessor(root, job):
    plan = load_registration(root, job['plan_sha256'])
    if _registration().check_predecessor(root, plan) is not True:
        raise ValueError(_ERROR) from None


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    if type(operation) is not str or operation not in {'enqueue', 'pulse', 'stop', 'status'}:
        raise ValueError(_ERROR) from None
    _validate_job(job)
    if operation == 'enqueue':
        _check_predecessor(root, job)  # A failed preflight does not create even the guard file.
    with _operation_guard(root):
        if operation == 'enqueue':
            _check_predecessor(root, job)
        # The inherited operation rechecks registration under the existing global lock.
        return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    _validate_job(job)
    return _get_core().record_gpu_entry(root, job, gpu)


def fenced_release(runtime, job, known_pids):
    _validate_job(job)
    return _get_fence().fenced_release(runtime, job, known_pids)


def serve(job, *, root=ROOT):
    _validate_job(job)
    return _get_core().serve(job, root=root)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    if name == 'ReleaseFenceError':
        return _get_fence().ReleaseFenceError
    return getattr(_get_core(), name)


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
