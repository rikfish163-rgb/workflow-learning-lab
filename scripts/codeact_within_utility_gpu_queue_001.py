"""Eighteen future CodeAct utility leases using isolated writer admission code.

The adapter preserves the global GPU queue, independent idle ticket, operation
guard, token-count service and release fence. Import and job construction do not
register a study, contact a service, or operate a queue or GPU.
"""

import argparse
import hashlib
import importlib.util
import re
import threading
from importlib import import_module
from pathlib import Path

from scripts import coverage_contract_gpu_queue as previous

ID = 'qwen38-codeact-within-utility-001'
PLAN = 'artifacts/qwen38-codeact-within-utility-plan-001.json'
PORT, BATCH_COUNT, TASKS_PER_BATCH = 18994, 18, 3
WAIT_SECONDS, GPU_SECONDS, CLIENT_SECONDS = 86400, 7200, 6000
MODEL_START_SECONDS, CLEANUP_SECONDS = 900, 180
BATCH_GPU_SECONDS = GPU_SECONDS
GUARD_NAME = 'codeact-within-utility-operations-001.lock'
WRITER_SOURCE = Path(__file__).with_name('codeact_within_writer_gpu_queue_001.py')
WRITER_SHA256 = 'd856646b4698a08c19b999b8c90546472327667f5d01143da2e9599fbc4dc9c1'
BASE_SHA256 = '928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8'
FENCE_SHA256 = 'ae221cd0ad1a88c2f4833cb605c501e831c5068ad8e3aeca4491f02c31881419'
ROOT, REMOTE, PYTHON = previous.ROOT, previous.REMOTE, previous.PYTHON
SERVICE_PLAN, SERVICE_SHA = previous.SERVICE_PLAN, previous.SERVICE_SHA
GATE, GATE_SHA = previous.GATE, previous.GATE_SHA
GATE_B3, GATE_B3_SHA = previous.GATE_B3, previous.GATE_B3_SHA
MODEL_IDS = dict(previous.MODEL_IDS)
_ADAPTER = None
_ADAPTER_LOCK = threading.Lock()
_ERROR = 'codeact_within_utility_gpu_queue_invalid'


def make_job(index, owner, plan_sha):
    if (type(index) is not int or not 0 <= index < BATCH_COUNT
            or type(owner) is not str or re.fullmatch(r'[a-f0-9]{32}', owner) is None
            or type(plan_sha) is not str or re.fullmatch(r'[a-f0-9]{64}', plan_sha) is None):
        raise ValueError(_ERROR) from None
    return {'id': f'{ID}-comparison-{index}', 'kind': 'comparison', 'batch_index': index,
            'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': GPU_SECONDS,
            'argv': [PYTHON, '-B', '-m', 'scripts.codeact_within_utility_gpu_queue_001', '--serve',
                     '--plan-sha256', plan_sha, '--batch', str(index), '--owner', owner]}


def _registration():
    return import_module('scripts.codeact_within_utility_registration_001')


def _isolated(*, raw=None):
    if raw is None:
        raw = previous.public_read(ROOT, 'scripts/codeact_within_writer_gpu_queue_001.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != WRITER_SHA256:
        raise ValueError(_ERROR) from None
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_utility_queue_base', WRITER_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR) from None
    module = importlib.util.module_from_spec(spec)
    # Only the hash-bound human-authored writer queue is compiled.
    exec(compile(raw, str(WRITER_SOURCE), 'exec', dont_inherit=True), module.__dict__)
    if (module.BASE_SHA256 != BASE_SHA256 or module.FENCE_SHA256 != FENCE_SHA256
            or module.WAIT_SECONDS != WAIT_SECONDS or module.GPU_SECONDS != GPU_SECONDS
            or module.RELEASE_SECONDS != 180.0):
        raise ValueError(_ERROR) from None
    module.ID, module.PLAN, module.PORT, module.GUARD_NAME = ID, PLAN, PORT, GUARD_NAME
    module.BATCH_COUNT, module.TASKS_PER_BATCH = BATCH_COUNT, TASKS_PER_BATCH
    module.CLIENT_SECONDS, module.MODEL_START_SECONDS, module.CLEANUP_SECONDS = (
        CLIENT_SECONDS, MODEL_START_SECONDS, CLEANUP_SECONDS)
    module.MODEL_IDS = dict(MODEL_IDS)
    module.make_job, module._registration = make_job, _registration
    return module


def _get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        with _ADAPTER_LOCK:
            if _ADAPTER is None:
                _ADAPTER = _isolated()
    return _ADAPTER


def _get_core():
    return _get_adapter()._get_core()


def _get_fence():
    return _get_adapter()._get_fence()


def _operation_guard(root):
    return _get_adapter()._operation_guard(root)


def load_registration(root, sha, *, private=False):
    return _get_adapter().load_registration(root, sha, private=private)


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    return _get_adapter().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    return _get_adapter().record_gpu_entry(root, job, gpu)


def fenced_release(runtime, job, known_pids):
    return _get_adapter().fenced_release(runtime, job, known_pids)


def serve(job, *, root=ROOT):
    return _get_adapter().serve(job, root=root)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_adapter(), name)


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
