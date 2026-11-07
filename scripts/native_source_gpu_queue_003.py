"""V3 queue namespace over authenticated V2 guard and release-fence code."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_source_gpu_queue_002.py'
BASE_SHA256 = 'ae221cd0ad1a88c2f4833cb605c501e831c5068ad8e3aeca4491f02c31881419'
ID = 'qwen38-native-source-pilot-003'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-003.json'
PORT = 18991
REMOTE = Path('/home/hts/workflow-learning-lab')
GUARD_NAME = 'native-source-operations-003.lock'
_CORE = None


def _isolated(*, raw=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/native_source_gpu_queue_002.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('native_source_gpu_queue_003_source_invalid')
    source = raw.decode('utf-8')
    for old, new in (
        ('qwen38-native-source-pilot-002', ID),
        ('artifacts/qwen38-native-source-pilot-plan-002.json', PLAN),
        ('native-source-operations-002.lock', GUARD_NAME),
        ('scripts.native_source_gpu_queue_002', 'scripts.native_source_gpu_queue_003'),
        ('native_source_registration_002', 'native_source_registration_003'),
        ('_native_source_gpu_queue_002_base', '_native_source_gpu_queue_003_base'),
    ):
        if source.count(old) != 1:
            raise ValueError('native_source_gpu_queue_003_binding_invalid')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._native_source_gpu_queue_003', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError('native_source_gpu_queue_003_source_invalid')
    core = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    load = core.load_registration

    def bound_registration(root, sha):
        plan = load(root, sha)
        if plan['source_sha256'].get('scripts/native_source_gpu_queue_002.py') != BASE_SHA256:
            raise ValueError('V2 queue source binding differs')
        return plan

    core.load_registration = bound_registration
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def make_job(index, owner, plan_sha):
    return _get_core().make_job(index, owner, plan_sha)


def load_registration(root, sha):
    return _get_core().load_registration(root, sha)


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    return _get_core().record_gpu_entry(root, job, gpu)


def fenced_release(runtime, job, known_pids):
    return _get_core().fenced_release(runtime, job, known_pids)


def serve(job, *, root=ROOT):
    return _get_core().serve(job, root=root)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owner', required=True)
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == '__main__':
    main()
