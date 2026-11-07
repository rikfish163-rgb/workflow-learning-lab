"""Four controlled-fault batches on the pinned existing serial admission core.

Import, job construction and help do not operate the queue or load a model.
"""
import hashlib
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).with_name('codeact_within_gpu_queue_001.py')
SOURCE_SHA256 = '11ddd2f3b4985bee5470e9421b10f5f6a45d9cf57ec649f3911a2c4532e7165e'


def _load(*, raw=None):
    if raw is None:
        if SOURCE.is_symlink():
            raise ValueError('codeact_fault_queue_source')
        raw = SOURCE.read_bytes()
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('codeact_fault_queue_source')
    source = raw.decode('utf-8')
    for old, new in (
        ('qwen38-codeact-within-source-001', 'qwen38-codeact-fault-source-001'),
        ('qwen38-codeact-within-source-plan-001.json', 'qwen38-codeact-fault-source-plan-001.json'),
        ('18992', '18995'),
        ('scripts.codeact_within_gpu_queue_001', 'scripts.codeact_fault_gpu_queue_001'),
        ('scripts.codeact_within_registration_001', 'scripts.codeact_fault_registration_001'),
    ):
        if source.count(old) != 1:
            raise ValueError('codeact_fault_queue_transform')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._codeact_fault_gpu_queue_core', SOURCE)
    core = importlib.util.module_from_spec(spec)
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


_SOURCE_CORE = _load()
ID, PLAN, PORT = _SOURCE_CORE.ID, _SOURCE_CORE.PLAN, _SOURCE_CORE.PORT
BATCH_COUNT, TASKS_PER_BATCH = _SOURCE_CORE.BATCH_COUNT, _SOURCE_CORE.TASKS_PER_BATCH
WAIT_SECONDS, GPU_SECONDS, CLIENT_SECONDS = _SOURCE_CORE.WAIT_SECONDS, _SOURCE_CORE.GPU_SECONDS, _SOURCE_CORE.CLIENT_SECONDS
MODEL_START_SECONDS, CLEANUP_SECONDS = _SOURCE_CORE.MODEL_START_SECONDS, _SOURCE_CORE.CLEANUP_SECONDS
ROOT, REMOTE, PYTHON = _SOURCE_CORE.ROOT, _SOURCE_CORE.REMOTE, _SOURCE_CORE.PYTHON
MODEL_IDS = dict(_SOURCE_CORE.MODEL_IDS)
make_job, load_registration = _SOURCE_CORE.make_job, _SOURCE_CORE.load_registration
queue_operation, record_gpu_entry = _SOURCE_CORE.queue_operation, _SOURCE_CORE.record_gpu_entry
serve, main, _isolated = _SOURCE_CORE.serve, _SOURCE_CORE.main, _SOURCE_CORE._isolated


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_SOURCE_CORE, name)


if __name__ == '__main__':
    main()
