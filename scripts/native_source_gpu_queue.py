"""Three fixed native source-pilot leases using the frozen B3 GPU queue.

Only public source and registration metadata are read by this adapter. The
private CPU preflight is separate. Import and CLI help do not start a monitor,
reserve a job, contact a host, bind a port, or initialize a model.
"""

import argparse
import hashlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
SOURCE = ROOT / 'scripts/completion_gpu_queue.py'
SOURCE_SHA256 = '928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8'
ID = 'qwen38-native-source-pilot-001'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-001.json'
PORT = 18991
SERVICE_PLAN = 'artifacts/qwen38-b3-service-plan-001.json'
SERVICE_SHA = 'd7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5'
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200


def make_job(index, owner, plan_sha):
    if not (type(index) is int and 0 <= index < 3
            and type(owner) is str and re.fullmatch('[a-f0-9]{32}', owner)
            and type(plan_sha) is str and re.fullmatch('[a-f0-9]{64}', plan_sha)):
        raise ValueError('invalid native source-pilot job')
    return {'id': f'{ID}-comparison-{index}', 'kind': 'comparison', 'batch_index': index,
            'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': BATCH_GPU_SECONDS,
            'argv': [PYTHON, '-B', '-m', 'scripts.native_source_gpu_queue', '--serve',
                     '--plan-sha256', plan_sha, '--batch', str(index), '--owner', owner]}


def load_registration(root, sha):
    # GPU admission is public-only; private asset checks belong to CPU preflight.
    from scripts import native_source_registration as registration

    if (registration.ID, registration.PLAN, registration.PORT, registration.SERVICE_PLAN,
            registration.SERVICE_SHA, registration.PYTHON, registration.WAIT_SECONDS,
            registration.BATCH_GPU_SECONDS) != (
            ID, PLAN, PORT, SERVICE_PLAN, SERVICE_SHA, PYTHON, WAIT_SECONDS, BATCH_GPU_SECONDS):
        raise ValueError('native source-pilot registration constants differ')
    plan = registration.load_registration(root, sha, private=False)
    if (type(plan) is not dict or type(plan.get('source_sha256')) is not dict
            or plan['source_sha256'].get('scripts/completion_gpu_queue.py') != SOURCE_SHA256):
        raise ValueError('frozen queue source binding differs')
    return plan


def _isolated(*, raw=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/completion_gpu_queue.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('frozen queue source changed')
    spec = importlib.util.spec_from_file_location('scripts._native_source_completion_queue', SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError('frozen queue source unavailable')
    core = importlib.util.module_from_spec(spec)
    # These are authenticated repository bytes, never model-generated code.
    exec(compile(raw, str(SOURCE), 'exec', dont_inherit=True), core.__dict__)
    if (core.REMOTE, core.PYTHON, core.SERVICE_PLAN, core.SERVICE_SHA, core.WAIT_SECONDS) != (
            REMOTE, PYTHON, SERVICE_PLAN, SERVICE_SHA, WAIT_SECONDS):
        raise ValueError('frozen queue service constants differ')
    core.ID, core.PLAN, core.PORT = ID, PLAN, PORT
    core.make_job, core.load_registration = make_job, load_registration
    return core


_CORE = None


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    # Numeric equality alone would let 7200.0 compare equal to 7200, while the
    # monitor correctly requires an integer before issuing a ticket.
    if type(job) is not dict or type(job.get('max_runtime_seconds')) is not int:
        raise ValueError('invalid native source-pilot job')
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
    if type(job) is not dict or type(job.get('max_runtime_seconds')) is not int:
        raise ValueError('invalid native source-pilot job')
    return _get_core().serve(job, root=root)


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
