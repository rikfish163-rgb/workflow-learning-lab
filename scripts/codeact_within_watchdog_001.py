"""Fixed source slots over the unchanged independent CPU/container watchdog."""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

from scripts import codeact_within_registration_001 as registration

STUDY_ID = registration.ID
FROZEN_SHA256 = '6f02c121f9840e471768aa7b20d1d5bb3fba526e2460bc38c09fc5f3b1b647e7'


def client_command(stage, plan_sha):
    stage = Path(stage)
    if (type(plan_sha) is not str or re.fullmatch('[0-9a-f]{64}', plan_sha) is None
            or stage not in [registration.OUTPUT / f'batch-{i}' for i in range(4)]):
        raise ValueError('codeact_watchdog_binding')
    request = stage / 'request.json'
    if not request.is_file() or request.is_symlink():
        raise ValueError('codeact_watchdog_request')
    return [sys.executable, '-u', '-B', '-m', 'scripts.run_codeact_within_source_001',
        '--plan-sha256', plan_sha, '--client-request', str(request),
        '--client-request-sha256', hashlib.sha256(request.read_bytes()).hexdigest()]


def _slot(row):
    if (type(row) is not dict or 'stage' in row
            or type(row.get('task_id')) is not str or row['task_id'] not in registration.TASK_IDS
            or row.get('arm') != 'first_attempt' or row.get('phase') != 'source'):
        raise ValueError('codeact_watchdog_slot')
    return row['task_id'], row['arm']


def _isolated():
    source = Path(__file__).with_name('completion_watchdog.py')
    raw = source.read_bytes()
    if source.is_symlink() or hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError('codeact_watchdog_source')
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_watchdog_001', source)
    if spec is None or spec.loader is None:
        raise ImportError('codeact_watchdog_source')
    module = importlib.util.module_from_spec(spec)
    # Fixed hash-bound human implementation, never actor Python.
    exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)
    module.STUDY_ID, module.client_command, module._slot = STUDY_ID, client_command, _slot
    return module


_CORE = _isolated()
_base_cleanup = _CORE.cleanup_containers


def cleanup_containers(stage, who):
    # Always attempt exact owned cleanup first, even when its journal is bad.
    result = _base_cleanup(stage, who)
    path = Path(stage) / 'run/events.jsonl'
    try:
        raw = path.read_bytes() if path.is_file() and not path.is_symlink() else b''
        complete_journal = bool(raw) and raw.endswith(b'\n')
    except OSError:
        complete_journal = False
    if not complete_journal:
        result['errors'].append('missing_or_incomplete_event_journal')
        result['verified'] = False
    return result


_CORE.cleanup_containers = cleanup_containers
watch, recover = _CORE.watch, _CORE.recover
