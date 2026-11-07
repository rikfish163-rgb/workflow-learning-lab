"""Controlled-fault slots over the pinned independent cleanup watchdog."""
import hashlib
import importlib
import importlib.util
import re
import sys
from pathlib import Path

STUDY_ID = 'qwen38-codeact-fault-source-001'
OUTPUT = Path(__file__).resolve().parents[1] / 'artifacts' / STUDY_ID
FROZEN_SHA256 = '6f02c121f9840e471768aa7b20d1d5bb3fba526e2460bc38c09fc5f3b1b647e7'


def _registration():
    return importlib.import_module('scripts.codeact_fault_registration_001')


def client_command(stage, plan_sha):
    stage = Path(stage)
    if (type(plan_sha) is not str or re.fullmatch('[0-9a-f]{64}', plan_sha) is None
            or stage not in [OUTPUT / f'batch-{i}' for i in range(4)]):
        raise ValueError('codeact_fault_watchdog_binding')
    request = stage / 'request.json'
    if not request.is_file() or request.is_symlink():
        raise ValueError('codeact_fault_watchdog_request')
    return [sys.executable, '-u', '-B', '-m', 'scripts.run_codeact_fault_source_001',
            '--plan-sha256', plan_sha, '--client-request', str(request),
            '--client-request-sha256', hashlib.sha256(request.read_bytes()).hexdigest()]


def _slot(row):
    if (type(row) is not dict or 'stage' in row or type(row.get('task_id')) is not str
            or row['task_id'] not in _registration().TASK_IDS
            or row.get('arm') != 'first_attempt' or row.get('phase') != 'controlled_fault_source'):
        raise ValueError('codeact_fault_watchdog_slot')
    return row['task_id'], row['arm']


def _isolated():
    source = Path(__file__).with_name('completion_watchdog.py')
    raw = source.read_bytes()
    if source.is_symlink() or hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError('codeact_fault_watchdog_source')
    spec = importlib.util.spec_from_file_location('scripts._codeact_fault_watchdog_core', source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)
    module.STUDY_ID, module.client_command, module._slot = STUDY_ID, client_command, _slot
    return module


_CORE = _isolated()
_base_cleanup = _CORE.cleanup_containers


def cleanup_containers(stage, who):
    result = _base_cleanup(stage, who)
    path = Path(stage) / 'run/events.jsonl'
    try:
        raw = path.read_bytes() if path.is_file() and not path.is_symlink() else b''
        complete = bool(raw) and raw.endswith(b'\n')
    except OSError:
        complete = False
    if not complete:
        result['errors'].append('missing_or_incomplete_event_journal')
        result['verified'] = False
    return result


_CORE.cleanup_containers = cleanup_containers
watch, recover = _CORE.watch, _CORE.recover
