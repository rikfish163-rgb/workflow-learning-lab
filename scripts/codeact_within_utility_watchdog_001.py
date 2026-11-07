"""Independent cleanup, with each owned intent bound to its registered slot."""
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

STUDY_ID = 'qwen38-codeact-within-utility-001'
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts' / STUDY_ID
FROZEN_SHA256 = '6f02c121f9840e471768aa7b20d1d5bb3fba526e2460bc38c09fc5f3b1b647e7'


def _registration():
    return importlib.import_module('scripts.codeact_within_utility_registration_001')


def client_command(stage, plan_sha):
    stage = Path(stage)
    if (type(plan_sha) is not str or re.fullmatch('[0-9a-f]{64}', plan_sha) is None
            or stage not in [OUTPUT / f'batch-{i}' for i in range(18)]):
        raise ValueError('utility_watchdog_binding')
    request = stage / 'request.json'
    if not request.is_file() or request.is_symlink():
        raise ValueError('utility_watchdog_request')
    return [sys.executable, '-u', '-B', '-m', 'scripts.run_codeact_within_utility_batch_001',
            '--plan-sha256', plan_sha, '--client-request', str(request),
            '--client-request-sha256', hashlib.sha256(request.read_bytes()).hexdigest()]


def _slot(row):
    if (type(row) is not dict or 'stage' in row or type(row.get('task_id')) is not str
            or not row['task_id'] or row.get('arm') not in ('control', 'experience')
            or row.get('phase') != 'utility'):
        raise ValueError('utility_watchdog_slot')
    return row['task_id'], row['arm']


def _isolated():
    source = Path(__file__).with_name('completion_watchdog.py')
    raw = source.read_bytes()
    if source.is_symlink() or hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError('utility_watchdog_source')
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_utility_watchdog_core', source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)
    module.STUDY_ID, module.client_command, module._slot = STUDY_ID, client_command, _slot
    return module


_CORE = _isolated()
_base_cleanup = _CORE.cleanup_containers


def cleanup_containers(stage, who):
    # Exact owned cleanup must happen even if a corrupt journal cannot authenticate.
    result = _base_cleanup(stage, who)
    try:
        stage = Path(stage)
        index = [OUTPUT / f'batch-{i}' for i in range(18)].index(stage)
        request_path = stage / 'request.json'
        journal_path = stage / 'run/events.jsonl'
        if request_path.is_symlink() or journal_path.is_symlink():
            raise ValueError('unsafe path')
        request = json.loads(request_path.read_bytes())
        plan = _registration().load_registration(ROOT, request['plan_sha256'])
        if request['batch_index'] != index:
            raise ValueError('batch mismatch')
        from scripts.run_codeact_within_utility_batch_001 import _planned
        planned = _planned(plan, index)
        raw = journal_path.read_bytes()
        if not raw or not raw.endswith(b'\n'):
            raise ValueError('incomplete journal')
        rows = [json.loads(line) for line in raw.splitlines()]
        intents = [_slot(r) for r in rows if r.get('event') == 'container_intent']
        expected = [(s['task_id'], s['arm']) for s in planned]
        if intents != expected or len(set(intents)) != 3:
            raise ValueError('unbound intents')
    except Exception:
        result['errors'].append('utility_slot_or_journal_unverified')
        result['verified'] = False
    return result


_CORE.cleanup_containers = cleanup_containers
watch, recover = _CORE.watch, _CORE.recover
