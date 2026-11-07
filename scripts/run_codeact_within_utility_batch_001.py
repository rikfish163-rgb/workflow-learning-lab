"""Isolated paired utility client with the source's controlled fault policy.

Import performs no registration, GPU access, task execution or private reads.
"""
import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path

from scripts.codeact_fault_worker_state_001 import POLICY_SHA256

SOURCE = Path(__file__).with_name('run_codeact_within_source_001.py')
SOURCE_SHA256 = '4a05f9cfd7b40bdb6d785d9609920bfc00ea546f115b7a17312e680d7ac0160b'
ID = 'qwen38-codeact-within-utility-001'
PREFIX = 'A previous training experience may apply; check its conditions:\n'
FAULT_MODULE_PATHS = {
    'scripts.codeact_execution_observer_001': 'scripts/codeact_execution_observer_001.py',
    'scripts.codeact_within_worker_state_001': 'scripts/codeact_fault_worker_state_001.py',
    'scripts.codeact_within_worker_001': 'scripts/codeact_fault_worker_001.py',
}


def _fault_valid(value, counts=None):
    fields = {'injected_refusals', 'injected_at_request_attempt', 'injected_error_observed',
              'policy_sha256', 'backend_dispatched_for_fault'}
    if (type(value) is not dict or set(value) != fields
            or type(value['injected_refusals']) is not int or value['injected_refusals'] not in (0, 1)
            or type(value['injected_error_observed']) is not bool or value['policy_sha256'] != POLICY_SHA256
            or value['backend_dispatched_for_fault'] is not False):
        return False
    if counts is not None:
        if (type(counts) is not dict or type(counts.get('api_request_dispatches')) is not int
                or not 0 <= counts['api_request_dispatches'] <= 500):
            return False
        total = counts['api_request_dispatches']
    else:
        total = 500
    marker = value['injected_at_request_attempt']
    return ((value['injected_refusals'] == 0 and marker is None and value['injected_error_observed'] is False)
            or (value['injected_refusals'] == 1 and type(marker) is int and 1 <= marker <= total))


def _fault_projection(holder, result):
    value = holder.get('fault_provenance')
    if not _fault_valid(value):
        return None
    outcome = result.get('outcome') if type(result) is dict else None
    if type(outcome) is dict and 'counts' in outcome and not _fault_valid(value, outcome['counts']):
        return None
    return deepcopy(value)


def _check_experience(plan, value):
    if type(value) is not str or not value:
        raise ValueError('utility_experience_invalid')
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()
    if plan.get('experience_sha256') != digest or plan.get('experience_bundle_id') != 'codeact-experience-001:' + digest:
        raise ValueError('utility_experience_invalid')
    return value


def _planned(plan, index):
    if type(index) is not int or not 0 <= index < 18:
        raise ValueError('utility_slots_invalid')
    slots = plan['slots'][index * 3:index * 3 + 3]
    if (len(slots) != 3 or len(set(plan['batches'][index])) != 3
            or [s.get('task_id') for s in slots] != plan['batches'][index]
            or any(set(s) != {'task_id', 'arm', 'pair_index'} or type(s['task_id']) is not str
                   or s['arm'] not in ('control', 'experience') or type(s['pair_index']) is not int
                   or not 0 <= s['pair_index'] < 27 for s in slots)):
        raise ValueError('utility_slots_invalid')
    return [{'task_id': s['task_id'], 'arm': s['arm'], 'phase': 'utility'} for s in slots]


def _check_initial(history, experience):
    from wll.prompts import SYSTEMS, TUTORIAL
    prefix = [{'role': 'system', 'content': SYSTEMS['appworld-v3']}] + TUTORIAL
    expected_len = len(prefix) + 1 + (experience is not None)
    if (type(history) is not list or len(history) != expected_len or history[:len(prefix)] != prefix
            or type(history[len(prefix)]) is not dict
            or set(history[len(prefix)]) != {'role', 'content'}
            or history[len(prefix)]['role'] != 'user'
            or type(history[len(prefix)]['content']) is not str
            or (experience is not None and history[-1] != {'role': 'user', 'content': PREFIX + experience})):
        raise ValueError('utility_initial_messages_invalid')


def _load():
    raw = SOURCE.read_bytes()
    if SOURCE.is_symlink() or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('utility_source_invalid')
    source = raw.decode('utf-8')
    replacements = [
        ("qwen38-codeact-within-source-001", ID),
        ("codeact-within-source-001", 'codeact-within-utility-001'),
        ("wll-codeact-within-source-batch-v1", 'wll-codeact-within-utility-batch-v1'),
        ('18992', '18994'),
        ('codeact_within_registration_001', 'codeact_within_utility_registration_001'),
        ('codeact_within_gpu_queue_001', 'codeact_within_utility_gpu_queue_001'),
        ('codeact_within_watchdog_001', 'codeact_within_utility_watchdog_001'),
        ('queue.BATCH_COUNT == 4', 'queue.BATCH_COUNT == 18'),
        ('0 <= index < 4', '0 <= index < 18'),
        ('range(4)', 'range(18)'),
        ('len(plan[\'batches\']) == 4', "len(plan['batches']) == 18"),
        ("def _scoped_events(events, task_id):", "def _scoped_events(events, task_id, arm):"),
        ("events.emit(event, task_id=task_id, arm='first_attempt', phase='source', **fields)",
         "events.emit(event, task_id=task_id, arm=arm, phase='utility', **fields)"),
        ('def __init__(self, delegate, plan, expected, tunnel, deadline, events, task_id, progress):',
         'def __init__(self, delegate, plan, expected, tunnel, deadline, events, task_id, progress, arm):\n        self.arm = arm'),
        ("arm='first_attempt',\n                         phase='source'", "arm=self.arm,\n                         phase='utility'"),
        ('def _run_one(plan, task_id, offset, expected, tunnel, deadline, events, private):',
         "def _run_one(plan, slot, offset, expected, tunnel, deadline, events, private, experience):\n    task_id, arm = slot['task_id'], slot['arm']"),
        ('_scoped_events(events, task_id)', '_scoped_events(events, task_id, arm)'),
        ('events, task_id, progress)', 'events, task_id, progress, arm)'),
        ('seed=17, experience=None)', "seed=17, experience=experience if arm == 'experience' else None)"),
        ("'arm': 'first_attempt', 'phase': 'source', 'status':", "'arm': arm, 'phase': 'utility', 'status':"),
        ("planned = [{'task_id': task_id, 'arm': 'first_attempt', 'phase': 'source'}\n               for task_id in plan['batches'][index]]", 'planned = _planned(plan, index)'),
        ("row = _run_one(plan, slot['task_id'], offset, request['expected_backend'], tunnel,\n                           deadline, events, private)",
         "row = _run_one(plan, slot, offset, request['expected_backend'], tunnel,\n                           deadline, events, private, experience)"),
        ("    rows, events, failed = [], None, False", "    experience = _check_experience(plan, registration.load_experience(plan))\n    rows, events, failed = [], None, False"),
        ("row.get('arm') == 'first_attempt' and row.get('phase') == 'source'", "row.get('arm') == slot['arm'] and row.get('phase') == 'utility'"),
        ('def _verify_private(actor_raw, wire_raw, episode, expected_backend):', 'def _verify_private(actor_raw, wire_raw, episode, expected_backend, experience):'),
        ("    actor, model = _TraceCursor(actor_raw), _TraceCursor(wire_raw)", "    _check_initial(history, experience)\n    actor, model = _TraceCursor(actor_raw), _TraceCursor(wire_raw)"),
        ("                            request['expected_backend'])", "                            request['expected_backend'], experience if slot['arm'] == 'experience' else None)"),
        ("    expected_names = set()", "    experience = _check_experience(plan, _registration().load_experience(plan))\n    expected_names = set()"),
    ]
    for old, new in replacements:
        if old not in source:
            raise ValueError('utility_source_transform_invalid')
        source = source.replace(old, new)
    fault_replacements = [
        ('from scripts.codeact_owned_environment_001 import CodeActOwnedEnvironment',
         'from scripts.codeact_fault_owned_environment_001 import CodeActFaultOwnedEnvironment'),
        ('CodeActOwnedEnvironment(task_id', 'CodeActFaultOwnedEnvironment(task_id'),
        ("sources={name: ROOT / (name.replace('.', '/') + '.py') for name in MODULES},\n"
         "                    source_sha256={name: plan['source_sha256'][name.replace('.', '/') + '.py'] for name in MODULES},",
         "sources={name: ROOT / path for name, path in FAULT_MODULE_PATHS.items()},\n"
         "                    source_sha256={name: plan['source_sha256'][path] for name, path in FAULT_MODULE_PATHS.items()},"),
        ("return run_episode(task, world, model, record=lambda name, **fields:\n"
         "                                   _trace_actor(actor, name, **fields), seed=17, experience=experience if arm == 'experience' else None)",
         "result = run_episode(task, world, model, record=lambda name, **fields:\n"
         "                                   _trace_actor(actor, name, **fields), seed=17, experience=experience if arm == 'experience' else None)\n"
         "                holder['fault_provenance'] = holder['world'].fault_status()\n                return result"),
        ("valid = _outcome_valid(outcome) and _owner_verified(projected_owner, outcome, progress)",
         "valid = (_outcome_valid(outcome) and _owner_verified(projected_owner, outcome, progress)\n"
         "                 and _fault_valid(holder.get('fault_provenance'), outcome['counts']))"),
        ("'owner': projected_owner, 'observed_model': dict(progress), 'private_bindings': bindings}",
         "'owner': projected_owner, 'observed_model': dict(progress), 'private_bindings': bindings,\n"
         "                'fault_provenance': _fault_projection(holder, result)}"),
        ("'observed_model': dict(progress), 'private_bindings': bindings}\n\n\ndef run_client",
         "'observed_model': dict(progress), 'private_bindings': bindings,\n"
         "                'fault_provenance': _fault_projection(holder, result)}\n\n\ndef run_client"),
        ("'owner', 'observed_model', 'private_bindings'} | set(_OUTCOME_FIELDS)",
         "'owner', 'observed_model', 'private_bindings', 'fault_provenance'} | set(_OUTCOME_FIELDS)"),
        ("_require(_outcome_valid(outcome) and _owner_verified(row['owner'], outcome, row['observed_model']))",
         "_require(_outcome_valid(outcome) and _owner_verified(row['owner'], outcome, row['observed_model'])\n"
         "                 and _fault_valid(row['fault_provenance'], outcome['counts']))"),
    ]
    for old, new in fault_replacements:
        if source.count(old) != 1:
            raise ValueError('utility_fault_transform_invalid')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_utility_batch_core', SOURCE)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(_planned=_planned, _check_experience=_check_experience, _check_initial=_check_initial,
                           _fault_valid=_fault_valid, _fault_projection=_fault_projection,
                           FAULT_MODULE_PATHS=FAULT_MODULE_PATHS)
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), module.__dict__)
    return module


_CORE = _load()
run_stage = _CORE.run_stage
run_client = _CORE.run_client
verify_batch_report = _CORE.verify_batch_report
_lifecycle = _CORE._lifecycle


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_CORE, name)


if __name__ == '__main__':
    raise SystemExit(_CORE.main())
