"""Source-bound CodeAct client with one controlled Requester refusal per world.

Only pinned trusted source is compiled; importing does not start a task or GPU.
"""
import hashlib
import importlib.util
from pathlib import Path

from scripts.codeact_fault_worker_state_001 import POLICY_SHA256

SOURCE = Path(__file__).with_name('run_codeact_within_source_001.py')
SOURCE_SHA256 = '4a05f9cfd7b40bdb6d785d9609920bfc00ea546f115b7a17312e680d7ac0160b'
ID = 'qwen38-codeact-fault-source-001'


def _fault_valid(value, counts):
    if (type(value) is not dict or set(value) != {'injected_refusals', 'injected_at_request_attempt', 'injected_error_observed',
            'policy_sha256', 'backend_dispatched_for_fault'} or type(counts) is not dict
            or type(counts.get('api_request_dispatches')) is not int
            or value['policy_sha256'] != POLICY_SHA256 or value['backend_dispatched_for_fault'] is not False
            or type(value['injected_refusals']) is not int or type(value['injected_error_observed']) is not bool):
        return False
    count, marker = value['injected_refusals'], value['injected_at_request_attempt']
    return ((count == 0 and marker is None and value['injected_error_observed'] is False) or
            (count == 1 and type(marker) is int and 1 <= marker <= counts['api_request_dispatches'] <= 500))


def _load():
    raw = SOURCE.read_bytes()
    if SOURCE.is_symlink() or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('codeact_fault_source_binding')
    source = raw.decode('utf-8')
    changes = [
        ('qwen38-codeact-within-source-001', ID),
        ('codeact-within-source-001', 'codeact-fault-source-001'),
        ('wll-codeact-within-source-batch-v1', 'wll-codeact-fault-source-batch-v1'),
        ('18992', '18995'),
        ('codeact_within_registration_001', 'codeact_fault_registration_001'),
        ('codeact_within_gpu_queue_001', 'codeact_fault_gpu_queue_001'),
        ('codeact_within_watchdog_001', 'codeact_fault_watchdog_001'),
        ('from scripts.codeact_owned_environment_001 import CodeActOwnedEnvironment',
         'from scripts.codeact_fault_owned_environment_001 import CodeActFaultOwnedEnvironment'),
        ('CodeActOwnedEnvironment(task_id', 'CodeActFaultOwnedEnvironment(task_id'),
        ("sources={name: ROOT / (name.replace('.', '/') + '.py') for name in MODULES},\n                    source_sha256={name: plan['source_sha256'][name.replace('.', '/') + '.py'] for name in MODULES},",
         "sources={name: ROOT / path for name, path in FAULT_MODULE_PATHS.items()},\n                    source_sha256={name: plan['source_sha256'][path] for name, path in FAULT_MODULE_PATHS.items()},"),
        ("return run_episode(task, world, model, record=lambda name, **fields:\n                                   _trace_actor(actor, name, **fields), seed=17, experience=None)",
         "result = run_episode(task, world, model, record=lambda name, **fields:\n                                   _trace_actor(actor, name, **fields), seed=17, experience=None)\n                holder['fault_provenance'] = holder['world'].fault_status()\n                return result"),
        ("valid = _outcome_valid(outcome) and _owner_verified(projected_owner, outcome, progress)",
         "valid = (_outcome_valid(outcome) and _owner_verified(projected_owner, outcome, progress)\n                 and _fault_valid(holder.get('fault_provenance'), outcome['counts']))"),
        ("'owner': projected_owner, 'observed_model': dict(progress), 'private_bindings': bindings}",
         "'owner': projected_owner, 'observed_model': dict(progress), 'private_bindings': bindings,\n                'fault_provenance': holder.get('fault_provenance')}"),
        ("'observed_model': dict(progress), 'private_bindings': bindings}\n\n\ndef run_client",
         "'observed_model': dict(progress), 'private_bindings': bindings,\n                'fault_provenance': holder.get('fault_provenance')}\n\n\ndef run_client"),
        ("phase='source'", "phase='controlled_fault_source'"),
        ("'phase': 'source'", "'phase': 'controlled_fault_source'"),
        ("row.get('phase') == 'source'", "row.get('phase') == 'controlled_fault_source'"),
        ("'owner', 'observed_model', 'private_bindings'} | set(_OUTCOME_FIELDS)",
         "'owner', 'observed_model', 'private_bindings', 'fault_provenance'} | set(_OUTCOME_FIELDS)"),
        ("_require(_outcome_valid(outcome) and _owner_verified(row['owner'], outcome, row['observed_model']))",
         "_require(_outcome_valid(outcome) and _owner_verified(row['owner'], outcome, row['observed_model'])\n                 and _fault_valid(row['fault_provenance'], outcome['counts']))"),
    ]
    for old, new in changes:
        if old not in source:
            raise ValueError('codeact_fault_source_transform')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._codeact_fault_source_core', SOURCE)
    core = importlib.util.module_from_spec(spec)
    core.__dict__.update(_fault_valid=_fault_valid, FAULT_MODULE_PATHS={
        'scripts.codeact_execution_observer_001': 'scripts/codeact_execution_observer_001.py',
        'scripts.codeact_within_worker_state_001': 'scripts/codeact_fault_worker_state_001.py',
        'scripts.codeact_within_worker_001': 'scripts/codeact_fault_worker_001.py'})
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


_CORE = _load()
run_stage, run_client = _CORE.run_stage, _CORE.run_client
verify_batch_report, _lifecycle = _CORE.verify_batch_report, _CORE._lifecycle


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_CORE, name)


if __name__ == '__main__':
    raise SystemExit(_CORE.main())
