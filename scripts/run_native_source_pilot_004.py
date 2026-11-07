"""Independent V4 pilot with strict, cancellation-aware cost admission.

The authenticated prior lifecycle still owns failures, cleanup and source gates.
Cost evidence is always retained before invalid known receipts are rejected.
Structural journal completeness alone never makes an unknown receipt known.
"""

import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path

from scripts.native_generation_accounting_004 import generation_accounting

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/run_native_source_pilot_003.py'
BASE_SHA256 = 'bf0d5b64ff3fe039a33f092dac141840716fe533bf16709d83a523eca65f8614'
ID = 'qwen38-native-source-pilot-004'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-004.json'
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
_CORE = None


def _costs(receipt, journal, *, _base_costs):
    """Retain observed costs, certifying only consistent known V4 receipts."""
    try:
        result = _base_costs(receipt, journal)
    except Exception:
        # Invalid counters cannot discard lower bounds from the durable journal.
        outcome = receipt.get('actor_outcome') if type(receipt) is dict else None
        observed = outcome.get('cost') if type(outcome) is dict else None
        observed = observed if type(observed) is dict else {}
        safe = {key: value if type(value) is int and value >= 0 else None
                for key, value in observed.items()}
        sanitized = dict(receipt) if type(receipt) is dict else {}
        sanitized['actor_outcome'] = {'cost': safe, 'outcome_unknown': True, 'unknown_cost': True}
        rows = [row for row in journal if type(row) is dict] if type(journal) in (list, tuple) else []
        result = _base_costs(sanitized, rows)
    result['generation_accounting_valid'] = False
    result['confirmed_not_started'] = None
    result['generation_attempts_after_cancellations'] = None
    outcome = receipt.get('actor_outcome') if type(receipt) is dict else None
    try:
        accounting = generation_accounting(journal)
    except ValueError:
        result['generation_accounting_error'] = 'native_generation_accounting_invalid'
    else:
        result.update(accounting)
        result['generation_accounting_valid'] = True
        if type(receipt) is dict and receipt.get('status') == 'saved_and_closed':
            consistent = (type(outcome) is dict
                and type(outcome.get('status')) is str
                and outcome['status'] in {'incomplete', 'model_reported_complete'}
                and outcome.get('outcome_unknown') is False and outcome.get('unknown_cost') is False
                and receipt.get('is_policy_episode') is True
                and type(receipt.get('worker_exit_code')) is int and receipt['worker_exit_code'] == 0
                and result['counts_complete'] is True
                and accounting['unreturned_generation_intents_upper_bound'] == 0
                and result['generations'] == accounting['generation_attempts_after_cancellations']
                and result['prompt_tokens'] == accounting['journal_returned_prompt_tokens_lower_bound']
                and result['completion_tokens'] == accounting['journal_returned_completion_tokens_lower_bound'])
            if not consistent:
                result['generation_accounting_valid'] = False
                result['generation_accounting_error'] = 'native_generation_receipt_cost_mismatch'
    known = (type(receipt) is dict and receipt.get('status') == 'saved_and_closed'
             and result['generation_accounting_valid'] is True and result['counts_complete'] is True)
    result['counts_complete'] = known
    result['unknown_cost'] = not known
    return result


def _isolated(*, raw=None, _modules=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/run_native_source_pilot_003.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('native_source_pilot_004_source_invalid')
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_004'),
        'episode': importlib.import_module('scripts.native_source_episode_004'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_004'),
        'actor': importlib.import_module('scripts.native_appworld_actor_004'),
    }
    if (type(modules) is not dict or set(modules) != {'registration', 'episode', 'queue', 'actor'}
            or any(modules[name].ID != ID or modules[name].PLAN != PLAN
                   for name in ('registration', 'episode', 'queue'))
            or not callable(modules['queue'].fenced_release)):
        raise ValueError('native_source_pilot_004_binding_invalid')
    source = raw.decode('utf-8')
    for old, new in (
        ('qwen38-native-source-pilot-003', ID),
        ('artifacts/qwen38-native-source-pilot-plan-003.json', PLAN),
        ('\'"_003"\'', '\'"_004"\''),
        ('scripts.native_source_episode_003', 'scripts.native_source_episode_004'),
        ('scripts.run_native_source_pilot_003', 'scripts.run_native_source_pilot_004'),
        ('scripts._native_source_pilot_003_core', 'scripts._native_source_pilot_004_core'),
    ):
        # Episode appears once in imports and once in the inner CLI replacement.
        expected = 2 if old == 'scripts.native_source_episode_003' else 1
        if source.count(old) != expected:
            raise ValueError('native_source_pilot_004_binding_invalid')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._native_source_pilot_004_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError('native_source_pilot_004_source_invalid')
    base = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    core = base._isolated(_modules=modules)
    original_costs, original_evidence = core._costs, core.actor_evidence
    core._costs = lambda receipt, journal: _costs(receipt, journal, _base_costs=original_costs)

    def actor_evidence(receipt, rows, **kwargs):
        # The lifecycle already appended episode costs and set cost_recorded.
        # Reject before prefix selection, scoring or the next source begins.
        costs = core._costs(receipt, rows)
        core._require(costs['generation_accounting_valid'] is True and costs['counts_complete'] is True)
        return original_evidence(receipt, rows, **kwargs)

    core.actor_evidence = actor_evidence
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_pilot(plan_sha, root=None, _runtime=None):
    core = _get_core()
    return core.run_pilot(plan_sha, root=core.WORKSPACE if root is None else root, _runtime=_runtime)


def launch_pilot(plan_sha, *, root=None):
    core = _get_core()
    return core.launch_pilot(plan_sha, root=core.WORKSPACE if root is None else root)


def unit_command(*args, **kwargs):
    return _get_core().unit_command(*args, **kwargs)


def initial_slots():
    return _get_core().initial_slots()


def __getattr__(name):
    if name in {'_Runtime', '_Heartbeat', '_queue_state', 'NativeSourcePilotError'}:
        return getattr(_get_core(), name)
    raise AttributeError(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--run', action='store_true')
    choices.add_argument('--launch', action='store_true')
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
