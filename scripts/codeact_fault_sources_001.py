"""The unchanged 12/4/20 gate, anchored to a trusted injected Exception.

The caller authenticates the worker provenance and all original lifecycle/cost
evidence. This pure helper does not infer exception identity from observation
text, authenticate a receipt, run an actor, or claim same-intent semantic repair.
"""
import hashlib
import importlib.util
from pathlib import Path

BASE=Path(__file__).with_name('codeact_within_sources_001.py')
BASE_SHA256='0a9cccdd48fa9f4cd3952aa349a5917f2dac9bfd533787d4c778c37be3eb8c24'
POLICY_SHA256='b86be3262839cb7604bd9716222b6e6f6c3b57f8b414dc298bac6588e0f19ee1'
ERROR='codeact_fault_sources_invalid'


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def validate_fault(value,total=None):
    _require(type(value) is dict and set(value)=={'injected_refusals','injected_at_request_attempt',
        'injected_error_observed','policy_sha256','backend_dispatched_for_fault'}
        and type(value['injected_refusals']) is int and value['injected_refusals'] in (0,1)
        and type(value['injected_error_observed']) is bool
        and value['policy_sha256']==POLICY_SHA256 and value['backend_dispatched_for_fault'] is False)
    marker=value['injected_at_request_attempt']
    _require((value['injected_refusals']==0 and marker is None and value['injected_error_observed'] is False)
        or (value['injected_refusals']==1 and type(marker) is int and 1<=marker<=500
            and (total is None or (type(total) is int and marker<=total))))
    return value


def _isolated():
    raw=BASE.read_bytes()
    _require(hashlib.sha256(raw).hexdigest()==BASE_SHA256)
    source=raw.decode()
    old_condition="""        if (metadata['status'] in {'error_before_exec', 'error_in_exec'}
                and metadata['error_type'] in ERROR_TYPES and metadata['hook_calls'] == 1
                and (metadata['error_type'] != 'Exception'
                     or measured['api_request_dispatches'] > requests_before)):
"""
    new_condition="""        if (metadata['status'] in {'error_before_exec', 'error_in_exec'}
                and metadata['error_type'] == 'Exception' and metadata['hook_calls'] == 1
                and fault_observed and fault_marker is not None
                and requests_before < fault_marker <= measured['api_request_dispatches']):
"""
    replacements=[
        ('def _trace(private, outcome):','def _trace(private, outcome, fault_marker, fault_observed):'),
        (old_condition,new_condition),
        ("{'source_index', 'task_id', 'outcome', 'private', 'verified'}",
         "{'source_index', 'task_id', 'outcome', 'private', 'verified', 'fault_provenance'}"),
        ("        _outcome(outcome)\n", "        _outcome(outcome)\n        validate_fault(envelope['fault_provenance'], outcome['counts']['api_request_dispatches'])\n"),
        ("_trace(envelope['private'], outcome) if outcome['costs_known'] else ([], [])",
         "_trace(envelope['private'], outcome, envelope['fault_provenance']['injected_at_request_attempt'], envelope['fault_provenance']['injected_error_observed']) if outcome['costs_known'] else ([], [])"),
    ]
    for old,new in replacements:
        _require(source.count(old)==1)
        source=source.replace(old,new)
    spec=importlib.util.spec_from_file_location('scripts._codeact_fault_sources_core',BASE)
    module=importlib.util.module_from_spec(spec)
    module.validate_fault=validate_fault
    exec(compile(source,str(BASE),'exec',dont_inherit=True),module.__dict__)
    return module


_CORE=_isolated()


def qualify_sources(episodes):
    try:
        return _CORE.qualify_sources(episodes)
    except Exception:
        raise ValueError(ERROR) from None
