"""Independent controlled-fault source; the natural source remains frozen.

The source roster, complete-twelve/four-scene/twenty-target gate, model and
budgets are retained. The prospective condition adds one known nonexecuted
Requester refusal; this is not a natural-error, transfer or RSI claim.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/codeact_within_registration_001.py'
BASE_SHA256='ff654594a0809b7a901e6474f4405a90f349800773886fc9bdc69c3613d013db'
ID='qwen38-codeact-fault-source-001'
PLAN='artifacts/qwen38-codeact-fault-source-plan-001.json'
PRIVATE=Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-fault-source-001')
OUTPUT=ROOT/'artifacts'/ID
PORT=18995
PREDECESSOR='artifacts/codeact-within-source-terminal-root-001.json'
PREDECESSOR_SHA256='8b104274f0221be94bce7725508de1e5baea4de32835f18f7340f6a46bbda624'
FAULT_CPU_PROOF='artifacts/codeact-fault-source-root-CPU-001.json'
FAULT_PROBE='artifacts/codeact-fault-probe-root-CPU-002.json'
FAULT_PROBE_PLAN='artifacts/codeact-fault-probe-plan-CPU-002.json'
FAULT_POLICY=dict(id='controlled_single_transient_request_refusal_v1',
    policy_sha256='b86be3262839cb7604bd9716222b6e6f6c3b57f8b414dc298bac6588e0f19ee1',
    condition='first_non_status_Requester_attempt_outside_admin_api_docs_supervisor',
    maximum_injected_refusals=1,backend_execution_for_injected_request=False,
    observation='Transient tool failure: request was not executed.',
    requester_counter_includes_injected_refusal=True,HTTP_dispatch_claimed=False,
    error_anchor='same_Exception_object_in_IPython_result_and_request_attempt',
    actor_caught_fault_without_cell_error_is_eligible=False,
    same_intent_semantic_repair_claimed=False)
NEW_SOURCES={
    'scripts/codeact_fault_worker_state_001.py','scripts/codeact_fault_worker_001.py',
    'scripts/codeact_fault_owned_environment_001.py','scripts/codeact_fault_registration_001.py',
    'scripts/codeact_fault_sources_001.py','scripts/codeact_fault_gpu_queue_001.py',
    'scripts/codeact_fault_watchdog_001.py','scripts/run_codeact_fault_source_001.py',
    'scripts/run_codeact_fault_source_study_001.py',
}
ERROR='codeact_fault_registration_invalid'


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def check_predecessor(root,plan):
    _require(plan['input_sha256'].get(PREDECESSOR)==PREDECESSOR_SHA256)
    raw=_CORE.public_read(root,PREDECESSOR)
    _require(hashlib.sha256(raw).hexdigest()==PREDECESSOR_SHA256)
    proof=json.loads(raw)
    _require(proof['status']=='verified_complete_insufficient_source'
        and proof['source_parent_exited'] is proof['all_four_releases_verified'] is True
        and proof['fixed_denominator']==proof['authenticated_episodes']==12
        and proof['source_gate']['qualified'] is False
        and proof['costs']['all_costs_known'] is True
        and proof['writer_calls']==proof['utility_calls']==0)
    return True


def _additional_validation(root,plan):
    _require(plan.get('source_condition')=='controlled_single_tool_refusal'
        and plan.get('fault_policy')==FAULT_POLICY)
    values={}
    for name in (FAULT_CPU_PROOF,FAULT_PROBE,FAULT_PROBE_PLAN):
        raw=_CORE.public_read(root,name)
        _require(hashlib.sha256(raw).hexdigest()==plan['input_sha256'].get(name))
        values[name]=json.loads(raw)
    cpu,probe,probe_plan=(values[name] for name in (FAULT_CPU_PROOF,FAULT_PROBE,FAULT_PROBE_PLAN))
    _require(cpu['status']=='passed' and cpu['synthetic_only'] is True
        and type(cpu['tests_passed']) is int and cpu['tests_passed']>0
        and all(plan['source_sha256'].get(name)==sha for name,sha in cpu['source_test_sha256'].items())
        and probe['status']=='passed' and probe['human_authored'] is True
        and probe['source_sha256_unchanged'] is True
        and probe['plan_sha256']==plan['input_sha256'][FAULT_PROBE_PLAN]
        and probe['owner']['safe_to_continue'] is probe['owner']['cleanup_verified'] is True
        and all(probe[key]==0 for key in ('model_calls','GPU_calls','official_scorer_calls'))
        and all(plan['source_sha256'].get(name)==sha for name,sha in probe_plan['source_sha256'].items()))


def _isolated():
    raw=BASE.read_bytes()
    _require(hashlib.sha256(raw).hexdigest()==BASE_SHA256)
    spec=importlib.util.spec_from_file_location('scripts._codeact_fault_registration_core',BASE)
    module=importlib.util.module_from_spec(spec)
    exec(compile(raw,str(BASE),'exec',dont_inherit=True),module.__dict__)
    module.ID,module.PLAN,module.PRIVATE,module.OUTPUT,module.PORT=ID,PLAN,PRIVATE,OUTPUT,PORT
    module.REQUIRED_NEW_SOURCES=set(module.REQUIRED_NEW_SOURCES)|NEW_SOURCES
    original_validate=module.validate_metadata
    def validate(plan,*,_service_validator=None,root=ROOT):
        result=original_validate(plan,_service_validator=_service_validator,root=root)
        _require(plan.get('source_condition')=='controlled_single_tool_refusal'
            and plan.get('fault_policy')==FAULT_POLICY)
        return result
    module.validate_metadata,module.check_predecessor=validate,check_predecessor
    return module


_CORE=_isolated()
TASK_IDS,BATCHES,PROTOCOL=_CORE.TASK_IDS,copy.deepcopy(_CORE.BATCHES),copy.deepcopy(_CORE.PROTOCOL)


def load_registration(root,sha,*,private=False):
    plan=_CORE.load_registration(root,sha,private=private)
    _additional_validation(root,plan)
    check_predecessor(root,plan)
    return plan


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_CORE,name)
