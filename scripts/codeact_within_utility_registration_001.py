"""Prospective, train-only CodeAct utility bindings after one sealed writer.

No input, plan, world, model or queue is created at import. Source and writer
authentication belongs to the terminal preparation proof, not a model verdict.
The 27 pairs are all exposed train data and cannot establish unseen transfer.
"""
import copy
import hashlib
import os
import re
from pathlib import Path
from types import FunctionType

from scripts import codeact_fault_registration_001 as fault_source
from scripts import codeact_within_registration_001 as source
from scripts import codeact_within_writer_registration_001 as writer
from scripts.native_appworld_worker_client import _decode, _same
from scripts.queue_recovery_evidence_lesson import verify_public_frozen_plan

ROOT=Path(__file__).resolve().parents[1]
ID='qwen38-codeact-within-utility-001'
PLAN='artifacts/qwen38-codeact-within-utility-plan-001.json'
PORT=18994
MODEL_ARM,MODEL_ID='B3',writer.MODEL_ID
PYTHON,SERVICE_PLAN,SERVICE_SHA=writer.PYTHON,writer.SERVICE_PLAN,writer.SERVICE_SHA
WAIT_SECONDS,BATCH_GPU_SECONDS=86400,7200
PRIVATE=Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-utility-001')
OUTPUT=ROOT/'artifacts'/ID
TMPDIR=source.TMPDIR
IMAGE,DATA,DEPS=source.IMAGE,source.DATA,source.DEPS
EXPERIENCE_INPUT='/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-utility-preparation-001/bank.json'
PREPARATION_PROOF='artifacts/codeact-within-utility-inputs-CPU-001.json'
COHORT=source.COHORT
COHORT_SHA256=source.PROOF_INPUTS[COHORT]
SOURCE_PLAN_SHA=writer.SOURCE_PLAN_SHA
SOURCE_SCENARIOS=tuple(task[:-2] for task in source.TASK_IDS)
UTILITY_SCENARIOS=('229360a','34d9492','6104387','76f2c72','7d7fbf6','aa8502b','afc0fce','ce359b5','e7a10f8')
UTILITY_TASK_IDS=tuple(scene+'_'+str(variant) for scene in UTILITY_SCENARIOS for variant in (1,2,3))
ERROR='codeact_within_utility_registration_invalid'
public_read=writer.public_read
REQUIRED_NEW_SOURCES={
    'scripts/codeact_within_utility_registration_001.py',
    'scripts/codeact_within_utility_001.py', 'scripts/codeact_within_utility_gpu_queue_001.py',
    'scripts/codeact_within_utility_watchdog_001.py', 'scripts/run_codeact_within_utility_batch_001.py',
    'scripts/run_codeact_within_utility_study_001.py',
    'scripts/codeact_fault_worker_state_001.py','scripts/codeact_fault_worker_001.py',
    'scripts/codeact_fault_owned_environment_001.py','scripts/codeact_fault_sources_001.py',
    'scripts/codeact_fault_registration_001.py',
}


def slots():
    result=[]
    for offset in range(0,27,3):
        for second in (False,True):
            for index in range(offset,offset+3):
                first='control' if index%2==0 else 'experience'
                arm=({'control':'experience','experience':'control'}[first] if second else first)
                result.append(dict(task_id=UTILITY_TASK_IDS[index],arm=arm,pair_index=index))
    return result


SLOTS=slots()
TASK_IDS=tuple(row['task_id'] for row in SLOTS)
BATCHES=[list(TASK_IDS[index:index+3]) for index in range(0,54,3)]
EXPECTED_PAIRS=[[task,task[:-2],17] for task in UTILITY_TASK_IDS]
PROTOCOL={key:copy.deepcopy(source.PROTOCOL[key]) for key in (
    'prompt_version','temperature','seed','environment_seed','pythonhashseed','max_generations',
    'max_actions','max_api_dispatches','max_input_tokens','max_output_tokens','max_context_tokens',
    'model_request_seconds','cell_timeout_seconds','actor_soft_seconds','world_startup_seconds',
    'world_operation_seconds','batch_client_seconds','model_start_seconds','gpu_seconds',
    'GPU_release_poll_seconds','admission_wait_seconds','uniform_per_world_cleanup_deadline_claimed')}
PROTOCOL.update(pairs=27,scenarios=9,arms=54,serial_gpu_jobs=18,fresh_world_per_arm=True,
    replay=False,comparison='ordinary_experience_vs_no_experience',split='exposed_train',
    memory_placement='complete_user_message_after_task',pairs_cross_model_leases=True,
    bootstrap_samples=2000,bootstrap_seed=0,minimum_scenarios=8,maximum_harm_rate=0.05,
    both_lower_bounds_strictly_positive=True,require_complete_planned_pairs=True,
    separate_scorer_process_claimed=False)
PROTOCOL.update(source_condition='controlled_single_tool_refusal',
                fault_policy=copy.deepcopy(fault_source.FAULT_POLICY),
                fault_condition_same_in_both_arms=True,
                no_fault_or_caught_fault_episodes_retained_in_utility=True)


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}',value) is not None


def expected_identity(plan):
    from wll.protocol import fingerprint

    return dict(model_id=MODEL_ID,scaffold_id='codeact-within-utility-001',
                backend_id='B3:'+fingerprint(plan['expected_backend_static']['B3']),
                budget_id='codeact-budget-001:'+fingerprint(PROTOCOL))


def _read_bank(path):
    _require(path==EXPERIENCE_INPUT)
    # Descriptor-safe traversal, bound to the existing private-state prefix.
    from scripts import native_source_episode as episode
    scope=dict(episode.__dict__)
    scope['PRIVATE_RUN_ROOT']=Path('/home/hetaisheng/.local/share/workflow-learning-lab')
    for name in ('_private_directory','_read_private'):
        function=getattr(episode,name)
        scope[name]=FunctionType(function.__code__,scope,name,function.__defaults__,function.__closure__)
    return scope['_read_private'](path,maximum=64_000_000)


def render_bundle(rows):
    """Include every frozen pending row, preserving fields and source order."""
    from scripts import native_experience_content_001 as content

    _require(type(rows) is list and 1<=len(rows)<=12)
    indices,texts=[],[]
    fields={'source_index','source_scenario','candidate','insertion','candidate_sha256',
            'insertion_sha256','review_result_sha256','status','accepted','content_certified','training_authorized'}
    for row in rows:
        _require(type(row) is dict and set(row)==fields
            and type(row['source_index']) is int and 0<=row['source_index']<12
            and row['source_scenario']==SOURCE_SCENARIOS[row['source_index']]
            and row['status']=='plausible_pending_utility'
            and row['accepted'] is row['content_certified'] is row['training_authorized'] is False
            and _sha(row['review_result_sha256']))
        text=content.insertion(row['candidate'])
        _require(row['insertion']==text
            and row['candidate_sha256']==hashlib.sha256(content._json(row['candidate']).encode()).hexdigest()
            and row['insertion_sha256']==hashlib.sha256(text.encode()).hexdigest())
        indices.append(row['source_index'])
        texts.append(text)
    _require(indices==sorted(set(indices)))
    return '\n\n'.join(texts)


def load_experience(plan):
    try:
        raw=_read_bank(plan['experience_input'])
        _require(hashlib.sha256(raw).hexdigest()==plan['experience_input_sha256'])
        bank=_decode(raw)
        _require(type(bank) is dict and set(bank)=={'status','accepted','training_authorized',
            'insertion_text','source_scenario_ids','rows','source_authentication','writer_authentication'}
            and bank['status']=='plausible_pending_utility'
            and bank['accepted'] is bank['training_authorized'] is False
            and bank['source_scenario_ids']==list(SOURCE_SCENARIOS)
            and type(bank['rows']) is list and 1<=len(bank['rows'])<=12
            and type(bank['insertion_text']) is str and bool(bank['insertion_text'].strip())
            and bank['source_authentication']['plan_sha256']==SOURCE_PLAN_SHA
            and bank['writer_authentication']['plan_sha256']==plan['writer_plan_sha256'])
        text=bank['insertion_text']
        _require(text==render_bundle(bank['rows']))
        digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
        _require(digest==plan['experience_sha256']
                 and plan['experience_bundle_id']=='codeact-experience-001:'+digest)
        _require(_read_bank(plan['experience_input'])==raw)
        return text
    except Exception:
        raise ValueError(ERROR) from None


def validate_metadata(plan,*,root=ROOT,_service_validator=None):
    fixed=dict(id=ID,status='registered',port=PORT,model_id=MODEL_ID,model_arm=MODEL_ARM,
        task_ids=list(TASK_IDS),batches=BATCHES,slots=SLOTS,expected_pairs=EXPECTED_PAIRS,
        protocol=PROTOCOL,private_directory=str(PRIVATE),tmp_dir=TMPDIR,image=IMAGE,data_dir=DATA,deps_dir=DEPS,
        source_condition='controlled_single_tool_refusal',fault_policy=fault_source.FAULT_POLICY,
        experience_input=EXPERIENCE_INPUT,preparation_proof=PREPARATION_PROOF,
        gpu_seconds=7200,client_seconds=6000,model_start_seconds=900,cleanup_seconds=180,
        admission_wait_seconds=86400,automatic_retry=False,writer_authorized=False,
        training_authorized=False,utility_authorized=True,dev_or_test_authorized=False)
    _require(type(plan) is dict and all(key in plan and _same(plan[key],value) for key,value in fixed.items())
        and all(_sha(plan.get(key)) for key in ('experience_input_sha256','experience_sha256','writer_plan_sha256'))
        and plan.get('experience_bundle_id')=='codeact-experience-001:'+plan['experience_sha256']
        and type(plan.get('source_sha256')) is dict and REQUIRED_NEW_SOURCES<=plan['source_sha256'].keys()
        and type(plan.get('input_sha256')) is dict and plan['input_sha256'].get(COHORT)==COHORT_SHA256
        and _sha(plan['input_sha256'].get(PREPARATION_PROOF))
        and type(plan.get('native_bindings')) is dict and bool(plan['native_bindings']))
    _require(type(plan.get('identity')) is dict and _same(plan['identity'],expected_identity(plan)))
    if _service_validator is None:
        from scripts.coverage_contract_gpu_queue import _public_service
        _service_validator=_public_service
    _service_validator(root,plan,plan['source_sha256'],plan['input_sha256'])
    return plan


def check_predecessor(root,plan):
    proof_raw=public_read(root,PREPARATION_PROOF)
    _require(hashlib.sha256(proof_raw).hexdigest()==plan['input_sha256'][PREPARATION_PROOF])
    proof=_decode(proof_raw)
    _require(proof['id']=='codeact-within-utility-inputs-CPU-001'
        and proof['status']=='authenticated_frozen_bank_before_utility'
        and proof['source_all_twelve_authenticated'] is proof['writer_all_costs_known']
            is proof['writer_release_verified'] is True
        and proof['writer_plan_sha256']==plan['writer_plan_sha256']
        and proof['experience_input_sha256']==plan['experience_input_sha256']
        and proof['experience_sha256']==plan['experience_sha256']
        and proof['experience_bundle_id']==plan['experience_bundle_id']
        and type(proof['pending_count']) is int and 1<=proof['pending_count']<=12
        and proof['utility_calls']==0 and proof['training_authorized'] is False)
    registered=writer.load_registration(root,plan['writer_plan_sha256'],private=False)
    _require(writer.check_predecessor(root,registered) is True)
    return True


def collect_native_assets():
    scope=dict(source.collect_native_assets.__globals__)
    scope['TASK_IDS']=UTILITY_TASK_IDS
    collect=FunctionType(source.collect_native_assets.__code__,scope,
        'collect_native_assets',source.collect_native_assets.__defaults__)
    collect.__kwdefaults__=source.collect_native_assets.__kwdefaults__
    return collect()


def load_registration(root,sha,*,private=False):
    _require(_sha(sha) and type(private) is bool)
    plan=verify_public_frozen_plan(root,Path(root)/PLAN,sha)
    validate_metadata(plan,root=root)
    raw=public_read(root,COHORT)
    _require(hashlib.sha256(raw).hexdigest()==COHORT_SHA256
        and _decode(raw)['utility_task_ids']==list(UTILITY_TASK_IDS))
    check_predecessor(root,plan)
    if private:
        from scripts.remote_runtime_storage import verify_storage
        verify_storage(root,TMPDIR)
        _require(os.environ.get('CUDA_VISIBLE_DEVICES')=='')
        from scripts.control_plane_diagnostics_v2 import validate_directory
        validate_directory(PRIVATE)
        _require(collect_native_assets()==plan['native_bindings'])
        load_experience(plan)
    return plan
