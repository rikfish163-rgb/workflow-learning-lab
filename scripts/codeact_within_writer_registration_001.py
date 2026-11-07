"""Frozen bindings for a writer after the complete CodeAct fault source gate.

Public admission reads metadata only. Desktop source export authenticates all
actor evidence before sealing; the amax preflight rehashes that sealed input.
Import does not register, inspect private data, enqueue, or initialize a model.
"""
import copy
import hashlib
import json
import os
import re
import shutil
import socket
from pathlib import Path
from types import FunctionType

from scripts import native_source_registration as public
from scripts.native_appworld_worker_client import _decode
from scripts.native_appworld_worker_client import _encode as _encode
from scripts.queue_recovery_evidence_lesson import verify_public_frozen_plan

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-within-writer-001'
PLAN = 'artifacts/qwen38-codeact-within-writer-plan-001.json'
SOURCE_ID = 'qwen38-codeact-fault-source-001'
SOURCE_PLAN = 'artifacts/qwen38-codeact-fault-source-plan-001.json'
SOURCE_PLAN_SHA = '76bc11bed6f1fb93ff5d91a69d33d231dc3033718d60c803fbf02acc61446876'
SOURCE_REPORT = f'artifacts/{SOURCE_ID}/report.json'
SOURCE_EXPORT_ID = 'codeact-fault-source-export-001'
PREPARATION_PROOF = 'artifacts/codeact-within-writer-inputs-CPU-001.json'
PORT, MODEL_ARM, MODEL_ID = 18993, 'B3', 'qwen38-b3-matched-b3-001'
SERVICE_PLAN, SERVICE_SHA = public.SERVICE_PLAN, public.SERVICE_SHA
PYTHON, WAIT_SECONDS, BATCH_GPU_SECONDS = public.PYTHON, 86400, 7200
PRIVATE_STATE_ROOT = public.PRIVATE_STATE_ROOT
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + '/codeact-within-writer-001'
PRIVATE_INPUT = PRIVATE_STATE_ROOT + '/codeact-within-writer-preparation-001/source-private.json'
TMPDIR = PRIVATE_STATE_ROOT + '/runtime-tmp'
MINIMUM_FREE_BYTES = 5 * 1024**3
ERROR = 'codeact_within_writer_registration_invalid'
TASK_IDS = ('22cc237_3','27e1026_3','287e338_3','2a163ab_3','60d0b5b_3','692c77d_3',
            '82e2fac_3','b0a8eae_3','b7a9ee9_3','c901732_3','cf6abd2_3','d0b1f43_3')
REQUIRED_NEW_SOURCES = frozenset({
    'scripts/codeact_within_writer_registration_001.py',
    'scripts/codeact_within_writer_inputs_001.py', 'scripts/codeact_within_source_export_001.py',
    'scripts/codeact_fault_source_export_001.py',
    'scripts/codeact_within_experience_001.py', 'scripts/codeact_within_writer_gpu_queue_001.py',
    'scripts/run_codeact_within_writer_cases_001.py', 'scripts/run_codeact_within_writer_001.py',
    'scripts/prepare_codeact_within_writer_001.py',
})
public_read, _public_hash = public.public_read, public._public_hash


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def value_hash(value):
    return _hash(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def writer_policy():
    return dict(max_sources=12, max_calls_per_source=1, source_order='ascending_source_index',
                max_input_tokens=15360, max_output_tokens=128, temperature=0.7,
                seed_base=17, seed_rule='17 + source_index', request_window_seconds=120,
                per_call_seconds=240, model_ready_seconds=900, client_seconds=6000,
                content_review=dict(kind='fallible_local_model_review_pending_utility',
                    max_calls_per_candidate=1, max_output_tokens=256, max_input_tokens=15360,
                    per_call_seconds=240, temperature=0.7, seed_base=117,
                    seed_rule='117 + source_index', deterministic_screen_required=True,
                    accepted=False, content_certified=False, training_authorized=False),
                release_seconds=180, shutdown_reserve_seconds=180)


def validate_metadata(plan, *, root=ROOT, _service_validator=None):
    fixed = dict(id=ID, status='registered', model_arm=MODEL_ARM, model_id=MODEL_ID,
                 port=PORT, python=PYTHON, private_run_root=PRIVATE_RUN_ROOT,
                 private_input=PRIVATE_INPUT, tmp_dir=TMPDIR, gpu_seconds=7200,
                 total_gpu_seconds=7200, client_seconds=6000, model_start_seconds=900,
                 cleanup_seconds=180, admission_wait_seconds=86400, automatic_retry=False,
                 writer_authorized=True, training_authorized=False, utility_authorized=False,
                 dev_or_test_authorized=False, preparation_proof=PREPARATION_PROOF)
    _require(type(plan) is dict and all(type(plan.get(k)) is type(v) and plan[k] == v
                                       for k,v in fixed.items())
             and plan.get('writer') == writer_policy()
             and _sha(plan.get('private_input_sha256')) and _sha(plan.get('source_report_sha256'))
             and type(plan.get('source_sha256')) is dict
             and REQUIRED_NEW_SOURCES <= plan['source_sha256'].keys()
             and type(plan.get('input_sha256')) is dict
             and _sha(plan['input_sha256'].get(PREPARATION_PROOF))
             and plan['input_sha256'].get(SOURCE_PLAN) == SOURCE_PLAN_SHA
             and plan['input_sha256'].get(SOURCE_REPORT) == plan['source_report_sha256']
             and type(plan.get('runtime_sha256')) is dict and bool(plan['runtime_sha256'])
             and all(type(k) is str and Path(k).is_absolute() and '..' not in Path(k).parts and _sha(v)
                     for k,v in plan['runtime_sha256'].items()))
    binding = plan.get('registered_inputs')
    _require(type(binding) is dict and set(binding) == {
        'preparation_manifest_sha256','origin_ids','selection'}
        and _sha(binding['preparation_manifest_sha256'])
        and type(binding['origin_ids']) is list and len(binding['origin_ids']) == 12
        and all(type(v) is str and re.fullmatch('[0-9a-f]{7}',v) for v in binding['origin_ids'])
        and type(binding['selection']) is dict)
    if _service_validator is None:
        from scripts.coverage_contract_gpu_queue import _public_service
        _service_validator = _public_service
    _service_validator(root,plan,plan['source_sha256'],plan['input_sha256'])
    return plan


def _source_complete(report):
    try:
        return (report['id'] == SOURCE_ID and report['plan_sha256'] == SOURCE_PLAN_SHA
            and report['status'] == 'complete' and type(report['fixed_denominator']) is int
            and report['fixed_denominator'] == 12 and report['automatic_retry'] is False
            and report['unattempted_batches'] == [] and report['costs']['all_costs_known'] is True
            and report['qualification']['qualified'] is True
            and type(report['qualification']['known_episode_count']) is int
            and report['qualification']['known_episode_count'] == 12
            and type(report['sources']) is list and len(report['sources']) == 12
            and all(r['auth_complete'] is r['costs_known'] is r['attempted'] is True
                    and type(r['source_index']) is int and r['source_index'] == i
                    and type(r['official_success']) is type(r['model_reported_complete']) is bool
                    for i,r in enumerate(report['sources']))
            and type(report['batches']) is list and len(report['batches']) == 4
            and all(type(r['batch_index']) is int and r['batch_index'] == i
                    and r['auth_complete'] is r['accepted'] is True and r['status'] == 'complete'
                    and all(r['release'][k] is True for k in ('cpu_client','containers','model_lease'))
                    for i,r in enumerate(report['batches'])))
    except (KeyError,TypeError):
        return False


def check_predecessor(root, plan):
    raw = public_read(root,SOURCE_REPORT)
    _require(_hash(raw) == plan['source_report_sha256'])
    report = _decode(raw)
    _require(_source_complete(report))
    raw = public_read(root,PREPARATION_PROOF)
    _require(_hash(raw) == plan['input_sha256'][PREPARATION_PROOF])
    proof = _decode(raw)
    exported = proof.get('source_export') if type(proof) is dict else None
    _require(type(exported) is dict and exported.get('id') == SOURCE_EXPORT_ID
             and exported.get('status') == 'authenticated_writer_preparation_only'
             and type(exported.get('source_authentication')) is dict)
    auth = exported['source_authentication']
    binding = plan['registered_inputs']
    _require(proof['id'] == 'codeact-within-writer-inputs-CPU-001'
             and proof['status'] == 'authenticated_sealed_writer_inputs_not_registered'
             and proof['model_calls'] == proof['GPU_calls'] == proof['writer_calls'] == 0
             and proof['private_input_sha256'] == plan['private_input_sha256']
             and proof['registered_inputs'] == binding
             and auth.get('study_id') == SOURCE_ID and auth.get('plan') == SOURCE_PLAN
             and auth.get('plan_sha256') == SOURCE_PLAN_SHA
             and auth.get('qualification_proof') == SOURCE_REPORT
             and auth.get('report_sha256') == plan['source_report_sha256']
             and auth['qualified'] is auth['all_costs_known'] is True
             and auth['episode_count'] == 12
             and auth['origin_ids'] == binding['origin_ids']
             == [task.rsplit('_',1)[0] for task in TASK_IDS]
             and auth['gate_sha256'] == value_hash(report['qualification'])
             and exported.get('preparation') == binding['selection']
             and value_hash(dict(origin_ids=binding['origin_ids'], source_authentication=auth,
                                 selection=binding['selection'])) == binding['preparation_manifest_sha256'])
    return True


def load_registration(root, sha, *, private=False):
    _require(_sha(sha) and type(private) is bool)
    plan = verify_public_frozen_plan(root,Path(root)/PLAN,sha)
    validate_metadata(plan,root=root)
    check_predecessor(root,plan)
    if private:
        _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
                 and os.environ.get('TMPDIR') == TMPDIR)
        for name in (PRIVATE_STATE_ROOT,PRIVATE_RUN_ROOT,TMPDIR,str(Path(PRIVATE_INPUT).parent)):
            public._private_directory(Path(name))
            _require(shutil.disk_usage(name).free >= MINIMUM_FREE_BYTES)
        for name,expected in plan['runtime_sha256'].items():
            _require(public._absolute_hash(name) == expected)
    return plan


def build_registration(root, *, source_sha256, input_sha256, runtime_sha256,
                       preparation_proof, expected_backend_static, wll_code_fingerprint):
    """Return a prospective value; never write, reserve or launch a study."""
    from scripts.coverage_contract_gpu_queue import SERVICE_PLAN, SERVICE_SHA

    policy_raw = public_read(root,'config/gpu-policy.json')
    plan = dict(id=ID, status='registered', model_arm=MODEL_ARM, model_id=MODEL_ID,
                port=PORT, python=PYTHON, private_run_root=PRIVATE_RUN_ROOT,
                private_input=PRIVATE_INPUT, tmp_dir=TMPDIR, gpu_seconds=7200,
                total_gpu_seconds=7200, client_seconds=6000, model_start_seconds=900,
                cleanup_seconds=180, admission_wait_seconds=86400, automatic_retry=False,
                writer_authorized=True, training_authorized=False, utility_authorized=False,
                dev_or_test_authorized=False, preparation_proof=PREPARATION_PROOF,
                private_input_sha256=preparation_proof['private_input_sha256'],
                source_report_sha256=preparation_proof['source_export']['source_authentication']['report_sha256'],
                registered_inputs=copy.deepcopy(preparation_proof['registered_inputs']),
                writer=writer_policy(), service_plan=SERVICE_PLAN, service_plan_sha256=SERVICE_SHA,
                gpu_policy=_decode(policy_raw), gpu_policy_sha256=_hash(policy_raw),
                source_sha256=dict(source_sha256), input_sha256=dict(input_sha256),
                runtime_sha256=dict(runtime_sha256),
                expected_backend_static=copy.deepcopy(expected_backend_static),
                wll_code_fingerprint=wll_code_fingerprint)
    validate_metadata(plan,root=root)
    check_predecessor(root,plan)
    return plan


def private_preflight(root, sha):
    _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    plan = load_registration(root,sha,private=True)
    # Rebind the descriptor-safe reader's private prefix without changing its globals.
    from scripts.native_source_episode import _read_private
    scope = dict(_read_private.__globals__)
    scope['PRIVATE_RUN_ROOT'] = Path(PRIVATE_STATE_ROOT)
    from scripts import native_source_episode as episode
    for name in ('_private_directory','_read_private'):
        f = getattr(episode,name)
        scope[name] = FunctionType(f.__code__,scope,name,f.__defaults__,f.__closure__)
    raw = scope['_read_private'](PRIVATE_INPUT,maximum=64_000_000)
    _require(_hash(raw) == plan['private_input_sha256'])
    value = _decode(raw)
    from scripts.run_codeact_within_writer_cases_001 import validate_preparation
    validate_preparation(value,plan['registered_inputs'])
    _require(_hash(scope['_read_private'](PRIVATE_INPUT,maximum=64_000_000))
             == plan['private_input_sha256'])
    return plan,value
