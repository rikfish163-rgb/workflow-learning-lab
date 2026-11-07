"""Public bindings for one prospective CodeAct within-workflow source study.

Import never registers or starts an experiment. The registration is created only
after the final controller, watchdog, source gate and CPU checks are bound.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from scripts.queue_recovery_evidence_lesson import (
    public_read,
    verify_public_frozen_plan,
)

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-within-source-001'
PLAN = 'artifacts/qwen38-codeact-within-source-plan-001.json'
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-source-001')
OUTPUT = ROOT / 'artifacts' / ID
TMPDIR = '/home/hetaisheng/.local/share/workflow-learning-lab/runtime-tmp'
IMAGE = 'sha256:9256a65732145731805985e32e9632d2920d7bddc38777e5cd05c24b3672e1fa'
DATA = '/media/hetaisheng/044A81D94A81C83E/wll-runtime/data'
DEPS = '/media/hetaisheng/044A81D94A81C83E/wll-runtime/deps'
PORT = 18992
MODEL_ID = 'qwen38-b3-matched-b3-001'
TASK_IDS = ('22cc237_3', '27e1026_3', '287e338_3', '2a163ab_3', '60d0b5b_3', '692c77d_3',
            '82e2fac_3', 'b0a8eae_3', 'b7a9ee9_3', 'c901732_3', 'cf6abd2_3', 'd0b1f43_3')
BATCHES = [list(TASK_IDS[offset:offset+3]) for offset in range(0, 12, 3)]
PROTOCOL = {
    'prompt_version': 'appworld-v3', 'temperature': 0.7, 'seed': 17, 'environment_seed': 100,
    'pythonhashseed': 100, 'max_generations': 40, 'max_actions': 40, 'max_api_dispatches': 500,
    'max_input_tokens': 15360, 'max_output_tokens': 1024, 'max_context_tokens': 16384,
    'model_request_seconds': 180, 'cell_timeout_seconds': 30, 'actor_soft_seconds': 1200,
    'world_startup_seconds': 150, 'world_operation_seconds': 1500,
    'batch_client_seconds': 6000, 'model_start_seconds': 900, 'gpu_seconds': 7200,
    'GPU_release_poll_seconds': 180, 'admission_wait_seconds': 86400,
    'minimum_source_scenarios': 4, 'minimum_distinct_targets': 20,
    'full_source_tasks': 12, 'full_utility_pairs': 27,
    'uniform_per_world_cleanup_deadline_claimed': False,
    'source_error_types': ['NameError', 'TypeError', 'ValueError', 'KeyError', 'IndexError',
                           'AttributeError', 'ZeroDivisionError', 'AssertionError', 'Exception'],
    'generic_exception_requires_requester_dispatch': True,
}
COHORT = 'artifacts/codeact-within-workflow-source-cohort-001.json'
FINAL_CPU_PROOF = 'artifacts/codeact-within-source-root-CPU-001.json'
PREDECESSOR = 'artifacts/native-tool-recovery-source-terminal-root-001.json'
PROOF_INPUTS = {
    COHORT: 'a0077498a10fdec0cf59db6820754d4a34140178062a1d953a4f2fdc9b0b6acb',
    PREDECESSOR: 'a056513acc570f82957129afed36c95480bf3e4efa24eef36a5b59e477958984',
    'artifacts/codeact-within-components-root-CPU-001.json':
        '276cc917f94150891e902bc339a4287605db73aebced402e04784feec6775184',
    'artifacts/codeact-within-probe-plan-CPU-001.json':
        'cb8f6c15436d101104c75ed239ce0077a9e69152d5dfdc40f79085bef10e7333',
    'artifacts/codeact-within-probe-root-CPU-001.json':
        '9173e488d1b0eecf7e98a341e508ec8392fd6c6669f80ea236536f7579ad6721',
}
REQUIRED_NEW_SOURCES = {
    'scripts/codeact_model_client_001.py', 'scripts/codeact_execution_observer_001.py',
    'scripts/codeact_within_worker_state_001.py', 'scripts/codeact_within_worker_001.py',
    'scripts/codeact_owned_worker_001.py', 'scripts/codeact_owned_environment_001.py',
    'scripts/codeact_within_episode_001.py', 'scripts/codeact_within_sources_001.py',
    'scripts/codeact_within_registration_001.py', 'scripts/codeact_within_gpu_queue_001.py',
    'scripts/codeact_within_watchdog_001.py', 'scripts/run_codeact_within_source_001.py',
    'scripts/run_codeact_within_source_study_001.py',
}


def _require(condition):
    if not condition:
        raise ValueError('codeact_within_registration_invalid')


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def collect_native_assets(*, data_dir=DATA, deps_dir=DEPS):
    """Hash execution inputs only; never enumerate/read task ground_truth.

    Protected specifications/databases are hashed internally, not returned as
    bodies or read by the actor. Caller records these digests in its plan.
    """
    from scripts.codeact_within_worker_001 import SCORING_SOURCES, registry

    data, deps = Path(data_dir), Path(deps_dir)
    files = {data / 'datasets/train.txt', data / 'version.txt'}
    for directory in (data / 'api_docs', data / 'base_dbs'):
        _require(directory.is_dir() and not directory.is_symlink())
        files.update(path for path in directory.rglob('*') if path.is_file())
    for task in TASK_IDS:
        files.add(data / 'tasks' / task / 'specs.json')
        directory = data / 'tasks' / task / 'dbs'
        _require(directory.is_dir() and not directory.is_symlink())
        files.update(path for path in directory.rglob('*') if path.is_file())
    files.update(Path(os.path.normpath(str(deps / 'appworld' / name)))
                 for name in registry.SOURCE_SHA256 | SCORING_SOURCES)
    bindings = {}
    for path in sorted(files):
        _require('ground_truth' not in path.parts and path.is_file() and not path.is_symlink()
                 and path.resolve(strict=True) == path)
        bindings[str(path)] = _hash(path.read_bytes())
    return bindings


def verify_native_bindings(plan):
    expected = plan.get('native_bindings')
    _require(type(expected) is dict and bool(expected))
    # Re-enumeration detects added/removed execution inputs too.
    _require(collect_native_assets() == expected)
    return True


def validate_metadata(plan, *, _service_validator=None, root=ROOT):
    """Check new study semantics after authenticating its public bytes."""
    _require(type(plan) is dict and plan.get('id') == ID and plan.get('status') == 'registered'
             and plan.get('automatic_retry') is False and plan.get('training_authorized') is False
             and plan.get('writer_authorized') is False and plan.get('utility_authorized') is False
             and plan.get('task_ids') == list(TASK_IDS) and plan.get('batches') == BATCHES
             and json.dumps(plan.get('protocol'), sort_keys=True) == json.dumps(PROTOCOL, sort_keys=True)
             and type(plan.get('port')) is int and plan['port'] == PORT
             and plan.get('model_id') == MODEL_ID and plan.get('image') == IMAGE
             and plan.get('data_dir') == DATA and plan.get('deps_dir') == DEPS
             and plan.get('private_directory') == str(PRIVATE) and plan.get('tmp_dir') == TMPDIR
             and type(plan.get('gpu_seconds')) is int and plan['gpu_seconds'] == 7200
             and type(plan.get('client_seconds')) is int and plan['client_seconds'] == 6000
             and type(plan.get('model_start_seconds')) is int and plan['model_start_seconds'] == 900
             and type(plan.get('cleanup_seconds')) is int and plan['cleanup_seconds'] == 180
             and type(plan.get('admission_wait_seconds')) is int and plan['admission_wait_seconds'] == 86400
             and type(plan.get('source_sha256')) is dict
             and REQUIRED_NEW_SOURCES <= plan['source_sha256'].keys()
             and type(plan.get('input_sha256')) is dict
             and all(plan['input_sha256'].get(k) == v for k, v in PROOF_INPUTS.items()))
    final_cpu_sha = plan['input_sha256'].get(FINAL_CPU_PROOF)
    _require(type(final_cpu_sha) is str and re.fullmatch('[0-9a-f]{64}', final_cpu_sha) is not None)
    assets = plan.get('native_bindings')
    _require(type(assets) is dict and bool(assets)
             and all(type(name) is str and Path(name).is_absolute()
                     and 'ground_truth' not in Path(name).parts
                     and (Path(DATA) in Path(name).parents or Path(DEPS) in Path(name).parents)
                     and str(Path(name)) == name and '..' not in Path(name).parts
                     and type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None
                     for name, digest in assets.items()))
    if _service_validator is None:
        from scripts.coverage_contract_gpu_queue import _public_service

        _service_validator = _public_service
    _service_validator(root, plan, plan['source_sha256'], plan['input_sha256'])
    return plan


def load_registration(root, sha, *, private=False):
    _require(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha) is not None)
    root = Path(root)
    plan = verify_public_frozen_plan(root, root / PLAN, sha)
    validate_metadata(plan, root=root)
    cohort = json.loads(public_read(root, COHORT))
    _require(cohort['source_task_ids'] == list(TASK_IDS) and cohort['source_task_count'] == 12
             and cohort['scenario_roles_pairwise_disjoint'] is True
             and len(cohort['utility_task_ids']) == 27)
    probe = json.loads(public_read(root, 'artifacts/codeact-within-probe-root-CPU-001.json'))
    _require(probe['status'] == 'passed' and probe['owner']['safe_to_continue'] is True
             and probe['owner']['cleanup_verified'] is True and probe['source_sha256_unchanged'] is True
             and all(probe[k] == 0 for k in ('model_calls', 'GPU_calls', 'official_scorer_calls')))
    probe_plan = json.loads(public_read(root, 'artifacts/codeact-within-probe-plan-CPU-001.json'))
    for name, expected in probe_plan['source_sha256'].items():
        _require(plan['source_sha256'].get(name) == expected)
    component_proof = json.loads(public_read(root, 'artifacts/codeact-within-components-root-CPU-001.json'))
    for name, expected in component_proof['source_test_sha256'].items():
        _require(plan['source_sha256'].get(name) == expected)
    final_cpu = json.loads(public_read(root, FINAL_CPU_PROOF))
    _require(final_cpu['status'] == 'passed' and final_cpu['synthetic_only'] is True
             and type(final_cpu['tests_passed']) is int and final_cpu['tests_passed'] > 0)
    for name, expected in final_cpu['source_test_sha256'].items():
        _require(plan['source_sha256'].get(name) == expected)
    if private:
        _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('TMPDIR') == TMPDIR)
        for path in (Path(TMPDIR), PRIVATE):
            info = path.stat()
            _require(path.is_dir() and not path.is_symlink() and info.st_uid == os.getuid()
                     and (info.st_mode & 0o777) == 0o700)
        verify_native_bindings(plan)
    return plan


def check_predecessor(root, plan):
    """Public complete-negative evidence; queue admission also checks live release.

    This function is intentionally read-only and starts no task/scorer/target
    extraction. The launch preflight separately confirms the old parent exited.
    """
    _require(plan.get('input_sha256', {}).get(PREDECESSOR) == PROOF_INPUTS[PREDECESSOR])
    raw = public_read(root, PREDECESSOR)
    _require(_hash(raw) == PROOF_INPUTS[PREDECESSOR])
    proof = json.loads(raw)
    snapshot = proof['snapshot']
    _require(proof['status'] == 'verified_complete_insufficient_source'
             and snapshot['terminal']['authenticated'] is True
             and snapshot['terminal']['status'] == 'complete'
             and snapshot['terminal']['private_result_sha256']
             == 'c889c1c3c42f137bd1a52f355e93e2b9f2e6773ddca274ae2b4cc26de57396b2'
             and snapshot['classified_sources'] == 12 and snapshot['whole_study_costs_known'] is True
             and len(snapshot['releases']) == 12 and all(r['known'] is True for r in snapshot['releases'])
             and snapshot['classified_source_counts'].get('recovered', 0) == 0
             and proof['parent_service']['MainPID'] == '0'
             and proof['parent_service']['ControlPID'] == '0'
             and proof['parent_service']['Result'] == 'success'
             and proof['last_model_proc_absent'] is True
             and proof['last_model_absent_from_compute_apps'] is True)
    return True
