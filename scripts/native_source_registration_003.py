"""Independent V3 registration over authenticated, unchanged V2 metadata code."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_source_registration_002.py'
BASE_SHA256 = '45a951b9e2235b9a4a6aa1d71b92b4fd398b6bc2fc3e8d354c158af8cc311112'
ID = 'qwen38-native-source-pilot-003'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-003.json'
PRIVATE_RUN_ROOT = '/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-003'
LAUNCH_PROOF = 'artifacts/native-source-launch-root-CPU-003.json'
PORT = 18991
SERVICE_PLAN = 'artifacts/qwen38-b3-service-plan-001.json'
SERVICE_SHA = 'd7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5'
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200
PREDECESSOR_PLAN = 'artifacts/qwen38-native-source-pilot-plan-002.json'
PREDECESSOR_PLAN_SHA256 = '881853d06fc91869b8bb9890e55e9e7e9e1eda7d99b7a91dccdfd912e26cbcb2'
PREDECESSOR_TERMINAL = 'artifacts/native-source-pilot-terminal-root-002.json'
PREDECESSOR_TERMINAL_SHA256 = '74745bf4408c763bca058a89f8ec2ee818b0102043201ca14067e6157991d61a'
V2_PINS = {
    'scripts/native_source_registration_002.py': BASE_SHA256,
    'scripts/native_source_gpu_queue_002.py': 'ae221cd0ad1a88c2f4833cb605c501e831c5068ad8e3aeca4491f02c31881419',
    'scripts/native_source_episode_002.py': '700a7b9d083f946860ae6f422d912862c2407723379e4ac0fd36e17fb14a082f',
    'scripts/run_native_source_pilot_002.py': '5d7819334cbdd6a1034aa2d09e71a029df372c3ddf24a63f75b865b12d74ca29',
}
V3_SOURCES = frozenset(
    f'{directory}/{prefix}{name}_003.py'
    for name in ('native_verification_token_counter', 'native_appworld_model_client',
                 'native_appworld_actor', 'native_appworld_controller', 'native_source_registration',
                 'native_source_gpu_queue', 'native_source_episode', 'run_native_source_pilot')
    for directory, prefix in (('scripts', ''), ('tests', 'test_')))
_CORE = None


def execution_revision():
    return {'revision': 3, 'cross_process_queue_serialization': True, 'operation_guard_wait_seconds': 10,
            'completed_watcher_cycles_after_cancellation': 2, 'release_seconds': 90,
            'shutdown_reserve_seconds': 150, 'private_initiating_exception_capture': True,
            'actor_pre_generation_count_deadline_is_typed': True,
            'actor_budget_stop_requires_deadline_provenance': True,
            'generic_cpu_count_failure_remains_unknown': True,
            'no_generation_on_pre_generation_deadline': True,
            'independent_study': True, 'automatic_resume': False, 'outcomes_imported': False,
            'predecessor_id': 'qwen38-native-source-pilot-002', 'predecessor_policy_episodes': 1,
            'predecessor_unknown_first_attempts': 1, 'predecessor_unstarted_first_attempts': 11,
            'predecessor_scorer_calls': 0, 'predecessor_qualified_experiences': 0,
            'predecessor_plan': PREDECESSOR_PLAN, 'predecessor_plan_sha256': PREDECESSOR_PLAN_SHA256,
            'predecessor_terminal': PREDECESSOR_TERMINAL,
            'predecessor_terminal_sha256': PREDECESSOR_TERMINAL_SHA256,
            'historical_experiment_name_prefix': 'native-source-001'}


def _isolated(*, raw=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/native_source_registration_002.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('native_source_registration_003_source_invalid')
    spec = importlib.util.spec_from_file_location('scripts._native_source_registration_003_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError('native_source_registration_003_source_invalid')
    base = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    base.ID, base.PLAN, base.PRIVATE_RUN_ROOT, base.LAUNCH_PROOF = ID, PLAN, PRIVATE_RUN_ROOT, LAUNCH_PROOF
    core = base._isolated()
    core.NEW_SOURCES = core.NEW_SOURCES | V3_SOURCES
    core.REQUIRED_SOURCES = core.REQUIRED_SOURCES | V3_SOURCES | V2_PINS.keys()
    core.FROZEN_SOURCE_SHA256 = {**core.FROZEN_SOURCE_SHA256, **V2_PINS}
    core.FIXED_INPUT_SHA256 = {**core.FIXED_INPUT_SHA256, PREDECESSOR_PLAN: PREDECESSOR_PLAN_SHA256,
                               PREDECESSOR_TERMINAL: PREDECESSOR_TERMINAL_SHA256}
    previous_design = core.expected_design

    def expected_design(root):
        design = previous_design(root)
        design['execution_revision'] = execution_revision()
        return design

    core.expected_design = expected_design
    previous_validate = core.validate_registration

    def validate_registration(root, plan, *, private=False):
        try:
            raw_previous = core.public_read(root, PREDECESSOR_PLAN)
            if hashlib.sha256(raw_previous).hexdigest() != PREDECESSOR_PLAN_SHA256:
                raise ValueError
            previous = core._json(raw_previous)
            if type(previous) is not dict:
                raise ValueError
            validated = previous_validate(root, plan, private=private)
            for field in ('source_sha256', 'input_sha256', 'runtime_sha256'):
                old, current = previous.get(field), validated.get(field)
                if (type(old) is not dict or type(current) is not dict
                        or any(type(name) is not str or not core._sha(digest)
                               or current.get(name) != digest for name, digest in old.items())):
                    raise ValueError
            return validated
        except Exception:
            raise ValueError('native_source_registration_invalid') from None

    core.validate_registration = validate_registration
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def load_registration(root, sha, *, private=False):
    return _get_core().load_registration(root, sha, private=private)


def validate_registration(root, plan, *, private=False):
    return _get_core().validate_registration(root, plan, private=private)


def register(root, **kwargs):
    return _get_core().register(root, **kwargs)


def expected_design(root):
    return _get_core().expected_design(root)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_core(), name)
