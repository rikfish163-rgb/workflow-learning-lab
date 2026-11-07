"""Independent source pilot with request admission and conservative accounting.

Reuses authenticated registration code and preserves every predecessor binding.
No writer, utility or training execution is authorized by this source plan.
"""

import hashlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_registration_003.py"
BASE_SHA256 = "eacfc54a26200909d7dcf7fe9c3592bf114acd320b37727de5be930d914110a6"
ID = "qwen38-native-source-pilot-004"
PLAN = "artifacts/qwen38-native-source-pilot-plan-004.json"
PRIVATE_RUN_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-004"
LAUNCH_PROOF = "artifacts/native-source-launch-root-CPU-004.json"
PREDECESSOR_PLAN = "artifacts/qwen38-native-source-pilot-plan-003.json"
PREDECESSOR_PLAN_SHA256 = "2ce39bdd1210b05c3fd0f55d256f22f9c20cc1177f29562c397daeabe91ec202"
PREDECESSOR_TERMINAL = "artifacts/native-source-pilot-terminal-root-003.json"
PREDECESSOR_TERMINAL_SHA256 = "4a181b741b144a2b8e4ebcf903f9e7a387bc7ef3db5b6eb754dc02cd1341d9f5"
CPU_PROOF = "artifacts/native-request-window-save-score-root-CPU-004.json"
CPU_PROOF_SHA256 = "9f0839cb29ea2fc71ec9aaf387b5675c521d480008866661c768570a8f99799a"
V4_SOURCES = frozenset(
    f"{directory}/{prefix}{name}_004.py"
    for name in ("native_appworld_model_client", "native_appworld_actor", "native_appworld_controller",
                 "native_generation_accounting", "native_source_registration", "native_source_gpu_queue",
                 "native_source_episode", "run_native_source_pilot")
    for directory, prefix in (("scripts", ""), ("tests", "test_"))
) | {"tests/test_native_admission_integration_004.py"}
_CORE = None


def execution_revision():
    return {"revision": 4, "cross_process_queue_serialization": True,
            "operation_guard_wait_seconds": 10, "completed_watcher_cycles_after_cancellation": 2,
            "release_seconds": 90, "shutdown_reserve_seconds": 150,
            "private_initiating_exception_capture": True,
            "actor_pre_generation_count_deadline_is_typed": True,
            "actor_budget_stop_requires_deadline_provenance": True,
            "generic_cpu_count_failure_remains_unknown": True,
            "no_generation_on_pre_generation_deadline": True,
            "generation_request_window_seconds": 120,
            "generation_intent_journal_reserve_seconds": 5,
            "no_request_proof_required_for_post_intent_cancellation": True,
            "cancellation_journal_before_cost_decrement": True,
            "strict_generation_accounting_before_scoring": True,
            "post_transport_failure_remains_unknown": True,
            "independent_study": True, "automatic_resume": False, "outcomes_imported": False,
            "predecessor_id": "qwen38-native-source-pilot-003", "predecessor_policy_episodes": 1,
            "predecessor_unknown_first_attempts": 1, "predecessor_unstarted_first_attempts": 11,
            "predecessor_scorer_calls": 0, "predecessor_qualified_experiences": 0,
            "predecessor_plan": PREDECESSOR_PLAN, "predecessor_plan_sha256": PREDECESSOR_PLAN_SHA256,
            "predecessor_terminal": PREDECESSOR_TERMINAL,
            "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
            "historical_experiment_name_prefix": "native-source-001"}


def _preserve_maps(previous, current):
    for field in ("source_sha256", "input_sha256", "runtime_sha256"):
        old, new = previous.get(field), current.get(field)
        if (type(old) is not dict or type(new) is not dict
                or any(type(name) is not str or type(sha) is not str
                       or re.fullmatch(r"[a-f0-9]{64}", sha) is None or new.get(name) != sha
                       for name, sha in old.items())):
            raise ValueError("native_source_registration_invalid")


def _isolated(*, raw=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, "scripts/native_source_registration_003.py") if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("native_source_registration_004_source_invalid")
    spec = importlib.util.spec_from_file_location("scripts._native_source_registration_004_base", BASE_SOURCE)
    base = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), "exec", dont_inherit=True), base.__dict__)
    base.ID, base.PLAN, base.PRIVATE_RUN_ROOT, base.LAUNCH_PROOF = ID, PLAN, PRIVATE_RUN_ROOT, LAUNCH_PROOF
    base.execution_revision = execution_revision
    core = base._isolated()
    core.NEW_SOURCES = core.NEW_SOURCES | V4_SOURCES
    core.REQUIRED_SOURCES = core.REQUIRED_SOURCES | V4_SOURCES
    core.FROZEN_SOURCE_SHA256 = {**core.FROZEN_SOURCE_SHA256,
                                "scripts/native_source_registration_003.py": BASE_SHA256}
    core.FIXED_INPUT_SHA256 = {**core.FIXED_INPUT_SHA256,
                               PREDECESSOR_PLAN: PREDECESSOR_PLAN_SHA256,
                               PREDECESSOR_TERMINAL: PREDECESSOR_TERMINAL_SHA256,
                               CPU_PROOF: CPU_PROOF_SHA256}
    validate = core.validate_registration

    def validate_registration(root, plan, *, private=False):
        try:
            raw_previous = core.public_read(root, PREDECESSOR_PLAN)
            if hashlib.sha256(raw_previous).hexdigest() != PREDECESSOR_PLAN_SHA256:
                raise ValueError
            previous = core._json(raw_previous)
            validated = validate(root, plan, private=private)
            _preserve_maps(previous, validated)
            return validated
        except Exception:
            raise ValueError("native_source_registration_invalid") from None

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
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)
